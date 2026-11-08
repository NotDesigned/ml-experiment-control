//! Descriptor-confined, resumable removal after explicit administration. The
//! active managed-worker lock is held and each target is deleted through its
//! retained inode. Receipts survive alongside process metadata. A link or special
//! file in a target is removed itself and never followed or opened.
use ml_exp_core::maintenance::Plan;
use ml_exp_io::directory::{Directory, EntryType, LockOptions};
use serde::{Deserialize, Serialize};
use std::{ffi::OsStr, io};
fn invalid() -> io::Error {
    io::Error::other("maintenance scope, worker lock or retained deletion evidence is invalid")
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Target {
    path: String,
    inode: Option<u64>,
    done: bool,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    plan: Plan,
    targets: Vec<Target>,
    complete: bool,
}
fn parent(root: &Directory, path: &str) -> io::Result<(Directory, String)> {
    let mut components = path.split('/').collect::<Vec<_>>();
    let name = components.pop().ok_or_else(invalid)?.to_owned();
    let mut parent = root.clone();
    for p in components {
        parent = parent.child(OsStr::new(p), false)?;
    }
    Ok((parent, name))
}
fn inspect(root: &Directory, depth: usize, budget: &mut usize) -> io::Result<()> {
    if depth > 128 || *budget > 1000000 {
        return Err(invalid());
    }
    for name in root.entry_names_limited(1000000)? {
        *budget += 1;
        if root.entry_type(&name)? == Some(EntryType::Directory) {
            inspect(&root.child(&name, false)?, depth + 1, budget)?;
        }
    }
    Ok(())
}
fn remove(root: &Directory, depth: usize, budget: &mut usize) -> io::Result<()> {
    if depth > 128 || *budget > 1000000 {
        return Err(invalid());
    }
    root.source_mode(false)?;
    for name in root.entry_names_limited(1000000)? {
        *budget += 1;
        if root.entry_type(&name)? == Some(EntryType::Directory) {
            let directory = root.child(&name, false)?;
            remove(&directory, depth + 1, budget)?;
            root.remove_staged_empty_child(&directory)?;
        } else {
            root.remove_non_directory(&name)?;
        }
    }
    Ok(())
}
pub fn run(plan: &Plan) -> io::Result<serde_json::Value> {
    plan.validate().map_err(|_| invalid())?;
    let worker = Directory::open(&plan.config.attempt_root, false)?;
    let root = Directory::open(&plan.storage_root(), false)?;
    // The phase ledger: it is never created here, because a recreated ledger
    // reads as unstarted and could re-run training.
    let _lock = worker.lock(
        OsStr::new(".managed-launch.lock"),
        LockOptions {
            write: true,
            ..Default::default()
        },
    )?;
    let manifest = worker.read_regular(
        OsStr::new("launch-manifest.json"),
        ml_exp_core::launch::MAX_BYTES as u64,
    )?;
    // The file is created by managed::initialize, not client-authored metadata.
    let binding = plan.config.binding().map_err(|_| invalid())?;
    ml_exp_core::launch::verify_binding(&manifest, &binding).map_err(|_| invalid())?;
    let store = if worker.entry_exists(OsStr::new(".storage-maintenance"))? {
        worker.child(OsStr::new(".storage-maintenance"), false)?
    } else {
        worker.create_new_child(OsStr::new(".storage-maintenance"))?
    };
    let id = plan.id().map_err(|_| invalid())?;
    let name = format!("{id}.json");
    let mut saved = if store.entry_exists(OsStr::new(&name))? {
        let saved: Saved =
            serde_json::from_slice(&store.read_regular(OsStr::new(&name), 256 * 1024)?)?;
        if saved.plan != *plan
            || saved
                .targets
                .iter()
                .map(|t| &t.path)
                .ne(plan.paths().map_err(|_| invalid())?.iter())
        {
            return Err(invalid());
        }
        saved
    } else {
        let mut targets = Vec::new();
        for path in plan.paths().map_err(|_| invalid())? {
            let inode = match parent(&root, &path) {
                Ok((parent, name)) => {
                    if parent.entry_exists(OsStr::new(&name))? {
                        let directory = parent.child(OsStr::new(&name), false)?;
                        inspect(&directory, 0, &mut 0)?;
                        Some(directory.identity()?.1)
                    } else {
                        None
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::NotFound => None,
                Err(e) => return Err(e),
            };
            targets.push(Target {
                path,
                inode,
                done: false,
            });
        }
        let saved = Saved {
            plan: plan.clone(),
            targets,
            complete: false,
        };
        store.publish_new(OsStr::new(&name), &serde_json::to_vec(&saved)?)?;
        saved
    };
    if saved.complete {
        return plan.receipt().map_err(|_| invalid());
    }
    for index in 0..saved.targets.len() {
        if saved.targets[index].done {
            continue;
        }
        let target = &saved.targets[index];
        if let Some(inode) = target.inode {
            match parent(&root, &target.path) {
                Ok((parent, leaf)) if parent.entry_exists(OsStr::new(&leaf))? => {
                    let directory = parent.child(OsStr::new(&leaf), false)?;
                    if directory.identity()?.1 != inode {
                        return Err(invalid());
                    }
                    // The reviewed plan and worker lock already fence execution.
                    // Delete through this retained exact inode, keeping its
                    // original pathname attached; CPFS needs no directory rename.
                    remove(&directory, 0, &mut 0)?;
                    parent.remove_staged_empty_child(&directory)?;
                }
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }
        saved.targets[index].done = true;
        store.replace_regular(OsStr::new(&name), &serde_json::to_vec(&saved)?)?;
    }
    saved.complete = true;
    store.replace_regular(OsStr::new(&name), &serde_json::to_vec(&saved)?)?;
    plan.receipt().map_err(|_| invalid())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_exp_core::{
        checkpoints::{self, Context},
        launch,
        maintenance::Selection,
        managed::Config,
    };
    use serde_json::json;
    use std::{
        collections::BTreeMap,
        fs,
        os::unix::fs::{MetadataExt, PermissionsExt, symlink},
        path::Path,
    };

    fn config(attempt: &Path) -> Config {
        let identity = launch::Identity {
            project: "demo".into(),
            run_id: "train".into(),
            attempt_id: "attempt-001".into(),
        };
        let manifest = launch::Manifest {
            contract: launch::CONTRACT.into(),
            identity: identity.clone(),
            environment: BTreeMap::from([(
                "OUTPUT_DIR".into(),
                attempt.join("outputs").to_str().unwrap().into(),
            )]),
            argv: vec!["/bin/true".into()],
            workdir: "/tmp".into(),
            timeout_seconds: 60,
        };
        let bytes = manifest.encode().unwrap();
        fs::create_dir_all(attempt.join("worker")).unwrap();
        fs::write(attempt.join("worker/launch-manifest.json"), &bytes).unwrap();
        fs::write(attempt.join("worker/.managed-launch.lock"), b"").unwrap();
        Config {
            manifest_path: attempt.join("launch-manifest.json"),
            endpoint: String::new(),
            attempt_root: attempt.join("worker"),
            workdir: "/tmp".into(),
            maximum_timeout_seconds: 60,
            base_environment: BTreeMap::new(),
            local_binding: Some(launch::Binding {
                identity,
                sha256: ml_exp_core::slurm::helper::sha256_bytes(&bytes)[7..].into(),
            }),
            output_directory: Some(attempt.join("outputs")),
            runtime_environment: Vec::new(),
            checkpoint_restore: None,
            checkpoint_restore_root: None,
            datasets: Vec::new(),
            data_preparation: None,
        }
    }
    fn plan(config: &Config, selection: Selection, request: char) -> Plan {
        Plan {
            contract: "native-storage-maintenance.v1".into(),
            config: config.clone(),
            selection,
            request_id: request.to_string().repeat(32),
        }
    }
    fn sealed_generation(attempt: &Path) -> serde_json::Value {
        let generation = attempt.join("state/checkpoints/step-4");
        fs::create_dir_all(&generation).unwrap();
        fs::write(generation.join("state.bin"), b"weights").unwrap();
        let ready = json!({"step":4,"files":[{"path":"checkpoints/step-4/state.bin","bytes":7,
            "sha256":"9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"}]});
        let state = Directory::open(&attempt.join("state"), false).unwrap();
        ml_exp_io::checkpoints::verify(&state, &checkpoints::manifest(&ready).unwrap(), true)
            .unwrap();
        let context = Context {
            project: "demo".into(),
            run_id: "train".into(),
            attempt_id: "attempt-001".into(),
            source_id: format!("source.{}", "a".repeat(64)),
            image_id: format!("sha256:{}", "b".repeat(64)),
            storage_scope: format!("sha256:{}", "c".repeat(64)),
            state_root: attempt.join("state").to_str().unwrap().into(),
        };
        checkpoints::document(&context, &ready).unwrap()
    }

    #[test]
    fn outputs_state_and_one_generation_are_removed_independently() {
        let temp = tempfile::tempdir().unwrap();
        let attempt = temp.path().join("runs/demo/train/attempts/attempt-001");
        let config = config(&attempt);
        // A user's own checkpoints directory is an ordinary result, not a refusal.
        fs::create_dir_all(attempt.join("outputs/checkpoints")).unwrap();
        fs::write(attempt.join("outputs/checkpoints/kept.pt"), b"weights").unwrap();
        let document = sealed_generation(&attempt);
        fs::create_dir_all(attempt.join("state/checkpoints/step-8")).unwrap();
        fs::write(attempt.join("state/checkpoints/step-8/other"), b"x").unwrap();

        let outputs = plan(&config, Selection::Outputs, 'a');
        assert_eq!(run(&outputs).unwrap()["paths"], json!(["outputs"]));
        assert!(!attempt.join("outputs").exists());
        assert!(attempt.join("state/checkpoints/step-4/state.bin").exists());

        let generation = plan(
            &config,
            Selection::Checkpoint {
                checkpoint: Box::new(document),
            },
            'b',
        );
        assert_eq!(
            run(&generation).unwrap()["paths"],
            json!(["state/checkpoints/step-4"])
        );
        assert!(!attempt.join("state/checkpoints/step-4").exists());
        assert!(attempt.join("state/checkpoints/step-8/other").exists());

        let state = plan(&config, Selection::State, 'c');
        assert_eq!(run(&state).unwrap()["paths"], json!(["state"]));
        assert!(!attempt.join("state").exists());
        assert!(attempt.join("worker/launch-manifest.json").exists());
        // A repeated plan returns its saved receipt and removes nothing new.
        assert!(run(&outputs).is_ok());
    }

    #[test]
    fn a_corrupt_generation_is_deleted_without_proof_of_its_content() {
        let temp = tempfile::tempdir().unwrap();
        let attempt = temp.path().join("runs/demo/train/attempts/attempt-001");
        let config = config(&attempt);
        let document = sealed_generation(&attempt);
        let generation = attempt.join("state/checkpoints/step-4");
        fs::set_permissions(&generation, fs::Permissions::from_mode(0o700)).unwrap();
        let file = generation.join("state.bin");
        fs::set_permissions(&file, fs::Permissions::from_mode(0o600)).unwrap();
        fs::write(&file, b"damaged, and a different size").unwrap();
        fs::write(generation.join("extra"), b"not in the manifest").unwrap();
        let selection = Selection::Checkpoint {
            checkpoint: Box::new(document),
        };
        assert_eq!(
            run(&plan(&config, selection, 'a')).unwrap()["paths"],
            json!(["state/checkpoints/step-4"])
        );
        assert!(!generation.exists());
    }

    #[test]
    fn links_and_special_files_are_removed_and_what_they_point_at_stays() {
        let temp = tempfile::tempdir().unwrap();
        let outside = temp.path().join("outside");
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("kept"), b"not ours").unwrap();
        let attempt = temp.path().join("runs/demo/train/attempts/attempt-001");
        let config = config(&attempt);
        let outputs = attempt.join("outputs");
        fs::create_dir_all(outputs.join("nested")).unwrap();
        fs::write(outputs.join("nested/result.bin"), b"x").unwrap();
        symlink(&outside, outputs.join("to-directory")).unwrap();
        symlink(outside.join("kept"), outputs.join("nested/to-file")).unwrap();
        symlink("/absent", outputs.join("dangling")).unwrap();
        assert!(
            std::process::Command::new("mkfifo")
                .arg(outputs.join("nested/pipe"))
                .status()
                .unwrap()
                .success()
        );
        assert_eq!(
            run(&plan(&config, Selection::Outputs, 'a')).unwrap()["paths"],
            json!(["outputs"])
        );
        assert!(fs::symlink_metadata(&outputs).is_err());
        assert_eq!(fs::read(outside.join("kept")).unwrap(), b"not ours");
    }

    #[test]
    fn a_failed_read_refuses_the_plan_instead_of_passing_a_directory_as_a_file() {
        if ml_exp_io::directory::effective_uid() == 0 {
            // Root opens a mode 000 directory, so there is no failed read to observe.
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let attempt = temp.path().join("runs/demo/train/attempts/attempt-001");
        let config = config(&attempt);
        let locked = attempt.join("outputs/locked");
        fs::create_dir_all(&locked).unwrap();
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
        let plan = plan(&config, Selection::Outputs, 'a');
        assert_eq!(
            run(&plan).unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );
        assert!(locked.exists());
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o700)).unwrap();
        run(&plan).unwrap();
        assert!(!attempt.join("outputs").exists());
    }

    #[test]
    fn a_resumed_removal_binds_the_target_by_inode_alone() {
        let temp = tempfile::tempdir().unwrap();
        let attempt = temp.path().join("runs/demo/train/attempts/attempt-001");
        let config = config(&attempt);
        fs::create_dir_all(attempt.join("state")).unwrap();
        let plan = plan(&config, Selection::State, 'a');
        let inode = fs::metadata(attempt.join("state")).unwrap().ino();
        let store = attempt.join("worker/.storage-maintenance");
        fs::create_dir(&store).unwrap();
        let saved = |inode| {
            serde_json::to_vec(&Saved {
                plan: plan.clone(),
                targets: vec![Target {
                    path: "state".into(),
                    inode: Some(inode),
                    done: false,
                }],
                complete: false,
            })
            .unwrap()
        };
        let name = store.join(format!("{}.json", plan.id().unwrap()));
        fs::write(&name, saved(inode + 1)).unwrap();
        assert!(run(&plan).is_err());
        assert!(attempt.join("state").exists());
        fs::write(&name, saved(inode)).unwrap();
        run(&plan).unwrap();
        assert!(!attempt.join("state").exists());
    }
}
