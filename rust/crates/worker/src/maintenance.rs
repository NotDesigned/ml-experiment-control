//! Descriptor-confined, resumable removal after explicit administration. The
//! active managed-worker lock is held and each target is deleted through its
//! retained inode. Receipts survive alongside process metadata.
use ml_exp_core::maintenance::Plan;
use ml_exp_io::directory::{Directory, LockOptions};
use serde::{Deserialize, Serialize};
use std::{ffi::OsStr, io};
fn invalid() -> io::Error {
    io::Error::other("maintenance scope, worker lock or retained deletion evidence is invalid")
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Target {
    path: String,
    inode: Option<(u64, u64)>,
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
        if let Ok(directory) = root.child(&name, false) {
            inspect(&directory, depth + 1, budget)?;
        } else {
            root.open_regular_read(&name)?;
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
        if let Ok(directory) = root.child(&name, false) {
            remove(&directory, depth + 1, budget)?;
            root.remove_staged_empty_child(&directory)?;
        } else {
            root.remove_staged_regular(&name)?;
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
                        Some(directory.identity()?)
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
        if let ml_exp_core::maintenance::Selection::Checkpoint { checkpoint } = &plan.selection {
            let (context, manifest) =
                ml_exp_core::checkpoints::validate_document(checkpoint, None, None)
                    .map_err(|_| invalid())?;
            let state = Directory::open(std::path::Path::new(&context.state_root), false)?;
            ml_exp_io::checkpoints::verify_sealed(&state, &manifest).map_err(|_| invalid())?;
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
                    if directory.identity()? != inode {
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
    use std::{collections::BTreeMap, fs, path::Path};

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
}
