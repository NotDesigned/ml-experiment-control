//! Descriptor-confined, resumable removal after explicit administration. The
//! active managed-worker lock is held; exact target directories are quarantined
//! before deleting regular leaves. Receipts survive alongside process metadata.
use ml_exp_core::maintenance::Plan;
use ml_exp_io::directory::Directory;
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
            let file = root.open_regular_read(&name)?;
            root.remove_staged_regular(&name, &file)?;
        }
    }
    Ok(())
}
pub fn run(plan: &Plan) -> io::Result<serde_json::Value> {
    plan.validate().map_err(|_| invalid())?;
    let worker = Directory::open(&plan.config.attempt_root, false)?;
    worker.verify_private_parent()?;
    let root = Directory::open(&plan.storage_root(), false)?;
    root.verify_private_parent()?;
    let lock = worker.open_existing_lock_rw(OsStr::new(".managed-launch.lock"))?;
    fs2::FileExt::try_lock_exclusive(&lock)?;
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
                        if matches!(plan.selection, ml_exp_core::maintenance::Selection::Outputs)
                            && directory.entry_exists(OsStr::new("checkpoints"))?
                        {
                            return Err(io::Error::other("CHECKPOINTS_MUST_BE_REMOVED_SEPARATELY"));
                        }
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
            let temporary = format!(".ml-expd-delete-{id}-{index}");
            let directory = if root.entry_exists(OsStr::new(&temporary))? {
                root.child(OsStr::new(&temporary), false)?
            } else {
                let (parent, name) = match parent(&root, &target.path) {
                    Ok(value) => value,
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {
                        saved.targets[index].done = true;
                        store.replace_regular(OsStr::new(&name), &serde_json::to_vec(&saved)?)?;
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                if !parent.entry_exists(OsStr::new(&name))? {
                    saved.targets[index].done = true;
                    store.replace_regular(
                        OsStr::new(&format!("{id}.json")),
                        &serde_json::to_vec(&saved)?,
                    )?;
                    continue;
                }
                let directory = parent.child(OsStr::new(&name), false)?;
                if directory.identity()? != inode {
                    return Err(invalid());
                }
                directory.publish_directory(&root, OsStr::new(&temporary))?
            };
            if directory.identity()? != inode {
                return Err(invalid());
            }
            remove(&directory, 0, &mut 0)?;
            root.remove_staged_empty_child(&directory)?;
        }
        saved.targets[index].done = true;
        store.replace_regular(OsStr::new(&name), &serde_json::to_vec(&saved)?)?;
    }
    saved.complete = true;
    store.replace_regular(OsStr::new(&name), &serde_json::to_vec(&saved)?)?;
    worker.validate_regular_entry(OsStr::new(".managed-launch.lock"), &lock)?;
    plan.receipt().map_err(|_| invalid())
}
