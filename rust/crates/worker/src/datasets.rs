//! Gateway effects for one explicitly initialized dataset root. No scheduler,
//! credentials or arbitrary executable can be selected by a dataset request.
use ml_exp_core::datasets::{self, Command, Receipt, Reply, Request, State};
use ml_exp_io::{
    datasets as tree,
    directory::{Directory, LockOptions, Wait},
};
use serde::{Deserialize, Serialize};
use std::{
    ffi::OsStr,
    io::{self, Read, Seek, SeekFrom, Write},
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};
const MARKER: &str = ".native-data.json";
const STATE: &str = "state.json";
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub root: PathBuf,
    pub target: ml_exp_core::slurm::helper::Target,
}
#[derive(Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Marker {
    contract: String,
    target: ml_exp_core::slurm::helper::Target,
    identity: (u64, u64),
}
/// One message covers many checks, so name the check that failed.
#[track_caller]
fn invalid() -> io::Error {
    io::Error::other(format!(
        "dataset storage is unavailable, incomplete or conflicts with frozen data (check at line {})",
        std::panic::Location::caller().line()
    ))
}
fn json(value: &impl Serialize) -> io::Result<Vec<u8>> {
    serde_json::to_vec(value).map_err(|_| invalid())
}
pub struct Store {
    config: Config,
    root: Directory,
}
#[cfg(test)]
mod tests;
impl Store {
    fn configured(config: Config) -> io::Result<Self> {
        config.target.validate().map_err(|_| invalid())?;
        if !config.root.to_str().is_some_and(datasets::root_path) {
            return Err(invalid());
        }
        let root = Directory::open(&config.root, false)?;
        Ok(Self { config, root })
    }
    fn marker(&self) -> io::Result<Marker> {
        Ok(Marker {
            contract: datasets::CONTRACT.into(),
            target: self.config.target.clone(),
            identity: self.root.identity()?,
        })
    }
    pub fn initialize(config: Config) -> io::Result<()> {
        let store = Self::configured(config)?;
        if !store.root.entry_names_limited(1)?.is_empty() {
            return Err(invalid());
        }
        store
            .root
            .publish_new(OsStr::new(MARKER), &json(&store.marker()?)?)?;
        Ok(())
    }
    pub fn open(config: Config) -> io::Result<Self> {
        let store = Self::configured(config)?;
        store.verify()?;
        Ok(store)
    }
    /// Open a cloud store, first claiming a root that has no marker. Anything may have
    /// made that directory: training code writing to DATA_DIR, a download script, or a
    /// crash before the marker. Every dataset is still verified on its own, so a claimed
    /// root can only hold nothing useful yet. A marker for another target is refused.
    pub fn claim(config: Config) -> io::Result<Self> {
        let store = Self::configured(config)?;
        if !store.root.entry_exists(OsStr::new(MARKER))? {
            // A store claimed at the same moment is fine; verifying decides.
            let _ = store
                .root
                .publish_new(OsStr::new(MARKER), &json(&store.marker()?)?);
        }
        store.verify()?;
        Ok(store)
    }
    fn verify(&self) -> io::Result<()> {
        let actual: Marker =
            serde_json::from_slice(&self.root.read_regular(OsStr::new(MARKER), 4096)?)
                .map_err(|_| invalid())?;
        let expected = self.marker()?;
        // st_dev identifies a mount in this node's namespace, not a shared
        // filesystem. CPFS/NAS clients can assign different devices to the same
        // inode. Keep the logical authority and inode binding, which refuses
        // an empty or wrong mount at open.
        if actual.contract != expected.contract
            || actual.target != expected.target
            || actual.identity.1 != expected.identity.1
        {
            return Err(invalid());
        }
        Ok(())
    }
    fn read(&self, request: &Request) -> io::Result<(Directory, State)> {
        let project = self.root.child(OsStr::new(&request.project), false)?;
        let directory = project.child(OsStr::new(&request.asset_id()), false)?;
        let bytes = directory.read_regular(OsStr::new(STATE), datasets::MAX_METADATA as u64)?;
        let state: State = serde_json::from_value(serde_json::Value::Object(
            ml_exp_core::json::mapping(&bytes).map_err(|_| invalid())?,
        ))
        .map_err(|_| invalid())?;
        state.validate().map_err(|_| invalid())?;
        if state.project != request.project
            || state.target != request.target
            || state.asset_id != request.asset_id()
        {
            return Err(invalid());
        }
        Ok((directory, state))
    }
    fn save(directory: &Directory, state: &State) -> io::Result<()> {
        state.validate().map_err(|_| invalid())?;
        directory.replace_regular(OsStr::new(STATE), &json(state)?)
    }
    fn begin(&self, request: &Request) -> io::Result<()> {
        let Command::Begin { definition } = &request.command else {
            return Ok(());
        };
        let project = self.root.child(OsStr::new(&request.project), true)?;
        if project.entry_exists(OsStr::new(&request.asset_id()))? {
            let (_, state) = self.read(request)?;
            if &state.definition != definition {
                return Err(invalid());
            }
            return Ok(());
        }
        let state = State::new(
            request.project.clone(),
            request.target.clone(),
            definition.clone(),
        )
        .map_err(|_| invalid())?;
        // The asset lock excludes another uploader. Allocate the final slot
        // with mkdir(EXCL), then publish its control receipt last. Readers
        // reject incomplete reservations instead of repairing/adopting them.
        let slot = project.create_new_child(OsStr::new(&state.asset_id))?;
        slot.create_new_child(OsStr::new("parts"))?;
        slot.publish_new(OsStr::new(STATE), &json(&state)?)
    }
    fn part(
        &self,
        directory: &Directory,
        state: &State,
        number: u64,
        digest: &str,
        payload: &[u8],
    ) -> io::Result<State> {
        let record = ml_exp_core::multipart::PartRecord {
            sha256: digest.into(),
            bytes: payload.len() as u64,
        };
        let next = state
            .record_part(number, record.clone())
            .map_err(|_| invalid())?;
        if state.ready.is_some() {
            return Ok(next);
        }
        let capacity = directory.capacity()?;
        if capacity.insufficient(record.bytes + ml_exp_core::multipart::PART_BYTES, 2) {
            eprintln!(
                "ML_EXPD_DATA_CAPACITY available_bytes={} available_inodes={:?}",
                capacity.available_bytes, capacity.available_inodes
            );
            return Err(io::Error::from_raw_os_error(libc::ENOSPC));
        }
        let parts = directory.child(OsStr::new("parts"), false)?;
        let name = number.to_string();
        if !parts.entry_exists(OsStr::new(&name))? {
            parts.publish_new(OsStr::new(&name), payload)?;
        }
        let mut file = parts.open_regular_read(OsStr::new(&name))?;
        let (sha, bytes) = ml_exp_io::artifacts::hash(&mut file)?;
        if sha != record.sha256 || bytes != record.bytes {
            return Err(invalid());
        }
        ml_exp_io::directory::seal_regular(&file, false)?;
        Self::save(directory, &next)?;
        Ok(next)
    }
    fn cleanup(directory: &Directory) -> io::Result<()> {
        if directory.entry_exists(OsStr::new("archive.tar"))? {
            directory.remove_staged_regular(OsStr::new("archive.tar"))?;
        }
        let parts = directory.child(OsStr::new("parts"), false)?;
        for name in parts.entry_names_limited(ml_exp_core::multipart::MAX_PARTS as usize)? {
            let number = name
                .to_str()
                .and_then(|s| s.parse::<u64>().ok())
                .ok_or_else(invalid)?;
            if name != OsStr::new(&number.to_string()) {
                return Err(invalid());
            }
            parts.remove_staged_regular(&name)?;
        }
        Ok(())
    }
    fn receipt(&self, state: &State) -> io::Result<Receipt> {
        let root = self.config.root.to_str().ok_or_else(invalid)?;
        Ok(Receipt {
            contract: datasets::CONTRACT.into(),
            project: state.project.clone(),
            target: state.target.clone(),
            asset_id: state.asset_id.clone(),
            definition: state.definition.clone(),
            root: format!("{root}/{}/{}/tree", state.project, state.asset_id),
        })
    }
    /// Publish a state with no receipt yet. Any tree here came from a job that did
    /// not finish, so no Run has accepted it, and its bytes can be made again.
    fn complete(
        &self,
        request: &Request,
        directory: &Directory,
        state: &State,
        cancelled: &AtomicBool,
    ) -> io::Result<State> {
        // Pure completeness admission precedes any concatenation or extraction.
        let next = state.publish(self.receipt(state)?).map_err(|_| invalid())?;
        let stale = directory.entry_exists(OsStr::new("tree"))?;
        let sealed = stale
            && tree::verify(
                &directory.child(OsStr::new("tree"), false)?,
                &state.definition,
                true,
            )
            .is_ok();
        if !sealed {
            if stale {
                eprintln!("ML_EXPD_DATA_TREE_REBUILD no receipt and verification failed");
            }
            self.rebuild(request, directory, state, stale, cancelled)?;
        }
        Self::save(directory, &next)?;
        Self::cleanup(directory)?;
        Ok(next)
    }
    /// Make the tree from the parts, which stay until READY is saved. A stale tree
    /// goes only once the parts have proved the whole archive.
    fn rebuild(
        &self,
        request: &Request,
        directory: &Directory,
        state: &State,
        stale: bool,
        cancelled: &AtomicBool,
    ) -> io::Result<()> {
        if directory.entry_exists(OsStr::new("archive.tar"))? {
            directory.remove_staged_regular(OsStr::new("archive.tar"))?;
        }
        let definition = &state.definition;
        let expanded = definition
            .expanded_bytes()
            .map_err(|_| invalid())?
            .checked_add(ml_exp_core::multipart::PART_BYTES)
            .ok_or_else(invalid)?;
        let room = |bytes: u64| -> io::Result<()> {
            if directory
                .capacity()?
                .insufficient(bytes, (definition.files.len() as u64) * 2 + 4)
            {
                return Err(io::Error::other(
                    "dataset publication capacity is insufficient",
                ));
            }
            Ok(())
        };
        // A stale tree's space comes back before extraction starts.
        room(if stale {
            definition.archive_bytes
        } else {
            definition
                .archive_bytes
                .checked_add(expanded)
                .ok_or_else(invalid)?
        })?;
        let mut archive = directory.create_new_regular(OsStr::new("archive.tar"))?;
        let parts = directory.child(OsStr::new("parts"), false)?;
        let mut buffer = [0u8; 65536];
        for (number, part) in &state.parts {
            let mut file = parts.open_regular_read(OsStr::new(&number.to_string()))?;
            let (sha, bytes) = ml_exp_io::artifacts::hash(&mut file)?;
            if sha != part.sha256 || bytes != part.bytes {
                return Err(invalid());
            }
            file.seek(SeekFrom::Start(0))?;
            loop {
                if cancelled.load(Ordering::Relaxed) {
                    return Err(invalid());
                }
                let count = file.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                archive.write_all(&buffer[..count])?;
            }
        }
        archive.sync_all()?;
        archive.seek(SeekFrom::Start(0))?;
        let (sha, bytes) = ml_exp_io::artifacts::hash(&mut archive)?;
        if sha != definition.sha256 || bytes != definition.archive_bytes {
            return Err(invalid());
        }
        if cancelled.load(Ordering::Relaxed) {
            return Err(invalid());
        }
        if stale {
            self.discard_unpublished(request, directory)?;
            room(expanded)?;
        }
        // Only a complete verified tree plus the READY state is usable.
        // Exclusive final-name allocation works on CPFS without a
        // check-then-rename fallback or an additional payload copy.
        let tree = directory.create_new_child(OsStr::new("tree"))?;
        let prepared = (|| {
            tree::extract(&mut archive, &tree, definition)?;
            tree::seal(&tree)?;
            if cancelled.load(Ordering::Relaxed) {
                return Err(invalid());
            }
            Ok(())
        })();
        if let Err(error) = prepared {
            // Only this invocation's exclusively allocated, unready tree.
            tree::remove_staging(directory, &tree)?;
            return Err(error);
        }
        Self::still_published(directory, &tree, definition)
    }
    /// The one place that deletes a tree at the final name. With the lock working
    /// nothing can have published since this request read the state, and this check
    /// is what stands in the way if flock does not exclude across nodes.
    fn discard_unpublished(&self, request: &Request, directory: &Directory) -> io::Result<()> {
        if self.read(request)?.1.ready.is_some() {
            return Err(invalid());
        }
        let stale = directory.child(OsStr::new("tree"), false)?;
        tree::remove_staging(directory, &stale)
    }
    /// Extraction hashed this directory. READY also needs it still at the final name,
    /// complete, in case another node removed or replaced it with no lock between
    /// them. A failure here leaves no rollback to do: the next Complete verifies
    /// whatever is there. A Run's own hash at job start still decides what it reads.
    fn still_published(
        directory: &Directory,
        tree: &Directory,
        definition: &datasets::Definition,
    ) -> io::Result<()> {
        let current = directory.child(OsStr::new("tree"), false)?;
        if current.identity()? != tree.identity()? {
            return Err(invalid());
        }
        tree::verify_file_sizes(&current, definition)
    }
    /// The part of a request that needs this asset to itself. Each asset has its own
    /// lock, beside its slot, so that a long import cannot time out another asset's
    /// Run. A Complete of a published tree returns it to be hashed after the lock is
    /// released: published bytes do not change, and hashing under the lock would time
    /// out every other Run of this asset.
    fn exclusive(
        &self,
        request: &Request,
        payload: &[u8],
        cancelled: &AtomicBool,
    ) -> io::Result<(State, Option<Directory>)> {
        let begin = matches!(request.command, Command::Begin { .. });
        let project = self.root.child(OsStr::new(&request.project), begin)?;
        let asset = request.asset_id();
        // Only Begin allocates, so a query for an unknown asset leaves no lock file.
        if !begin && !project.entry_exists(OsStr::new(&asset))? {
            return Err(invalid());
        }
        let _lock = project
            .lock(
                OsStr::new(&format!("{asset}.lock")),
                LockOptions {
                    create: true,
                    wait: Wait::Until {
                        deadline: Instant::now() + Duration::from_secs(10),
                        cancelled: Some(cancelled),
                    },
                    ..Default::default()
                },
            )
            .map_err(|e| {
                if e.kind() == io::ErrorKind::WouldBlock {
                    invalid()
                } else {
                    e
                }
            })?;
        self.verify()?;
        if cancelled.load(Ordering::Relaxed) {
            return Err(invalid());
        }
        self.begin(request)?;
        let (directory, state) = self.read(request)?;
        Ok(match &request.command {
            Command::Part { number, sha256, .. } => (
                self.part(&directory, &state, *number, sha256, payload)?,
                None,
            ),
            Command::Complete { .. } if state.ready.is_some() => {
                state
                    .publish(self.receipt(&state)?)
                    .map_err(|_| invalid())?;
                // Finish a cleanup that a kill cut short after READY was saved.
                Self::cleanup(&directory)?;
                (state, Some(directory))
            }
            Command::Complete { .. } => {
                (self.complete(request, &directory, &state, cancelled)?, None)
            }
            _ => (state, None),
        })
    }
    pub fn handle(
        &self,
        request: &Request,
        payload: &[u8],
        cancelled: &AtomicBool,
    ) -> io::Result<Reply> {
        request.validate().map_err(|_| invalid())?;
        if request.target != self.config.target {
            return Err(invalid());
        }
        match &request.command {
            Command::Part { bytes, sha256, .. } => {
                if *bytes != payload.len() as u64 {
                    return Err(invalid());
                }
                use sha2::{Digest, Sha256};
                if format!("{:x}", Sha256::digest(payload)) != *sha256 {
                    return Err(invalid());
                }
            }
            _ => {
                if !payload.is_empty() {
                    return Err(invalid());
                }
            }
        }
        if cancelled.load(Ordering::Relaxed) {
            return Err(invalid());
        }
        let (state, published) = self.exclusive(request, payload, cancelled)?;
        if let Some(directory) = published {
            // Sealed READY content is only ever read: a failure here changes nothing.
            let tree = directory.child(OsStr::new("tree"), false)?;
            tree::verify(&tree, &state.definition, true)?;
        }
        self.verify()?;
        let reply = Reply {
            version: 1,
            request_sha256: request.digest().map_err(|_| invalid())?,
            state,
        };
        reply.validate(request).map_err(|_| invalid())?;
        Ok(reply)
    }
}

#[cfg(test)]
mod evidence_tests {
    use super::*;
    #[test]
    fn storage_check_failures_name_the_check_and_capacity_reports_full_storage() {
        let first = invalid().to_string();
        let second = invalid().to_string();
        assert!(first.contains("check at line"));
        assert_ne!(first, second);
        assert_eq!(
            io::Error::from_raw_os_error(libc::ENOSPC).raw_os_error(),
            Some(libc::ENOSPC)
        );
    }
}
