//! Gateway effects for one explicitly initialized dataset root. No scheduler,
//! credentials or arbitrary executable can be selected by a dataset request.
use fs2::FileExt;
use ml_exp_core::datasets::{self, Command, Receipt, Reply, Request, State};
use ml_exp_io::{datasets as tree, directory::Directory};
use serde::{Deserialize, Serialize};
use std::{
    ffi::OsStr,
    io::{self, Read, Seek, SeekFrom, Write},
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};
const MARKER: &str = ".native-data.json";
const LOCK: &str = ".native-data.lock";
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
fn invalid() -> io::Error {
    io::Error::other("dataset storage is unavailable, incomplete or conflicts with frozen data")
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
        root.verify_private_parent()?;
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
        store.root.open_lock(OsStr::new(LOCK))?;
        Ok(())
    }
    pub fn open(config: Config) -> io::Result<Self> {
        let store = Self::configured(config)?;
        store.verify()?;
        Ok(store)
    }
    fn verify(&self) -> io::Result<()> {
        let actual: Marker =
            serde_json::from_slice(&self.root.read_regular(OsStr::new(MARKER), 4096)?)
                .map_err(|_| invalid())?;
        if actual != self.marker()? {
            return Err(invalid());
        }
        self.root.verify_private_parent()
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
        let mut random = [0u8; 16];
        getrandom::fill(&mut random).map_err(|_| invalid())?;
        let stage = project.create_new_child(OsStr::new(&format!(
            ".dataset-{:032x}",
            u128::from_be_bytes(random)
        )))?;
        let state = State::new(
            request.project.clone(),
            request.target.clone(),
            definition.clone(),
        )
        .map_err(|_| invalid())?;
        let result = (|| {
            stage.publish_new(OsStr::new(STATE), &json(&state)?)?;
            stage.create_new_child(OsStr::new("parts"))?;
            stage.publish_directory(&project, OsStr::new(&state.asset_id))?;
            Ok(())
        })();
        if stage.verify_attached().is_ok() {
            tree::remove_staging(&project, &stage)?;
        }
        result
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
        if capacity.available_bytes < record.bytes + 16 * 1024 * 1024
            || capacity.available_inodes < 2
        {
            return Err(io::Error::other("dataset part capacity is insufficient"));
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
        parts.seal_regular(OsStr::new(&name), &file, false)?;
        Self::save(directory, &next)?;
        Ok(next)
    }
    fn cleanup(directory: &Directory) -> io::Result<()> {
        if directory.entry_exists(OsStr::new("archive.tar"))? {
            let file = directory.open_regular_read(OsStr::new("archive.tar"))?;
            directory.remove_staged_regular(OsStr::new("archive.tar"), &file)?;
        }
        let parts = directory.child(OsStr::new("parts"), false)?;
        for name in parts.entry_names_limited(ml_exp_core::multipart::MAX_MAX_PARTS as usize)? {
            let number = name
                .to_str()
                .and_then(|s| s.parse::<u64>().ok())
                .ok_or_else(invalid)?;
            if name != OsStr::new(&number.to_string()) {
                return Err(invalid());
            }
            let file = parts.open_regular_read(&name)?;
            parts.remove_staged_regular(&name, &file)?;
        }
        Ok(())
    }
    fn complete(
        &self,
        directory: &Directory,
        state: &State,
        cancelled: &AtomicBool,
    ) -> io::Result<State> {
        let root = self.config.root.to_str().ok_or_else(invalid)?;
        let receipt = Receipt {
            contract: datasets::CONTRACT.into(),
            project: state.project.clone(),
            target: state.target.clone(),
            asset_id: state.asset_id.clone(),
            definition: state.definition.clone(),
            root: format!("{root}/{}/{}/tree", state.project, state.asset_id),
        };
        // Pure completeness admission precedes any concatenation or extraction.
        let next = state.publish(receipt).map_err(|_| invalid())?;
        if directory.entry_exists(OsStr::new("tree"))? {
            let tree = directory.child(OsStr::new("tree"), false)?;
            tree::verify(&tree, &state.definition, true)?;
        } else {
            if state.ready.is_some() {
                return Err(invalid());
            }
            if directory.entry_exists(OsStr::new("unpacked"))? {
                let old = directory.child(OsStr::new("unpacked"), false)?;
                tree::remove_staging(directory, &old)?;
            }
            if directory.entry_exists(OsStr::new("archive.tar"))? {
                let old = directory.open_regular_read(OsStr::new("archive.tar"))?;
                directory.remove_staged_regular(OsStr::new("archive.tar"), &old)?;
            }
            let required = state
                .definition
                .archive_bytes
                .checked_add(state.definition.expanded_bytes().map_err(|_| invalid())?)
                .and_then(|v| v.checked_add(16 * 1024 * 1024))
                .ok_or_else(invalid)?;
            let capacity = directory.capacity()?;
            if capacity.available_bytes < required
                || capacity.available_inodes < (state.definition.files.len() as u64) * 2 + 4
            {
                return Err(io::Error::other(
                    "dataset publication capacity is insufficient",
                ));
            }
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
            if sha != state.definition.sha256 || bytes != state.definition.archive_bytes {
                return Err(invalid());
            }
            if cancelled.load(Ordering::Relaxed) {
                return Err(invalid());
            }
            let tree = directory.create_new_child(OsStr::new("unpacked"))?;
            tree::extract(&mut archive, &tree, &state.definition)?;
            tree::seal(&tree)?;
            if cancelled.load(Ordering::Relaxed) {
                return Err(invalid());
            }
            tree.publish_directory(directory, OsStr::new("tree"))?;
        }
        Self::save(directory, &next)?;
        Self::cleanup(directory)?;
        Ok(next)
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
        let lock = self.root.open_existing_lock(OsStr::new(LOCK))?;
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match lock.try_lock_exclusive() {
                Ok(()) => break,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    if cancelled.load(Ordering::Relaxed) || Instant::now() >= deadline {
                        return Err(invalid());
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(e) => return Err(e),
            }
        }
        self.verify()?;
        self.root.validate_regular_entry(OsStr::new(LOCK), &lock)?;
        if cancelled.load(Ordering::Relaxed) {
            return Err(invalid());
        }
        self.begin(request)?;
        let (directory, state) = self.read(request)?;
        let state = match &request.command {
            Command::Part { number, sha256, .. } => {
                self.part(&directory, &state, *number, sha256, payload)?
            }
            Command::Complete { .. } => self.complete(&directory, &state, cancelled)?,
            _ => state,
        };
        self.verify()?;
        self.root.validate_regular_entry(OsStr::new(LOCK), &lock)?;
        let reply = Reply {
            version: 1,
            request_sha256: request.digest().map_err(|_| invalid())?,
            state,
        };
        reply.validate(request).map_err(|_| invalid())?;
        Ok(reply)
    }
}
