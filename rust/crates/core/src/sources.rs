//! Source archive validation and exact legacy tree identities. Files, directory
//! observations, time and archive digests are explicit inputs. No source is run.
use crate::{canonical, identity};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const CHUNK_BYTES: usize = 1024 * 1024;
pub const MAX_ARCHIVE_BYTES: u64 = 512 * 1024 * 1024;
pub const MAX_SOURCE_BYTES: u64 = 1024 * 1024 * 1024;
pub const MAX_SOURCE_FILES: usize = 100_000;
pub const MAX_PATH_BYTES: usize = 4096;
pub const MAX_DEPTH: usize = 128;
pub const MAX_TOPOLOGY_ENTRIES: usize = 100_000;
pub const MAX_TOPOLOGY_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_METADATA_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Policy {
    pub max_archive_bytes: u64,
    pub max_source_bytes: u64,
    pub max_source_files: usize,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            max_archive_bytes: 64 * 1024 * 1024,
            max_source_bytes: 256 * 1024 * 1024,
            max_source_files: 20_000,
        }
    }
}
impl Policy {
    pub fn validate(&self) -> Result<()> {
        if self.max_archive_bytes == 0
            || self.max_archive_bytes > MAX_ARCHIVE_BYTES
            || self.max_source_bytes == 0
            || self.max_source_bytes > MAX_SOURCE_BYTES
            || self.max_source_files == 0
            || self.max_source_files > MAX_SOURCE_FILES
        {
            return Err(Error::Policy);
        }
        Ok(())
    }
    /// Includes tar/PAX framing, not just regular-file payload. The I/O boundary
    /// must apply this bound to the complete decompressed stream as Python does.
    pub fn expanded_archive_limit(&self) -> Result<u64> {
        self.validate()?;
        Ok(self.max_source_bytes + self.max_source_files as u64 * 2048 + CHUNK_BYTES as u64)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid finite source archive policy")]
    Policy,
    #[error("invalid or escaping source archive path")]
    Path,
    #[error("source archive path is protected")]
    Protected,
    #[error("source archive has duplicate paths")]
    Duplicate,
    #[error("source kind or Git proposal operation is not implemented")]
    Unsupported,
    #[error("source archive or metadata exceeds its explicit bound")]
    Limit,
    #[error("source topology is incomplete or contradictory")]
    Topology,
    #[error("source tree chunk differs from the exact requested frame")]
    Chunk,
    #[error("source archive or tree hash is incomplete")]
    Incomplete,
    #[error("invalid or mismatched source identity")]
    Identity,
    #[error("invalid source metadata or observation")]
    Metadata,
}
pub type Result<T> = std::result::Result<T, Error>;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemberKind {
    Directory,
    File,
    Symlink,
    Other,
}
fn parts(raw: &str) -> Vec<&str> {
    raw.split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect()
}
/// Python `_is_protected` semantics, including case-normalized components and
/// pathlib's final suffix rule. A leading-dot filename alone has no suffix.
pub fn is_protected(raw: &str) -> bool {
    parts(raw).iter().any(|part| {
        let lowered = part.to_lowercase();
        [
            ".git",
            ".ssh",
            ".aws",
            ".gnupg",
            "credentials",
            "keys",
            "secrets",
            ".env",
        ]
        .contains(&lowered.as_str())
            || lowered.starts_with(".env.")
            || lowered.starts_with("credentials.")
            || lowered.starts_with("secrets.")
            || lowered.rfind('.').is_some_and(|at| {
                at > 0
                    && at + 1 < lowered.len()
                    && [".key", ".pem", ".p12", ".pfx"].contains(&&lowered[at..])
            })
    })
}
fn relative(raw: &str) -> Result<String> {
    if raw.is_empty()
        || raw.starts_with('/')
        || raw.contains(['\\', '\0'])
        || raw.len() > MAX_PATH_BYTES
    {
        return Err(Error::Path);
    }
    let components = parts(raw);
    if components.is_empty()
        || components.len() > MAX_DEPTH
        || components.iter().any(|part| *part == "..")
    {
        return Err(Error::Path);
    }
    Ok(components.join("/"))
}
/// None is only the ignored `.`/`./` root directory. Symlinks, hard links and
/// special entries are unsupported even when their textual path looks safe.
pub fn archive_path(raw: &str, kind: MemberKind, policy: &Policy) -> Result<Option<String>> {
    policy.validate()?;
    if kind == MemberKind::Directory && [".", "./"].contains(&raw) {
        return Ok(None);
    }
    let path = relative(raw)?;
    if is_protected(raw) {
        return Err(Error::Protected);
    }
    if !matches!(kind, MemberKind::Directory | MemberKind::File) {
        return Err(Error::Unsupported);
    }
    Ok(Some(path))
}
pub struct ArchiveState {
    policy: Policy,
    seen: BTreeSet<String>,
    path_bytes: usize,
    files: usize,
    bytes: u64,
}
pub struct ArchiveSummary {
    pub entries: usize,
    pub files: usize,
    pub expanded_bytes: u64,
}
impl ArchiveState {
    pub fn new(policy: Policy) -> Result<Self> {
        policy.validate()?;
        Ok(Self {
            policy,
            seen: BTreeSet::new(),
            path_bytes: 0,
            files: 0,
            bytes: 0,
        })
    }
    pub fn accept(
        mut self,
        raw: &str,
        kind: MemberKind,
        size: u64,
    ) -> Result<(Self, Option<String>)> {
        let Some(path) = archive_path(raw, kind, &self.policy)? else {
            return Ok((self, None));
        };
        if !self.seen.insert(path.clone()) {
            return Err(Error::Duplicate);
        }
        self.path_bytes = self
            .path_bytes
            .checked_add(path.len())
            .ok_or(Error::Limit)?;
        if self.seen.len() > self.policy.max_source_files || self.path_bytes > MAX_TOPOLOGY_BYTES {
            return Err(Error::Limit);
        }
        if kind == MemberKind::File {
            self.files += 1;
            self.bytes = self.bytes.checked_add(size).ok_or(Error::Limit)?;
            if self.bytes > self.policy.max_source_bytes {
                return Err(Error::Limit);
            }
        }
        Ok((self, Some(path)))
    }
    pub fn finish(self) -> Result<ArchiveSummary> {
        if self.files == 0 {
            return Err(Error::Incomplete);
        }
        Ok(ArchiveSummary {
            entries: self.seen.len(),
            files: self.files,
            expanded_bytes: self.bytes,
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeKind {
    Directory,
    File { size: u64, executable: bool },
}
#[derive(Clone, PartialEq, Eq)]
pub struct TreeEntry {
    pub path: String,
    pub kind: TreeKind,
}
pub struct TreePlan {
    entries: Vec<TreeEntry>,
    bytes: u64,
    files: usize,
}
impl TreePlan {
    /// Entries are actual observed topology, not tar members: all parent
    /// directories must appear. This planner grants no archive/path authority.
    pub fn new(entries: Vec<TreeEntry>, policy: &Policy) -> Result<Self> {
        policy.validate()?;
        if entries.len() > MAX_TOPOLOGY_ENTRIES {
            return Err(Error::Limit);
        }
        let mut table = BTreeMap::new();
        let mut children: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut path_bytes = 0usize;
        let mut bytes = 0u64;
        let mut files = 0usize;
        for entry in entries {
            let normalized = relative(&entry.path)?;
            if normalized != entry.path {
                return Err(Error::Topology);
            }
            path_bytes = path_bytes
                .checked_add(entry.path.len())
                .ok_or(Error::Limit)?;
            if path_bytes > MAX_TOPOLOGY_BYTES {
                return Err(Error::Limit);
            }
            if let TreeKind::File { size, .. } = entry.kind {
                bytes = bytes.checked_add(size).ok_or(Error::Limit)?;
                files += 1;
                if bytes > policy.max_source_bytes || files > policy.max_source_files {
                    return Err(Error::Limit);
                }
            }
            if table.insert(entry.path, entry.kind).is_some() {
                return Err(Error::Duplicate);
            }
        }
        for path in table.keys() {
            let parent = path.rsplit_once('/').map_or("", |(parent, _)| parent);
            if !parent.is_empty() && !matches!(table.get(parent), Some(TreeKind::Directory)) {
                return Err(Error::Topology);
            }
            children
                .entry(parent.into())
                .or_default()
                .push(path.clone());
        }
        // BTreeMap iteration sorts UTF-8 names in Unicode code-point order,
        // as Python's sorted(children,key=lambda p:p.name) does for valid UTF-8.
        let mut ordered = Vec::with_capacity(table.len());
        let mut pending = vec![String::new()];
        while let Some(parent) = pending.pop() {
            let mut directories = Vec::new();
            if let Some(paths) = children.get(&parent) {
                for path in paths {
                    let kind = table[path];
                    ordered.push(TreeEntry {
                        path: path.clone(),
                        kind,
                    });
                    if kind == TreeKind::Directory {
                        directories.push(path.clone());
                    }
                }
            }
            pending.extend(directories.into_iter().rev());
        }
        if ordered.len() != table.len() {
            return Err(Error::Topology);
        }
        Ok(Self {
            entries: ordered,
            bytes,
            files,
        })
    }
    pub fn entries(&self) -> &[TreeEntry] {
        &self.entries
    }
    pub fn total_bytes(&self) -> u64 {
        self.bytes
    }
    pub fn file_count(&self) -> usize {
        self.files
    }
}
pub struct ChunkRequest<'a> {
    pub path: &'a str,
    pub offset: u64,
    pub len: usize,
}
/// Consuming state holds only bounded topology and a SHA256 state. Payload bytes
/// are hashed immediately and never copied or retained. Every file read must
/// fill the exact requested 1MiB frame or final remainder before `consume`.
pub struct TreeDigest {
    plan: TreePlan,
    index: usize,
    offset: u64,
    digest: Sha256,
}
fn field(digest: &mut Sha256, value: &[u8]) {
    digest.update((value.len() as u64).to_be_bytes());
    digest.update(value);
}
impl TreeDigest {
    pub fn new(plan: TreePlan) -> Self {
        let mut state = Self {
            plan,
            index: 0,
            offset: 0,
            digest: Sha256::new(),
        };
        state.advance_headers();
        state
    }
    fn advance_headers(&mut self) {
        while let Some(entry) = self.plan.entries.get(self.index) {
            match entry.kind {
                TreeKind::Directory => {
                    field(&mut self.digest, b"D");
                    field(&mut self.digest, entry.path.as_bytes());
                    self.index += 1;
                }
                TreeKind::File { size, executable } => {
                    field(&mut self.digest, b"F");
                    field(&mut self.digest, entry.path.as_bytes());
                    field(&mut self.digest, if executable { b"X" } else { b"-" });
                    if size > 0 {
                        return;
                    }
                    self.index += 1;
                }
            }
        }
    }
    pub fn next_chunk(&self) -> Option<ChunkRequest<'_>> {
        let entry = self.plan.entries.get(self.index)?;
        let TreeKind::File { size, .. } = entry.kind else {
            return None;
        };
        Some(ChunkRequest {
            path: &entry.path,
            offset: self.offset,
            len: (size - self.offset).min(CHUNK_BYTES as u64) as usize,
        })
    }
    pub fn consume(mut self, bytes: &[u8]) -> Result<Self> {
        let expected = self.next_chunk().ok_or(Error::Chunk)?;
        if bytes.len() != expected.len {
            return Err(Error::Chunk);
        }
        field(&mut self.digest, bytes);
        self.offset += bytes.len() as u64;
        let TreeKind::File { size, .. } = self.plan.entries[self.index].kind else {
            return Err(Error::Topology);
        };
        if self.offset == size {
            self.index += 1;
            self.offset = 0;
            self.advance_headers();
        }
        Ok(self)
    }
    pub fn finish(self) -> Result<String> {
        if self.index != self.plan.entries.len() {
            return Err(Error::Incomplete);
        }
        Ok(format!("sha256:{:x}", self.digest.finalize()))
    }
}
pub fn validate_archive_digest(value: &str) -> Result<&str> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(Error::Identity);
    }
    Ok(value)
}
pub fn source_id(tree_digest: &str) -> Result<String> {
    let value = tree_digest.strip_prefix("sha256:").ok_or(Error::Identity)?;
    validate_archive_digest(value)?;
    Ok(format!("source.{value}"))
}
pub fn validate_source_id(value: &str) -> Result<&str> {
    let suffix = value.strip_prefix("source.").ok_or(Error::Identity)?;
    validate_archive_digest(suffix)?;
    Ok(value)
}
fn bounded_json(value: &Value) -> Result<()> {
    let mut pending = vec![(value, 0usize)];
    let mut nodes = 0usize;
    let mut text = 0usize;
    while let Some((value, depth)) = pending.pop() {
        nodes += 1;
        if nodes > 8192 || depth > 32 {
            return Err(Error::Limit);
        }
        match value {
            Value::String(value) => text = text.checked_add(value.len()).ok_or(Error::Limit)?,
            Value::Object(fields) => {
                if fields.len() > 8192usize.saturating_sub(nodes + pending.len()) {
                    return Err(Error::Limit);
                }
                for (k, v) in fields {
                    text = text.checked_add(k.len()).ok_or(Error::Limit)?;
                    pending.push((v, depth + 1));
                }
            }
            Value::Array(values) => {
                if values.len() > 8192usize.saturating_sub(nodes + pending.len()) {
                    return Err(Error::Limit);
                }
                pending.extend(values.iter().map(|v| (v, depth + 1)));
            }
            Value::Number(v) => {
                let raw = v.to_string();
                text = text.checked_add(raw.len()).ok_or(Error::Limit)?;
                if raw.contains(['.', 'e', 'E']) && !v.as_f64().is_some_and(f64::is_finite) {
                    return Err(Error::Metadata);
                }
            }
            _ => {}
        }
        if text > MAX_METADATA_BYTES || pending.len() > 8192 {
            return Err(Error::Limit);
        }
    }
    if canonical::python_json_bytes(value)
        .map_err(|_| Error::Metadata)?
        .len()
        > MAX_METADATA_BYTES
    {
        return Err(Error::Limit);
    }
    Ok(())
}
fn observation(value: &Value) -> Result<()> {
    bounded_json(value)?;
    let fields = value.as_object().ok_or(Error::Metadata)?;
    if fields.get("kind").and_then(Value::as_str) != Some("archive") {
        return Err(Error::Unsupported);
    }
    validate_archive_digest(
        fields
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or(Error::Metadata)?,
    )?;
    Ok(())
}
/// Preserves bounded unknown provenance; none of it authorizes execution. The
/// archive-only native path deliberately refuses Git/proposal metadata.
#[derive(Clone)]
pub struct Metadata {
    value: Value,
}
impl Metadata {
    pub fn new(
        project: &str,
        tree_digest: &str,
        created_at: &str,
        observation_value: &Value,
        policy: &Policy,
    ) -> Result<Self> {
        policy.validate()?;
        identity::validate_id(project).map_err(|_| Error::Identity)?;
        if created_at.is_empty() || created_at.len() > 128 {
            return Err(Error::Metadata);
        }
        observation(observation_value)?;
        let source = source_id(tree_digest)?;
        Self::from_value(
            &json!({"schema_version":1,"project":project,"source_id":source,"tree":"tree","tree_digest":tree_digest,"created_at":created_at,"observation":observation_value}),
            project,
            &source,
            policy,
        )
    }
    pub fn from_value(value: &Value, project: &str, source: &str, policy: &Policy) -> Result<Self> {
        policy.validate()?;
        bounded_json(value)?;
        identity::validate_id(project).map_err(|_| Error::Identity)?;
        validate_source_id(source)?;
        let fields = value.as_object().ok_or(Error::Metadata)?;
        if fields.get("schema_version").and_then(Value::as_u64) != Some(1)
            || fields.get("project").and_then(Value::as_str) != Some(project)
            || fields.get("source_id").and_then(Value::as_str) != Some(source)
            || fields.get("tree").and_then(Value::as_str) != Some("tree")
            || fields
                .get("tree_digest")
                .and_then(Value::as_str)
                .map(source_id)
                .transpose()?
                .as_deref()
                != Some(source)
            || !fields
                .get("created_at")
                .and_then(Value::as_str)
                .is_some_and(|v| !v.is_empty() && v.len() <= 128)
        {
            return Err(Error::Metadata);
        }
        observation(fields.get("observation").ok_or(Error::Unsupported)?)?;
        Ok(Self {
            value: value.clone(),
        })
    }
    pub fn to_value(&self) -> Value {
        self.value.clone()
    }
    pub fn project(&self) -> &str {
        self.value["project"].as_str().expect("validated project")
    }
    pub fn source_id(&self) -> &str {
        self.value["source_id"].as_str().expect("validated source")
    }
    pub fn tree_digest(&self) -> &str {
        self.value["tree_digest"]
            .as_str()
            .expect("validated digest")
    }
    pub fn observation(&self) -> &Value {
        &self.value["observation"]
    }
}
pub fn response(
    metadata: &Metadata,
    current_observation: &Value,
    registration: Value,
    policy: &Policy,
) -> Result<Value> {
    policy.validate()?;
    observation(current_observation)?;
    bounded_json(&registration)?;
    let result = json!({"project":metadata.project(),"source_id":metadata.source_id(),"tree_digest":metadata.tree_digest(),"observation":current_observation,"registration":registration});
    bounded_json(&result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn archive_paths_are_bounded_and_protected_without_fs_effects() {
        let policy = Policy::default();
        assert_eq!(
            archive_path(".", MemberKind::Directory, &policy).unwrap(),
            None
        );
        assert!(archive_path("../escape", MemberKind::File, &policy).is_err());
        assert!(archive_path(".env.local", MemberKind::File, &policy).is_err());
        assert!(archive_path("keys/model.bin", MemberKind::File, &policy).is_err());
        assert!(archive_path("link", MemberKind::Symlink, &policy).is_err());
        assert_eq!(
            archive_path("src/main.py", MemberKind::File, &policy).unwrap(),
            Some("src/main.py".into())
        );
    }

    #[test]
    fn streaming_tree_digest_matches_python_length_framed_identity() {
        let policy = Policy::default();
        let plan = TreePlan::new(
            vec![
                TreeEntry {
                    path: "a.txt".into(),
                    kind: TreeKind::File {
                        size: 3,
                        executable: false,
                    },
                },
                TreeEntry {
                    path: "dir".into(),
                    kind: TreeKind::Directory,
                },
                TreeEntry {
                    path: "dir/b.sh".into(),
                    kind: TreeKind::File {
                        size: 1,
                        executable: true,
                    },
                },
            ],
            &policy,
        )
        .unwrap();
        assert_eq!(
            plan.entries()
                .iter()
                .map(|entry| entry.path.as_str())
                .collect::<Vec<_>>(),
            vec!["a.txt", "dir", "dir/b.sh"]
        );
        let mut digest = TreeDigest::new(plan);
        while let Some(chunk) = digest.next_chunk() {
            let bytes = match chunk.path {
                "dir/b.sh" => b"x".as_slice(),
                "a.txt" => b"abc".as_slice(),
                _ => unreachable!(),
            };
            digest = digest.consume(bytes).unwrap();
        }
        assert_eq!(
            digest.finish().unwrap(),
            "sha256:8cd50fc85e242683b983058d966f957a2cbc96b341665ba079d3634c793073ee"
        );
    }

    #[test]
    fn metadata_and_response_keep_transport_sha_separate_from_tree_identity() {
        let policy = Policy::default();
        let observation = json!({"kind":"archive","sha256":"a".repeat(64)});
        let metadata = Metadata::new(
            "demo",
            "sha256:8cd50fc85e242683b983058d966f957a2cbc96b341665ba079d3634c793073ee",
            "2026-10-28T00:00:00Z",
            &observation,
            &policy,
        )
        .unwrap();
        let response =
            response(&metadata, &observation, json!({"existing":true}), &policy).unwrap();
        assert_eq!(
            response["source_id"],
            "source.8cd50fc85e242683b983058d966f957a2cbc96b341665ba079d3634c793073ee"
        );
        assert_eq!(response["observation"]["sha256"], "a".repeat(64));
        assert!(Metadata::from_value(&response, "demo", metadata.source_id(), &policy).is_err());
    }
}
