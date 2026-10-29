//! Immutable dataset identities and resumable gateway delivery. All filesystem,
//! network and clock observations are supplied by the effect shells.
use crate::{artifacts::FileReceipt, canonical, identity, multipart, slurm::helper::Target};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
pub const CONTRACT: &str = "native-dataset.v1";
pub mod import;
pub const MAX_METADATA: usize = 8 * 1024 * 1024;
pub const PART_BYTES: u64 = multipart::DEFAULT_PART_BYTES;
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid dataset identity, delivery state or input binding")]
pub struct Error;
pub type Result<T> = std::result::Result<T, Error>;
/// Payload-free discovery. Cache readiness and archive delivery stay distinct.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Summary {
    pub asset_id: String,
    pub archive_bytes: u64,
    pub file_count: usize,
    pub part_count: u64,
    pub received_parts: u64,
    pub status: String,
}
impl Summary {
    pub fn from_state(state: &State, archive_verified: bool) -> Result<Self> {
        state.validate()?;
        let result = Self {
            asset_id: state.asset_id.clone(),
            archive_bytes: state.definition.archive_bytes,
            file_count: state.definition.files.len(),
            part_count: state.definition.part_count()?,
            received_parts: state.parts.len() as u64,
            status: if state.ready.is_some() {
                "READY"
            } else if archive_verified {
                "ARCHIVED"
            } else {
                "UPLOADING"
            }
            .into(),
        };
        result.validate()?;
        Ok(result)
    }
    pub fn validate(&self) -> Result<()> {
        if !asset_id(&self.asset_id)
            || self.file_count == 0
            || self.file_count > crate::artifacts::MAX_FILES
            || self.part_count
                != multipart::part_count(self.archive_bytes, multipart::Policy::default())
                    .map_err(|_| Error)?
            || self.received_parts > self.part_count
            || !matches!(self.status.as_str(), "READY" | "ARCHIVED" | "UPLOADING")
            || self.status != "UPLOADING" && self.received_parts != self.part_count
        {
            return Err(Error);
        }
        Ok(())
    }
}
fn digest(value: &impl Serialize) -> Result<String> {
    canonical::protocol_identity(&serde_json::to_value(value).map_err(|_| Error)?)
        .map_err(|_| Error)
}
pub fn asset_id(value: &str) -> bool {
    value
        .strip_prefix("asset.")
        .is_some_and(|s| crate::sources::validate_archive_digest(s).is_ok())
}
pub fn path(value: &str) -> bool {
    crate::artifacts::receipt_path(value)
        && value.split('/').count() <= 128
        && !value.chars().any(char::is_control)
        && !value.split('/').any(|s| {
            matches!(s.to_ascii_lowercase().as_str(), "credentials" | "secrets")
                || [".pem", ".key", ".p12", ".pfx"]
                    .iter()
                    .any(|suffix| s.to_ascii_lowercase().ends_with(suffix))
        })
}
pub fn root_path(value: &str) -> bool {
    crate::slurm::absolute_path(value)
        && value != "/"
        && !value.ends_with('/')
        && !value.contains([':', ','])
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Definition {
    pub sha256: String,
    pub archive_bytes: u64,
    pub files: Vec<FileReceipt>,
}
impl Definition {
    pub fn validate(&self) -> Result<()> {
        crate::sources::validate_archive_digest(&self.sha256).map_err(|_| Error)?;
        multipart::part_count(self.archive_bytes, multipart::Policy::default())
            .map_err(|_| Error)?;
        if self.files.is_empty() || self.files.len() > crate::artifacts::MAX_FILES {
            return Err(Error);
        }
        let mut names = BTreeSet::new();
        let mut previous = None;
        let mut total = 0u64;
        for file in &self.files {
            if !path(&file.path)
                || previous.is_some_and(|p: &str| p >= file.path.as_str())
                || !names.insert(file.path.as_str())
            {
                return Err(Error);
            }
            crate::sources::validate_archive_digest(&file.sha256).map_err(|_| Error)?;
            total = total.checked_add(file.bytes).ok_or(Error)?;
            previous = Some(file.path.as_str());
        }
        for file in &self.files {
            let mut parent = file.path.as_str();
            while let Some((next, _)) = parent.rsplit_once('/') {
                if names.contains(next) {
                    return Err(Error);
                }
                parent = next;
            }
        }
        if total > self.archive_bytes
            || serde_json::to_vec(self).map_err(|_| Error)?.len() > MAX_METADATA / 2
        {
            return Err(Error);
        }
        Ok(())
    }
    pub fn asset_id(&self) -> String {
        format!("asset.{}", self.sha256)
    }
    pub fn part_count(&self) -> Result<u64> {
        self.validate()?;
        multipart::part_count(self.archive_bytes, multipart::Policy::default()).map_err(|_| Error)
    }
    pub fn part_size(&self, number: u64) -> Result<u64> {
        multipart::part_size(self.archive_bytes, multipart::Policy::default(), number)
            .map_err(|_| Error)
    }
    pub fn expanded_bytes(&self) -> Result<u64> {
        self.files
            .iter()
            .try_fold(0u64, |sum, f| sum.checked_add(f.bytes).ok_or(Error))
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub contract: String,
    pub project: String,
    pub target: Target,
    pub asset_id: String,
    pub definition: Definition,
    pub root: String,
}
impl Receipt {
    pub fn validate(&self) -> Result<()> {
        identity::validate_id(&self.project).map_err(|_| Error)?;
        self.target.validate().map_err(|_| Error)?;
        self.definition.validate()?;
        if self.contract != CONTRACT
            || self.asset_id != self.definition.asset_id()
            || !root_path(&self.root)
            || !self
                .root
                .ends_with(&format!("/{}/{}/tree", self.project, self.asset_id))
        {
            return Err(Error);
        }
        Ok(())
    }
    pub fn bind(&self, project: &str, target: &Target, data_root: &str) -> Result<()> {
        self.validate()?;
        if !root_path(data_root)
            || self.project != project
            || &self.target != target
            || self.root != format!("{data_root}/{project}/{}/tree", self.asset_id)
        {
            return Err(Error);
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub contract: String,
    pub project: String,
    pub target: Target,
    pub asset_id: String,
    pub definition: Definition,
    pub parts: BTreeMap<u64, multipart::PartRecord>,
    pub ready: Option<Receipt>,
}
impl State {
    pub fn new(project: String, target: Target, definition: Definition) -> Result<Self> {
        let value = Self {
            contract: CONTRACT.into(),
            project,
            target,
            asset_id: definition.asset_id(),
            definition,
            parts: BTreeMap::new(),
            ready: None,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<()> {
        self.definition.validate()?;
        self.target.validate().map_err(|_| Error)?;
        identity::validate_id(&self.project).map_err(|_| Error)?;
        if self.contract != CONTRACT || self.asset_id != self.definition.asset_id() {
            return Err(Error);
        }
        for (number, part) in &self.parts {
            if self.definition.part_size(*number)? != part.bytes
                || crate::sources::validate_archive_digest(&part.sha256).is_err()
            {
                return Err(Error);
            }
        }
        if let Some(ready) = &self.ready {
            ready.validate()?;
            if ready.project != self.project
                || ready.target != self.target
                || ready.definition != self.definition
                || ready.asset_id != self.asset_id
                || self.parts.len() as u64 != self.definition.part_count()?
            {
                return Err(Error);
            }
        }
        if serde_json::to_vec(self).map_err(|_| Error)?.len() > MAX_METADATA - 4096 {
            return Err(Error);
        }
        Ok(())
    }
    pub fn record_part(&self, number: u64, part: multipart::PartRecord) -> Result<Self> {
        self.validate()?;
        if self.definition.part_size(number)? != part.bytes
            || crate::sources::validate_archive_digest(&part.sha256).is_err()
        {
            return Err(Error);
        }
        if let Some(old) = self.parts.get(&number) {
            return if old == &part {
                Ok(self.clone())
            } else {
                Err(Error)
            };
        }
        if self.ready.is_some() {
            return Err(Error);
        }
        let mut next = self.clone();
        next.parts.insert(number, part);
        next.validate()?;
        Ok(next)
    }
    pub fn publish(&self, receipt: Receipt) -> Result<Self> {
        self.validate()?;
        let mut next = self.clone();
        if self.ready.as_ref().is_some_and(|old| old != &receipt) {
            return Err(Error);
        }
        next.ready = Some(receipt);
        next.validate()?;
        Ok(next)
    }
    pub fn validate_successor(&self, next: &Self) -> Result<()> {
        self.validate()?;
        next.validate()?;
        if self.project != next.project
            || self.target != next.target
            || self.definition != next.definition
            || self.parts.iter().any(|(n, p)| next.parts.get(n) != Some(p))
            || self
                .ready
                .as_ref()
                .is_some_and(|r| next.ready.as_ref() != Some(r))
        {
            return Err(Error);
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub name: String,
    pub asset_id: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mount {
    pub name: String,
    pub receipt: Receipt,
}
pub fn inputs(values: &[Input]) -> Result<()> {
    let mut names = BTreeSet::new();
    if values.len() > 32 {
        return Err(Error);
    }
    for v in values {
        identity::validate_id(&v.name).map_err(|_| Error)?;
        if !asset_id(&v.asset_id) || !names.insert(&v.name) {
            return Err(Error);
        }
    }
    Ok(())
}
pub fn mounts(values: &[Mount]) -> Result<()> {
    if serde_json::to_vec(values).map_err(|_| Error)?.len() > 256 * 1024 {
        return Err(Error);
    }
    inputs(
        &values
            .iter()
            .map(|v| Input {
                name: v.name.clone(),
                asset_id: v.receipt.asset_id.clone(),
            })
            .collect::<Vec<_>>(),
    )?;
    for value in values {
        value.receipt.validate()?;
    }
    Ok(())
}
pub fn select(
    inputs_: &[Input],
    receipts: &[Receipt],
    project: &str,
    target: &Target,
    data_root: &str,
) -> Result<Vec<Mount>> {
    inputs(inputs_)?;
    if inputs_.len() != receipts.len() {
        return Err(Error);
    }
    inputs_
        .iter()
        .zip(receipts)
        .map(|(input, receipt)| {
            receipt.bind(project, target, data_root)?;
            if input.asset_id != receipt.asset_id {
                return Err(Error);
            }
            Ok(Mount {
                name: input.name.clone(),
                receipt: receipt.clone(),
            })
        })
        .collect()
}
/// Read only server-frozen bindings, and confine their roots to this exact
/// Project/backend shared-data namespace. Absence preserves old encodings.
pub fn document_mounts(
    document: &serde_json::Value,
    project: &str,
    target: &Target,
    project_data_root: &str,
) -> Result<Vec<Mount>> {
    let values: Vec<Mount> = document
        .get("native_datasets")
        .map(|v| serde_json::from_value(v.clone()).map_err(|_| Error))
        .transpose()?
        .unwrap_or_default();
    mounts(&values)?;
    if !root_path(project_data_root) {
        return Err(Error);
    }
    for value in &values {
        if value.receipt.project != project
            || &value.receipt.target != target
            || value.receipt.root != format!("{project_data_root}/{}/tree", value.receipt.asset_id)
        {
            return Err(Error);
        }
    }
    Ok(values)
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    Begin {
        definition: Definition,
    },
    Status {
        asset_id: String,
    },
    Part {
        asset_id: String,
        number: u64,
        sha256: String,
        bytes: u64,
    },
    Complete {
        asset_id: String,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub version: u16,
    pub project: String,
    pub target: Target,
    pub command: Command,
}
impl Request {
    pub fn validate(&self) -> Result<()> {
        if self.version != 1 {
            return Err(Error);
        }
        identity::validate_id(&self.project).map_err(|_| Error)?;
        self.target.validate().map_err(|_| Error)?;
        match &self.command {
            Command::Begin { definition } => definition.validate()?,
            Command::Status { asset_id: id } | Command::Complete { asset_id: id } => {
                if !asset_id(id) {
                    return Err(Error);
                }
            }
            Command::Part {
                asset_id: id,
                number,
                sha256,
                bytes,
            } => {
                if !asset_id(id)
                    || *number >= multipart::MAX_MAX_PARTS
                    || *bytes == 0
                    || *bytes > PART_BYTES
                    || crate::sources::validate_archive_digest(sha256).is_err()
                {
                    return Err(Error);
                }
            }
        }
        Ok(())
    }
    pub fn asset_id(&self) -> String {
        match &self.command {
            Command::Begin { definition } => definition.asset_id(),
            Command::Status { asset_id }
            | Command::Part { asset_id, .. }
            | Command::Complete { asset_id } => asset_id.clone(),
        }
    }
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        digest(self)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reply {
    pub version: u16,
    pub request_sha256: String,
    pub state: State,
}
impl Reply {
    pub fn validate(&self, request: &Request) -> Result<()> {
        request.validate()?;
        self.state.validate()?;
        if self.version != 1
            || self.request_sha256 != request.digest()?
            || self.state.project != request.project
            || self.state.target != request.target
            || self.state.asset_id != request.asset_id()
        {
            return Err(Error);
        }
        match &request.command {
            Command::Begin { definition } => {
                if self.state.definition != *definition {
                    return Err(Error);
                }
            }
            Command::Part {
                number,
                sha256,
                bytes,
                ..
            } => {
                if self.state.parts.get(number)
                    != Some(&multipart::PartRecord {
                        sha256: sha256.clone(),
                        bytes: *bytes,
                    })
                {
                    return Err(Error);
                }
            }
            Command::Complete { .. } => {
                if self.state.ready.is_none() {
                    return Err(Error);
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn encode(value: &impl Serialize) -> Result<Vec<u8>> {
    let body = serde_json::to_vec(value).map_err(|_| Error)?;
    if body.is_empty() || body.len() > MAX_METADATA {
        return Err(Error);
    }
    let mut frame = (body.len() as u32).to_be_bytes().to_vec();
    frame.extend(body);
    Ok(frame)
}
fn decode<T: serde::de::DeserializeOwned>(frame: &[u8]) -> Result<T> {
    if frame.len() < 5 || frame.len() > MAX_METADATA + 4 {
        return Err(Error);
    }
    let count = u32::from_be_bytes(frame[..4].try_into().map_err(|_| Error)?) as usize;
    if count + 4 != frame.len() {
        return Err(Error);
    }
    let value = crate::json::mapping(&frame[4..]).map_err(|_| Error)?;
    serde_json::from_value(serde_json::Value::Object(value)).map_err(|_| Error)
}
pub fn encode_request(request: &Request) -> Result<Vec<u8>> {
    request.validate()?;
    encode(request)
}
pub fn decode_request(frame: &[u8]) -> Result<Request> {
    let request: Request = decode(frame)?;
    request.validate()?;
    Ok(request)
}
pub fn encode_reply(reply: &Reply) -> Result<Vec<u8>> {
    reply.state.validate()?;
    encode(reply)
}
pub fn decode_reply(frame: &[u8], request: &Request) -> Result<Reply> {
    let reply: Reply = decode(frame)?;
    reply.validate(request)?;
    Ok(reply)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn definition() -> Definition {
        Definition {
            sha256: "1".repeat(64),
            archive_bytes: 5 * 1024 * 1024 * 1024,
            files: vec![FileReceipt {
                path: "train.bin".into(),
                sha256: "2".repeat(64),
                bytes: 5 * 1024 * 1024 * 1024 - 4096,
            }],
        }
    }
    fn target() -> Target {
        Target {
            backend_id: "slurm".into(),
            profile_sha256: format!("sha256:{}", "3".repeat(64)),
            storage_scope: format!("sha256:{}", "4".repeat(64)),
        }
    }
    #[test]
    fn datasets_above_four_gib_are_chunked_without_truncation() {
        let d = definition();
        d.validate().unwrap();
        assert_eq!(d.part_count().unwrap(), 320);
        assert_eq!(d.part_size(319).unwrap(), PART_BYTES);
        assert!(d.part_size(320).is_err());
    }
    #[test]
    fn durable_parts_cannot_disappear_or_change_and_incomplete_data_is_not_ready() {
        let s = State::new("demo".into(), target(), definition()).unwrap();
        let part = multipart::PartRecord {
            sha256: "5".repeat(64),
            bytes: PART_BYTES,
        };
        let next = s.record_part(0, part.clone()).unwrap();
        assert!(next.validate_successor(&s).is_err());
        assert_eq!(next.record_part(0, part).unwrap(), next);
        assert!(
            next.record_part(
                0,
                multipart::PartRecord {
                    sha256: "6".repeat(64),
                    bytes: PART_BYTES
                }
            )
            .is_err()
        );
        let receipt = Receipt {
            contract: CONTRACT.into(),
            project: s.project.clone(),
            target: s.target.clone(),
            asset_id: s.asset_id.clone(),
            definition: s.definition.clone(),
            root: format!("/data/demo/{}/tree", s.asset_id),
        };
        assert!(s.publish(receipt).is_err());
    }
    #[test]
    fn unsafe_paths_and_file_directory_collisions_are_rejected() {
        for p in [
            "../x",
            "/x",
            "hidden/.env",
            "secrets/token",
            "token.key",
            "a\nb",
        ] {
            assert!(!path(p));
        }
        let mut d = definition();
        d.files.push(FileReceipt {
            path: "train.bin/x".into(),
            sha256: "2".repeat(64),
            bytes: 0,
        });
        assert!(d.validate().is_err());
        assert!(!root_path("/data:rw"));
    }
    #[test]
    fn summaries_never_promote_an_incomplete_upload_to_a_ready_cache() {
        let state = State::new("demo".into(), target(), definition()).unwrap();
        let summary = Summary::from_state(&state, false).unwrap();
        assert_eq!(summary.status, "UPLOADING");
        assert_eq!(summary.archive_bytes, 5 * 1024 * 1024 * 1024);
        assert!(Summary::from_state(&state, true).is_err());
        let mut complete = state;
        for number in 0..complete.definition.part_count().unwrap() {
            complete.parts.insert(
                number,
                multipart::PartRecord {
                    sha256: "3".repeat(64),
                    bytes: complete.definition.part_size(number).unwrap(),
                },
            );
        }
        assert_eq!(
            Summary::from_state(&complete, true).unwrap().status,
            "ARCHIVED"
        );
        let receipt = Receipt {
            contract: CONTRACT.into(),
            project: complete.project.clone(),
            target: complete.target.clone(),
            asset_id: complete.asset_id.clone(),
            definition: complete.definition.clone(),
            root: format!("/data/demo/{}/tree", complete.asset_id),
        };
        let ready = complete.publish(receipt).unwrap();
        assert_eq!(Summary::from_state(&ready, true).unwrap().status, "READY");
    }
}
