//! Object-store copies of registered checkpoints. Pure layout and receipts only.
//!
//! Each file of a registered generation is cut into `PART_BYTES` objects named by
//! the checkpoint's identity, the file's SHA256 and the part number, so a key names
//! exactly one byte range of one immutable file. The worker sends the bytes directly
//! to the object store with short-lived tickets, computing each part's CRC-64/XZ and
//! each file's SHA256 from the bytes it sends. No byte passes through the server: it
//! asks the object store for each part's size and its own CRC-64 (`HEAD`), requires
//! both to match what the worker sent, and requires every file's SHA256 to be the
//! registered one before it records a receipt. Only that receipt makes the copy the
//! checkpoint's authority.
use super::{Error, FileRecord, Result, validate_checkpoint_id};
use crate::multipart::PART_BYTES;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

pub const CONTRACT: &str = "native-checkpoint-archive.v1";
/// Printed by `ml-exp-worker-rs capabilities`, so these exact bytes are in every worker
/// build that copies checkpoints and reads a delivered `.restore` copy. A server freezes
/// the copy in a new Run only when the worker the Run freezes holds them.
pub const WORKER_CAPABILITY: &str = "ml-expd-worker-capability:checkpoint-archive.v1";

/// Whether a worker file holds the capability marker.
pub fn declares_capability(worker: &[u8]) -> bool {
    worker
        .windows(WORKER_CAPABILITY.len())
        .any(|window| window == WORKER_CAPABILITY.as_bytes())
}
/// The server's independent read of every object; never only the worker's word.
/// The object store's own CRC-64 of every part, as the server read it, matched the
/// worker's; the worker's SHA256 of the same bytes matched the registered file.
pub const VERIFICATION: &str = "oss-crc64+worker-sha256";
/// At most this many tickets per request, so each one is used well before it expires.
pub const TICKETS_PER_REQUEST: u64 = 8;
/// Every part of every file of one checkpoint: 4 TiB. Its CRC list fits one request.
pub const MAX_PARTS: u64 = 1 << 18;
/// Below a restoring Attempt's own `STATE_DIR`. The generation it restores is
/// placed at `.restore/<checkpoint hash>/checkpoints/<generation>`.
pub const RESTORE_DIRECTORY: &str = ".restore";

const CRC64_REFLECTED: u64 = 0xC96C_5795_D787_0F42;
const fn crc64_table() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u64;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ CRC64_REFLECTED
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}
const CRC64_TABLE: [u64; 256] = crc64_table();

/// CRC-64/XZ (ECMA-182 polynomial, reflected, all-ones in and out), which OSS reports
/// as `x-oss-hash-crc64ecma`, as an unsigned decimal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc64(u64);
impl Default for Crc64 {
    fn default() -> Self {
        Self(u64::MAX)
    }
}
impl Crc64 {
    pub fn update(&mut self, bytes: &[u8]) {
        let mut crc = self.0;
        for byte in bytes {
            crc = CRC64_TABLE[((crc ^ u64::from(*byte)) & 0xff) as usize] ^ (crc >> 8);
        }
        self.0 = crc;
    }
    pub fn finish(self) -> u64 {
        self.0 ^ u64::MAX
    }
}
pub fn crc64(bytes: &[u8]) -> u64 {
    let mut crc = Crc64::default();
    crc.update(bytes);
    crc.finish()
}

/// One object: `part` of the file at `file` in the sorted manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Part {
    /// The object's place among every part of the checkpoint, from zero.
    pub index: u64,
    pub file: usize,
    pub part: u64,
    pub offset: u64,
    pub bytes: u64,
}

/// Every object of these files, in manifest order. An empty file has none.
pub fn parts(files: &[FileRecord]) -> Result<Vec<Part>> {
    let mut result = Vec::new();
    for (file, record) in files.iter().enumerate() {
        let count = record.bytes.div_ceil(PART_BYTES);
        for part in 0..count {
            if result.len() as u64 >= MAX_PARTS {
                return Err(Error::Manifest);
            }
            let offset = part * PART_BYTES;
            result.push(Part {
                index: result.len() as u64,
                file,
                part,
                offset,
                bytes: (record.bytes - offset).min(PART_BYTES),
            });
        }
    }
    Ok(result)
}

/// The total object count, without building the list.
pub fn part_count(files: &[FileRecord]) -> Result<u64> {
    let count = files
        .iter()
        .try_fold(0u64, |n, f| n.checked_add(f.bytes.div_ceil(PART_BYTES)))
        .ok_or(Error::Manifest)?;
    if count > MAX_PARTS {
        return Err(Error::Manifest);
    }
    Ok(count)
}

pub fn total_bytes(files: &[FileRecord]) -> Result<u64> {
    files
        .iter()
        .try_fold(0u64, |n, f| n.checked_add(f.bytes))
        .ok_or(Error::Manifest)
}

fn hex(checkpoint_id: &str) -> Result<&str> {
    validate_checkpoint_id(checkpoint_id)?
        .strip_prefix("checkpoint.")
        .ok_or(Error::Identity)
}

/// The object key below the operator's prefix. Every component is already validated
/// by the document, so nothing in it comes from a request.
pub fn object_suffix(
    project: &str,
    run_id: &str,
    attempt_id: &str,
    checkpoint_id: &str,
    file: &FileRecord,
    part: u64,
) -> Result<String> {
    crate::identity::validate_id(project).map_err(|_| Error::Identity)?;
    crate::identity::validate_id(run_id).map_err(|_| Error::Identity)?;
    crate::identity::validate_attempt(attempt_id).map_err(|_| Error::Identity)?;
    if file.sha256.len() != 64 || !file.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(Error::Manifest);
    }
    Ok(format!(
        "checkpoints/{project}/{run_id}/{attempt_id}/{}/{}/{part:05}",
        hex(checkpoint_id)?,
        file.sha256
    ))
}

/// Where a restoring Attempt keeps its own copy of the generation it restores.
pub fn restore_root(attempt_root: &Path, checkpoint_id: &str) -> Result<PathBuf> {
    Ok(attempt_root
        .join("state")
        .join(RESTORE_DIRECTORY)
        .join(hex(checkpoint_id)?))
}

/// The state root that a restoring Attempt's frozen `RESUME_DIR` names: the original
/// Attempt's, or this Attempt's own copy. Nothing else is accepted.
pub fn resume_state_root(
    plan: &super::RestorePlan,
    attempt_root: &Path,
    resume_dir: &str,
) -> Result<String> {
    let own = restore_root(attempt_root, &plan.checkpoint_id)?;
    let own = own.to_str().ok_or(Error::Identity)?;
    for root in [plan.state_root.as_str(), own] {
        if resume_dir == format!("{root}/checkpoints/{}", plan.generation) {
            return Ok(root.to_owned());
        }
    }
    Err(Error::Identity)
}

/// The server's record that the object-store copy holds every byte of the
/// registered document. It is written once and never changed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub contract: String,
    pub checkpoint_id: String,
    pub part_bytes: u64,
    pub parts: u64,
    pub bytes: u64,
    pub verification: String,
    pub verified_at: String,
    /// The object store's CRC-64/XZ of each part, in part order. A restore checks each
    /// downloaded part against it before the file's SHA256.
    pub part_crc64: Vec<u64>,
}
impl Receipt {
    pub fn new(document: &Value, verified_at: &str, part_crc64: Vec<u64>) -> Result<Self> {
        let (_, manifest) = super::validate_document(document, None, None)?;
        if OffsetDateTime::parse(verified_at, &Rfc3339).is_err() {
            return Err(Error::Timestamp);
        }
        if part_crc64.len() as u64 != part_count(&manifest.files)? {
            return Err(Error::Document);
        }
        Ok(Self {
            contract: CONTRACT.into(),
            checkpoint_id: document["checkpoint_id"]
                .as_str()
                .ok_or(Error::Document)?
                .into(),
            part_bytes: PART_BYTES,
            parts: part_count(&manifest.files)?,
            bytes: total_bytes(&manifest.files)?,
            verification: VERIFICATION.into(),
            verified_at: verified_at.into(),
            part_crc64,
        })
    }
    /// This receipt describes exactly this document.
    pub fn validate(&self, document: &Value) -> Result<()> {
        let expected = Self::new(document, &self.verified_at, self.part_crc64.clone())?;
        if *self != expected {
            return Err(Error::Document);
        }
        Ok(())
    }
}

/// One worker request on its Attempt capability.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub package_sha256: String,
    pub operation: Operation,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    /// This Attempt's registered checkpoints and which of them have a verified copy.
    List,
    /// One registered document of this Attempt.
    Document { checkpoint_id: String },
    /// Write tickets for parts `start..` of one of this Attempt's checkpoints.
    Upload { checkpoint_id: String, start: u64 },
    /// What the worker computed from the bytes it sent: each part's CRC-64/XZ in part
    /// order and each file's SHA256 in manifest order. The server checks the parts
    /// against the object store and the files against the registered document.
    Complete {
        checkpoint_id: String,
        part_crc64: Vec<u64>,
        file_sha256: Vec<String>,
    },
    /// Read tickets for parts `start..` of the checkpoint this Attempt restores.
    Download { start: u64 },
    /// From a sync job of an Attempt whose worker never reported its end: it delivered
    /// a packet built from storage as it is now and every registered checkpoint has a
    /// verified copy. The server keeps it only once the training job has ended.
    Synced,
}
impl Operation {
    pub fn checkpoint_id(&self) -> Option<&str> {
        match self {
            Self::Document { checkpoint_id }
            | Self::Upload { checkpoint_id, .. }
            | Self::Complete { checkpoint_id, .. } => Some(checkpoint_id),
            Self::List | Self::Download { .. } | Self::Synced => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn record(path: &str, bytes: u64) -> FileRecord {
        FileRecord {
            path: path.into(),
            bytes,
            sha256: "a".repeat(64),
        }
    }

    #[test]
    fn only_a_worker_that_declares_it_can_copy() {
        let mut build = b"\x7fELF...".to_vec();
        assert!(!declares_capability(&build));
        build.extend_from_slice(WORKER_CAPABILITY.as_bytes());
        build.extend_from_slice(b"...");
        assert!(declares_capability(&build));
        assert!(!declares_capability(
            b"ml-expd-worker-capability:checkpoint-archive.v"
        ));
    }

    #[test]
    fn crc64_is_the_one_oss_reports() {
        // CRC-64/XZ check values: OSS's x-oss-hash-crc64ecma.
        assert_eq!(crc64(b""), 0);
        assert_eq!(crc64(b"123456789"), 0x995D_C9BB_DF19_39FA);
        let mut split = Crc64::default();
        split.update(b"1234");
        split.update(b"56789");
        assert_eq!(split.finish(), 0x995D_C9BB_DF19_39FA);
        assert_ne!(crc64(b"123456780"), 0x995D_C9BB_DF19_39FA);
    }

    #[test]
    fn parts_cover_every_byte_once_and_skip_empty_files() {
        let files = [
            record("checkpoints/g/a", PART_BYTES * 2 + 5),
            record("checkpoints/g/b", 0),
            record("checkpoints/g/c", 3),
        ];
        let parts = parts(&files).unwrap();
        assert_eq!(parts.len() as u64, part_count(&files).unwrap());
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[2].bytes, 5);
        assert_eq!(parts[2].offset, PART_BYTES * 2);
        assert_eq!((parts[3].file, parts[3].part, parts[3].bytes), (2, 0, 3));
        assert!(parts.iter().enumerate().all(|(i, p)| p.index == i as u64));
        assert_eq!(total_bytes(&files).unwrap(), PART_BYTES * 2 + 8);
    }

    #[test]
    fn keys_and_restore_paths_name_only_validated_components() {
        let id = format!("checkpoint.{}", "b".repeat(64));
        let file = record("checkpoints/g/a", 1);
        assert_eq!(
            object_suffix("demo", "train", "attempt-002", &id, &file, 3).unwrap(),
            format!(
                "checkpoints/demo/train/attempt-002/{}/{}/00003",
                "b".repeat(64),
                "a".repeat(64)
            )
        );
        assert!(object_suffix("../x", "train", "attempt-002", &id, &file, 0).is_err());
        assert!(object_suffix("demo", "train", "attempt-002", "checkpoint.x", &file, 0).is_err());
        assert_eq!(
            restore_root(Path::new("/r/demo/train/attempt-002"), &id).unwrap(),
            PathBuf::from(format!(
                "/r/demo/train/attempt-002/state/.restore/{}",
                "b".repeat(64)
            ))
        );
    }

    #[test]
    fn resume_dir_names_only_the_original_or_the_attempts_own_copy() {
        let plan = super::super::RestorePlan {
            checkpoint_id: format!("checkpoint.{}", "e".repeat(64)),
            step: 1,
            generation: "s1".into(),
            state_root: "/r/demo/train/attempt-001/state".into(),
            files: Vec::new(),
        };
        let attempt = Path::new("/r/demo/train/attempt-002");
        let own = format!(
            "/r/demo/train/attempt-002/state/.restore/{}",
            "e".repeat(64)
        );
        assert_eq!(
            resume_state_root(
                &plan,
                attempt,
                "/r/demo/train/attempt-001/state/checkpoints/s1"
            )
            .unwrap(),
            plan.state_root
        );
        assert_eq!(
            resume_state_root(&plan, attempt, &format!("{own}/checkpoints/s1")).unwrap(),
            own
        );
        for other in [
            "/r/demo/train/attempt-001/state/checkpoints/s2",
            "/tmp/checkpoints/s1",
            "/r/demo/train/attempt-002/state/checkpoints/s1",
        ] {
            assert!(resume_state_root(&plan, attempt, other).is_err(), "{other}");
        }
    }

    #[test]
    fn a_receipt_binds_the_exact_document() {
        let context = super::super::Context {
            project: "demo".into(),
            run_id: "trial".into(),
            attempt_id: "attempt-001".into(),
            source_id: format!("source.{}", "a".repeat(64)),
            image_id: format!("sha256:{}", "b".repeat(64)),
            storage_scope: format!("storage.{}", "c".repeat(64)),
            state_root: "/data/demo/trial/attempt-001/state".into(),
        };
        let ready = json!({"step":1,"files":[{"path":"checkpoints/s1/m.pt","bytes":7,"sha256":"d".repeat(64)}]});
        let document = super::super::document(&context, &ready).unwrap();
        let receipt = Receipt::new(&document, "2026-11-08T00:00:00Z", vec![42]).unwrap();
        assert_eq!((receipt.parts, receipt.bytes), (1, 7));
        assert_eq!(receipt.verification, "oss-crc64+worker-sha256");
        assert!(Receipt::new(&document, "2026-11-08T00:00:00Z", vec![]).is_err());
        assert!(receipt.validate(&document).is_ok());
        let mut other = receipt.clone();
        other.parts = 2;
        assert!(other.validate(&document).is_err());
        assert!(Receipt::new(&document, "yesterday", vec![42]).is_err());
        let request: Request = serde_json::from_value(json!({"package_sha256":"x","operation":{"kind":"upload","checkpoint_id":"c","start":0}})).unwrap();
        assert_eq!(request.operation.checkpoint_id(), Some("c"));
        assert!(
            serde_json::from_value::<Request>(
                json!({"package_sha256":"x","operation":{"kind":"delete","checkpoint_id":"c"}})
            )
            .is_err()
        );
    }
}
