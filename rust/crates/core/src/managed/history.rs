//! Pure cursor and bounded page validation for an append-only metrics file.
//! The gateway hashes a stable file prefix; the service commits events before
//! advancing its private cursor. Lost replies can repeat only that same page.
use crate::slurm::helper::{self, valid_digest};
use serde::{Deserialize, Serialize};

pub const PAGE_BYTES: usize = 128 * 1024;
pub const MAX_FILE_BYTES: u64 = 128 * 1024 * 1024;
pub const CONTRACT: &str = "managed-metric-history.v1";
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cursor {
    pub device: u64,
    pub inode: u64,
    pub offset_bytes: u64,
    pub prefix_sha256: String,
}
impl Cursor {
    pub fn validate(&self) -> helper::Result<()> {
        if self.offset_bytes > MAX_FILE_BYTES || !valid_digest(&self.prefix_sha256) {
            return Err(helper::Error::Metadata);
        }
        Ok(())
    }
}
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Page {
    pub contract: String,
    pub config_sha256: String,
    pub start: Option<Cursor>,
    pub next: Cursor,
    pub bytes: Vec<u8>,
    pub total_bytes: u64,
    pub process_finished: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub missing_after_finish: bool,
}
impl std::fmt::Debug for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetricHistoryPage")
            .field("next", &self.next)
            .field("bytes", &self.bytes.len())
            .finish_non_exhaustive()
    }
}
impl Page {
    pub fn validate(&self, start: Option<&Cursor>) -> helper::Result<()> {
        self.next.validate()?;
        if let Some(start) = start {
            start.validate()?;
        }
        let offset = start.map(|v| v.offset_bytes).unwrap_or(0);
        if self.contract != CONTRACT
            || !valid_digest(&self.config_sha256)
            || self.start.as_ref() != start
            || self.bytes.len() > PAGE_BYTES
            || self.next.offset_bytes != offset + self.bytes.len() as u64
            || self.next.offset_bytes > self.total_bytes
            || self.total_bytes > MAX_FILE_BYTES
            || self.missing_after_finish
                && (!self.process_finished
                    || self.total_bytes != 0
                    || self.next.device != 0
                    || self.next.inode != 0
                    || start.is_some())
            || self.bytes.len() != ((self.total_bytes - offset).min(PAGE_BYTES as u64) as usize)
            || start.is_some_and(|v| {
                v.device != self.next.device
                    || v.inode != self.next.inode
                    || self.bytes.is_empty() && v.prefix_sha256 != self.next.prefix_sha256
            })
        {
            return Err(helper::Error::Metadata);
        }
        // The first page contains the complete hashed prefix, so verify it
        // locally too. Later pages remain bound by the exact request/receipt.
        if offset == 0 && helper::sha256_bytes(&self.bytes) != self.next.prefix_sha256 {
            return Err(helper::Error::Content);
        }
        Ok(())
    }
    pub fn complete(&self) -> bool {
        self.process_finished && self.next.offset_bytes == self.total_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_metrics_requires_finished_empty_first_page_and_keeps_old_encoding() {
        let page = Page {
            contract: CONTRACT.into(),
            config_sha256: helper::sha256_bytes(b"config"),
            start: None,
            next: Cursor {
                device: 0,
                inode: 0,
                offset_bytes: 0,
                prefix_sha256: helper::sha256_bytes(&[]),
            },
            bytes: Vec::new(),
            total_bytes: 0,
            process_finished: true,
            missing_after_finish: true,
        };
        page.validate(None).unwrap();
        assert!(page.complete());
        let mut bad = page.clone();
        bad.process_finished = false;
        assert!(bad.validate(None).is_err());
        let mut bad = page.clone();
        bad.next.inode = 1;
        assert!(bad.validate(None).is_err());
        let mut bad = page.clone();
        bad.start = Some(page.next.clone());
        assert!(bad.validate(bad.start.as_ref()).is_err());
        let mut present = page;
        present.missing_after_finish = false;
        assert!(
            serde_json::to_value(present)
                .unwrap()
                .get("missing_after_finish")
                .is_none()
        );
    }
    #[test]
    fn page_binding_rejects_mutated_content_regression_and_replaced_files() {
        let bytes = b"{\"step\":1}\n".to_vec();
        let page = Page {
            contract: CONTRACT.into(),
            config_sha256: helper::sha256_bytes(b"config"),
            start: None,
            next: Cursor {
                device: 1,
                inode: 2,
                offset_bytes: bytes.len() as u64,
                prefix_sha256: helper::sha256_bytes(&bytes),
            },
            total_bytes: bytes.len() as u64,
            bytes,
            process_finished: false,
            missing_after_finish: false,
        };
        page.validate(None).unwrap();
        assert!(!page.complete());
        let mut bad = page.clone();
        bad.bytes[0] = b'X';
        assert!(bad.validate(None).is_err());
        let mut bad = page.clone();
        bad.next.offset_bytes -= 1;
        assert!(bad.validate(None).is_err());
        let tail = Page {
            start: Some(page.next.clone()),
            next: page.next.clone(),
            bytes: Vec::new(),
            process_finished: true,
            ..page
        };
        tail.validate(tail.start.as_ref()).unwrap();
        assert!(tail.complete());
        let mut bad = tail.clone();
        bad.next.inode += 1;
        assert!(bad.validate(tail.start.as_ref()).is_err());
        let mut bad = tail.clone();
        bad.next.prefix_sha256 = helper::sha256_bytes(b"changed");
        assert!(bad.validate(tail.start.as_ref()).is_err());
    }
}
