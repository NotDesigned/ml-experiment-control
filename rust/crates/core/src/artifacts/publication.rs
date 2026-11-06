//! Backend-resident result bundles. Choices and identities are pure; only an
//! explicit verified ACK can authorize deleting the generated download copy.
use super::*;
use crate::slurm::helper::{FrozenIntent, valid_digest};

pub const CONTRACT: &str = "backend-artifacts.v1";
pub const MAX_SKIPPED_ROWS: usize = 1024;
const MAX_SKIPPED_PATH: usize = 512;
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    #[serde(default)]
    pub required_files: Vec<String>,
}
impl Policy {
    pub fn validate(&self) -> Result<()> {
        let mut names = BTreeSet::new();
        // Required paths are relative to OUTPUT_DIR. A leading `outputs` is almost
        // certainly that mistake, and accepting it would only surface after training.
        if self.required_files.len() > MAX_FILES
            || self.required_files.iter().any(|p| {
                !export_path(p) || p == "outputs" || p.starts_with("outputs/") || !names.insert(p)
            })
        {
            return Err(Error::Receipt);
        }
        Ok(())
    }
    pub fn check(&self, files: &[FileReceipt]) -> Result<()> {
        self.validate()?;
        if self
            .required_files
            .iter()
            .any(|p| !files.iter().any(|f| f.path == *p))
        {
            return Err(Error::Receipt);
        }
        Ok(())
    }
}
pub fn export_path(path: &str) -> bool {
    receipt_path(path) && !path.chars().any(char::is_control)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkipReason {
    SymbolicLink,
    SpecialFile,
    /// The name is not UTF-8, or holds a control character or backslash, or the path
    /// is too long or too deep.
    UnsupportedPath,
    /// The archive holds as many files as a receipt can list.
    FileLimit,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkippedPath {
    pub path: String,
    pub reason: SkipReason,
}
/// What the archive leaves out. This list is the only record of what is missing:
/// nothing else compares the archive with the original directory. A row naming a
/// directory covers what lies beneath it, though a required file there is still
/// archived. `.` is OUTPUT_DIR itself. A path is for reading, not for opening: control
/// characters are escaped and a very long path is cut short. `unlisted` counts entries
/// skipped after the rows ran out or past a directory's listing limit.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Skipped {
    pub paths: Vec<SkippedPath>,
    pub unlisted: u64,
}
impl Skipped {
    pub fn add(&mut self, path: &str, reason: SkipReason) {
        if self.paths.len() >= MAX_SKIPPED_ROWS {
            self.unlisted = self.unlisted.saturating_add(1);
            return;
        }
        let mut shown = String::new();
        for c in path.chars() {
            if shown.len() >= MAX_SKIPPED_PATH {
                shown.push_str("...");
                break;
            }
            if c.is_control() {
                shown.extend(c.escape_default());
            } else {
                shown.push(c);
            }
        }
        self.paths.push(SkippedPath {
            path: shown,
            reason,
        });
    }
    fn validate(&self) -> Result<()> {
        if self.paths.len() > MAX_SKIPPED_ROWS
            || self.paths.iter().any(|row| {
                row.path.is_empty()
                    || row.path.len() > MAX_SKIPPED_PATH + 16
                    || row.path.chars().any(char::is_control)
            })
        {
            return Err(Error::Receipt);
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bundle {
    pub contract: String,
    pub intent_sha256: String,
    pub config_sha256: String,
    pub archive: ArchiveReceipt,
    pub skipped: Skipped,
}
impl Bundle {
    pub fn validate(&self) -> Result<()> {
        validate_receipt(&self.archive)?;
        self.skipped.validate()?;
        if self.contract != CONTRACT
            || !valid_digest(&self.intent_sha256)
            || !valid_digest(&self.config_sha256)
            || self.archive.files.iter().any(|f| !export_path(&f.path))
        {
            return Err(Error::Receipt);
        }
        Ok(())
    }
    pub fn bind(&self, intent: &FrozenIntent, config: Option<&str>) -> Result<()> {
        self.validate()?;
        if self.intent_sha256 != intent.digest().map_err(|_| Error::Identity)?
            || self.archive.project != intent.scope.project
            || self.archive.run_id != intent.scope.run_id
            || self.archive.attempt_id != intent.scope.attempt_id
            || config.is_some_and(|v| v != self.config_sha256)
        {
            return Err(Error::Identity);
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub bundle: Bundle,
    pub retention: Retention,
    pub acknowledgement: Option<Acknowledgement>,
}
impl State {
    pub fn new(bundle: Bundle) -> Result<Self> {
        bundle.validate()?;
        Ok(Self {
            retention: Retention {
                status: RetentionStatus::Keep,
                sha256: bundle.archive.sha256.clone(),
                release_after: None,
                released_at: None,
            },
            bundle,
            acknowledgement: None,
        })
    }
    pub fn validate(&self) -> Result<()> {
        self.bundle.validate()?;
        validate_retention(&self.retention, &self.bundle.archive.sha256)?;
        if let Some(ack) = &self.acknowledgement {
            proof_matches(&self.bundle.archive, &ack.proof)?;
            timestamp(&ack.verified_at)?;
            if self.retention.status != RetentionStatus::Keep
                && self.retention.release_after.as_ref() != Some(&ack.verified_at)
            {
                return Err(Error::Transition);
            }
        } else if self.retention.status != RetentionStatus::Keep {
            return Err(Error::Transition);
        }
        Ok(())
    }
    /// Remote observations may complete an ACK, but cannot undo a durable
    /// local release intent after its response was lost.
    pub fn validate_successor(&self, next: &Self) -> Result<()> {
        self.validate()?;
        next.validate()?;
        let rank = |status| match status {
            RetentionStatus::Keep => 0,
            RetentionStatus::Scheduled => 1,
            RetentionStatus::Releasing => 2,
            RetentionStatus::Released => 3,
        };
        if self.bundle != next.bundle
            || rank(next.retention.status) < rank(self.retention.status)
            || self.retention.status == RetentionStatus::Released && self != next
        {
            return Err(Error::Transition);
        }
        if let Some(old) = &self.acknowledgement {
            if next.acknowledgement.as_ref().map(|a| &a.proof) != Some(&old.proof) {
                return Err(Error::ProofConflict);
            }
        }
        Ok(())
    }
    pub fn available(&self) -> Result<bool> {
        self.validate()?;
        Ok(self.retention.status == RetentionStatus::Keep)
    }
    /// An exact duplicate ACK is idempotent, including after release. A KEEP
    /// receipt can later be explicitly released; release cannot revert to KEEP.
    pub fn acknowledge(
        &self,
        proof: DownloadProof,
        verified_at: &str,
        release: bool,
    ) -> Result<Self> {
        self.validate()?;
        let (ack, retention) =
            super::acknowledge(&self.bundle.archive, proof, verified_at, release)?;
        if let Some(old) = &self.acknowledgement {
            if old.proof != ack.proof {
                return Err(Error::ProofConflict);
            }
            if self.retention.status != RetentionStatus::Keep || !release {
                return Ok(self.clone());
            }
        }
        Ok(Self {
            bundle: self.bundle.clone(),
            retention,
            acknowledgement: Some(ack),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    Collect,
    Download {
        sha256: String,
    },
    Acknowledge {
        proof: DownloadProof,
        verified_at: String,
        release: bool,
    },
}
impl Command {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Download { sha256: value } if !sha256(value) => Err(Error::Identity),
            Self::Acknowledge {
                proof, verified_at, ..
            } => {
                timestamp(verified_at)?;
                if !sha256(&proof.archive_sha256)
                    || proof.archive_bytes == 0
                    || proof.files.len() > MAX_FILES
                {
                    return Err(Error::ProofConflict);
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bundle() -> Bundle {
        Bundle {
            contract: CONTRACT.into(),
            intent_sha256: format!("sha256:{}", "1".repeat(64)),
            config_sha256: format!("sha256:{}", "2".repeat(64)),
            archive: ArchiveReceipt {
                project: "demo".into(),
                run_id: "train".into(),
                attempt_id: "attempt-001".into(),
                sha256: "3".repeat(64),
                bytes: 2048,
                files: vec![FileReceipt {
                    path: "weights.bin".into(),
                    bytes: 4,
                    sha256: "4".repeat(64),
                }],
                object_key: format!("demo/train/attempt-001/{}.tar", "3".repeat(64)),
            },
            skipped: Skipped::default(),
        }
    }
    fn proof(state: &State) -> DownloadProof {
        DownloadProof {
            archive_sha256: state.bundle.archive.sha256.clone(),
            archive_bytes: state.bundle.archive.bytes,
            files: state
                .bundle
                .archive
                .files
                .iter()
                .map(|f| (format!("outputs/{}", f.path), f.clone()))
                .collect(),
        }
    }
    #[test]
    fn verified_ack_is_monotonic_and_released_bundles_never_resurrect() {
        let state = State::new(bundle()).unwrap();
        assert!(state.available().unwrap());
        let keep = state
            .acknowledge(proof(&state), "2026-10-29T00:00:00Z", false)
            .unwrap();
        assert!(keep.available().unwrap());
        let mut release = keep
            .acknowledge(proof(&state), "2026-10-29T00:01:00Z", true)
            .unwrap();
        assert!(!release.available().unwrap());
        release.retention = begin_release(&release.bundle.archive, &release.retention).unwrap();
        release.retention = finish_release(
            &release.bundle.archive,
            &release.retention,
            "2026-10-29T00:02:00Z",
        )
        .unwrap();
        assert_eq!(
            release
                .acknowledge(proof(&state), "2026-10-29T00:03:00Z", false)
                .unwrap(),
            release
        );
        assert!(!release.available().unwrap());
        let mut bad = proof(&state);
        bad.files.clear();
        assert!(
            release
                .acknowledge(bad, "2026-10-29T00:04:00Z", true)
                .is_err()
        );
        let mut bad = release.clone();
        bad.acknowledgement = None;
        assert!(bad.validate().is_err());
    }
    #[test]
    fn stale_remote_keep_cannot_undo_a_saved_release_intent() {
        let state = State::new(bundle()).unwrap();
        let keep = state
            .acknowledge(proof(&state), "2026-10-29T00:00:00Z", false)
            .unwrap();
        let release = keep
            .acknowledge(proof(&state), "2026-10-29T00:01:00Z", true)
            .unwrap();
        assert!(keep.validate_successor(&state).is_err());
        assert!(release.validate_successor(&keep).is_err());
        assert!(state.validate_successor(&release).is_ok());
        let mut releasing = release.clone();
        releasing.retention = begin_release(&release.bundle.archive, &release.retention).unwrap();
        assert!(releasing.validate_successor(&release).is_err());
        let mut completed = releasing.clone();
        completed.retention = finish_release(
            &release.bundle.archive,
            &releasing.retention,
            "2026-10-29T00:02:00Z",
        )
        .unwrap();
        assert!(release.validate_successor(&completed).is_ok());
        assert!(completed.validate_successor(&completed).is_ok());
        assert!(completed.validate_successor(&releasing).is_err());
    }
    #[test]
    fn scientific_output_requirements_can_explicitly_include_checkpoints() {
        let mut policy = Policy {
            required_files: vec!["weights.bin".into()],
        };
        assert!(policy.check(&bundle().archive.files).is_ok());
        policy.required_files.push("validation.json".into());
        assert!(policy.check(&bundle().archive.files).is_err());
        for path in ["../other", "/weights", "data\nfile"] {
            assert!(!export_path(path));
        }
        let mut invalid = bundle();
        invalid.archive.files[0].path = "checkpoints/state.pt".into();
        assert!(invalid.validate().is_ok());
    }
    #[test]
    fn a_required_path_may_be_hidden_but_is_never_written_from_above_output_dir() {
        for path in [".hydra/config.yaml", "models/outputs/final.pt"] {
            let policy = Policy {
                required_files: vec![path.into()],
            };
            assert!(policy.validate().is_ok(), "{path}");
        }
        for path in ["outputs", "outputs/final.pt", ".", "../final.pt"] {
            let policy = Policy {
                required_files: vec![path.into()],
            };
            assert!(policy.validate().is_err(), "{path}");
        }
        // What the walk archives is wider: such a file still leaves and can be named.
        assert!(export_path("outputs/final.pt"));
        let mut hidden = bundle();
        hidden.archive.files[0].path = "outputs/.hydra/config.yaml".into();
        assert!(hidden.validate().is_ok());
    }
    #[test]
    fn the_skip_list_is_bounded_readable_and_counts_what_it_cannot_list() {
        let mut skipped = Skipped::default();
        skipped.add("run\nlog\u{7}", SkipReason::UnsupportedPath);
        assert_eq!(skipped.paths[0].path, "run\\nlog\\u{7}");
        skipped.add(&"long/".repeat(1000), SkipReason::SymbolicLink);
        assert!(skipped.paths[1].path.len() <= MAX_SKIPPED_PATH + 3);
        for _ in 2..MAX_SKIPPED_ROWS + 5 {
            skipped.add("link", SkipReason::SymbolicLink);
        }
        assert_eq!(skipped.paths.len(), MAX_SKIPPED_ROWS);
        assert_eq!(skipped.unlisted, 5);
        let mut bundle = bundle();
        bundle.skipped = skipped;
        assert!(bundle.validate().is_ok());
        let encoded = serde_json::to_string(&bundle.skipped.paths[0]).unwrap();
        assert!(
            encoded.contains(r#""reason":"unsupported_path""#),
            "{encoded}"
        );
        bundle.skipped.paths[0].path = "raw\ncontrol".into();
        assert!(bundle.validate().is_err());
        bundle.skipped.paths[0].path = String::new();
        assert!(bundle.validate().is_err());
    }
}
