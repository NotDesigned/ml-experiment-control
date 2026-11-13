//! The rest of an Attempt's `metrics.jsonl`, from its published result archive. The
//! live stream sends one page per tick and stops when its drain budget runs out, so a
//! Run that logged quickly can end with only a prefix in the journal. The archive holds
//! the whole file. Its bytes before the stream's cursor must hash to the cursor's prefix,
//! or nothing is imported; after it, the same importer state continues where the stream
//! stopped. Every event ID derives from content, so a repeat after a crash adds nothing.
//! Nothing here starts a job or changes the archive.
use super::*;
use ml_exp_core::{
    artifacts::{ArchiveReceipt, FileReceipt},
    records::{Event, ImportCounts, JsonlImporter},
};
use sha2::{Digest, Sha256};
use std::io::{Read, Seek, Write};

const CONTRACT: &str = "native-metric-gap-fill.v1";
const METRICS: &str = "metrics.jsonl";
/// The largest archived metrics file read. A larger one is refused and says so.
pub const MAX_METRIC_BYTES: u64 = 256 * 1024 * 1024;
const LEADER: &str = ".metric-gap.lock";
/// How often the background fill looks for newly published archives.
const PASS_SECONDS: u64 = 30;
/// Input imported under one hold of the Run's record guard, so a deletion waits for
/// one slice rather than the whole file.
const SLICE_BYTES: u64 = 1024 * 1024;

/// What the fill of one Attempt did. Written once, when it has an answer; a transient
/// failure writes nothing and the next pass tries again.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Filled {
    contract: String,
    /// `FILLED`, or `REFUSED` with the reason in `diagnostic`. Either is final.
    pub(super) status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    diagnostic: Option<String>,
    archive_sha256: String,
    /// Where the live stream stopped.
    from_bytes: u64,
    /// The end of the archived file, once it was imported.
    pub(super) to_bytes: u64,
    /// Lines read from the archive after the stream's cursor.
    pub(super) records: u64,
    pub(super) invalid_lines: u64,
    at: String,
}
/// Where an Attempt's fill stands.
pub(super) enum Gap {
    /// Nothing to fill and nothing will be: the stream completed, or no archive is made.
    Absent,
    /// The archive is not published yet, or the stream may still deliver.
    Waiting,
    Done(Filled),
}
enum Fetched {
    /// The verified bytes after the stream's cursor, in an anonymous file.
    Suffix(std::fs::File),
    Refused(&'static str),
}
fn json_error(_: serde_json::Error) -> cloud::Error {
    cloud::Error::evidence()
}
fn storage(_: std::io::Error) -> cloud::Error {
    cloud::Error {
        code: "METRIC_GAP_FILL_UNAVAILABLE".into(),
        http_status: Some(503),
        uncertain: false,
    }
}

impl Service {
    fn gap_name(prepared: &Prepared) -> cloud::Result<String> {
        let job = &prepared.execution.job;
        Ok(format!(
            "{}.metric-gap.json",
            Self::attempt_slot(&job.project, &job.run_id, &job.attempt_id)?
        ))
    }
    pub(super) fn gap_saved(&self, prepared: &Prepared) -> cloud::Result<Option<Filled>> {
        let name = Self::gap_name(prepared)?;
        if !self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            return Ok(None);
        }
        let saved: Filled = serde_json::from_value(read(&self.slots, &name)?)
            .map_err(why!(cloud::Error::evidence()))?;
        if saved.contract != CONTRACT
            || !matches!(saved.status.as_str(), "FILLED" | "REFUSED")
            || (saved.status == "REFUSED") != saved.diagnostic.is_some()
        {
            return Err(cloud::Error::evidence());
        }
        Ok(Some(saved))
    }
    /// The first answer stays: a second fill that finished meanwhile changes nothing.
    fn gap_save(&self, prepared: &Prepared, record: Filled) -> cloud::Result<Filled> {
        let _lock = self.lock("metric_gap.save")?;
        if let Some(saved) = self.gap_saved(prepared)? {
            return Ok(saved);
        }
        let b = &prepared.execution.binding;
        self.require_live(&b.project, &b.run_id)?;
        self.slots
            .replace_regular(OsStr::new(&Self::gap_name(prepared)?), &encode(&record)?)
            .map_err(why!(cloud::Error {
                code: "METRIC_GAP_FILL_COMMIT_UNCERTAIN".into(),
                http_status: None,
                uncertain: true,
            }))?;
        Ok(record)
    }
    /// What a status view shows: the fill's record, if it has one.
    pub(super) fn gap_document(&self, prepared: &Prepared) -> cloud::Result<Value> {
        Ok(self
            .gap_saved(prepared)?
            .map_or(Value::Null, |saved| json!(saved)))
    }
    /// Import whatever of the archived `metrics.jsonl` the live stream did not deliver.
    /// Safe to repeat at any point: the journal skips every event it already holds.
    pub(super) fn fill_metric_gap(&self, prepared: &Prepared, now: &str) -> cloud::Result<Gap> {
        if let Some(saved) = self.gap_saved(prepared)? {
            return Ok(Gap::Done(saved));
        }
        let Some(objects) = self.objects.as_ref() else {
            return Ok(Gap::Absent);
        };
        let package = prepared
            .execution
            .package()
            .map_err(why!(cloud::Error::evidence()))?;
        if !package.result_upload {
            return Ok(Gap::Absent);
        }
        let b = &prepared.execution.binding;
        self.require_live(&b.project, &b.run_id)?;
        let Some(stream) = self.exchange_saved(prepared)? else {
            return Ok(Gap::Waiting);
        };
        // A drained stream delivered the whole file: nothing is read from the archive.
        if stream.complete {
            return Ok(Gap::Absent);
        }
        let Some(importer) = stream.importer.clone() else {
            return Ok(Gap::Absent);
        };
        let Some(artifacts) = self.artifact_saved(prepared)? else {
            return Ok(Gap::Waiting);
        };
        let Some(state) = &artifacts.state else {
            return Ok(Gap::Waiting);
        };
        let archive = &state.bundle.archive;
        let from = stream.cursor.as_ref().map_or(0, |c| c.offset_bytes);
        // The stream already reached the archived file's end: there is nothing to import.
        if archive
            .files
            .iter()
            .any(|f| f.path == METRICS && f.bytes == from)
        {
            return Ok(Gap::Absent);
        }
        let record = |status: &str, diagnostic: Option<&str>| Filled {
            contract: CONTRACT.into(),
            status: status.into(),
            diagnostic: diagnostic.map(Into::into),
            archive_sha256: archive.sha256.clone(),
            from_bytes: from,
            to_bytes: from,
            records: 0,
            invalid_lines: 0,
            at: now.into(),
        };
        let refuse = |code: &'static str| -> cloud::Result<Gap> {
            eprintln!(
                "metric gap fill of {}/{}/{} refused: {code}",
                b.project, b.run_id, b.attempt_id
            );
            let saved = self.gap_save(prepared, record("REFUSED", Some(code)))?;
            Ok(Gap::Done(saved))
        };
        let available = state.available().map_err(why!(cloud::Error::evidence()))?;
        if artifacts.administrative_release.is_some() || !available {
            return refuse("METRICS_ARCHIVE_RELEASED");
        }
        let Some(file) = archive.files.iter().find(|f| f.path == METRICS) else {
            return refuse("METRICS_NOT_ARCHIVED");
        };
        if file.bytes > MAX_METRIC_BYTES {
            return refuse("METRICS_FILE_TOO_LARGE");
        }
        if file.bytes < from {
            return refuse("METRICS_PREFIX_MISMATCH");
        }
        let prefix = stream.cursor.as_ref().map(|c| c.prefix_sha256.as_str());
        let spool = match self.archived_metrics(objects, archive, file, from, prefix)? {
            Fetched::Suffix(spool) => spool,
            Fetched::Refused(code) => return refuse(code),
        };
        let records = self.exchange_records(prepared)?;
        let config = {
            let guard = records
                .guard(&b.project, &b.run_id)
                .map_err(why!(cloud::Error::evidence()))?;
            records
                .binding(&guard)
                .map_err(why!(cloud::Error::evidence()))?
                .ok_or_else(cloud::Error::evidence)?["config"]
                .clone()
        };
        if !importer.matches(METRICS, &config) {
            return refuse("METRICS_IMPORTER_MISMATCH");
        }
        let counts = self.import_suffix(prepared, &records, importer, spool)?;
        // The journal now ends where the archived file ends, as a drained stream would.
        let worker = &stream.evidence.worker;
        let payload = json!({"kind":"lifecycle","data":{"phase":"METRICS_DRAINED",
            "state":super::exchange::training_state(worker),"timestamp":null,
            "exit_code":worker.receipt.as_ref().and_then(|r|r.process.exit_code),
            "delivery":"result-archive","archive_sha256":archive.sha256}});
        let digest = canonical::digest(&payload).map_err(why!(cloud::Error::evidence()))?;
        let event = Event {
            event_id: format!("worker.cloud.archive.{digest}"),
            payload,
        };
        {
            let guard = records
                .guard(&b.project, &b.run_id)
                .map_err(why!(cloud::Error::evidence()))?;
            let scope = records
                .scope(&guard, &b.attempt_id)
                .map_err(why!(cloud::Error::evidence()))?;
            records
                .append(&guard, &scope, &[event])
                .map_err(why!(cloud::Error::evidence()))?;
        }
        let mut filled = record("FILLED", None);
        filled.to_bytes = file.bytes;
        filled.records = counts.records.saturating_sub(stream.counts.records);
        filled.invalid_lines = counts
            .invalid_lines
            .saturating_sub(stream.counts.invalid_lines);
        Ok(Gap::Done(self.gap_save(prepared, filled)?))
    }
    /// Read the member from the archive, check its header, its whole SHA256 and the
    /// stream's prefix hash, and keep the bytes after `from`. No lock is held.
    fn archived_metrics(
        &self,
        objects: &crate::objects::Store,
        archive: &ArchiveReceipt,
        file: &FileReceipt,
        from: u64,
        prefix: Option<&str>,
    ) -> cloud::Result<Fetched> {
        let layout = "METRICS_ARCHIVE_LAYOUT_UNRECOGNIZED";
        let Some(offset) = ml_exp_io::artifacts::result_member_offset(&archive.files, METRICS)
        else {
            return Ok(Fetched::Refused("METRICS_NOT_ARCHIVED"));
        };
        let Ok(expected) = ml_exp_io::artifacts::result_member_header(METRICS, file.bytes) else {
            return Ok(Fetched::Refused(layout));
        };
        let length = file.bytes + 512;
        if offset
            .checked_add(length)
            .is_none_or(|end| end > archive.bytes)
        {
            return Ok(Fetched::Refused(layout));
        }
        let mut reader = objects
            .read_archive_range(archive, offset, length)
            .map_err(storage)?;
        let mut header = [0u8; 512];
        reader.read_exact(&mut header).map_err(storage)?;
        if header != expected {
            return Ok(Fetched::Refused(layout));
        }
        // Anonymous, beside the registry rather than in a temporary filesystem that may
        // be memory: it disappears with the handle, including after a crash.
        let mut spool =
            tempfile::tempfile_in(self.settings.registry_root.join(NAME)).map_err(storage)?;
        let mut hash = Sha256::new();
        let mut before = from;
        let mut seen = if from == 0 {
            Some(helper::sha256_bytes(&[]))
        } else {
            None
        };
        let mut total = 0u64;
        let mut buffer = vec![0u8; 64 * 1024];
        loop {
            let count = reader.read(&mut buffer).map_err(storage)?;
            if count == 0 {
                break;
            }
            total += count as u64;
            let mut chunk = &buffer[..count];
            if before > 0 {
                let head = before.min(count as u64) as usize;
                hash.update(&chunk[..head]);
                before -= head as u64;
                chunk = &chunk[head..];
                if before == 0 {
                    seen = Some(format!("sha256:{:x}", hash.clone().finalize()));
                }
            }
            hash.update(chunk);
            spool.write_all(chunk).map_err(storage)?;
        }
        if total != file.bytes {
            let short = std::io::Error::other("archived metrics were cut short");
            return Err(storage(short));
        }
        if format!("{:x}", hash.finalize()) != file.sha256 {
            return Ok(Fetched::Refused("METRICS_ARCHIVE_DIGEST_MISMATCH"));
        }
        if prefix.is_some_and(|prefix| seen.as_deref() != Some(prefix)) {
            return Ok(Fetched::Refused("METRICS_PREFIX_MISMATCH"));
        }
        spool.rewind().map_err(storage)?;
        Ok(Fetched::Suffix(spool))
    }
    /// The importer the stream saved, fed the rest of the file in bounded chunks and
    /// finished. Each slice holds the Run's record guard, which keeps its fences.
    fn import_suffix(
        &self,
        prepared: &Prepared,
        records: &crate::records::Store,
        mut value: JsonlImporter,
        mut spool: std::fs::File,
    ) -> cloud::Result<ImportCounts> {
        let b = &prepared.execution.binding;
        let mut slice = Vec::with_capacity(SLICE_BYTES as usize);
        loop {
            slice.clear();
            (&mut spool)
                .take(SLICE_BYTES)
                .read_to_end(&mut slice)
                .map_err(storage)?;
            let guard = records
                .guard(&b.project, &b.run_id)
                .map_err(why!(cloud::Error::evidence()))?;
            let scope = records
                .scope(&guard, &b.attempt_id)
                .map_err(why!(cloud::Error::evidence()))?;
            if slice.is_empty() {
                let emission = value.finish().map_err(why!(cloud::Error::evidence()))?;
                records
                    .append_import(&guard, &scope, &emission)
                    .map_err(why!(cloud::Error::evidence()))?;
                return Ok(emission.counts);
            }
            for chunk in slice.chunks(JsonlImporter::INPUT_BOUND) {
                let (next, emission) = value.feed(chunk).map_err(why!(cloud::Error::evidence()))?;
                records
                    .append_import(&guard, &scope, &emission)
                    .map_err(why!(cloud::Error::evidence()))?;
                value = next;
            }
        }
    }
    /// The fill of one Action's Attempt, for a caller that names it.
    pub fn fill_metric_gap_for(&self, action: &str, now: &str) -> cloud::Result<Value> {
        let prepared = self.load(action)?;
        let gap = self.fill_metric_gap(&prepared, now)?;
        Ok(match gap {
            Gap::Absent => json!({"submission_id":action,"status":"NOT_NEEDED"}),
            Gap::Waiting => json!({"submission_id":action,"status":"WAITING"}),
            Gap::Done(saved) => {
                let mut view = json!(saved);
                view["submission_id"] = json!(action);
                view
            }
        })
    }
    /// Every Attempt looked at once. One Attempt's error never stops the others, and
    /// writes nothing, so the next pass tries again. `done` keeps the slots that need no
    /// further look.
    pub fn metric_gap_pass(&self, now: &str, done: &mut std::collections::BTreeSet<String>) {
        if !self.has_artifacts() {
            return;
        }
        let Ok(names) = self.slots.entry_names() else {
            return;
        };
        for name in names {
            let Some(name) = name.to_str() else {
                continue;
            };
            if done.contains(name) || !super::super::observer::preparation_slot(name) {
                continue;
            }
            let Ok(prepared) = read(&self.slots, name)
                .and_then(|v| serde_json::from_value::<Prepared>(v).map_err(json_error))
            else {
                continue;
            };
            let b = &prepared.execution.binding;
            // The same checks `saved_attempt` makes, without reading the slot again.
            if Self::attempt_slot(&b.project, &b.run_id, &b.attempt_id)
                .ok()
                .as_deref()
                != Some(name)
                || action::validate(&prepared.plan, &prepared.execution).is_err()
                || prepared.plan["native_cloud"]["approval"] != json!(prepared.approval)
                || self.require_live(&b.project, &b.run_id).is_err()
            {
                continue;
            }
            match self.fill_metric_gap(&prepared, now) {
                Ok(Gap::Waiting) => {}
                Ok(Gap::Absent | Gap::Done(_)) => {
                    done.insert(name.to_owned());
                }
                Err(error) => eprintln!(
                    "metric gap fill of {}/{}/{} deferred: {}",
                    b.project, b.run_id, b.attempt_id, error.code
                ),
            }
        }
    }
    /// One thread that fills the gaps of newly published archives, when results are
    /// uploaded. The leader lock is tried on every pass, so only one server fills.
    pub fn start_metric_gap_fill(self: Arc<Self>) -> Option<crate::background::Worker> {
        if !self.has_artifacts() {
            return None;
        }
        crate::background::Worker::spawn("native-metric-gap-fill", move |stop| {
            let mut leader = None;
            let mut done = std::collections::BTreeSet::new();
            while !stop.requested() {
                if leader.is_none() {
                    leader = self
                        .directory
                        .lock(
                            OsStr::new(LEADER),
                            LockOptions {
                                create: true,
                                ..Default::default()
                            },
                        )
                        .ok();
                }
                if leader.is_some() {
                    if let Ok((now, _)) = crate::projects::transition_context() {
                        self.metric_gap_pass(&now, &mut done);
                    }
                }
                for _ in 0..PASS_SECONDS {
                    if !stop.pause() {
                        return;
                    }
                }
            }
        })
        .ok()
    }
}
