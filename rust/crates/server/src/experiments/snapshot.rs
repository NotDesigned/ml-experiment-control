//! Bounded evidence reads. Only fixed names below an exact manifested Attempt
//! are opened. File mtime is provenance, never a daemon heartbeat or freshness.
use super::{Document, Error, Format, Provenance, parse_mapping};
use ml_exp_io::directory::Directory;
use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    ffi::OsStr,
    io::{Read, Seek, SeekFrom},
    time::UNIX_EPOCH,
};

pub(super) const MAX_ATTEMPTS: usize = 256;
const MAX_METADATA_TOTAL: u64 = 32 * 1024 * 1024;
const MAX_TAIL_BYTES: u64 = 1024 * 1024;
const MAX_TAIL_LINES: usize = 10_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EvidenceProblem {
    Io,
    InvalidJson,
    IdentityConflict,
    TooLarge,
    BudgetExceeded,
    Truncated,
    PartialFinalLine,
    InvalidUtf8,
    MalformedRecord,
    ForeignRecord,
    NotLoaded,
    NoExplicitCurrent,
    CurrentAttemptUnavailable,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Diagnostic {
    pub source: String,
    pub code: EvidenceProblem,
    pub count: usize,
}

// No Debug: any record or log can include user environment values.
#[derive(Clone, Serialize)]
#[serde(tag = "availability", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EvidenceDocument {
    Missing,
    Present {
        source: String,
        metadata: Map<String, Value>,
        raw_sha256: String,
        modified_at_unix: Option<f64>,
    },
    Unavailable {
        source: String,
        code: EvidenceProblem,
    },
}
impl EvidenceDocument {
    pub fn metadata(&self) -> Option<&Map<String, Value>> {
        match self {
            Self::Present { metadata, .. } => Some(metadata),
            _ => None,
        }
    }
}
#[derive(Clone, Serialize)]
pub struct LogTail {
    pub source: Option<String>,
    pub lines: Vec<String>,
    pub available_lines: usize,
    pub truncated: bool,
    pub offset_bytes: u64,
    pub sampled_bytes: u64,
    pub modified_at_unix: Option<f64>,
    pub diagnostics: Vec<Diagnostic>,
}
#[derive(Clone, Serialize)]
pub struct MetricTail {
    pub source: Option<String>,
    pub records: Vec<Value>,
    pub source_attempt_id: Option<String>,
    pub truncated: bool,
    pub diagnostics: Vec<Diagnostic>,
}
#[derive(Clone, Serialize)]
pub struct AttemptSnapshot {
    pub attempt_id: String,
    pub manifest: Document,
    pub provenance: Provenance,
    pub status: EvidenceDocument,
    pub backend: EvidenceDocument,
    pub collection: EvidenceDocument,
    pub decision: EvidenceDocument,
    pub submission: EvidenceDocument,
    pub stdout: LogTail,
    pub stderr: LogTail,
    pub metrics: MetricTail,
}
#[derive(Clone, Serialize)]
pub struct Snapshot {
    pub project: String,
    pub run_id: String,
    pub run: Document,
    pub current_status: EvidenceDocument,
    pub current_collection: EvidenceDocument,
    pub current_decision: EvidenceDocument,
    pub current_attempt_id: Option<String>,
    pub current_selection_source: Option<String>,
    pub attempts: Vec<AttemptSnapshot>,
    pub diagnostics: Vec<Diagnostic>,
}

pub(super) struct Budget {
    remaining: u64,
}
impl Budget {
    pub(super) fn new() -> Self {
        Self {
            remaining: MAX_METADATA_TOTAL,
        }
    }
    fn reserve(&mut self, bytes: u64) -> Result<(), EvidenceProblem> {
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .ok_or(EvidenceProblem::BudgetExceeded)?;
        Ok(())
    }
}
pub(super) fn manifest(
    directory: &Directory,
    sources: &[super::ManifestSource],
    budget: &mut Budget,
) -> Result<Document, Error> {
    for source in sources {
        let name = OsStr::new(source.filename());
        if !directory.entry_exists(name)? {
            continue;
        }
        let mut file = directory.open_regular_read(name)?;
        let length = file.metadata()?.len();
        if length > super::domain::MAX_BYTES as u64 {
            return Err(Error::Document(super::DocumentError::TooLarge));
        }
        budget.reserve(length).map_err(|_| Error::SnapshotLimit)?;
        let mut bytes = Vec::new();
        (&mut file).take(length + 1).read_to_end(&mut bytes)?;
        directory.validate_regular_entry(name, &file)?;
        if bytes.len() as u64 != length {
            return Err(Error::BytesChanged);
        }
        return Ok(Document::decode(&bytes, *source)?);
    }
    Err(Error::Missing)
}
fn diagnostic(source: &str, code: EvidenceProblem, count: usize) -> Diagnostic {
    Diagnostic {
        source: source.into(),
        code,
        count,
    }
}
fn modified(metadata: &std::fs::Metadata) -> Option<f64> {
    metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|v| v.as_secs_f64())
}
fn unavailable(source: &str, code: EvidenceProblem) -> EvidenceDocument {
    EvidenceDocument::Unavailable {
        source: source.into(),
        code,
    }
}
fn identity_conflict(
    map: &Map<String, Value>,
    project: &str,
    run: &str,
    attempt: Option<&str>,
) -> bool {
    [
        ("project", Some(project)),
        ("run_id", Some(run)),
        ("attempt_id", attempt),
    ]
    .into_iter()
    .any(|(key, expected)| {
        expected.is_some_and(|expected| {
            map.get(key)
                .is_some_and(|value| !value.is_null() && value.as_str() != Some(expected))
        })
    })
}
pub(super) fn evidence(
    directory: &Directory,
    filename: &str,
    source: &str,
    project: &str,
    run: &str,
    attempt: Option<&str>,
    budget: &mut Budget,
) -> EvidenceDocument {
    match directory.entry_exists(OsStr::new(filename)) {
        Ok(false) => return EvidenceDocument::Missing,
        Err(_) => return unavailable(source, EvidenceProblem::Io),
        Ok(true) => {}
    }
    let result = (|| {
        let mut file = directory
            .open_regular_read(OsStr::new(filename))
            .map_err(|_| EvidenceProblem::Io)?;
        let before = file.metadata().map_err(|_| EvidenceProblem::Io)?;
        if before.len() > super::domain::MAX_BYTES as u64 {
            return Err(EvidenceProblem::TooLarge);
        }
        budget.reserve(before.len())?;
        let mut bytes = Vec::new();
        (&mut file)
            .take(before.len() + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| EvidenceProblem::Io)?;
        directory
            .validate_regular_entry(OsStr::new(filename), &file)
            .map_err(|_| EvidenceProblem::Io)?;
        if bytes.len() as u64 != before.len() {
            return Err(EvidenceProblem::Io);
        }
        let map = parse_mapping(&bytes, Format::Json).map_err(|_| EvidenceProblem::InvalidJson)?;
        if identity_conflict(&map, project, run, attempt) {
            return Err(EvidenceProblem::IdentityConflict);
        }
        Ok(EvidenceDocument::Present {
            source: source.into(),
            metadata: map,
            raw_sha256: format!("sha256:{:x}", Sha256::digest(&bytes)),
            modified_at_unix: modified(&before),
        })
    })();
    result.unwrap_or_else(|code| unavailable(source, code))
}

pub(super) fn current(
    mirrors: &[&EvidenceDocument],
    attempts: &[String],
) -> (Option<String>, Option<String>, Vec<Diagnostic>) {
    for mirror in mirrors {
        match mirror {
            EvidenceDocument::Missing => continue,
            EvidenceDocument::Unavailable { source, code } => {
                return (None, None, vec![diagnostic(source, *code, 1)]);
            }
            EvidenceDocument::Present {
                source, metadata, ..
            } => {
                let Some(value) = metadata.get("attempt_id").filter(|value| !value.is_null())
                else {
                    continue;
                };
                let Some(attempt) = value
                    .as_str()
                    .filter(|id| ml_exp_core::identity::validate_id(id).is_ok())
                else {
                    return (
                        None,
                        None,
                        vec![diagnostic(source, EvidenceProblem::IdentityConflict, 1)],
                    );
                };
                if attempts.iter().filter(|id| id.as_str() == attempt).count() == 1 {
                    return (Some(attempt.into()), Some(source.clone()), Vec::new());
                }
                return (
                    None,
                    None,
                    vec![diagnostic(
                        source,
                        EvidenceProblem::CurrentAttemptUnavailable,
                        1,
                    )],
                );
            }
        }
    }
    (
        None,
        None,
        vec![diagnostic(
            "run_mirrors",
            EvidenceProblem::NoExplicitCurrent,
            1,
        )],
    )
}
struct TailBytes {
    bytes: Vec<u8>,
    offset: u64,
    sampled: u64,
    truncated: bool,
    modified: Option<f64>,
}
fn text_line_end(bytes: &[u8], offset: usize) -> Option<usize> {
    let suffix = &bytes[offset..];
    match suffix[0] {
        b'\r' => Some(offset + if suffix.get(1) == Some(&b'\n') { 2 } else { 1 }),
        b'\n' | 0x0b | 0x0c | 0x1c..=0x1e => Some(offset + 1),
        0xc2 if suffix.starts_with(b"\xc2\x85") => Some(offset + 2),
        0xe2 if suffix.starts_with(b"\xe2\x80\xa8") || suffix.starts_with(b"\xe2\x80\xa9") => {
            Some(offset + 3)
        }
        _ => None,
    }
}
/// Python's text splitlines contract, including CRLF as one separator and no
/// extra empty record after a terminal separator. JSONL keeps its LF framing.
fn text_lines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut offset = 0;
    while offset < text.len() {
        if let Some(end) = text_line_end(text.as_bytes(), offset) {
            lines.push(&text[start..offset]);
            start = end;
            offset = end;
        } else {
            offset += text[offset..]
                .chars()
                .next()
                .expect("nonempty suffix")
                .len_utf8();
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}
fn tail_bytes(
    directory: &Directory,
    filename: &str,
    text: bool,
) -> Result<TailBytes, EvidenceProblem> {
    let mut file = directory
        .open_regular_read(OsStr::new(filename))
        .map_err(|_| EvidenceProblem::Io)?;
    let before = file.metadata().map_err(|_| EvidenceProblem::Io)?;
    let start = before.len().saturating_sub(MAX_TAIL_BYTES);
    // Retain a preceding full separator so an aligned UTF-8 boundary or CRLF
    // does not discard its following complete line.
    let window = start.saturating_sub(if text { 3 } else { 1 });
    file.seek(SeekFrom::Start(window))
        .map_err(|_| EvidenceProblem::Io)?;
    let mut bytes = Vec::new();
    (&mut file)
        .take(before.len() - window)
        .read_to_end(&mut bytes)
        .map_err(|_| EvidenceProblem::Io)?;
    directory
        .validate_regular_entry(OsStr::new(filename), &file)
        .map_err(|_| EvidenceProblem::Io)?;
    if bytes.len() as u64 != before.len() - window {
        return Err(EvidenceProblem::Io);
    }
    let mut offset = window;
    if start > 0 {
        let minimum = (start - window) as usize;
        let drop = (0..bytes.len())
            .filter_map(|index| {
                if text {
                    text_line_end(&bytes, index)
                } else {
                    (bytes[index] == b'\n').then_some(index + 1)
                }
            })
            .find(|end| *end >= minimum)
            .unwrap_or(bytes.len());
        offset += drop as u64;
        bytes.drain(..drop);
    }
    Ok(TailBytes {
        sampled: bytes.len() as u64,
        bytes,
        offset,
        truncated: start > 0,
        modified: modified(&before),
    })
}
fn missing_log(not_loaded: bool) -> LogTail {
    LogTail {
        source: None,
        lines: Vec::new(),
        available_lines: 0,
        truncated: false,
        offset_bytes: 0,
        sampled_bytes: 0,
        modified_at_unix: None,
        diagnostics: if not_loaded {
            vec![diagnostic("attempt", EvidenceProblem::NotLoaded, 1)]
        } else {
            Vec::new()
        },
    }
}
pub(super) fn unloaded(attempt: &str) -> (LogTail, LogTail, MetricTail) {
    (
        missing_log(true),
        missing_log(true),
        MetricTail {
            source: None,
            records: Vec::new(),
            source_attempt_id: Some(attempt.into()),
            truncated: false,
            diagnostics: vec![diagnostic("attempt", EvidenceProblem::NotLoaded, 1)],
        },
    )
}
fn sources(directory: &Directory) -> Result<Vec<(Directory, &'static str)>, Error> {
    let mut result = Vec::new();
    for name in ["uploaded_outputs", "collected_run"] {
        if directory.entry_exists(OsStr::new(name))? {
            result.push((directory.child(OsStr::new(name), false)?, name));
        }
    }
    result.push((directory.clone(), ""));
    Ok(result)
}
fn relative(prefix: &str, root: &str, filename: &str) -> String {
    if root.is_empty() {
        format!("{prefix}/{filename}")
    } else {
        format!("{prefix}/{root}/{filename}")
    }
}
fn log(directory: &Directory, stream: &str, prefix: &str) -> Result<LogTail, Error> {
    let filename = format!("{stream}.log");
    let mut roots = vec![(directory.clone(), "")];
    if directory.entry_exists(OsStr::new("collected_run"))? {
        roots.push((
            directory.child(OsStr::new("collected_run"), false)?,
            "collected_run",
        ));
    }
    for (root, name) in roots {
        if !root.entry_exists(OsStr::new(&filename))? {
            continue;
        }
        let source = relative(prefix, name, &filename);
        return Ok(match tail_bytes(&root, &filename, true) {
            Err(code) => LogTail {
                source: Some(source.clone()),
                diagnostics: vec![diagnostic(&source, code, 1)],
                ..missing_log(false)
            },
            Ok(tail) => {
                let mut diagnostics = Vec::new();
                if tail.truncated {
                    diagnostics.push(diagnostic(&source, EvidenceProblem::Truncated, 1));
                }
                if std::str::from_utf8(&tail.bytes).is_err() {
                    diagnostics.push(diagnostic(&source, EvidenceProblem::InvalidUtf8, 1));
                }
                let text = String::from_utf8_lossy(&tail.bytes);
                let mut lines = text_lines(&text)
                    .into_iter()
                    .map(str::to_owned)
                    .collect::<Vec<_>>();
                let available_lines = lines.len();
                let line_truncated = available_lines > MAX_TAIL_LINES;
                if line_truncated {
                    lines.drain(..available_lines - MAX_TAIL_LINES);
                }
                LogTail {
                    source: Some(source),
                    lines,
                    available_lines,
                    truncated: tail.truncated || line_truncated,
                    offset_bytes: tail.offset,
                    sampled_bytes: tail.sampled,
                    modified_at_unix: tail.modified,
                    diagnostics,
                }
            }
        });
    }
    Ok(missing_log(false))
}
fn metrics(
    directory: &Directory,
    project: &str,
    run: &str,
    attempt: &str,
    prefix: &str,
) -> Result<MetricTail, Error> {
    let roots = sources(directory)?;
    for filenames in [
        &["train_metrics.jsonl", "metrics.jsonl"][..],
        &["observed_train_metrics.jsonl"][..],
    ] {
        for (root, name) in &roots {
            for filename in filenames {
                if !root.entry_exists(OsStr::new(filename))? {
                    continue;
                }
                let source = relative(prefix, name, filename);
                let mut result = MetricTail {
                    source: Some(source.clone()),
                    records: Vec::new(),
                    source_attempt_id: Some(attempt.into()),
                    truncated: false,
                    diagnostics: Vec::new(),
                };
                let tail = match tail_bytes(root, filename, false) {
                    Ok(tail) => tail,
                    Err(code) => {
                        result.diagnostics.push(diagnostic(&source, code, 1));
                        return Ok(result);
                    }
                };
                result.truncated = tail.truncated;
                if tail.truncated {
                    result
                        .diagnostics
                        .push(diagnostic(&source, EvidenceProblem::Truncated, 1));
                }
                let mut lines = tail.bytes.split(|b| *b == b'\n').collect::<Vec<_>>();
                if tail.bytes.ends_with(b"\n") {
                    // split includes a terminal empty slice, not another
                    // record. Do not charge it against the history budget.
                    lines.pop();
                } else if !tail.bytes.is_empty() {
                    lines.pop();
                    result.diagnostics.push(diagnostic(
                        &source,
                        EvidenceProblem::PartialFinalLine,
                        1,
                    ));
                }
                if lines.len() > MAX_TAIL_LINES {
                    let drop = lines.len() - MAX_TAIL_LINES;
                    lines.drain(..drop);
                    result.truncated = true;
                }
                for line in lines
                    .into_iter()
                    .filter(|line| !line.iter().all(u8::is_ascii_whitespace))
                {
                    let map = match parse_mapping(line, Format::Json) {
                        Ok(map) => map,
                        Err(_) => {
                            result.diagnostics.push(diagnostic(
                                &source,
                                EvidenceProblem::MalformedRecord,
                                1,
                            ));
                            continue;
                        }
                    };
                    if identity_conflict(&map, project, run, Some(attempt)) {
                        result.diagnostics.push(diagnostic(
                            &source,
                            EvidenceProblem::ForeignRecord,
                            1,
                        ));
                        continue;
                    }
                    result.records.push(Value::Object(map));
                }
                return Ok(result);
            }
        }
    }
    Ok(MetricTail {
        source: None,
        records: Vec::new(),
        source_attempt_id: Some(attempt.into()),
        truncated: false,
        diagnostics: Vec::new(),
    })
}
pub(super) fn heavy(
    directory: &Directory,
    project: &str,
    run: &str,
    attempt: &str,
) -> Result<(LogTail, LogTail, MetricTail), Error> {
    let prefix = format!("attempts/{attempt}");
    Ok((
        log(directory, "stdout", &prefix)?,
        log(directory, "stderr", &prefix)?,
        metrics(directory, project, run, attempt, &prefix)?,
    ))
}
