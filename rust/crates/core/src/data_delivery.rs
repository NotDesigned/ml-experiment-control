//! Pure contracts for SenseCore/CCR data delivery.
//!
//! The provider, builder and journal remain imperative shells.  This module
//! owns the identities and decisions which must survive a lost response or a
//! daemon restart: a frozen delivery definition, its NAS scope and receipt,
//! the legal phase transitions, exact CPU-job binding, and the rule that an
//! uncertain delivery is reconciled rather than submitted again.
use crate::{canonical, identity};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

pub const DELIVERY_PREFIX: &str = "delivery.";
pub const RECEIPT_SCHEMA: &str = "nas-data-ready-reuse.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Status {
    Prepared,
    BuildingImage,
    ImageReady,
    Submitting,
    Queued,
    Copying,
    Ready,
    Failed,
    ReconcileRequired,
}

impl Status {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "PREPARED" => Ok(Self::Prepared),
            "BUILDING_IMAGE" => Ok(Self::BuildingImage),
            "IMAGE_READY" => Ok(Self::ImageReady),
            "SUBMITTING" => Ok(Self::Submitting),
            "QUEUED" => Ok(Self::Queued),
            "COPYING" => Ok(Self::Copying),
            "READY" => Ok(Self::Ready),
            "FAILED" => Ok(Self::Failed),
            "RECONCILE_REQUIRED" => Ok(Self::ReconcileRequired),
            _ => Err(Error::InvalidStatus),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prepared => "PREPARED",
            Self::BuildingImage => "BUILDING_IMAGE",
            Self::ImageReady => "IMAGE_READY",
            Self::Submitting => "SUBMITTING",
            Self::Queued => "QUEUED",
            Self::Copying => "COPYING",
            Self::Ready => "READY",
            Self::Failed => "FAILED",
            Self::ReconcileRequired => "RECONCILE_REQUIRED",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Ready | Self::Failed)
    }
}

/// A delivery is deliberately not a scheduler state.  This reducer permits
/// the provider's fast callback races (`COPYING` or `READY` can arrive while
/// the local journal still says `IMAGE_READY`), but never goes backwards to a
/// phase which could silently issue a second create.
pub fn transition(current: Status, next: Status) -> Result<Status> {
    let allowed = match current {
        Status::Prepared => matches!(next, Status::BuildingImage | Status::ReconcileRequired),
        Status::BuildingImage => matches!(
            next,
            Status::ImageReady
                | Status::Copying
                | Status::Ready
                | Status::Failed
                | Status::ReconcileRequired
        ),
        Status::ImageReady => matches!(
            next,
            Status::Submitting
                | Status::Queued
                | Status::Copying
                | Status::Ready
                | Status::Failed
                | Status::ReconcileRequired
        ),
        Status::Submitting => matches!(
            next,
            Status::Queued
                | Status::Copying
                | Status::Ready
                | Status::Failed
                | Status::ReconcileRequired
        ),
        Status::Queued => matches!(
            next,
            Status::Queued
                | Status::Copying
                | Status::Ready
                | Status::Failed
                | Status::ReconcileRequired
        ),
        Status::Copying => matches!(
            next,
            Status::Copying
                | Status::Ready
                | Status::Failed
                | Status::ReconcileRequired
        ),
        Status::ReconcileRequired => {
            matches!(
                next,
                Status::BuildingImage | Status::Copying | Status::Ready | Status::Failed
            )
        }
        Status::Ready => next == Status::Ready,
        Status::Failed => next == Status::Failed,
    };
    if allowed {
        Ok(next)
    } else {
        Err(Error::InvalidTransition {
            from: current,
            to: next,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("data delivery value must be a JSON object")]
    Object,
    #[error("data delivery field is missing or invalid: {0}")]
    Field(&'static str),
    #[error("invalid data delivery identity")]
    Identity,
    #[error("invalid data delivery image")]
    Image,
    #[error("invalid data delivery status")]
    InvalidStatus,
    #[error("illegal data delivery transition {from:?} -> {to:?}")]
    InvalidTransition { from: Status, to: Status },
    #[error("data delivery receipt identity differs")]
    ReceiptIdentity,
    #[error("sealed READY delivery candidate is malformed")]
    ReadyCandidate,
    #[error("data delivery scheduler identity differs")]
    JobIdentity,
    #[error("ambiguous data delivery scheduler result")]
    AmbiguousJob,
    #[error("reconciliation cannot replay a data-copy submission")]
    NoReplay,
    #[error("data delivery requires inspection before another operation")]
    Uncertain,
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NasScope {
    pub workspace: String,
    pub storage_mount: String,
    pub data_root: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptIdentity {
    pub asset_id: String,
    pub archive_sha256: String,
    pub files_sha256: String,
    pub image: String,
    pub data_path: String,
}

impl ReceiptIdentity {
    pub fn value(&self) -> Value {
        json!({
            "asset_id": self.asset_id,
            "archive_sha256": self.archive_sha256,
            "files_sha256": self.files_sha256,
            "image": self.image,
            "data_path": self.data_path,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobIdentity {
    pub scheduler_name: String,
    pub uid: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BeginDecision {
    Start { value: Value, reconcile: bool },
    AlreadyReady { value: Value },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubmissionDecision {
    Submit,
    UseExisting(JobIdentity),
    NoReplay,
}

fn object(value: &Value) -> Result<&Map<String, Value>> {
    value.as_object().ok_or(Error::Object)
}

fn field<'a>(value: &'a Value, name: &'static str) -> Result<&'a Value> {
    object(value)?.get(name).ok_or(Error::Field(name))
}

fn string<'a>(value: &'a Value, name: &'static str, max: usize) -> Result<&'a str> {
    let value = field(value, name)?
        .as_str()
        .filter(|value| !value.is_empty() && value.len() <= max && !value.chars().any(char::is_control))
        .ok_or(Error::Field(name))?;
    Ok(value)
}

fn nested_string<'a>(value: &'a Value, parent: &'static str, name: &'static str, max: usize) -> Result<&'a str> {
    let object = field(value, parent)?.as_object().ok_or(Error::Field(parent))?;
    let value = object
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty() && value.len() <= max && !value.chars().any(char::is_control))
        .ok_or(Error::Field(name))?;
    Ok(value)
}

fn sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn asset_id(value: &str) -> bool {
    value.strip_prefix("asset.").is_some_and(sha256)
}

fn image(value: &str) -> bool {
    let Some((repository, digest)) = value.rsplit_once("@sha256:") else {
        return false;
    };
    !repository.is_empty()
        && repository.len() <= 1024
        && repository
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:/-".contains(&byte))
        && sha256(digest)
}

fn delivery_id_valid(value: &str) -> bool {
    value.strip_prefix(DELIVERY_PREFIX).is_some_and(sha256)
}

fn data_path(scope: &NasScope, asset_id: &str) -> String {
    format!("{}/{}", scope.data_root.trim_end_matches('/'), asset_id)
}

/// Read the three storage fields whose equality is required before a READY
/// receipt can be reused.  The full copy profile remains in the Python shell;
/// scheduler quota and CPU settings must not affect NAS identity.
pub fn nas_scope(value: &Value) -> Result<NasScope> {
    let workspace = nested_string(value, "copy_profile", "workspace", 4096)?.to_owned();
    let storage_mount = nested_string(value, "copy_profile", "storage_mount", 4096)?.to_owned();
    let data_root = nested_string(value, "copy_profile", "data_root", 4096)?.to_owned();
    if !data_root.starts_with('/') || data_root.split('/').any(|part| part == "..") {
        return Err(Error::Field("data_root"));
    }
    Ok(NasScope {
        workspace,
        storage_mount,
        data_root,
    })
}

/// Validate the immutable fields used by `prepare` and return their digest.
/// Unknown fields are retained in the digest for forward compatibility, just
/// as the Python definition digest is computed over the entire definition.
pub fn validate_definition(value: &Value) -> Result<()> {
    let project = string(value, "project", 128)?;
    identity::validate_id(project).map_err(|_| Error::Identity)?;
    let asset_id = string(value, "asset_id", 72)?;
    if !asset_id(asset_id) {
        return Err(Error::Identity);
    }
    for key in ["archive_sha256", "files_sha256", "worker_sha256"] {
        if !sha256(string(value, key, 64)?) {
            return Err(Error::Field(key));
        }
    }
    string(value, "control_revision", 128)?;
    nas_scope(value)?;
    if let Some(image_value) = object(value)?.get("image").and_then(Value::as_str) {
        if !image(image_value) {
            return Err(Error::Image);
        }
    }
    Ok(())
}

/// Python-compatible identity of the frozen definition (`delivery.<sha256>`).
pub fn delivery_id(definition: &Value) -> Result<String> {
    validate_definition(definition)?;
    let digest = canonical::digest(definition).map_err(|_| Error::Identity)?;
    Ok(format!("{DELIVERY_PREFIX}{digest}"))
}

pub fn confirmation(definition: &Value) -> Result<String> {
    validate_definition(definition)?;
    let digest = canonical::digest(definition).map_err(|_| Error::Identity)?;
    Ok(format!("prepare-data:{digest}"))
}

/// The immutable receipt identity used by callbacks and the sealed READY
/// candidate check.  A missing image is not a valid receipt.
pub fn receipt_identity(value: &Value) -> Result<ReceiptIdentity> {
    validate_definition(value)?;
    let image_value = string(value, "image", 1024)?;
    if !image(image_value) {
        return Err(Error::Image);
    }
    let scope = nas_scope(value)?;
    let asset_id = string(value, "asset_id", 72)?.to_owned();
    Ok(ReceiptIdentity {
        asset_id,
        archive_sha256: string(value, "archive_sha256", 64)?.to_owned(),
        files_sha256: string(value, "files_sha256", 64)?.to_owned(),
        image: image_value.to_owned(),
        data_path: data_path(&scope, string(value, "asset_id", 72)?),
    })
}

pub fn receipt_value(value: &Value) -> Result<Value> {
    Ok(receipt_identity(value)?.value())
}

fn status(value: &Value) -> Result<Status> {
    Status::parse(string(value, "status", 64)?)
}

fn same_basic_identity(definition: &Value, candidate: &Value) -> bool {
    ["project", "asset_id", "archive_sha256", "files_sha256"]
        .iter()
        .all(|key| definition.get(*key) == candidate.get(*key))
}

/// Check whether `candidate` is a sealed, exact-byte READY receipt on the
/// same NAS. `Ok(false)` means it is simply another delivery; malformed data
/// that claims to be the matching READY delivery is an error and must not be
/// silently reused.
pub fn ready_candidate_matches(definition: &Value, candidate: &Value) -> Result<bool> {
    validate_definition(definition)?;
    let candidate_status = match candidate.get("status").and_then(Value::as_str) {
        Some(value) => match Status::parse(value) {
            Ok(status) => status,
            Err(_) => return Ok(false),
        },
        None => return Ok(false),
    };
    if candidate_status != Status::Ready {
        return Ok(false);
    }
    if !same_basic_identity(definition, candidate) {
        return Ok(false);
    }
    if nas_scope(definition)? != nas_scope(candidate)? {
        return Ok(false);
    }
    let delivery_id = string(candidate, "delivery_id", 80)?;
    if !delivery_id_valid(delivery_id) {
        return Err(Error::ReadyCandidate);
    }
    let expected = receipt_identity(candidate).map_err(|_| Error::ReadyCandidate)?;
    let receipt = field(candidate, "receipt")?.as_object().ok_or(Error::ReadyCandidate)?;
    if receipt.get("status").and_then(Value::as_str) != Some(Status::Ready.as_str()) {
        return Err(Error::ReadyCandidate);
    }
    for (key, expected) in [
        ("asset_id", &expected.asset_id),
        ("archive_sha256", &expected.archive_sha256),
        ("files_sha256", &expected.files_sha256),
        ("image", &expected.image),
        ("data_path", &expected.data_path),
    ] {
        if receipt.get(key).and_then(Value::as_str) != Some(expected.as_str()) {
            return Err(Error::ReadyCandidate);
        }
    }
    Ok(true)
}

/// Compatibility spelling for callers that mirror the Python service method.
pub fn ready_candidate(definition: &Value, candidate: &Value) -> Result<bool> {
    ready_candidate_matches(definition, candidate)
}

/// Reduce the first execute/reconcile request.  Reconciliation may rebuild or
/// re-observe the original publication, but this result never authorizes a new
/// scheduler create; `submission_plan` enforces that second boundary.
pub fn begin(value: &Value, reconcile: bool) -> Result<BeginDecision> {
    let current = status(value)?;
    if current == Status::Ready {
        return Ok(BeginDecision::AlreadyReady { value: value.clone() });
    }
    let expected = if reconcile {
        Status::ReconcileRequired
    } else {
        Status::Prepared
    };
    if current != expected {
        return Err(if current == Status::ReconcileRequired || current != Status::Prepared {
            Error::Uncertain
        } else {
            Error::InvalidTransition {
                from: current,
                to: Status::BuildingImage,
            }
        });
    }
    transition(current, Status::BuildingImage)?;
    let mut next = value.clone();
    next["status"] = Value::String(Status::BuildingImage.as_str().into());
    next["error"] = Value::Null;
    Ok(BeginDecision::Start { value: next, reconcile })
}

fn scheduler_name(value: &Value) -> Result<String> {
    string(value, "scheduler_name", 256).map(str::to_owned)
}

fn uid(value: &Value) -> Result<String> {
    string(value, "uid", 256).map(str::to_owned)
}

fn observed_job(value: &Value, expected_name: &str, expected_uid: Option<&str>) -> Result<JobIdentity> {
    let name = string(value, "name", 256)?;
    let uid = uid(value)?;
    if name != expected_name || expected_uid.is_some_and(|expected| expected != uid) {
        return Err(Error::JobIdentity);
    }
    Ok(JobIdentity {
        scheduler_name: name.to_owned(),
        uid,
    })
}

/// `found` is an exact provider lookup result. An empty result is represented
/// by `None`; the shell must turn multiple rows into `Error::AmbiguousJob`.
pub fn submission_plan(
    value: &Value,
    found: Option<&Value>,
    reconcile: bool,
) -> Result<SubmissionDecision> {
    let name = scheduler_name(value)?;
    let expected_uid = object(value)?.get("job_uid").and_then(Value::as_str);
    if let Some(found) = found {
        return Ok(SubmissionDecision::UseExisting(observed_job(
            found,
            &name,
            expected_uid,
        )?));
    }
    if reconcile {
        return Ok(SubmissionDecision::NoReplay);
    }
    Ok(SubmissionDecision::Submit)
}

/// Bind one exact scheduler UID. A replacement with the same name is refused;
/// a callback may race the create response, so binding an already equal UID is
/// idempotent.
pub fn bind_job(value: &Value, observed: &Value) -> Result<Value> {
    let name = scheduler_name(value)?;
    let expected_uid = object(value)?.get("job_uid").and_then(Value::as_str);
    let job = observed_job(observed, &name, expected_uid)?;
    let mut next = value.clone();
    if let Some(existing) = expected_uid {
        if existing != job.uid {
            return Err(Error::JobIdentity);
        }
    } else {
        next["job_uid"] = Value::String(job.uid);
    }
    Ok(next)
}

fn receipt_status(receipt: &Value) -> Result<Status> {
    let status = Status::parse(
        receipt
            .as_object()
            .and_then(|object| object.get("status"))
            .and_then(Value::as_str)
            .ok_or(Error::Field("status"))?,
    )?;
    if !matches!(status, Status::Copying | Status::Ready | Status::Failed) {
        return Err(Error::InvalidStatus);
    }
    Ok(status)
}

fn receipt_matches(value: &Value, receipt: &Value) -> Result<()> {
    let expected = receipt_identity(value)?.value();
    let receipt = object(receipt)?;
    for key in ["asset_id", "archive_sha256", "files_sha256", "image", "data_path"] {
        if receipt.get(key) != expected.get(key) {
            return Err(Error::ReceiptIdentity);
        }
    }
    Ok(())
}

/// Validate a callback receipt without applying it to a delivery snapshot.
pub fn validate_receipt(value: &Value, receipt: &Value) -> Result<()> {
    let status = receipt_status(receipt)?;
    if object(value)?.get("image").and_then(Value::as_str).is_none() {
        return Err(Error::Field("image"));
    }
    if status != Status::Failed {
        receipt_matches(value, receipt)?;
    }
    Ok(())
}

/// Apply a worker callback without clocks or journal effects. Failed callbacks
/// carry diagnostics instead of a data receipt, while COPYING/READY must match
/// the exact image, archive, file inventory and NAS path.
pub fn apply_callback(value: &Value, receipt: &Value) -> Result<Value> {
    let current = status(value)?;
    let next_status = receipt_status(receipt)?;
    if object(value)?.get("image").and_then(Value::as_str).is_none() {
        return Err(Error::Field("image"));
    }
    transition(current, next_status)?;
    if next_status != Status::Failed {
        receipt_matches(value, receipt)?;
    }
    if current == Status::Ready {
        if next_status != Status::Ready || object(value)?.get("receipt") != Some(receipt) {
            return Err(Error::ReceiptIdentity);
        }
        return Ok(value.clone());
    }
    let mut next = value.clone();
    next["status"] = Value::String(next_status.as_str().into());
    next["receipt"] = receipt.clone();
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE: &str = "registry.example/data@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn definition(status: Option<&str>) -> Value {
        let mut value = json!({
            "project":"demo",
            "asset_id":format!("asset.{}", "a".repeat(64)),
            "archive_sha256":"b".repeat(64),
            "files_sha256":"c".repeat(64),
            "copy_profile":{
                "workspace":"ws",
                "storage_mount":"volume:/data",
                "data_root":"/nas/demo/data-assets",
                "aec2":"debug-pool",
                "quota_type":"reserved"
            },
            "worker_sha256":"d".repeat(64),
            "control_revision":"acp-data-copy.v5",
            "scheduler_name":"ml-expd-data-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        });
        if let Some(status) = status {
            value["status"] = json!(status);
            value["delivery_id"] = json!(format!("delivery.{}", "e".repeat(64)));
            value["image"] = json!(IMAGE);
        }
        value
    }

    fn receipt(value: &Value, status: &str) -> Value {
        let identity = receipt_identity(value).unwrap().value();
        let mut receipt = identity.as_object().unwrap().clone();
        receipt.insert("status".into(), json!(status));
        Value::Object(receipt)
    }

    #[test]
    fn definition_and_receipt_identity_are_stable_and_scope_excludes_cpu_choices() {
        let value = definition(None);
        assert_eq!(delivery_id(&value).unwrap(), format!("delivery.{}", canonical::digest(&value).unwrap()));
        assert_eq!(confirmation(&value).unwrap(), format!("prepare-data:{}", canonical::digest(&value).unwrap()));
        assert_eq!(nas_scope(&value).unwrap().data_root, "/nas/demo/data-assets");
        let with_cpu = {
            let mut changed = value.clone();
            changed["copy_profile"]["cpus"] = json!(2);
            changed["copy_profile"]["memory_gb"] = json!(4);
            changed
        };
        assert_eq!(nas_scope(&value).unwrap(), nas_scope(&with_cpu).unwrap());
        let mut ready = definition(Some("READY"));
        assert_eq!(receipt_identity(&ready).unwrap().data_path, "/nas/demo/data-assets/asset.".to_owned() + &"a".repeat(64));
        let ready_receipt = receipt(&ready, "READY");
        ready["receipt"] = ready_receipt;
        assert!(ready_candidate_matches(&definition(None), &ready).unwrap());
    }

    #[test]
    fn ready_candidate_requires_exact_receipt_and_same_nas() {
        let definition = definition(None);
        let mut ready = definition(Some("READY"));
        let ready_receipt = receipt(&ready, "READY");
        ready["receipt"] = ready_receipt;
        assert!(ready_candidate_matches(&definition, &ready).unwrap());
        ready["receipt"]["files_sha256"] = json!("f".repeat(64));
        assert_eq!(ready_candidate_matches(&definition, &ready), Err(Error::ReadyCandidate));
        let mut other = definition(Some("READY"));
        let other_receipt = receipt(&other, "READY");
        other["receipt"] = other_receipt;
        other["copy_profile"]["data_root"] = json!("/nas/other");
        assert!(!ready_candidate_matches(&definition, &other).unwrap());
    }

    #[test]
    fn legal_transitions_keep_terminal_states_closed() {
        for (from, to) in [
            (Status::Prepared, Status::BuildingImage),
            (Status::BuildingImage, Status::ImageReady),
            (Status::ImageReady, Status::Submitting),
            (Status::Submitting, Status::Queued),
            (Status::Queued, Status::Copying),
            (Status::Copying, Status::Ready),
            (Status::ReconcileRequired, Status::BuildingImage),
        ] {
            assert!(transition(from, to).is_ok(), "{from:?} -> {to:?}");
        }
        assert!(transition(Status::Ready, Status::Submitting).is_err());
        assert!(transition(Status::Failed, Status::BuildingImage).is_err());
    }

    #[test]
    fn begin_reconcile_does_not_replay_and_only_prepared_can_submit() {
        let mut prepared = definition(Some("PREPARED"));
        prepared["image"] = Value::Null;
        let start = begin(&prepared, false).unwrap();
        let BeginDecision::Start { value, reconcile } = start else { panic!() };
        assert!(!reconcile && value["status"] == "BUILDING_IMAGE");
        assert_eq!(begin(&value, false), Err(Error::Uncertain));
        let mut uncertain = definition(Some("RECONCILE_REQUIRED"));
        uncertain["job_uid"] = json!("original");
        let resumed = begin(&uncertain, true).unwrap();
        let BeginDecision::Start { value, reconcile } = resumed else { panic!() };
        assert!(reconcile && value["status"] == "BUILDING_IMAGE");
        assert_eq!(submission_plan(&value, None, true).unwrap(), SubmissionDecision::NoReplay);
    }

    #[test]
    fn exact_job_binding_is_idempotent_and_rejects_replacement() {
        let value = definition(Some("SUBMITTING"));
        let observed = json!({"name":value["scheduler_name"],"uid":"u-1","state":"RUNNING"});
        let bound = bind_job(&value, &observed).unwrap();
        assert_eq!(bound["job_uid"], "u-1");
        assert_eq!(bind_job(&bound, &observed).unwrap(), bound);
        let replacement = json!({"name":value["scheduler_name"],"uid":"u-2"});
        assert_eq!(bind_job(&bound, &replacement), Err(Error::JobIdentity));
        assert_eq!(submission_plan(&value, None, true).unwrap(), SubmissionDecision::NoReplay);
        assert!(matches!(submission_plan(&value, Some(&observed), true), Ok(SubmissionDecision::UseExisting(_))));
    }

    #[test]
    fn callback_requires_identity_and_seals_ready() {
        let mut value = definition(Some("QUEUED"));
        value["image"] = json!(IMAGE);
        let copying = receipt(&value, "COPYING");
        let copied = apply_callback(&value, &copying).unwrap();
        assert_eq!(copied["status"], "COPYING");
        let ready = apply_callback(&copied, &receipt(&copied, "READY")).unwrap();
        assert_eq!(ready["status"], "READY");
        assert_eq!(apply_callback(&ready, &ready["receipt"]), Ok(ready.clone()));
        let mut wrong = receipt(&ready, "READY");
        wrong["data_path"] = json!("/other");
        assert_eq!(apply_callback(&ready, &wrong), Err(Error::ReceiptIdentity));
        assert_eq!(apply_callback(&value, &json!({"status":"FAILED","error":"ENOSPC"})).unwrap()["status"], "FAILED");
    }
}
