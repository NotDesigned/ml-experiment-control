//! One durable zero-GPU job imports an OSS tar. Repeats reconcile its outbox;
//! object bytes never pass through the API host. Only its scoped verifier may
//! publish the destination-part hashes and discovered file inventory.
use super::*;
use crate::backends::cloud::adapter::Profile;
use crate::objects::ObjectIdentity;
use ml_exp_core::{
    cloud::{Job, pai::Allocation},
    datasets::{self, Command, oss::Import},
    multipart::{PART_BYTES, PartRecord},
};
use subtle::ConstantTimeEq;

const SCRIPT: &str = include_str!("oss_verify.py");
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    project: String,
    executor: String,
    source: Import,
    target: helper::Target,
    source_identity: Option<ObjectIdentity>,
    job: Job,
    request_id: String,
    uploads: BTreeMap<u64, String>,
    copied: BTreeMap<u64, ObjectIdentity>,
    status: String,
    error: Option<String>,
    provider: Option<Value>,
}
fn name(project: &str, executor: &str, asset: &str) -> cloud::Result<String> {
    Ok(format!(
        "oss-import-{}",
        Service::data_name(project, executor, asset)?
    ))
}
fn conflict() -> cloud::Error {
    cloud::Error {
        code: "OSS_IMPORT_IDENTITY_CONFLICT".into(),
        http_status: Some(409),
        uncertain: false,
    }
}
impl Service {
    pub fn has_oss_import(&self) -> bool {
        self.has_datasets()
            && self
                .backends
                .values()
                .any(|b| b.definition.oss_import_image.is_some() && b.collector.is_some())
    }
    fn oss_saved(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
    ) -> cloud::Result<Option<Saved>> {
        self.project_hash(project)?;
        let file = name(project, executor, asset)?;
        if !self.slots.entry_exists(OsStr::new(&file)).map_err(io)? {
            return Ok(None);
        }
        let saved: Saved = serde_json::from_value(read(&self.slots, &file)?)
            .map_err(why!(cloud::Error::evidence()))?;
        if saved.project != project
            || saved.executor != executor
            || saved.source.asset_id() != asset
            || saved.target
                != self
                    .backends
                    .get(executor)
                    .ok_or_else(cloud::Error::configuration)?
                    .definition
                    .target()?
        {
            return Err(conflict());
        }
        saved
            .source
            .validate()
            .map_err(why!(cloud::Error::evidence()))?;
        Ok(Some(saved))
    }
    fn oss_write(&self, saved: &Saved) -> cloud::Result<()> {
        self.slots
            .replace_regular(
                OsStr::new(&name(
                    &saved.project,
                    &saved.executor,
                    &saved.source.asset_id(),
                )?),
                &encode(saved)?,
            )
            .map_err(io)
    }
    fn oss_view(saved: &Saved) -> cloud::Result<Value> {
        Ok(
            json!({"contract":"oss-import.v2", "project":saved.project,"executor":saved.executor,
            "asset_id":saved.source.asset_id(),"import_source":saved.source,"status":saved.status,
            "copied_parts":saved.copied.len(),"total_parts":ml_exp_core::multipart::part_count(saved.source.archive_bytes).map_err(why!(cloud::Error::evidence()))?,
            "archive_verified":saved.status == "ARCHIVED", "error":saved.error,
            "request_id":saved.request_id,"job_id":saved.provider.as_ref().map(|v| &v["handle"]["job_id"]),
            "resources":{"gpus":0,"cpus":2,"memory_gb":4}}),
        )
    }
    pub(super) fn oss_import_view(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
    ) -> cloud::Result<Option<Value>> {
        self.oss_saved(project, executor, asset)?
            .as_ref()
            .map(Self::oss_view)
            .transpose()
    }
    fn oss_channel(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
    ) -> cloud::Result<datasets::import::DataChannel> {
        self.exchange
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?
            .dataset_channel(&self.workspace, project, executor, asset, "oss-import.v2")
    }
    pub fn import_oss_dataset(
        &self,
        project: &str,
        executor: &str,
        source: Import,
    ) -> cloud::Result<Value> {
        source
            .validate()
            .map_err(why!(cloud::Error::configuration()))?;
        self.project_hash(project)?;
        let backend = self
            .backends
            .get(executor)
            .ok_or_else(cloud::Error::configuration)?;
        let objects = self
            .objects
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        objects.validate_import(&source).map_err(io)?;
        let asset = source.asset_id();
        // A pre-upgrade import remains reusable, with no cloud or object effects.
        if let Some(saved) = self
            .data_saved(project, executor, &asset)?
            .filter(|s| s.archive_verified)
        {
            source
                .matches(&saved.state.definition)
                .map_err(why!(cloud::Error::evidence()))?;
            if saved
                .import_source
                .as_ref()
                .is_some_and(|old| old != &source)
            {
                return Err(conflict());
            }
            let total = saved
                .state
                .definition
                .part_count()
                .map_err(why!(cloud::Error::evidence()))?;
            return Ok(
                json!({"contract":"oss-import.v2","project":project,"executor":executor,"asset_id":asset,
                "import_source":source,"status":"ARCHIVED","copied_parts":total,"total_parts":total,
                "archive_verified":true,"error":null,"reused":true}),
            );
        }
        if !self.enabled {
            return Err(cloud::Error::configuration());
        }
        let collector = backend
            .collector
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let Profile::Pai(profile) = &collector.profile;
        if !objects.import_region(&profile.region) {
            return Err(cloud::Error::configuration());
        }
        let effect = self.slots.lock(
            OsStr::new(&format!("{}.lock", name(project, executor, &asset)?)),
            LockOptions {
                create: true,
                ..Default::default()
            },
        );
        let _effect = match effect {
            Ok(lock) => lock,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                let existing = self
                    .oss_saved(project, executor, &asset)?
                    .ok_or_else(missing)?;
                if existing.source != source {
                    return Err(conflict());
                }
                return Self::oss_view(&existing);
            }
            Err(error) => return Err((why_lock!(cloud::Error::evidence()))(error)),
        };
        let saved = match self.oss_saved(project, executor, &asset)? {
            Some(saved) => {
                if saved.source != source {
                    return Err(conflict());
                }
                saved
            }
            None => {
                let image = backend
                    .definition
                    .oss_import_image
                    .as_ref()
                    .ok_or_else(cloud::Error::configuration)?;
                let target = backend.definition.target()?;
                let id = canonical::digest(&json!([
                    "oss-import.v2",
                    self.workspace,
                    project,
                    executor,
                    source,
                    target
                ]))
                .map_err(why!(cloud::Error::evidence()))?;
                let request_id = id[..32].to_owned();
                let channel = self.oss_channel(project, executor, &asset)?;
                let config = json!({"url":channel.url,"token":channel.token,"request_id":request_id,"sha256":source.sha256,"archive_bytes":source.archive_bytes,
                    "part_bytes":PART_BYTES,"part_count":ml_exp_core::multipart::part_count(source.archive_bytes).map_err(why!(cloud::Error::configuration()))?});
                let job = Job {
                    project: project.into(),
                    run_id: format!("oss-import-{request_id}"),
                    attempt_id: "attempt-001".into(),
                    submission_token: request_id.clone(),
                    source_id: format!("source.{}", source.sha256),
                    image: image.clone(),
                    command: vec![
                        "python3".into(),
                        "-u".into(),
                        "-c".into(),
                        SCRIPT.into(),
                        serde_json::to_string(&config)
                            .map_err(why!(cloud::Error::configuration()))?,
                    ],
                    gpus: 0,
                    cpus: 2,
                    memory_gb: 4,
                    max_time_seconds: 3600,
                    priority: 5,
                    read_only_storage: true,
                };
                job.validate()
                    .map_err(why!(cloud::Error::configuration()))?;
                let saved = Saved {
                    project: project.into(),
                    executor: executor.into(),
                    source,
                    target,
                    source_identity: None,
                    job,
                    request_id,
                    uploads: BTreeMap::new(),
                    copied: BTreeMap::new(),
                    status: "SUBMITTING".into(),
                    error: None,
                    provider: None,
                };
                self.oss_write(&saved)?;
                saved
            }
        };
        if matches!(saved.status.as_str(), "ARCHIVED" | "FAILED") {
            return Self::oss_view(&saved);
        }
        let check = || self.project_hash(project).map(|_| ());
        // Write-ahead outbox owns uncertain submission responses. Repeated POSTs
        // reconcile the exact frozen job, including after a process restart.
        let outcome = self.submit_once_or_reconcile(
            collector,
            &saved.job,
            Allocation::Guaranteed,
            &check,
            &AtomicBool::new(false),
        );
        let _lock = self.lock("oss_import.submit")?;
        let mut current = self
            .oss_saved(project, executor, &asset)?
            .ok_or_else(missing)?;
        match outcome {
            Ok(status) => {
                if !matches!(current.status.as_str(), "ARCHIVED" | "FAILED") {
                    if let Ok(Some(observation)) = super::sync_jobs::proven_terminal(&status) {
                        // A provider exit alone never proves an archive. A successful
                        // worker must have durably acknowledged its completion first.
                        current.status = "FAILED".into();
                        current.error = Some(
                            if observation.state == ml_exp_core::state::RunState::Succeeded {
                                "OSS_IMPORT_REPORT_MISSING"
                            } else {
                                "OSS_IMPORT_JOB_FAILED"
                            }
                            .into(),
                        );
                    } else if current.status == "SUBMITTING" {
                        current.status = "QUEUED".into();
                    }
                }
                current.provider = Some(status);
            }
            Err(error) => {
                current.error = Some(error.code);
            }
        }
        self.oss_write(&current)?;
        Self::oss_view(&current)
    }
    fn oss_auth(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        token: &str,
    ) -> cloud::Result<Saved> {
        let channel = self.oss_channel(project, executor, asset)?;
        if !bool::from(channel.token.as_bytes().ct_eq(token.as_bytes())) {
            return Err(cloud::Error {
                code: "DATASET_AUTH_REQUIRED".into(),
                http_status: Some(401),
                uncertain: false,
            });
        }
        self.oss_saved(project, executor, asset)?
            .ok_or_else(missing)
    }
    pub fn oss_import_exchange(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        token: &str,
        operation: &str,
        body: Value,
    ) -> cloud::Result<Value> {
        let saved = self.oss_auth(project, executor, asset, token)?;
        if operation == "oss-copy" {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Request {
                part: u64,
            }
            let request: Request =
                serde_json::from_value(body).map_err(why!(cloud::Error::configuration()))?;
            return self.oss_copy(saved, request.part);
        }
        if body["request_id"] != saved.request_id {
            return Err(conflict());
        }
        if operation == "oss-failed" {
            let code = body["code"]
                .as_str()
                .filter(|s| {
                    !s.is_empty()
                        && s.len() <= 64
                        && s.bytes()
                            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == b'_')
                })
                .ok_or_else(cloud::Error::configuration)?;
            let _lock = self.lock("oss_import.failed")?;
            let mut saved = self
                .oss_saved(project, executor, asset)?
                .ok_or_else(missing)?;
            if self
                .data_saved(project, executor, asset)?
                .is_some_and(|s| s.archive_verified)
            {
                // The dataset commit precedes the progress commit. A lost final
                // response or a crash between them cannot turn verified bytes
                // into a failed import.
                saved.status = "ARCHIVED".into();
                saved.error = None;
                self.oss_write(&saved)?;
            } else if saved.status != "ARCHIVED" {
                saved.status = "FAILED".into();
                saved.error = Some(code.into());
                self.oss_write(&saved)?;
            }
            return Self::oss_view(&saved);
        }
        if operation != "oss-complete" {
            return Err(cloud::Error::configuration());
        }
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Report {
            request_id: String,
            definition: datasets::Definition,
            parts: BTreeMap<u64, PartRecord>,
        }
        let report: Report =
            serde_json::from_value(body).map_err(why!(cloud::Error::configuration()))?;
        saved
            .source
            .matches(&report.definition)
            .map_err(why!(cloud::Error::evidence()))?;
        report
            .definition
            .manifest_budget()
            .map_err(|_| cloud::Error::configuration())?;
        if report.request_id != saved.request_id
            || saved.status == "FAILED"
            || report.parts.len() as u64
                != report
                    .definition
                    .part_count()
                    .map_err(why!(cloud::Error::evidence()))?
        {
            return Err(conflict());
        }
        for (number, part) in &report.parts {
            if saved
                .copied
                .get(number)
                .is_none_or(|m| m.bytes != part.bytes)
            {
                return Err(conflict());
            }
        }
        self.dataset_transfer(
            project,
            executor,
            Command::Begin {
                definition: report.definition.clone(),
            },
            &[],
        )?;
        let _lock = self.lock("oss_import.complete")?;
        if self
            .oss_saved(project, executor, asset)?
            .ok_or_else(missing)?
            .status
            == "FAILED"
        {
            return Err(conflict());
        }
        let mut dataset = self
            .data_saved(project, executor, asset)?
            .ok_or_else(missing)?;
        for (number, part) in report.parts {
            dataset.state = dataset
                .state
                .record_part(number, part)
                .map_err(why!(cloud::Error::evidence()))?;
        }
        dataset.archive_verified = true;
        dataset.import_source = Some(saved.source);
        self.slots
            .replace_regular(
                OsStr::new(&Self::data_name(project, executor, asset)?),
                &encode(&dataset)?,
            )
            .map_err(io)?;
        let mut saved = self
            .oss_saved(project, executor, asset)?
            .ok_or_else(missing)?;
        saved.status = "ARCHIVED".into();
        saved.error = None;
        self.oss_write(&saved)?;
        Ok(json!({"asset_id":asset,"archive_verified":true,"status":"ARCHIVED"}))
    }
    fn oss_copy(&self, mut saved: Saved, part: u64) -> cloud::Result<Value> {
        let asset = saved.source.asset_id();
        let _part = self
            .slots
            .lock(
                OsStr::new(&format!(
                    "{}.part-{part}.lock",
                    name(&saved.project, &saved.executor, &asset)?
                )),
                LockOptions {
                    create: true,
                    ..Default::default()
                },
            )
            .map_err(why_lock!(cloud::Error::evidence()))?;
        if matches!(saved.status.as_str(), "ARCHIVED" | "FAILED") {
            return Err(conflict());
        }
        let objects = self
            .objects
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let size = ml_exp_core::multipart::part_size(saved.source.archive_bytes, part)
            .map_err(why!(cloud::Error::configuration()))?;
        let source_identity = match &saved.source_identity {
            Some(v) => v.clone(),
            None => objects.import_source_identity(&saved.source).map_err(io)?,
        };
        {
            let _lock = self.lock("oss_import.snapshot")?;
            saved = self
                .oss_saved(&saved.project, &saved.executor, &asset)?
                .ok_or_else(missing)?;
            if matches!(saved.status.as_str(), "FAILED" | "ARCHIVED") {
                return Err(conflict());
            }
            if saved
                .source_identity
                .as_ref()
                .is_some_and(|v| *v != source_identity)
            {
                return Err(conflict());
            }
            saved.source_identity = Some(source_identity.clone());
            saved.status = "COPYING_VERIFYING".into();
            saved.error = None;
            self.oss_write(&saved)?;
        }
        let identity = match objects
            .import_part_identity(&saved.project, &saved.executor, &saved.source, part)
            .map_err(io)?
        {
            Some(identity) => identity,
            None => {
                let upload = match saved.uploads.get(&part) {
                    Some(id) => id.clone(),
                    None => {
                        let id = objects
                            .begin_import_part(&saved.project, &saved.executor, &saved.source, part)
                            .map_err(io)?;
                        let _lock = self.lock("oss_import.upload")?;
                        saved = self
                            .oss_saved(&saved.project, &saved.executor, &asset)?
                            .ok_or_else(missing)?;
                        saved.uploads.insert(part, id.clone());
                        self.oss_write(&saved)?;
                        id
                    }
                };
                objects
                    .finish_import_part(
                        &saved.project,
                        &saved.executor,
                        &saved.source,
                        part,
                        &source_identity.etag,
                        &upload,
                    )
                    .map_err(io)?
            }
        };
        if identity.bytes != size {
            return Err(conflict());
        }
        {
            let _lock = self.lock("oss_import.copied")?;
            saved = self
                .oss_saved(&saved.project, &saved.executor, &asset)?
                .ok_or_else(missing)?;
            if saved.copied.get(&part).is_some_and(|v| *v != identity) {
                return Err(conflict());
            }
            saved.copied.insert(part, identity.clone());
            self.oss_write(&saved)?;
        }
        objects
            .import_read_ticket(
                &saved.project,
                &saved.executor,
                &saved.source,
                part,
                &identity,
            )
            .map_err(io)
    }
}

#[cfg(test)]
mod tests;
