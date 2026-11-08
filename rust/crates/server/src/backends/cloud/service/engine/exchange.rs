//! Scope-limited ingress. Authentication never grants ordinary API access.
//! Journal/checkpoint commits precede the stream cursor; replay is idempotent.
use super::*;
use hmac::{Hmac, Mac};
use ml_exp_core::{
    cloud::exchange::{Ack, Channel, Packet},
    managed::{history::Cursor, remote::Evidence},
    records::{ImportCounts, JsonlImporter},
};
use sha2::Sha256;
use subtle::ConstantTimeEq;

pub(super) struct Authority {
    url: String,
    key: Vec<u8>,
}
impl Authority {
    pub(super) fn dataset_channel(
        &self,
        workspace: &str,
        project: &str,
        executor: &str,
        asset: &str,
        access: &str,
    ) -> cloud::Result<ml_exp_core::datasets::import::DataChannel> {
        let mut mac = Hmac::<Sha256>::new_from_slice(&self.key).map_err(why!(
            cloud::Error::configuration(),
            "exchange signing key rejected"
        ))?;
        mac.update(&encode(&json!([
            "native-dataset-exchange.v1",
            workspace,
            project,
            executor,
            asset,
            access
        ]))?);
        Ok(ml_exp_core::datasets::import::DataChannel {
            url: format!(
                "{}/api/dataset-exchange/{project}/{executor}/{asset}",
                self.url
            ),
            token: format!("{:x}", mac.finalize().into_bytes()),
        })
    }
    pub(super) fn load(config: super::super::config::ExchangeSettings) -> cloud::Result<Self> {
        let key = ml_exp_io::read_private(&config.signing_key_file, 4096, true).map_err(io)?;
        if key.len() < 32 || config.public_url.ends_with('/') {
            return Err(cloud::Error::configuration());
        }
        let result = Self {
            url: config.public_url,
            key,
        };
        let dummy = Channel {
            url: format!("{}/api/worker-exchange/check/check/attempt-001", result.url),
            token: "0".repeat(64),
        };
        dummy
            .validate(&ml_exp_core::launch::Identity {
                project: "check".into(),
                run_id: "check".into(),
                attempt_id: "attempt-001".into(),
            })
            .map_err(why!(cloud::Error::configuration()))?;
        Ok(result)
    }
    pub(super) fn channel(&self, workspace: &str, execution: &Execution) -> cloud::Result<Channel> {
        self.channel_for(
            workspace,
            &execution.binding,
            &execution.job.submission_token,
        )
    }
    pub(super) fn channel_for(
        &self,
        workspace: &str,
        binding: &Binding,
        submission_token: &str,
    ) -> cloud::Result<Channel> {
        let mut mac = Hmac::<Sha256>::new_from_slice(&self.key).map_err(why!(
            cloud::Error::configuration(),
            "exchange signing key rejected"
        ))?;
        mac.update(&encode(&json!([
            "native-cloud-exchange.v1",
            workspace,
            binding,
            submission_token
        ]))?);
        let token = format!("{:x}", mac.finalize().into_bytes());
        Ok(Channel {
            url: format!(
                "{}/api/worker-exchange/{}/{}/{}",
                self.url, binding.project, binding.run_id, binding.attempt_id
            ),
            token,
        })
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Saved {
    pub sequence: u64,
    pub packet_sha256: String,
    pub package_sha256: String,
    pub observed_at: String,
    pub evidence: Evidence,
    pub cursor: Option<Cursor>,
    pub importer: Option<JsonlImporter>,
    pub counts: ImportCounts,
    pub complete: bool,
    pub inputs: Vec<ml_exp_core::input_progress::Record>,
    pub checkpoint_error: Option<String>,
}
pub struct CodeGrant {
    project: String,
    source: String,
}
pub struct WorkerGrant {
    worker: String,
}
/// An open file of the store whose hash has been proven, read through that same handle.
pub struct WorkerContent {
    pub bytes: u64,
    file: std::fs::File,
}
impl WorkerContent {
    pub fn each_chunk(
        self,
        mut send: impl FnMut(Vec<u8>) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        use std::io::Read;
        let mut file = self.file.take(self.bytes);
        let mut sent = 0;
        loop {
            let mut chunk = vec![0; 256 * 1024];
            let count = file.read(&mut chunk)?;
            if count == 0 {
                break;
            }
            chunk.truncate(count);
            send(chunk)?;
            sent += count as u64;
        }
        if sent != self.bytes {
            return Err(std::io::Error::other("the worker file became shorter"));
        }
        Ok(())
    }
}
impl Service {
    pub fn synchronize(
        &self,
        action: &str,
        request: &ml_exp_core::cloud::exchange::Sync,
        now: &str,
        cancelled: &AtomicBool,
    ) -> cloud::Result<Value> {
        request
            .validate(action)
            .map_err(why!(cloud::Error::configuration()))?;
        if !self.enabled || self.exchange.is_none() {
            return Err(cloud::Error {
                code: "CLOUD_SYNC_DISABLED".into(),
                http_status: Some(403),
                uncertain: false,
            });
        }
        if let Some(retry) = self.sync_recovery_request(action, &request.request_id)? {
            return self.retry_sync(action, &request.request_id, &retry, now, cancelled);
        }
        let _lock = self.lock("exchange.synchronize")?;
        let prepared = self.load(action)?;
        if self.actions.observe(action).map_err(op)?["execution"]["status"] != "VERIFIED" {
            return Err(cloud::Error::evidence());
        }
        let execution = &prepared.execution;
        let guard = self
            .fence
            .existing_observation_guard(&execution.job.project, &execution.job.run_id)
            .map_err(why!(cloud::Error::evidence()))?;
        let collector = self.backends[&execution.backend_id]
            .collector
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let job = request
            .job(&execution.job)
            .map_err(why!(cloud::Error::configuration()))?;
        let job = self.with_result_limits(&execution.backend_id, job)?;
        // Before the job exists, so a purge sees a sync that may still copy checkpoints.
        self.record_sync_job(&prepared, &job)?;
        let check = || self.still_live(&guard, &execution.job.project, &execution.job.run_id);
        let status = self.submit_once_or_reconcile(
            collector,
            &job,
            ml_exp_core::cloud::pai::Allocation::Guaranteed,
            &check,
            cancelled,
        )?;
        let mut view = self.get(action, now)?;
        view["synchronization"] = json!({"request_id":request.request_id,"resources":{"gpus":job.gpus,"cpus":job.cpus,"memory_gb":job.memory_gb},"max_time_seconds":job.max_time_seconds,"job":status,"training_replayed":false});
        Ok(view)
    }
    pub(super) fn restore_reference(
        &self,
        project: &str,
        executor: &str,
        reference: &ml_exp_core::checkpoints::Reference,
    ) -> cloud::Result<Value> {
        reference
            .validate()
            .map_err(why!(cloud::Error::evidence()))?;
        let prepared = self
            .saved_attempt(project, &reference.run_id, &reference.attempt_id)?
            .ok_or_else(missing)?;
        if prepared.execution.backend_id != executor
            || prepared.execution.job.attempt_id != reference.attempt_id
        {
            return Err(cloud::Error::evidence());
        }
        let b = &prepared.execution.binding;
        let root = self
            .authority(project, &reference.run_id)?
            .child(OsStr::new(&reference.run_id), false)
            .map_err(io)?;
        if helper::sha256_bytes(&encode(&read(&root, "manifest.json")?)?) != b.run_sha256 {
            return Err(cloud::Error::evidence());
        }
        let context = self.checkpoint_context(&prepared)?;
        crate::checkpoints::Store::open_existing(&self.settings.registry_root)
            .map_err(why!(cloud::Error::evidence()))?
            .read(&context, &reference.checkpoint_id)
            .map_err(why!(cloud::Error::evidence()))?
            .ok_or_else(missing)
    }
    /// The frozen package of exactly the Attempt this token was issued for.
    pub(super) fn worker_authorized(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        token: &str,
    ) -> cloud::Result<ml_exp_core::cloud::launch::Package> {
        let denied = || cloud::Error {
            code: "WORKER_AUTH_REQUIRED".into(),
            http_status: Some(401),
            uncertain: false,
        };
        let prepared = self
            .saved_attempt(project, run, attempt)?
            .ok_or_else(denied)?;
        let expected = self
            .exchange
            .as_ref()
            .ok_or_else(denied)?
            .channel(&self.workspace, &prepared.execution)?;
        if !bool::from(expected.token.as_bytes().ct_eq(token.as_bytes())) {
            return Err(denied());
        }
        self.require_live(project, run)?;
        prepared
            .execution
            .package()
            .map_err(why!(cloud::Error::evidence()))
    }
    pub fn worker_package(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        token: &str,
    ) -> cloud::Result<Value> {
        let package = self.worker_authorized(project, run, attempt, token)?;
        serde_json::to_value(package).map_err(why!(cloud::Error::evidence()))
    }
    /// Only the source frozen in this Attempt's package. A request names no source,
    /// so holding a token cannot reach any other code in the Project.
    fn worker_code_source(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        token: &str,
    ) -> cloud::Result<String> {
        self.worker_authorized(project, run, attempt, token)?
            .code_source_id
            .ok_or_else(missing)
    }
    pub fn worker_code(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        token: &str,
    ) -> cloud::Result<Value> {
        let source = self.worker_code_source(project, run, attempt, token)?;
        let manifest = self
            .sources
            .code_manifest(project, &source)
            .map_err(why!(cloud::Error::evidence()))?;
        serde_json::to_value(manifest).map_err(why!(cloud::Error::evidence()))
    }
    /// Proof that this token may read this Attempt's code, before a transfer is begun.
    pub fn worker_code_grant(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        token: &str,
    ) -> cloud::Result<CodeGrant> {
        Ok(CodeGrant {
            project: project.to_owned(),
            source: self.worker_code_source(project, run, attempt, token)?,
        })
    }
    pub fn worker_code_content(&self, grant: &CodeGrant) -> cloud::Result<sources::CodeContent> {
        self.sources
            .code_content(&grant.project, &grant.source)
            .map_err(why!(cloud::Error::evidence()))
    }
    /// Proof that this token may read this Attempt's worker, before a transfer is begun.
    /// A request names no worker, so a token reaches only the one its Attempt froze.
    pub fn worker_binary_grant(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        token: &str,
    ) -> cloud::Result<WorkerGrant> {
        Ok(WorkerGrant {
            worker: self
                .worker_authorized(project, run, attempt, token)?
                .worker_sha256,
        })
    }
    /// From the configured store only, with a name that is a validated hash, and refused
    /// unless the file really has that hash.
    pub fn worker_binary(&self, grant: &WorkerGrant) -> cloud::Result<WorkerContent> {
        self.stored_worker(&grant.worker)
    }
    pub(super) fn stored_worker(&self, worker: &str) -> cloud::Result<WorkerContent> {
        use sha2::Digest;
        use std::io::{Read, Seek};
        let unavailable = super::builds::worker_unavailable;
        let store = self
            .builds
            .as_ref()
            .and_then(|settings| settings.worker_store.as_ref())
            .ok_or_else(unavailable)?;
        if !helper::valid_digest(worker) {
            return Err(unavailable());
        }
        let name = worker.strip_prefix("sha256:").ok_or_else(unavailable)?;
        let mut file = Directory::open(store, false)
            .and_then(|directory| directory.open_regular_read(OsStr::new(name)))
            .map_err(why!(unavailable()))?;
        let bytes = file.metadata().map_err(why!(unavailable()))?.len();
        if bytes > ml_exp_core::cloud::launch::MAX_WORKER_BYTES {
            return Err(unavailable());
        }
        let mut hasher = Sha256::new();
        let mut buffer = vec![0; 256 * 1024];
        loop {
            let count = file.read(&mut buffer).map_err(why!(unavailable()))?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        if format!("{:x}", hasher.finalize()) != name {
            crate::why::record(file!(), line!(), &"a stored worker differs from its name");
            return Err(unavailable());
        }
        file.rewind().map_err(why!(unavailable()))?;
        Ok(WorkerContent { bytes, file })
    }
    pub fn has_exchange(&self) -> bool {
        self.exchange.is_some()
    }
    fn exchange_name(prepared: &Prepared) -> cloud::Result<String> {
        Ok(format!(
            "{}.exchange.json",
            Self::attempt_slot(
                &prepared.execution.job.project,
                &prepared.execution.job.run_id,
                &prepared.execution.job.attempt_id
            )?
        ))
    }
    pub(super) fn exchange_saved(&self, prepared: &Prepared) -> cloud::Result<Option<Saved>> {
        let name = Self::exchange_name(prepared)?;
        if !self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            return Ok(None);
        }
        let bytes = self.slots.read_whole(OsStr::new(&name)).map_err(io)?;
        let saved: Saved = serde_json::from_value(Value::Object(
            ml_exp_core::json::own_mapping(&bytes).map_err(why!(cloud::Error::evidence()))?,
        ))
        .map_err(why!(cloud::Error::evidence()))?;
        if saved.sequence == 0
            || saved.package_sha256
                != helper::sha256_bytes(&encode(
                    &prepared
                        .execution
                        .package()
                        .map_err(why!(cloud::Error::evidence()))?,
                )?)
            || !helper::valid_digest(&saved.packet_sha256)
            || saved.complete != saved.importer.is_none()
        {
            return Err(cloud::Error::evidence());
        }
        Ok(Some(saved))
    }
    pub(super) fn checkpoint_context(
        &self,
        prepared: &Prepared,
    ) -> cloud::Result<ml_exp_core::checkpoints::Context> {
        let execution = &prepared.execution;
        let package = execution
            .package()
            .map_err(why!(cloud::Error::evidence()))?;
        package
            .checkpoint_context
            .ok_or_else(cloud::Error::evidence)
    }
    pub(super) fn exchange_records(
        &self,
        prepared: &Prepared,
    ) -> cloud::Result<crate::records::Store> {
        let store = crate::records::Store::open(&self.settings.registry_root, &self.workspace)
            .map_err(why!(cloud::Error::evidence()))?;
        let b = &prepared.execution.binding;
        let root = self
            .authority(&b.project, &b.run_id)?
            .child(OsStr::new(&b.run_id), false)
            .map_err(io)?;
        let manifest = read(&root, "manifest.json")?;
        let binding = ml_exp_core::records::native_binding(&manifest, &b.run_sha256)
            .map_err(why!(cloud::Error::evidence()))?;
        let guard = store
            .guard(&b.project, &b.run_id)
            .map_err(why!(cloud::Error::evidence()))?;
        // bind checks equality of an existing route/config; ingress cannot reroute it.
        store
            .bind(&guard, &binding["route"], &binding["config"])
            .map_err(why!(cloud::Error::evidence()))?;
        drop(guard);
        Ok(store)
    }
    pub fn receive(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        token: &str,
        packet: &Packet,
        now: &str,
    ) -> cloud::Result<Ack> {
        let denied = || cloud::Error {
            code: "WORKER_AUTH_REQUIRED".into(),
            http_status: Some(401),
            uncertain: false,
        };
        // Authentication precedes journal creation, packet processing and any write.
        let _lock = self.lock("exchange.receive")?;
        let prepared = self
            .saved_attempt(project, run, attempt)?
            .ok_or_else(denied)?;
        if attempt != prepared.execution.job.attempt_id {
            return Err(denied());
        }
        let expected = self
            .exchange
            .as_ref()
            .ok_or_else(denied)?
            .channel(&self.workspace, &prepared.execution)?;
        if !bool::from(expected.token.as_bytes().ct_eq(token.as_bytes())) {
            return Err(denied());
        }
        self.require_live(project, run)?;
        if self
            .actions
            .observe(
                prepared.plan["action_id"]
                    .as_str()
                    .ok_or_else(cloud::Error::evidence)?,
            )
            .map_err(op)?["execution"]["status"]
            != "VERIFIED"
        {
            return Err(cloud::Error::evidence());
        }
        let guard = self
            .fence
            .existing_observation_guard(project, run)
            .map_err(why!(cloud::Error::evidence()))?;
        let package = prepared
            .execution
            .package()
            .map_err(why!(cloud::Error::evidence()))?;
        let package_sha256 = helper::sha256_bytes(&encode(&package)?);
        packet
            .validate(&package.config, &package_sha256)
            .map_err(why!(cloud::Error::evidence()))?;
        // The sealed launch hash labels worker evidence, while the provider
        // handle is independently retained in the scheduler outbox.
        if packet.evidence.job_id
            != package
                .config
                .binding()
                .map_err(why!(cloud::Error::evidence()))?
                .sha256
        {
            return Err(cloud::Error::evidence());
        }
        let previous = self.exchange_saved(&prepared)?;
        let ack = Ack::for_packet(packet).map_err(why!(cloud::Error::evidence()))?;
        if let Some(saved) = &previous {
            if saved.sequence == packet.sequence && saved.packet_sha256 == ack.packet_sha256 {
                return Ok(ack);
            }
            packet
                .evidence
                .follows(&saved.evidence)
                .map_err(why!(cloud::Error::evidence()))?;
        }
        packet
            .follows(
                previous.as_ref().map_or(0, |s| s.sequence),
                previous.as_ref().map(|s| s.packet_sha256.as_str()),
                previous.as_ref().and_then(|s| s.cursor.as_ref()),
                previous.as_ref().is_some_and(|s| s.complete),
            )
            .map_err(why!(cloud::Error::evidence()))?;
        if let Some(document) = &packet.checkpoint {
            let context = self.checkpoint_context(&prepared)?;
            let (_, manifest) =
                ml_exp_core::checkpoints::validate_document(document, Some(&context), None)
                    .map_err(why!(cloud::Error::evidence()))?;
            crate::checkpoints::Store::open_existing(&self.settings.registry_root)
                .map_err(why!(cloud::Error::evidence()))?
                .register(
                    &context,
                    &json!({"step":manifest.step,"files":document["files"]}),
                    now,
                )
                .map_err(why!(cloud::Error::evidence()))?;
        }
        let records = self.exchange_records(&prepared)?;
        let record_guard = records
            .guard(project, run)
            .map_err(why!(cloud::Error::evidence()))?;
        let scope = records
            .scope(&record_guard, attempt)
            .map_err(why!(cloud::Error::evidence()))?;
        let config = records
            .binding(&record_guard)
            .map_err(why!(cloud::Error::evidence()))?
            .ok_or_else(cloud::Error::evidence)?["config"]
            .clone();
        let (mut cursor, mut importer, mut counts, mut complete) = match previous {
            Some(saved) => (saved.cursor, saved.importer, saved.counts, saved.complete),
            None => (
                None,
                Some(
                    JsonlImporter::new("metrics.jsonl", config)
                        .map_err(why!(cloud::Error::evidence()))?,
                ),
                ImportCounts::default(),
                false,
            ),
        };
        if let Some(page) = &packet.history {
            let mut value = importer.take().ok_or_else(cloud::Error::evidence)?;
            for chunk in page.bytes.chunks(JsonlImporter::INPUT_BOUND) {
                let (next, emission) = value.feed(chunk).map_err(why!(cloud::Error::evidence()))?;
                records
                    .append_import(&record_guard, &scope, &emission)
                    .map_err(why!(cloud::Error::evidence()))?;
                counts = emission.counts;
                value = next;
            }
            complete = page.complete();
            cursor = Some(page.next.clone());
            importer = if complete {
                let emission = value.finish().map_err(why!(cloud::Error::evidence()))?;
                records
                    .append_import(&record_guard, &scope, &emission)
                    .map_err(why!(cloud::Error::evidence()))?;
                counts = emission.counts;
                None
            } else {
                Some(value)
            };
        }
        let worker = &packet.evidence.worker;
        let phase = if worker.phase == ml_exp_core::managed::evidence::Phase::Finished {
            if complete {
                "METRICS_DRAINED"
            } else {
                "TRAINING_COMPLETED"
            }
        } else if worker.phase == ml_exp_core::managed::evidence::Phase::Prepared {
            if packet
                .inputs
                .iter()
                .any(|p| p.phase == ml_exp_core::input_progress::Phase::Failed)
            {
                "DATA_FAILED"
            } else {
                "PREPARING_DATA"
            }
        } else {
            "TRAINING"
        };
        let status = worker
            .receipt
            .as_ref()
            .map(|r| match r.training {
                ml_exp_core::managed::evidence::Training::Completed => "SUCCEEDED",
                ml_exp_core::managed::evidence::Training::Cancelled => "CANCELLED",
                ml_exp_core::managed::evidence::Training::TimedOut => "TIMEOUT",
                ml_exp_core::managed::evidence::Training::Failed => "FAILED",
                _ => "UNKNOWN",
            })
            .unwrap_or("RUNNING");
        // No mutable receipt time inside an idempotent event. The accepted
        // server receive time lives separately in the observation envelope.
        let event = ml_exp_core::records::Event {
            event_id: format!("worker.cloud.{}", ack.packet_sha256),
            payload: json!({"kind":"lifecycle","data":{"phase":phase,"state":status,"timestamp":null,"exit_code":worker.receipt.as_ref().and_then(|r|r.process.exit_code)}}),
        };
        records
            .append(&record_guard, &scope, &[event])
            .map_err(why!(cloud::Error::evidence()))?;
        let saved = Saved {
            sequence: packet.sequence,
            packet_sha256: ack.packet_sha256.clone(),
            package_sha256,
            observed_at: now.into(),
            evidence: packet.evidence.clone(),
            cursor,
            importer,
            counts,
            complete,
            inputs: packet.inputs.clone(),
            checkpoint_error: packet.checkpoint_error.clone(),
        };
        guard
            .require_observed_live(project, run)
            .map_err(why!(cloud::Error::evidence()))?;
        let bytes = encode(&saved)?;
        self.slots
            .replace_regular(OsStr::new(&Self::exchange_name(&prepared)?), &bytes)
            .map_err(why!(cloud::Error {
                code: "WORKER_RECEIPT_UNCERTAIN".into(),
                http_status: None,
                uncertain: true,
            }))?;
        Ok(ack)
    }
    pub fn checkpoints(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        id: Option<&str>,
    ) -> cloud::Result<Value> {
        let prepared = self
            .saved_attempt(project, run, attempt)?
            .ok_or_else(missing)?;
        self.require_live(project, run)?;
        if prepared.execution.job.attempt_id != attempt {
            return Err(missing());
        }
        let context = self.checkpoint_context(&prepared)?;
        let store = crate::checkpoints::Store::open_existing(&self.settings.registry_root)
            .map_err(why!(cloud::Error::evidence()))?;
        match id {
            Some(id) => store
                .read(&context, id)
                .map_err(why!(cloud::Error::evidence()))?
                .ok_or_else(missing),
            None => {
                let rows = store
                    .list_archived(&context)
                    .map_err(why!(cloud::Error::evidence()))?;
                // Each verified object-store copy, by checkpoint. A checkpoint without
                // one is still only on backend storage.
                let archives = rows
                    .iter()
                    .filter_map(|(document, receipt)| {
                        Some((
                            document["checkpoint_id"].as_str()?.to_owned(),
                            json!(receipt.as_ref()?),
                        ))
                    })
                    .collect::<serde_json::Map<_, _>>();
                Ok(json!({
                    "project": project,
                    "run_id": run,
                    "attempt_id": attempt,
                    "checkpoints": rows.into_iter().map(|(document, _)| document).collect::<Vec<_>>(),
                    "archives": archives,
                    "state_purge": self.purge_summary(&prepared)?,
                }))
            }
        }
    }
    pub fn tracking_status(&self, project: &str, run: &str) -> cloud::Result<Value> {
        let prepared = self.saved(project, run)?.ok_or_else(missing)?;
        let records = self.exchange_records(&prepared)?;
        let guard = records
            .guard(project, run)
            .map_err(why!(cloud::Error::evidence()))?;
        let binding = records
            .binding(&guard)
            .map_err(why!(cloud::Error::evidence()))?
            .ok_or_else(cloud::Error::evidence)?;
        let route = &binding["route"];
        let mut scopes = records
            .run_scopes(&guard)
            .map_err(why!(cloud::Error::evidence()))?;
        for scope in &mut scopes {
            scope["pending"] = json!(
                scope["total"].as_i64().ok_or_else(cloud::Error::evidence)?
                    - scope["confirmed"]
                        .as_i64()
                        .ok_or_else(cloud::Error::evidence)?
            );
            // Post-hoc batches are listed apart from the training journal.
            let id = scope["id"]
                .as_str()
                .ok_or_else(cloud::Error::evidence)?
                .to_owned();
            let batches = records
                .derived_batches(&guard, &id)
                .map_err(why!(cloud::Error::evidence()))?;
            if !batches.is_empty() {
                scope["derived_batches"] = json!(
                    batches
                        .iter()
                        .map(crate::records::DerivedBatch::document)
                        .collect::<Vec<_>>()
                );
            }
            // Each W&B rebuild keeps the id it replaced, which is never deleted.
            let rebuilds = records
                .wandb_rebuilds(&guard, &id)
                .map_err(why!(cloud::Error::evidence()))?;
            if !rebuilds.is_empty() {
                scope["wandb_rebuilds"] = json!(
                    rebuilds
                        .iter()
                        .map(crate::records::WandbRebuild::document)
                        .collect::<Vec<_>>()
                );
            }
            if route["enabled"] == true {
                scope["url"] = json!(format!(
                    "https://wandb.ai/{}/{}/runs/{}",
                    route["entity"]
                        .as_str()
                        .ok_or_else(cloud::Error::evidence)?,
                    route["project"]
                        .as_str()
                        .ok_or_else(cloud::Error::evidence)?,
                    scope["wandb_id"]
                        .as_str()
                        .ok_or_else(cloud::Error::evidence)?
                ));
            }
        }
        Ok(
            json!({"project":project,"run_id":run,"wandb":route,"scopes":scopes,"source":"local-canonical-journal","cloud_query_performed":false}),
        )
    }
    pub fn history(&self, action: &str) -> cloud::Result<Value> {
        let prepared = self.load(action)?;
        let saved = self.exchange_saved(&prepared)?;
        Ok(
            json!({"submission_id":action,"status":if saved.as_ref().is_some_and(|s|s.complete){"COMPLETE"}else{"PARTIAL"},
            "received_bytes":saved.as_ref().and_then(|s|s.cursor.as_ref()).map_or(0,|c|c.offset_bytes),"records":saved.as_ref().map_or(0,|s|s.counts.records),
            "invalid_lines":saved.as_ref().map_or(0,|s|s.counts.invalid_lines),"scheduler_mutation_performed":false,"source":"native-shared-metric-history",
            "transport":"native-cloud-exchange","received_at":saved.as_ref().map(|s|&s.observed_at),"diagnostic":if saved.is_none(){json!("AWAITING_WORKER_EXCHANGE")}else{Value::Null}}),
        )
    }
    pub fn latest_registered_checkpoint(&self, action: &str) -> cloud::Result<Value> {
        let prepared = self.load(action)?;
        let context = self.checkpoint_context(&prepared)?;
        let mut values = crate::checkpoints::Store::open_existing(&self.settings.registry_root)
            .map_err(why!(cloud::Error::evidence()))?
            .list(&context)
            .map_err(why!(cloud::Error::evidence()))?;
        values.sort_by_key(|v| v["step"].as_u64());
        let checkpoint = values.pop().ok_or_else(missing)?;
        if values
            .last()
            .is_some_and(|v| v["step"] == checkpoint["step"])
        {
            return Err(cloud::Error::evidence());
        }
        Ok(
            json!({"submission_id":action,"checkpoint":checkpoint,"scheduler_mutation_performed":false,"source":"accepted-cloud-worker-checkpoint","remote_refresh_performed":false}),
        )
    }
}
impl crate::tracking::Source for Service {
    fn publisher_leader(&self) -> Result<FileLock, crate::tracking::Error> {
        self.directory
            .lock(
                OsStr::new(".publisher.lock"),
                LockOptions {
                    create: true,
                    ..Default::default()
                },
            )
            .map_err(why_lock!(crate::tracking::Error::Storage))
    }
    fn publisher_store(&self) -> Result<crate::records::Store, crate::tracking::Error> {
        crate::records::Store::open(&self.settings.registry_root, &self.workspace)
            .map_err(why!(crate::tracking::Error::Storage))
    }
    fn tracking_journal(
        &self,
        project: &str,
        run: &str,
    ) -> Result<crate::records::Store, crate::tracking::Error> {
        let prepared = self
            .saved(project, run)
            .map_err(why!(crate::tracking::Error::Identity))?
            .ok_or(crate::tracking::Error::Identity)?;
        self.exchange_records(&prepared)
            .map_err(why!(crate::tracking::Error::Identity))
    }
    /// The scheduler outbox's last provider observation, only where it proves the job
    /// is over.
    /// Nothing is fetched: the observer and client refreshes keep it current.
    fn provider_terminal(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
    ) -> Result<Option<String>, crate::tracking::Error> {
        let Some(prepared) = self
            .saved_attempt(project, run, attempt)
            .map_err(why!(crate::tracking::Error::Identity))?
        else {
            return Ok(None);
        };
        let Some(state) = self
            .outbox
            .optional_status(&prepared.execution.job)
            .map_err(why!(crate::tracking::Error::Storage))?
        else {
            return Ok(None);
        };
        Ok(super::sync_jobs::proven_terminal(&state)
            .map_err(why!(crate::tracking::Error::Storage))?
            .and_then(|_| state["observation"]["state"].as_str().map(Into::into)))
    }
}
