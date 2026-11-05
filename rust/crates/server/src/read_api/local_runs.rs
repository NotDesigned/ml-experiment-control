//! Local execution ingress shares the journal and publisher, never cloud launch.
use super::*;
use fs2::FileExt;
use ml_exp_core::{
    local::{Cursor, FORMAT, Packet, Registration},
    records::Event,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::json;

fn invalid() -> Error {
    error(
        409,
        "LOCAL_EVIDENCE_INVALID",
        "local execution metadata or accepted stream differs",
    )
}
fn io(_: std::io::Error) -> Error {
    invalid()
}
fn decision(e: ml_exp_core::local::Error) -> Error {
    error(
        409,
        e.0,
        "local execution or ordered JSONL identity differs; keep the same saved state",
    )
}
fn record(_: crate::records::Error) -> Error {
    invalid()
}
fn load<T: DeserializeOwned>(d: &Directory, name: &str) -> Result<T, Error> {
    serde_json::from_slice(&d.read_regular(OsStr::new(name), 1024 * 1024).map_err(io)?)
        .map_err(why!(invalid()))
}
fn save(d: &Directory, name: &str, v: &impl Serialize) -> Result<(), Error> {
    let bytes = serde_json::to_vec(v).map_err(why!(invalid()))?;
    if d.entry_exists(OsStr::new(name)).map_err(io)? {
        d.replace_regular(OsStr::new(name), &bytes).map_err(io)
    } else {
        d.publish_new(OsStr::new(name), &bytes).map_err(io)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedCursor {
    stream: Cursor,
    last_ingress_at: Option<String>,
}
fn cursor(d: &Directory, config: &Value) -> Result<SavedCursor, Error> {
    let saved: SavedCursor = load(d, "cursor.json")?;
    if !saved.stream.importer.matches("metrics.jsonl", config)
        || ml_exp_core::sources::validate_archive_digest(&saved.stream.chain_sha256).is_err()
        || saved.stream.offset == 0
            && saved.stream.finished.is_none()
            && saved.stream.chain_sha256 != ml_exp_core::local::INITIAL_CHAIN
    {
        return Err(invalid());
    }
    Ok(saved)
}
impl Api {
    pub fn with_local_tracking(mut self, settings: ml_exp_core::tracking::RoutingSettings) -> Self {
        self.local_tracking = settings;
        self
    }
    pub fn local_enabled(&self) -> bool {
        !self.local_roots.is_empty()
    }
    fn local_root(&self, project: &str, run: &str) -> Result<Directory, Error> {
        validate_id(project)
            .and_then(|_| validate_observation_id(run))
            .map_err(|_| invalid_query())?;
        self.validate_registry()?;
        self.ensure_active(project)?;
        let p = self.local_roots.get(project).ok_or_else(|| {
            error(
                404,
                "LOCAL_PROJECT_NOT_FOUND",
                "project has no canonical native Run root",
            )
        })?;
        let root = Directory::open(p, false).map_err(io)?;
        root.verify_private_parent().map_err(io)?;
        Ok(root)
    }
    /// Nonlocal Runs return None. Broken local authority is never a cloud fallback.
    fn local_document(
        &self,
        project: &str,
        run: &str,
    ) -> Result<Option<(Directory, Value, String)>, Error> {
        if !self.local_roots.contains_key(project) {
            return Ok(None);
        }
        let root = self.local_root(project, run)?;
        if !root.entry_exists(OsStr::new(run)).map_err(io)? {
            return Ok(None);
        }
        let dir = root.child(OsStr::new(run), false).map_err(io)?;
        if !dir.entry_exists(OsStr::new("manifest.json")).map_err(io)? {
            // Existing Slurm Runs can have only manifest.yaml. A local ingress
            // directory, however, must retain its registered JSON manifest.
            return if dir.entry_exists(OsStr::new("local-ingress")).map_err(io)? {
                Err(invalid())
            } else {
                Ok(None)
            };
        }
        let raw = dir
            .read_regular(OsStr::new("manifest.json"), 1024 * 1024)
            .map_err(io)?;
        let manifest: Value = serde_json::from_slice(&raw).map_err(why!(invalid()))?;
        if manifest["native_creation"]["format"] != FORMAT {
            return Ok(None);
        }
        let request: Registration = load(&dir, "request.json")?;
        if manifest
            != request
                .manifest(
                    project,
                    manifest["tracking_route"].clone(),
                    manifest["created_at"].as_str().ok_or_else(invalid)?,
                )
                .map_err(decision)?
            || request.run_id != run
        {
            return Err(invalid());
        }
        Ok(Some((
            dir,
            manifest,
            ml_exp_core::slurm::helper::sha256_bytes(&raw),
        )))
    }
    fn local_store(
        &self,
        project: &str,
        run: &str,
        manifest: &Value,
        sha: &str,
        write: bool,
    ) -> Result<crate::records::Store, Error> {
        let expected =
            ml_exp_core::records::native_binding(manifest, sha).map_err(why!(invalid()))?;
        let store = if write {
            crate::records::Store::open(
                &self.registry_root,
                self.workspace_id
                    .strip_prefix("rust-preview-")
                    .ok_or_else(invalid)?,
            )
            .map_err(record)?
        } else {
            crate::records::Store::read_only(&self.registry_root)
        };
        let guard = store.guard(project, run).map_err(record)?;
        if store.binding(&guard).map_err(record)?.as_ref() != Some(&expected) {
            return Err(invalid());
        }
        drop(guard);
        Ok(store)
    }
    pub fn local_journal(
        &self,
        project: &str,
        run: &str,
    ) -> Result<Option<crate::records::Store>, Error> {
        let Some((_, manifest, sha)) = self.local_document(project, run)? else {
            return Ok(None);
        };
        self.local_store(project, run, &manifest, &sha, true)
            .map(Some)
    }
    pub fn local_publisher_leader(&self) -> Result<std::fs::File, crate::tracking::Error> {
        let f = self
            .registry
            .open_lock(OsStr::new(".local-publisher.lock"))
            .map_err(why!(crate::tracking::Error::Storage))?;
        f.try_lock_exclusive()
            .map_err(|_| crate::tracking::Error::Storage)?;
        Ok(f)
    }
    pub fn validate_local_publisher_leader(
        &self,
        f: &std::fs::File,
    ) -> Result<(), crate::tracking::Error> {
        self.registry
            .validate_regular_entry(OsStr::new(".local-publisher.lock"), f)
            .map_err(|_| crate::tracking::Error::Storage)
    }
    pub fn local_publisher_store(&self) -> Result<crate::records::Store, crate::tracking::Error> {
        crate::records::Store::open(
            &self.registry_root,
            self.workspace_id
                .strip_prefix("rust-preview-")
                .ok_or(crate::tracking::Error::Storage)?,
        )
        .map_err(why!(crate::tracking::Error::Storage))
    }
    pub fn register_local(
        &self,
        project: &str,
        request: Registration,
        now: &str,
    ) -> Result<Value, Error> {
        if !self.local_enabled() {
            return Err(error(
                403,
                "LOCAL_RUN_CREATION_DISABLED",
                "local Run creation is unavailable",
            ));
        }
        request.validate(project).map_err(decision)?;
        let _admission = crate::administration::authorship_guard(&self.registry_root)
            .map_err(why!(invalid()))?;
        let fence = self
            .fence
            .existing_observation_guard(project, &request.run_id)
            .map_err(why!(invalid()))?;
        let root = self.local_root(project, &request.run_id)?;
        let registration_lock = self
            .registry
            .open_lock(OsStr::new(".local-registration.lock"))
            .map_err(io)?;
        registration_lock
            .try_lock_exclusive()
            .map_err(|_| error(503, "LOCAL_INGRESS_BUSY", "local registration is busy"))?;
        if root.entry_exists(OsStr::new(&request.run_id)).map_err(io)? {
            let Some((dir, _, _)) = self.local_document(project, &request.run_id)? else {
                return Err(invalid());
            };
            let original: Registration = load(&dir, "request.json")?;
            if json!(original) != json!(request) {
                return Err(error(
                    409,
                    "LOCAL_IDENTITY_CONFLICT",
                    "Run is frozen to a different local execution or definition",
                ));
            }
        } else {
            let route = ml_exp_core::tracking::resolve(
                request.wandb.as_ref(),
                &self.local_tracking,
                project,
            );
            let manifest = request.manifest(project, route, now).map_err(decision)?;
            let raw = serde_json::to_vec(&manifest).map_err(why!(invalid()))?;
            let binding = ml_exp_core::records::native_binding(
                &manifest,
                &ml_exp_core::slurm::helper::sha256_bytes(&raw),
            )
            .map_err(why!(invalid()))?;
            let (_, random) = crate::projects::transition_context()?;
            let staging = self
                .registry
                .child(OsStr::new(".local-staging"), true)
                .map_err(io)?
                .create_new_child(OsStr::new(&random))
                .map_err(io)?;
            staging
                .publish_new(OsStr::new("manifest.json"), &raw)
                .map_err(io)?;
            save(
                &staging,
                "status.json",
                &json!({"project":project,"run_id":request.run_id,"attempt_id":"attempt-001","execution_kind":"local"}),
            )?;
            save(&staging, "request.json", &request)?;
            let attempts = staging
                .create_new_child(OsStr::new("attempts"))
                .map_err(io)?;
            let attempt = attempts
                .create_new_child(OsStr::new("attempt-001"))
                .map_err(io)?;
            let mut document = manifest.clone();
            document["attempt_id"] = json!("attempt-001");
            save(&attempt, "attempt.json", &document)?;
            let ingress = staging
                .create_new_child(OsStr::new("local-ingress"))
                .map_err(io)?;
            ingress
                .create_new_child(OsStr::new("packets"))
                .map_err(io)?;
            save(
                &ingress,
                "cursor.json",
                &SavedCursor {
                    stream: Cursor::new(binding["config"].clone()).map_err(decision)?,
                    last_ingress_at: None,
                },
            )?;
            staging
                .open_lock(OsStr::new(".local-ingress.lock"))
                .map_err(io)?;
            fence
                .require_observed_live(project, &request.run_id)
                .map_err(why!(invalid()))?;
            staging
                .publish_directory(&root, OsStr::new(&request.run_id))
                .map_err(io)?;
        }
        let (dir, manifest, sha) = self
            .local_document(project, &request.run_id)?
            .ok_or_else(invalid)?;
        let store = crate::records::Store::open(
            &self.registry_root,
            self.workspace_id
                .strip_prefix("rust-preview-")
                .ok_or_else(invalid)?,
        )
        .map_err(record)?;
        let expected =
            ml_exp_core::records::native_binding(&manifest, &sha).map_err(why!(invalid()))?;
        {
            let guard = store.guard(project, &request.run_id).map_err(record)?;
            store
                .bind(&guard, &expected["route"], &expected["config"])
                .map_err(record)?;
        }
        let _ingress = dir
            .open_existing_lock(OsStr::new(".local-ingress.lock"))
            .map_err(io)?;
        _ingress
            .try_lock_exclusive()
            .map_err(|_| error(503, "LOCAL_INGRESS_BUSY", "local stream is busy"))?;
        let guard = store.guard(project, &request.run_id).map_err(record)?;
        let scope = store.scope(&guard, "attempt-001").map_err(record)?;
        store.append(&guard,&scope,&[Event{event_id:format!("local-attached.{}",request.execution_id),payload:json!({"kind":"lifecycle","data":{"phase":"LOCAL_ATTACHED","state":"RUNNING"}})}]).map_err(record)?;
        drop(guard);
        drop(_ingress);
        self.local_status(project, &request.run_id)
    }
    pub fn local_tracking_status(&self, project: &str, run: &str) -> Result<Option<Value>, Error> {
        if self.local_document(project, run)?.is_none() {
            return Ok(None);
        }
        self.local_status(project, run).map(Some)
    }
    pub fn local_status(&self, project: &str, run: &str) -> Result<Value, Error> {
        let _fence = self
            .fence
            .existing_observation_guard(project, run)
            .map_err(why!(invalid()))?;
        let Some((dir, manifest, sha)) = self.local_document(project, run)? else {
            return Err(error(
                404,
                "LOCAL_RUN_NOT_FOUND",
                "local execution is not registered",
            ));
        };
        let ingress = dir.child(OsStr::new("local-ingress"), false).map_err(io)?;
        let binding =
            ml_exp_core::records::native_binding(&manifest, &sha).map_err(why!(invalid()))?;
        let saved = cursor(&ingress, &binding["config"])?;
        let cursor = saved.stream;
        let store = self.local_store(project, run, &manifest, &sha, false)?;
        let guard = store.guard(project, run).map_err(record)?;
        let scopes = store.run_scopes(&guard).map_err(record)?;
        Ok(
            json!({"project":project,"run_id":run,"attempt_id":"attempt-001","execution_id":manifest["local_execution"]["execution_id"],
            "execution_kind":"local","process_control":false,"process_liveness":"UNKNOWN","completion_source":"client_reported",
            "submission_id":null,"provider_job_id":null,"last_ingress_at":saved.last_ingress_at,
            "state":cursor.finished.map(|f|json!(f)).unwrap_or(json!("ATTACHED")),
            "cursor":{"offset":cursor.offset,"chain_sha256":cursor.chain_sha256,"finished":cursor.finished,"records":cursor.counts.records,"invalid_lines":cursor.counts.invalid_lines},
            "wandb":manifest["tracking_route"],"scopes":scopes,"source":"local-jsonl-ingress","provider_queries_performed":false}),
        )
    }
    pub fn ingest_local(
        &self,
        project: &str,
        run: &str,
        packet: Packet,
        now: &str,
    ) -> Result<Value, Error> {
        let digest = packet.digest().map_err(decision)?;
        let _fence = self
            .fence
            .existing_observation_guard(project, run)
            .map_err(why!(invalid()))?;
        let Some((dir, manifest, sha)) = self.local_document(project, run)? else {
            return Err(error(
                404,
                "LOCAL_RUN_NOT_FOUND",
                "local execution is not registered",
            ));
        };
        if manifest["local_execution"]["execution_id"] != packet.execution_id {
            return Err(error(
                409,
                "LOCAL_IDENTITY_CONFLICT",
                "local execution differs",
            ));
        }
        let lock = dir
            .open_existing_lock(OsStr::new(".local-ingress.lock"))
            .map_err(io)?;
        lock.try_lock_exclusive().map_err(|_| {
            error(
                503,
                "LOCAL_INGRESS_BUSY",
                "local stream is busy; repeat the same packet",
            )
        })?;
        dir.validate_regular_entry(OsStr::new(".local-ingress.lock"), &lock)
            .map_err(io)?;
        let ingress = dir.child(OsStr::new("local-ingress"), false).map_err(io)?;
        let receipts = ingress.child(OsStr::new("packets"), false).map_err(io)?;
        let binding =
            ml_exp_core::records::native_binding(&manifest, &sha).map_err(why!(invalid()))?;
        let before = cursor(&ingress, &binding["config"])?;
        let name = format!(
            "{}-{}.json",
            packet.offset,
            if packet.bytes.is_empty() {
                "finish"
            } else {
                "data"
            }
        );
        let receipt = if receipts.entry_exists(OsStr::new(&name)).map_err(io)? {
            Some(load::<Value>(&receipts, &name)?)
        } else {
            None
        };
        if let Some(saved) = &receipt {
            if saved["packet_sha256"] != digest {
                return Err(error(
                    409,
                    "LOCAL_PACKET_CONFLICT",
                    "this offset is already bound to different bytes or completion",
                ));
            }
            if saved["project"] != project
                || saved["run_id"] != run
                || saved["execution_id"] != packet.execution_id
                || saved["offset"]
                    != packet
                        .offset
                        .checked_add(packet.bytes.len() as u64)
                        .ok_or_else(invalid)?
                || saved["chain_sha256"] != digest
                || saved["finished"] != json!(packet.finish)
            {
                return Err(invalid());
            }
            if packet.offset < before.stream.offset || before.stream.finished.is_some() {
                return Ok(saved.clone());
            }
        }
        let (next, events) = before.stream.advance(&packet).map_err(decision)?;
        let intent_name = format!("{name}.intent");
        if receipts
            .entry_exists(OsStr::new(&intent_name))
            .map_err(io)?
        {
            if load::<Value>(&receipts, &intent_name)? != json!({"packet_sha256":digest}) {
                return Err(error(
                    409,
                    "LOCAL_PACKET_CONFLICT",
                    "this offset is already bound to a different packet",
                ));
            }
        } else {
            save(&receipts, &intent_name, &json!({"packet_sha256":digest}))?;
        }
        let store = self.local_store(project, run, &manifest, &sha, true)?;
        let guard = store.guard(project, run).map_err(record)?;
        let scope = store.scope(&guard, "attempt-001").map_err(record)?;
        let through = store.append(&guard, &scope, &events).map_err(record)?;
        let ack = json!({"project":project,"run_id":run,"execution_id":packet.execution_id,"packet_sha256":digest,"offset":next.offset,"chain_sha256":next.chain_sha256,"finished":next.finished,"journal_through":through,"accepted_at":receipt.as_ref().and_then(|r|r["accepted_at"].as_str()).unwrap_or(now)});
        if let Some(saved) = receipt {
            if saved != ack {
                return Err(invalid());
            }
        } else {
            save(&receipts, &name, &ack)?;
        }
        // Journal and exact receipt precede the cursor. A crash replays the same Event IDs.
        save(
            &ingress,
            "cursor.json",
            &SavedCursor {
                stream: next,
                last_ingress_at: Some(ack["accepted_at"].as_str().ok_or_else(invalid)?.into()),
            },
        )?;
        self.ensure_active(project)?;
        Ok(ack)
    }
}
