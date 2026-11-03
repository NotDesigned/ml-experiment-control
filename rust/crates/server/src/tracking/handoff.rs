//! Explicit pending-scope handoff after a quiescent, verified migration.
//! A separate historical journal remains read-only to API authorship. Only
//! publication receipts and monotonic confirmations for these pinned scopes
//! are writable; completed and opted-out scopes are never discovered here.
use super::{Error, Result, Source};
use crate::{
    history,
    records::{RecordGuard, Store},
};
use fs2::FileExt;
use ml_exp_core::{canonical, records::Scope};
use ml_exp_io::directory::Directory;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    ffi::OsStr,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScopePin {
    pub scope: String,
    pub wandb_id: String,
    pub binding_sha256: String,
    pub events_sha256: String,
    pub confirmed: i64,
    pub total: i64,
    /// Old in-memory sends cannot be reconstructed. Pin their entire possible
    /// suffix as uncertain; remote ACK is required and missing rows never replay.
    pub reserved_through: i64,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema_version: u32,
    pub registry_root: PathBuf,
    pub workspace_id: String,
    pub scopes: Vec<ScopePin>,
}
pub struct Handoff {
    config: Config,
    root: Directory,
    history: Arc<history::Bridge>,
}
impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            ml_exp_io::read_private(path, 1024 * 1024, true).map_err(|_| Error::Configuration)?;
        let config: Self = serde_json::from_value(Value::Object(
            ml_exp_core::json::mapping(&bytes).map_err(|_| Error::Configuration)?,
        ))
        .map_err(|_| Error::Configuration)?;
        config.validate()?;
        Ok(config)
    }
    fn validate(&self) -> Result<()> {
        let mut scopes = BTreeSet::new();
        if self.schema_version != 1
            || !self.registry_root.is_absolute()
            || self.registry_root.components().any(|c| {
                matches!(
                    c,
                    std::path::Component::ParentDir | std::path::Component::CurDir
                )
            })
            || self.workspace_id.len() != 32
            || !self
                .workspace_id
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || self.scopes.is_empty()
            || self.scopes.len() > 256
        {
            return Err(Error::Configuration);
        }
        for pin in &self.scopes {
            Scope::parse(&pin.scope).map_err(|_| Error::Configuration)?;
            if !scopes.insert(&pin.scope)
                || pin.confirmed < 0
                || pin.total <= pin.confirmed
                || pin.total > 100_000
                || pin.reserved_through < pin.confirmed
                || pin.reserved_through > pin.total
                || pin.wandb_id.is_empty()
                || pin.wandb_id.len() > 128
                || !pin.wandb_id.bytes().all(|b| b.is_ascii_alphanumeric())
                || [&pin.binding_sha256, &pin.events_sha256]
                    .iter()
                    .any(|v| !ml_exp_core::slurm::helper::valid_digest(v))
            {
                return Err(Error::Configuration);
            }
        }
        Ok(())
    }
    /// Read-only preflight. It never creates tables, changes a cursor, opens an
    /// SDK session or accepts an initially completed/disabled scope.
    pub fn check(&self) -> Result<()> {
        self.validate()?;
        let root = Directory::open(&self.registry_root, false).map_err(|_| Error::Storage)?;
        root.verify_private_parent().map_err(|_| Error::Storage)?;
        if crate::read_config::has_native_marker(&root).map_err(|_| Error::Storage)? {
            return Err(Error::Identity);
        }
        let store = Store::read_only(&self.registry_root);
        if store.existing_workspace_id().map_err(|_| Error::Storage)? != self.workspace_id {
            return Err(Error::Identity);
        }
        for pin in &self.scopes {
            validate_pin(&store, pin, true)?;
        }
        Ok(())
    }
}
impl Handoff {
    pub fn open(
        config: Config,
        historical: history::Settings,
        native_root: &Path,
    ) -> Result<Arc<Self>> {
        config.check()?;
        if historical.reads.registry_root != config.registry_root
            || historical.journal_workspace_id.as_deref() != Some(config.workspace_id.as_str())
            || native_root.starts_with(&config.registry_root)
            || config.registry_root.starts_with(native_root)
        {
            return Err(Error::Identity);
        }
        let history = history::Bridge::open(historical).map_err(|_| Error::Identity)?;
        history
            .validate_separation(&[native_root.to_owned()])
            .map_err(|_| Error::Identity)?;
        for pin in &config.scopes {
            let parsed = Scope::parse(&pin.scope).map_err(|_| Error::Identity)?;
            if history
                .api()
                .historical_tracking(&parsed.project, &parsed.run)
                .map_err(|_| Error::Identity)?
                .is_none()
            {
                return Err(Error::Identity);
            }
        }
        let root = Directory::open(&config.registry_root, false).map_err(|_| Error::Storage)?;
        Ok(Arc::new(Self {
            config,
            root,
            history,
        }))
    }
    fn store(&self) -> Result<Store> {
        self.root
            .verify_private_parent()
            .map_err(|_| Error::Identity)?;
        let store = Store::open(&self.config.registry_root, &self.config.workspace_id)
            .map_err(|_| Error::Storage)?;
        if store.existing_workspace_id().map_err(|_| Error::Storage)? != self.config.workspace_id {
            return Err(Error::Identity);
        }
        Ok(store)
    }
}
impl Source for Handoff {
    fn publisher_leader(&self) -> Result<File> {
        let file = self
            .root
            .open_lock(OsStr::new(".historical-publisher.lock"))
            .map_err(|_| Error::Storage)?;
        file.try_lock_exclusive().map_err(|_| Error::Storage)?;
        self.validate_publisher_leader(&file)?;
        Ok(file)
    }
    fn validate_publisher_leader(&self, file: &File) -> Result<()> {
        self.root
            .validate_regular_entry(OsStr::new(".historical-publisher.lock"), file)
            .map_err(|_| Error::Identity)
    }
    fn publisher_store(&self) -> Result<Store> {
        let store = self.store()?;
        for pin in &self.config.scopes {
            validate_pin(&store, pin, true)?;
            let scope = Scope::parse(&pin.scope).map_err(|_| Error::Identity)?;
            let guard = store
                .guard(&scope.project, &scope.run)
                .map_err(|_| Error::Identity)?;
            let mut state = ml_exp_core::tracking::publication::PublicationState {
                confirmed: pin.confirmed,
                ..Default::default()
            };
            let mut after = pin.confirmed;
            while after < pin.reserved_through {
                let rows = store
                    .events(
                        &guard,
                        &pin.scope,
                        after,
                        (pin.reserved_through - after).min(128),
                    )
                    .map_err(|_| Error::Storage)?;
                if rows.is_empty() {
                    return Err(Error::Identity);
                }
                for row in rows {
                    state.sent.insert(row.seq, row.event_id);
                    after = row.seq;
                }
            }
            store.initialize_publication_state(&guard,&pin.scope,&json!({"state":state,"finish_requested":false,"remote_finished":false,"ever_opened":pin.reserved_through>0})).map_err(|_|Error::Storage)?;
        }
        Ok(store)
    }
    fn tracking_journal(&self, project: &str, run: &str) -> Result<Store> {
        if !self
            .config
            .scopes
            .iter()
            .any(|p| Scope::parse(&p.scope).is_ok_and(|s| s.project == project && s.run == run))
            || self
                .history
                .api()
                .historical_tracking(project, run)
                .map_err(|_| Error::Identity)?
                .is_none()
        {
            return Err(Error::Identity);
        }
        let store = self.store()?;
        for pin in
            self.config.scopes.iter().filter(|p| {
                Scope::parse(&p.scope).is_ok_and(|s| s.project == project && s.run == run)
            })
        {
            validate_pin(&store, pin, false)?;
        }
        Ok(store)
    }
    fn publication_scopes(&self, store: &Store) -> Result<BTreeSet<String>> {
        let mut result = BTreeSet::new();
        for pin in &self.config.scopes {
            let parsed = Scope::parse(&pin.scope).map_err(|_| Error::Identity)?;
            let guard = store
                .guard(&parsed.project, &parsed.run)
                .map_err(|_| Error::Identity)?;
            validate_pin(store, pin, false)?;
            if store
                .publication_state(&guard, &pin.scope)
                .map_err(|_| Error::Storage)?
                .is_some_and(|s| s["remote_finished"] == true)
            {
                continue;
            }
            result.insert(pin.scope.clone());
        }
        Ok(result)
    }
    fn historical_lifecycle(&self) -> bool {
        true
    }
}
fn validate_pin(store: &Store, pin: &ScopePin, verify_events: bool) -> Result<()> {
    let scope = Scope::parse(&pin.scope).map_err(|_| Error::Identity)?;
    let guard = store
        .guard(&scope.project, &scope.run)
        .map_err(|_| Error::Identity)?;
    let status = store
        .scope_status(&guard, &pin.scope)
        .map_err(|_| Error::Storage)?
        .ok_or(Error::Identity)?;
    let binding = store
        .binding(&guard)
        .map_err(|_| Error::Storage)?
        .ok_or(Error::Identity)?;
    if status["wandb_id"] != pin.wandb_id
        || status["total"] != pin.total
        || !status["confirmed"]
            .as_i64()
            .is_some_and(|n| pin.confirmed <= n && n <= pin.total)
        || binding["route"]["enabled"] != true
        || status["status"] == "LOCAL_ONLY"
        || canonical::protocol_identity(&binding).map_err(|_| Error::Identity)?
            != pin.binding_sha256
    {
        return Err(Error::Identity);
    }
    if verify_events && event_digest(store, &guard, &pin.scope, pin.total)? != pin.events_sha256 {
        return Err(Error::Identity);
    }
    Ok(())
}
fn event_digest(store: &Store, guard: &RecordGuard<'_>, scope: &str, total: i64) -> Result<String> {
    let mut hash = Sha256::new();
    let mut after = 0;
    while after < total {
        let rows = store
            .events(guard, scope, after, (total - after).min(128))
            .map_err(|_| Error::Storage)?;
        if rows.is_empty() {
            return Err(Error::Identity);
        }
        for row in rows {
            if row.seq != after + 1 {
                return Err(Error::Identity);
            }
            hash.update(
                canonical::python_json_bytes(&row.document()).map_err(|_| Error::Identity)?,
            );
            hash.update(b"\n");
            after = row.seq;
        }
    }
    Ok(format!("sha256:{:x}", hash.finalize()))
}

#[cfg(test)]
mod tests;
