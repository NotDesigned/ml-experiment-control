//! What one publisher thread knows only while it runs. Losing all of this in a
//! restart costs a journal replay of the display, never a row or a cursor.
use super::*;
use std::collections::BTreeMap;

/// Rows one visit sends. W&B is not read back, so a batch costs one journal
/// read, one summary and one SDK acknowledgement however large it is.
const BATCH: i64 = 8192;
/// Journal payload one visit may load.
pub(super) const WINDOW_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Next {
    /// More to do for this scope now: rows left, or a Run to finish.
    More,
    /// Nothing to do until the journal grows.
    Idle,
}

pub(super) struct Session {
    pub(super) sdk: Option<Sdk>,
    /// Rows one visit sends; [`BATCH`] outside tests.
    pub(super) batch: i64,
    idle: BTreeMap<String, i64>,
    projections: BTreeMap<String, tracking::Projection>,
    /// The W&B id each scope's entries above are about.
    bound: BTreeMap<String, String>,
}
impl Default for Session {
    fn default() -> Self {
        Self {
            sdk: None,
            batch: BATCH,
            idle: BTreeMap::new(),
            projections: BTreeMap::new(),
            bound: BTreeMap::new(),
        }
    }
}
impl Session {
    pub(super) fn open(&mut self, runtime: &Runtime) -> Result<&mut Sdk> {
        if self.sdk.is_none() {
            self.sdk = Some(Sdk::open(&runtime.config.python)?);
        }
        self.sdk.as_mut().ok_or(Error::Transport)
    }
    /// Killing the child loses what wandb-core still holds; W&B may then miss
    /// rows the SDK had accepted. That is the mirror's accepted cost.
    pub(super) fn lost(&mut self) {
        self.sdk = None;
    }
    pub(super) fn shutdown(&mut self) {
        if let Some(sdk) = self.sdk.as_mut() {
            let _ = sdk.shutdown();
        }
    }
    pub(super) fn retain(&mut self, scopes: &BTreeSet<String>) {
        self.idle.retain(|scope, _| scopes.contains(scope));
        self.projections.retain(|scope, _| scopes.contains(scope));
        self.bound.retain(|scope, _| scopes.contains(scope));
    }
    /// A scope a W&B rebuild moved to another id keeps nothing of the old Run.
    pub(super) fn rebind(&mut self, scope: &str, wandb_id: &str) {
        if self.bound.get(scope).is_some_and(|bound| bound == wandb_id) {
            return;
        }
        self.bound.insert(scope.into(), wandb_id.into());
        self.idle.remove(scope);
        self.projections.remove(scope);
    }
    pub(super) fn is_idle(&self, scope: &str, total: i64) -> bool {
        self.idle.get(scope) == Some(&total)
    }
    pub(super) fn mark_idle(&mut self, scope: &str, total: i64) {
        self.idle.insert(scope.into(), total);
    }
    /// The journal is append-only, so the display only ever needs the records
    /// since the last call, instead of a replay per batch.
    pub(super) fn display(
        &mut self,
        store: &Store,
        guard: &RecordGuard<'_>,
        scope: &str,
        through: i64,
    ) -> crate::records::Result<serde_json::Value> {
        let projection = self
            .projections
            .remove(scope)
            .filter(|p| p.cursor() <= through)
            .unwrap_or_default();
        let projection = store.advance_display(guard, scope, projection, through)?;
        let document = store.display_document(&projection);
        self.projections.insert(scope.into(), projection);
        document
    }
    /// One request for this scope's W&B Run. An error frame leaves the pipe
    /// usable but maybe not this Run's session, so that session is closed and
    /// the next request opens it again.
    pub(super) fn call(
        &mut self,
        runtime: &Runtime,
        request: &serde_json::Value,
        cancelled: &AtomicBool,
    ) -> Result<serde_json::Value> {
        let result = self.open(runtime)?.call(request, cancelled);
        if matches!(result, Err(Error::SdkRequest(_))) {
            let close = json!({"operation":"close","wandb_id":request["wandb_id"]});
            let closed = self.sdk.as_mut().map(|sdk| sdk.call(&close, cancelled));
            if !matches!(closed, Some(Ok(_))) {
                self.lost();
            }
        }
        result
    }
}
