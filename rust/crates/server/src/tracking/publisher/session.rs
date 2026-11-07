//! What one publisher thread knows only while it runs. Losing all of this in a
//! restart costs a settle delay and a journal replay, never a proof or a row.
use super::*;

/// Rows one scope may have awaiting remote history, and so the most W&B reads
/// one proof covers. Rate is this over the time rows take to become visible.
const WINDOW: i64 = 4096;
/// Rows awaiting history across all Runs: what wandb-core may hold in memory.
const IN_FLIGHT: i64 = 16384;
const MIN_SHARE: i64 = 128;
/// Journal payload one visit may load.
pub(super) const WINDOW_BYTES: usize = 16 * 1024 * 1024;
/// The publication state document is bounded at 1 MiB, and during a repair it
/// holds the reservations and the repair's copy of them.
pub(super) const RESERVATION_BYTES: usize = 450 * 1024;
/// Remote history must stay unchanged this long before rows no live SDK child
/// holds are treated as absent: a dying wandb-core or W&B ingestion may yet
/// deliver them. Measured from when this process first saw that evidence, so a
/// restart can only lengthen it.
const SETTLE: Duration = Duration::from_secs(180);
/// A live session showing no remote progress for this long is closed, which
/// flushes it, and its rows then settle like any other uncertain send.
pub(super) const STALL: Duration = Duration::from_secs(600);

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Next {
    /// Rows are on their way or being observed: read W&B again soon.
    Poll,
    /// Nothing to do until the journal grows.
    Idle,
}

struct Watch {
    since: Instant,
    evidence: serde_json::Value,
}

pub(super) struct Session {
    pub(super) sdk: Option<Sdk>,
    /// Scopes whose reservations were handed to the running SDK child, which
    /// may still deliver them. Every other reservation is uncertain.
    held: BTreeSet<String>,
    backlog: BTreeSet<String>,
    idle: BTreeMap<String, i64>,
    watches: BTreeMap<String, Watch>,
    projections: BTreeMap<String, tracking::Projection>,
    /// The W&B id each scope's entries above are about.
    bound: BTreeMap<String, String>,
    pub(super) settle: Duration,
}
impl Default for Session {
    fn default() -> Self {
        Self {
            sdk: None,
            held: BTreeSet::new(),
            backlog: BTreeSet::new(),
            idle: BTreeMap::new(),
            watches: BTreeMap::new(),
            projections: BTreeMap::new(),
            bound: BTreeMap::new(),
            settle: SETTLE,
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
    /// Killing the child can make rows it held arrive late. Their settle delay starts
    /// from here: a scan unchanged since before the kill says nothing about after it.
    pub(super) fn lost(&mut self) {
        self.sdk = None;
        self.held.clear();
        self.watches.clear();
    }
    pub(super) fn shutdown(&mut self) {
        if let Some(sdk) = self.sdk.as_mut() {
            let _ = sdk.shutdown();
        }
        self.held.clear();
    }
    pub(super) fn retain(&mut self, scopes: &BTreeSet<String>) {
        self.backlog.retain(|scope| scopes.contains(scope));
        self.idle.retain(|scope, _| scopes.contains(scope));
        self.watches.retain(|scope, _| scopes.contains(scope));
        self.projections.retain(|scope, _| scopes.contains(scope));
        self.bound.retain(|scope, _| scopes.contains(scope));
    }
    /// A scope a W&B rebuild moved to another id keeps nothing of the old Run:
    /// not its display, its idle mark, its remote evidence or its reservations.
    pub(super) fn rebind(&mut self, scope: &str, wandb_id: &str) {
        if self.bound.get(scope).is_some_and(|bound| bound == wandb_id) {
            return;
        }
        self.bound.insert(scope.into(), wandb_id.into());
        self.held.remove(scope);
        self.backlog.remove(scope);
        self.idle.remove(scope);
        self.watches.remove(scope);
        self.projections.remove(scope);
    }
    /// An equal share per Run with work, so a large backlog can hold only its
    /// share of the rows in flight while a live Run keeps its own.
    pub(super) fn share(&self, scope: &str) -> i64 {
        let others = self.backlog.iter().filter(|s| *s != scope).count() as i64;
        (IN_FLIGHT / (others + 1)).clamp(MIN_SHARE, WINDOW)
    }
    pub(super) fn note_backlog(&mut self, scope: &str, behind: bool) {
        if behind {
            self.backlog.insert(scope.into());
        } else {
            self.backlog.remove(scope);
        }
    }
    pub(super) fn is_idle(&self, scope: &str, total: i64) -> bool {
        self.idle.get(scope) == Some(&total)
    }
    pub(super) fn mark_idle(&mut self, scope: &str, total: i64) {
        self.idle.insert(scope.into(), total);
    }
    pub(super) fn holds(&self, scope: &str) -> bool {
        self.held.contains(scope)
    }
    pub(super) fn forget_watch(&mut self, scope: &str) {
        self.watches.remove(scope);
    }
    /// How long this exact remote evidence has been seen. `None` on a first
    /// sighting or a change, so two scans always separate any decision from
    /// the start of its clock.
    pub(super) fn stable_for(
        &mut self,
        scope: &str,
        evidence: &serde_json::Value,
    ) -> Option<Duration> {
        match self.watches.get(scope) {
            Some(watch) if watch.evidence == *evidence => Some(watch.since.elapsed()),
            _ => {
                self.watches.insert(
                    scope.into(),
                    Watch {
                        since: Instant::now(),
                        evidence: evidence.clone(),
                    },
                );
                None
            }
        }
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
    /// Hand effects to the SDK. The caller has already saved the reservation.
    pub(super) fn send(
        &mut self,
        scope: &str,
        job: &serde_json::Value,
        frames: Vec<serde_json::Value>,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        self.held.insert(scope.into());
        for frame in frames {
            let applied = self
                .sdk
                .as_mut()
                .ok_or(Error::Transport)?
                .call(&frame, cancelled);
            match applied {
                Ok(value) if value["applied"] == true => {}
                Ok(_) => return Err(Error::Transport),
                Err(error) => {
                    // The pipe still works but this Run's session may not.
                    if matches!(error, Error::SdkRequest(_)) {
                        self.abandon(scope, job, cancelled);
                    }
                    return Err(error);
                }
            }
        }
        Ok(())
    }
    /// Make this scope's rows uncertain: end its session, which flushes what the
    /// SDK holds, so nothing else can arrive once they have settled. Another
    /// Run's session is lost only if the SDK cannot do even that.
    pub(super) fn abandon(&mut self, scope: &str, job: &serde_json::Value, cancelled: &AtomicBool) {
        self.held.remove(scope);
        self.watches.remove(scope);
        let close = json!({"operation":"close","wandb_id":job["wandb_id"]});
        let closed = self.sdk.as_mut().map(|sdk| sdk.call(&close, cancelled));
        if !matches!(closed, Some(Ok(_))) {
            self.lost();
        }
    }
}
