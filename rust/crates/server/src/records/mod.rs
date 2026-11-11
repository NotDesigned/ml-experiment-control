//! Canonical experiment journal I/O. SQLite acceptance is independent of optional
//! publishers. All normal reads and writes require an exact Run guard.
//! No HTTP, provider, scheduler or scientific payload file access.
mod store;
pub use store::{DerivedBatch, EventRow, ImportSummary, RecordGuard, Store, WandbRebuild};

pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Storage(#[from] crate::storage::Error),
    #[error("record journal I/O failed")]
    Io(#[from] std::io::Error),
    #[error("record journal SQLite operation failed")]
    Sqlite(#[from] rusqlite::Error),
    #[error("record journal JSON is invalid")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Decision(#[from] ml_exp_core::records::Error),
    #[error(transparent)]
    Fence(#[from] crate::deletion::Error),
    #[error("record journal is read-only")]
    ReadOnly,
    #[error("record mutation requires this Store's explicit Run guard")]
    Guard,
    #[error("tracking identity is already frozen")]
    FrozenIdentity,
    #[error("event ID is already bound to different content")]
    EventConflict,
    #[error("Run records are not bound")]
    Unbound,
    #[error("record scope does not exist or belongs to another Run")]
    Scope,
    #[error("a publication cursor advance does not match the frozen identity or journal")]
    Confirmation,
    #[error("derived metrics require an Attempt whose Run has ended")]
    NotEnded,
    #[error("a derived metric reuses a training-time metric name of this Attempt")]
    DerivedNameConflict,
    #[error("a W&B rebuild requires a finished W&B publication of every record")]
    RebuildNotFinished,
    #[error("invalid record storage metadata: {0}")]
    Invalid(&'static str),
}

#[cfg(test)]
mod tests;
