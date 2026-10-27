//! Disposable SQLite Run projections. This is not run_detail or failure assessment.
mod store;
pub use store::{Error, Result, RunIndex};

#[cfg(test)]
mod tests;
