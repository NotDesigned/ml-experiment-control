//! One bounded SSH invocation of the native Slurm helper. The caller must have
//! persisted/authorized its immutable intent before calling; this library does
//! not select Attempts, consume budgets, retry or execute legacy command plans.
mod transport;
pub use transport::*;
pub mod action;

#[cfg(test)]
mod tests;
