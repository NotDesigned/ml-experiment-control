//! One tree per backend: each holds its provider effect shell and the service above
//! it. `runs` chooses between them, and the HTTP handlers that serve either backend sit
//! above this tree. Slurm names nothing in the cloud tree. Cloud implements the one
//! trait Slurm asks a builder through, `RuntimeBuilder`, and otherwise each service
//! receives what it needs from the caller as data.
pub mod cloud;
pub mod slurm;
