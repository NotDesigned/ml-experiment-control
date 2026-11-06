//! One tree per backend: each holds its provider effect shell and the service
//! above it. `runs` chooses between them. Neither service names the other's `Service`;
//! each asks through a one-question trait.
pub mod cloud;
pub mod slurm;
