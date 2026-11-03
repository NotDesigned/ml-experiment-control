//! Irreversible restrictions belong to a dedicated thread, never a pooled
//! worker or the calling service. SQLite's READ_ONLY flag alone can create WAL
//! auxiliaries; deny filesystem writes in the kernel before opening state.
#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("kernel-enforced read-only observation is unavailable")]
pub struct Error;

#[cfg(target_os = "linux")]
fn restrict() -> Result<(), Error> {
    use landlock::{ABI, AccessFs, CompatLevel, Compatible, Ruleset, RulesetAttr, RulesetStatus};
    let restriction = Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .handle_access(AccessFs::from_write(ABI::V3))
        .map_err(|_| Error)?
        .create()
        .map_err(|_| Error)?
        .restrict_self()
        .map_err(|_| Error)?;
    if restriction.ruleset != RulesetStatus::FullyEnforced || !restriction.no_new_privs {
        return Err(Error);
    }
    Ok(())
}
#[cfg(not(target_os = "linux"))]
fn restrict() -> Result<(), Error> {
    Err(Error)
}

pub(crate) fn run<T: Send>(operation: impl FnOnce() -> T + Send) -> Result<T, Error> {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("historical-read-only".into())
            .spawn_scoped(scope, move || {
                restrict()?;
                Ok(operation())
            })
            .map_err(|_| Error)?
            .join()
            .map_err(|_| Error)?
    })
}
