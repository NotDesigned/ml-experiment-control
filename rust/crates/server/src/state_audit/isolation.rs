//! Kernel-enforced denial of new writable opens, creation, removal, rename and
//! truncation. No allow-write rules and no weaker fallback. Only the dedicated
//! audit thread is restricted; neither its caller nor production services are.
use super::{Error, Result};
#[cfg(target_os = "linux")]
pub(super) fn restrict() -> Result<()> {
    use landlock::{ABI, AccessFs, CompatLevel, Compatible, Ruleset, RulesetAttr, RulesetStatus};
    let restriction = Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .handle_access(AccessFs::from_write(ABI::V3))
        .map_err(|_| Error::Isolation)?
        .create()
        .map_err(|_| Error::Isolation)?
        .restrict_self()
        .map_err(|_| Error::Isolation)?;
    if restriction.ruleset != RulesetStatus::FullyEnforced || !restriction.no_new_privs {
        return Err(Error::Isolation);
    }
    Ok(())
}
#[cfg(not(target_os = "linux"))]
pub(super) fn restrict() -> Result<()> {
    Err(Error::Isolation)
}
