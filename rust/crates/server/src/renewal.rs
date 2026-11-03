//! Small immutable gate-replacement records shared by both provider shells.
use ml_exp_io::directory::Directory;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Record<T> {
    pub original: String,
    pub request: ml_exp_core::renewal::Request,
    pub prepared: T,
}
use std::{ffi::OsStr, io};
fn invalid() -> io::Error {
    io::Error::other("renewal record differs or is unavailable")
}
pub(crate) fn read<T: DeserializeOwned>(slots: &Directory, name: &str) -> io::Result<Option<T>> {
    if !slots.entry_exists(OsStr::new(name))? {
        return Ok(None);
    }
    let bytes = slots.read_regular(OsStr::new(name), 32 * 1024 * 1024)?;
    let value = ml_exp_core::json::mapping(&bytes).map_err(|_| invalid())?;
    serde_json::from_value(serde_json::Value::Object(value))
        .map(Some)
        .map_err(|_| invalid())
}
pub(crate) fn retain(slots: &Directory, name: &str, value: &impl Serialize) -> io::Result<()> {
    let bytes = serde_json::to_vec(value)?;
    if slots.entry_exists(OsStr::new(name))? {
        if slots.read_regular(OsStr::new(name), 32 * 1024 * 1024)? != bytes {
            return Err(invalid());
        }
    } else {
        slots.publish_new(OsStr::new(name), &bytes)?;
    }
    Ok(())
}
pub(crate) fn expiry(now: &str) -> Result<String, crate::read_api::Error> {
    let invalid = || crate::read_api::Error {
        status: 422,
        code: "INVALID_RENEWAL_TIME",
        detail: "submission renewal time is invalid",
    };
    let start = time::OffsetDateTime::parse(now, &time::format_description::well_known::Rfc3339)
        .map_err(|_| invalid())?;
    (start + time::Duration::minutes(30))
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|_| invalid())
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Link {
    original: String,
    replacement: String,
    request_id: String,
}
pub(crate) fn link(
    slots: &Directory,
    original: &str,
    replacement: &str,
    request: &ml_exp_core::renewal::Request,
) -> io::Result<()> {
    request.validate(original).map_err(|_| invalid())?;
    ml_exp_core::operations::validate_submission_id(replacement).map_err(|_| invalid())?;
    let value = Link {
        original: original.into(),
        replacement: replacement.into(),
        request_id: request.request_id.clone(),
    };
    retain(slots, &format!("renewal-origin-{replacement}.json"), &value)?;
    retain(slots, &format!("renewal-next-{original}.json"), &value)
}
pub(crate) fn decorate(
    slots: &Directory,
    mut view: serde_json::Value,
) -> io::Result<serde_json::Value> {
    let action = view["submission_id"]
        .as_str()
        .ok_or_else(invalid)?
        .to_owned();
    ml_exp_core::operations::validate_submission_id(&action).map_err(|_| invalid())?;
    for (side, key) in [
        ("origin", "renewed_from"),
        ("next", "replacement_submission_id"),
    ] {
        if let Some(link) = read::<Link>(slots, &format!("renewal-{side}-{action}.json"))? {
            ml_exp_core::renewal::Request {
                request_id: link.request_id.clone(),
                confirmation: format!("RENEW {}", link.original),
            }
            .validate(&link.original)
            .map_err(|_| invalid())?;
            ml_exp_core::operations::validate_submission_id(&link.replacement)
                .map_err(|_| invalid())?;
            if side == "origin" && link.replacement != action
                || side == "next" && link.original != action
            {
                return Err(invalid());
            }
            view[key] = serde_json::json!(if side == "origin" {
                link.original
            } else {
                link.replacement
            });
            if side == "origin" {
                view["renewal_request_id"] = serde_json::json!(link.request_id);
            }
        }
    }
    Ok(view)
}
