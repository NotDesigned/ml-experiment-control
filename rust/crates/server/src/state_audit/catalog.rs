//! Only fixed control leaves are read. Source trees, image layers and dataset
//! archives are never enumerated or opened by this metadata catalogue.
use super::*;
const MAX_CONTROL_BYTES: usize = 64 * 1024 * 1024;
pub(super) fn capture(
    settings: &read_config::Settings,
    root: &Directory,
    audit: &mut Audit,
) -> Result<()> {
    let mut bytes = 0usize;
    for project in &settings.projects {
        if let Some(directory) = nested(root, &["source-revisions", "sources", &project.project])? {
            for leaf in directory
                .entry_names_limited(4097)
                .map_err(|_| Error::Evidence)?
            {
                let name = leaf.to_str().ok_or(Error::Evidence)?;
                if !name.starts_with("source.") {
                    continue;
                }
                ml_exp_core::sources::validate_source_id(name).map_err(|_| Error::Evidence)?;
                let source = directory.child(&leaf, false).map_err(|_| Error::Evidence)?;
                let (value, sha) = mapping(&source, "source.json")?.ok_or(Error::Evidence)?;
                budget(&value, &mut bytes)?;
                if value["project"] != project.project
                    || value["source_id"] != name
                    || value["tree_digest"]
                        != format!("sha256:{}", name.trim_start_matches("source."))
                {
                    audit.issue(
                        key(&project.project, None, None, "source-receipt", Some(name)),
                        "SOURCE_IDENTITY_CONFLICT",
                    )?;
                }
                audit.push(
                    key(&project.project, None, None, "source-receipt", Some(name)),
                    &json!({"raw_sha256":sha}),
                    1,
                )?;
            }
        }
        for (namespace, kind) in [
            ("runtime-bundles", "runtime-receipt"),
            ("experiment-preparations", "preparation-receipt"),
        ] {
            let Some(directory) = nested(root, &[namespace, &project.project])? else {
                continue;
            };
            for leaf in directory
                .entry_names_limited(8193)
                .map_err(|_| Error::Evidence)?
            {
                let name = leaf.to_str().ok_or(Error::Evidence)?;
                let Some(id) = name.strip_suffix(".json") else {
                    continue;
                };
                if name.starts_with('.') {
                    continue;
                }
                ml_exp_core::identity::validate_observation_id(id).map_err(|_| Error::Evidence)?;
                let (value, sha) = mapping(&directory, name)?.ok_or(Error::Evidence)?;
                budget(&value, &mut bytes)?;
                // Some historical durable records wrap the document in value.
                let value = value
                    .get("value")
                    .filter(|v| v.is_object())
                    .unwrap_or(&value);
                let id_key = if namespace == "runtime-bundles" {
                    "runtime_id"
                } else {
                    "preparation_id"
                };
                if value["project"]
                    .as_str()
                    .is_some_and(|v| v != project.project)
                    || value[id_key].as_str().is_some_and(|v| v != id)
                {
                    audit.issue(
                        key(&project.project, None, None, kind, Some(id)),
                        "CATALOGUE_IDENTITY_CONFLICT",
                    )?;
                }
                audit.push(
                    key(&project.project, None, None, kind, Some(id)),
                    &json!({"raw_sha256":sha}),
                    1,
                )?;
            }
        }
    }
    Ok(())
}
fn budget(value: &Value, bytes: &mut usize) -> Result<()> {
    *bytes = bytes
        .checked_add(
            serde_json::to_vec(value)
                .map_err(|_| Error::Evidence)?
                .len(),
        )
        .filter(|v| *v <= MAX_CONTROL_BYTES)
        .ok_or(Error::Evidence)?;
    Ok(())
}
