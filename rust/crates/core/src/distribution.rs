//! A release is selected by exact version and platform, never by a mutable
//! filename or unchecked download response.
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub version: String,
    pub protocol_version: u32,
    pub assets: Vec<Asset>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub name: String,
    pub target: String,
    pub executable: String,
    pub bytes: u64,
    pub sha256: String,
}
pub fn version(s: &str) -> bool {
    let parts: Vec<_> = s.split('.').collect();
    parts.len() == 3
        && parts.iter().all(|p| {
            !p.is_empty()
                && p.len() <= 6
                && p.bytes().all(|b| b.is_ascii_digit())
                && (p.len() == 1 || !p.starts_with('0'))
        })
}

fn version_key(value: &str) -> Result<[u32; 3], &'static str> {
    if !version(value) {
        return Err("invalid native version");
    }
    let mut key = [0; 3];
    for (index, part) in value.split('.').enumerate() {
        key[index] = part.parse().map_err(|_| "invalid native version")?;
    }
    Ok(key)
}

/// GitHub's latest release may belong to the former Python distribution.
/// Select stable native tags numerically from a bounded, complete listing.
pub fn latest_native(releases: &[serde_json::Value]) -> Result<String, &'static str> {
    if releases.len() > 1000 {
        return Err("native release listing exceeds limit");
    }
    let mut latest: Option<([u32; 3], String)> = None;
    for release in releases {
        let draft = release["draft"]
            .as_bool()
            .ok_or("invalid release listing")?;
        let prerelease = release["prerelease"]
            .as_bool()
            .ok_or("invalid release listing")?;
        let tag = release["tag_name"]
            .as_str()
            .ok_or("invalid release listing")?;
        if draft || prerelease {
            continue;
        }
        let Some(candidate) = tag.strip_prefix("native-v") else {
            continue;
        };
        if !version(candidate) {
            continue;
        }
        let key = version_key(candidate)?;
        if latest.as_ref().is_none_or(|(old, _)| key > *old) {
            latest = Some((key, candidate.into()));
        }
    }
    latest
        .map(|(_, version)| version)
        .ok_or("no stable native release found")
}

pub fn prevent_default_downgrade(installed: &str, selected: &str) -> Result<(), &'static str> {
    if version_key(selected)? < version_key(installed)? {
        return Err(
            "latest stable native release is older than this client; refusing automatic downgrade; select --version explicitly if intended",
        );
    }
    Ok(())
}

#[cfg(test)]
mod latest_tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn selects_numeric_native_stable_tags_and_refuses_incomplete_or_older_choices() {
        let row =
            |tag, draft, prerelease| json!({"tag_name":tag,"draft":draft,"prerelease":prerelease});
        let rows = vec![
            row("v1.5.1", false, false),
            row("native-v0.1.9", false, false),
            row("native-v0.1.10", false, false),
            row("native-v9.0.0", false, true),
            row("native-v8.0.0", true, false),
            row("native-v10.0.0-rc1", false, false),
        ];
        assert_eq!(latest_native(&rows).unwrap(), "0.1.10");
        assert!(latest_native(&[json!({"tag_name":"native-v0.1.10"})]).is_err());
        assert!(latest_native(&[]).is_err());
        assert!(prevent_default_downgrade("0.1.10", "0.1.9").is_err());
        assert!(prevent_default_downgrade("0.1.9", "0.1.10").is_ok());
        assert!(prevent_default_downgrade("0.1.10", "0.1.10").is_ok());
    }
}
impl Manifest {
    pub fn select(
        &self,
        requested: &str,
        target: &str,
        executable: &str,
    ) -> Result<&Asset, &'static str> {
        if self.schema_version != 1
            || self.version != requested
            || !version(&self.version)
            || self.protocol_version != crate::API_PROTOCOL_VERSION
            || self.assets.is_empty()
            || self.assets.len() > 32
        {
            return Err("release manifest identity differs");
        }
        let mut names = std::collections::BTreeSet::new();
        let mut keys = std::collections::BTreeSet::new();
        for asset in &self.assets {
            if asset.bytes == 0
                || asset.bytes > 128 * 1024 * 1024
                || crate::sources::validate_archive_digest(&asset.sha256).is_err()
                || asset.name.is_empty()
                || asset.name.len() > 200
                || !asset
                    .name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
                || !names.insert(&asset.name)
                || !keys.insert((&asset.target, &asset.executable))
            {
                return Err("invalid release asset");
            }
        }
        self.assets
            .iter()
            .find(|a| a.target == target && a.executable == executable)
            .ok_or("release has no binary for this platform")
    }
}
