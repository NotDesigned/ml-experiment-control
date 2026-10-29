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
