//! An existing object is an input reference, never authority to choose a bucket,
//! endpoint or another user's storage. These checks have no network effects.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Import {
    pub object_key: String,
    pub sha256: String,
    pub archive_bytes: u64,
}
impl Import {
    pub fn validate(&self) -> Result<()> {
        crate::sources::validate_archive_digest(&self.sha256).map_err(|_| Error)?;
        multipart::part_count(self.archive_bytes, multipart::Policy::default())
            .map_err(|_| Error)?;
        if self.object_key.len() > 1024
            || !path(&self.object_key)
            || self.object_key.split('/').any(|s| {
                s.starts_with('.')
                    || !s
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
            })
        {
            return Err(Error);
        }
        Ok(())
    }
    pub fn bind(&self, approved_prefix: &str) -> Result<()> {
        self.validate()?;
        let parts = approved_prefix.split('/').collect::<Vec<_>>();
        if parts.len() < 4
            || parts[0] != "users"
            || parts[1].is_empty()
            || !approved_prefix.ends_with('/')
            || parts[..parts.len() - 1].iter().any(|s| {
                s.is_empty()
                    || matches!(*s, "." | "..")
                    || !s
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
            })
            || !(self
                .object_key
                .starts_with(&format!("users/{}/datasets/", parts[1]))
                || self.object_key.starts_with("public/datasets/"))
        {
            return Err(Error);
        }
        Ok(())
    }
    pub fn asset_id(&self) -> String {
        format!("asset.{}", self.sha256)
    }
    pub fn matches(&self, definition: &Definition) -> Result<()> {
        definition.validate()?;
        if definition.sha256 != self.sha256 || definition.archive_bytes != self.archive_bytes {
            return Err(Error);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(key: &str) -> Import {
        Import {
            object_key: key.into(),
            sha256: "a".repeat(64),
            archive_bytes: 2048,
        }
    }
    #[test]
    fn only_approved_personal_and_public_datasets_are_inputs() {
        for key in [
            "users/alice/datasets/train.tar",
            "public/datasets/fineweb/train.tar",
        ] {
            request(key).bind("users/alice/ml-expd/").unwrap();
        }
        for key in [
            "users/bob/datasets/train.tar",
            "users/alice/results/train.tar",
            "users/alice/datasets/../results/train.tar",
            "public/datasets/.secret.tar",
            "public/datasets/%2e%2e/train.tar",
            "public/datasets/credentials/a.tar",
            "public/datasets//a.tar",
            "https://example.org/data.tar",
            "public/datasets/a.tar?x=1",
        ] {
            assert!(request(key).bind("users/alice/ml-expd/").is_err(), "{key}");
        }
        assert!(
            request("public/datasets/a.tar")
                .bind("users//ml-expd/")
                .is_err()
        );
    }
    #[test]
    fn a_reference_requires_exact_nonempty_archive_identity() {
        let mut value = request("public/datasets/train.tar");
        value.archive_bytes = 0;
        assert!(value.validate().is_err());
        value.archive_bytes = 2048;
        value.sha256 = "A".repeat(64);
        assert!(value.validate().is_err());
        assert!(serde_json::from_str::<Import>(r#"{"object_key":"public/datasets/a.tar","sha256":"a","archive_bytes":1,"endpoint":"evil"}"#).is_err());
    }
}
