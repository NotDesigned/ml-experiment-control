//! User configuration is frozen data, never a source of platform credentials.
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid user environment; platform variables are reserved")]
pub struct Error;

pub fn validate(values: &BTreeMap<String, String>) -> Result<(), Error> {
    for (key, value) in values {
        let mut chars = key.bytes();
        if !chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == b'_')
            || !chars.all(|c| c.is_ascii_alphanumeric() || c == b'_')
            || value.contains('\0')
            || key.starts_with("ML_EXPD_")
            || matches!(
                key.as_str(),
                "PROJECT_NAME"
                    | "RUN_ID"
                    | "ATTEMPT_ID"
                    | "SOURCE_ID"
                    | "OUTPUT_DIR"
                    | "STATE_DIR"
                    | "DATA_DIR"
                    | "DATASETS_JSON"
                    | "METRICS_PATH"
                    | "RESUME_DIR"
                    | "RESUME_CHECKPOINT_ID"
                    | "RESUME_STEP"
            )
        {
            return Err(Error);
        }
    }
    Ok(())
}

/// Missing fields retain the old empty-environment encoding and identity.
pub fn from_document(document: &serde_json::Value) -> Result<BTreeMap<String, String>, Error> {
    let values = document
        .get("env")
        .map(|v| serde_json::from_value(v.clone()))
        .transpose()
        .map_err(|_| Error)?
        .unwrap_or_default();
    validate(&values)?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn user_values_preserve_literal_text_and_cannot_override_platform_contracts() {
        let values = BTreeMap::from([
            ("LABEL".into(), "two words; $(literal)\n中文".into()),
            ("OMP_NUM_THREADS".into(), "4".into()),
        ]);
        assert!(validate(&values).is_ok());
        for key in [
            "DATA_DIR",
            "STATE_DIR",
            "METRICS_PATH",
            "RESUME_DIR",
            "RESUME_CHECKPOINT_ID",
            "RESUME_STEP",
            "RUN_ID",
            "ML_EXPD_API_TOKEN",
            "1bad",
            "X=Y",
        ] {
            assert!(
                validate(&BTreeMap::from([(key.into(), "value".into())])).is_err(),
                "{key}"
            );
        }
        // The platform no longer sets these names, so they are ordinary user variables.
        for key in ["INPUT_DIR", "BACKEND_JOB_ID"] {
            assert!(
                validate(&BTreeMap::from([(key.into(), "/mine".into())])).is_ok(),
                "{key}"
            );
        }
        assert!(from_document(&serde_json::json!({})).unwrap().is_empty());
        for bad in [
            serde_json::json!({"env":null}),
            serde_json::json!({"env":{"X":1}}),
            serde_json::json!({"env":{"X":"\u{0000}"}}),
        ] {
            assert!(from_document(&bad).is_err());
        }
        assert!(validate(&BTreeMap::from([("X".into(), "x".repeat(8193))])).is_ok());
        let oversized = (0..5)
            .map(|i| (format!("K{i}"), "x".repeat(8192)))
            .collect();
        assert!(validate(&oversized).is_ok());
    }
}
