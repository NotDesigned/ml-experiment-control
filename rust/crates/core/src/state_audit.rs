//! Pure comparison of bounded, metadata-only state observations. Equality here
//! never authorizes a provider operation, data deletion or production cutover.
use crate::{canonical, identity};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const CONTRACT: &str = "native-state-audit.v1";
pub const MAX_ENTRIES: usize = 100_000;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Key {
    pub project: String,
    pub run: Option<String>,
    pub attempt: Option<String>,
    pub kind: String,
    pub item: Option<String>,
}
impl Key {
    pub fn validate(&self) -> Result<(), &'static str> {
        identity::validate_id(&self.project).map_err(|_| "audit project")?;
        if let Some(run) = &self.run {
            identity::validate_observation_id(run).map_err(|_| "audit run")?;
        }
        if let Some(attempt) = &self.attempt {
            identity::validate_attempt(attempt).map_err(|_| "audit attempt")?;
            if self.run.is_none() {
                return Err("audit orphan Attempt");
            }
        }
        for name in std::iter::once(&self.kind).chain(self.item.iter()) {
            identity::validate_observation_id(name).map_err(|_| "audit item")?;
        }
        Ok(())
    }
}
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fingerprint {
    pub key: Key,
    pub sha256: String,
    pub records: u64,
}
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Issue {
    pub key: Key,
    pub code: String,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub contract: String,
    pub scope: String,
    pub entries: Vec<Fingerprint>,
    pub issues: Vec<Issue>,
}
fn hash(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|v| {
        v.len() == 64
            && v.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    })
}
impl Snapshot {
    pub fn new(
        mut entries: Vec<Fingerprint>,
        mut issues: Vec<Issue>,
    ) -> Result<Self, &'static str> {
        entries.sort_by(|a, b| a.key.cmp(&b.key));
        issues.sort();
        issues.dedup();
        let snapshot = Self {
            contract: CONTRACT.into(),
            scope: "FROZEN_METADATA_AND_JOURNALS".into(),
            entries,
            issues,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.contract != CONTRACT
            || self.scope != "FROZEN_METADATA_AND_JOURNALS"
            || self.entries.len() > MAX_ENTRIES
            || self.issues.len() > MAX_ENTRIES
        {
            return Err("audit contract or budget");
        }
        let mut keys = BTreeSet::new();
        for entry in &self.entries {
            entry.key.validate()?;
            if !hash(&entry.sha256) || !keys.insert(&entry.key) {
                return Err("audit duplicate or invalid fingerprint");
            }
        }
        for issue in &self.issues {
            issue.key.validate()?;
            if issue.code.is_empty()
                || issue.code.len() > 128
                || !issue
                    .code
                    .bytes()
                    .all(|c| c.is_ascii_uppercase() || c == b'_')
            {
                return Err("audit issue code");
            }
        }
        Ok(())
    }
}
pub fn fingerprint(key: Key, value: &Value, records: u64) -> Result<Fingerprint, &'static str> {
    key.validate()?;
    Ok(Fingerprint {
        key,
        sha256: canonical::protocol_identity(value).map_err(|_| "audit identity")?,
        records,
    })
}
#[derive(Serialize)]
pub struct Comparison {
    pub contract: &'static str,
    pub scope: String,
    pub equivalent: bool,
    pub missing: Vec<Key>,
    pub added: Vec<Key>,
    pub changed: Vec<Key>,
    pub baseline_issues: Vec<Issue>,
    pub candidate_issues: Vec<Issue>,
    pub provider_execution_authorized: bool,
    pub production_cutover_authorized: bool,
}
pub fn compare(baseline: &Snapshot, candidate: &Snapshot) -> Result<Comparison, &'static str> {
    baseline.validate()?;
    candidate.validate()?;
    let index = |snapshot: &Snapshot| {
        snapshot
            .entries
            .iter()
            .map(|v| (v.key.clone(), (v.sha256.clone(), v.records)))
            .collect::<BTreeMap<_, _>>()
    };
    let a = index(baseline);
    let b = index(candidate);
    let missing: Vec<_> = a.keys().filter(|k| !b.contains_key(*k)).cloned().collect();
    let added: Vec<_> = b.keys().filter(|k| !a.contains_key(*k)).cloned().collect();
    let changed: Vec<_> = a
        .iter()
        .filter(|(k, v)| b.get(*k).is_some_and(|other| other != *v))
        .map(|(k, _)| k.clone())
        .collect();
    Ok(Comparison {
        contract: CONTRACT,
        scope: baseline.scope.clone(),
        equivalent: missing.is_empty()
            && added.is_empty()
            && changed.is_empty()
            && baseline.issues.is_empty()
            && candidate.issues.is_empty(),
        missing,
        added,
        changed,
        baseline_issues: baseline.issues.clone(),
        candidate_issues: candidate.issues.clone(),
        provider_execution_authorized: false,
        production_cutover_authorized: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn entry(item: &str, value: Value) -> Fingerprint {
        fingerprint(
            Key {
                project: "demo".into(),
                run: Some("train".into()),
                attempt: Some("attempt-001".into()),
                kind: "metrics".into(),
                item: Some(item.into()),
            },
            &value,
            1,
        )
        .unwrap()
    }
    #[test]
    fn order_of_entries_is_irrelevant_but_receive_order_and_context_are_not() {
        let a = Snapshot::new(vec![entry("journal", json!([{"variant_id":"a","unit":"nats/token","step":2},{"variant_id":"b","step":1}]))], vec![]).unwrap();
        assert!(compare(&a, &a).unwrap().equivalent);
        for value in [
            json!([{"variant_id":"b","step":1},{"variant_id":"a","unit":"nats/token","step":2}]),
            json!([{"variant_id":"a","unit":"bits/byte","step":2},{"variant_id":"b","step":1}]),
        ] {
            let b = Snapshot::new(vec![entry("journal", value)], vec![]).unwrap();
            assert!(!compare(&a, &b).unwrap().equivalent);
        }
        let mut b = a.clone();
        b.issues.push(Issue {
            key: b.entries[0].key.clone(),
            code: "SOURCE_CHANGED".into(),
        });
        assert!(!compare(&b, &b).unwrap().equivalent);
        assert!(!compare(&a, &a).unwrap().production_cutover_authorized);
        b.entries.push(b.entries[0].clone());
        assert!(compare(&a, &b).is_err());
    }
    #[test]
    fn a_missing_scope_or_changed_count_never_matches() {
        let a = Snapshot::new(
            vec![entry("one", json!({})), entry("two", json!({}))],
            vec![],
        )
        .unwrap();
        let mut b = a.clone();
        b.entries.pop();
        assert_eq!(compare(&a, &b).unwrap().missing.len(), 1);
        b.entries[0].records += 1;
        assert_eq!(compare(&a, &b).unwrap().changed.len(), 1);
    }
}
