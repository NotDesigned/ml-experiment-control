//! Authored schema validation and path planning only. Campaign file contents
//! must be loaded and bound by a separate shell; these plans authorize no I/O.
use super::{Error, PathEvidence, Result};
use crate::{identity, metrics::MetricSchema};
use serde_json::{Map, Number, Value, json};
use std::collections::BTreeSet;

#[derive(Clone)]
pub struct ProjectDefinition {
    value: Value,
}
impl ProjectDefinition {
    pub fn from_value(value: &Value) -> Result<Self> {
        let fields = object(value)?;
        forbid(
            fields,
            &[
                "schema_version",
                "project",
                "title",
                "run_roots",
                "controller",
                "campaigns",
                "metrics_schema",
                "base_dir",
                "authored_file",
                "daemon_run_root",
            ],
        )?;
        let version = fields
            .get("schema_version")
            .map(integer)
            .transpose()?
            .unwrap_or(json!(1));
        if version != json!(1) {
            return Err(Error::Definition);
        }
        let project = string(fields.get("project"))?;
        identity::validate_id(&project).map_err(|_| Error::Definition)?;
        let title = string(fields.get("title"))?;
        let run_roots = strings(fields.get("run_roots"), None)?;
        let controller = optional(fields.get("controller"), controller)?;
        let mut campaigns = Vec::new();
        let mut names = BTreeSet::new();
        for item in array_default(fields.get("campaigns"))? {
            let item = campaign(item)?;
            if !names.insert(item["name"].as_str().expect("validated name").to_owned()) {
                return Err(Error::Definition);
            }
            campaigns.push(item);
        }
        let metrics_schema = optional(fields.get("metrics_schema"), |value| {
            MetricSchema::from_value(value)
                .map(|v| v.to_value())
                .map_err(|_| Error::Definition)
        })?;
        let base_dir = optional(fields.get("base_dir"), |v| {
            Ok(json!(path_string(&string(Some(v))?)))
        })?;
        for key in ["authored_file", "daemon_run_root"] {
            optional(fields.get(key), |v| {
                Ok(json!(path_string(&string(Some(v))?)))
            })?;
        }
        Ok(Self {
            value: json!({"schema_version":1,"project":project,"title":title,"run_roots":run_roots,"controller":controller,"campaigns":campaigns,"metrics_schema":metrics_schema,"base_dir":base_dir}),
        })
    }
    pub fn to_value(&self) -> Value {
        self.value.clone()
    }
    pub fn project(&self) -> &str {
        self.value["project"].as_str().expect("validated project")
    }
    pub fn title(&self) -> &str {
        self.value["title"].as_str().expect("validated title")
    }
    pub fn run_roots(&self) -> impl Iterator<Item = &str> {
        self.value["run_roots"]
            .as_array()
            .expect("validated roots")
            .iter()
            .map(|v| v.as_str().expect("validated string"))
    }
    /// These references still need independent file loading. A caller must not
    /// label an authored `current_revision` as observed campaign-file evidence.
    pub fn campaign_files(&self) -> impl Iterator<Item = &str> {
        self.value["campaigns"]
            .as_array()
            .expect("validated campaigns")
            .iter()
            .filter_map(|c| c["file"].as_str().filter(|s| !s.is_empty()))
    }
}
fn object(value: &Value) -> Result<&Map<String, Value>> {
    value.as_object().ok_or(Error::Definition)
}
fn forbid(fields: &Map<String, Value>, allowed: &[&str]) -> Result<()> {
    if fields.keys().any(|k| !allowed.contains(&k.as_str())) {
        Err(Error::Definition)
    } else {
        Ok(())
    }
}
fn string(value: Option<&Value>) -> Result<String> {
    value
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or(Error::Definition)
}
fn optional(
    value: Option<&Value>,
    normalizer: impl FnOnce(&Value) -> Result<Value>,
) -> Result<Value> {
    match value {
        None | Some(Value::Null) => Ok(Value::Null),
        Some(v) => normalizer(v),
    }
}
fn array_default(value: Option<&Value>) -> Result<&[Value]> {
    match value {
        None => Ok(&[]),
        Some(Value::Array(v)) => Ok(v),
        _ => Err(Error::Definition),
    }
}
fn strings(value: Option<&Value>, limit: Option<usize>) -> Result<Vec<String>> {
    let values = value.and_then(Value::as_array).ok_or(Error::Definition)?;
    if limit.is_some_and(|limit| values.len() > limit) {
        return Err(Error::Definition);
    }
    values.iter().map(|v| string(Some(v))).collect()
}
fn bool_value(value: &Value) -> Result<Value> {
    let result = match value {
        Value::Bool(v) => *v,
        Value::Number(v) => match v.as_f64() {
            Some(0.0) => false,
            Some(1.0) => true,
            _ => return Err(Error::Definition),
        },
        Value::String(v) => match v.to_ascii_lowercase().as_str() {
            "0" | "off" | "f" | "false" | "n" | "no" => false,
            "1" | "on" | "t" | "true" | "y" | "yes" => true,
            _ => return Err(Error::Definition),
        },
        _ => return Err(Error::Definition),
    };
    Ok(json!(result))
}
fn integer(value: &Value) -> Result<Value> {
    let raw = match value {
        Value::Bool(v) => return Ok(json!(u8::from(*v))),
        Value::Number(number) => {
            let raw = number.to_string();
            if !raw.contains(['.', 'e', 'E']) {
                return Ok(if raw == "-0" { json!(0) } else { value.clone() });
            }
            let float = number.as_f64().ok_or(Error::Definition)?;
            if !float.is_finite()
                || float.fract() != 0.0
                || float <= i64::MIN as f64
                || float >= 9_223_372_036_854_775_808.0
            {
                return Err(Error::Definition);
            }
            return Ok(json!(float as i64));
        }
        Value::String(v) => v.trim(),
        _ => return Err(Error::Definition),
    };
    let (negative, digits) = match raw.as_bytes().first() {
        Some(b'-') => (true, &raw[1..]),
        Some(b'+') => (false, &raw[1..]),
        _ => (false, raw),
    };
    let mut sections = digits.split('.');
    let whole = sections.next().unwrap_or("");
    let fraction = sections.next();
    if sections.next().is_some()
        || whole.is_empty()
        || !whole.as_bytes().first().is_some_and(u8::is_ascii_digit)
        || !whole.as_bytes().last().is_some_and(u8::is_ascii_digit)
        || whole.as_bytes().windows(2).any(|pair| pair == b"__")
        || whole.bytes().any(|b| !b.is_ascii_digit() && b != b'_')
        || fraction.is_some_and(|s| s.is_empty() || s.bytes().any(|b| b != b'0'))
    {
        return Err(Error::Definition);
    }
    let digits = whole.replace('_', "");
    let significant = digits.trim_start_matches('0');
    let significant = if significant.is_empty() {
        "0"
    } else {
        significant
    };
    if significant.len() > 4300 {
        return Err(Error::Definition);
    }
    let normalized = if negative && significant != "0" {
        format!("-{significant}")
    } else {
        significant.into()
    };
    Ok(Value::Number(
        normalized
            .parse::<Number>()
            .map_err(|_| Error::Definition)?,
    ))
}
fn string_map(value: Option<&Value>) -> Result<Value> {
    let mut out = Map::new();
    if let Some(value) = value {
        for (k, v) in object(value)? {
            out.insert(k.clone(), json!(string(Some(v))?));
        }
    }
    Ok(Value::Object(out))
}
fn controller(value: &Value) -> Result<Value> {
    let fields = object(value)?;
    let mut capabilities = Map::new();
    if let Some(raw) = fields.get("capabilities") {
        for (k, v) in object(raw)? {
            capabilities.insert(k.clone(), bool_value(v)?);
        }
    }
    Ok(
        json!({"python":string(fields.get("python"))?,"experimentctl":string(fields.get("experimentctl"))?,"workdir":match fields.get("workdir"){None=>".".into(),v=>string(v)?},"capabilities":capabilities,"execution_bundle":optional(fields.get("execution_bundle"),bundle)?}),
    )
}
fn bundle(value: &Value) -> Result<Value> {
    let fields = object(value)?;
    forbid(
        fields,
        &[
            "entry_module",
            "paths",
            "python_packages",
            "require_clean_git",
        ],
    )?;
    Ok(
        json!({"entry_module":string(fields.get("entry_module"))?,"paths":match fields.get("paths"){None=>vec![],v=>strings(v,Some(64))?},"python_packages":match fields.get("python_packages"){None=>vec![],v=>strings(v,Some(32))?},"require_clean_git":fields.get("require_clean_git").map(bool_value).transpose()?.unwrap_or(json!(true))}),
    )
}
fn campaign(value: &Value) -> Result<Value> {
    let fields = object(value)?;
    let name = string(fields.get("name"))?;
    identity::validate_id(&name).map_err(|_| Error::Definition)?;
    Ok(
        json!({"name":name,"file":optional(fields.get("file"),|v|Ok(json!(string(Some(v))?)))?,"role_notes":string_map(fields.get("role_notes"))?,"current_revision":optional(fields.get("current_revision"),revision)?}),
    )
}
fn revision(value: &Value) -> Result<Value> {
    let fields = object(value)?;
    let mut memberships = Vec::new();
    for item in array_default(fields.get("memberships"))? {
        memberships.push(membership(item)?);
    }
    Ok(
        json!({"campaign":string(fields.get("campaign"))?,"project":string(fields.get("project"))?,"revision_id":string(fields.get("revision_id"))?,"file":string(fields.get("file"))?,"research_contract":optional(fields.get("research_contract"),|v|{object(v)?;Ok(v.clone())})?,"memberships":memberships,"source_bindings":string_map(fields.get("source_bindings"))?}),
    )
}
fn membership(value: &Value) -> Result<Value> {
    let fields = object(value)?;
    let run_id = string(fields.get("run_id"))?;
    identity::validate_id(&run_id).map_err(|_| Error::Definition)?;
    let kind = match fields.get("kind") {
        None => "materialize".into(),
        v => string(v)?,
    };
    if kind != "materialize" && kind != "reuse" {
        return Err(Error::Definition);
    }
    let mut result = json!({"run_id":run_id,"kind":kind,"replicate":optional(fields.get("replicate"),integer)?,"included_in_analysis":fields.get("included_in_analysis").map(bool_value).transpose()?.unwrap_or(json!(true))});
    for key in ["role", "arm", "purpose"] {
        result[key] = optional(fields.get(key), |v| Ok(json!(string(Some(v))?)))?;
    }
    Ok(result)
}
/// pathlib's lexical normalization, without home expansion, symlink resolution
/// or filesystem access. `..` remains a path component; it is not authorization.
fn path_string(value: &str) -> String {
    let prefix = if value.starts_with("//") && !value.starts_with("///") {
        "//"
    } else if value.starts_with('/') {
        "/"
    } else {
        ""
    };
    let parts = value
        .split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .collect::<Vec<_>>();
    let joined = parts.join("/");
    if joined.is_empty() && prefix.is_empty() {
        ".".into()
    } else {
        format!("{prefix}{joined}")
    }
}

pub struct ProjectPaths {
    pub base_dir: String,
    pub authored_file: String,
    pub run_roots: Vec<String>,
}
pub struct ProjectLayout {
    pub base_dir: String,
    pub authored_file: String,
    pub run_roots: Vec<String>,
}
/// Return unresolved candidates. Callers resolve and approve each before using
/// them; neither authored base_dir nor daemon_run_root confers authority.
pub fn project_paths(
    definition: &ProjectDefinition,
    manifest: &PathEvidence,
) -> Result<ProjectPaths> {
    manifest.validate()?;
    let (parent, _) = manifest
        .canonical_absolute
        .rsplit_once('/')
        .ok_or(Error::Path)?;
    let parent = if parent.is_empty() { "/" } else { parent };
    let base = if parent.rsplit('/').next() == Some("experiments") {
        parent
            .rsplit_once('/')
            .map(|(p, _)| if p.is_empty() { "/" } else { p })
            .unwrap_or("/")
    } else {
        parent
    };
    let roots = definition
        .run_roots()
        .map(|root| {
            if root.starts_with('/') {
                path_string(root)
            } else {
                path_string(&format!("{base}/{root}"))
            }
        })
        .collect();
    Ok(ProjectPaths {
        base_dir: base.into(),
        authored_file: manifest.canonical_absolute.clone(),
        run_roots: roots,
    })
}
pub fn project_layout(
    definition: &ProjectDefinition,
    manifest: &PathEvidence,
    resolved_run_roots: &[PathEvidence],
    daemon_run_root: Option<&PathEvidence>,
) -> Result<ProjectLayout> {
    let paths = project_paths(definition, manifest)?;
    let mut roots = Vec::new();
    let mut seen = BTreeSet::new();
    for (requested, authored) in paths.run_roots.iter().zip(definition.run_roots()) {
        let matching = resolved_run_roots
            .iter()
            .filter(|p| &p.provided == requested)
            .collect::<Vec<_>>();
        let first = matching.first().ok_or(Error::PathEvidence)?;
        first.validate()?;
        if matching.iter().any(|path| *path != *first) {
            return Err(Error::PathEvidence);
        }
        // Absolute authored paths are deliberately not resolved by Python's
        // resolved_run_roots; preserve their lexical form, while still requiring
        // explicit shell approval before any caller performs filesystem I/O.
        let path = if authored.starts_with('/') {
            requested.clone()
        } else {
            first.canonical_absolute.clone()
        };
        if seen.insert(path.clone()) {
            roots.push(path);
        }
    }
    if let Some(path) = daemon_run_root {
        path.validate()?;
        if seen.insert(path.canonical_absolute.clone()) {
            roots.push(path.canonical_absolute.clone());
        }
    }
    Ok(ProjectLayout {
        base_dir: paths.base_dir,
        authored_file: paths.authored_file,
        run_roots: roots,
    })
}
