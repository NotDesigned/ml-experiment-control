//! Provider name constraints and the exact issued recipe are part of the
//! ownership boundary, including after an uncertain creation response.
use base64::Engine;
use ml_exp_core::{
    canonical,
    cloud::debug::{Create, Profile, Recipe},
};
use serde_json::json;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

fn profile() -> Profile {
    serde_json::from_value(json!({"provider":"pai_dsw","image_id":"image-approved","access":{"kind":"intranet"},
        "profile":{"region":"cn-hangzhou","workspace_id":"workspace","quota_id":"quota",
        "max_gpus":0,"max_cpus":4,"max_memory_gb":8,
        "storage":{"cpfs_uri":"bmcpfs://cpfs-example.cn-hangzhou.cpfs.aliyuncs.com/","workspace_user":"researcher","role_arn":"acs:ram::1234567890123456:role/pai-data-researcher"},
        "vpc":{"VpcId":"vpc-approved","VSwitchId":"vsw-approved","SecurityGroupId":"sg-approved"}}})).unwrap()
}
fn request() -> Create {
    let mut wire = Vec::new();
    wire.extend(11u32.to_be_bytes());
    wire.extend(b"ssh-ed25519");
    wire.extend(32u32.to_be_bytes());
    wire.extend([1u8; 32]);
    Create {
        profile: "cpu-debug".into(),
        request_id: "a".repeat(32),
        ssh_public_key: format!(
            "ssh-ed25519 {}",
            base64::engine::general_purpose::STANDARD.encode(wire)
        ),
        lease_seconds: 600,
        gpus: 0,
        cpus: 4,
        memory_gb: 8,
    }
}
const NOW: &str = "2026-10-29T00:00:00Z";

#[test]
fn dsw_name_meets_actual_api_constraints_and_full_ownership_is_still_required() {
    let profile = profile();
    let recipe = profile.recipe(request(), NOW).unwrap();
    assert_eq!(recipe.contract, "native-debug-machine.v4");
    assert!(recipe.name.len() <= 27);
    assert!(
        recipe
            .name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_')
    );
    let body = profile.create(&recipe).unwrap().body.unwrap();
    assert_eq!(body["InstanceName"], recipe.name);
    assert_eq!(
        profile.stop(&recipe, "dsw-owned", false).unwrap().body,
        Some(json!({}))
    );
    assert_eq!(body["WorkspaceSource"], "/mnt/ml-expd-personal");
    assert!(
        body["Datasets"]
            .as_array()
            .unwrap()
            .iter()
            .any(|mount| mount["MountPath"] == body["WorkspaceSource"]
                && mount["MountAccess"] == "RW")
    );
    assert!(
        body["Datasets"]
            .as_array()
            .unwrap()
            .iter()
            .all(|mount| mount["MountPath"] != "/mnt/workspace")
    );
    assert_eq!(
        profile.get(&recipe, None).unwrap().query["InstanceName"],
        recipe.name
    );
    let mut row = body.clone();
    row["InstanceId"] = json!("dsw-owned");
    assert_eq!(
        profile.checked(&recipe, Some("dsw-owned"), &row).unwrap(),
        "dsw-owned"
    );
    row["Labels"] = json!([]);
    assert!(profile.checked(&recipe, None, &row).is_err());
    row = body.clone();
    row["InstanceId"] = json!("dsw-other");
    assert!(profile.checked(&recipe, Some("dsw-owned"), &row).is_err());
    row["InstanceId"] = json!("dsw-owned");
    row["ResourceId"] = json!("foreign-quota");
    assert!(profile.checked(&recipe, Some("dsw-owned"), &row).is_err());
}

#[test]
fn dsw_workspace_binding_accepts_the_provider_integer_and_nothing_else() {
    let mut profile = profile();
    let Profile::PaiDsw { profile: dsw, .. } = &mut profile;
    dsw.workspace_id = "253".into();
    let recipe = profile.recipe(request(), NOW).unwrap();
    let body = profile.create(&recipe).unwrap().body.unwrap();
    assert_eq!(body["WorkspaceSource"], "/mnt/ml-expd-personal");
    let mut row = body.clone();
    row["InstanceId"] = json!("dsw-owned");
    row["WorkspaceId"] = json!(253);
    assert_eq!(
        profile.checked(&recipe, Some("dsw-owned"), &row).unwrap(),
        "dsw-owned"
    );
    for workspace in [
        json!(254),
        json!(253.0),
        json!("0253"),
        json!(null),
        json!(false),
    ] {
        row["WorkspaceId"] = workspace;
        assert!(profile.checked(&recipe, Some("dsw-owned"), &row).is_err());
    }
}

#[test]
fn a_changed_name_contract_or_identity_is_refused() {
    let profile = profile();
    let recipe = profile.recipe(request(), NOW).unwrap();
    profile.check_recipe(&recipe).unwrap();
    assert_eq!(
        profile.get(&recipe, None).unwrap().query["InstanceName"],
        recipe.name
    );
    let tampered = |change: fn(&mut Recipe)| {
        let mut recipe = recipe.clone();
        change(&mut recipe);
        recipe
    };
    for recipe in [
        tampered(|r| r.name = "ml_exp_different".into()),
        tampered(|r| r.contract = "native-debug-machine.v5".into()),
        tampered(|r| r.identity = format!("debug.{}", "0".repeat(64))),
        tampered(|r| r.expires_at += 1),
    ] {
        assert!(profile.check_recipe(&recipe).is_err());
        assert!(profile.create(&recipe).is_err());
    }
}

#[test]
fn dsw_lifecycle_path_and_reason_match_the_current_state() {
    let profile = profile();
    let recipe = profile.recipe(request(), NOW).unwrap();
    let request = profile.lifecycle(&recipe, "dsw-owned").unwrap();
    assert_eq!(request.path, "/api/v2/instances/dsw-owned/lifecycle");
    assert_eq!(request.query["Limit"], "1");
    let rows = json!({"Lifecycle":[[{"Status":"Queuing","ReasonMessage":"quota wait"},{"Status":"Failed","ReasonMessage":"mountPath must be unique"}]]});
    assert_eq!(
        ml_exp_core::cloud::debug::lifecycle_reason(&rows, "Failed").unwrap(),
        Some("mountPath must be unique")
    );
    assert_eq!(
        ml_exp_core::cloud::debug::lifecycle_reason(&rows, "Running").unwrap(),
        None
    );
    assert!(
        ml_exp_core::cloud::debug::lifecycle_reason(&json!({"Lifecycle":[{}]}), "Failed").is_err()
    );
}

/// A self-consistent recipe under another contract, with the name that contract
/// would use, is refused even though its digest is correct.
fn forged(profile: &Profile, request: Create, contract: &str, name: fn(&str) -> String) -> Recipe {
    let profile_sha256 = profile.digest().unwrap();
    let digest = canonical::digest(&json!([contract, profile_sha256, request])).unwrap();
    Recipe {
        contract: contract.into(),
        identity: format!("debug.{digest}"),
        name: name(&digest),
        profile_sha256,
        expires_at: OffsetDateTime::parse(NOW, &Rfc3339)
            .unwrap()
            .unix_timestamp()
            + request.lease_seconds as i64,
        request,
        created_at: NOW.into(),
    }
}
fn dsw_name(digest: &str) -> String {
    format!("ml_exp_{}", &digest[..20])
}
/// The name format of the retired SenseCore contract `native-debug-machine.v1`.
fn retired_name(digest: &str) -> String {
    format!("ml-expd-debug-{}", &digest[..32])
}

#[test]
fn a_dsw_profile_accepts_only_its_own_contract() {
    type Name = fn(&str) -> String;
    let dsw = profile();
    let genuine = forged(&dsw, request(), "native-debug-machine.v4", dsw_name);
    assert!(genuine == dsw.recipe(request(), NOW).unwrap());
    dsw.check_recipe(&genuine).unwrap();
    let refused: [(&str, Name); 4] = [
        ("native-debug-machine.v1", dsw_name),
        ("native-debug-machine.v1", retired_name),
        ("native-debug-machine.v2", dsw_name),
        ("native-debug-machine.v3", dsw_name),
    ];
    for (contract, name) in refused {
        let recipe = forged(&dsw, request(), contract, name);
        assert!(dsw.check_recipe(&recipe).is_err(), "{contract}");
        assert!(dsw.create(&recipe).is_err(), "{contract}");
    }
}
