//! Provider name constraints and issued-recipe compatibility are part of the
//! ownership boundary, including after an uncertain creation response.
use base64::Engine;
use ml_exp_core::{
    canonical,
    cloud::debug::{Create, Profile, Recipe},
};
use serde_json::json;
use time::format_description::well_known::Rfc3339;

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
fn legacy_uncertain_create_is_observed_by_its_original_name_without_identity_rewrite() {
    let profile = profile();
    let request = request();
    let profile_sha256 = profile.digest().unwrap();
    let digest =
        canonical::digest(&json!(["native-debug-machine.v1", profile_sha256, request])).unwrap();
    let mut legacy = Recipe {
        contract: "native-debug-machine.v1".into(),
        identity: format!("debug.{digest}"),
        name: format!("ml-expd-debug-{}", &digest[..32]),
        profile_sha256,
        request: request.clone(),
        created_at: NOW.into(),
        expires_at: time::OffsetDateTime::parse(
            NOW,
            &time::format_description::well_known::Rfc3339,
        )
        .unwrap()
        .unix_timestamp()
            + 600,
    };
    let original = serde_json::to_vec(&legacy).unwrap();
    profile.check_recipe(&legacy).unwrap();
    assert_eq!(
        profile.get(&legacy, None).unwrap().query["InstanceName"],
        legacy.name
    );
    assert_eq!(
        profile.create(&legacy).unwrap().body.unwrap()["InstanceName"],
        legacy.name
    );
    assert_eq!(serde_json::to_vec(&legacy).unwrap(), original);
    assert_ne!(
        profile.recipe(request, NOW).unwrap().identity,
        legacy.identity
    );
    legacy.name = "ml_exp_different".into();
    assert!(profile.check_recipe(&legacy).is_err());
    legacy.contract = "native-debug-machine.v5".into();
    assert!(profile.check_recipe(&legacy).is_err());
}

#[test]
fn issued_v2_body_is_unchanged_and_workspace_integer_response_still_binds_exact_scope() {
    let mut profile = profile();
    if let Profile::PaiDsw { profile, .. } = &mut profile {
        profile.workspace_id = "253".into();
    }
    let request = request();
    let profile_sha256 = profile.digest().unwrap();
    let digest =
        canonical::digest(&json!(["native-debug-machine.v2", profile_sha256, request])).unwrap();
    let legacy = Recipe {
        contract: "native-debug-machine.v2".into(),
        identity: format!("debug.{digest}"),
        name: format!("ml_exp_{}", &digest[..20]),
        profile_sha256,
        request,
        created_at: NOW.into(),
        expires_at: time::OffsetDateTime::parse(NOW, &Rfc3339)
            .unwrap()
            .unix_timestamp()
            + 600,
    };
    profile.check_recipe(&legacy).unwrap();
    let body = profile.create(&legacy).unwrap().body.unwrap();
    assert_eq!(body["WorkspaceSource"], "/mnt/ml-expd-personal");
    let mut row = body.clone();
    row["InstanceId"] = json!("dsw-owned");
    row["WorkspaceId"] = json!(253);
    assert_eq!(
        profile.checked(&legacy, Some("dsw-owned"), &row).unwrap(),
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
        assert!(profile.checked(&legacy, Some("dsw-owned"), &row).is_err());
    }
}

#[test]
fn issued_v3_mounts_remain_frozen_and_lifecycle_reason_matches_current_state() {
    let profile = profile();
    let request = request();
    let profile_sha256 = profile.digest().unwrap();
    let digest =
        canonical::digest(&json!(["native-debug-machine.v3", profile_sha256, request])).unwrap();
    let legacy = Recipe {
        contract: "native-debug-machine.v3".into(),
        identity: format!("debug.{digest}"),
        name: format!("ml_exp_{}", &digest[..20]),
        profile_sha256,
        request,
        created_at: NOW.into(),
        expires_at: time::OffsetDateTime::parse(NOW, &Rfc3339)
            .unwrap()
            .unix_timestamp()
            + 600,
    };
    profile.check_recipe(&legacy).unwrap();
    let body = profile.create(&legacy).unwrap().body.unwrap();
    assert_eq!(body["WorkspaceSource"], "/mnt/workspace");
    assert_eq!(body["Datasets"][0]["MountPath"], "/mnt/workspace");
    let request = profile.lifecycle(&legacy, "dsw-owned").unwrap();
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
