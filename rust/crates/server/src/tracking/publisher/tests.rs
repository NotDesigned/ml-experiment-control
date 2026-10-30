use super::*;
use ml_exp_core::records::{Event, normalized_metric};
use serde_json::Value;
use std::{fs, os::unix::fs::PermissionsExt};

#[test]
fn uncertain_history_is_observed_without_opening_or_recreating_remote_runs() {
    let root = tempfile::tempdir().unwrap();
    let cloud = root.path().join("cloud.json");
    fs::write(
        &cloud,
        r#"{"hidden":false,"runs":{},"inits":0,"logs":0,"finishes":0}"#,
    )
    .unwrap();
    fs::write(root.path().join("fake_sdk.py"), include_str!("fake_sdk.py")).unwrap();
    let wrapper = root.path().join("python");
    fs::write(&wrapper,format!("#!/usr/bin/python3\nimport sys,os\nsys.path.insert(0,{root:?})\nos.environ['TEST_CLOUD']={cloud:?}\nimport fake_sdk\nsys.modules['wandb']=fake_sdk\nexec(compile(sys.argv[4],'<sdk>','exec'),{{'__name__':'__main__'}})\n",root=root.path().to_str().unwrap(),cloud=cloud.to_str().unwrap())).unwrap();
    fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o700)).unwrap();
    let mut sdk = Sdk::open(&wrapper).unwrap();
    let error = sdk
        .call(
            &json!({"operation":"observe","api_key":"never-print",
        "route":{"entity":"fixture","project":"science"},"wandb_id":"missing",
        "scope":{"project":"demo","run":"train","attempt":"attempt-001"},
        "config":{},"confirmed":0,"through":128,"open_session":true,"resume":"must"}),
            &AtomicBool::new(false),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::SdkRequest("WANDB_REMOTE_RUN_MISSING")
    ));
    assert!(!error.to_string().contains("sensitive"));
    let state: Value = serde_json::from_slice(&fs::read(&cloud).unwrap()).unwrap();
    assert_eq!(state["inits"], 0);
    assert_eq!(state["logs"], 0);
}

#[test]
fn historical_terminal_uses_original_outcomes_without_claiming_a_running_run_completed() {
    for value in [
        json!({"state":"CANCELLED"}),
        json!({"phase":"RESULTS_UPLOADED","exit_code":0}),
        json!({"phase":"RESULTS_UPLOAD_FAILED"}),
    ] {
        assert!(historical_terminal(&value));
    }
    assert!(!historical_terminal(&json!({"state":"RUNNING"})));
    assert!(!historical_terminal(&json!({})));
}

#[test]
fn delayed_remote_ack_restart_variants_and_terminal_finish_preserve_exact_cursors() {
    let root = tempfile::tempdir().unwrap();
    let cloud = root.path().join("cloud.json");
    fs::write(
        &cloud,
        r#"{"hidden":true,"runs":{},"inits":0,"logs":0,"finishes":0}"#,
    )
    .unwrap();
    let module = root.path().join("fake_sdk.py");
    fs::write(&module, include_str!("fake_sdk.py")).unwrap();
    let wrapper = root.path().join("python");
    fs::write(&wrapper,format!("#!/usr/bin/python3\nimport sys,os\nsys.path.insert(0,{root:?})\nos.environ['TEST_CLOUD']={cloud:?}\nimport fake_sdk\nsys.modules['wandb']=fake_sdk\nexec(compile(sys.argv[4],'<official-sdk-effects>','exec'),{{'__name__':'__main__'}})\n",root=root.path().to_str().unwrap(),cloud=cloud.to_str().unwrap())).unwrap();
    fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o700)).unwrap();
    let runtime = Runtime {
        config: super::super::Config {
            schema_version: 1,
            python: wrapper,
            api_key_file: root.path().join("unused-key"),
            entity: Some("fixture".into()),
            project: None,
            enabled: true,
            historical_handoff: None,
        },
        key: "test-key-never-stored".into(),
        routing: Default::default(),
        diagnostic: None,
    };
    let store = Store::open(root.path(), "0123456789abcdef0123456789abcdef").unwrap();
    let guard = store.guard("demo", "train").unwrap();
    store
        .bind(
            &guard,
            &json!({"enabled":true,"entity":"fixture","project":"science"}),
            &json!({"ml_expd":{"run_id":"train","parameters":{"recipe":{"true_path_gamma":1.0,"fast_weight":0.0},"weights":[2.0,3],"upstream_result_val":3.4976104655686546},"protocol":{"definitions":{}}}}),
        )
        .unwrap();
    let scope = store.scope(&guard, "attempt-001").unwrap();
    let metric = |id: &str, step: u64, value: f64, variant: &str| {
        Event{event_id:id.into(),payload:normalized_metric(&json!({"name":"loss","unit":"nats/token","step":step,"value":value,"variant_id":variant}),&json!({})).unwrap()}
    };
    store
        .append(
            &guard,
            &scope,
            &[
                metric("m1", 1, 3.0, "a"),
                metric("m2", 2, 2.0, "a"),
                metric("m3", 1, 5.0, "b"),
                metric("m4", 2, 4.0, "b"),
            ],
        )
        .unwrap();
    let flag = AtomicBool::new(false);
    let mut sdk = None;
    let cycle = |sdk: &mut Option<Sdk>| {
        publish_bound(&store, &runtime, sdk, &scope, &flag, false, &|| Ok(()))
    };
    let read_cloud = || serde_json::from_slice::<Value>(&fs::read(&cloud).unwrap()).unwrap();
    cycle(&mut sdk).unwrap();
    assert_eq!(
        store.scope_status(&guard, &scope).unwrap().unwrap()["confirmed"],
        0
    );
    assert_eq!(read_cloud()["logs"], 4);
    let other = store.guard("demo", "missing").unwrap();
    store
        .bind(
            &other,
            &json!({"enabled":true,"entity":"fixture","project":"science"}),
            &json!({}),
        )
        .unwrap();
    let other_scope = store.scope(&other, "attempt-001").unwrap();
    let error = sdk
        .as_mut()
        .unwrap()
        .call(
            &json!({"operation":"observe","api_key":"test-key-never-stored",
        "route":{"entity":"fixture","project":"science"},"wandb_id":"absent",
        "scope":{"project":"demo","run":"missing","attempt":"attempt-001"},
        "config":{},"confirmed":0,"through":0,"open_session":true,"resume":"must"}),
            &flag,
        )
        .unwrap_err();
    record_failure(&store, &other_scope, &mut sdk, &error);
    assert!(
        sdk.is_some(),
        "a missing remote Run must not kill another Run's buffered SDK session"
    );
    assert_eq!(
        store.scope_status(&other, &other_scope).unwrap().unwrap()["error"],
        "WANDB_REMOTE_RUN_MISSING"
    );
    cycle(&mut sdk).unwrap();
    assert_eq!(
        store.scope_status(&guard, &scope).unwrap().unwrap()["status"],
        "ACK_PENDING"
    );
    assert_eq!(read_cloud()["logs"], 4);
    sdk = None;
    cycle(&mut sdk).unwrap();
    assert_eq!(
        read_cloud()["logs"],
        4,
        "a lost SDK process cannot replay reserved rows"
    );
    let mut remote = read_cloud();
    remote["hidden"] = json!(false);
    fs::write(&cloud, remote.to_string()).unwrap();
    cycle(&mut sdk).unwrap();
    assert_eq!(
        store.scope_status(&guard, &scope).unwrap().unwrap()["confirmed"],
        4
    );
    cycle(&mut sdk).unwrap();
    assert_eq!(
        store.scope_status(&guard, &scope).unwrap().unwrap()["status"],
        "SYNCED"
    );
    assert_eq!(
        read_cloud()["finishes"],
        0,
        "a synced running run stays open"
    );
    let id = store.scope_status(&guard, &scope).unwrap().unwrap()["wandb_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let remote = read_cloud();
    assert_eq!(
        remote["runs"][&id]["config"]["ml_expd"]["parameters"]["upstream_result_val"],
        json!(3.497610465568654)
    );
    let binding = store.binding(&guard).unwrap().unwrap();
    assert_eq!(
        remote["runs"][&id]["config"]["ml_expd_identity_json"],
        publication::config_identity(
            &binding["config"]["ml_expd"],
            &binding["route"],
            &scope,
            &id
        )
        .unwrap()
    );
    let rows = remote["runs"][&id]["history"].as_array().unwrap();
    let curve_keys: Vec<_> = rows
        .iter()
        .flat_map(|r| {
            r.as_object()
                .unwrap()
                .keys()
                .filter(|k| k.starts_with("curves/") && k.contains("/variants/"))
        })
        .collect();
    assert!(curve_keys.len() >= 4, "variant curves remain distinct");
    assert!(
        !remote["runs"][&id]["summary"]
            .as_object()
            .unwrap()
            .keys()
            .any(|k| k.starts_with("results/")),
        "active losses are curves, not final comparison values"
    );
    store.append(&guard,&scope,&[Event{event_id:"terminal".into(),payload:json!({"kind":"lifecycle","data":{"phase":"METRICS_DRAINED","state":"SUCCEEDED"}})}]).unwrap();
    for _ in 0..4 {
        cycle(&mut sdk).unwrap();
    }
    let status = store.scope_status(&guard, &scope).unwrap().unwrap();
    assert_eq!(status["confirmed"], 5);
    assert_eq!(status["status"], "FINISHED");
    assert_eq!(read_cloud()["finishes"], 1);
    assert_eq!(
        read_cloud()["inits"],
        2,
        "finish verification must not reopen the run"
    );
    let remote = read_cloud();
    let finals = remote["runs"][&id]["summary"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| k.starts_with("results/"))
        .collect::<Vec<_>>();
    assert_eq!(finals.len(), 2);
    assert!(
        finals.iter().any(|(_, v)| **v == json!(2.0))
            && finals.iter().any(|(_, v)| **v == json!(4.0))
    );
    assert!(
        !fs::read(store.path())
            .unwrap()
            .windows(b"test-key-never-stored".len())
            .any(|w| w == b"test-key-never-stored")
    );
    assert!(
        !fs::read_to_string(&cloud)
            .unwrap()
            .contains("test-key-never-stored")
    );
    drop(sdk);
}
