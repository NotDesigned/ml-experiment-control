use super::*;
use ml_exp_core::records::{Event, normalized_metric};
use serde_json::Value;
use std::{fs, os::unix::fs::PermissionsExt};

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
            &json!({"ml_expd":{"run_id":"train"}}),
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
    let cycle =
        |sdk: &mut Option<Sdk>| publish_bound(&store, &runtime, sdk, &scope, &flag, &|| Ok(()));
    let read_cloud = || serde_json::from_slice::<Value>(&fs::read(&cloud).unwrap()).unwrap();
    cycle(&mut sdk).unwrap();
    assert_eq!(
        store.scope_status(&guard, &scope).unwrap().unwrap()["confirmed"],
        0
    );
    assert_eq!(read_cloud()["logs"], 4);
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
