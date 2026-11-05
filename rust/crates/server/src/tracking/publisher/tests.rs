use super::*;
use ml_exp_core::records::{Event, normalized_metric};
use serde_json::Value;
use std::{fs, os::unix::fs::PermissionsExt};

#[test]
fn explicit_gap_repair_appends_original_ids_and_normal_publication_resumes_without_duplicates() {
    for version in [3, 4] {
        exercise_gap_repair(false, 2, version);
    }
}

#[test]
fn explicit_tail_repair_confirms_appended_tail_and_continues_without_duplicates() {
    for version in [3, 4] {
        exercise_gap_repair(false, 4, version);
    }
}

#[test]
fn uncertain_repair_is_observed_without_resending_or_clearing_original_reservations() {
    for version in [3, 4] {
        exercise_gap_repair(true, 2, version);
        exercise_gap_repair(true, 4, version);
    }
}

fn exercise_gap_repair(uncertain: bool, missing_sequence: i64, version: i64) {
    struct FixtureSource(std::path::PathBuf);
    impl Source for FixtureSource {
        fn publisher_leader(&self) -> Result<ml_exp_io::directory::FileLock> {
            ml_exp_io::directory::Directory::open(&self.0, false)
                .and_then(|directory| {
                    directory.lock(
                        std::ffi::OsStr::new("leader"),
                        ml_exp_io::directory::LockOptions {
                            create: true,
                            shared: true,
                            ..Default::default()
                        },
                    )
                })
                .map_err(|_| Error::Storage)
        }
        fn publisher_store(&self) -> Result<Store> {
            Store::open(&self.0, "0123456789abcdef0123456789abcdef").map_err(|_| Error::Storage)
        }
        fn tracking_journal(&self, _: &str, _: &str) -> Result<Store> {
            self.publisher_store()
        }
    }
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
    let runtime = Runtime {
        config: super::super::Config {
            schema_version: 1,
            python: wrapper,
            api_key_file: root.path().join("unused"),
            entity: Some("fixture".into()),
            project: None,
            enabled: true,
        },
        key: "never-print".into(),
        routing: Default::default(),
        diagnostic: None,
    };
    let source = Arc::new(FixtureSource(root.path().to_owned()));
    let store = source.publisher_store().unwrap();
    let guard = store.guard("demo", "train").unwrap();
    let route = json!({"enabled":true,"entity":"fixture","project":"science"});
    store
        .bind(&guard, &route, &json!({"ml_expd":{"run_id":"train"}}))
        .unwrap();
    let scope = store.scope(&guard, "attempt-001").unwrap();
    if version == 3 {
        let saved = Saved {
            display_contract: Some(3),
            ..Default::default()
        };
        store
            .save_publication_state(&guard, &scope, &json!(saved), "PENDING", None, "fixture")
            .unwrap();
    }
    let metric = |step| Event {
        event_id: format!("m{step}"),
        payload: normalized_metric(
            &json!({"name":"loss","unit":"nats/token","step":step,"value":1.0/(step as f64)}),
            &json!({}),
        )
        .unwrap(),
    };
    store
        .append(
            &guard,
            &scope,
            &[metric(1), metric(2), metric(3), metric(4)],
        )
        .unwrap();
    let flag = AtomicBool::new(false);
    let mut sdk = None;
    publish_bound(&store, &runtime, &mut sdk, &scope, &flag).unwrap();
    let status = store.scope_status(&guard, &scope).unwrap().unwrap();
    let id = status["wandb_id"].as_str().unwrap().to_owned();
    sdk.as_mut().unwrap().shutdown().unwrap();
    let mut data: Value = serde_json::from_slice(&fs::read(&cloud).unwrap()).unwrap();
    assert_eq!(data["runs"][&id]["pending"], json!([]));
    assert_eq!(data["runs"][&id]["history"].as_array().unwrap().len(), 4);
    data["repeat_pages"] = json!(true);
    let rows = data["runs"][&id]["history"].take();
    data["runs"][&id]["history"] = json!(
        rows.as_array()
            .unwrap()
            .iter()
            .filter(|r| r["ml_expd/sequence"] != missing_sequence)
            .collect::<Vec<_>>()
    );
    fs::write(&cloud, serde_json::to_vec(&data).unwrap()).unwrap();
    drop(sdk.take());
    let original = store.publication_state(&guard, &scope).unwrap().unwrap();
    if uncertain {
        let mut reserved = original.clone();
        reserved["history_repair"] = json!({"events":{missing_sequence.to_string():format!("m{missing_sequence}")},"attempted":true,"remote_confirmed":false});
        store
            .save_publication_state(
                &guard,
                &scope,
                &reserved,
                "REPAIR_ACK_PENDING",
                None,
                "fixture",
            )
            .unwrap();
    }
    drop(guard);
    let result = repair(source.clone(), runtime.clone(), &scope).unwrap();
    if uncertain {
        assert_eq!(result["status"], "REPAIR_ACK_PENDING");
        assert_eq!(result["writes"], 0);
        let guard = store.guard("demo", "train").unwrap();
        assert_eq!(
            store.publication_state(&guard, &scope).unwrap().unwrap()["state"],
            original["state"]
        );
        let mut data: Value = serde_json::from_slice(&fs::read(&cloud).unwrap()).unwrap();
        assert_eq!(data["logs"], 4, "an uncertain repair cannot be sent again");
        // Model the delayed remote visibility of the already-attempted append.
        let mut late = rows
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["ml_expd/sequence"] == missing_sequence)
            .unwrap()
            .clone();
        late["_step"] = json!(4);
        data["runs"][&id]["history"]
            .as_array_mut()
            .unwrap()
            .push(late);
        fs::write(&cloud, serde_json::to_vec(&data).unwrap()).unwrap();
    } else {
        assert_eq!(result["status"], "REPAIRED");
        assert_eq!(result["appended"], 1);
        assert_eq!(result["confirmed"], 4);
    }
    let repeated = repair(source.clone(), runtime.clone(), &scope).unwrap();
    assert_eq!(repeated["writes"], 0);
    let guard = store.guard("demo", "train").unwrap();
    store.append(&guard, &scope, &[metric(5)]).unwrap();
    publish_bound(&store, &runtime, &mut sdk, &scope, &flag).unwrap();
    publish_bound(&store, &runtime, &mut sdk, &scope, &flag).unwrap();
    assert_eq!(
        store.scope_status(&guard, &scope).unwrap().unwrap()["confirmed"],
        5
    );
    sdk.as_mut().unwrap().shutdown().unwrap();
    let data: Value = serde_json::from_slice(&fs::read(&cloud).unwrap()).unwrap();
    let rows = data["runs"][&id]["history"].as_array().unwrap();
    assert_eq!(rows.len(), 5);
    let ids: BTreeSet<_> = rows
        .iter()
        .map(|r| r["ml_expd/event_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids.len(), 5);
    let repaired = rows
        .iter()
        .find(|r| r["ml_expd/sequence"] == missing_sequence)
        .unwrap();
    // The simulated delayed write uses physical4; an immediate tail append
    // follows the three visible rows at physical3. Science still uses its
    // original training step in either case.
    assert_eq!(
        repaired["_step"],
        if missing_sequence == 4 && !uncertain {
            3
        } else {
            4
        }
    );
    assert_eq!(
        repaired[if version == 4 {
            "ml_expd/axes/loss/step"
        } else {
            "axes/loss/step"
        }],
        missing_sequence
    );
    assert_eq!(store.events(&guard, &scope, 0, 128).unwrap().len(), 5);
}

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
fn delayed_remote_ack_restart_variants_and_terminal_finish_preserve_exact_cursors() {
    exercise_delayed_publication(4);
}

#[test]
fn already_started_legacy_display_keeps_names_and_reservations_across_restart() {
    exercise_delayed_publication(2);
}

#[test]
fn started_v3_keeps_prefixes_across_restart() {
    exercise_delayed_publication(3);
}

fn exercise_delayed_publication(version: i64) {
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
            &json!({"ml_expd":{"run_id":"train","parameters":{"recipe":{"true_path_gamma":1.0,"fast_weight":0.0},"weights":[2.0,3],"upstream_result_val":3.4976104655686546},"protocol":{"definitions":{}}}}),
        )
        .unwrap();
    let scope = store.scope(&guard, "attempt-001").unwrap();
    if version != 4 {
        // Exact shape of an older saved state, with no display_contract field.
        let mut saved = serde_json::to_value(Saved::default()).unwrap();
        saved["state"]["display_version"] = json!(version);
        assert!(saved.get("display_contract").is_none());
        if version == 3 {
            saved["display_contract"] = json!(3);
        }
        store
            .save_publication_state(&guard, &scope, &saved, "PENDING", None, "fixture")
            .unwrap();
    }
    let metric = |id: &str, step: u64, value: f64, variant: &str| {
        let name = if version == 4 {
            format!("loss/{variant}")
        } else {
            "loss".into()
        };
        Event{event_id:id.into(),payload:normalized_metric(&json!({"name":name,"unit":"nats/token","step":step,"value":value,"variant_id":variant}),&json!({})).unwrap()}
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
    let cycle = |sdk: &mut Option<Sdk>| publish_bound(&store, &runtime, sdk, &scope, &flag);
    let read_cloud = || serde_json::from_slice::<Value>(&fs::read(&cloud).unwrap()).unwrap();
    cycle(&mut sdk).unwrap();
    let saved = store.publication_state(&guard, &scope).unwrap().unwrap();
    assert_eq!(saved["display_contract"], version);
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
    if version != 4 {
        assert!(curve_keys.len() >= 4, "variant curves remain distinct");
    }
    if version == 3 {
        assert_eq!(rows[0]["curves/loss/variants/a"], 3.0);
        assert_eq!(rows[2]["curves/loss/variants/b"], 5.0);
        assert!(rows[0].get("curves/loss").is_none());
        assert_eq!(
            store.scope_status(&guard, &scope).unwrap().unwrap()["display_version"],
            3
        );
    }
    if version == 4 {
        assert!(curve_keys.is_empty());
        assert_eq!(rows[0]["loss/a"], 3.0);
        assert_eq!(rows[2]["loss/b"], 5.0);
        assert_eq!(remote["runs"][&id]["summary"]["loss/a"], 2.0);
        assert_eq!(remote["runs"][&id]["summary"]["loss/b"], 4.0);
    }
    assert!(
        !remote["runs"][&id]["summary"]
            .as_object()
            .unwrap()
            .keys()
            .any(|k| k.starts_with("results/")),
        "active losses are curves, not final comparison values"
    );
    // Official Summary is not a Mapping: membership falls back to integer
    // indexing and fails. Exercise deletion of actual legacy auto-summary keys.
    let mut stale_summary = read_cloud();
    stale_summary["runs"][&id]["summary"]["curves/legacy"] = json!(123.0);
    fs::write(&cloud, stale_summary.to_string()).unwrap();
    store.append(&guard,&scope,&[Event{event_id:"terminal".into(),payload:json!({"kind":"lifecycle","data":{"phase":"METRICS_DRAINED","state":"SUCCEEDED"}})}]).unwrap();
    for _ in 0..4 {
        cycle(&mut sdk).unwrap();
    }
    let status = store.scope_status(&guard, &scope).unwrap().unwrap();
    assert_eq!(status["confirmed"], 5);
    assert_eq!(status["status"], "FINISHED");
    assert_eq!(read_cloud()["finishes"], 1);
    assert_eq!(
        read_cloud()["runs"][&id]["summary"]
            .get("curves/legacy")
            .is_none(),
        version != 4
    );
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
        .filter(|(k, _)| {
            if version == 4 {
                ["loss/a", "loss/b"].contains(&k.as_str())
            } else {
                k.starts_with("results/")
            }
        })
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

#[test]
fn display_version_selection_never_upgrades_started_or_uncertain_scopes() {
    assert_eq!(Saved::default().display_version(0, 0).unwrap(), 4);
    assert_eq!(Saved::default().display_version(1, 0).unwrap(), 2);
    assert_eq!(Saved::default().display_version(0, 2).unwrap(), 2);
    let mut old = Saved {
        ever_opened: true,
        ..Default::default()
    };
    assert_eq!(old.display_version(0, 0).unwrap(), 2);
    old.ever_opened = false;
    old.state.sent.insert(1, "unknown-send".into());
    assert_eq!(old.display_version(0, 0).unwrap(), 2);
    old.display_contract = Some(3);
    let restored: Saved = serde_json::from_value(serde_json::to_value(&old).unwrap()).unwrap();
    assert_eq!(restored.display_version(0, 0).unwrap(), 3);
    assert_eq!(restored.state.sent, old.state.sent);
    old.display_contract = Some(4);
    assert_eq!(old.display_version(0, 0).unwrap(), 4);
    old.display_contract = Some(5);
    assert!(old.display_version(0, 0).is_err());
}

#[test]
fn stable_projection_ignores_destination_reservations_and_reports_ambiguous_names() {
    let root = tempfile::tempdir().unwrap();
    let store = Store::open(root.path(), "0123456789abcdef0123456789abcdef").unwrap();
    for (index, run) in ["train-a", "train-b"].iter().enumerate() {
        let guard = store.guard("demo", run).unwrap();
        store
            .bind(
                &guard,
                &json!({"enabled":true,"entity":"fixture","project":"science"}),
                &json!({}),
            )
            .unwrap();
        let scope = store.scope(&guard, "attempt-001").unwrap();
        let event = |protocol: &str| Event {
            event_id: format!("{run}-{protocol}"),
            payload: json!({"kind":"metrics","observations":[{"name":"train_loss","unit":"nats/token","protocol_id":protocol,"value":3.0,"step":1,"status":"VALID"}]}),
        };
        store.append(&guard, &scope, &[event(run)]).unwrap();
        // Exercise the old allocator first so its reservation cannot influence v3.
        let legacy = store.display(&guard, &scope, 1, 2).unwrap();
        let stable = store.display(&guard, &scope, 1, 3).unwrap();
        let row = stable["series"]
            .as_object()
            .unwrap()
            .values()
            .next()
            .unwrap();
        assert_eq!(row["curve"], "curves/train_loss");
        if index == 1 {
            assert_ne!(
                legacy["series"]
                    .as_object()
                    .unwrap()
                    .values()
                    .next()
                    .unwrap()["curve"],
                row["curve"]
            );
        }
        store
            .append(&guard, &scope, &[event("different-protocol")])
            .unwrap();
        assert!(matches!(
            store.display(&guard, &scope, 2, 3),
            Err(crate::records::Error::Invalid("DISPLAY_NAME_CONFLICT"))
        ));
        assert_eq!(
            store.total(&guard, &scope).unwrap(),
            2,
            "display errors preserve the accepted journal"
        );
    }
}

#[test]
fn sdk_repeated_pages_require_identical_full_rows_and_reject_distinct_duplicate_rows() {
    let root = tempfile::tempdir().unwrap();
    let cloud = root.path().join("cloud.json");
    fs::write(root.path().join("fake_sdk.py"), include_str!("fake_sdk.py")).unwrap();
    let wrapper = root.path().join("python");
    fs::write(&wrapper,format!("#!/usr/bin/python3\nimport sys,os\nsys.path.insert(0,{root:?})\nos.environ['TEST_CLOUD']={cloud:?}\nimport fake_sdk\nsys.modules['wandb']=fake_sdk\nexec(compile(sys.argv[4],'<sdk>','exec'),{{'__name__':'__main__'}})\n",root=root.path().to_str().unwrap(),cloud=cloud.to_str().unwrap())).unwrap();
    fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o700)).unwrap();
    let base = json!({"hidden":false,"inits":0,"logs":0,"finishes":0,"runs":{"original":{
        "entity":"fixture","project":"science","state":"running","config":{},"summary":{},"pending":[],
        "history":[{"_step":700,"ml_expd/sequence":2,"ml_expd/event_id":"m2","curves/loss":0.5,"axes/loss/step":10}]}}});
    let job = json!({"operation":"observe","api_key":"never-print",
        "route":{"entity":"fixture","project":"science"},"wandb_id":"original",
        "scope":{},"config":{},"confirmed":0,"through":2,"open_session":false,"resume":"must"});
    let mut sdk = Sdk::open(&wrapper).unwrap();
    for flag in ["repeat_pages", "conflicting_page", "duplicate_sequence"] {
        let mut data = base.clone();
        data[flag] = json!(true);
        fs::write(&cloud, data.to_string()).unwrap();
        let observed = sdk.call(&job, &AtomicBool::new(false));
        if flag == "repeat_pages" {
            assert_eq!(
                observed.unwrap()["evidence"]["history"],
                json!([{"sequence":2,"event_id":"m2"}])
            );
        } else {
            assert!(matches!(
                observed,
                Err(Error::SdkRequest("WANDB_HISTORY_READ_FAILED"))
            ));
        }
        let data: Value = serde_json::from_slice(&fs::read(&cloud).unwrap()).unwrap();
        assert_eq!(data["logs"], 0);
        assert_eq!(data["inits"], 0);
    }
}
