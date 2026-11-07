use super::*;
use ml_exp_core::records::{Event, normalized_metric};
use serde_json::Value;
use std::{fs, os::unix::fs::PermissionsExt};

#[test]
fn explicit_gap_repair_appends_original_ids_and_normal_publication_resumes_without_duplicates() {
    exercise_gap_repair(false, 2);
}

#[test]
fn explicit_tail_repair_confirms_appended_tail_and_continues_without_duplicates() {
    exercise_gap_repair(false, 4);
}

#[test]
fn uncertain_repair_is_observed_without_resending_or_clearing_original_reservations() {
    exercise_gap_repair(true, 2);
    exercise_gap_repair(true, 4);
}

fn exercise_gap_repair(uncertain: bool, missing_sequence: i64) {
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
    let mut session = Session::default();
    publish_bound(&store, &runtime, &mut session, &scope, &flag).unwrap();
    let status = store.scope_status(&guard, &scope).unwrap().unwrap();
    let id = status["wandb_id"].as_str().unwrap().to_owned();
    session.shutdown();
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
    session.lost();
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
    publish_bound(&store, &runtime, &mut session, &scope, &flag).unwrap();
    publish_bound(&store, &runtime, &mut session, &scope, &flag).unwrap();
    assert_eq!(
        store.scope_status(&guard, &scope).unwrap().unwrap()["confirmed"],
        5
    );
    session.shutdown();
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
    assert_eq!(repaired["ml_expd/axes/loss/step"], missing_sequence);
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
    let root = tempfile::tempdir().unwrap();
    let cloud = root.path().join("cloud.json");
    fs::write(
        &cloud,
        r#"{"hidden":true,"flush_on_crash":true,"runs":{},"inits":0,"logs":0,"finishes":0}"#,
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
    let metric = |id: &str, step: u64, value: f64, variant: &str| {
        let name = format!("loss/{variant}");
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
    let mut session = Session::default();
    let cycle = |session: &mut Session| publish_bound(&store, &runtime, session, &scope, &flag);
    let read_cloud = || serde_json::from_slice::<Value>(&fs::read(&cloud).unwrap()).unwrap();
    cycle(&mut session).unwrap();
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
    let error = session
        .sdk
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
    record_failure(&store, &other_scope, &mut session, &error);
    assert!(
        session.sdk.is_some(),
        "a missing remote Run must not kill another Run's buffered SDK session"
    );
    assert_eq!(
        store.scope_status(&other, &other_scope).unwrap().unwrap()["error"],
        "WANDB_REMOTE_RUN_MISSING"
    );
    cycle(&mut session).unwrap();
    assert_eq!(
        store.scope_status(&guard, &scope).unwrap().unwrap()["status"],
        "ACK_PENDING"
    );
    assert_eq!(read_cloud()["logs"], 4);
    session.lost();
    cycle(&mut session).unwrap();
    assert_eq!(
        read_cloud()["logs"],
        4,
        "a lost SDK process cannot replay reserved rows"
    );
    let mut remote = read_cloud();
    remote["hidden"] = json!(false);
    fs::write(&cloud, remote.to_string()).unwrap();
    cycle(&mut session).unwrap();
    assert_eq!(
        store.scope_status(&guard, &scope).unwrap().unwrap()["confirmed"],
        4
    );
    cycle(&mut session).unwrap();
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
    assert_eq!(rows[0]["loss/a"], 3.0);
    assert_eq!(rows[2]["loss/b"], 5.0);
    assert_eq!(remote["runs"][&id]["summary"]["loss/a"], 2.0);
    assert_eq!(remote["runs"][&id]["summary"]["loss/b"], 4.0);
    assert!(
        !remote["runs"][&id]["summary"]
            .as_object()
            .unwrap()
            .keys()
            .any(|k| k.starts_with("results/")),
        "active losses are curves, not final comparison values"
    );
    // A summary key that no observation in this journal names belongs to the
    // caller, so publication leaves it alone.
    let mut client_summary = read_cloud();
    client_summary["runs"][&id]["summary"]["curves/client-owned"] = json!(123.0);
    fs::write(&cloud, client_summary.to_string()).unwrap();
    store.append(&guard,&scope,&[Event{event_id:"terminal".into(),payload:json!({"kind":"lifecycle","data":{"phase":"METRICS_DRAINED","state":"SUCCEEDED"}})}]).unwrap();
    for _ in 0..4 {
        cycle(&mut session).unwrap();
    }
    let status = store.scope_status(&guard, &scope).unwrap().unwrap();
    assert_eq!(status["confirmed"], 5);
    assert_eq!(status["status"], "FINISHED");
    assert_eq!(read_cloud()["finishes"], 1);
    assert_eq!(
        read_cloud()["runs"][&id]["summary"]["curves/client-owned"],
        123.0
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
        .filter(|(k, _)| ["loss/a", "loss/b"].contains(&k.as_str()))
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
    drop(session);
}

#[test]
fn a_name_collision_in_one_history_row_reports_its_own_code() {
    assert!(matches!(
        history_error(tracking::Error::Display("DISPLAY_NAME_CONFLICT")),
        Error::Publication("DISPLAY_NAME_CONFLICT")
    ));
    assert!(matches!(
        history_error(tracking::Error::Display("event payload")),
        Error::Publication("JOURNAL_HISTORY_PROJECTION_INVALID")
    ));
}

#[test]
fn sdk_repeats_of_a_record_are_counted_and_conflicting_rows_are_refused() {
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
    for flag in [
        "repeat_pages",
        "duplicate_sequence",
        "conflicting_page",
        "conflicting_sequence",
    ] {
        let mut data = base.clone();
        data[flag] = json!(true);
        fs::write(&cloud, data.to_string()).unwrap();
        let observed = sdk.call(&job, &AtomicBool::new(false));
        if flag == "repeat_pages" || flag == "duplicate_sequence" {
            let observed = observed.unwrap();
            assert_eq!(
                observed["evidence"]["history"],
                json!([{"sequence":2,"event_id":"m2"}])
            );
            assert_eq!(
                observed["evidence"]["duplicate_rows"],
                u64::from(flag == "duplicate_sequence")
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

/// A journal, a fake W&B and a publisher session, as one restartable test Run.
struct Fixture {
    root: tempfile::TempDir,
    cloud: std::path::PathBuf,
    runtime: Runtime,
    store: Store,
    session: Session,
    flag: AtomicBool,
}
impl Fixture {
    fn new(cloud_flags: Value) -> Self {
        let root = tempfile::tempdir().unwrap();
        let cloud = root.path().join("cloud.json");
        let mut data = json!({"hidden":false,"runs":{},"inits":0,"logs":0,"finishes":0});
        data.as_object_mut()
            .unwrap()
            .extend(cloud_flags.as_object().unwrap().clone());
        fs::write(&cloud, data.to_string()).unwrap();
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
        let store = Store::open(root.path(), "0123456789abcdef0123456789abcdef").unwrap();
        Self {
            root,
            cloud,
            runtime,
            store,
            session: Session::default(),
            flag: AtomicBool::new(false),
        }
    }
    fn scope(&self, run: &str) -> String {
        let guard = self.store.guard("demo", run).unwrap();
        self.store
            .bind(
                &guard,
                &json!({"enabled":true,"entity":"fixture","project":"science"}),
                &json!({"ml_expd":{"run_id":run}}),
            )
            .unwrap();
        self.store.scope(&guard, "attempt-001").unwrap()
    }
    fn append(&self, scope: &str, steps: std::ops::RangeInclusive<u64>) {
        let parsed = Scope::parse(scope).unwrap();
        let guard = self.store.guard("demo", &parsed.run).unwrap();
        let events: Vec<_> = steps
            .map(|step| Event {
                event_id: format!("m{step}"),
                payload: normalized_metric(
                    &json!({"name":"loss","unit":"nats/token","step":step,"value":step as f64 / 4.0}),
                    &json!({}),
                )
                .unwrap(),
            })
            .collect();
        self.store.append(&guard, scope, &events).unwrap();
    }
    fn succeed(&self, scope: &str) {
        let parsed = Scope::parse(scope).unwrap();
        let guard = self.store.guard("demo", &parsed.run).unwrap();
        self.store.append(&guard,scope,&[Event{event_id:"terminal".into(),payload:json!({"kind":"lifecycle","data":{"phase":"METRICS_DRAINED","state":"SUCCEEDED"}})}]).unwrap();
    }
    /// The server restarts: the SDK child dies and nothing in memory survives.
    fn restart(&mut self) {
        self.session = Session::default();
        self.session.settle = Duration::ZERO;
    }
    fn visit(&mut self, scope: &str) -> Result<Next> {
        publish_bound(
            &self.store,
            &self.runtime,
            &mut self.session,
            scope,
            &self.flag,
        )
    }
    /// A visit once the provider has proven the Attempt ended in `state`.
    fn visit_ended(&mut self, scope: &str, state: &str) -> Result<Next> {
        publish_ended(
            &self.store,
            &self.runtime,
            &mut self.session,
            scope,
            Some(state),
            &self.flag,
        )
    }
    fn status(&self, scope: &str) -> Value {
        let parsed = Scope::parse(scope).unwrap();
        let guard = self.store.guard("demo", &parsed.run).unwrap();
        self.store.scope_status(&guard, scope).unwrap().unwrap()
    }
    fn state(&self, scope: &str) -> Value {
        let parsed = Scope::parse(scope).unwrap();
        let guard = self.store.guard("demo", &parsed.run).unwrap();
        self.store
            .publication_state(&guard, scope)
            .unwrap()
            .unwrap()
    }
    fn cloud(&self) -> Value {
        serde_json::from_slice(&fs::read(&self.cloud).unwrap()).unwrap()
    }
    fn edit_cloud(&self, edit: impl FnOnce(&mut Value)) {
        let mut data = self.cloud();
        edit(&mut data);
        fs::write(&self.cloud, data.to_string()).unwrap();
    }
    fn rows(&self, scope: &str) -> Vec<Value> {
        let id = self.status(scope)["wandb_id"].as_str().unwrap().to_owned();
        self.cloud()["runs"][&id]["history"]
            .as_array()
            .unwrap()
            .clone()
    }
    /// Each journal record must be on W&B once, exact, on a rising physical step.
    fn assert_each_record_once(&self, scope: &str, steps: u64) {
        let rows = self.rows(scope);
        let physical: Vec<_> = rows.iter().map(|r| r["_step"].as_i64().unwrap()).collect();
        assert!(physical.windows(2).all(|pair| pair[0] < pair[1]));
        for step in 1..=steps {
            let row: Vec<_> = rows
                .iter()
                .filter(|r| r["ml_expd/event_id"] == format!("m{step}"))
                .collect();
            assert_eq!(row.len(), 1, "record {step} must appear exactly once");
            assert_eq!(row[0]["ml_expd/sequence"], step);
            assert_eq!(row[0]["loss"], step as f64 / 4.0);
            assert_eq!(row[0]["ml_expd/axes/loss/step"], step);
            assert!(
                row[0]["ml_expd/record"]
                    .as_str()
                    .unwrap()
                    .contains("nats/token")
            );
        }
        assert_eq!(self.cloud().get("dropped"), None);
    }
}

#[test]
fn one_visit_sends_a_whole_window_and_one_scan_confirms_the_range() {
    let mut fixture = Fixture::new(json!({}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=600);
    assert_eq!(fixture.visit(&scope).unwrap(), Next::Poll);
    assert_eq!(fixture.cloud()["logs"], 600);
    assert_eq!(fixture.visit(&scope).unwrap(), Next::Poll);
    assert_eq!(fixture.status(&scope)["confirmed"], 600);
    assert_eq!(fixture.state(&scope)["state"]["sent"], json!({}));
    fixture.session.shutdown();
    fixture.assert_each_record_once(&scope, 600);
}

#[test]
fn waiting_for_visibility_is_not_counted_as_a_failure() {
    let mut fixture = Fixture::new(json!({"hidden":true}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=2);
    for _ in 0..3 {
        fixture.visit(&scope).unwrap();
    }
    let status = fixture.status(&scope);
    assert_eq!(status["error"], "REMOTE_ACK_PENDING");
    assert_eq!(status["failures"], 0);
}

#[test]
fn a_synced_scope_costs_no_cloud_request_until_its_journal_grows() {
    let mut fixture = Fixture::new(json!({}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=2);
    while fixture.status(&scope)["status"] != "SYNCED"
        || fixture.visit(&scope).unwrap() != Next::Idle
    {
        fixture.visit(&scope).unwrap();
    }
    let scans = fixture.cloud()["scans"].clone();
    assert_eq!(fixture.visit(&scope).unwrap(), Next::Idle);
    assert_eq!(fixture.cloud()["scans"], scans);
    fixture.append(&scope, 3..=3);
    assert_eq!(fixture.visit(&scope).unwrap(), Next::Poll);
    assert_eq!(fixture.cloud()["logs"], 3);
}

#[test]
fn a_large_backlog_cannot_keep_a_live_run_from_sending() {
    let mut fixture = Fixture::new(json!({"hidden":true}));
    let backlog = fixture.scope("backlog");
    let live = fixture.scope("live");
    fixture.append(&backlog, 1..=300);
    fixture.append(&live, 1..=2);
    // A share never drops below what one Run needs however many Runs are busy.
    for index in 0..1000 {
        fixture
            .session
            .note_backlog(&format!("other-{index}"), true);
    }
    assert_eq!(fixture.session.share(&backlog), 128);
    fixture.visit(&backlog).unwrap();
    assert_eq!(fixture.cloud()["logs"], 128);
    fixture.visit(&live).unwrap();
    assert_eq!(
        fixture.cloud()["logs"],
        130,
        "the live Run has its own share"
    );
    for index in 0..1000 {
        fixture
            .session
            .note_backlog(&format!("other-{index}"), false);
    }
    // Shares grow back as Runs catch up, up to one Run's window.
    assert_eq!(fixture.session.share(&live), 4096);
    for other in ["c", "d", "e", "f"] {
        fixture.session.note_backlog(other, true);
    }
    assert_eq!(fixture.session.share("g"), 16384 / 7);
}

#[test]
fn frames_keep_order_and_never_exceed_the_pipe_bound() {
    let job = json!({"operation":"apply"});
    let effects: Vec<_> = (0..40)
        .map(|index| json!({"index":index,"record":"x".repeat(1024 * 1024)}))
        .collect();
    let frames = Sdk::frames(&job, effects).unwrap();
    assert!(frames.len() > 4);
    let mut next = 0;
    for frame in &frames {
        assert!(Sdk::validate_request(frame).is_ok());
        for effect in frame["effects"].as_array().unwrap() {
            assert_eq!(effect["index"], next);
            next += 1;
        }
    }
    assert_eq!(next, 40);
}

#[test]
fn a_restart_recovers_a_lost_tail_without_a_person_and_then_finishes() {
    let mut fixture = Fixture::new(json!({"hidden":true}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=2);
    fixture.visit(&scope).unwrap();
    fixture.edit_cloud(|data| data["hidden"] = json!(false));
    fixture.visit(&scope).unwrap();
    assert_eq!(fixture.status(&scope)["confirmed"], 2);
    fixture.edit_cloud(|data| data["hidden"] = json!(true));
    // Rows reserved and handed to the SDK, which then dies with the server.
    fixture.append(&scope, 3..=6);
    fixture.visit(&scope).unwrap();
    assert_eq!(fixture.cloud()["logs"], 6);
    fixture.restart();
    fixture.edit_cloud(|data| data["hidden"] = json!(false));
    // The Run finishes while its publication is still stranded.
    fixture.append(&scope, 7..=7);
    fixture.succeed(&scope);
    // The first scan only starts the clock: nothing is sent on one observation.
    fixture.visit(&scope).unwrap();
    assert_eq!(fixture.cloud()["logs"], 6);
    assert_eq!(fixture.status(&scope)["status"], "RECOVERY_SETTLING");
    let id = fixture.status(&scope)["wandb_id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(fixture.cloud()["runs"][&id]["state"], "crashed");
    // The second, unchanged scan proves absence, and only the lost rows go.
    fixture.visit(&scope).unwrap();
    assert_eq!(fixture.cloud()["logs"], 10);
    let state = fixture.state(&scope);
    assert_eq!(state["history_repair"]["source"], "automatic");
    assert!(state["history_repair"].get("attempts").is_none());
    assert_eq!(state["history_repair"]["events"]["3"], "m3");
    assert_eq!(state["history_repair"]["remote_confirmed"], false);
    assert_eq!(fixture.status(&scope)["status"], "REPAIR_ACK_PENDING");
    for _ in 0..8 {
        fixture.visit(&scope).unwrap();
    }
    let status = fixture.status(&scope);
    assert_eq!(status["status"], "FINISHED");
    assert_eq!(status["confirmed"], 8);
    assert_eq!(
        fixture.state(&scope)["history_repair"]["remote_confirmed"],
        true
    );
    assert_eq!(fixture.cloud()["runs"][&id]["state"], "finished");
    fixture.assert_each_record_once(&scope, 7);
}

#[test]
fn rows_arriving_during_the_settle_delay_are_never_sent_again() {
    // wandb-core delivers after the crash, slower than the scans read.
    let mut fixture = Fixture::new(json!({"hidden":true,"flush_on_crash":true,"reveal":0}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=4);
    fixture.visit(&scope).unwrap();
    fixture.restart();
    fixture.edit_cloud(|data| data["reveal"] = json!(1));
    // A settle of zero still needs an unchanged scan, and each of these changed.
    for confirmed in 0..4 {
        fixture.visit(&scope).unwrap();
        assert!(fixture.status(&scope)["confirmed"].as_i64().unwrap() <= confirmed + 1);
        assert_eq!(fixture.cloud()["logs"], 4);
    }
    fixture.edit_cloud(|data| data["hidden"] = json!(false));
    fixture.visit(&scope).unwrap();
    assert_eq!(fixture.status(&scope)["confirmed"], 4);
    assert_eq!(fixture.cloud()["logs"], 4);
    assert_eq!(fixture.state(&scope).get("history_repair"), None);
    fixture.session.shutdown();
    fixture.assert_each_record_once(&scope, 4);
}

#[test]
fn the_default_settle_delay_holds_uncertain_rows_back() {
    let mut fixture = Fixture::new(json!({"hidden":true}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=2);
    fixture.visit(&scope).unwrap();
    fixture.session.lost();
    for _ in 0..3 {
        fixture.visit(&scope).unwrap();
    }
    assert_eq!(fixture.cloud()["logs"], 2);
    assert_eq!(fixture.status(&scope)["status"], "RECOVERY_SETTLING");
}

#[test]
fn a_lost_repair_is_observed_then_retried_for_as_long_as_it_stays_missing() {
    let mut fixture = Fixture::new(json!({"hidden":true}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=2);
    fixture.visit(&scope).unwrap();
    for attempt in 1..=5 {
        fixture.restart();
        fixture.visit(&scope).unwrap();
        assert_eq!(fixture.cloud()["logs"], 2 * attempt);
        fixture.visit(&scope).unwrap();
        assert_eq!(fixture.cloud()["logs"], 2 * attempt + 2);
        let saved = &fixture.state(&scope)["history_repair"]["attempts"];
        assert_eq!(saved.as_u64().unwrap_or(1), attempt);
        assert_eq!(fixture.status(&scope)["status"], "REPAIR_ACK_PENDING");
    }
}

#[test]
fn a_settle_clock_ends_when_the_child_is_lost_or_a_read_fails() {
    let mut fixture = Fixture::new(json!({}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=1);
    fixture.visit(&scope).unwrap();
    let evidence = json!({"history":[]});
    let session = &mut fixture.session;
    assert!(session.stable_for(&scope, &evidence).is_none());
    assert!(session.stable_for(&scope, &evidence).is_some());
    session.lost();
    assert!(session.stable_for(&scope, &evidence).is_none());
    assert!(session.stable_for(&scope, &evidence).is_some());
    record_failure(
        &fixture.store,
        &scope,
        &mut fixture.session,
        &Error::SdkRequest("WANDB_HISTORY_READ_FAILED"),
    );
    assert!(fixture.session.stable_for(&scope, &evidence).is_none());
}

#[test]
fn a_saved_single_attempt_reads_as_an_older_server_wrote_it() {
    let saved = |attempts| {
        serde_json::to_value(HistoryRepair {
            events: BTreeMap::new(),
            attempted: true,
            remote_confirmed: false,
            attempts,
            source: None,
            at: None,
        })
        .unwrap()
    };
    assert!(saved(1).get("attempts").is_none());
    assert_eq!(saved(2)["attempts"], 2);
}

#[test]
fn a_failed_finish_leaves_no_dead_session_and_the_run_still_ends_finished() {
    let mut fixture = Fixture::new(json!({"finish_failures":1}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=2);
    fixture.succeed(&scope);
    let mut failed = false;
    for _ in 0..10 {
        failed |= fixture.visit(&scope).is_err();
    }
    assert!(failed);
    assert_eq!(fixture.status(&scope)["status"], "FINISHED");
    let id = fixture.status(&scope)["wandb_id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(fixture.cloud()["runs"][&id]["state"], "finished");
    fixture.assert_each_record_once(&scope, 2);
}

#[test]
fn production_state_with_a_confirmed_repair_loads_and_a_new_gap_can_be_repaired() {
    struct One(std::path::PathBuf);
    impl Source for One {
        fn publisher_leader(&self) -> Result<ml_exp_io::directory::FileLock> {
            ml_exp_io::directory::Directory::open(&self.0, false)
                .and_then(|directory| {
                    directory.lock(
                        std::ffi::OsStr::new("leader"),
                        ml_exp_io::directory::LockOptions {
                            create: true,
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
    let mut fixture = Fixture::new(json!({"hidden":true}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=2);
    fixture.visit(&scope).unwrap();
    fixture.session.lost();
    // Exactly the shape the previous release saved.
    let mut state = fixture.state(&scope);
    state["history_repair"] = json!({"events":{"1":"m0"},"attempted":true,"remote_confirmed":true});
    let guard = fixture.store.guard("demo", "train").unwrap();
    fixture
        .store
        .save_publication_state(&guard, &scope, &state, "ACK_PENDING", None, "fixture")
        .unwrap();
    drop(guard);
    let source = Arc::new(One(fixture.root.path().to_owned()));
    fixture.edit_cloud(|data| data["hidden"] = json!(false));
    let result = repair(source, fixture.runtime.clone(), &scope).unwrap();
    assert_eq!(result["status"], "REPAIRED");
    assert_eq!(result["appended"], 2);
    let saved = fixture.state(&scope);
    assert_eq!(saved["repairs"][0]["first"], 1);
    assert_eq!(saved["history_repair"]["source"], "operator");
    fixture.assert_each_record_once(&scope, 2);
}

#[test]
fn an_incremental_display_is_the_same_document_as_a_full_replay() {
    let fixture = Fixture::new(json!({}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=300);
    let guard = fixture.store.guard("demo", "train").unwrap();
    let mut session = Session::default();
    for through in [10, 11, 280, 300] {
        assert_eq!(
            session
                .display(&fixture.store, &guard, &scope, through)
                .unwrap(),
            fixture.store.display(&guard, &scope, through).unwrap()
        );
    }
    // A smaller window after a larger one rebuilds rather than overstating.
    assert_eq!(
        session.display(&fixture.store, &guard, &scope, 50).unwrap(),
        fixture.store.display(&guard, &scope, 50).unwrap()
    );
}

/// Visit until the publication is finished on W&B, as the publisher thread would.
fn until_finished(fixture: &mut Fixture, scope: &str) {
    for _ in 0..12 {
        if fixture.status(scope)["status"] == "FINISHED"
            && fixture.visit(scope).unwrap() == Next::Idle
        {
            return;
        }
        fixture.visit(scope).unwrap();
    }
    panic!("publication did not finish: {}", fixture.status(scope));
}

fn exercise_derived_batch(state: &str, remote_state: &str) {
    let mut fixture = Fixture::new(json!({}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=2);
    {
        let guard = fixture.store.guard("demo", "train").unwrap();
        let end = json!({"kind":"lifecycle","data":{"phase":"METRICS_DRAINED","state":state}});
        fixture
            .store
            .append(
                &guard,
                &scope,
                &[Event {
                    event_id: "terminal".into(),
                    payload: end,
                }],
            )
            .unwrap();
    }
    until_finished(&mut fixture, &scope);
    let id = fixture.status(&scope)["wandb_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let before = fixture.cloud();
    assert_eq!(before["runs"][&id]["state"], remote_state);
    assert_eq!(before["finishes"], 1);
    assert!(!candidate_scopes(&fixture.store).unwrap().contains(&scope));

    // Evaluated after the Run ended, at steps before its last training step.
    let text = "{\"name\":\"eval/fid\",\"value\":3.0,\"unit\":\"fid\",\"step\":1}\n\
                {\"name\":\"eval/fid\",\"value\":2.5,\"unit\":\"fid\",\"step\":2}\n";
    let observations =
        ml_exp_core::records::derived_observations(text.as_bytes(), &json!({})).unwrap();
    let sha256 = "a".repeat(64);
    let append = |fixture: &Fixture, at: &str| {
        let guard = fixture.store.guard("demo", "train").unwrap();
        fixture
            .store
            .append_derived(&guard, &scope, &sha256, at, &observations)
            .unwrap()
    };
    let batch = append(&fixture, "2026-11-07T00:00:00Z");
    assert_eq!((batch.first, batch.last, batch.records), (4, 5, 2));
    assert!(candidate_scopes(&fixture.store).unwrap().contains(&scope));
    until_finished(&mut fixture, &scope);

    let status = fixture.status(&scope);
    assert_eq!(status["confirmed"], 5);
    let after = fixture.cloud();
    assert_eq!(
        after["runs"].as_object().unwrap().len(),
        1,
        "the same W&B Run"
    );
    assert_eq!(after["runs"][&id]["state"], remote_state);
    assert_eq!(after["finishes"], 2);
    assert_eq!(after["inits"], before["inits"].as_i64().unwrap() + 1);
    assert_eq!(after["logs"], before["logs"].as_i64().unwrap() + 2);
    let rows = fixture.rows(&scope);
    assert_eq!(rows.len(), 5);
    let derived: Vec<_> = rows
        .iter()
        .filter(|row| row["ml_expd/source"] == "derived")
        .collect();
    assert_eq!(derived.len(), 2);
    for (row, (step, value)) in derived.iter().zip([(1, 3.0), (2, 2.5)]) {
        assert_eq!(row["eval/fid"], value);
        assert_eq!(row["ml_expd/axes/eval/fid/step"], step);
        assert_eq!(row["ml_expd/kind"], "derived_metrics");
        assert_eq!(row["ml_expd/derived_sha256"], sha256.as_str());
    }
    assert_eq!(
        rows.iter()
            .filter(|row| row.get("ml_expd/source").is_some())
            .count(),
        2,
        "training rows are not marked"
    );
    assert_eq!(after["runs"][&id]["summary"]["eval/fid"], 2.5);
    assert_eq!(after["runs"][&id]["summary"]["loss"], 0.5);
    fixture.assert_each_record_once(&scope, 2);

    // The same file again publishes nothing and returns the same batch.
    assert_eq!(append(&fixture, "2026-11-08T00:00:00Z"), batch);
    assert_eq!(fixture.visit(&scope).unwrap(), Next::Idle);
    assert!(!candidate_scopes(&fixture.store).unwrap().contains(&scope));
    let repeated = fixture.cloud();
    assert_eq!(repeated["logs"], after["logs"]);
    assert_eq!(repeated["inits"], after["inits"]);
    assert_eq!(repeated["finishes"], 2);
}

#[test]
fn a_derived_batch_reaches_the_same_finished_wandb_run_which_ends_finished_again() {
    exercise_derived_batch("SUCCEEDED", "finished");
}

#[test]
fn a_derived_batch_for_a_failed_run_leaves_its_wandb_run_failed_again() {
    exercise_derived_batch("FAILED", "failed");
}

/// Visit until the scope is synced and idle, with no terminal evidence.
fn until_synced(fixture: &mut Fixture, scope: &str) {
    for _ in 0..12 {
        if fixture.status(scope)["status"] == "SYNCED"
            && fixture.visit(scope).unwrap() == Next::Idle
        {
            return;
        }
        fixture.visit(scope).unwrap();
    }
    panic!("publication did not sync: {}", fixture.status(scope));
}

/// A preempted or killed job never writes METRICS_DRAINED. Once the provider proves it
/// ended, its W&B Run is finished as failed, with no new row and no invented event.
#[test]
fn a_job_the_provider_ended_finishes_its_wandb_run_as_failed_without_new_rows() {
    let mut fixture = Fixture::new(json!({}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=3);
    {
        let guard = fixture.store.guard("demo", "train").unwrap();
        let running = json!({"kind":"lifecycle","data":{"phase":"TRAINING","state":"RUNNING"}});
        fixture
            .store
            .append(
                &guard,
                &scope,
                &[Event {
                    event_id: "running".into(),
                    payload: running,
                }],
            )
            .unwrap();
    }
    until_synced(&mut fixture, &scope);
    let total = fixture.status(&scope)["total"].clone();
    let id = fixture.status(&scope)["wandb_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let before = fixture.cloud();
    assert_eq!(before["runs"][&id]["state"], "running");
    assert_eq!(before["finishes"], 0);
    // Still running as far as the provider knows: nothing finishes.
    assert_eq!(fixture.visit(&scope).unwrap(), Next::Idle);
    assert_eq!(fixture.cloud()["finishes"], 0);

    let mut finished = false;
    for _ in 0..12 {
        if fixture.status(&scope)["status"] == "FINISHED"
            && fixture.visit_ended(&scope, "PREEMPTED").unwrap() == Next::Idle
        {
            finished = true;
            break;
        }
        fixture.visit_ended(&scope, "PREEMPTED").unwrap();
    }
    assert!(finished, "{}", fixture.status(&scope));
    let after = fixture.cloud();
    assert_eq!(after["runs"][&id]["state"], "failed");
    assert_eq!(after["finishes"], 1);
    assert_eq!(after["logs"], before["logs"], "no row is sent again");
    let status = fixture.status(&scope);
    assert_eq!(status["total"], total, "no event is invented");
    assert_eq!(status["confirmed"], total);
    assert!(!candidate_scopes(&fixture.store).unwrap().contains(&scope));
    // Finished: later visits cost nothing.
    assert_eq!(
        fixture.visit_ended(&scope, "PREEMPTED").unwrap(),
        Next::Idle
    );
    assert_eq!(fixture.cloud()["finishes"], 1);
    fixture.assert_each_record_once(&scope, 3);
}

/// A person deleted the W&B Run. The publisher never recreates it: the scope says so
/// and stops, and later visits make no W&B request.
#[test]
fn a_deleted_wandb_run_is_never_recreated_and_its_scope_stops() {
    let mut fixture = Fixture::new(json!({}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=2);
    until_synced(&mut fixture, &scope);
    let id = fixture.status(&scope)["wandb_id"]
        .as_str()
        .unwrap()
        .to_owned();
    fixture.append(&scope, 3..=3);
    fixture.restart();
    fixture.edit_cloud(|data| {
        data["runs"].as_object_mut().unwrap().remove(&id);
    });
    let before = fixture.cloud();
    assert_eq!(fixture.visit(&scope).unwrap(), Next::Idle);
    let status = fixture.status(&scope);
    assert_eq!(status["status"], "REMOTE_RUN_MISSING");
    assert_eq!(status["error"], "WANDB_REMOTE_RUN_MISSING");
    assert_eq!(status["confirmed"], 2);
    assert!(!candidate_scopes(&fixture.store).unwrap().contains(&scope));
    for _ in 0..3 {
        assert_eq!(fixture.visit(&scope).unwrap(), Next::Idle);
        assert_eq!(
            fixture.visit_ended(&scope, "PREEMPTED").unwrap(),
            Next::Idle
        );
    }
    fixture.restart();
    assert_eq!(fixture.visit(&scope).unwrap(), Next::Idle);
    let after = fixture.cloud();
    assert!(after["runs"].get(&id).is_none(), "never recreated");
    assert_eq!(after["inits"], before["inits"]);
    assert_eq!(after["logs"], before["logs"]);
    assert_eq!(after["finishes"], before["finishes"]);
    assert_eq!(fixture.status(&scope)["status"], "REMOTE_RUN_MISSING");
}

const EXCLUDED: &str = "loss/old\neval/old\n\nnever/seen\n";

/// Each step writes `loss` and, in a row of its own, the name the client wants gone.
fn append_named(fixture: &Fixture, scope: &str, steps: std::ops::RangeInclusive<u64>) {
    let guard = fixture.store.guard("demo", "train").unwrap();
    let mut events = Vec::new();
    for step in steps {
        for (id, name) in [("m", "loss"), ("o", "loss/old")] {
            events.push(Event {
                event_id: format!("{id}{step}"),
                payload: normalized_metric(
                    &json!({"name":name,"unit":"nats/token","step":step,"value":step as f64 / 4.0}),
                    &json!({}),
                )
                .unwrap(),
            });
        }
    }
    fixture.store.append(&guard, scope, &events).unwrap();
}

fn append_derived_text(fixture: &Fixture, scope: &str, text: &str) {
    use sha2::{Digest, Sha256};
    let guard = fixture.store.guard("demo", "train").unwrap();
    let observations =
        ml_exp_core::records::derived_observations(text.as_bytes(), &json!({})).unwrap();
    let sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
    fixture
        .store
        .append_derived(
            &guard,
            scope,
            &sha256,
            "2026-11-07T00:00:00Z",
            &observations,
        )
        .unwrap();
}

/// What the HTTP route stores for one exclusion list.
fn rebuild_with(
    fixture: &Fixture,
    scope: &str,
    names: &str,
    provider_ended: bool,
) -> crate::records::Result<(crate::records::WandbRebuild, bool)> {
    use sha2::{Digest, Sha256};
    let guard = fixture.store.guard("demo", "train").unwrap();
    let sha256 = format!("{:x}", Sha256::digest(names.as_bytes()));
    fixture.store.rebuild_wandb(
        &guard,
        scope,
        &sha256,
        names,
        "2026-11-07T12:00:00Z",
        provider_ended,
    )
}

fn end(fixture: &Fixture, scope: &str, state: &str) {
    let guard = fixture.store.guard("demo", "train").unwrap();
    let end = json!({"kind":"lifecycle","data":{"phase":"METRICS_DRAINED","state":state}});
    fixture
        .store
        .append(
            &guard,
            scope,
            &[Event {
                event_id: "terminal".into(),
                payload: end,
            }],
        )
        .unwrap();
}

/// A finished Attempt whose old W&B Run holds `loss/old` and `eval/old`.
fn finished_with_old_names(state: &str, flags: Value) -> (Fixture, String, String) {
    let mut fixture = Fixture::new(flags);
    let scope = fixture.scope("train");
    append_named(&fixture, &scope, 1..=4);
    end(&fixture, &scope, state);
    until_finished(&mut fixture, &scope);
    append_derived_text(
        &fixture,
        &scope,
        "{\"name\":\"eval/fid\",\"value\":3.0,\"unit\":\"fid\",\"step\":1}\n\
         {\"name\":\"eval/old\",\"value\":9.0,\"unit\":\"fid\",\"step\":1}\n\
         {\"name\":\"eval/fid\",\"value\":2.5,\"unit\":\"fid\",\"step\":2}\n",
    );
    until_finished(&mut fixture, &scope);
    let old = fixture.status(&scope)["wandb_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let cloud = fixture.cloud();
    assert!(cloud["runs"][&old]["summary"].get("loss/old").is_some());
    (fixture, scope, old)
}

/// Every sequence once on `rows`, and no trace of an excluded name.
fn assert_rebuilt_rows(rows: &[Value], total: i64) {
    let mut sequences: Vec<_> = rows
        .iter()
        .map(|row| row["ml_expd/sequence"].as_i64().unwrap())
        .collect();
    sequences.sort();
    assert_eq!(sequences, (1..=total).collect::<Vec<_>>());
    for row in rows {
        for name in ["loss/old", "eval/old"] {
            assert!(row.get(name).is_none(), "{row}");
            assert!(!row.to_string().contains(name), "{row}");
        }
    }
    let count = |name: &str| rows.iter().filter(|row| row.get(name).is_some()).count();
    assert_eq!(count("loss"), 4);
    assert_eq!(count("eval/fid"), 2);
    // A row whose only metric was excluded keeps its sequence and identity.
    let empty = rows
        .iter()
        .find(|row| row["ml_expd/event_id"] == "o1")
        .unwrap();
    assert_eq!(empty["ml_expd/kind"], "metrics");
    assert_eq!(
        empty
            .as_object()
            .unwrap()
            .keys()
            .filter(|key| !key.starts_with("ml_expd/") && *key != "_step")
            .count(),
        0
    );
}

fn exercise_rebuild(state: &str, remote_state: &str) {
    let (mut fixture, scope, old) = finished_with_old_names(state, json!({}));
    let before = fixture.cloud();
    let total = fixture.status(&scope)["total"].as_i64().unwrap();
    assert_eq!(total, 12);
    let training = {
        let guard = fixture.store.guard("demo", "train").unwrap();
        fixture.store.metric_records(&guard, "attempt-001").unwrap()
    };

    let (rebuild, created) = rebuild_with(&fixture, &scope, EXCLUDED, false).unwrap();
    assert!(created);
    assert_eq!(rebuild.previous_wandb_id, old);
    assert_ne!(rebuild.wandb_id, old);
    let document = rebuild.document();
    assert_eq!(document["status"], "PUBLISHING");
    assert_eq!(document["names"], 3);
    assert_eq!(document["excluded"], json!({"eval/old":1,"loss/old":4}));
    assert_eq!(document["excluded_observations"], 5);
    assert_eq!(document["unmatched_names"], json!(["never/seen"]));
    assert_eq!(document["emptied_rows"], 5);
    assert_eq!(document["previous_wandb_run_deleted"], false);
    let status = fixture.status(&scope);
    assert_eq!(status["wandb_id"], rebuild.wandb_id.as_str());
    assert_eq!(status["confirmed"], 0);
    assert!(candidate_scopes(&fixture.store).unwrap().contains(&scope));
    // The scope is still the same scope to every writer.
    {
        let guard = fixture.store.guard("demo", "train").unwrap();
        assert_eq!(fixture.store.scope(&guard, "attempt-001").unwrap(), scope);
    }

    until_finished(&mut fixture, &scope);
    let after = fixture.cloud();
    let new = &after["runs"][&rebuild.wandb_id];
    assert_eq!(new["state"], remote_state);
    assert_eq!(new["name"], "train/attempt-001");
    assert_eq!(after["runs"][&old]["name"], "train/attempt-001");
    assert_eq!(after["runs"].as_object().unwrap().len(), 2);
    // The old W&B Run was not opened, written or finished again.
    assert_eq!(after["runs"][&old], before["runs"][&old]);
    assert_eq!(after["finishes"], before["finishes"].as_i64().unwrap() + 1);
    let rows = fixture.rows(&scope);
    assert_eq!(rows.len() as i64, total);
    assert_eq!(
        after["logs"],
        before["logs"].as_i64().unwrap() + total,
        "each row is sent once"
    );
    assert_rebuilt_rows(&rows, total);
    for name in ["loss/old", "eval/old"] {
        assert!(new["summary"].get(name).is_none());
        assert!(new["definitions"].get(name).is_none());
    }
    assert_eq!(new["summary"]["loss"], 1.0);
    assert_eq!(new["summary"]["eval/fid"], 2.5);
    let finished = fixture.status(&scope);
    assert_eq!(finished["status"], "FINISHED");
    assert_eq!(finished["confirmed"], total);
    assert!(!candidate_scopes(&fixture.store).unwrap().contains(&scope));
    {
        let guard = fixture.store.guard("demo", "train").unwrap();
        let rebuilds = fixture.store.wandb_rebuilds(&guard, &scope).unwrap();
        assert_eq!(rebuilds.len(), 1);
        assert_eq!(rebuilds[0].document()["status"], "FINISHED");
        // The journal still holds every excluded record.
        assert_eq!(
            fixture.store.metric_records(&guard, "attempt-001").unwrap(),
            training
        );
    }

    // The same list again is the same rebuild: nothing is sent or opened.
    let (again, created) = rebuild_with(&fixture, &scope, EXCLUDED, false).unwrap();
    assert!(!created);
    assert_eq!(again.wandb_id, rebuild.wandb_id);
    assert_eq!(fixture.visit(&scope).unwrap(), Next::Idle);
    let repeated = fixture.cloud();
    assert_eq!(repeated["logs"], after["logs"]);
    assert_eq!(repeated["inits"], after["inits"]);

    // A later derived batch goes to the new W&B Run only.
    append_derived_text(
        &fixture,
        &scope,
        "{\"name\":\"eval/late\",\"value\":1.5,\"unit\":\"fid\",\"step\":4}\n\
         {\"name\":\"eval/old\",\"value\":8.0,\"unit\":\"fid\",\"step\":4}\n",
    );
    until_finished(&mut fixture, &scope);
    let late = fixture.cloud();
    assert_eq!(late["runs"][&old], before["runs"][&old]);
    assert_eq!(late["runs"][&rebuild.wandb_id]["state"], remote_state);
    let rows = fixture.rows(&scope);
    assert_eq!(rows.len() as i64, total + 2);
    assert_eq!(
        rows.iter()
            .filter(|row| row.get("eval/late").is_some())
            .count(),
        1
    );
    assert!(rows.iter().all(|row| !row.to_string().contains("eval/old")));

    // Another list starts from the current id, never from the old one.
    let (second, created) = rebuild_with(&fixture, &scope, "eval/late\n", false).unwrap();
    assert!(created);
    assert_eq!(second.ordinal, 2);
    assert_eq!(second.previous_wandb_id, rebuild.wandb_id);
    assert!(![old.as_str(), rebuild.wandb_id.as_str()].contains(&second.wandb_id.as_str()));
    // A third list waits until that publication has finished.
    assert!(matches!(
        rebuild_with(&fixture, &scope, "loss\n", false),
        Err(crate::records::Error::RebuildNotFinished)
    ));
    // The list in progress repeats as itself.
    assert!(
        !rebuild_with(&fixture, &scope, "eval/late\n", false)
            .unwrap()
            .1
    );
}

#[test]
fn a_rebuild_publishes_a_new_finished_wandb_run_without_the_excluded_names() {
    exercise_rebuild("SUCCEEDED", "finished");
}

#[test]
fn a_rebuild_of_a_failed_run_ends_its_new_wandb_run_failed() {
    exercise_rebuild("FAILED", "failed");
}

#[test]
fn a_rebuild_needs_a_finished_publication_and_an_ended_run() {
    let mut fixture = Fixture::new(json!({}));
    let scope = fixture.scope("train");
    append_named(&fixture, &scope, 1..=2);
    // Running, published but not finished.
    until_synced(&mut fixture, &scope);
    assert!(matches!(
        rebuild_with(&fixture, &scope, EXCLUDED, false),
        Err(crate::records::Error::RebuildNotFinished)
    ));
    // Ended, but its last rows are not yet on W&B.
    end(&fixture, &scope, "SUCCEEDED");
    assert!(matches!(
        rebuild_with(&fixture, &scope, EXCLUDED, false),
        Err(crate::records::Error::RebuildNotFinished)
    ));
    until_finished(&mut fixture, &scope);
    // A derived batch not yet published.
    append_derived_text(
        &fixture,
        &scope,
        "{\"name\":\"eval/fid\",\"value\":3.0,\"unit\":\"fid\",\"step\":1}\n",
    );
    assert!(matches!(
        rebuild_with(&fixture, &scope, EXCLUDED, false),
        Err(crate::records::Error::RebuildNotFinished)
    ));
    let guard = fixture.store.guard("demo", "train").unwrap();
    assert!(
        fixture
            .store
            .wandb_rebuilds(&guard, &scope)
            .unwrap()
            .is_empty()
    );
    assert_eq!(fixture.cloud()["runs"].as_object().unwrap().len(), 1);
}

#[test]
fn a_run_the_provider_ended_is_rebuilt_only_with_that_proof() {
    let mut fixture = Fixture::new(json!({}));
    let scope = fixture.scope("train");
    append_named(&fixture, &scope, 1..=2);
    for _ in 0..12 {
        fixture.visit_ended(&scope, "PREEMPTED").unwrap();
    }
    assert_eq!(fixture.status(&scope)["status"], "FINISHED");
    // The journal never recorded the end of this Run.
    assert!(matches!(
        rebuild_with(&fixture, &scope, "loss/old\n", false),
        Err(crate::records::Error::NotEnded)
    ));
    let (rebuild, _) = rebuild_with(&fixture, &scope, "loss/old\n", true).unwrap();
    for _ in 0..12 {
        fixture.visit_ended(&scope, "PREEMPTED").unwrap();
    }
    assert_eq!(fixture.status(&scope)["status"], "FINISHED");
    let cloud = fixture.cloud();
    assert_eq!(cloud["runs"][&rebuild.wandb_id]["state"], "failed");
    let rows = fixture.rows(&scope);
    assert_eq!(rows.len(), 4);
    assert!(rows.iter().all(|row| row.get("loss/old").is_none()));
}

#[test]
fn a_rebuild_interrupted_by_a_restart_resumes_the_same_new_run_without_duplicates() {
    let (mut fixture, scope, old) = finished_with_old_names("SUCCEEDED", json!({}));
    let total = fixture.status(&scope)["total"].as_i64().unwrap();
    let before = fixture.cloud();
    let (rebuild, _) = rebuild_with(&fixture, &scope, EXCLUDED, false).unwrap();
    // The new Run is created and every row handed to the SDK, which then dies
    // with the server before W&B shows any of them.
    fixture.edit_cloud(|data| data["hidden"] = json!(true));
    fixture.visit(&scope).unwrap();
    let sent = before["logs"].as_i64().unwrap() + total;
    assert_eq!(fixture.cloud()["logs"], sent);
    assert!(fixture.cloud()["runs"].get(&rebuild.wandb_id).is_some());
    fixture.restart();
    fixture.edit_cloud(|data| data["hidden"] = json!(false));
    // After a restart the rebuild is the same: same list, same new id.
    let (again, created) = rebuild_with(&fixture, &scope, EXCLUDED, false).unwrap();
    assert!(!created);
    assert_eq!(again.wandb_id, rebuild.wandb_id);
    assert_eq!(
        fixture.status(&scope)["wandb_id"],
        rebuild.wandb_id.as_str()
    );
    // Nothing is sent again on one observation of their absence.
    fixture.visit(&scope).unwrap();
    assert_eq!(fixture.cloud()["logs"], sent);
    until_finished(&mut fixture, &scope);
    let after = fixture.cloud();
    assert_eq!(after["runs"].as_object().unwrap().len(), 2, "no third Run");
    assert_eq!(after["runs"][&old], before["runs"][&old]);
    assert_eq!(after["runs"][&rebuild.wandb_id]["state"], "finished");
    assert_eq!(after["logs"], sent + total, "only the lost rows, once");
    assert!(fixture.state(&scope).get("duplicate_rows").is_none());
    let rows = fixture.rows(&scope);
    assert_eq!(rows.len() as i64, total);
    assert_rebuilt_rows(&rows, total);
}
