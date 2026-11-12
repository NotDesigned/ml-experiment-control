use super::*;
use ml_exp_core::records::{Event, normalized_metric};
use std::{fs, os::unix::fs::PermissionsExt};

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
        let mut data = json!({"runs":{},"inits":0,"logs":0,"finishes":0});
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

/// Visit until the scope has nothing left to do now.
fn settle(fixture: &mut Fixture, scope: &str) {
    for _ in 0..32 {
        if fixture.visit(scope).unwrap() == Next::Idle {
            return;
        }
    }
    panic!("publication did not settle: {}", fixture.status(scope));
}

#[test]
fn rows_are_sent_once_in_order_and_the_run_finishes() {
    let mut fixture = Fixture::new(json!({}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=4);
    // One visit sends every row, and the cursor moves on the SDK's word alone.
    assert_eq!(fixture.visit(&scope).unwrap(), Next::More);
    assert_eq!(fixture.cloud()["logs"], 4);
    let status = fixture.status(&scope);
    assert_eq!(status["confirmed"], 4);
    assert_eq!(status["status"], "SYNCED");
    assert_eq!(status["display_version"], tracking::DISPLAY_VERSION);
    settle(&mut fixture, &scope);
    assert_eq!(fixture.cloud()["finishes"], 0, "a running Run stays open");
    let id = fixture.status(&scope)["wandb_id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(fixture.cloud()["runs"][&id]["summary"]["loss"], 1.0);
    // A summary key that no observation in this journal names belongs to the
    // caller, so publication leaves it alone.
    fixture.edit_cloud(|data| data["runs"][&id]["summary"]["curves/client-owned"] = json!(123.0));
    fixture.append(&scope, 5..=6);
    fixture.succeed(&scope);
    until_finished(&mut fixture, &scope);
    let status = fixture.status(&scope);
    assert_eq!(status["confirmed"], 7);
    assert_eq!(status["status"], "FINISHED");
    let cloud = fixture.cloud();
    assert_eq!(cloud["logs"], 7);
    assert_eq!(cloud["finishes"], 1);
    assert_eq!(cloud["inits"], 1, "one session for the whole Run");
    assert_eq!(cloud["runs"][&id]["state"], "finished");
    assert_eq!(cloud["runs"][&id]["summary"]["loss"], 1.5);
    assert_eq!(cloud["runs"][&id]["summary"]["curves/client-owned"], 123.0);
    fixture.assert_each_record_once(&scope, 6);
    let sequences: Vec<_> = fixture
        .rows(&scope)
        .iter()
        .map(|row| row["ml_expd/sequence"].as_i64().unwrap())
        .collect();
    assert_eq!(sequences, (1..=7).collect::<Vec<_>>());
    let saved = fixture.state(&scope);
    assert_eq!(saved["remote_finished"], true);
    assert!(
        !fs::read(fixture.store.path())
            .unwrap()
            .windows(b"never-print".len())
            .any(|w| w == b"never-print")
    );
    assert!(
        !fs::read_to_string(&fixture.cloud)
            .unwrap()
            .contains("never-print")
    );
}

#[test]
fn a_send_error_retries_the_same_rows_from_the_cursor() {
    // The first row of the batch fails: nothing was logged, so nothing repeats.
    let mut fixture = Fixture::new(json!({"fail_logs":[1]}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=3);
    let error = fixture.visit(&scope).unwrap_err();
    assert!(matches!(error, Error::SdkRequest("WANDB_WRITE_FAILED")));
    record_failure(&fixture.store, &scope, &mut fixture.session, &error);
    let status = fixture.status(&scope);
    assert_eq!(status["confirmed"], 0);
    assert_eq!(status["status"], "RETRY_PENDING");
    assert_eq!(status["error"], "WANDB_WRITE_FAILED");
    assert!(
        fixture.session.sdk.is_some(),
        "an error frame keeps the SDK"
    );
    settle(&mut fixture, &scope);
    let status = fixture.status(&scope);
    assert_eq!(status["confirmed"], 3);
    assert_eq!(status["status"], "SYNCED");
    assert_eq!(status["error"], Value::Null);
    assert_eq!(status["failures"], 0);
    assert_eq!(fixture.cloud()["logs"], 4);
    fixture.assert_each_record_once(&scope, 3);

    // A row fails inside a batch: the batch is sent again whole, so W&B holds
    // the rows before the failure twice, and every row at least once.
    fixture.append(&scope, 4..=6);
    fixture.edit_cloud(|data| data["fail_logs"] = json!([6]));
    assert!(fixture.visit(&scope).is_err());
    assert_eq!(fixture.status(&scope)["confirmed"], 3);
    settle(&mut fixture, &scope);
    assert_eq!(fixture.status(&scope)["confirmed"], 6);
    let count = |id: &str| {
        fixture
            .rows(&scope)
            .iter()
            .filter(|row| row["ml_expd/event_id"] == id)
            .count()
    };
    assert_eq!((count("m4"), count("m5"), count("m6")), (2, 1, 1));
    assert_eq!(fixture.cloud()["logs"], 4 + 2 + 3);
}

#[test]
fn a_restart_continues_from_the_cursor_without_resending_accepted_rows() {
    let mut fixture = Fixture::new(json!({}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=10);
    fixture.session.batch = 4;
    assert_eq!(fixture.visit(&scope).unwrap(), Next::More);
    assert_eq!(fixture.status(&scope)["confirmed"], 4);
    assert_eq!(fixture.status(&scope)["status"], "PENDING");
    // The server restarts: the SDK child dies and nothing in memory survives.
    fixture.restart();
    fixture.succeed(&scope);
    until_finished(&mut fixture, &scope);
    let cloud = fixture.cloud();
    assert_eq!(cloud["logs"], 11, "no accepted row is sent again");
    assert_eq!(cloud["inits"], 2, "the same W&B Run, resumed");
    assert_eq!(cloud["runs"].as_object().unwrap().len(), 1);
    assert_eq!(fixture.status(&scope)["status"], "FINISHED");
    fixture.assert_each_record_once(&scope, 10);
}

#[test]
fn one_visit_sends_a_whole_batch() {
    let mut fixture = Fixture::new(json!({}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=300);
    fixture.session.batch = 128;
    assert_eq!(fixture.visit(&scope).unwrap(), Next::More);
    assert_eq!(fixture.cloud()["logs"], 128);
    assert_eq!(fixture.status(&scope)["confirmed"], 128);
    settle(&mut fixture, &scope);
    assert_eq!(fixture.status(&scope)["confirmed"], 300);
    assert_eq!(fixture.cloud()["logs"], 300);
    fixture.assert_each_record_once(&scope, 300);
}

#[test]
fn a_synced_scope_costs_no_cloud_request_until_its_journal_grows() {
    let mut fixture = Fixture::new(json!({}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=2);
    settle(&mut fixture, &scope);
    let before = fixture.cloud();
    assert_eq!(fixture.visit(&scope).unwrap(), Next::Idle);
    assert_eq!(fixture.cloud(), before);
    fixture.append(&scope, 3..=3);
    assert_eq!(fixture.visit(&scope).unwrap(), Next::More);
    assert_eq!(fixture.cloud()["logs"], 3);
}

#[test]
fn a_missing_remote_run_is_never_created_by_a_resuming_request() {
    let fixture = Fixture::new(json!({}));
    let mut sdk = Sdk::open(&fixture.runtime.config.python).unwrap();
    let error = sdk
        .call(
            &json!({"operation":"apply","api_key":"never-print",
        "route":{"entity":"fixture","project":"science"},"wandb_id":"missing",
        "scope":{"project":"demo","run":"train","attempt":"attempt-001"},
        "config":{},"resume":"must","definitions":[],"history":[{"loss":1.0}],"display":null}),
            &AtomicBool::new(false),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::SdkRequest("WANDB_REMOTE_RUN_MISSING")
    ));
    assert!(!error.to_string().contains("sensitive"));
    let state = fixture.cloud();
    assert_eq!(state["inits"], 0);
    assert_eq!(state["logs"], 0);
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
fn frames_keep_order_never_exceed_the_pipe_bound_and_end_with_the_display() {
    let job = json!({"operation":"apply"});
    let history: Vec<_> = (0..40)
        .map(|index| json!({"index":index,"record":"x".repeat(1024 * 1024)}))
        .collect();
    let display = json!({"set":{"loss":1.0},"delete":[]});
    let frames = Sdk::frames(&job, history, display.clone()).unwrap();
    assert!(frames.len() > 4);
    let mut next = 0;
    for (index, frame) in frames.iter().enumerate() {
        assert!(Sdk::validate_request(frame).is_ok());
        let last = index == frames.len() - 1;
        assert_eq!(
            frame["display"],
            if last { display.clone() } else { Value::Null }
        );
        for record in frame["history"].as_array().unwrap() {
            assert_eq!(record["index"], next);
            next += 1;
        }
    }
    assert_eq!(next, 40);
    let only = Sdk::frames(&job, Vec::new(), display.clone()).unwrap();
    assert_eq!(only.len(), 1);
    assert_eq!(only[0]["history"], json!([]));
    assert_eq!(only[0]["display"], display);
}

/// Exactly what an earlier release saved while it waited to read rows back from
/// W&B history, and while it repaired them.
fn legacy_state(sent: Value) -> Value {
    json!({"state":{"confirmed":2,"sent":sent,"display_version":4,"display_through":2,"finished":false},
        "finish_requested":true,"remote_finished":false,"ever_opened":true,
        "history_repair":{"events":{"3":"m3"},"attempted":true,"remote_confirmed":false,
            "attempts":2,"source":"automatic","at":"2026-11-01T00:00:00Z"},
        "repairs":[{"first":1,"last":1,"count":1,"attempts":1,"source":null,"at":null}],
        "duplicate_rows":1})
}

#[test]
fn an_earlier_releases_reserved_and_repairing_states_load_and_move_on() {
    for (phase, error) in [
        ("ACK_PENDING", "REMOTE_ACK_PENDING"),
        ("REPAIR_ACK_PENDING", "REMOTE_REPAIR_ACK_PENDING"),
        ("RECOVERY_SETTLING", "REMOTE_ACK_PENDING"),
    ] {
        let mut fixture = Fixture::new(json!({}));
        let scope = fixture.scope("train");
        fixture.append(&scope, 1..=6);
        fixture.session.batch = 2;
        fixture.visit(&scope).unwrap();
        assert_eq!(fixture.status(&scope)["confirmed"], 2);
        // Rows 3 and 5 were handed to the SDK by the earlier release; 4 was not.
        {
            let guard = fixture.store.guard("demo", "train").unwrap();
            fixture
                .store
                .save_publication_state(
                    &guard,
                    &scope,
                    &legacy_state(json!({"3":"m3","5":"m5"})),
                    phase,
                    Some(error),
                    "2026-11-01T00:00:00Z",
                )
                .unwrap();
        }
        fixture.restart();
        fixture.succeed(&scope);
        until_finished(&mut fixture, &scope);
        let status = fixture.status(&scope);
        assert_eq!(status["status"], "FINISHED");
        assert_eq!(status["confirmed"], 7);
        assert_eq!(status["error"], Value::Null);
        // Row 3 counts as sent; reservations after the gap at 4 do not.
        assert_eq!(fixture.cloud()["logs"], 2 + 4, "{phase}");
        let saved = fixture.state(&scope);
        for gone in [
            "history_repair",
            "repairs",
            "duplicate_rows",
            "finish_requested",
        ] {
            assert!(saved.get(gone).is_none(), "{phase}: {gone}");
        }
        assert!(saved["state"].get("sent").is_none());
        assert_eq!(fixture.cloud()["runs"].as_object().unwrap().len(), 1);
    }
}

#[test]
fn an_earlier_releases_state_with_every_row_reserved_moves_to_synced() {
    let mut fixture = Fixture::new(json!({}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=4);
    fixture.session.batch = 2;
    fixture.visit(&scope).unwrap();
    {
        let guard = fixture.store.guard("demo", "train").unwrap();
        fixture
            .store
            .save_publication_state(
                &guard,
                &scope,
                &legacy_state(json!({"3":"m3","4":"m4"})),
                "ACK_PENDING",
                Some("REMOTE_ACK_PENDING"),
                "2026-11-01T00:00:00Z",
            )
            .unwrap();
    }
    fixture.restart();
    settle(&mut fixture, &scope);
    let status = fixture.status(&scope);
    assert_eq!(status["status"], "SYNCED");
    assert_eq!(status["confirmed"], 4);
    assert_eq!(status["display_confirmed"], 4);
    assert_eq!(fixture.cloud()["logs"], 2, "nothing is sent twice");
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
    assert_eq!(fixture.cloud()["logs"], 3, "a failed finish resends no row");
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
    // The new Run is created and part of it sent before the server stops.
    fixture.session.batch = 5;
    assert_eq!(fixture.visit(&scope).unwrap(), Next::More);
    assert_eq!(fixture.status(&scope)["confirmed"], 5);
    assert!(fixture.cloud()["runs"].get(&rebuild.wandb_id).is_some());
    fixture.restart();
    // After a restart the rebuild is the same: same list, same new id.
    let (again, created) = rebuild_with(&fixture, &scope, EXCLUDED, false).unwrap();
    assert!(!created);
    assert_eq!(again.wandb_id, rebuild.wandb_id);
    assert_eq!(
        fixture.status(&scope)["wandb_id"],
        rebuild.wandb_id.as_str()
    );
    until_finished(&mut fixture, &scope);
    let after = fixture.cloud();
    assert_eq!(after["runs"].as_object().unwrap().len(), 2, "no third Run");
    assert_eq!(after["runs"][&old], before["runs"][&old]);
    assert_eq!(after["runs"][&rebuild.wandb_id]["state"], "finished");
    assert_eq!(
        after["logs"],
        before["logs"].as_i64().unwrap() + total,
        "each row once"
    );
    let rows = fixture.rows(&scope);
    assert_eq!(rows.len() as i64, total);
    assert_rebuilt_rows(&rows, total);
}

/// One journal, two destinations: W&B and MLflow keep their own cursors and
/// diagnostics, and one failing never holds back the other.
#[test]
fn wandb_and_mlflow_publish_one_journal_with_independent_cursors() {
    use crate::tracking::mlflow::{self, fake};
    let mut fixture = Fixture::new(json!({}));
    let route = json!({"enabled":true,"entity":"fixture","project":"science",
        "mlflow":{"enabled":true,"reason":null,"experiment":"demo"}});
    let bind = |run: &str| {
        let guard = fixture.store.guard("demo", run).unwrap();
        fixture
            .store
            .bind(&guard, &route, &json!({"ml_expd":{"run_id":run}}))
            .unwrap();
        fixture.store.scope(&guard, "attempt-001").unwrap()
    };
    let (train, eval) = (bind("train"), bind("eval"));
    for scope in [&train, &eval] {
        fixture.append(scope, 1..=3);
        fixture.succeed(scope);
    }
    let mlflow_state = |fixture: &Fixture, scope: &str| {
        let parsed = Scope::parse(scope).unwrap();
        let guard = fixture.store.guard("demo", &parsed.run).unwrap();
        fixture
            .store
            .mlflow_state(&guard, scope)
            .unwrap()
            .unwrap_or(json!({}))
    };
    let flag = AtomicBool::new(false);
    let mut session = mlflow::Session::default();

    // MLflow is down: W&B publishes and finishes `train` all the same.
    let down = fake::runtime(&fake::closed_url(), fixture.root.path(), "secret");
    let error = mlflow::publish_ended(&fixture.store, &down, &mut session, &train, None, &flag)
        .unwrap_err();
    mlflow::record_failure(&fixture.store, &train, error);
    until_finished(&mut fixture, &train);
    assert_eq!(fixture.status(&train)["confirmed"], 4);
    assert_eq!(fixture.status(&train)["status"], "FINISHED");
    let state = mlflow_state(&fixture, &train);
    assert_eq!(state["error"], "MLFLOW_UNREACHABLE");
    assert_eq!(state["confirmed"], 0);

    // W&B is broken: MLflow publishes and finishes `eval` all the same.
    fixture.runtime.config.python = fixture.root.path().join("missing-python");
    fixture.restart();
    assert!(fixture.visit(&eval).is_err());
    let server = fake::Server::start("admin", "secret");
    let up = fake::runtime(&server.url, fixture.root.path(), "secret");
    for _ in 0..8 {
        mlflow::publish_ended(&fixture.store, &up, &mut session, &eval, None, &flag).unwrap();
    }
    let state = mlflow_state(&fixture, &eval);
    assert_eq!(state["status"], "MLFLOW_FINISHED");
    assert_eq!(state["confirmed"], 4);
    assert_eq!(fixture.status(&eval)["confirmed"], 0);
    assert_eq!(server.runs().len(), 1);
    assert_eq!(server.runs()[0].name, "eval/attempt-001");
    assert_eq!(server.runs()[0].points.len(), 3);

    // MLflow comes back for `train`; its W&B publication is untouched.
    for _ in 0..8 {
        mlflow::publish_ended(&fixture.store, &up, &mut session, &train, None, &flag).unwrap();
    }
    assert_eq!(mlflow_state(&fixture, &train)["status"], "MLFLOW_FINISHED");
    assert_eq!(server.runs().len(), 2);
    assert_eq!(fixture.status(&train)["status"], "FINISHED");
    assert_eq!(fixture.status(&train)["confirmed"], 4);
}

#[test]
fn a_resumed_run_whose_sdk_step_trails_its_history_keeps_publishing() {
    // After a resume the real SDK can report a `step` behind the Run's history.
    // Rows logged below it are dropped, so publication stalled with
    // WANDB_WRITE_FAILED on every retry.
    let mut fixture = Fixture::new(json!({}));
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=10);
    fixture.session.batch = 4;
    assert_eq!(fixture.visit(&scope).unwrap(), Next::More);
    assert_eq!(fixture.status(&scope)["confirmed"], 4);
    fixture.edit_cloud(|data| data["resume_step_lag"] = json!(3));
    fixture.restart();
    fixture.succeed(&scope);
    until_finished(&mut fixture, &scope);
    assert_eq!(fixture.status(&scope)["status"], "FINISHED");
    assert_eq!(fixture.cloud().get("dropped"), None, "no row is logged below the history");
    fixture.assert_each_record_once(&scope, 10);
}
