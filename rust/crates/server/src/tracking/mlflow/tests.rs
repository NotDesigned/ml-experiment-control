use super::fake::{self, Server};
use super::*;
use ml_exp_core::records::{Event, normalized_metric};
use serde_json::json;

const SEED: &str = "0123456789abcdef0123456789abcdef";
const PASSWORD: &str = "secret";

fn mlflow_route() -> Value {
    json!({"enabled":false,"reason":"CREDENTIALS_NOT_CONFIGURED","mlflow":{"enabled":true,"reason":null,"experiment":"demo"}})
}

struct Fixture {
    root: tempfile::TempDir,
    store: Store,
    runtime: Runtime,
    session: Session,
    flag: AtomicBool,
    server: Server,
}
impl Fixture {
    fn new() -> Self {
        let server = Server::start("admin", PASSWORD);
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), SEED).unwrap();
        let runtime = fake::runtime(&server.url, root.path(), PASSWORD);
        Self {
            root,
            store,
            runtime,
            session: Session::default(),
            flag: AtomicBool::new(false),
            server,
        }
    }
    fn bind(&self, run: &str, route: &Value) -> String {
        let guard = self.store.guard("demo", run).unwrap();
        self.store
            .bind(
                &guard,
                route,
                &json!({"ml_expd":{"run_id":run,"source_id":"src-1","seed":42,
                    "parameters":{"lr":0.001,"opt":{"beta1":0.9,"name":"adamw"},"bad@key":1}}}),
            )
            .unwrap();
        self.store.scope(&guard, "attempt-001").unwrap()
    }
    fn scope(&self, run: &str) -> String {
        self.bind(run, &mlflow_route())
    }
    fn events(&self, scope: &str, events: Vec<Event>) {
        let parsed = Scope::parse(scope).unwrap();
        let guard = self.store.guard("demo", &parsed.run).unwrap();
        self.store.append(&guard, scope, &events).unwrap();
    }
    fn append(&self, scope: &str, steps: std::ops::RangeInclusive<u64>) {
        self.events(
            scope,
            steps
                .map(|step| Event {
                    event_id: format!("m{step}"),
                    payload: normalized_metric(
                        &json!({"name":"loss","unit":"nats/token","step":step,"value":step as f64 / 4.0}),
                        &json!({}),
                    )
                    .unwrap(),
                })
                .collect(),
        );
    }
    fn end(&self, scope: &str, state: &str) {
        self.events(
            scope,
            vec![Event {
                event_id: "terminal".into(),
                payload: json!({"kind":"lifecycle","data":{"phase":"METRICS_DRAINED","state":state}}),
            }],
        );
    }
    fn visit(&mut self, scope: &str) -> Result<Next> {
        publish_ended(
            &self.store,
            &self.runtime,
            &mut self.session,
            scope,
            None,
            &self.flag,
        )
    }
    fn until_idle(&mut self, scope: &str) {
        for _ in 0..64 {
            if self.visit(scope).unwrap() == Next::Idle {
                return;
            }
        }
        panic!("MLflow publication did not settle: {}", self.state(scope));
    }
    fn state(&self, scope: &str) -> Value {
        let parsed = Scope::parse(scope).unwrap();
        let guard = self.store.guard("demo", &parsed.run).unwrap();
        self.store
            .mlflow_state(&guard, scope)
            .unwrap()
            .unwrap_or(json!({}))
    }
    fn view(&self, scope: &str) -> Value {
        let parsed = Scope::parse(scope).unwrap();
        let guard = self.store.guard("demo", &parsed.run).unwrap();
        self.store
            .run_scopes(&guard)
            .unwrap()
            .into_iter()
            .find(|s| s["id"] == scope)
            .unwrap()
    }
    fn total(&self, scope: &str) -> i64 {
        self.view(scope)["total"].as_i64().unwrap()
    }
    /// Every point the journal says MLflow must hold for `key`, in MLflow's form.
    fn expected(&self, scope: &str, key: &str) -> BTreeSet<(i64, i64, u64)> {
        let parsed = Scope::parse(scope).unwrap();
        let guard = self.store.guard("demo", &parsed.run).unwrap();
        let base = self.state(scope)["timestamp_base"].as_i64().unwrap();
        let mut points = BTreeSet::new();
        for event in self.store.events(&guard, scope, 0, 100_000).unwrap() {
            for item in event.payload["observations"]
                .as_array()
                .into_iter()
                .flatten()
            {
                let name = if item["summary"] == true {
                    format!("summary/{}", item["name"].as_str().unwrap())
                } else {
                    item["name"].as_str().unwrap().to_owned()
                };
                if name == key && item["status"] == "VALID" {
                    points.insert((
                        base + event.seq,
                        item["step"].as_i64().unwrap_or(0),
                        item["value"].as_f64().unwrap().to_bits(),
                    ));
                }
            }
        }
        points
    }
    /// What MLflow returns for `key`, read back over REST.
    fn read_back(&self, scope: &str, key: &str) -> Vec<(i64, i64, u64)> {
        let run = self.state(scope)["run_id"].as_str().unwrap().to_owned();
        self.runtime
            .client
            .history(&run, key)
            .unwrap()
            .iter()
            .map(|point| {
                (
                    point["timestamp"].as_i64().unwrap(),
                    point["step"].as_i64().unwrap(),
                    point["value"].as_f64().unwrap().to_bits(),
                )
            })
            .collect()
    }
}

#[test]
fn a_finished_attempt_is_published_once_and_reads_back_exactly() {
    let mut fixture = Fixture::new();
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=5);
    fixture.events(
        &scope,
        vec![Event {
            event_id: "result".into(),
            payload: normalized_metric(
                &json!({"name":"loss","unit":"nats/token","value":0.125,"summary":true}),
                &json!({}),
            )
            .unwrap(),
        }],
    );
    fixture.end(&scope, "SUCCEEDED");
    fixture.until_idle(&scope);

    let state = fixture.state(&scope);
    assert_eq!(state["status"], "MLFLOW_FINISHED");
    assert_eq!(state["terminal_status"], "FINISHED");
    assert_eq!(state["confirmed"], fixture.total(&scope));
    assert!(state.get("error").is_none());
    let runs = fixture.server.runs();
    assert_eq!(runs.len(), 1);
    let run = &runs[0];
    assert_eq!(run.name, "train/attempt-001");
    assert_eq!(run.status, "FINISHED");
    assert_eq!(run.id, state["run_id"]);
    assert_eq!(run.start_time, state["timestamp_base"]);
    for (key, value) in [
        ("ml_expd.project", "demo"),
        ("ml_expd.run", "train"),
        ("ml_expd.attempt", "attempt-001"),
        ("ml_expd.scope", "demo/train/attempt-001"),
        ("ml_expd.source_id", "src-1"),
    ] {
        assert_eq!(run.tags[key], value, "{key}");
    }
    assert_eq!(run.params["lr"], "0.001");
    assert_eq!(run.params["opt.beta1"], "0.9");
    assert_eq!(run.params["opt.name"], "adamw");
    assert_eq!(run.params["ml_expd.seed"], "42");
    assert!(!run.params.contains_key("bad@key"));
    assert_eq!(state["skipped_params"], json!(["bad@key"]));

    let loss = fixture.read_back(&scope, "loss");
    assert_eq!(loss.len(), 5);
    assert_eq!(
        loss.into_iter().collect::<BTreeSet<_>>(),
        fixture.expected(&scope, "loss")
    );
    assert_eq!(
        fixture.read_back(&scope, "summary/loss"),
        fixture
            .expected(&scope, "summary/loss")
            .into_iter()
            .collect::<Vec<_>>()
    );
    assert_eq!(fixture.read_back(&scope, "summary/loss").len(), 1);

    let view = fixture.view(&scope)["mlflow"].clone();
    assert_eq!(view["status"], "MLFLOW_FINISHED");
    assert_eq!(view["pending"], 0);
    assert_eq!(view["experiment"], "demo");
    assert!(
        view["url"]
            .as_str()
            .unwrap()
            .ends_with(&format!("/#/experiments/1/runs/{}", run.id))
    );
    // A finished scope is no longer a candidate and costs no request.
    assert!(!candidate_scopes(&fixture.store).unwrap().contains(&scope));
    let calls = fixture.server.with(|data| data.calls.values().sum::<u64>());
    assert_eq!(fixture.visit(&scope).unwrap(), Next::Idle);
    assert_eq!(
        fixture.server.with(|data| data.calls.values().sum::<u64>()),
        calls
    );
}

#[test]
fn a_batch_replayed_after_a_lost_response_adds_nothing() {
    let mut fixture = Fixture::new();
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=3);
    fixture.until_idle(&scope);
    assert_eq!(fixture.state(&scope)["status"], "MLFLOW_SYNCED");
    fixture.append(&scope, 4..=6);
    fixture.server.with(|data| {
        data.drop_after_commit.insert("runs/log-batch".into(), 1);
    });
    let before = fixture.server.calls("runs/log-batch");
    let error = fixture.visit(&scope).unwrap_err();
    assert_eq!(error.code(), "MLFLOW_UNREACHABLE");
    record_failure(&fixture.store, &scope, error);
    let state = fixture.state(&scope);
    assert_eq!(state["confirmed"], 3, "an uncertain batch never advances");
    assert_eq!(state["status"], "MLFLOW_RETRY_PENDING");
    assert_eq!(state["error"], "MLFLOW_UNREACHABLE");
    assert_eq!(state["failures"], 1);
    // MLflow committed the batch whose response was lost.
    let run = state["run_id"].as_str().unwrap().to_owned();
    let committed = fixture.server.with(|data| data.runs[&run].points.len());
    assert_eq!(committed, 6);

    fixture.until_idle(&scope);
    let state = fixture.state(&scope);
    assert_eq!(state["confirmed"], 6);
    assert_eq!(state["status"], "MLFLOW_SYNCED");
    assert!(state.get("error").is_none());
    assert_eq!(fixture.server.calls("runs/log-batch"), before + 2);
    // The same deterministic batch again: the stored points did not change.
    assert_eq!(fixture.server.with(|data| data.runs[&run].points.len()), 6);
    assert_eq!(
        fixture
            .read_back(&scope, "loss")
            .into_iter()
            .collect::<BTreeSet<_>>(),
        fixture.expected(&scope, "loss")
    );
}

#[test]
fn the_run_is_created_once_across_restarts_and_a_lost_create_response() {
    let mut fixture = Fixture::new();
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=2);
    fixture.server.with(|data| {
        data.drop_after_commit.insert("runs/create".into(), 1);
    });
    assert_eq!(
        fixture.visit(&scope).unwrap_err(),
        Error::Unreachable,
        "the created run's response is lost"
    );
    assert!(fixture.state(&scope).get("run_id").is_none());
    // A restart forgets everything in memory.
    fixture.session = Session::default();
    fixture.runtime = fake::runtime(&fixture.server.url, fixture.root.path(), PASSWORD);
    fixture.until_idle(&scope);
    let run = fixture.state(&scope)["run_id"].as_str().unwrap().to_owned();
    assert_eq!(fixture.server.runs().len(), 1);
    assert_eq!(fixture.server.runs()[0].id, run);
    assert_eq!(fixture.server.calls("runs/create"), 1);

    fixture.session = Session::default();
    fixture.append(&scope, 3..=4);
    fixture.end(&scope, "SUCCEEDED");
    fixture.until_idle(&scope);
    assert_eq!(fixture.server.runs().len(), 1);
    assert_eq!(fixture.server.calls("runs/create"), 1);
    assert_eq!(fixture.state(&scope)["run_id"], run);
    assert_eq!(fixture.read_back(&scope, "loss").len(), 4);
}

#[test]
fn the_terminal_status_follows_the_wandb_rule() {
    for (run, end, provider, status) in [
        ("ok", Some("SUCCEEDED"), None, "FINISHED"),
        ("bad", Some("FAILED"), None, "FAILED"),
        ("cancel", Some("CANCELLED"), None, "FAILED"),
        // Preempted or killed: no METRICS_DRAINED, but the provider proved the end.
        ("killed", None, Some("PREEMPTED"), "FAILED"),
        ("live", None, None, "RUNNING"),
    ] {
        let mut fixture = Fixture::new();
        let scope = fixture.scope(run);
        fixture.append(&scope, 1..=2);
        if let Some(end) = end {
            fixture.end(&scope, end);
        }
        for _ in 0..8 {
            publish_ended(
                &fixture.store,
                &fixture.runtime,
                &mut fixture.session,
                &scope,
                provider,
                &fixture.flag,
            )
            .unwrap();
        }
        assert_eq!(fixture.server.runs()[0].status, status, "{run}");
        let state = fixture.state(&scope);
        if status == "RUNNING" {
            assert_eq!(state["status"], "MLFLOW_SYNCED");
            assert!(candidate_scopes(&fixture.store).unwrap().contains(&scope));
        } else {
            assert_eq!(state["status"], "MLFLOW_FINISHED");
            assert_eq!(state["terminal_status"], status);
        }
    }
}

#[test]
fn a_derived_batch_after_the_end_lands_in_the_same_run() {
    use sha2::{Digest, Sha256};
    let mut fixture = Fixture::new();
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=3);
    fixture.end(&scope, "SUCCEEDED");
    fixture.until_idle(&scope);
    assert_eq!(fixture.state(&scope)["status"], "MLFLOW_FINISHED");
    let text = "{\"name\":\"eval/fid\",\"value\":3.0,\"unit\":\"fid\",\"step\":1}\n\
                {\"name\":\"eval/fid\",\"value\":2.5,\"unit\":\"fid\",\"step\":2}\n";
    let observations =
        ml_exp_core::records::derived_observations(text.as_bytes(), &json!({})).unwrap();
    let sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
    {
        let guard = fixture.store.guard("demo", "train").unwrap();
        fixture
            .store
            .append_derived(
                &guard,
                &scope,
                &sha256,
                "2026-11-07T00:00:00Z",
                &observations,
            )
            .unwrap();
    }
    assert!(candidate_scopes(&fixture.store).unwrap().contains(&scope));
    assert_eq!(fixture.view(&scope)["mlflow"]["pending"], 2);
    fixture.until_idle(&scope);
    let state = fixture.state(&scope);
    assert_eq!(state["confirmed"], fixture.total(&scope));
    assert_eq!(state["status"], "MLFLOW_FINISHED");
    assert_eq!(fixture.server.runs().len(), 1);
    assert_eq!(fixture.server.runs()[0].status, "FINISHED");
    assert_eq!(fixture.server.calls("runs/update"), 2);
    let fid = fixture.read_back(&scope, "eval/fid");
    assert_eq!(
        fid.into_iter().collect::<BTreeSet<_>>(),
        fixture.expected(&scope, "eval/fid")
    );
    assert_eq!(fixture.read_back(&scope, "loss").len(), 3);
    assert!(!candidate_scopes(&fixture.store).unwrap().contains(&scope));
}

#[test]
fn an_authentication_failure_is_shown_and_clears_once_fixed() {
    let mut fixture = Fixture::new();
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=2);
    let good = std::mem::replace(
        &mut fixture.runtime,
        fake::runtime(&fixture.server.url, fixture.root.path(), "wrong"),
    );
    let error = fixture.visit(&scope).unwrap_err();
    assert_eq!(error, Error::AuthFailed);
    record_failure(&fixture.store, &scope, error);
    let view = fixture.view(&scope)["mlflow"].clone();
    assert_eq!(view["status"], "MLFLOW_RETRY_PENDING");
    assert_eq!(view["error"], "MLFLOW_AUTH_FAILED");
    assert_eq!(view["pending"], 2);
    assert!(fixture.server.runs().is_empty());
    fixture.runtime = good;
    fixture.until_idle(&scope);
    let view = fixture.view(&scope)["mlflow"].clone();
    assert_eq!(view["status"], "MLFLOW_SYNCED");
    assert!(view.get("error").is_none());
    assert_eq!(view["pending"], 0);
}

#[test]
fn unreachable_failing_and_deleted_destinations_have_their_own_codes() {
    let mut fixture = Fixture::new();
    let scope = fixture.scope("train");
    fixture.append(&scope, 1..=2);
    let up = std::mem::replace(
        &mut fixture.runtime,
        fake::runtime(&fake::closed_url(), fixture.root.path(), PASSWORD),
    );
    assert_eq!(
        fixture.visit(&scope).unwrap_err().code(),
        "MLFLOW_UNREACHABLE"
    );
    fixture.runtime = up;
    fixture.server.with(|data| data.fail_with = Some(503));
    assert_eq!(
        fixture.visit(&scope).unwrap_err().code(),
        "MLFLOW_SERVER_ERROR"
    );
    fixture.server.with(|data| data.fail_with = None);
    fixture.until_idle(&scope);
    // A person deletes the run: it is never recreated, and the scope stops.
    fixture.server.with(|data| {
        for run in data.runs.values_mut() {
            run.lifecycle = "deleted".into();
        }
    });
    fixture.append(&scope, 3..=3);
    assert_eq!(fixture.visit(&scope).unwrap(), Next::Idle);
    let state = fixture.state(&scope);
    assert_eq!(state["status"], "MLFLOW_STOPPED");
    assert_eq!(state["error"], "MLFLOW_RUN_MISSING");
    assert_eq!(state["confirmed"], 2);
    assert!(!candidate_scopes(&fixture.store).unwrap().contains(&scope));
    assert_eq!(fixture.server.calls("runs/create"), 1);
    // A deleted experiment is reported, not recreated.
    let other = fixture.scope("other");
    fixture.append(&other, 1..=1);
    fixture.session = Session::default();
    fixture
        .server
        .with(|data| data.experiments[0].2 = "deleted".into());
    assert_eq!(
        fixture.visit(&other).unwrap_err().code(),
        "MLFLOW_EXPERIMENT_DELETED"
    );
}

#[test]
fn without_an_mlflow_route_there_is_no_mlflow_work() {
    let fixture = Fixture::new();
    let scope = fixture.bind(
        "train",
        &json!({"enabled":false,"reason":"CREDENTIALS_NOT_CONFIGURED"}),
    );
    fixture.append(&scope, 1..=2);
    assert!(candidate_scopes(&fixture.store).unwrap().is_empty());
    assert!(fixture.view(&scope).get("mlflow").is_none());
    let disabled = fixture.bind(
        "off",
        &json!({"enabled":false,"reason":"CREDENTIALS_NOT_CONFIGURED","mlflow":{"enabled":false,"reason":"PROJECT_NOT_SELECTED","experiment":"demo"}}),
    );
    fixture.append(&disabled, 1..=2);
    assert!(candidate_scopes(&fixture.store).unwrap().is_empty());
    assert!(fixture.view(&disabled).get("mlflow").is_none());
    assert_eq!(fixture.server.with(|data| data.calls.len()), 0);

    // No `mlflow` server settings: routes are exactly what they were before.
    let settings = ml_exp_core::tracking::RoutingSettings::default();
    let route = ml_exp_core::tracking::resolve(None, &settings, "demo");
    assert!(route.get("mlflow").is_none());
    let settings = ml_exp_core::tracking::RoutingSettings {
        mlflow: Some(MlflowRouting {
            enabled: true,
            experiment_prefix: "ml-expd/".into(),
            projects: Some(["demo".to_owned()].into()),
        }),
        ..Default::default()
    };
    assert_eq!(
        ml_exp_core::tracking::resolve(None, &settings, "demo")["mlflow"],
        json!({"enabled":true,"reason":null,"experiment":"ml-expd/demo"})
    );
    assert_eq!(
        ml_exp_core::tracking::resolve(None, &settings, "other")["mlflow"]["reason"],
        "PROJECT_NOT_SELECTED"
    );
    assert!(
        ml_exp_core::tracking::resolve_preparation(None, &settings, "demo")
            .get("mlflow")
            .is_none()
    );
}

#[test]
fn the_tracking_file_reads_both_destinations_and_stays_backward_compatible() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let key = root.path().join("wandb.key");
    let write = |name: &str, value: Value| {
        let path = root.path().join(name);
        std::fs::write(&path, value.to_string()).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        path
    };
    let wandb = json!({"schema_version":1,"python":"/usr/bin/python3","api_key_file":key});
    let settings = super::super::Settings::load(&write("old.json", wandb.clone())).unwrap();
    assert!(settings.wandb.is_some());
    assert!(settings.mlflow.is_none());

    let mut both = wandb.clone();
    both["mlflow"] = json!({"url":"https://mlflow.example:5000/","username":"admin","password_file":"/etc/ml-exp/mlflow.password","experiment_prefix":"ml-expd/"});
    let path = write("both.json", both);
    let settings = super::super::Settings::load(&path).unwrap();
    assert!(settings.wandb.is_some());
    let mlflow = settings.mlflow.unwrap();
    assert_eq!(mlflow.base().unwrap(), "https://mlflow.example:5000");
    assert!(mlflow.enabled);
    // The W&B loader alone still accepts the file.
    assert!(super::super::Config::load(&path).is_ok());

    let only = write(
        "only.json",
        json!({"schema_version":1,"mlflow":{"url":"http://127.0.0.1:5000"}}),
    );
    let settings = super::super::Settings::load(&only).unwrap();
    assert!(settings.wandb.is_none());
    assert!(settings.mlflow.is_some());

    for bad in [
        json!({"url":"http://mlflow.example:5000","username":"admin","password_file":"/p"}),
        json!({"url":"https://admin:pw@mlflow.example"}),
        json!({"url":"ftp://mlflow.example"}),
        json!({"url":"https://mlflow.example","username":"admin"}),
        json!({"url":"https://mlflow.example","password_file":"relative"}),
        json!({"url":"https://mlflow.example","unknown":true}),
        json!({"url":"https://mlflow.example","projects":["../x"]}),
        json!({"url":"https://mlflow.example","timeout_seconds":0}),
    ] {
        assert!(Config::from_value(bad.clone()).is_err(), "{bad}");
    }
    assert!(
        Config::from_value(
            json!({"url":"https://mlflow.example","projects":["demo"],"enabled":false})
        )
        .is_ok()
    );
}

#[test]
fn mapping_keeps_mlflow_limits_and_skips_what_does_not_fit() {
    for key in ["loss", "train/loss", "eval acc", "a-b_c.d", "summary/loss"] {
        assert!(mapping::valid_key(key), "{key}");
    }
    let long = "k".repeat(251);
    for key in [
        "",
        "/a",
        "a/",
        "a//b",
        "./a",
        "a/../b",
        "a@b",
        "a:b",
        long.as_str(),
    ] {
        assert!(!mapping::valid_key(key), "{key}");
    }
    let observation = |name: &str, step: u64| json!({"name":name,"value":step as f64,"unit":"u","status":"VALID","step":step});
    let event = |seq: i64, observations: Vec<Value>| crate::records::EventRow {
        seq,
        event_id: format!("e{seq}"),
        payload: json!({"kind":"metrics","observations":observations}),
    };
    // One event with more points than one batch may hold is split, completed by its last part.
    let wide: Vec<Value> = (0..2500)
        .map(|i| observation(&format!("m{i}"), 1))
        .collect();
    let batch = mapping::next_batch(
        &[event(1, wide), event(2, vec![observation("loss", 1)])],
        0,
        1000,
    );
    assert_eq!(
        batch.chunks.iter().map(Vec::len).collect::<Vec<_>>(),
        [1000, 1000, 500]
    );
    assert_eq!(batch.through, 1);
    // Batches stop before an event that would overflow; repeats inside an event collapse.
    let events: Vec<_> = (1..=1500)
        .map(|seq| event(seq, vec![observation("loss", 1), observation("loss", 1)]))
        .collect();
    let batch = mapping::next_batch(&events, 0, 1000);
    assert_eq!(batch.chunks.len(), 1);
    assert_eq!(batch.chunks[0].len(), 1000);
    assert_eq!(batch.through, 1000);
    assert_eq!(batch.chunks[0][0]["timestamp"], 1001);
    // Invalid names, failed values and lifecycle events are not points.
    let mixed = vec![
        event(
            1,
            vec![
                observation("bad@name", 1),
                json!({"name":"x","value":null,"status":"FAILED"}),
            ],
        ),
        crate::records::EventRow {
            seq: 2,
            event_id: "life".into(),
            payload: json!({"kind":"lifecycle","data":{"phase":"TRAINING"}}),
        },
    ];
    let batch = mapping::next_batch(&mixed, 0, 0);
    assert!(batch.chunks.is_empty());
    assert_eq!(batch.through, 2);
    assert_eq!(batch.skipped.count, 1);
    assert!(batch.skipped.names.contains("bad@name"));
    let (mut params, skipped) = mapping::params(
        &json!({"ml_expd":{"parameters":{"lr":0.1,"list":[1,2],"long":"x".repeat(6001)}}}),
    );
    params.sort();
    assert_eq!(
        params,
        [
            ("list".to_owned(), "[1,2]".to_owned()),
            ("lr".to_owned(), "0.1".to_owned())
        ]
    );
    assert_eq!(skipped, ["long"]);
}
