"""API routes against the real fixture data (A1 + smoke run)."""

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import ml_exp_server.api.app as api_app
from ml_exp_server.api_contract import CLIENT_PROTOCOL_HEADER
from ml_exp_server.api.app import create_app
from ml_exp_server.api.routes import _attention
from ml_exp_server.schemas import RunIndexRow
from ml_exp_server.schemas import ResearchProject, ServerConfig, ProjectRef
from tests.conftest import FIXTURES


@pytest.fixture
def client(tmp_path):
    exp = tmp_path / "experiments"
    hyp = exp / "research_questions"
    hyp.mkdir(parents=True)
    campaign_file = exp / "campaign_h100.yml"
    campaign_file.write_text(textwrap.dedent("""\
        schema_version: 1
        project: elf
        campaign: fusion-len256-gate-h100-20260711
        runs:
          - run_id: elf-a1-frozen-t5-l256-s42-h100-v1
            research_role: a1
    """))
    (exp / "research_project.yaml").write_text(textwrap.dedent(f"""\
        schema_version: 1
        project: elf
        title: ELF plan-token fusion
        run_roots: [{FIXTURES / 'runs'}]
        campaigns:
          - name: fusion-len256-gate-h100-20260711
            file: {campaign_file}
            role_notes: {{a1: frozen Sentence-T5}}
        research_questions_dir: experiments/research_questions
    """))
    (hyp / "h1.yml").write_text(textwrap.dedent(f"""\
        schema_version: 1
        id: H1
        title: sentence plan viability
        status: OPEN
        summary: test question
        links:
          campaigns: [fusion-len256-gate-h100-20260711]
    """))
    config = ServerConfig(
        index_db=str(tmp_path / "index.sqlite"),
        action_root=str(tmp_path / "actions"),
        # Fixture records have no controller. This test exercises explicit
        # snapshot mode; server defaults are covered separately below.
        collector_enabled=False,
        projects=[ProjectRef(project_file=str(exp / "research_project.yaml"))],
    )
    app = create_app(config)
    with TestClient(app) as test_client:
        yield test_client


A1 = "elf-a1-frozen-t5-l256-s42-h100-v1"


def test_projects_list(client):
    payload = client.get("/api/projects").json()
    assert payload[0]["project"] == "elf"
    assert payload[0]["research_question_count"] == 1
    assert payload[0]["run_counts"].get("RUNNING") == 1  # A1


def test_terminal_snapshot_is_server_read_model(client):
    payload = client.get("/api/terminal/snapshot").json()
    assert payload["projects"][0]["project"] == "elf"
    assert payload["runs"]["elf"][0]["run_id"] == A1
    assert payload["campaign_statuses"]
    assert "loaded_at" in payload
    assert payload["scale"]["projects"] == 1
    assert payload["scale"]["runs"] == len(payload["runs"]["elf"])
    assert payload["scale"]["target_statuses"] == {
        "returned": 0, "total": 0, "limit": 500, "truncated": False,
    }
    filtered = client.get("/api/terminal/snapshot", params={"project": "elf"})
    assert filtered.status_code == 200
    assert filtered.json()["scale"]["project_filter"] == "elf"
    assert client.get(
        "/api/terminal/snapshot", params={"project": "missing"},
    ).status_code == 404


def test_tui_session_endpoints_are_server_owned(client):
    operations = client.get("/api/operations", params={
        "project": "elf", "scope_type": "project", "object_id": "elf",
    })
    assert operations.status_code == 200
    assert operations.json()
    operation_ids = {item["operation"]["operation_id"] for item in operations.json()}
    assert "campaign.create" in operation_ids
    assert "research.recommend" not in operation_ids

    bundle = client.get(f"/api/attempts/elf/{A1}::attempt-001/bundle")
    assert bundle.status_code == 200
    assert set(bundle.json()) == {
        "show", "validation", "metrics", "checkpoints", "artifacts",
    }
    show = bundle.json()["show"]
    assessment = show["failure_assessment"]
    assert assessment["failure_summary"] is None
    assert assessment["diagnostic_evidence"]
    diagnostic = assessment["diagnostic_evidence"][0]
    assert diagnostic["kind"] == "preliminary_failure_classification"
    assert diagnostic["failure_class"] == "unknown"
    assert diagnostic["applicability"] == "NON_APPLICABLE"
    assert diagnostic["source_binding"] == "BOUND_BY_EXACT_ROOT_COLLECTION"
    assert diagnostic["evidence_source"].endswith("/decision.json")
    checkpoint_gate = next(
        gate for gate in bundle.json()["validation"]["gates"]
        if gate["id"] == "attempt.checkpoint_evidence"
    )
    assert checkpoint_gate["status"] == "UNKNOWN"


def test_overview_contains_research_question_and_attention(client):
    payload = client.get("/api/projects/elf/overview").json()
    hyp = payload["research_questions"][0]
    assert hyp["id"] == "H1" and hyp["status"] == "OPEN"
    roles = {r["role"]: r for r in hyp["roles"]}
    assert roles["a1"]["scheduler_state"] == "RUNNING"
    assert roles["a1"]["stale"] is True  # worker evidence hours old
    kinds = {item["kind"] for item in payload["attention"]}
    assert "stale_evidence" in kinds
    assert payload["collector"]["enabled"] is False
    assert payload["campaigns"][0]["lifecycle_state"] == "ACTIVE"


def test_campaign_lifecycle_endpoint(client):
    campaign = "fusion-len256-gate-h100-20260711"
    payload = client.get(f"/api/campaigns/elf/{campaign}").json()
    assert payload["campaign"] == campaign
    assert payload["lifecycle_state"] == "ACTIVE"
    assert payload["validation"]["status"] == "PASS"
    assert "completion" not in payload
    assert payload["runs"][0]["latest_metrics"]["step"] == 3700


def test_research_question_detail_matrix(client):
    payload = client.get("/api/research-questions/elf/H1").json()
    campaign = payload["campaigns"][0]
    role = next(r for r in campaign["roles"] if r["role"] == "a1")
    assert role["role_note"] == "frozen Sentence-T5"
    assert role["evidence"]["worker"]["stale"] is True
    assert role["key_metrics"]["step"] == 3700
    assert role["key_metrics"]["plan_ppl_gap"] > 0
    assert role["evaluation_snapshot"]["latest_metric_complete"]["step"] == 2000
    assert role["evaluation_snapshot"]["latest_metric_complete"]["metric_sources"][
        "plan_ppl_gap"
    ]["step"] == 2000
    assert role["canonical_eval_variant_id"] is None
    assert len(role["eval_variants"]) == 4
    assert role["decision"]["action"] == "OBSERVE"
    assert "failure_class" not in role["decision"]
    assert role["failure_assessment"]["failure_summary"] is None
    assert role["failure_assessment"]["diagnostic_evidence"][0][
        "failure_class"
    ] == "unknown"
    assert all(
        "failure_class" not in str(item.get("decision") or {})
        for item in payload["decision_timeline"]
    )


def test_run_detail_five_layers(client):
    payload = client.get(f"/api/runs/elf/{A1}").json()
    evidence = payload["evidence"]
    assert evidence["scheduler"]["state"] == "RUNNING"
    # The fixture's last scheduler poll is from 2026-07-11; at wall-clock time
    # it is correctly flagged stale (freshness with injected clocks is covered
    # in test_runscan).
    assert evidence["scheduler"]["stale"] is True
    assert evidence["worker"]["stale"] is True
    assert "worker evidence" in evidence["worker"]["stale_reason"]
    assert evidence["model"]["detail"]["step"] == 3700
    assert len(evidence["evaluation"]["detail"]) == 4
    assert payload["provenance"]["seed"] == 42
    assert payload["decision"] == {
        "action": "OBSERVE", "reason": "run is nonterminal",
        "retries_allowed": 0, "retries_used": 0,
    }
    assert all("failure_class" not in attempt["decision"]
               for attempt in payload["attempts"])
    assert payload["failure_assessment"]["failure_summary"] is None
    assert payload["failure_assessment"]["diagnostic_evidence"][0][
        "failure_class"
    ] == "unknown"


def test_terminal_snapshot_plain_json_sanitizes_nested_failure_fields(client):
    payload = client.get("/api/terminal/snapshot").json()
    row = next(item for item in payload["runs"]["elf"] if item["run_id"] == A1)

    assert "failure_class" not in row["decision"]
    assert all("failure_class" not in item["decision"] for item in row["attempts"])
    assert all(
        "failure_class" not in str(item.get("decision") or {})
        for item in row["decision_history"]
    )
    assert row["failure_assessment"]["failure_summary"] is None
    assert row["failure_assessment"]["diagnostic_evidence"][0][
        "failure_class"
    ] == "unknown"


def test_run_metrics_series(client):
    payload = client.get(
        f"/api/runs/elf/{A1}/metrics",
        params={"keys": "train_loss,train_plan_emb_batch_var"},
    ).json()
    assert payload["total_records"] == 37
    assert payload["downsampled"] is False
    first, last = payload["points"][0], payload["points"][-1]
    assert first["step"] == 100 and last["step"] == 3700
    assert first["train_loss"] > last["train_loss"]  # loss went down
    assert "train_plan_emb_batch_var" in first


def test_run_metrics_downsampling(client):
    payload = client.get(f"/api/runs/elf/{A1}/metrics", params={"max_points": 10}).json()
    assert payload["downsampled"] is True
    assert len(payload["points"]) == 10
    assert payload["points"][-1]["step"] == 3700  # latest point kept

    invalid = client.get(f"/api/runs/elf/{A1}/metrics", params={"max_points": 0})
    assert invalid.status_code == 422


def test_run_metrics_reports_missing_keys(client):
    payload = client.get(
        f"/api/runs/elf/{A1}/metrics",
        params={"keys": "train_loss,does_not_exist", "max_points": 5},
    ).json()
    assert payload["missing_keys"] == ["does_not_exist"]
    assert len(payload["points"]) == 5


def test_run_eval_variants_use_the_indexed_coherent_snapshot(client, monkeypatch):
    monkeypatch.setattr(
        "ml_exp_server.ingest.runscan.evaluation_variants",
        lambda *_args, **_kwargs: pytest.fail("run eval must not rescan live files"),
    )
    payload = client.get(f"/api/runs/elf/{A1}/eval").json()
    names = [v["variant"] for v in payload["variants"]]
    assert any("oracle-plan" in n for n in names)
    assert any("shuffled-plan" in n for n in names)
    oracle = next(v for v in payload["variants"] if "oracle-plan" in v["variant"])
    assert oracle["latest"]["step"] == 3700 or oracle["latest"]["step"] >= 2000
    assert oracle["history"][-1] == oracle["latest"]
    assert oracle["history_total"] == oracle["records"]
    assert oracle["history_limit"] == 32
    assert oracle["history_truncated"] is False
    assert oracle["history_omitted_records"] == 0
    complete = payload["evaluation_snapshot"]["latest_metric_complete"]
    assert complete["state"] == "COMPLETE"
    assert complete["metric_sources"]["oracle_plan_ppl"]["variant_id"] == oracle[
        "variant"
    ]


def test_run_events_timeline(client):
    payload = client.get(f"/api/runs/elf/{A1}/events").json()
    events = payload["events"]
    assert events[0]["event"] == "control_attempt_created"
    assert events[-1]["event"] == "scheduler_observed"
    assert events[-1]["payload"]["state"] == "RUNNING"
    assert events[0]["ts"] is not None


def test_404s(client):
    assert client.get("/api/projects/nope/overview").status_code == 404
    assert client.get("/api/research-questions/elf/H9").status_code == 404
    assert client.get("/api/runs/elf/ghost-run").status_code == 404
    missing = client.get("/api/does-not-exist")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "API route not found"}


def test_api_404_contract_is_machine_readable(tmp_path):
    config = ServerConfig(
        index_db=str(tmp_path / "empty-index.sqlite"),
        action_root=str(tmp_path / "actions"),
        projects=[],
    )

    with TestClient(api_app.create_app(config)) as test_client:
        assert test_client.get("/api").json() == {"detail": "API route not found"}
        missing = test_client.get("/api/does-not-exist")
        assert missing.status_code == 404
        assert missing.json() == {"detail": "API route not found"}


def test_api_resources_require_protocol_header_before_dispatch(tmp_path):
    config = ServerConfig(
        index_db=str(tmp_path / "index.sqlite"),
        action_root=str(tmp_path / "actions"),
        collector_enabled=False,
        projects=[],
    )
    with TestClient(create_app(config)) as test_client:
        del test_client.headers[CLIENT_PROTOCOL_HEADER]
        # Health is the one headerless bootstrap exception.
        assert test_client.get("/api/health").status_code == 200
        snapshot = test_client.get("/api/terminal/snapshot")
        assert snapshot.status_code == 426
        assert snapshot.headers["X-ML-Expd-Error-Code"] == (
            "INCOMPATIBLE_API_PROTOCOL"
        )
        mutation = test_client.post(
            "/api/project-lifecycle/unregister-all", json={},
        )
        assert mutation.status_code == 426
        assert mutation.headers["X-ML-Expd-Error-Code"] == (
            "INCOMPATIBLE_API_PROTOCOL"
        )


def test_app_factory_has_no_durable_side_effect_before_lifespan(tmp_path):
    index_path = tmp_path / "index.sqlite"
    action_root = tmp_path / "actions"
    registry_root = tmp_path / "projects"
    empty = ServerConfig(
        index_db=str(index_path), action_root=str(action_root),
        project_registry_root=str(registry_root), projects=[],
    )

    app = create_app(empty)
    assert app.state.runtime is None
    assert not index_path.exists()
    assert not action_root.exists()
    assert not (registry_root / "registry.json").exists()

    project_file = tmp_path / "research_project.yaml"
    project_file.write_text(
        "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\n",
        encoding="utf-8",
    )
    seeded = empty.model_copy(update={
        "projects": [ProjectRef(project_file=str(project_file))],
    })
    with TestClient(create_app(seeded)) as test_client:
        records = test_client.get("/api/project-lifecycle").json()["projects"]
        assert [record["project"] for record in records] == ["demo"]


def test_health_negotiates_versioned_protocol_and_native_bearer_auth(tmp_path):
    token = "daemon-test-token-" + "x" * 32
    token_file = tmp_path / "daemon.token"
    token_file.write_text(token + "\n", encoding="utf-8")
    token_file.chmod(0o600)
    config = ServerConfig(
        index_db=str(tmp_path / "index.sqlite"),
        action_root=str(tmp_path / "actions"),
        collector_enabled=False,
        http_auth={"bearer_token_file": str(token_file)},
        projects=[],
    )

    with TestClient(create_app(config)) as authenticated:
        missing = authenticated.get("/api/health")
        assert missing.status_code == 401
        assert missing.headers["X-ML-Expd-Error-Code"] == "AUTHENTICATION_REQUIRED"
        headers = {
            "Authorization": f"Bearer {token}",
            "X-ML-Expd-Client-Protocol": "1",
        }
        health = authenticated.get("/api/health", headers=headers)
        payload = health.json()
        assert health.status_code == 200
        assert health.headers["X-ML-Expd-Protocol"] == "1"
        assert payload["api_protocol_version"] == 1
        assert payload["authentication"] == "bearer"
        assert "actions.v1" in payload["capabilities"]
        assert payload["openapi_path"] == "/api/v1/openapi.json"
        assert authenticated.get(payload["openapi_path"], headers=headers).status_code == 200

        incompatible = authenticated.get("/api/health", headers={
            **headers, "X-ML-Expd-Client-Protocol": "2",
        })
        assert incompatible.status_code == 426
        assert incompatible.headers["X-ML-Expd-Error-Code"] == "INCOMPATIBLE_API_PROTOCOL"


def test_collector_status_endpoint(client):
    payload = client.get("/api/collector/status").json()
    assert payload["enabled"] is False
    assert payload["requested"] is False
    assert payload["owner"] is False
    assert payload["last_error"] is None
    assert payload["runs"] == []


def test_server_enables_collector_by_default(tmp_path):
    config = ServerConfig(index_db=str(tmp_path / "index.sqlite"), projects=[])
    with TestClient(create_app(config)) as test_client:
        payload = test_client.get("/api/collector/status").json()
    assert payload["enabled"] is True
    assert payload["requested"] is True
    assert payload["owner"] is True


def test_second_server_fails_startup_when_workspace_lease_is_owned(tmp_path):
    config = ServerConfig(index_db=str(tmp_path / "index.sqlite"), projects=[])
    with TestClient(create_app(config)) as first:
        assert first.get("/api/collector/status").json()["owner"] is True
        with pytest.raises(RuntimeError, match="another ml-expd process owns"):
            with TestClient(create_app(config)):
                pass
        assert first.get("/api/health").status_code == 200


def test_lease_loss_fails_before_runtime_construction(monkeypatch, tmp_path):
    indexed = []
    initialized = []

    class StandbyLease:
        def __init__(self, path):
            pass

        def acquire(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(api_app, "CollectorLease", StandbyLease)
    monkeypatch.setattr(api_app, "index_project", lambda index, project: indexed.append(project.project))
    config = ServerConfig(
        index_db=str(tmp_path / "index.sqlite"),
        action_root=str(tmp_path / "actions"),
        observability={
            "credential_root": str(tmp_path / "credentials"),
            "log_archive_root": str(tmp_path / "logs"),
        },
        projects=[],
    )
    project = ResearchProject(project="demo", title="Demo", run_roots=[])
    app = create_app(config, poll=False, projects=[project])
    app.state.runtime_initializers.append(lambda runtime: initialized.append(runtime))

    with pytest.raises(RuntimeError, match="a second daemon cannot start safely"):
        with TestClient(app):
            pass

    assert indexed == []
    assert initialized == []
    assert not (tmp_path / "index.sqlite").exists()
    assert not (tmp_path / "index.observability.sqlite").exists()
    assert not (tmp_path / "index.projects").exists()
    assert not (tmp_path / "actions").exists()
    assert not (tmp_path / "credentials").exists()
    assert not (tmp_path / "logs").exists()


def test_snapshot_daemons_still_require_exclusive_workspace_lease(tmp_path):
    config = ServerConfig(index_db=str(tmp_path / "index.sqlite"), projects=[])
    with TestClient(create_app(config, poll=False)) as first:
        assert first.post("/api/project-lifecycle/unregister-all", json={}).status_code == 200
        with pytest.raises(RuntimeError, match="another ml-expd process owns"):
            with TestClient(create_app(config, poll=False)):
                pass


def test_startup_failure_releases_workspace_lease(tmp_path):
    config = ServerConfig(index_db=str(tmp_path / "index.sqlite"), projects=[])
    broken = create_app(config)

    def fail_recovery():
        raise RuntimeError("startup recovery failed")

    broken.state.runtime_initializers.append(
        lambda runtime: setattr(runtime.action_store, "list_all", fail_recovery)
    )
    with pytest.raises(RuntimeError, match="startup recovery failed"):
        with TestClient(broken):
            pass

    with TestClient(create_app(config)) as recovered:
        assert recovered.get("/api/collector/status").json()["owner"] is True


def test_thread_start_failure_releases_workspace_lease(tmp_path, monkeypatch):
    config = ServerConfig(index_db=str(tmp_path / "index.sqlite"), projects=[])
    original_start = api_app.threading.Thread.start
    failed_once = False

    def fail_publisher_once(thread):
        nonlocal failed_once
        if thread.name == "wandb-publisher" and not failed_once:
            failed_once = True
            raise RuntimeError("publisher thread failed to start")
        return original_start(thread)

    monkeypatch.setattr(api_app.threading.Thread, "start", fail_publisher_once)
    with pytest.raises(RuntimeError, match="publisher thread failed"):
        with TestClient(create_app(config)):
            pass

    with TestClient(create_app(config)) as recovered:
        assert recovered.get("/api/collector/status").json()["owner"] is True


def test_attention_includes_failed_runs_and_collector_errors():
    failed = RunIndexRow(
        project="elf", run_id="failed", run_dir="/tmp/failed",
        scheduler_state="FAILED", decision={"failure_class": "unknown"},
    )
    plain = failed.model_copy(update={"run_id": "plain", "decision": {}})
    collector = SimpleNamespace(run_id="failed", last_error="scheduler unavailable")
    items = _attention([failed, plain], [collector], {
        "failed": {"failure_summary": {
            "failure_class": "OOM", "applicability": "APPLICABLE",
        }},
        "plain": {"failure_summary": None, "diagnostic_evidence": [{
            "failure_class": "unknown", "applicability": "NON_APPLICABLE",
        }]},
    })
    assert [item["kind"] for item in items].count("failed_run") == 2
    assert any("OOM" in item["detail"] for item in items)
    assert all("unknown" not in item["detail"] for item in items)
    assert any(item["kind"] == "collector_error" for item in items)
