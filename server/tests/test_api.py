"""API routes against the real fixture data (A1 + smoke run)."""

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import ml_exp_server.api.app as api_app
from ml_exp_server.api.app import create_app
from ml_exp_server.api.routes import _attention, _contract_view, _match_check
from ml_exp_server.schemas import RunIndexRow
from ml_exp_server.schemas import ServerConfig, ProjectRef
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
        agent_root=str(tmp_path / "agents"),
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


def test_tui_session_endpoints_are_server_owned(client):
    operations = client.get("/api/operations", params={
        "project": "elf", "scope_type": "project", "object_id": "elf",
    })
    assert operations.status_code == 200
    assert operations.json()
    assert {item["operation"]["operation_id"] for item in operations.json()} >= {
        "research.recommend", "campaign.create",
    }

    bundle = client.get(f"/api/attempts/elf/{A1}::attempt-001/bundle")
    assert bundle.status_code == 200
    assert set(bundle.json()) == {
        "show", "validation", "metrics", "checkpoints", "artifacts",
    }


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
    assert payload["campaigns"][0]["lifecycle_state"] == "INVALID"


def test_campaign_lifecycle_endpoint(client):
    campaign = "fusion-len256-gate-h100-20260711"
    payload = client.get(f"/api/campaigns/elf/{campaign}").json()
    assert payload["campaign"] == campaign
    assert payload["lifecycle_state"] == "INVALID"
    assert payload["validation"]["status"] == "FAIL"
    assert next(gate for gate in payload["validation"]["gates"]
                if gate["name"] == "research_contract")["status"] == "FAIL"
    assert payload["completion"]["ready"] is False


def test_research_question_detail_matrix(client):
    payload = client.get("/api/research-questions/elf/H1").json()
    campaign = payload["campaigns"][0]
    assert campaign["contract"] is None  # h100 campaign has no inline contract
    assert campaign["contract_source"] is None
    assert campaign["match_check"]["status"] == "NO_CONTRACT"
    role = next(r for r in campaign["roles"] if r["role"] == "a1")
    assert role["role_note"] == "frozen Sentence-T5"
    assert role["evidence"]["worker"]["stale"] is True
    assert role["key_metrics"]["step"] == 3700
    assert "plan_ppl_gap" not in role["key_metrics"]
    assert role["canonical_eval_variant_id"] is None
    assert len(role["eval_variants"]) == 4
    assert role["decision"]["action"] == "OBSERVE"


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


def test_run_eval_variants(client):
    payload = client.get(f"/api/runs/elf/{A1}/eval").json()
    names = [v["variant"] for v in payload["variants"]]
    assert any("oracle-plan" in n for n in names)
    assert any("shuffled-plan" in n for n in names)
    oracle = next(v for v in payload["variants"] if "oracle-plan" in v["variant"])
    assert oracle["latest"]["step"] == 3700 or oracle["latest"]["step"] >= 2000


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
        agent_root=str(tmp_path / "agents"),
        action_root=str(tmp_path / "actions"),
        projects=[],
    )

    with TestClient(api_app.create_app(config)) as test_client:
        assert test_client.get("/api").json() == {"detail": "API route not found"}
        missing = test_client.get("/api/does-not-exist")
        assert missing.status_code == 404
        assert missing.json() == {"detail": "API route not found"}


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


def test_second_server_reports_collector_lease_loss(tmp_path):
    config = ServerConfig(index_db=str(tmp_path / "index.sqlite"), projects=[])
    with TestClient(create_app(config)) as first, TestClient(create_app(config)) as second:
        assert first.get("/api/collector/status").json()["owner"] is True
        payload = second.get("/api/collector/status").json()
    assert payload["enabled"] is False
    assert payload["requested"] is True
    assert payload["owner"] is False
    assert "owns this workspace collector" in payload["last_error"]


def test_frozen_contract_wins_and_match_check_stays_pending_without_block_decision():
    contract = {"schema_version": 1, "question": "frozen", "required_roles": ["a0"]}
    row = RunIndexRow(
        project="elf", run_id="run-a", run_dir="/tmp/run-a", role="a0",
        research_contract=contract, research_contract_source="manifest",
    )

    view = _contract_view([row], {**contract, "question": "edited"})
    match = _match_check([row], view)

    assert view["contract"]["question"] == "frozen"
    assert view["contract_source"] == "frozen_manifest"
    assert view["authored_contract_match"] is False
    assert match["status"] == "PENDING"
    assert match["comparable"] is None


def test_contract_view_and_match_check_cover_scientific_gate_outcomes():
    contract = {"required_roles": ["a0", "a1"]}
    a0 = RunIndexRow(
        project="elf", run_id="run-a0", run_dir="/tmp/a0", role="a0",
        research_contract=contract,
    )
    missing_contract = RunIndexRow(
        project="elf", run_id="run-a1", run_dir="/tmp/a1", role="a1",
    )
    conflicting = RunIndexRow(
        project="elf", run_id="run-other", run_dir="/tmp/other", role="a1",
        research_contract={"required_roles": ["a0"]},
    )
    view = _contract_view([a0, missing_contract, conflicting], None)
    assert any("different frozen" in item for item in view["contract_warnings"])
    assert any("no frozen" in item for item in view["contract_warnings"])

    authored = _contract_view([], contract)
    assert authored["contract_source"] == "authored_campaign_reference"
    assert _match_check([], authored)["status"] == "UNVERIFIED_CONTRACT"

    frozen = _contract_view([a0], contract)
    assert _match_check([a0], frozen)["status"] == "INCOMPLETE"
    mismatched = a0.model_copy(update={
        "role": "a1", "decision": {"block_mismatches": [{"field": "seed"}]},
    })
    assert _match_check([a0, mismatched], frozen)["status"] == "MISMATCHED"
    decided = mismatched.model_copy(update={
        "decision": {"block_outcome": "PASS", "block_mismatches": []},
    })
    assert _match_check([a0, decided], frozen)["status"] == "COMPARABLE"


def test_attention_includes_failed_runs_and_collector_errors():
    failed = RunIndexRow(
        project="elf", run_id="failed", run_dir="/tmp/failed",
        scheduler_state="FAILED", decision={"failure_class": "OOM"},
    )
    plain = failed.model_copy(update={"run_id": "plain", "decision": {}})
    collector = SimpleNamespace(run_id="failed", last_error="scheduler unavailable")
    items = _attention([failed, plain], [collector])
    assert [item["kind"] for item in items].count("failed_run") == 2
    assert any("OOM" in item["detail"] for item in items)
    assert any(item["kind"] == "collector_error" for item in items)
