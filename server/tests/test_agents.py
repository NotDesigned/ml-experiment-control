"""Scoped agent persistence, API, and provider-neutral runtime boundary."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from ml_exp_server.api.app import create_app
from ml_exp_server.api.agent_routes import _compact_evidence
from ml_exp_server.agents.store import AgentStore
from ml_exp_server.schemas import (
    AgentScope,
    AgentScopeType,
    CampaignMembershipBinding,
    CampaignRunMembership,
    ServerConfig,
    ProjectRef,
)
from tests.conftest import FIXTURES


@pytest.fixture
def client(tmp_path):
    experiments = tmp_path / "experiments"
    research_questions = experiments / "research_questions"
    research_questions.mkdir(parents=True)
    campaign = experiments / "campaign.yml"
    campaign.write_text(textwrap.dedent("""\
        schema_version: 1
        project: elf
        campaign: fusion-len256-gate-h100-20260711
        runs:
          - run_id: elf-a1-frozen-t5-l256-s42-h100-v1
            research_role: a1
    """))
    project_file = experiments / "research_project.yaml"
    project_file.write_text(textwrap.dedent(f"""\
        schema_version: 1
        project: elf
        title: ELF plan-token fusion
        run_roots: [{FIXTURES / 'runs'}]
        campaigns:
          - name: fusion-len256-gate-h100-20260711
            file: {campaign}
        research_questions_dir: experiments/research_questions
    """))
    (research_questions / "h1.yml").write_text(textwrap.dedent(f"""\
        schema_version: 1
        id: H1
        title: sentence plan viability
        status: OPEN
        links:
          campaigns: [fusion-len256-gate-h100-20260711]
    """))
    config = ServerConfig(
        index_db=str(tmp_path / "index.sqlite"),
        agent_root=str(tmp_path / "agents"),
        action_root=str(tmp_path / "actions"),
        projects=[ProjectRef(project_file=str(project_file))],
    )
    with TestClient(create_app(config)) as test_client:
        yield test_client


@pytest.mark.parametrize("scope_type,object_id", [
    ("project", "elf"),
    ("research_question", "H1"),
    ("campaign", "fusion-len256-gate-h100-20260711"),
    ("run", "elf-a1-frozen-t5-l256-s42-h100-v1"),
    ("attempt", "elf-a1-frozen-t5-l256-s42-h100-v1::attempt-001"),
])
def test_all_five_agent_scopes_are_addressable(client, scope_type, object_id):
    response = client.get("/api/agent", params={
        "project": "elf", "scope_type": scope_type, "object_id": object_id,
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["scope"]["scope_type"] == scope_type
    assert payload["scope"]["object_id"] == object_id
    assert payload["state"] == "IDLE"
    assert payload["goal"]


def test_operation_availability_explains_scope_and_safety_blockers(client):
    application = client.app.state.application
    project_ops = {
        item.operation.operation_id: item for item in application.operation_availability(
            "elf", AgentScopeType.PROJECT, "elf",
        )
    }
    assert project_ops["research.recommend"].available
    assert project_ops["question.create"].available
    assert project_ops["campaign.create"].available

    run_id = "elf-a1-frozen-t5-l256-s42-h100-v1"
    run_ops = {
        item.operation.operation_id: item for item in application.operation_availability(
            "elf", AgentScopeType.RUN, run_id,
        )
    }
    assert run_ops["run.submit"].status == "BLOCKED"
    assert "controller" in " ".join(run_ops["run.submit"].reasons).lower()
    assert run_ops["run.evaluate"].status == "BLOCKED"
    assert "controller" in " ".join(run_ops["run.evaluate"].reasons).lower()
    assert run_ops["report.generate"].available

    attempt_ops = {
        item.operation.operation_id: item for item in application.operation_availability(
            "elf", AgentScopeType.ATTEMPT, f"{run_id}::attempt-001",
        )
    }
    assert attempt_ops["attempt.retry"].status == "BLOCKED"
    assert attempt_ops["attempt.cancel"].status == "BLOCKED"
    assert any("state" in reason.lower() for reason in attempt_ops["attempt.retry"].reasons)


def test_run_agent_receives_authored_campaign_membership_and_comparator_context(client):
    application = client.app.state.application
    scope, project, row = application.resolve_scope(
        "elf", AgentScopeType.RUN, "elf-a1-frozen-t5-l256-s42-h100-v1",
    )

    evidence = application.bounded_evidence(scope, project, row)

    assert evidence["run"]["campaign_memberships"][0]["campaign"] == \
        "fusion-len256-gate-h100-20260711"
    context = evidence["campaign_contexts"][0]
    assert context["membership"]["role"] == "a1"
    assert context["comparator_runs"][0]["run_id"] == row.run_id


def test_orphaned_historical_campaign_context_never_crashes_scope_resolution(client):
    application = client.app.state.application
    _, project, row = application.resolve_scope(
        "elf", AgentScopeType.RUN, "elf-a1-frozen-t5-l256-s42-h100-v1",
    )
    row.campaign_memberships.append(CampaignMembershipBinding(
        campaign="removed-campaign", revision_id="campaign.old",
        membership=CampaignRunMembership(
            run_id=row.run_id, kind="reuse", role="historical",
        ),
    ))

    contexts = application.campaign_contexts(project, row)

    orphan = next(item for item in contexts if item["campaign"] == "removed-campaign")
    assert orphan["orphaned_campaign"] is True
    assert orphan["lifecycle"]["lifecycle_state"] == "UNKNOWN"


def test_external_agent_turn_persists_result_and_nonexecuting_approval(client):
    scope = {"project": "elf", "scope_type": "research_question", "object_id": "H1"}
    queued = client.post("/api/agent/turns", json={**scope, "message": "分析当前证据"})
    assert queued.status_code == 200
    turn = queued.json()["turn"]
    assert turn["status"] == "PENDING"

    claimed = client.post(
        f"/api/agent/turns/{turn['request_id']}/claim",
        json={**scope, "client_id": "test-client", "provider": "fake"},
    )
    assert claimed.status_code == 200
    context = claimed.json()
    assert context["turn"]["status"] == "RUNNING"
    assert context["evidence_digest"].startswith("sha256:")
    assert context["project_context"]["workspace_root"]
    assert context["project_context"]["source_identity"]["kind"] == "directory"
    assert context["project_context"]["project_file_relative"] == \
        "experiments/research_project.yaml"
    assert context["project_context"]["project_file_digest"].startswith("sha256:")

    completed = client.post(
        f"/api/agent/turns/{turn['request_id']}/result",
        json={
            **scope, "status": "COMPLETED", "client_id": "test-client",
            "session_id": "session-phase2",
            "message": "当前证据仍不完整，应继续观察。",
            "evidence_digest": context["evidence_digest"],
            "proposals": [{
                "kind": "CREATE_REPORT_DRAFT", "title": "补齐评估",
                "target": "research_question://elf/H1",
                "change_summary": "新增评估报告", "resource_estimate": "none",
                "rationale": "evaluation evidence 缺失", "risk": "none",
                "draft": "oracle-vs-shuffled evidence report",
            }],
        },
    )
    assert completed.status_code == 200
    payload = completed.json()
    assert payload["thread_id"] == "session-phase2"
    assert payload["state"] == "WAITING_FOR_APPROVAL"
    assert [item["role"] for item in payload["messages"]] == ["user", "assistant"]
    proposal = payload["proposals"][0]
    assert payload["created_proposals"][0]["proposal_id"] == proposal["proposal_id"]
    assert proposal["status"] == "PENDING"
    assert "oracle-vs-shuffled" in proposal["draft"]
    assert proposal["target"] == "research_question://elf/H1"
    assert payload["current_evidence_digest"].startswith("sha256:")
    assert payload["messages"][-1]["evidence_captured_at"]

    decided = client.post("/api/agent/proposal-decision", json={
        **scope,
        "proposal_id": proposal["proposal_id"],
        "decision": "APPROVED",
        "note": "仅批准记录，不执行",
    })
    assert decided.status_code == 200
    approval = decided.json()["approval"]
    assert approval["decision"] == "APPROVED"
    assert approval["execution_enabled"] is False
    assert decided.json()["agent"]["state"] == "IDLE"


def test_external_agent_output_with_wrong_scope_is_rejected(client):
    scope = {"project": "elf", "scope_type": "research_question", "object_id": "H1"}
    turn = client.post("/api/agent/turns", json={**scope, "message": "Evaluate it"}).json()["turn"]
    context = client.post(
        f"/api/agent/turns/{turn['request_id']}/claim",
        json={**scope, "client_id": "test", "provider": "fake"},
    ).json()
    response = client.post(f"/api/agent/turns/{turn['request_id']}/result", json={
        **scope, "status": "COMPLETED", "client_id": "test",
        "session_id": "wrong-scope-session",
        "message": "Try evaluation", "evidence_digest": context["evidence_digest"],
        "proposals": [{
            "kind": "RUN_EVALUATION", "title": "Wrong scope", "target": "run-a",
            "change_summary": "evaluate", "resource_estimate": "1 GPUh",
            "rationale": "test", "risk": "GPU", "draft": "max_gpu_hours: 1",
        }],
    })
    payload = response.json()
    assert payload["created_proposals"] == []
    assert payload["state"] == "IDLE"
    assert "not valid in research_question scope" in payload["messages"][-1]["content"]


def test_claim_context_uses_project_defaults_guidance_and_operations(client):
    project = client.app.state.runtime.project("elf")
    project.proposal_defaults = {
        "max_gpu_hours": 0.25,
        "resources": {"nodes": 1, "gpus": 1, "backend": "local-cuda"},
    }
    project.agent_guidance = ["Use one immutable local harness Attempt."]

    queued = client.post("/api/agent/turns", json={
        "project": "elf", "scope_type": "project", "object_id": "elf",
        "message": "Analyze the local run.",
    })
    turn = queued.json()["turn"]
    context = client.post(
        f"/api/agent/turns/{turn['request_id']}/claim", json={
            "project": "elf", "scope_type": "project", "object_id": "elf",
            "client_id": "test", "provider": "fake",
        },
    ).json()
    assert context["project_context"]["proposal_defaults"] == project.proposal_defaults
    assert context["project_context"]["agent_guidance"] == project.agent_guidance
    assert any(item["operation"]["operation_id"] == "campaign.create"
               for item in context["operations"])


def test_claim_next_and_stale_result_fail_closed(client, monkeypatch):
    scope = {"project": "elf", "scope_type": "project", "object_id": "elf"}
    queued = client.post("/api/agent/turns", json={**scope, "message": "analyze"}).json()
    claimed = client.post("/api/agent/turns/claim-next", json={
        "client_id": "worker-1", "provider": "fake", "project": "elf",
    }).json()["turn_context"]
    assert claimed["turn"]["request_id"] == queued["turn"]["request_id"]
    assert client.post("/api/agent/turns/claim-next", json={
        "client_id": "worker-2", "provider": "fake", "project": "elf",
    }).json()["turn_context"] is None

    wrong_owner = client.post(
        f"/api/agent/turns/{claimed['turn']['request_id']}/result", json={
            **scope, "status": "COMPLETED", "client_id": "worker-2",
            "message": "not mine", "evidence_digest": claimed["evidence_digest"],
            "proposals": [],
        },
    )
    assert wrong_owner.headers["X-ML-Expd-Error-Code"] == \
        "AGENT_TURN_OWNER_MISMATCH"

    monkeypatch.setattr(
        client.app.state.application, "bounded_evidence", lambda *args: {"changed": True},
    )
    response = client.post(
        f"/api/agent/turns/{claimed['turn']['request_id']}/result", json={
            **scope, "status": "COMPLETED", "client_id": "worker-1",
            "message": "old analysis",
            "evidence_digest": claimed["evidence_digest"], "proposals": [],
        },
    )
    assert response.status_code == 409
    assert response.headers["X-ML-Expd-Error-Code"] == "STALE_AGENT_RESULT"


def test_invalid_campaign_draft_is_marked_and_cannot_be_approved(client):
    scope = AgentScope(project="elf", scope_type="research_question", object_id="H1")
    created = client.app.state.agent_store.add_proposals(scope, [{
        "kind": "CREATE_CAMPAIGN_DRAFT", "title": "broken campaign",
        "target": "campaign://elf/broken", "change_summary": "invalid shape",
        "resource_estimate": "unknown", "rationale": "test", "risk": "invalid",
        "draft": "schema_version: 1\nproject: elf\nruns: []\n",
    }], evidence_digest="sha256:old")
    proposal = created[0]

    assert proposal["validation"]["status"] == "INVALID"
    assert "research_contract mapping is required" in proposal["validation"]["errors"]
    with pytest.raises(ValueError, match="invalid proposal draft"):
        client.app.state.agent_store.decide_proposal(
            scope, proposal["proposal_id"], "APPROVED"
        )


def test_derived_run_budget_and_resources_are_machine_validated(client):
    scope = AgentScope(project="elf", scope_type="research_question", object_id="H1")
    common = {
        "kind": "DERIVE_RUN_DRAFT", "title": "A2 smoke", "target": "elf/a2-smoke",
        "change_summary": "derive", "resource_estimate": "40 H100 GPU-hours",
        "rationale": "OOM recovery", "risk": "resource",
    }
    invalid = client.app.state.agent_store.add_proposals(scope, [{
        **common,
        "draft": yaml.safe_dump({
            "schema_version": 1, "project": "elf", "campaign": "a2-smoke",
            "research_contract": {}, "budget": {"max_gpu_hours": "unknown"},
            "default_resources": {"gpus": 4},
            "runs": [{"run_id": "a2-smoke-run"}],
        }),
    }], evidence_digest="sha256:evidence")[0]
    assert invalid["validation"]["status"] == "INVALID"
    assert "budget.max_gpu_hours must be a positive number" in invalid["validation"]["errors"]
    assert "use defaults.resources; default_resources is not canonical" in invalid["validation"]["errors"]

    valid = client.app.state.agent_store.add_proposals(scope, [{
        **common,
        "draft": yaml.safe_dump({
            "schema_version": 1, "project": "elf", "campaign": "a2-smoke",
            "research_contract": {}, "budget": {"max_gpu_hours": 40},
            "defaults": {"resources": {"nodes": 1, "gpus": 4}},
            "runs": [{"run_id": "a2-smoke-run"}],
        }),
    }], evidence_digest="sha256:evidence")[0]
    assert valid["validation"] == {"status": "VALID", "errors": []}


def test_campaign_reuse_requires_explicit_run_refs_and_no_scheduler_budget(client):
    scope = AgentScope(project="elf", scope_type="research_question", object_id="H1")
    common = {
        "kind": "CREATE_CAMPAIGN_DRAFT", "title": "reuse baseline",
        "target": "campaign://elf/reuse", "change_summary": "reuse existing evidence",
        "resource_estimate": "none", "rationale": "shared baseline", "risk": "low",
    }
    explicit = client.app.state.agent_store.add_proposals(scope, [{
        **common,
        "draft": yaml.safe_dump({
            "schema_version": 1, "project": "elf", "campaign": "reuse",
            "research_contract": {},
            "run_refs": [{"run_id": "elf-a1-frozen-t5-l256-s42-h100-v1",
                          "research_role": "baseline"}],
        }),
    }], evidence_digest="sha256:evidence")[0]
    assert explicit["validation"] == {"status": "VALID", "errors": []}

    implicit = client.app.state.agent_store.add_proposals(scope, [{
        **common,
        "draft": yaml.safe_dump({
            "schema_version": 1, "project": "elf", "campaign": "implicit-reuse",
            "research_contract": {},
            "runs": [{"run_id": "elf-a1-frozen-t5-l256-s42-h100-v1"}],
        }),
    }], evidence_digest="sha256:evidence")[0]
    assert implicit["validation"]["status"] == "INVALID"
    assert "budget or defaults.resources is required" in implicit["validation"]["errors"]

    wrong_role = client.app.state.agent_store.add_proposals(scope, [{
        **common,
        "draft": yaml.safe_dump({
            "schema_version": 1, "project": "elf", "campaign": "wrong-role",
            "research_contract": {"required_roles": ["baseline", "candidate"]},
            "run_refs": [
                {"run_id": "baseline-run", "role": "baseline"},
                {"run_id": "candidate-run", "membership": {"role": "candidate"}},
            ],
        }),
    }], evidence_digest="sha256:evidence")[0]
    errors = wrong_role["validation"]["errors"]
    assert wrong_role["validation"]["status"] == "INVALID"
    assert "membership fields must be top-level and use research_role, not role" in errors
    assert "every membership requires research_role when required_roles is declared" in errors
    assert "memberships do not cover required_roles: baseline, candidate" in errors


def test_project_adapter_proposal_requires_canonical_field_shapes(client):
    scope = AgentScope(project="adapter_demo", scope_type="project", object_id="adapter_demo")
    store = client.app.state.agent_store
    store.ensure(scope, default_goal="onboard")
    common = {
        "kind": "CREATE_PROJECT_ADAPTER_DRAFT", "title": "adapter",
        "target": "adapter_demo", "change_summary": "onboard",
        "resource_estimate": "none", "rationale": "test", "risk": "review",
    }
    invalid = store.add_proposals(scope, [{
        **common, "draft": "project: adapter_demo\ntrain:\n  command:\n    argv: [python, train.py]\n",
    }], evidence_digest="sha256:evidence")[0]
    valid = store.add_proposals(scope, [{
        **common,
        "draft": yaml.safe_dump({
            "project": "adapter_demo", "title": "Adapter",
            "source": {"identity": "git:" + "a" * 40, "required_paths": ["train.py"]},
            "train": {"command": ["python", "train.py", "--output", "{run_dir}"]},
            "parameters": {"seed": {"type": "integer", "default": 1,
                                        "required": True, "description": "seed"}},
            "container": {"base_image": "image@sha256:" + "b" * 64,
                          "image": "image@sha256:" + "c" * 64,
                          "install_command": ["uv", "sync", "--frozen"]},
            "backend_profile": {"kind": "slurm"},
            "assets": [],
            "checkpoint": {"expected_first_minutes": 5,
                           "max_uncheckpointed_minutes": 10},
            "outputs": {"metrics": "metrics.jsonl", "checkpoints": "checkpoints",
                        "artifacts": "artifacts"},
        }),
    }], evidence_digest="sha256:evidence")[0]

    assert invalid["validation"]["status"] == "INVALID"
    assert "train.command must be a non-empty argv list" in invalid["validation"]["errors"]
    assert valid["validation"] == {"status": "VALID", "errors": []}


def test_proposal_snapshot_is_chronological_not_id_sorted(tmp_path):
    store = AgentStore(tmp_path / "agents")
    scope = AgentScope(project="demo", scope_type="project", object_id="demo")
    directory = store.agent_dir(scope)
    store.ensure(scope, default_goal="test")
    older = {"proposal_id": "proposal-ffffffffffff", "title": "older",
             "created_at": "2026-01-01T00:00:00Z", "status": "PENDING"}
    newer = {"proposal_id": "proposal-000000000000", "title": "newer",
             "created_at": "2026-01-02T00:00:00Z", "status": "PENDING"}
    (directory / "proposals" / "proposal-ffffffffffff.json").write_text(json.dumps(older))
    (directory / "proposals" / "proposal-000000000000.json").write_text(json.dumps(newer))

    assert [item["title"] for item in store.snapshot(scope)["proposals"]] == [
        "older", "newer",
    ]


def test_outdated_proposal_cannot_be_approved(client):
    scope = AgentScope(project="elf", scope_type="research_question", object_id="H1")
    created = client.app.state.agent_store.add_proposals(scope, [{
        "kind": "CREATE_REPORT_DRAFT", "title": "old report",
        "target": "research_question://elf/H1", "change_summary": "summarize",
        "resource_estimate": "none", "rationale": "test", "risk": "stale",
        "draft": "old evidence",
    }], evidence_digest="sha256:old")
    response = client.post("/api/agent/proposal-decision", json={
        "project": "elf", "scope_type": "research_question", "object_id": "H1",
        "proposal_id": created[0]["proposal_id"], "decision": "APPROVED",
    })

    assert response.status_code == 409
    assert "outdated proposal" in response.json()["detail"]


def test_goal_update_is_durable(client):
    request = {
        "project": "elf", "scope_type": "project", "object_id": "elf",
        "goal": "先完成 H1 的证据闭环",
    }
    assert client.put("/api/agent/goal", json=request).status_code == 200
    payload = client.get("/api/agent", params={
        "project": "elf", "scope_type": "project", "object_id": "elf",
    }).json()
    assert payload["goal"] == request["goal"]
    assert any(item["event"] == "goal_updated" for item in payload["journal"])


def test_agent_store_uses_hashed_scope_path(tmp_path):
    store = AgentStore(tmp_path / "agents")
    scope = AgentScope(
        project="elf", scope_type=AgentScopeType.ATTEMPT,
        object_id="run/../../unsafe::attempt-001",
    )
    snapshot = store.ensure(scope, default_goal="diagnose")
    assert ".." not in snapshot["agent_id"]
    assert store.agent_dir(scope).parent == tmp_path / "agents"


def test_chart_proposal_creates_versioned_draft_artifact(tmp_path):
    store = AgentStore(tmp_path / "agents")
    scope = AgentScope(
        project="elf", scope_type=AgentScopeType.RESEARCH_QUESTION, object_id="H1",
    )
    store.ensure(scope, default_goal="review evidence")
    created = store.add_proposals(scope, [{
        "kind": "CREATE_CHART_SPEC",
        "title": "A0-A3 comparison",
        "target": "research_question/H1/charts",
        "change_summary": "add chart spec",
        "resource_estimate": "none",
        "rationale": "compare roles",
        "risk": "missing series",
        "draft": "x: step\ny: train_loss",
    }], evidence_digest="sha256:evidence")

    snapshot = store.snapshot(scope)
    assert created[0]["artifact_id"].startswith("chart-")
    artifact = snapshot["draft_artifacts"][0]
    assert artifact["artifact_type"] == "chart"
    assert artifact["version"] == 1
    assert artifact["evidence_digest"] == "sha256:evidence"


def test_older_chart_proposal_is_migrated_to_draft_artifact(tmp_path):
    store = AgentStore(tmp_path / "agents")
    scope = AgentScope(
        project="elf", scope_type=AgentScopeType.RESEARCH_QUESTION, object_id="H1",
    )
    store.ensure(scope, default_goal="review evidence")
    proposal = {
        "proposal_id": "proposal-old",
        "kind": "CREATE_CHART_SPEC",
        "title": "legacy chart",
        "target": "research_question/H1/charts",
        "draft": "x: step",
        "evidence_digest": "sha256:old",
        "created_at": "2026-07-12T00:00:00Z",
    }
    proposal_path = store.agent_dir(scope) / "proposals" / "proposal-old.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")

    snapshot = store.ensure(scope, default_goal="review evidence")

    assert snapshot["draft_artifacts"][0]["proposal_id"] == "proposal-old"
    assert json.loads(proposal_path.read_text())["artifact_id"].startswith("chart-")


def test_agent_evidence_drops_logs_and_bounds_large_values():
    compact = _compact_evidence({
        "model": {
            "stdout_tail": ["secretly huge"],
            "stderr_tail": ["also huge"],
            "metric": 1.25,
            "note": "x" * 3000,
            "records": list(range(50)),
        }
    })

    assert "stdout_tail" not in compact["model"]
    assert "stderr_tail" not in compact["model"]
    assert compact["model"]["metric"] == 1.25
    assert compact["model"]["note"].endswith("…[truncated]")
    assert compact["model"]["records"][-1] == "[10 additional records omitted]"
