"""Shared submission identity and generic managed-Run read contracts."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ml_exp_server.application import ExperimentServerApplication
from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.ingest.runscan import evaluation_variants, scan_run_dir
from ml_exp_server.ingest.freshness import apply_freshness
from ml_exp_server.runs.first_submission import first_submission_intent
from ml_exp_server.runs.submissions import ExperimentSubmissionService
from ml_exp_server.schemas import (
    CampaignRef, CampaignRevision, CampaignRunMembership, ControllerConfig,
    OperationScope, ResearchProject, EvidenceLayer, EvidenceLayers, RunIndexRow,
)


def test_first_submit_entrypoints_freeze_same_source_attempt_and_budget(tmp_path):
    campaign = tmp_path / "study.yml"
    campaign.write_text("campaign: study\n")
    revision = CampaignRevision(
        campaign="study", project="demo", revision_id="campaign.revision",
        file=str(campaign), memberships=[CampaignRunMembership(run_id="run-a", kind="materialize")],
    )
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[], base_dir=tmp_path,
        controller=ControllerConfig(python="python", experimentctl="controller.py", workdir="."),
        campaigns=[CampaignRef(name="study", file=str(campaign), current_revision=revision)],
    )
    source_id = "source." + "a" * 64
    row = SimpleNamespace(
        scheduler_state="NOT_SUBMITTED", attempts=[], campaign="study", run_id="run-a",
        campaign_memberships=[SimpleNamespace(campaign="study", membership=SimpleNamespace(kind="materialize"))],
        provenance={"source_binding": "campaign_file", "source_id": source_id},
        model_dump=lambda **kw: {"run_id": "run-a"},
    )
    intents = []
    source_reads = []

    def prepare(scope, configured, intent):
        intents.append(intent)
        return {"action_id": "action-a", "operation": "SUBMIT_RUN", "execution": {"status": "PREPARED"}}

    runtime = SimpleNamespace(
        config=SimpleNamespace(), project=lambda name: project,
        index=SimpleNamespace(get_run=lambda *a: row),
        action_store=SimpleNamespace(list_for_scope=lambda scope: []),
        action_service=SimpleNamespace(prepare=prepare),
    )
    application = ExperimentServerApplication(runtime)
    application.source_revision_service = SimpleNamespace(get=lambda *args: source_reads.append(args))
    scope = OperationScope(project="demo", scope_type="run", object_id="run-a")
    application.resolve_scope = lambda *args: (scope, project, row)
    application._require_operation_available = lambda *args: None
    application.bounded_evidence = lambda *args: {"run_id": "run-a"}
    application.prepare_run_submit("demo", "run-a", max_gpu_hours=1)
    ExperimentSubmissionService(application, runtime).prepare_first_attempt("demo", "run-a", max_gpu_hours=1, reason="")
    drafts = [yaml.safe_load(intent["draft"]) for intent in intents]
    assert drafts[0] == drafts[1] == {
        "campaign_file": str(campaign), "run_id": "run-a", "attempt_id": "attempt-001",
        "resource_approval": "budget_cap", "expected_source_id": source_id, "max_gpu_hours": 1,
    }
    assert source_reads == [("demo", source_id), ("demo", source_id)]
    assert "idempotency_key" not in intents[0]
    assert intents[1]["idempotency_key"].startswith("submission-")


@pytest.mark.parametrize("budget", [None, 0, -1, float("nan"), float("inf")])
def test_first_submit_invalid_budget_cannot_read_evidence_or_prepare(budget):
    with pytest.raises(ApplicationError) as error:
        first_submission_intent(
            None, None, run_id="run-a", campaign_file=lambda: pytest.fail("campaign read"),
            max_gpu_hours=budget, resource_approval="budget_cap", reason="",
            evidence=lambda: pytest.fail("evidence read"), source_service=None,
        )
    assert error.value.code == "INVALID_GPU_BUDGET"


def test_first_submit_review_exact_keeps_cap_optional(tmp_path):
    intent = first_submission_intent(
        SimpleNamespace(project="demo", controller=object()), None,
        run_id="run-a", campaign_file=lambda: tmp_path / "study.yml",
        max_gpu_hours=None, resource_approval="review_exact", reason="review",
        evidence=lambda: {}, source_service=None,
    )
    assert "max_gpu_hours" not in yaml.safe_load(intent["draft"])
    assert "human Action confirmation" in intent["resource_estimate"]


def test_managed_run_uses_its_generic_configuration_and_not_project_guesses(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("DEPTH = 999\nDEVICE_BATCH_SIZE = 999\n")
    manifest = {
        "identity_version": 2, "config_path": "container_execution", "run_id": "elf-a1-candidate", "project": "demo", "campaign": "study",
        "source": {"source_sha256": "abc"},
        "resolved_config": {"learning_rate": 0.01, "tokenizer": "custom", "nested": {"private": 1}},
    }
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    row = scan_run_dir(tmp_path, "demo")
    assert row.provenance["resolved_config_excerpt"] == {"learning_rate": 0.01, "tokenizer": "custom"}
    assert row.role is None
    old_metrics = tmp_path / "train_sampling_eval" / "old" / "metrics.jsonl"
    old_metrics.parent.mkdir(parents=True)
    old_metrics.write_text('{"name":"loss","unit":"nats/token","value":1}\n')
    assert evaluation_variants(tmp_path)[0] == []


def test_model_freshness_does_not_guess_log_interval_from_throughput():
    row = RunIndexRow(
        project="demo", run_id="run-a", run_dir="/unused", campaign="study",
        scheduler_state="RUNNING", latest_metrics={"steps_per_sec": 1e-9},
        evidence=EvidenceLayers(model=EvidenceLayer(state="TRAINING", as_of=1)),
    )
    apply_freshness(row, now=3601)
    assert row.evidence.model.stale


def test_historical_v2_project_adapter_remains_read_only(tmp_path):
    manifest = {"identity_version": 2, "run_id": "elf-a1-old", "project": "demo", "campaign": "study",
                "resolved_config": {"use_sentence_plan": True, "learning_rate": 0.1}}
    path = tmp_path / "manifest.yaml"
    original = yaml.safe_dump(manifest)
    path.write_text(original)
    row = scan_run_dir(tmp_path, "demo")
    assert row.provenance["resolved_config_excerpt"] == {"use_sentence_plan": True}
    assert row.role == "a1"
    assert path.read_text() == original


def test_shared_http_boundary_keeps_typed_status_and_hides_untyped_errors():
    import asyncio
    from fastapi import HTTPException
    from ml_exp_server.api.errors import invoke

    def typed():
        raise ApplicationError("retry later", status_code=503, code="BUSY")

    with pytest.raises(HTTPException) as error:
        asyncio.run(invoke(typed))
    assert error.value.status_code == 503
    assert error.value.headers["X-ML-Expd-Error-Code"] == "BUSY"

    def untyped():
        raise KeyError("private-provider-reply")

    with pytest.raises(HTTPException) as error:
        asyncio.run(invoke(untyped))
    assert error.value.status_code == 409
    assert "private-provider-reply" not in str(error.value.detail)


def test_transfer_without_content_length_grows_its_reservation(tmp_path):
    import asyncio
    from ml_exp_server.api.transfers import receive_body

    async def stream():
        yield b"abc"
        yield b"def"

    reservations = []
    value = asyncio.run(receive_body(
        SimpleNamespace(stream=stream), SimpleNamespace(root=tmp_path, limit=1024),
        lambda stream, size: (stream.read(), size), (),
        SimpleNamespace(resize=lambda size: reservations.append(size)), 0,
    ))
    assert value == (b"abcdef", 6)
    assert reservations == [9, 18]
    assert list(tmp_path.iterdir()) == []


def test_transfer_buffer_backpressure_returns_retry_after():
    import asyncio
    from fastapi import HTTPException
    from ml_exp_server.api.errors import invoke

    def busy():
        raise ApplicationError("transfer buffer is busy", status_code=503, code="TRANSFER_BUFFER_BUSY")

    with pytest.raises(HTTPException) as error:
        asyncio.run(invoke(busy))
    assert error.value.status_code == 503
    assert error.value.headers["Retry-After"] == "1"
