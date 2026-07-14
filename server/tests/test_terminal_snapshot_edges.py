"""Attention ordering and payload validation for terminal snapshots."""

from __future__ import annotations

from types import SimpleNamespace

from ml_exp_server import terminal_snapshot as module
from ml_exp_server.schemas import (
    CampaignRef,
    EvidenceLayer,
    EvidenceLayers,
    ResearchProject,
    RunIndexRow,
)
from ml_exp_server.terminal_snapshot import (
    build_snapshot,
    is_current_collector_error,
    snapshot_from_payload,
    snapshot_payload,
)


def row(run_id, campaign, state, *, stale=False):
    return RunIndexRow(
        project="demo", run_id=run_id, campaign=campaign,
        run_dir=f"/{run_id}", scheduler_state=state,
        evidence=EvidenceLayers(
            scheduler=EvidenceLayer(as_of=10),
            worker=EvidenceLayer(stale=stale, stale_reason="worker delayed" if stale else None),
        ),
    )


def test_current_collector_error_requires_error_and_recent_poll():
    current = row("run", "active", "RUNNING")
    assert is_current_collector_error(
        current, SimpleNamespace(last_error=None, last_poll_at=20),
    ) is False
    assert is_current_collector_error(
        current, SimpleNamespace(last_error="old", last_poll_at=9),
    ) is False
    current.evidence.scheduler.as_of = None
    assert is_current_collector_error(
        current, SimpleNamespace(last_error="new", last_poll_at=None),
    ) is True


def test_build_snapshot_classifies_active_historical_stale_and_collector(
    monkeypatch,
):
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[], campaigns=[
            CampaignRef(name="active"), CampaignRef(name="archived"),
        ],
    )
    rows = [
        row("active-failed", "active", "FAILED"),
        row("historical-failed", "archived", "PREEMPTED"),
        row("running-stale", "archived", "RUNNING", stale=True),
        row("active-stale", "active", "SUCCEEDED", stale=True),
    ]

    class Index:
        def __init__(self):
            self.reindexed = False

        def list_runs(self, _project):
            return list(rows)

        def collector_statuses(self, _project):
            return [
                SimpleNamespace(
                    run_id="running-stale", last_error="poll failed", last_poll_at=20,
                ),
                SimpleNamespace(run_id="missing", last_error="ignored", last_poll_at=20),
            ]

    index = Index()
    monkeypatch.setattr(
        module, "index_project", lambda *_args: setattr(index, "reindexed", True),
    )
    monkeypatch.setattr(module, "authored_run_placeholders", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        module, "campaign_snapshot",
        lambda _index, _project, campaign: {
            "lifecycle_state": "ARCHIVED" if campaign == "archived" else "ACTIVE",
        },
    )

    snapshot = build_snapshot(index, [project], reindex=True)

    assert index.reindexed is True
    assert snapshot.historical_failures == {"demo": 1}
    assert snapshot.attention["demo"][0][:2] == ("STALE", "running-stale")
    assert ("FAILED", "active-failed", "FAILED") in snapshot.attention["demo"]
    assert snapshot.collector_errors[("demo", "running-stale")] == "poll failed"


def test_snapshot_payload_round_trip_filters_malformed_collections():
    payload = {
        "projects": [{"project": "demo", "title": "Demo", "run_roots": []}],
        "runs": {
            "demo": [{"project": "demo", "run_id": "run-a", "run_dir": ""}],
            "bad": "not-a-list",
        },
        "attention": {"demo": [["FAILED", "run-a", "FAILED"], "bad"]},
        "campaign_statuses": [
            {"project": "demo", "campaign": "study", "status": {"state": "ACTIVE"}},
            {"project": "bad", "campaign": "bad", "status": "invalid"},
        ],
        "collector_errors": [
            {"project": "demo", "run_id": "run-a", "error": "poll"},
            {"project": "demo", "run_id": "run-b", "error": ""},
        ],
        "historical_failures": {"demo": "2"},
        "loaded_at": 123,
    }
    snapshot = snapshot_from_payload(payload)
    encoded = snapshot_payload(snapshot)
    assert list(snapshot.runs) == ["demo"]
    assert snapshot.attention["demo"] == [("FAILED", "run-a", "FAILED")]
    assert snapshot.historical_failures == {"demo": 2}
    assert encoded["campaign_statuses"][0]["campaign"] == "study"
    assert encoded["collector_errors"][0]["error"] == "poll"


def test_snapshot_from_empty_payload_uses_defaults():
    snapshot = snapshot_from_payload({})
    assert snapshot.projects == [] and snapshot.runs == {}
    assert snapshot.loaded_at > 0
