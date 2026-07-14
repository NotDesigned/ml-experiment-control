"""SQLite migration and reconciliation edges for the rebuildable Run index."""

from __future__ import annotations

import sqlite3

from ml_exp_server.ingest.indexer import RunIndex, _reconcile_campaign_binding
from ml_exp_server.schemas import (
    CampaignRef,
    CampaignRevision,
    CampaignRunMembership,
    ResearchProject,
    RunIndexRow,
)


def test_run_index_migrates_legacy_collector_status_table(tmp_path):
    path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE collector_status ("
        "project TEXT NOT NULL, run_id TEXT NOT NULL, last_poll_at REAL, "
        "last_verb TEXT, last_error TEXT, PRIMARY KEY (project, run_id))"
    )
    connection.commit()
    connection.close()

    index = RunIndex(path)
    columns = {
        row[1] for row in index._conn.execute(
            "PRAGMA table_info(collector_status)"
        ).fetchall()
    }
    assert "verb_results" in columns
    index.close()


def test_list_runs_without_filters_and_invalid_legacy_poll_json(tmp_path):
    index = RunIndex(tmp_path / "index.sqlite")
    row = RunIndexRow(
        project="demo", campaign="study", run_id="run-a", run_dir="/run-a",
    )
    index.upsert_run(row)
    assert [item.run_id for item in index.list_runs()] == ["run-a"]

    with index._lock:
        index._conn.execute(
            "INSERT INTO collector_status "
            "(project, run_id, last_poll_at, last_verb, last_error, verb_results) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("demo", "run-a", 0, "status", None, "not-json"),
        )
        index._conn.commit()
    index.record_poll("demo", "run-a", "collect", now=1)
    status = index.collector_statuses("demo")[0]
    assert status.verb_results["collect"]["outcome"] == "succeeded"
    index.close()


def test_campaign_membership_without_role_preserves_observed_role():
    membership = CampaignRunMembership(run_id="run-a", kind="materialize")
    revision = CampaignRevision(
        campaign="study", project="demo", revision_id="campaign.revision",
        file="study.yml", memberships=[membership],
    )
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[],
        campaigns=[CampaignRef(name="study", current_revision=revision)],
    )
    row = RunIndexRow(
        project="demo", campaign="study", run_id="run-a", run_dir="/run-a",
        role="observed", role_source="manifest",
    )
    row.campaign_binding.origin_revision = revision.revision_id

    _reconcile_campaign_binding(row, project)

    assert row.role == "observed"
    assert row.campaign_binding.relationship.value == "MATCHED"
