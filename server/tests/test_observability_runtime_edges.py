"""Coordinator backfill, publication, and source-discovery edges."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server.observability_archive import ArchiveSource
from ml_exp_server.observability_runtime import ObservabilityCoordinator, _observed_at
from ml_exp_server.observability_store import AttemptRef, OutboxItem, TargetStatus
from ml_exp_server.schemas import (
    AttemptSummary,
    EvidenceLayer,
    EvidenceLayers,
    LocalWandbConfig,
    RunIndexRow,
    WandbCloudConfig,
)
from ml_exp_server.wandb_publisher import PublishResult, TargetConfig, TargetKind


class Store:
    def __init__(self, items=(), statuses=(), terminal=False):
        self.items = list(items)
        self.target_statuses = list(statuses)
        self.terminal = terminal
        self.calls = []

    def revive_terminal(self, target):
        self.calls.append(("revive", target))

    def claim(self, target, worker, limit):
        self.calls.append(("claim", target, limit))
        if target == "local":
            result, self.items = self.items, []
            return result
        return []

    def set_target_state(self, attempt, target, state, **kwargs):
        self.calls.append(("state", target, state, kwargs))

    def retry(self, item_id, worker, error):
        self.calls.append(("retry", item_id, error))
        return self.terminal

    def statuses(self, **kwargs):
        return self.target_statuses

    def acknowledge(self, item_id, worker):
        self.calls.append(("ack", item_id))

    def record_archive_error(self, reference, error):
        self.calls.append(("archive_error", reference.source_key, error))

    def get_cursor(self, reference):
        return None


def coordinator(tmp_path, store=None, *, credential=lambda _ref: "secret"):
    return ObservabilityCoordinator(
        workspace_id="workspace",
        archive_root=tmp_path / "archive",
        store=store or Store(),
        local=LocalWandbConfig(
            enabled=True, managed=False, external_url="http://127.0.0.1:8080",
            publisher_entity="team", publisher_credential_ref="local",
        ),
        cloud=WandbCloudConfig(),
        credential_provider=credential,
    )


def outbox(kind="metrics"):
    return OutboxItem(
        id=1,
        attempt=AttemptRef("workspace", "demo", "run-a", "attempt-001"),
        target="local", record_key="record", kind=kind,
        payload={"loss": 1}, observed_at=1.0, created_at=0,
        available_at=0, attempt_count=0, lease_owner="worker", lease_until=1,
    )


def config(tmp_path):
    return TargetConfig(
        kind=TargetKind.LOCAL, api_url="http://127.0.0.1:8080",
        dashboard_url="http://127.0.0.1:8080", entity="team",
        project="demo", working_dir=tmp_path / "publisher",
        credential_ref="local",
    )


def test_backfill_validates_target_availability_workspace_and_bounds(tmp_path):
    value = coordinator(tmp_path)
    with pytest.raises(ValueError, match="unsupported observability target"):
        value.backfill("invalid", [])
    with pytest.raises(RuntimeError, match="PublisherUnavailable"):
        value.backfill("cloud", [])
    with pytest.raises(ValueError, match="workspace does not match"):
        value.backfill("local", [AttemptRef("other", "demo", "run", "attempt")])
    with pytest.raises(ValueError, match="between 1 and 500"):
        value.backfill("local", [])


def test_collect_rows_skips_authored_placeholder_without_directory(tmp_path):
    value = coordinator(tmp_path)
    value.collect_rows([RunIndexRow(
        project="demo", run_id="run-a", run_dir="",
    )])


def test_publish_once_validates_limit_and_handles_missing_target_config(tmp_path):
    store = Store([outbox()])
    value = coordinator(tmp_path, store)
    with pytest.raises(ValueError, match="must be positive"):
        value.publish_once(limit_per_target=0)
    value._target_config = lambda *_args: None
    value.publish_once()
    assert ("retry", 1, "PublisherUnavailable") in store.calls
    assert any(call[:3] == ("state", "local", "DEGRADED") for call in store.calls)


def test_publish_once_maps_construction_exception_and_terminal_retry(tmp_path):
    store = Store([outbox("unsupported")], terminal=True)
    value = coordinator(tmp_path, store)
    value._target_config = lambda *_args: config(tmp_path)
    value.publish_once()
    assert ("retry", 1, "ValueError") in store.calls
    assert any(call[:3] == ("state", "local", "FAILED") for call in store.calls)


def test_publish_once_maps_unacknowledged_result(tmp_path):
    store = Store([outbox()])
    value = coordinator(tmp_path, store)
    value._target_config = lambda *_args: config(tmp_path)
    value.publisher = SimpleNamespace(publish=lambda *_args: PublishResult(
        acknowledged=False, target=TargetKind.LOCAL, record_key="record",
        run_id="run", dashboard_url=None, error_class=None,
    ))
    value.publish_once()
    assert ("retry", 1, "PublisherError") in store.calls


@pytest.mark.parametrize(("pending", "terminal", "expected"), [
    (0, 0, "READY"),
    (1, 0, "PENDING"),
    (0, 1, "FAILED"),
])
def test_publish_once_acknowledgement_derives_target_state(
    tmp_path, pending, terminal, expected,
):
    attempt = outbox().attempt
    status = TargetStatus(
        attempt=attempt, target="local", state="SYNCING", dashboard_url=None,
        last_error=None, updated_at=0, pending=pending, leased=0,
        delivered=1, terminal=terminal,
    )
    store = Store([outbox()], statuses=[status])
    value = coordinator(tmp_path, store)
    value._target_config = lambda *_args: config(tmp_path)
    value.publisher = SimpleNamespace(publish=lambda *_args: PublishResult(
        acknowledged=True, target=TargetKind.LOCAL, record_key="record",
        run_id="run", dashboard_url="http://dashboard",
    ))
    value.publish_once()
    assert ("ack", 1) in store.calls
    assert any(call[:3] == ("state", "local", expected) for call in store.calls)


def test_publish_once_acknowledgement_without_status_defaults_ready(tmp_path):
    store = Store([outbox()], statuses=[])
    value = coordinator(tmp_path, store)
    value._target_config = lambda *_args: config(tmp_path)
    value.publisher = SimpleNamespace(publish=lambda *_args: PublishResult(
        acknowledged=True, target=TargetKind.LOCAL, record_key="record",
        run_id="run", dashboard_url=None,
    ))
    value.publish_once()
    assert any(call[:3] == ("state", "local", "READY") for call in store.calls)


def test_collect_source_records_archive_error_without_advancing(tmp_path):
    store = Store()
    value = coordinator(tmp_path, store)
    value.archive = SimpleNamespace(
        scan=lambda *_args: (_ for _ in ()).throw(OSError("disk")),
    )
    value._collect_source(ArchiveSource(
        workspace_id="workspace", project="demo", run_id="run-a",
        attempt_id="attempt-001", name="metrics.jsonl",
        path=tmp_path / "metrics.jsonl", kind="metrics",
    ))
    assert store.calls[-1][0] == "archive_error"
    assert store.calls[-1][2] == "OSError"


def test_target_policy_handles_unrequested_missing_and_failing_credentials(tmp_path):
    value = coordinator(tmp_path)
    assert value._target_enabled(TargetKind.CLOUD) is False
    value._credential_provider = lambda _ref: (_ for _ in ()).throw(RuntimeError())
    assert value._target_enabled(TargetKind.LOCAL) is False

    value.local.publisher_entity = None
    assert value._target_config(TargetKind.LOCAL, "demo") is None
    assert value._target_config(TargetKind.CLOUD, "demo") is None


def test_target_configs_sanitize_project_name_for_local_and_cloud(tmp_path):
    value = coordinator(tmp_path)
    local = value._target_config(TargetKind.LOCAL, "bad project/name")
    assert local is not None and local.project == "bad-project-name"
    value.cloud = WandbCloudConfig(
        enabled=True, default_credential_ref="cloud", entity="team",
    )
    cloud = value._target_config(TargetKind.CLOUD, "***")
    assert cloud is not None and cloud.project == "ml-expd"


def test_source_discovery_covers_attempt_evidence_and_root_files(tmp_path):
    root = tmp_path / "run"
    attempt = root / "attempts" / "attempt-002"
    attempt.mkdir(parents=True)
    (attempt / "metrics.jsonl").write_text("{}\n")
    (attempt / "events.jsonl").write_text("{}\n")
    (attempt / "stderr.log").write_text("line\n")
    (attempt / "ignored.txt").write_text("ignored\n")
    (attempt / "nested").mkdir()
    (root / "collected_run").mkdir()
    (root / "train_metrics.jsonl").write_text("{}\n")
    (root / "events.jsonl").write_text("{}\n")
    (root / "job.out").write_text("line\n")
    (root / "ignored.txt").write_text("ignored\n")
    (root / "attempts" / "nested.log").write_text("ignored\n")
    evidence = EvidenceLayers(
        scheduler=EvidenceLayer(attempt_id="attempt-002"),
    )
    row = RunIndexRow(
        project="demo", run_id="run-a", run_dir=str(root), evidence=evidence,
    )
    sources = coordinator(tmp_path)._sources(row)
    kinds = {item.kind for item in sources}
    assert kinds == {"metrics", "events", "log"}
    assert {item.attempt_id for item in sources} == {"attempt-002"}


def test_source_discovery_skips_duplicate_and_symlink_files(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    metric = root / "metrics.jsonl"
    metric.write_text("{}\n")
    link = root / "events.jsonl"
    link.symlink_to(metric)
    value = coordinator(tmp_path)
    row = RunIndexRow(project="demo", run_id="run-a", run_dir=str(root))
    sources = value._sources(row)
    assert [item.name for item in sources] == ["metrics.jsonl"]


def test_source_discovery_defaults_attempt_and_rejects_symlink_root(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "metrics.jsonl").write_text("{}\n")
    row = RunIndexRow(project="demo", run_id="run-a", run_dir=str(root))
    sources = coordinator(tmp_path)._sources(row)
    assert sources[0].attempt_id == "attempt-001"

    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    row.run_dir = str(link)
    assert coordinator(tmp_path)._sources(row) == []


@pytest.mark.parametrize(("payload", "expected"), [
    ({"timestamp": 1}, 1.0), ({"ts": 2.5}, 2.5), ({"time": "bad"}, None),
])
def test_observed_at_aliases(payload, expected):
    assert _observed_at(payload) == expected
