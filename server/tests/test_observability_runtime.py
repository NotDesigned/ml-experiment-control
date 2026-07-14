from pathlib import Path

from ml_exp_server.observability_runtime import ObservabilityCoordinator
from ml_exp_server.observability_store import ObservabilityStore
from ml_exp_server.schemas import (
    AttemptSummary,
    LocalWandbConfig,
    RunIndexRow,
    WandbCloudConfig,
)
from ml_exp_server.wandb_publisher import WandbPublisher


class _FakeAdapter:
    def __init__(self):
        self.records = []

    def publish(self, request, *, environment):
        self.records.append((request, environment))


def _coordinator(
    tmp_path: Path, store: ObservabilityStore, *, local_enabled=False,
    credential_ready=True,
):
    local = (
        LocalWandbConfig(
            enabled=True,
            managed=False,
            external_url="http://127.0.0.1:8080",
            publisher_entity="research",
            publisher_credential_ref="local-primary",
        )
        if local_enabled else LocalWandbConfig()
    )
    return ObservabilityCoordinator(
        workspace_id="workspace",
        archive_root=tmp_path / "archive",
        store=store,
        local=local,
        cloud=WandbCloudConfig(),
        credential_provider=lambda reference: (
            "local-secret"
            if credential_ready and reference == "local-primary" else None
        ),
    )


def test_collect_restart_partial_line_sanitization_and_late_cloud_target(tmp_path: Path):
    run_dir = tmp_path / "run"
    attempt_dir = run_dir / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    metrics = attempt_dir / "train_metrics.jsonl"
    metrics.write_text('{"step":1,"loss":2.0}\n{"step":2', encoding="utf-8")
    (attempt_dir / "stdout.log").write_text(
        "WANDB_API_KEY=top-secret\n", encoding="utf-8",
    )
    row = RunIndexRow(
        project="demo", run_id="run-a", run_dir=str(run_dir),
        attempts=[AttemptSummary(attempt_id="attempt-001")],
    )
    store = ObservabilityStore(tmp_path / "observability.sqlite")
    first = _coordinator(tmp_path, store)
    first.collect_rows([row])
    # Archive remains active, but a disabled publisher creates no misleading
    # target and no unbounded Local outbox backlog.
    assert store.statuses(limit=10) == []

    archived = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "archive").rglob("*.json")
    )
    assert "top-secret" not in archived
    assert "[REDACTED]" in archived

    # Recreating the coordinator uses the durable byte cursors and cannot
    # duplicate already archived/outbox records.
    second = _coordinator(tmp_path, store)
    second.collect_rows([row])
    assert store.statuses(limit=10) == []

    second.enable_cloud("demo", "run-a", "attempt-001")
    with metrics.open("a", encoding="utf-8") as handle:
        handle.write(',"loss":1.5}\n')
    second.collect_rows([row])
    by_target = {item.target: item for item in store.statuses(limit=10)}
    # First activation rewinds this Attempt atomically, so Cloud receives the
    # complete sanitized history plus the newly completed metric.
    assert by_target["cloud"].pending == 3


def test_backfill_replays_archive_after_canonical_source_is_truncated(tmp_path: Path):
    run_dir = tmp_path / "run"
    attempt_dir = run_dir / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    metrics = attempt_dir / "metrics.jsonl"
    metrics.write_text('{"step":1,"loss":2}\n', encoding="utf-8")
    row = RunIndexRow(
        project="demo", run_id="run-a", run_dir=str(run_dir),
        attempts=[AttemptSummary(attempt_id="attempt-001")],
    )
    store = ObservabilityStore(tmp_path / "observability.sqlite")
    coordinator = _coordinator(tmp_path, store)
    coordinator.collect_rows([row])
    metrics.write_text("", encoding="utf-8")

    coordinator.enable_cloud("demo", "run-a", "attempt-001")

    claimed = store.claim("cloud", "worker", limit=10)
    assert len(claimed) == 1
    assert claimed[0].payload == {"step": 1, "loss": 2}


def test_local_publisher_acknowledges_outbox_and_sets_dashboard_url(tmp_path: Path):
    run_dir = tmp_path / "run"
    attempt_dir = run_dir / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "metrics.jsonl").write_text('{"step":1,"loss":2}\n')
    row = RunIndexRow(
        project="demo", run_id="run-a", run_dir=str(run_dir),
        attempts=[AttemptSummary(attempt_id="attempt-001")],
    )
    store = ObservabilityStore(tmp_path / "observability.sqlite")
    coordinator = _coordinator(tmp_path, store, local_enabled=True)
    fake = _FakeAdapter()
    coordinator.publisher = WandbPublisher(
        fake, credential_provider=lambda _reference: "local-secret",
    )
    coordinator.collect_rows([row])
    coordinator.publish_once(limit_per_target=10)

    status = store.statuses(limit=10)[0]
    assert status.state == "READY"
    assert status.pending == 0
    assert status.delivered == 1
    assert status.dashboard_url and "/research/demo/runs/" in status.dashboard_url
    assert len(fake.records) == 1


def test_requested_local_target_queues_before_credential_is_ready(tmp_path: Path):
    run_dir = tmp_path / "run"
    attempt_dir = run_dir / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "metrics.jsonl").write_text('{"step":1,"loss":2}\n')
    row = RunIndexRow(
        project="demo", run_id="run-a", run_dir=str(run_dir),
        attempts=[AttemptSummary(attempt_id="attempt-001")],
    )
    store = ObservabilityStore(tmp_path / "observability.sqlite")
    coordinator = _coordinator(
        tmp_path, store, local_enabled=True, credential_ready=False,
    )
    coordinator.collect_rows([row])
    status = store.statuses(limit=10)[0]
    assert status.target == "local"
    assert status.pending == 1
    coordinator.publish_once()
    assert store.statuses(limit=10)[0].pending == 1


def test_failed_lower_sequence_blocks_higher_sequence(tmp_path: Path):
    run_dir = tmp_path / "run"
    attempt_dir = run_dir / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "metrics.jsonl").write_text(
        '{"step":1,"loss":2}\n{"step":2,"loss":1}\n'
    )
    row = RunIndexRow(
        project="demo", run_id="run-a", run_dir=str(run_dir),
        attempts=[AttemptSummary(attempt_id="attempt-001")],
    )
    store = ObservabilityStore(tmp_path / "observability.sqlite")
    coordinator = _coordinator(tmp_path, store, local_enabled=True)

    class FailingAdapter(_FakeAdapter):
        def publish(self, request, *, environment):
            super().publish(request, environment=environment)
            raise RuntimeError("failure")

    fake = FailingAdapter()
    coordinator.publisher = WandbPublisher(
        fake, credential_provider=lambda _reference: "local-secret",
    )
    coordinator.collect_rows([row])
    coordinator.publish_once(limit_per_target=10)
    coordinator.publish_once(limit_per_target=10)

    status = store.statuses(limit=10)[0]
    assert status.state == "DEGRADED"
    assert status.pending == 2
    assert len(fake.records) == 1
