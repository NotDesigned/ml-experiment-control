import hashlib
import math
import subprocess
from pathlib import Path

import pytest

from ml_exp_server.wandb_publisher import (
    AttemptIdentity,
    PublicationItem,
    PublisherProcessError,
    SubprocessWandbAdapter,
    TargetConfig,
    TargetKind,
    WandbPublisher,
    build_publisher_environment,
)


class FakeAdapter:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def publish(self, request, *, environment):
        self.calls.append((request, dict(environment)))
        if self.error is not None:
            raise self.error


@pytest.fixture
def identity():
    return AttemptIdentity("workspace-a", "science", "run-42", "attempt-002")


def target(tmp_path: Path, kind: TargetKind) -> TargetConfig:
    if kind is TargetKind.LOCAL:
        return TargetConfig(
            kind=kind,
            api_url="http://127.0.0.1:8080/api/",
            dashboard_url="http://127.0.0.1:8080/",
            entity="local-team",
            project="mirror",
            working_dir=tmp_path / "local",
        )
    return TargetConfig(
        kind=kind,
        api_url="https://api.wandb.ai",
        dashboard_url="https://wandb.ai",
        entity="cloud-team",
        project="mirror",
        working_dir=tmp_path / "cloud",
        credential_ref="team-cloud",
    )


def item(kind: TargetKind, *, sequence: int = 7) -> PublicationItem:
    return PublicationItem(
        target=kind,
        record_key="sha256:record-7",
        sequence=sequence,
        kind="metric",
        payload={"loss": 1.25, "step": 7},
        timestamp="2026-07-14T01:02:03Z",
    )


def test_attempt_identity_is_stable_and_target_independent(identity):
    expected = hashlib.sha256(
        b"workspace-a\x00science\x00run-42\x00attempt-002"
    ).hexdigest()[:32]
    assert identity.wandb_run_id == expected
    assert identity.wandb_run_id == AttemptIdentity(
        "workspace-a", "science", "run-42", "attempt-002"
    ).wandb_run_id
    assert len(identity.wandb_run_id) == 32


def test_local_publish_has_minimal_environment_and_dashboard_url(
    tmp_path: Path, identity, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("WANDB_API_KEY", "must-not-leak")
    monkeypatch.setenv("HTTPS_PROXY", "https://user:password@proxy.invalid")
    fake = FakeAdapter()
    configured = target(tmp_path, TargetKind.LOCAL)

    result = WandbPublisher(fake).publish(configured, identity, item(TargetKind.LOCAL))

    assert result.acknowledged is True
    assert result.dashboard_url == (
        f"http://127.0.0.1:8080/local-team/mirror/runs/{identity.wandb_run_id}"
    )
    request, environment = fake.calls[0]
    assert request.item.sequence == 7
    assert set(environment) == {
        "HOME", "WANDB_BASE_URL", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR",
        "WANDB_CONSOLE", "WANDB_DIR", "WANDB_MODE", "WANDB_SILENT",
    }
    assert "must-not-leak" not in repr(environment)
    assert "password" not in repr(environment)
    assert "WANDB_API_KEY" not in environment


def test_cloud_uses_only_requested_credential_and_is_independent_from_local(
    tmp_path: Path, identity,
):
    fake = FakeAdapter()
    looked_up = []

    def credentials(ref):
        looked_up.append(ref)
        return "cloud-secret"

    publisher = WandbPublisher(fake, credential_provider=credentials)
    cloud_result = publisher.publish(
        target(tmp_path, TargetKind.CLOUD), identity, item(TargetKind.CLOUD)
    )
    local_result = publisher.publish(
        target(tmp_path, TargetKind.LOCAL), identity, item(TargetKind.LOCAL)
    )

    assert cloud_result.acknowledged and local_result.acknowledged
    assert cloud_result.run_id == local_result.run_id
    assert looked_up == ["team-cloud"]
    assert fake.calls[0][1]["WANDB_API_KEY"] == "cloud-secret"
    assert "WANDB_API_KEY" not in fake.calls[1][1]


def test_missing_cloud_credential_does_not_call_adapter(tmp_path: Path, identity):
    fake = FakeAdapter()
    result = WandbPublisher(fake, credential_provider=lambda ref: None).publish(
        target(tmp_path, TargetKind.CLOUD), identity, item(TargetKind.CLOUD)
    )
    assert result.acknowledged is False
    assert result.error_class == "CredentialUnavailable"
    assert result.dashboard_url is None
    assert fake.calls == []


def test_target_mismatch_is_rejected_without_affecting_other_target(tmp_path: Path, identity):
    fake = FakeAdapter()
    result = WandbPublisher(fake).publish(
        target(tmp_path, TargetKind.LOCAL), identity, item(TargetKind.CLOUD)
    )
    assert result.acknowledged is False
    assert result.error_class == "TargetMismatch"
    assert fake.calls == []


def test_adapter_error_retains_only_sanitized_class(tmp_path: Path, identity):
    secret = "api-key-that-must-not-appear"

    class DangerousError(RuntimeError):
        pass

    fake = FakeAdapter(DangerousError(f"failed https://user:{secret}@host"))
    result = WandbPublisher(fake).publish(
        target(tmp_path, TargetKind.LOCAL), identity, item(TargetKind.LOCAL)
    )
    assert result.acknowledged is False
    assert result.error_class == "DangerousError"
    assert secret not in repr(result)


@pytest.mark.parametrize(
    "payload",
    [
        {"api_key": "secret"},
        {"nested": {"authorization": "Bearer x"}},
        {"loss": math.nan},
        {"object": object()},
        {"text": "WANDB_API_KEY=secret"},
        {"text": "failed at https://user:secret@host/path"},
    ],
)
def test_publication_item_defensively_rejects_unsafe_payload(payload):
    with pytest.raises(ValueError):
        PublicationItem(TargetKind.LOCAL, "record", 1, "metric", payload)


def test_target_urls_reject_credentials_and_dashboard_escapes_names(tmp_path: Path, identity):
    with pytest.raises(ValueError, match="user information"):
        TargetConfig(
            TargetKind.LOCAL, "https://user:secret@host/api", "https://host",
            "team", "project", tmp_path,
        )
    configured = TargetConfig(
        TargetKind.LOCAL, "https://host/api", "https://host", "team/name",
        "project name", tmp_path,
    )
    assert "/team%2Fname/project%20name/runs/" in configured.run_url(identity)


def test_subprocess_adapter_passes_json_on_stdin_and_no_parent_environment(
    tmp_path: Path, identity,
):
    seen = {}

    def runner(command, **kwargs):
        seen["command"] = command
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "", "")

    configured = target(tmp_path, TargetKind.LOCAL)
    adapter = SubprocessWandbAdapter(runner=runner)
    environment = build_publisher_environment(configured)
    request_item = item(TargetKind.LOCAL)
    adapter.publish(
        __import__("ml_exp_server.wandb_publisher", fromlist=["PublishRequest"]).PublishRequest(
            configured, identity, request_item
        ),
        environment=environment,
    )
    assert seen["command"][1:] == ["-m", "ml_exp_server.wandb_publisher", "--worker"]
    assert '"record_key":"sha256:record-7"' in seen["input"]
    assert "cloud-secret" not in seen["input"]
    assert seen["env"] == environment
    assert seen["cwd"] == str(configured.working_dir)
    assert configured.working_dir.stat().st_mode & 0o777 == 0o700


def test_subprocess_adapter_discards_child_details(tmp_path: Path, identity):
    secret = "server-secret"

    def failed(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", f"failed with {secret}")

    fake_process = SubprocessWandbAdapter(runner=failed)
    publisher = WandbPublisher(fake_process)
    result = publisher.publish(
        target(tmp_path, TargetKind.LOCAL), identity, item(TargetKind.LOCAL)
    )
    assert result.error_class == "WandbWorkerFailed"
    assert secret not in repr(result)


def test_subprocess_timeout_has_bounded_error(tmp_path: Path, identity):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output="secret")

    publisher = WandbPublisher(SubprocessWandbAdapter(runner=timeout))
    result = publisher.publish(
        target(tmp_path, TargetKind.LOCAL), identity, item(TargetKind.LOCAL)
    )
    assert result.acknowledged is False
    assert result.error_class == "PublisherTimeout"


def test_worker_payload_supports_event_and_log_records(tmp_path: Path, identity):
    from ml_exp_server.wandb_publisher import PublishRequest, _worker_log_payload

    configured = target(tmp_path, TargetKind.LOCAL)
    event = PublicationItem(
        TargetKind.LOCAL, "event-1", 8, "event", {"event": "started"}
    )
    log = PublicationItem(
        TargetKind.LOCAL, "log-1", 9, "log", {"stream": "stderr", "text": "warning"}
    )
    event_payload = PublishRequest(configured, identity, event).worker_payload()["item"]
    log_payload = PublishRequest(configured, identity, log).worker_payload()["item"]
    assert '"event":"started"' in _worker_log_payload(event_payload)["_ml_expd/event"]
    assert _worker_log_payload(log_payload) == {
        "_ml_expd/record_key": "log-1",
        "_ml_expd/kind": "log",
        "_ml_expd/log": "warning",
        "_ml_expd/stream": "stderr",
    }
