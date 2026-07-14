"""Validation and isolated-worker edges for W&B publication."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server import wandb_publisher as module
from ml_exp_server.wandb_publisher import (
    AttemptIdentity,
    PublicationItem,
    PublishRequest,
    PublisherProcessError,
    SubprocessWandbAdapter,
    TargetConfig,
    TargetKind,
    WandbPublisher,
    _contains_url_userinfo,
    _json_copy,
    _run_worker,
    _sanitize_error_class,
    _worker_log_payload,
    build_publisher_environment,
)


def test_payload_helpers_cover_empty_sequences_plain_values_and_multiple_urls():
    module._validate_payload([])
    assert _json_copy([1, {"value": True}]) == [1, {"value": True}]
    assert _contains_url_userinfo(
        "https://example.com then https://user:secret@example.org"
    ) is True


def identity():
    return AttemptIdentity("workspace", "demo", "run-a", "attempt-001")


def target(tmp_path: Path, kind=TargetKind.LOCAL, **updates):
    values = {
        "kind": kind,
        "api_url": "http://127.0.0.1:8080",
        "dashboard_url": "http://127.0.0.1:8080",
        "entity": "team",
        "project": "mirror",
        "working_dir": tmp_path,
        "credential_ref": "cloud" if kind is TargetKind.CLOUD else None,
    }
    values.update(updates)
    return TargetConfig(**values)


def item(target_kind=TargetKind.LOCAL, **updates):
    values = {
        "target": target_kind, "record_key": "record", "sequence": 1,
        "kind": "metric", "payload": {"loss": 1.0},
    }
    values.update(updates)
    return PublicationItem(**values)


@pytest.mark.parametrize("values", [
    ("", "demo", "run-a", "attempt-001"),
    ("workspace", "bad\0project", "run-a", "attempt-001"),
])
def test_attempt_identity_rejects_empty_or_nul_fields(values):
    with pytest.raises(ValueError, match="non-empty string"):
        AttemptIdentity(*values)


@pytest.mark.parametrize(("updates", "message"), [
    ({"record_key": ""}, "record_key"),
    ({"record_key": "x" * 257}, "record_key"),
    ({"sequence": -1}, "sequence"),
    ({"kind": "unknown"}, "unsupported publication kind"),
    ({"payload": {1: "value"}}, "keys must be strings"),
    ({"payload": {"values": [1, float("inf")]}}, "non-finite"),
])
def test_publication_item_rejects_identity_and_nested_payload_edges(updates, message):
    with pytest.raises(ValueError, match=message):
        item(**updates)


def test_worker_payload_revalidates_mutated_nested_container(tmp_path):
    nested = {"safe": "value"}
    publication = item(payload={"nested": nested})
    nested["api_key"] = "secret"
    with pytest.raises(ValueError, match="secret-like"):
        PublishRequest(target(tmp_path), identity(), publication).worker_payload()


def test_target_requires_entity_project_and_cloud_credential(tmp_path):
    with pytest.raises(ValueError, match="entity and project"):
        target(tmp_path, entity="")
    with pytest.raises(ValueError, match="credential reference"):
        target(tmp_path, TargetKind.CLOUD, credential_ref=None)


def test_subprocess_adapter_validates_timeout_and_maps_spawn_error(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        SubprocessWandbAdapter(timeout_seconds=0)
    adapter = SubprocessWandbAdapter(
        runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("spawn")),
    )
    with pytest.raises(PublisherProcessError) as caught:
        adapter.publish(
            PublishRequest(target(tmp_path), identity(), item()), environment={},
        )
    assert caught.value.error_class == "OSError"


def test_publisher_handles_missing_and_failing_credential_provider(tmp_path):
    class Adapter:
        def publish(self, *_args, **_kwargs):
            raise AssertionError("must not publish")

    configured = target(tmp_path, TargetKind.CLOUD)
    no_provider = WandbPublisher(Adapter()).publish(
        configured, identity(), item(TargetKind.CLOUD),
    )
    assert no_provider.error_class == "CredentialUnavailable"

    failing = WandbPublisher(
        Adapter(), credential_provider=lambda _ref: (_ for _ in ()).throw(KeyError("secret")),
    ).publish(configured, identity(), item(TargetKind.CLOUD))
    assert failing.error_class == "KeyError"


def test_cloud_environment_requires_key_and_local_can_receive_explicit_key(tmp_path):
    with pytest.raises(ValueError, match="requires an API key"):
        build_publisher_environment(target(tmp_path, TargetKind.CLOUD))
    environment = build_publisher_environment(target(tmp_path), api_key="explicit")
    assert environment["WANDB_API_KEY"] == "explicit"


@pytest.mark.parametrize(("updates", "message"), [
    ({"api_url": "relative"}, "absolute HTTP"),
    ({"dashboard_url": "https://example.com/path?secret=x"}, "query or fragment"),
    ({"api_url": "https://example.com:bad"}, "invalid port"),
])
def test_target_rejects_malformed_base_urls(tmp_path, updates, message):
    with pytest.raises(ValueError, match=message):
        target(tmp_path, **updates)


def test_url_and_error_sanitizers_fail_closed():
    assert _contains_url_userinfo("request http://[bad") is True
    assert _sanitize_error_class("***") == "PublisherError"


def test_worker_log_payload_covers_metric_and_optional_log_fields():
    assert _worker_log_payload({
        "record_key": "metric", "kind": "metric", "payload": {"loss": 1},
        "timestamp": "now",
    }) == {
        "_ml_expd/record_key": "metric", "_ml_expd/kind": "metric",
        "_ml_expd/timestamp": "now", "loss": 1,
    }
    assert _worker_log_payload({
        "record_key": "log", "kind": "log", "payload": {"text": "line"},
        "timestamp": None,
    })["_ml_expd/log"] == "line"


def test_isolated_worker_success_none_and_failure(monkeypatch, tmp_path):
    request = PublishRequest(target(tmp_path), identity(), item()).worker_payload()
    calls = []

    class Run:
        def log(self, payload, **kwargs):
            calls.append((payload, kwargs))

        def finish(self, **kwargs):
            calls.append(kwargs)

    fake = SimpleNamespace(
        Settings=lambda **kwargs: kwargs,
        init=lambda **kwargs: Run(),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    assert _run_worker(json.dumps(request)) == 0
    assert calls

    fake.init = lambda **kwargs: None
    assert _run_worker(json.dumps(request)) == 2
    assert _run_worker("not-json") == 1


def test_worker_main_dispatches_only_explicit_worker_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["publisher", "--worker"])
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(read=lambda: "not-json"))
    assert module._main() == 1
    monkeypatch.setattr(sys, "argv", ["publisher"])
    assert module._main() == 2
