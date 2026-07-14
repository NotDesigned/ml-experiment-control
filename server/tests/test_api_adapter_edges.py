"""Transport adapter branches that are independent of domain behavior."""

from __future__ import annotations

import asyncio
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from ml_exp_server.api import action_routes, routes, submission_routes
from ml_exp_server.application import ApplicationError
from ml_exp_server.schemas import ActionRuntimeConfig, OperationScopeType


ERROR = ApplicationError("blocked", status_code=418, code="BLOCKED")


class Failing:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: (_ for _ in ()).throw(ERROR)


def request(*, application=None, submission_service=None, **state):
    values = {
        "application": application or Failing(),
        "submission_service": submission_service or Failing(),
        "config": SimpleNamespace(action_runtime=ActionRuntimeConfig()),
    }
    values.update(state)
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(**values)),
        url=SimpleNamespace(scheme="http"),
    )


def assert_http_error(call):
    with pytest.raises(HTTPException) as caught:
        call()
    assert caught.value.status_code == 418


def test_action_route_policy_and_reconcile_error():
    req = request()
    assert action_routes.action_policy(req)["allow_project_writes"] is False
    data = action_routes.ActionRequest(action_id="action-0123456789abcdef")
    assert_http_error(lambda: asyncio.run(action_routes.reconcile_action(data, req)))
    assert_http_error(lambda: action_routes.list_actions(
        req, "p", OperationScopeType.RUN, "r",
    ))
    prepared = action_routes.PrepareActionRequest(
        project="p", scope_type=OperationScopeType.RUN, object_id="r",
        intent={
            "kind": "CREATE_CAMPAIGN_DRAFT", "title": "title", "target": "target",
            "change_summary": "change", "resource_estimate": "none",
            "rationale": "reason", "risk": "risk", "draft": "draft",
            "evidence_digest": "sha256:evidence", "idempotency_key": "key",
        },
    )
    assert_http_error(lambda: asyncio.run(action_routes.prepare_action(prepared, req)))
    authorized = action_routes.AuthorizeActionRequest(
        action_id="action-0123456789abcdef",
    )
    assert_http_error(lambda: action_routes.authorize_action(authorized, req))


def test_submission_routes_map_every_application_error():
    req = request()
    assert_http_error(lambda: submission_routes.list_submissions("p", "r", req))
    assert_http_error(lambda: asyncio.run(submission_routes.prepare_submission(
        "p", "r", submission_routes.PrepareSubmissionRequest(max_gpu_hours=1), req,
    )))
    assert_http_error(lambda: submission_routes.get_submission("s", req))
    assert_http_error(lambda: submission_routes.authorize_submission(
        "s", submission_routes.AuthorizeSubmissionRequest(), req,
    ))
    assert_http_error(lambda: asyncio.run(submission_routes.execute_submission(
        "s", submission_routes.ExecuteSubmissionRequest(confirmation="yes"), req,
    )))
    assert_http_error(lambda: asyncio.run(
        submission_routes.reconcile_submission("s", req),
    ))


def test_health_source_version_fallback(monkeypatch):
    monkeypatch.setattr(
        "importlib.metadata.version",
        lambda _name: (_ for _ in ()).throw(PackageNotFoundError()),
    )
    runtime = SimpleNamespace(
        workspace_id="workspace", telemetry=SimpleNamespace(enabled=False),
        wandb_service=SimpleNamespace(status=lambda: {"state": "DISABLED"}),
    )
    req = request(
        runtime=runtime, auth_mode="none", projects=[], collector=None,
        collector_owner=False, collector_error=None,
        index=SimpleNamespace(get_meta=lambda _key: None),
        project_write_recovery_errors=[],
        publisher_last_success_at=None, publisher_last_error=None,
        publisher_consecutive_failures=0,
    )
    assert routes.health(req).server_version == "0.1.0+source"


class Status:
    def __init__(self, target, state):
        self.target = target
        self.state = state
        self.pending = 0
        self.terminal = 0


@pytest.mark.parametrize(("states", "expected"), [
    ([], "PENDING"),
    ([Status("local", "FAILED")], "FAILED"),
    ([Status("local", "READY")], "READY"),
    ([Status("local", "OTHER")], "PENDING"),
])
def test_observability_payload_publisher_state_matrix(states, expected):
    local = SimpleNamespace(
        enabled=True, publisher_entity="team", publisher_credential_ref="local",
        url=lambda: "http://127.0.0.1:8080",
    )
    cloud = SimpleNamespace(
        enabled=False, default_credential_ref=None, entity=None,
        dashboard_url="https://wandb.ai",
    )
    store = SimpleNamespace(archive_summary=lambda: {"degraded_sources": 0})
    runtime = SimpleNamespace(
        config=SimpleNamespace(observability=SimpleNamespace(
            local_wandb=local, wandb_cloud=cloud,
        )),
        credential_store=SimpleNamespace(
            status=lambda _ref: SimpleNamespace(configured=True),
        ),
        observability_store=store,
        wandb_service=SimpleNamespace(status=lambda: {"state": "READY"}),
    )
    payload = routes._observability_payload(
        request(runtime=runtime), states, target_total=len(states),
    )
    assert payload["local_wandb"]["publisher_state"] == expected
    assert payload["cloud"]["state"] == "DISABLED"


def test_observability_payload_unavailable_and_dashboard_url_validation():
    assert routes._public_dashboard_url("http://example.com:bad") is None
    assert routes._public_dashboard_url("https://example.com/dashboard") == (
        "https://example.com/dashboard"
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(observability=SimpleNamespace(
            local_wandb=SimpleNamespace(
                enabled=True, publisher_entity="team",
                publisher_credential_ref="local", url=lambda: "http://localhost",
            ),
            wandb_cloud=SimpleNamespace(
                enabled=True, default_credential_ref="cloud", entity="team",
                dashboard_url="https://wandb.ai",
            ),
        )),
        credential_store=SimpleNamespace(
            status=lambda _ref: SimpleNamespace(configured=False),
        ),
        observability_store=SimpleNamespace(
            archive_summary=lambda: {"degraded_sources": 0},
        ),
        wandb_service=SimpleNamespace(status=lambda: {"state": "READY"}),
    )
    payload = routes._observability_payload(request(runtime=runtime), [], target_total=0)
    assert payload["local_wandb"]["publisher_state"] == "UNAVAILABLE"
    assert payload["cloud"]["state"] == "UNAVAILABLE"


def test_route_error_adapters_and_attempt_view_dispatch():
    req = request()
    with pytest.raises(HTTPException) as missing:
        routes._run_or_404(SimpleNamespace(get_run=lambda *_args: None), "p", "r")
    assert missing.value.status_code == 404
    row = object()
    assert routes._run_or_404(
        SimpleNamespace(get_run=lambda *_args: row), "p", "r",
    ) is row
    data = routes.ProjectLifecycleRequest()
    assert_http_error(lambda: routes.project_lifecycle_unregister("p", data, req))
    assert_http_error(lambda: routes.object_show(
        req, "p", OperationScopeType.RUN, "r",
    ))
    assert_http_error(lambda: routes.run_attempts("p", "r", req))
    assert_http_error(lambda: routes.run_validate("p", "r", req))
    assert_http_error(lambda: routes.attempt_bundle("p", "a", req))
    assert_http_error(lambda: routes.attempt_retry(
        "p", "a", routes.AttemptRetryRequest(max_gpu_hours=1), req,
    ))
    assert_http_error(lambda: routes.attempt_cancel(
        "p", "a", routes.AttemptCancelRequest(), req,
    ))

    class Views:
        def __getattr__(self, name):
            return lambda *_args, **_kwargs: name

    view_request = request(application=Views())
    expected = {
        "show": "attempt_show", "logs": "attempt_logs",
        "checkpoints": "attempt_checkpoints", "artifacts": "attempt_artifacts",
        "metrics": "attempt_metrics", "eval": "attempt_eval",
        "events": "attempt_events", "validate": "attempt_validate",
    }
    for view, result in expected.items():
        assert routes.attempt_view("p", "a", view, view_request) == result
    assert_http_error(lambda: routes.attempt_view("p", "a", "show", req))
    with pytest.raises(HTTPException, match="unknown Attempt view"):
        routes.attempt_view("p", "a", "unknown", view_request)


def test_terminal_refresh_rejects_unknown_project():
    req = request(projects=[], index=object())
    with pytest.raises(HTTPException, match="unknown project"):
        routes.terminal_refresh(routes.RefreshRequest(project="missing"), req)


def test_observability_endpoint_and_attention_ignore_unmatched_collector_status():
    store = SimpleNamespace(
        statuses=lambda **_kwargs: [], status_count=lambda **_kwargs: 0,
        archive_summary=lambda: {"degraded_sources": 0},
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(observability=SimpleNamespace(
            local_wandb=SimpleNamespace(
                enabled=False, publisher_entity=None,
                publisher_credential_ref=None,
            ),
            wandb_cloud=SimpleNamespace(
                enabled=False, default_credential_ref=None, entity=None,
            ),
        )),
        credential_store=SimpleNamespace(), observability_store=store,
        wandb_service=SimpleNamespace(status=lambda: {"state": "DISABLED"}),
    )
    payload = routes.observability(request(runtime=runtime))
    assert payload["limits"]["target_statuses"]["returned"] == 0
    status = SimpleNamespace(run_id="missing", last_error="ignored")
    assert routes._attention([], [status]) == []


def test_terminal_snapshot_groups_target_status(monkeypatch):
    target = SimpleNamespace(
        attempt=SimpleNamespace(project="demo", run_id="run-a", attempt_id="a1"),
        target="local", state="READY", dashboard_url="https://example.com",
        pending=0, delivered=1, terminal=0, updated_at=1, last_error=None,
    )
    payload = {"projects": [], "runs": {"demo": [{
        "run_id": "run-a", "attempts": [],
    }]}}
    monkeypatch.setattr(routes, "build_snapshot", lambda *_args: object())
    monkeypatch.setattr(routes, "snapshot_payload", lambda _snapshot: payload)
    store = SimpleNamespace(
        statuses=lambda **_kwargs: [target], status_count=lambda **_kwargs: 1,
        archive_summary=lambda: {"degraded_sources": 0},
    )
    runtime = SimpleNamespace(
        observability_store=store,
        config=SimpleNamespace(observability=SimpleNamespace(
            local_wandb=SimpleNamespace(
                enabled=False, publisher_entity=None, publisher_credential_ref=None,
            ),
            wandb_cloud=SimpleNamespace(
                enabled=False, default_credential_ref=None, entity=None,
            ),
        )),
        credential_store=SimpleNamespace(),
        wandb_service=SimpleNamespace(status=lambda: {"state": "DISABLED"}),
    )
    req = request(
        runtime=runtime, projects=[SimpleNamespace(project="demo")],
        index=SimpleNamespace(get_run=lambda *_args: None),
    )
    result = routes.terminal_snapshot(req)
    assert result["runs"]["demo"][0]["observability"]["attempts"]["a1"][0][
        "state"
    ] == "READY"
