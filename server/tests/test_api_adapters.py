"""Transport adapter contracts: stable HTTP errors and SSE delivery."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from ml_exp_server.api import action_routes, operation_routes
from ml_exp_server.api.sse import EventBroker
from ml_exp_server.application import ApplicationError
from ml_exp_server.schemas import OperationScope, OperationScopeType


class FakeApplication:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if self.fail:
                raise ApplicationError("adapter failure", status_code=418, code="ADAPTER_TEST")
            return {"operation": name}
        return call


def request_for(application: FakeApplication):
    state = SimpleNamespace(
        application=application,
        projects=[SimpleNamespace(project="demo")],
        config=SimpleNamespace(
            action_runtime=SimpleNamespace(model_dump=lambda: {"enabled": True}),
        ),
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def assert_http_error(call):
    with pytest.raises(HTTPException) as caught:
        call()
    assert caught.value.status_code == 418
    assert caught.value.detail == "adapter failure"


def test_action_adapter_maps_application_errors():
    request = request_for(FakeApplication(fail=True))
    scope = {
        "project": "demo", "scope_type": OperationScopeType.PROJECT, "object_id": "demo",
    }
    prepare = action_routes.PrepareActionRequest(
        **scope,
        intent={
            "kind": "CREATE_RESEARCH_QUESTION_DRAFT",
            "title": "Question",
            "draft": "id: Q1\ntitle: Question\n",
        },
    )
    authorize = action_routes.AuthorizeActionRequest(
        action_id="action-0123456789abcdef", note="ok",
    )
    execute = action_routes.ExecuteActionRequest(
        action_id="action-0123456789abcdef", confirmation="EXECUTE",
    )

    calls = [
        lambda: action_routes.list_actions(request, **scope),
        lambda: asyncio.run(action_routes.prepare_action(prepare, request)),
        lambda: action_routes.authorize_action(authorize, request),
        lambda: asyncio.run(action_routes.execute_action(execute, request)),
    ]
    for call in calls:
        assert_http_error(call)


def test_operation_adapter_delegates_and_maps_application_errors():
    application = FakeApplication()
    request = request_for(application)
    common = {
        "project": "demo", "scope_type": OperationScopeType.PROJECT, "object_id": "demo",
    }
    payload = operation_routes.OperationInvokeRequest(
        **common, operation_id="object.archive", parameters={"reason": "done"},
    )

    assert operation_routes.operation_availability(request, **common) == {
        "operation": "operation_availability",
    }
    assert asyncio.run(operation_routes.invoke_direct_operation(payload, request)) == {
        "operation": "invoke_direct_operation",
    }
    assert [call[0] for call in application.calls] == [
        "operation_availability", "invoke_direct_operation",
    ]

    failing_request = request_for(FakeApplication(fail=True))
    for call in (
        lambda: operation_routes.operation_availability(failing_request, **common),
        lambda: asyncio.run(operation_routes.invoke_direct_operation(payload, failing_request)),
    ):
        assert_http_error(call)


def test_event_broker_threadsafe_publish_and_stream():
    async def scenario():
        broker = EventBroker()
        broker.publish_threadsafe({"ignored": True})

        loop = asyncio.get_running_loop()
        broker.bind_loop(loop)
        stream = broker.stream()
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        broker.publish_threadsafe({"run_id": "run-a"})
        event = await asyncio.wait_for(pending, timeout=1)
        assert event == 'data: {"run_id": "run-a"}\n\n'
        await stream.aclose()
        assert not broker._subscribers

        broker._loop = SimpleNamespace(is_closed=lambda: True)
        broker.publish_threadsafe({"ignored": True})

    asyncio.run(scenario())


def test_event_broker_emits_keepalive(monkeypatch):
    async def timeout(awaitable, *args, **kwargs):
        awaitable.close()
        raise asyncio.TimeoutError

    async def scenario():
        broker = EventBroker()
        monkeypatch.setattr(asyncio, "wait_for", timeout)
        stream = broker.stream()
        assert await anext(stream) == ": keepalive\n\n"
        await stream.aclose()

    asyncio.run(scenario())
