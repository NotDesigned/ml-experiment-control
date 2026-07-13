"""Provider epoch migration and durable turn-request identity coverage."""

import json

import pytest

from ml_exp_server.agents.store import AgentStore
from ml_exp_server.schemas import AgentScope, AgentScopeType


SCOPE = AgentScope(
    project="demo", scope_type=AgentScopeType.RUN, object_id="run-a",
)


def test_provider_epoch_archives_codex_conversation_without_touching_proposals(tmp_path):
    store = AgentStore(tmp_path)
    store.ensure(SCOPE, default_goal="goal")
    store.append_message(SCOPE, role="user", content="legacy", thread_id="codex-thread")
    proposal = store.add_proposals(SCOPE, [{
        "kind": "ANALYSIS_ONLY", "title": "keep", "draft": "",
    }], evidence_digest="sha256:old")[0]

    snapshot = store.begin_provider_epoch(SCOPE, "openai_agents")

    assert snapshot["conversation_provider"] == "openai_agents"
    assert snapshot["conversation_epoch"] == 1
    assert snapshot["messages"] == []
    assert snapshot["thread_id"] is None
    assert snapshot["archived_conversation_epochs"] == 1
    assert [item["proposal_id"] for item in snapshot["proposals"]] == [
        proposal["proposal_id"],
    ]
    archive = store.agent_dir(SCOPE) / "conversation_epochs" / "epoch-0000.json"
    assert json.loads(archive.read_text())["messages"][0]["content"] == "legacy"

    repeated = store.begin_provider_epoch(SCOPE, "openai_agents")
    assert repeated["conversation_epoch"] == 1
    assert repeated["archived_conversation_epochs"] == 1


def test_turn_requests_are_durable_and_validate_identity(tmp_path):
    store = AgentStore(tmp_path)
    store.ensure(SCOPE, default_goal="goal")
    request = store.create_turn_request(
        SCOPE, message="analyze", enforce_operation_availability=True,
    )
    loaded = store.turn_request(SCOPE, request["request_id"])
    assert loaded["status"] == "PENDING"
    assert loaded["scope"]["object_id"] == "run-a"
    completed = store.set_turn_request(
        SCOPE, request["request_id"], status="COMPLETED",
        result={"created_proposals": []},
    )
    assert completed["result"] == {"created_proposals": []}
    with pytest.raises(ValueError, match="invalid turn request"):
        store.turn_request(SCOPE, "../escape")
