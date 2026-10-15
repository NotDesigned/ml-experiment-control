"""An Attempt query never borrows another Attempt's current Run summary."""
import json
from types import SimpleNamespace

import pytest

from ml_exp_server.runs.queries import RunQueries


@pytest.mark.parametrize("collection", [{}, {"latest_completed_checkpoint_step": 0,
    "latest_completed_checkpoint": "checkpoint-zero", "artifacts": {"files": 1}}])
def test_checkpoint_and_artifact_summaries_are_exact_attempt(tmp_path, collection):
    root = tmp_path / "attempts" / "attempt-001"
    root.mkdir(parents=True)
    (root / "collection.json").write_text(json.dumps(collection))
    row = SimpleNamespace(run_id="trial", checkpoint={
        "latest_completed_checkpoint": "attempt-002/checkpoint-final",
        "latest_completed_checkpoint_step": 100}, artifacts={"files": 99})
    query = RunQueries()
    query._attempt_context = lambda *args: (None, None, row,
        SimpleNamespace(attempt_id="attempt-001"), root)
    query._read_mapping = lambda path: json.loads(path.read_text())
    checkpoint = query.attempt_checkpoints("demo", "trial::attempt-001")
    artifacts = query.attempt_artifacts("demo", "trial::attempt-001")
    assert checkpoint["latest_completed_checkpoint"] == collection.get("latest_completed_checkpoint")
    assert checkpoint["latest_completed_checkpoint_step"] == collection.get("latest_completed_checkpoint_step")
    assert artifacts["summary"] == collection.get("artifacts", {})
