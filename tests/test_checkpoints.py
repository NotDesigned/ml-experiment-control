import json

from experiment_control.checkpoints import (
    discover_latest_completed_checkpoint,
    select_latest_checkpoint_name,
)


def write_checkpoint(root, step, payload=b"payload", *, complete=True, declared_bytes=None):
    path = root / f"checkpoint_{step}"
    path.write_bytes(payload)
    if complete:
        (root / f"checkpoint_{step}.complete").write_text(
            json.dumps({"step": step, "bytes": len(payload) if declared_bytes is None else declared_bytes}),
            encoding="utf-8",
        )
    return path


def test_latest_checkpoint_requires_valid_marker_and_matching_payload(tmp_path):
    write_checkpoint(tmp_path, 8)
    write_checkpoint(tmp_path, 9, complete=False)
    write_checkpoint(tmp_path, 10, declared_bytes=999)
    latest = discover_latest_completed_checkpoint(tmp_path)
    assert latest == {
        "path": str(tmp_path / "checkpoint_8"), "step": 8,
        "bytes": 7, "completed_at": None,
    }


def test_remote_checkpoint_names_are_filtered_and_ordered():
    assert select_latest_checkpoint_name(
        ["checkpoint_9", "noise", "checkpoint_100", "checkpoint_bad"]
    ) == ("checkpoint_100", 100)


def test_checkpoint_discovery_ignores_missing_malformed_and_non_object_markers(tmp_path):
    (tmp_path / "checkpoint_1.complete").write_text("{}")
    write_checkpoint(tmp_path, 2)
    (tmp_path / "checkpoint_2.complete").write_text("not-json")
    write_checkpoint(tmp_path, 3)
    (tmp_path / "checkpoint_3.complete").write_text("[]")

    assert discover_latest_completed_checkpoint(tmp_path) is None
    assert select_latest_checkpoint_name(["noise", "checkpoint_bad"]) is None
