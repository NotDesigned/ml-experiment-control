"""Explicit Action migration and stopped-service rollback export.

Usage: python -m ml_exp_server.actions.migrate --root ACTION_ROOT
       python -m ml_exp_server.actions.migrate --root ACTION_ROOT --export-legacy NEW_ROOT
Neither command executes or authorizes an Action. See docs/action-storage.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from ..storage import DurableJsonState, StorageError, _jsonl_mappings, atomic_json, atomic_text, read_json
from .database import _encode
from .store import ActionStore


def _read_legacy(directory: Path):
    # Legacy recovery may normalize a torn journal tail; use private copies.
    with tempfile.TemporaryDirectory(prefix=".import-", dir=directory.parent) as temporary:
        copied = Path(temporary)
        for name in ("execution.json", "journal.jsonl"):
            source = directory / name
            if source.is_file():
                shutil.copyfile(source, copied / name)
        legacy = DurableJsonState(copied / "execution.json", copied / "journal.jsonl")
        snapshot = legacy.snapshot({})
        legacy.repair_journal(snapshot)
        events = _jsonl_mappings(copied / "journal.jsonl")
    if snapshot.value and snapshot.last_transition is None and any("transition_id" in event for event in events):
        raise StorageError(f"Action journal has transitions without authoritative metadata: {directory.name}")
    value = dict(snapshot.value)
    if value:
        if value.get("revision", snapshot.revision) != snapshot.revision:
            raise StorageError(f"execution revision does not match durable state: {directory.name}")
        value["revision"] = snapshot.revision
    elif events:
        raise StorageError(f"Action journal exists without execution state: {directory.name}")
    return value, snapshot.revision, events


def _import_legacy(store: ActionStore):
    paths = sorted(store.root.glob("action-*/plan.json"))
    with store.database.transaction() as connection:
        for path in paths:
            plan = read_json(path, {})
            if not isinstance(plan, dict) or plan.get("action_id") != path.parent.name:
                raise StorageError("unreadable Action plan; migration incomplete")
            action_id = str(plan["action_id"])
            store.directory(action_id)
            if connection.execute("SELECT 1 FROM executions WHERE action_id=?", (action_id,)).fetchone():
                continue  # never overwrite committed SQLite with legacy files
            value, revision, events = _read_legacy(path.parent)
            connection.execute("INSERT INTO executions VALUES (?,?,?)", (action_id, revision, _encode(value)))
            for index, event in enumerate(events):
                key = ("transition:" + event["transition_id"] if "transition_id" in event
                       else "event:" + event["journal_event_id"] if "journal_event_id" in event
                       else f"legacy:{index}")
                connection.execute(
                    "INSERT INTO events(action_id,event_key,revision,payload) VALUES (?,?,?,?)",
                    (action_id, key, event.get("revision") if "transition_id" in event else None, _encode(event)),
                )


def migrate(root: Path, export_legacy: Path | None = None) -> dict:
    store = ActionStore(root)
    with store.locked():
        _import_legacy(store)
        snapshots = store.list_all()
        # Initialize the database even for an empty workspace.
        with store.database.transaction() as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise StorageError("Action database integrity check failed")
        if export_legacy is not None:
            destination = export_legacy.resolve()
            if destination == root.resolve() or root.resolve() in destination.parents:
                raise ValueError("export destination must be outside the Action root")
            # copytree refuses an existing destination. No live state is overwritten.
            shutil.copytree(root, destination, ignore=shutil.ignore_patterns(
                "actions.sqlite3*", ".sqlite-authoritative", ".actions.lock", ".import-*",
            ))
            for item in snapshots:
                action_id = item["action_id"]
                journal = store.database.journal(action_id, limit=-1)
                last = next((event for event in reversed(journal) if "transition_id" in event), None)
                execution = item["execution"]
                if last is not None:
                    execution = {**execution, "_durability": {
                        "revision": execution["revision"], "last_transition": last,
                    }}
                atomic_json(destination / action_id / "execution.json", execution)
                atomic_text(destination / action_id / "journal.jsonl", "".join(
                    json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in journal
                ))
            atomic_json(destination / "rollback-export.json", {
                "format": "legacy-action-json", "actions": len(snapshots), "complete": True,
            })
        return {"actions": len(snapshots), "database": str(store.database.path),
                "exported": export_legacy is not None}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--export-legacy", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(migrate(args.root, args.export_legacy)))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
