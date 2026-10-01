"""Explicit Action migration and stopped-service rollback export.

Usage: python -m ml_exp_server.actions.migrate --root ACTION_ROOT
       python -m ml_exp_server.actions.migrate --root ACTION_ROOT --export-legacy NEW_ROOT
Neither command executes or authorizes an Action. See docs/action-storage.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ..storage import StorageError, atomic_json, atomic_text
from .store import ActionStore


def migrate(root: Path, export_legacy: Path | None = None) -> dict:
    store = ActionStore(root)
    with store.locked():
        snapshots = store.list_all()
        if len(snapshots) != len(list(root.glob("action-*/plan.json"))):
            raise StorageError("unreadable Action plan; migration incomplete")
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
                state = store._execution_state(action_id)
                journal = state.journal(limit=-1)
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
