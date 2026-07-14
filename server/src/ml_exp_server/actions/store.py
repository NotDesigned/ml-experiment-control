"""Durable action plans and execution records with idempotent identities."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from ..schemas import OperationScope
from ..storage import atomic_json, exclusive_file_lock, read_json, utc_now


class ActionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._lock_state = threading.local()
        self.lock_path = self.root / ".actions.lock"

    @contextmanager
    def locked(self):
        """Serialize durable Action state across threads and daemon processes."""

        with self._lock:
            depth = getattr(self._lock_state, "depth", 0)
            if depth:
                self._lock_state.depth = depth + 1
                try:
                    yield
                finally:
                    self._lock_state.depth -= 1
                return
            with exclusive_file_lock(self.lock_path):
                self._lock_state.depth = 1
                try:
                    yield
                finally:
                    self._lock_state.depth = 0

    @staticmethod
    def action_id(scope: OperationScope, intent_key: str) -> str:
        raw = f"{scope.project}:{scope.scope_type.value}:{scope.object_id}:{intent_key}"
        return "action-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def directory(self, action_id: str) -> Path:
        if not action_id.startswith("action-") or not action_id[7:].isalnum():
            raise ValueError("invalid action_id")
        return self.root / action_id

    def save_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        with self.locked():
            directory = self.directory(str(plan["action_id"]))
            directory.mkdir(parents=True, exist_ok=True)
            existing = read_json(directory / "plan.json", {})
            if existing:
                if existing.get("request_digest") != plan.get("request_digest"):
                    raise RuntimeError(
                        "idempotency_key is already bound to a different operation intent"
                    )
                return existing
            atomic_json(directory / "plan.json", plan)
            atomic_json(directory / "execution.json", {
                "status": "PREPARED" if plan.get("ready") else "BLOCKED",
                "authorized_at": None,
                "authorization_note": "",
                "started_at": None,
                "finished_at": None,
                "result": None,
                "error": None,
            })
            self.append_journal(plan["action_id"], "action_prepared", {
                "ready": plan.get("ready"), "operation": plan.get("operation"),
            })
            return self.snapshot(plan["action_id"])

    def append_journal(self, action_id: str, event: str, payload: dict[str, Any]) -> None:
        with self.locked():
            path = self.directory(action_id) / "journal.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "timestamp": utc_now(), "event": event, "payload": payload,
                }, ensure_ascii=False, sort_keys=True) + "\n")

    def execution(self, action_id: str) -> dict[str, Any]:
        with self.locked():
            return read_json(self.directory(action_id) / "execution.json", {})

    def write_command(
        self, action_id: str, phase: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist sensitive/free-form input outside immutable action plans."""
        if phase not in {"authorize", "execute"}:
            raise ValueError("unsupported action command phase")
        with self.locked():
            path = self.directory(action_id) / "commands" / f"{phase}.json"
            existing = read_json(path, {})
            if existing:
                if existing.get("payload") != payload:
                    raise ValueError(f"{phase} command is already recorded for this action")
                return existing
            record = {
                "action_id": action_id,
                "phase": phase,
                "payload": payload,
                "created_at": utc_now(),
            }
            atomic_json(path, record)
            self.append_journal(action_id, f"{phase}_command_recorded", {})
            return record

    def read_command(self, action_id: str, phase: str) -> dict[str, Any]:
        if phase not in {"authorize", "execute"}:
            raise ValueError("unsupported action command phase")
        record = read_json(
            self.directory(action_id) / "commands" / f"{phase}.json", {},
        )
        if not record:
            raise FileNotFoundError(f"{action_id}:{phase}")
        payload = record.get("payload")
        return payload if isinstance(payload, dict) else {}

    def write_activity_error(
        self, action_id: str, phase: str, message: str, category: str,
    ) -> None:
        if phase not in {"prepare", "authorize", "execute", "reconcile"}:
            raise ValueError("unsupported action activity phase")
        with self.locked():
            atomic_json(self.directory(action_id) / "activity_errors" / f"{phase}.json", {
                "action_id": action_id,
                "phase": phase,
                "message": message[:1000],
                "category": category,
                "created_at": utc_now(),
            })

    def activity_error(self, action_id: str, phase: str) -> dict[str, Any]:
        return read_json(
            self.directory(action_id) / "activity_errors" / f"{phase}.json", {},
        )

    def set_execution(self, action_id: str, payload: dict[str, Any],
                      *, event: str, expected_status: str | None = None) -> dict[str, Any]:
        with self.locked():
            if expected_status is not None:
                current = self.execution(action_id)
                if current.get("status") != expected_status:
                    raise RuntimeError(
                        f"action state changed; expected {expected_status}, "
                        f"found {current.get('status')}"
                    )
            atomic_json(self.directory(action_id) / "execution.json", payload)
            self.append_journal(action_id, event, {
                "status": payload.get("status"), "error": payload.get("error"),
            })
            return self.snapshot(action_id)

    def claim_execution(self, action_id: str) -> None:
        """Cross-process, create-once claim for one immutable action intent."""
        with self.locked():
            path = self.directory(action_id) / "execution.claim"
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as exc:
                raise RuntimeError("execution intent has already been claimed") from exc
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(json.dumps({"claimed_at": utc_now(), "pid": os.getpid()}) + "\n")

    def begin_execution(
        self, action_id: str, payload: dict[str, Any], *, intent_digest: str,
    ) -> dict[str, Any]:
        """Move AUTHORIZED to EXECUTING before writing audit claim metadata."""
        with self.locked():
            current = self.execution(action_id)
            if current.get("status") != "AUTHORIZED":
                raise RuntimeError(
                    f"action state changed; expected AUTHORIZED, found {current.get('status')}"
                )
            atomic_json(self.directory(action_id) / "execution.json", payload)
            atomic_json(self.directory(action_id) / "execution.claim", {
                "claimed_at": utc_now(), "pid": os.getpid(),
                "intent_digest": intent_digest,
            })
            self.append_journal(action_id, "execution_started", {
                "status": payload.get("status"), "error": payload.get("error"),
            })
            return self.snapshot(action_id)

    def snapshot(self, action_id: str) -> dict[str, Any]:
        with self.locked():
            directory = self.directory(action_id)
            plan = read_json(directory / "plan.json", {})
            if not plan:
                raise FileNotFoundError(action_id)
            journal: list[dict[str, Any]] = []
            path = directory / "journal.jsonl"
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines()[-100:]:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        journal.append(item)
            return {**plan, "execution": self.execution(action_id), "journal": journal}

    def list_for_scope(self, scope: OperationScope) -> list[dict[str, Any]]:
        items = []
        for path in sorted(self.root.glob("action-*/plan.json")):
            payload = read_json(path, {})
            if payload.get("scope") == scope.model_dump(mode="json"):
                try:
                    items.append(self.snapshot(str(payload["action_id"])))
                except (FileNotFoundError, ValueError):
                    continue
        return items

    def list_all(self) -> list[dict[str, Any]]:
        """Return every readable action snapshot for restart reconciliation."""
        items = []
        for path in sorted(self.root.glob("action-*/plan.json")):
            payload = read_json(path, {})
            try:
                items.append(self.snapshot(str(payload["action_id"])))
            except (KeyError, FileNotFoundError, ValueError):
                continue
        return items
