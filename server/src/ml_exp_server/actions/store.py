"""Durable action plans and execution records with idempotent identities."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4
from .database import ActionDatabase
from ..schemas import OperationScope
from ..storage import (
    atomic_json,
    exclusive_file_lock,
    read_json,
    utc_now,
)


class ActionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._lock_state = threading.local()
        self.lock_path = self.root / ".actions.lock"
        self.database = ActionDatabase(self.root)

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

    @staticmethod
    def _initial_execution(plan: dict[str, Any]) -> dict[str, Any]:
        return {
            "revision": 0,
            "status": "PREPARED" if plan.get("ready") else "BLOCKED",
            "authorized_at": None,
            "authorization_note": "",
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }

    def _execution(self, action_id: str, plan: dict[str, Any] | None = None):
        directory = self.directory(action_id)
        execution = self.database.read(action_id)
        if not execution:
            plan = plan or read_json(directory / "plan.json", {})
            if plan:
                execution = self.database.commit(
                    action_id, self._initial_execution(plan), event={
                        "timestamp": utc_now(), "event": "action_prepared",
                        "payload": {"ready": plan.get("ready"), "operation": plan.get("operation")},
                    },
                )
        return execution

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
                return self.snapshot(str(plan["action_id"]))
            atomic_json(directory / "plan.json", plan)
            return self.snapshot(plan["action_id"])

    def append_journal(
        self, action_id: str, event: str, payload: dict[str, Any], *,
        event_id: str | None = None,
    ) -> None:
        with self.locked():
            self.directory(action_id)
            self.database.append_event(action_id, {
                "timestamp": utc_now(), "event": event, "payload": payload,
            }, event_id=event_id)

    def execution(self, action_id: str) -> dict[str, Any]:
        with self.locked():
            return self._execution(action_id)

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
                record = existing
            else:
                record = {
                    "action_id": action_id,
                    "phase": phase,
                    "payload": payload,
                    "created_at": utc_now(),
                    "journal_event_id": uuid4().hex,
                }
                atomic_json(path, record)
            event_id = str(record.get("journal_event_id") or "")
            if not event_id:
                encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
                event_id = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            self.append_journal(
                action_id, f"{phase}_command_recorded", {}, event_id=event_id,
            )
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

    def _transition(self, action_id, payload, *, event, expected_status=None,
                    timestamp=None, details=None):
        self._execution(action_id)  # finish an interrupted initial plan write
        self.database.commit(action_id, payload, expected_status=expected_status, event={
            "timestamp": timestamp or utc_now(), "event": event,
            "payload": {"status": payload.get("status"), "error": payload.get("error"),
                        **(details or {})},
        })

    def set_execution(self, action_id: str, payload: dict[str, Any],
                      *, event: str, expected_status: str | None = None) -> dict[str, Any]:
        with self.locked():
            self._transition(action_id, payload, event=event, expected_status=expected_status)
            return self.snapshot(action_id)

    def begin_execution(
        self, action_id: str, payload: dict[str, Any], *, intent_digest: str,
    ) -> dict[str, Any]:
        """Claim execution with the same SQLite CAS used by all transitions."""
        with self.locked():
            claim = {"claimed_at": utc_now(), "pid": os.getpid(), "intent_digest": intent_digest}
            self._transition(
                action_id, payload, event="execution_started", expected_status="AUTHORIZED",
                timestamp=claim["claimed_at"], details={"intent_digest": intent_digest},
            )
            try:
                atomic_json(self.directory(action_id) / "execution.claim", claim)
            except OSError:
                # State and audit already contain the authoritative claim.
                pass
            return self.snapshot(action_id)

    def snapshot(self, action_id: str) -> dict[str, Any]:
        with self.locked():
            directory = self.directory(action_id)
            plan = read_json(directory / "plan.json", {})
            if not plan:
                raise FileNotFoundError(action_id)
            execution = self._execution(action_id, plan)
            journal = self.database.journal(action_id)
            return {**plan, "execution": execution, "journal": journal}

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
