"""Durable action plans and execution records with idempotent identities."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..agents.store import utc_now
from ..schemas import AgentScope


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


class ActionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def action_id(scope: AgentScope, proposal_id: str) -> str:
        raw = f"{scope.project}:{scope.scope_type.value}:{scope.object_id}:{proposal_id}"
        return "action-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def directory(self, action_id: str) -> Path:
        if not action_id.startswith("action-") or not action_id[7:].isalnum():
            raise ValueError("invalid action_id")
        return self.root / action_id

    def save_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            directory = self.directory(str(plan["action_id"]))
            directory.mkdir(parents=True, exist_ok=True)
            existing = read_json(directory / "plan.json", {})
            if existing:
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
        with self._lock:
            path = self.directory(action_id) / "journal.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "timestamp": utc_now(), "event": event, "payload": payload,
                }, ensure_ascii=False, sort_keys=True) + "\n")

    def execution(self, action_id: str) -> dict[str, Any]:
        return read_json(self.directory(action_id) / "execution.json", {})

    def write_command(
        self, action_id: str, phase: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist sensitive/free-form input outside immutable action plans."""
        if phase not in {"authorize", "execute"}:
            raise ValueError("unsupported action command phase")
        with self._lock:
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
                      *, event: str) -> dict[str, Any]:
        with self._lock:
            atomic_json(self.directory(action_id) / "execution.json", payload)
            self.append_journal(action_id, event, {
                "status": payload.get("status"), "error": payload.get("error"),
            })
            return self.snapshot(action_id)

    def claim_execution(self, action_id: str) -> None:
        """Cross-process, create-once claim for one immutable action intent."""
        path = self.directory(action_id) / "execution.claim"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise RuntimeError("execution intent has already been claimed") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps({"claimed_at": utc_now(), "pid": os.getpid()}) + "\n")

    def snapshot(self, action_id: str) -> dict[str, Any]:
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

    def list_for_scope(self, scope: AgentScope) -> list[dict[str, Any]]:
        items = []
        for path in sorted(self.root.glob("action-*/plan.json")):
            payload = read_json(path, {})
            if payload.get("scope") == scope.model_dump(mode="json"):
                try:
                    items.append(self.snapshot(str(payload["action_id"])))
                except (FileNotFoundError, ValueError):
                    continue
        return items
