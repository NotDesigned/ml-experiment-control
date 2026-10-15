"""Audited whole-Run resource deletion, independent of download acknowledgements."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import errno
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import stat
import time

import yaml

from ..application_errors import ApplicationError
from ..backends.run_deletion import BackendRunDeletion, digest as backend_digest
from ..deletion_state import RunFence
from ..records.store import RecordStore
from ..results.admin_deletion import AdminArtifacts
from ..results.artifacts import open_directory
from ..projects.source_imports import source_lock
from ..runs.intent_protocol import OperationIntent
from ..schemas import OperationScope, OperationScopeType
from ..storage import DurableJsonState, utc_now
from ..workers.delete_run import remove_tree

_ACTION = re.compile(r"^action-[0-9a-f]{16}$")
_ACTIVE = {"AUTHORIZED", "EXECUTING", "RECONCILE_REQUIRED"}


def digest(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def mapping(path):
    try:
        parent = open_directory(path.parent)
    except FileNotFoundError:
        return {}
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError("deletion evidence contains a link or invalid parent") from None
        raise
    try:
        try:
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        except FileNotFoundError:
            return {}
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError("deletion evidence is a link") from None
            raise
        with os.fdopen(descriptor, "r") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("deletion evidence is not a regular file")
            content = stream.read()
    finally:
        os.close(parent)
    value = yaml.safe_load(content) if path.suffix in {".yaml", ".yml"} else json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("deletion evidence is not a mapping")
    return value


def unlink_file(path):
    """Remove an exact regular file without following any ancestor or leaf link."""
    try:
        parent = open_directory(path.parent)
    except FileNotFoundError:
        return
    try:
        try:
            before = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("canonical deletion file is not regular")
        os.unlink(path.name, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)


def replace_yaml(path, value):
    parent = open_directory(path.parent)
    temporary = ".deletion-" + secrets.token_hex(16)
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(yaml.safe_dump(value, sort_keys=False))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
        os.fsync(parent)
    finally:
        try:
            os.unlink(temporary, dir_fd=parent)
        except FileNotFoundError:
            pass
        os.close(parent)


class RunDeletionService:
    def __init__(self, runtime, *, backend=None, sleep=time.sleep):
        self.runtime = runtime
        self.root = runtime.config.project_registry_root_path()
        self.fence = RunFence(self.root)
        self.backend = backend or BackendRunDeletion(runtime)
        self.sleep = sleep

    def _identity(self, project, run):
        self.fence.path(project, run)
        configured = self.runtime.project(project)
        managed = self.root / "managed" / project
        if (configured.base_dir is None or configured.base_dir.resolve() != managed.resolve()
                or configured.controller is None or not configured.controller.capabilities.get("container_execution")):
            raise ApplicationError("only daemon-managed Runs support deletion", code="RUN_DELETION_UNSUPPORTED")
        path = managed / "experiments" / "campaigns" / ("run-" + run + ".yaml")
        campaign = mapping(path)
        runs = campaign.get("runs", [])
        if campaign.get("project") != project or len(runs) != 1 or runs[0].get("run_id") != run:
            raise ApplicationError("unknown frozen managed Run", status_code=404, code="UNKNOWN_RUN")
        local = self.runtime.config.project_run_root_path(project) / ("run-" + run) / run
        return configured, path, campaign, local

    def _attempts(self, local):
        project, run = local.parent.parent.name, local.name
        known, transfers, jobs, submitted_ids = set(), {}, {}, set()

        def remember(identity, job=None):
            if not isinstance(identity, str) or not re.fullmatch(r"attempt-[0-9]{3,}", identity):
                raise ApplicationError("Attempt identity evidence is incomplete", code="RUN_DELETION_ATTEMPT_EVIDENCE_INCOMPLETE")
            known.add(identity)
            if job is not None:
                if not isinstance(job, str) or not job:
                    raise ValueError("invalid recorded scheduler job identity")
                jobs.setdefault(identity, set()).add(job)

        canonical = {}
        for path in sorted((local / "attempts").glob("attempt-*")):
            if path.is_symlink() or not path.is_dir():
                raise ValueError("invalid canonical Attempt directory")
            remember(path.name)
            canonical[path.name] = path
        row = self.runtime.index.get_run(project, run)
        if row is not None:
            if row.run_dir and (Path(row.run_dir) != local or not local.is_dir()):
                raise ApplicationError("canonical Run evidence is missing or differs from its index", code="RUN_DELETION_ATTEMPT_EVIDENCE_INCOMPLETE")
            for value in row.attempts:
                item = value if isinstance(value, dict) else value.model_dump(mode="json")
                remember(item["attempt_id"], item.get("backend_job_id"))
                if item.get("has_submission"):
                    submitted_ids.add(item["attempt_id"])
        for path in sorted((self.root / "artifact-transfers" / project / run).glob("attempt-*.json")):
            value = mapping(path)
            if ((value.get("project"), value.get("run_id"), value.get("attempt_id")) != (project, run, path.stem)
                    or value.get("run_dir") != str(local)):
                raise ValueError("Attempt transfer provenance differs from canonical Run")
            remember(path.stem)
            transfers[path.stem] = value
            diagnostics = value.get("training_diagnostics") or {}
            if isinstance(diagnostics, dict) and diagnostics.get("scheduler_name"):
                remember(path.stem, diagnostics["scheduler_name"])
        for path in (local / "backend.json", local / "status.json"):
            value = mapping(path)
            if value.get("attempt_id") or value.get("backend_job_id"):
                remember(value.get("attempt_id"), value.get("backend_job_id"))
        for action in self.runtime.action_store.list_all():
            scope = action.get("scope", {})
            if (scope.get("project") != project
                    or action.get("run_id") != run and scope.get("object_id", "").split("::", 1)[0] != run):
                continue
            identity = action.get("attempt_id")
            if identity is None and scope.get("scope_type") == "attempt":
                identity = scope.get("object_id", "").split("::", 1)[-1]
            if identity is None:
                continue
            remember(identity)
            result = action.get("execution", {}).get("result") or {}
            if isinstance(result, dict):
                for value in (result, result.get("submission"), result.get("observation")):
                    if isinstance(value, dict) and value.get("backend_job_id"):
                        if (value.get("run_id", run) != run or value.get("attempt_id", identity) != identity):
                            raise ValueError("Action scheduler evidence differs from its exact Attempt")
                        remember(identity, value["backend_job_id"])
        if known - canonical.keys():
            raise ApplicationError("known Attempt has no canonical directory", code="RUN_DELETION_ATTEMPT_EVIDENCE_INCOMPLETE")
        attempts = []
        for identity, path in sorted(canonical.items()):
            manifest = mapping(path / "attempt.yaml") or mapping(path / "control_attempt.yaml")
            if (manifest.get("project"), manifest.get("run_id"), manifest.get("attempt_id")) != (project, run, identity):
                raise ApplicationError("canonical Attempt manifest is missing or differs", code="RUN_DELETION_ATTEMPT_EVIDENCE_INCOMPLETE")
            backend = mapping(path / "backend.json")
            status = mapping(path / "status.json")
            outbox = mapping(path / "submission.json")
            job = backend.get("backend_job_id")
            for value in (backend, status, outbox):
                if (value.get("attempt_id", identity) != identity or value.get("project", project) != project
                        or value.get("run_id", run) != run):
                    raise ValueError("canonical Attempt state provenance differs")
                if value.get("backend_job_id"):
                    remember(identity, value["backend_job_id"])
            if jobs.get(identity) and (not job or jobs[identity] != {job}):
                raise ApplicationError("recorded scheduler jobs differ from canonical Attempt", code="RUN_DELETION_ATTEMPT_EVIDENCE_INCOMPLETE")
            transfer = transfers.get(identity, {})
            diagnostics = transfer.get("training_diagnostics") or {}
            if not isinstance(diagnostics, dict):
                raise ValueError("invalid durable training diagnostics")
            uids = [value[key] for value in (backend, outbox) for key in ("job_uid", "backend_job_uid") if value.get(key)]
            if job and diagnostics.get("scheduler_name") == job and diagnostics.get("job_uid"):
                uids.append(diagnostics["job_uid"])
            if any(not isinstance(uid, str) for uid in uids) or len(set(uids)) > 1:
                raise ApplicationError("original scheduler UID evidence conflicts", code="RUN_DELETION_ATTEMPT_EVIDENCE_INCOMPLETE")
            uid = next(iter(uids), None)
            deleted = self._worker_deleted(diagnostics, job, uid)
            produced = bool(transfer.get("receipt") or transfer.get("snapshots") or transfer.get("results_ready")
                            or job and diagnostics.get("scheduler_name") == job)
            submitted = bool(job or outbox or produced or identity in submitted_ids)
            state = status.get("state", "UNKNOWN" if submitted else "NOT_SUBMITTED")
            if not submitted and state in {"CREATED", "PREPARED"}:
                state = "NOT_SUBMITTED"
            attempts.append({"attempt_id": identity, "backend_job_id": job, "backend_job_uid": uid,
                             "state": state, "has_submission": submitted, "worker_deleted": deleted})
        return attempts

    @staticmethod
    def _worker_deleted(diagnostics, job, uid):
        if (not uid or not job or diagnostics.get("scheduler_name") != job
                or diagnostics.get("job_uid") != uid):
            return False
        pod = re.compile(r"(?<![A-Za-z0-9_-])pt-" + re.escape(uid) + r"-worker-[0-9]+(?![A-Za-z0-9_-])")
        return any(isinstance(event, dict) and event.get("reason") == "SuccessfulDelete"
                   and isinstance(event.get("message"), str) and pod.search(event["message"])
                   for event in diagnostics.get("events", []))

    @staticmethod
    def _cached_attempt_gates(manifest):
        from ..backends.states import TERMINAL
        unresolved, identities = [], []
        for attempt in manifest["attempts"]:
            job = attempt.get("backend_job_id")
            unsubmitted = not job and not attempt.get("has_submission") and attempt.get("state") == "NOT_SUBMITTED"
            if not unsubmitted and attempt.get("state") not in TERMINAL:
                unresolved.append(attempt["attempt_id"])
            if (not unsubmitted and not job
                    or job and manifest["backend"]["backend"] == "sensecore" and not attempt.get("backend_job_uid")):
                identities.append(attempt["attempt_id"])
        return [("original_attempts_terminal", not unresolved, json.dumps(unresolved)),
                ("original_job_identities_known", not identities, json.dumps(identities))]

    def _dependencies(self, project, run):
        from ..ingest.runscan import discover_run_dirs
        dependencies = {}

        def reference(other, value, kind):
            if (not isinstance(value, dict) or value.get("project", project) != project
                    or value.get("run_id") != run or not other or other == run
                    or self.fence.is_deleted(project, other)):
                return
            checkpoint = value.get("checkpoint_id")
            if checkpoint is not None and not isinstance(checkpoint, str):
                raise ValueError("invalid checkpoint dependency identity")
            dependencies.setdefault((other, checkpoint), {"run_id": other, "kind": kind, "checkpoint_id": checkpoint})

        def scientific(value, kind, *, owner=None):
            if value.get("project", project) != project:
                return
            other = value.get("run_id", owner)
            reference(other, value.get("resume_from"), kind)
            resolved = value.get("resolved_config")
            if isinstance(resolved, dict):
                reference(other, resolved.get("resume_from"), kind)

        campaigns = self.root / "managed" / project / "experiments" / "campaigns"
        for path in sorted({*campaigns.glob("*.yaml"), *campaigns.glob("*.yml")}):
            campaign = mapping(path)
            if campaign.get("project") == project:
                for other in campaign.get("runs", []):
                    scientific(other, "frozen_resume")
        for path in (self.root / "experiment-preparations" / project).glob("*.json"):
            value = mapping(path)
            request = value.get("request", {}).get("run", {})
            other = value.get("run_id")
            if value.get("project", project) == project:
                reference(other, request.get("resume_from"), "preparation_resume")
        # Authored Campaigns can be updated. Their original canonical science
        # and already-frozen Actions still bind any subsequent retry or restore.
        for directory in discover_run_dirs(self.runtime.config.project_run_root_path(project)):
            for name in ("manifest.yaml", "control_manifest.yaml"):
                scientific(mapping(directory / name), "canonical_resume", owner=directory.name)
            for path in (directory / "attempts").glob("attempt-*/*.yaml"):
                if path.name in {"attempt.yaml", "control_attempt.yaml"}:
                    scientific(mapping(path), "attempt_resume", owner=directory.name)
        for action in self.runtime.action_store.list_all():
            if action.get("scope", {}).get("project") != project:
                continue
            owner = action.get("run_id")
            if not owner or self.fence.is_deleted(project, owner):
                continue
            summary = action.get("preflight_summary")
            if isinstance(summary, dict):
                scientific(summary, "action_resume", owner=owner)
            raw = action.get("execution_campaign_file")
            if raw:
                path = Path(raw)
                if path.parent != self.runtime.action_store.directory(action["action_id"]):
                    raise ValueError("frozen Action Campaign is outside its Action")
                campaign = mapping(path)
                if campaign.get("project") == project:
                    for other in campaign.get("runs", []):
                        scientific(other, "action_resume")
        return sorted(dependencies.values(), key=lambda item: (item["run_id"], str(item["checkpoint_id"])))

    def _busy(self, project, run, *, own_action=None):
        busy = []
        for action in self.runtime.action_store.list_all():
            scope = action.get("scope", {})
            if (action["action_id"] != own_action and scope.get("project") == project
                    and (action.get("run_id") == run or scope.get("object_id", "").split("::")[0] == run)
                    and action.get("execution", {}).get("status") in _ACTIVE):
                busy.append({"kind": "action", "id": action["action_id"]})
        for path in (self.root / "experiment-preparations" / project).glob("*.json"):
            value = mapping(path)
            if value.get("run_id") == run and value.get("status") in {"EXECUTING", "RECONCILE_REQUIRED"}:
                busy.append({"kind": "preparation", "id": value.get("preparation_id")})
        for path in (self.root / "result-collections" / project / run).glob("attempt-*.json"):
            value = mapping(path)
            if value.get("status") in {"QUEUED", "COLLECTING", "SUBMITTING", "WAITING", "RECONCILE_REQUIRED"}:
                busy.append({"kind": "collection", "id": path.stem})
        for path in (self.root / "multipart-uploads").glob("*/upload.json"):
            value = mapping(path)
            binding = value.get("binding", {})
            if binding.get("project") == project and binding.get("run_id") == run and value.get("status") == "UPLOADING":
                busy.append({"kind": "upload", "id": value.get("upload_id")})
        return busy

    def manifest(self, project, run):
        _, path, campaign, local = self._identity(project, run)
        frozen = campaign["runs"][0]
        attempts = self._attempts(local)
        objects = AdminArtifacts(self.runtime.storage_service.objects).plan_objects(project, run)
        # Deletion paths are generated here, never accepted from HTTP input.
        paths = [local, path, self.root / "persistent-checkpoints" / project / run,
                 self.root / "result-collections" / project / run]
        paths.extend(path for path in (self.root / "experiment-preparations" / project).glob("*.json")
                     if mapping(path).get("run_id") == run)
        snapshots = sorted({asset for path in (self.root / "artifact-transfers" / project / run).glob("attempt-*.json")
                            for asset in mapping(path).get("snapshots", [])})
        return {"contract": "run-deletion-manifest.v1", "project": project, "run_id": run,
                "definition_sha256": digest(campaign), "attempts": attempts,
                "backend": self.backend.plan(frozen, attempts), "objects": objects,
                "local_paths": sorted(str(path) for path in paths),
                "retained_snapshot_assets": snapshots,
                "dependencies": self._dependencies(project, run),
                "retained": {"action_audit": True, "object_receipts": True, "tombstone": True,
                             "shared_assets": True, "wandb_cloud": True, "provider_history": "UNSUPPORTED"}}

    def prepare(self, project, run, reason, idempotency_key):
        if not reason.strip():
            raise ApplicationError("deletion reason is required", status_code=422, code="DELETION_REASON_REQUIRED")
        scope = OperationScope(project=project, scope_type=OperationScopeType.RUN, object_id=run)
        configured = self.runtime.project(project)
        intent = OperationIntent(kind="DELETE_RUN_RESOURCES", title="Delete managed Run resources", target=run,
                                 rationale=reason, risk="Irreversible deletion of all Run resources", draft="{}",
                                 idempotency_key=idempotency_key)
        return self.runtime.action_service.prepare(scope, configured, intent)

    def prepare_action(self, action_id, scope, project, intent):
        with self.fence.guard(scope.project, scope.object_id):
            manifest = self.manifest(scope.project, scope.object_id)
            busy = self._busy(scope.project, scope.object_id)
        enabled = self.runtime.config.action_runtime.allow_resource_deletions
        checks = [("deletion_policy", enabled, "administrator resource-deletion policy"),
                  ("no_active_or_uncertain_operations", not busy, json.dumps(busy)),
                  ("checkpoint_dependencies", not manifest["dependencies"], json.dumps(manifest["dependencies"]))]
        checks.extend(self._cached_attempt_gates(manifest))
        return {"action_id": action_id, "intent_id": intent["intent_id"], "intent_kind": "DELETE_RUN_RESOURCES",
                "scope": scope.model_dump(mode="json"), "operation": "DELETE_RUN_RESOURCES", "run_id": scope.object_id,
                "target": scope.object_id, "reason": intent["rationale"],
                "deletion_request": {"reason": intent["rationale"], "idempotency_key": intent["intent_id"]},
                "created_at": utc_now(), "target_manifest": manifest, "target_digest": digest(manifest),
                "gates": [{"name": name, "status": "PASS" if ok else "FAIL", "detail": detail} for name, ok, detail in checks],
                "ready": all(ok for _, ok, _ in checks),
                "risk": "Irreversible whole-Run data deletion; no download ACK is fabricated"}

    @contextmanager
    def state(self, action_id):
        if not _ACTION.fullmatch(action_id):
            raise ValueError("invalid deletion identity")
        root = self.root / "run-deletions"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (root / (action_id + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            store = DurableJsonState(root / (action_id + ".json"), root / (action_id + ".jsonl"))
            snapshot = store.snapshot({})
            store.repair_journal(snapshot)
            yield store, snapshot

    def update(self, action_id, **changes):
        with self.state(action_id) as (store, snapshot):
            if "backend" in changes and snapshot.value.get("backend", {}).get("receipt"):
                changes["backend"] = {**changes["backend"], "receipt": snapshot.value["backend"]["receipt"]}
            value = {**snapshot.value, **changes}
            store.commit(value, expected_revision=snapshot.revision,
                         event={"event": "run_deletion_progress", "status": value.get("status"), "timestamp": utc_now()})
            return value

    def receive(self, action_id, token, receipt):
        with self.state(action_id) as (store, snapshot):
            value = snapshot.value
            state = value.get("backend", {})
            if not token or not hmac.compare_digest(token, state.get("capability", "")) or value.get("status") not in {"DELETING", "DELETED"}:
                raise ApplicationError("invalid deletion receipt capability", status_code=403, code="DELETION_CAPABILITY_REQUIRED")
            expected = self.backend.payload(value["target_manifest"]["backend"], state)
            accepted = self.backend.receipt(expected, receipt)
            if state.get("receipt"):
                if state["receipt"] != accepted:
                    raise ApplicationError("deletion receipt conflicts", code="DELETION_RECEIPT_CONFLICT")
                return {"accepted": True}
            value["backend"] = {**state, "receipt": accepted}
            store.commit(value, expected_revision=snapshot.revision, event={"event": "run_deletion_receipt", "timestamp": utc_now()})
            return {"accepted": True}

    @contextmanager
    def execution_lease(self, action_id):
        if not _ACTION.fullmatch(action_id):
            raise ValueError("invalid deletion identity")
        root = self.root / "run-deletions"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (root / (action_id + ".execution.lock")).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ApplicationError("Run deletion is already executing", code="RUN_DELETION_BUSY") from None
            yield

    def _local_targets(self, manifest):
        project, run = manifest["project"], manifest["run_id"]
        self.fence.path(project, run)
        campaign = self.root / "managed" / project / "experiments" / "campaigns" / ("run-" + run + ".yaml")
        directories = [self.runtime.config.project_run_root_path(project) / ("run-" + run) / run,
                       self.root / "persistent-checkpoints" / project / run,
                       self.root / "result-collections" / project / run]
        files = [campaign]
        base = self.root / "experiment-preparations" / project
        supplied = manifest["local_paths"]
        if len(supplied) != len(set(supplied)):
            raise ValueError("duplicate deletion path")
        for raw in supplied:
            path = Path(raw)
            if path in directories or path == campaign:
                continue
            if path.parent != base or not re.fullmatch(r"preparation-[0-9a-f]{32}\.json", path.name):
                raise ValueError("deletion path is outside managed Run")
            value = mapping(path)
            if value and value.get("run_id") != run:
                raise ValueError("preparation belongs to another Run")
            files.append(path)
        if sorted(str(path) for path in [*directories, *files]) != supplied:
            raise ValueError("canonical deletion paths differ")
        return directories, files

    def _detach_catalogue(self, project, run):
        managed = self.root / "managed" / project
        path = managed / "experiments" / "research_project.yaml"
        campaign = managed / "experiments" / "campaigns" / ("run-" + run + ".yaml")
        # Same lock as create_run: removing a reference precedes unlinking its
        # definition, so a crash can leave an orphan, never a broken catalogue.
        with source_lock(self.root / "runtime-bundles", project):
            value = mapping(path)
            references = value["campaigns"]
            retained = []
            for reference in references:
                target = Path(reference["file"])
                if not target.is_absolute():
                    target = managed / target
                if reference["name"] == "run-" + run or target == campaign:
                    if reference["name"] != "run-" + run or target != campaign:
                        raise ValueError("campaign catalogue binding differs")
                else:
                    retained.append(reference)
            if retained != references:
                value["campaigns"] = retained
                replace_yaml(path, value)
            self.runtime.register_project(path)
        self.runtime.index.delete_run(project, run)

    def execute(self, plan):
        with self.execution_lease(plan["action_id"]):
            return self._execute(plan)

    def _execute(self, plan):
        action_id = plan["action_id"]
        if not self.runtime.config.action_runtime.allow_resource_deletions:
            raise ApplicationError("administrator resource deletion is disabled", code="RUN_DELETION_DISABLED")
        manifest = plan["target_manifest"]
        if digest(manifest) != plan["target_digest"]:
            raise ValueError("deletion manifest changed")
        project, run = manifest["project"], manifest["run_id"]
        if (plan.get("scope") != {"project": project, "scope_type": "run", "object_id": run}
                or plan.get("run_id") != run or plan.get("operation") != "DELETE_RUN_RESOURCES"):
            raise ValueError("deletion Action scope differs")
        directories, files = self._local_targets(manifest)
        with self.state(action_id) as (_, snapshot):
            value = snapshot.value
        if not value:
            # Positive provider evidence precedes the short mutation fence.
            _, _, campaign, _ = self._identity(project, run)
            terminal = self.backend.terminal(campaign["runs"][0], manifest["attempts"])
            with self.runtime.action_store.locked(), self.fence.exclusive():
                self.fence.require_live(project, run)
                if self._busy(project, run, own_action=action_id) or self._dependencies(project, run):
                    raise ApplicationError("Run has active work or checkpoint consumers", code="RUN_DELETION_IN_USE")
                if self.manifest(project, run) != manifest:
                    raise ApplicationError("Run resources changed after review", code="RUN_DELETION_PLAN_CONFLICT")
                state = {"deletion_id": action_id, "plan_sha256": backend_digest(manifest["backend"])}
                if manifest["backend"]["backend"] == "sensecore":
                    origin = self.runtime.storage_service.objects.config["public_transfer_base"].rstrip("/").rsplit("/", 1)[0]
                    state.update(receipt_url=origin + "/run-deletion-transfers/" + action_id + "/receipt", capability=secrets.token_urlsafe(32))
                value = self.update(action_id, status="DELETING", target_manifest=manifest,
                                    target_digest=plan["target_digest"], backend=state, terminal=terminal)
                self.fence.mark(project, run, {"deletion_id": action_id, "target_digest": plan["target_digest"], "reason": plan["reason"]})
        elif value.get("target_digest") != plan["target_digest"] or value.get("target_manifest") != manifest:
            raise ValueError("durable deletion target differs")
        if (value.get("status") not in {"DELETING", "DELETED"}
                or value.get("backend", {}).get("deletion_id") != action_id
                or value["backend"].get("plan_sha256") != backend_digest(manifest["backend"])):
            raise ValueError("durable deletion identity or state differs")
        if value["status"] == "DELETED":
            result = value.get("backend_result", {})
            if result.get("status") != "DELETED" or value["backend"].get("stage") != "DELETED":
                raise ValueError("completed deletion lacks backend evidence")
            payload = self.backend.payload(manifest["backend"], value["backend"])
            receipt = self.backend.receipt(payload, result.get("receipt"))
            if receipt != value["backend"].get("receipt"):
                raise ValueError("completed deletion receipt differs")
        if not self.fence.is_deleted(project, run):
            # Crash after the durable intent but before its fence: revalidate,
            # never infer that external effects ran without the fence.
            _, _, campaign, _ = self._identity(project, run)
            self.backend.terminal(campaign["runs"][0], manifest["attempts"])
            with self.runtime.action_store.locked(), self.fence.exclusive():
                if self._busy(project, run, own_action=action_id) or self._dependencies(project, run) or self.manifest(project, run) != manifest:
                    raise ApplicationError("Run resources changed before deletion fence", code="RUN_DELETION_PLAN_CONFLICT")
                self.fence.mark(project, run, {"deletion_id": action_id, "target_digest": plan["target_digest"], "reason": plan["reason"]})
        tombstone = mapping(self.fence.path(project, run))
        if any(tombstone.get(key) != expected for key, expected in {
                "project": project, "run_id": run, "deletion_id": action_id,
                "target_digest": plan["target_digest"], "status": "DELETED"}.items()):
            raise ValueError("Run deletion fence identity differs")
        if value.get("status") != "DELETED":
            while True:
                with self.state(action_id) as (_, snapshot):
                    value = snapshot.value
                backend = self.backend.execute(manifest["backend"], dict(value["backend"]), lambda state: self.update(action_id, backend=state))
                if backend["status"] == "DELETED":
                    break
                self.sleep(2)
            def persist(state):
                self.update(action_id, artifacts=state)
            artifacts = AdminArtifacts(self.runtime.storage_service.objects)
            artifacts.delete_objects(manifest["objects"], value.get("artifacts", {}), persist)
            # Actual metrics/events are removed; shared definitions and Actions
            # are retained. The fence prevents any delayed writer resurrecting them.
            RecordStore(self.root).delete_run(project, run)
            self._detach_catalogue(project, run)
            for target in directories:
                remove_tree(str(target))
            for target in files:
                unlink_file(target)
            value = self.update(action_id, status="DELETED", completed_at=utc_now(), backend_result=backend)
        return {"status": "DELETED", "project": project, "run_id": run, "target_digest": plan["target_digest"],
                "backend": value.get("backend_result"), "retained": manifest["retained"]}
