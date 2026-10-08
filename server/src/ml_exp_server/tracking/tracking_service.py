"""Server observations plus exact-Attempt worker records, without scheduler effects."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading

from ..ingest.runscan import train_metric_records
from .metric_contract import observations
from ..storage import utc_now
from .tracking_store import TrackingStore, digest
from ..wandb_exporter import export, PublisherProcess
from experiment_control.manifest import ExperimentStateStore


def store_for(runtime):
    return TrackingStore(runtime.config.project_registry_root_path())


def publication_config(project, run):
    # Server-private SSH aliases, credentials, transfer URLs, commands and raw
    # backend responses are intentionally absent from the external publication.
    return {"ml_expd": {"project": project, **{key: run[key] for key in
        ("run_id", "source_id", "image_id", "resources", "evaluation", "parameters", "resume_from") if key in run},
        "runtime_id": run.get("container", {}).get("runtime_id"),
        "backend": run.get("backend", {}).get("kind"),
        "inputs": [{key: item[key] for key in ("asset_id", "sha256", "mount_path") if key in item}
                   for item in run.get("inputs", [])]}}


def bind_run(runtime, project, run):
    store = store_for(runtime)
    store.bind(project, run["run_id"], run["tracking"], publication_config(project, run))
    return store


def preparation_record(runtime, value):
    route = value.get("tracking")
    if not route or not route["enabled"]:
        return
    store = store_for(runtime)
    # Preparation and actual Attempts have distinct W&B runs. Failure to build
    # does not fabricate a GPU Attempt or a successful training record.
    project, run = value["project"], value["run_id"]
    identity = run + "--preparation"
    store.bind(project, identity, route, {"ml_expd": {"project": project, "run_id": run,
               "preparation_id": value["preparation_id"], "record_type": "preparation"}})
    scope = store.scope(project, identity, "preparation")
    data = {key: value[key] for key in ("status", "phase", "updated_at", "runtime_id", "error_code") if key in value}
    data["state"] = data.pop("status")
    store.append(scope, [("preparation." + digest(data), {"kind": "lifecycle", "data": data})])


def normalized_metric(record, config):
    contract = config.get("ml_expd", {}).get("evaluation", {})
    return {"kind": "metrics", "observations": observations(record, contract.get("metrics_schema", {}),
                 protocol_id=contract.get("protocol_id"), source="exact-attempt-worker")}


def observe(runtime, store):
    for path in (runtime.config.project_registry_root_path() / "experiment-preparations").glob("*/*.json"):
        preparation_record(runtime, json.loads(path.read_text()))
    for row in runtime.index.list_runs():
        binding = store.binding(row.project, row.run_id)
        publication_run = row.run_id
        if binding is None or not binding["route"]["enabled"]:
            publication_run += "--backfill"
            binding = store.binding(row.project, publication_run)
        if binding is None or not binding["route"]["enabled"]:
            continue
        for attempt in row.attempts:
            scope = store.scope(row.project, publication_run, attempt.attempt_id)
            data = {"state": attempt.state, "backend": attempt.backend, "backend_job_id": attempt.backend_job_id}
            store.append(scope, [("scheduler." + digest(data), {"kind": "lifecycle", "data": data})])
            root = Path(row.run_dir) / "attempts" / attempt.attempt_id
            directory = runtime.config.project_registry_root_path() / "persistent-checkpoints" / row.project / row.run_id / attempt.attempt_id
            for path in directory.glob("checkpoint.*.json"):
                receipt = json.loads(path.read_text())
                value = {k: receipt[k] for k in ("checkpoint_id", "step", "files", "registered_at", "status")}
                store.append(scope, [("checkpoint." + digest(value), {"kind": "checkpoints", "data": value})])
            transfer = runtime.config.project_registry_root_path() / "artifact-transfers" / row.project / row.run_id / (attempt.attempt_id + ".json")
            if transfer.is_file():
                receipt = json.loads(transfer.read_text()).get("receipt")
                if receipt:
                    value = {k: receipt[k] for k in ("sha256", "bytes", "files", "received_at")}
                    store.append(scope, [("artifacts." + digest(value), {"kind": "artifacts", "data": value})])
            if not binding["route"].get("record_stream") or (root / "uploaded_outputs" / "metrics.jsonl").is_file():
                records, source, _ = train_metric_records(Path(row.run_dir), attempt_id=attempt.attempt_id, exact_attempt=True)
                for record in records:
                    payload = normalized_metric(record, binding["config"])
                    if not binding["route"].get("record_stream"):
                        payload.update(coverage="observed-legacy; completeness not guaranteed", source=source.name if source else None)
                    store.append(scope, [("history." + digest(payload), payload)])


def backfill(runtime, project, run, options):
    from ..application_errors import ApplicationError
    store = store_for(runtime)
    row = runtime.index.get_run(project, run)
    if row is None:
        raise ApplicationError("unknown Run", status_code=404, code="UNKNOWN_RUN")
    existing = store.binding(project, run)
    identity = run if existing and existing["route"]["enabled"] else run + "--backfill"
    binding = store.binding(project, identity)
    if binding:
        route = binding["route"]
        if (not options.enabled or options.entity is not None and options.entity != route["entity"]
                or options.project is not None and options.project != route["project"]):
            raise ValueError("publication target is already frozen")
    else:
        route = store.route(options, project)
        route["record_stream"] = False
        if not route["enabled"]:
            return {"wandb": route, "scopes": []}
        manifest = ExperimentStateStore(Path(row.run_dir)).load_manifest()
        run_definition = {**manifest, **manifest.get("resolved_config", {})}
        config = publication_config(project, run_definition)
        config["ml_expd"].update(record_type="backfill", history_coverage="saved evidence only; completeness not guaranteed")
        store.bind(project, identity, route, config)
    observe(runtime, store)
    return store.status(project, identity)


class TrackingPublisher:
    def __init__(self, runtime):
        self.runtime = runtime
        self.store = store_for(runtime)
        self.stop = threading.Event()
        self.publisher = PublisherProcess()
        self.thread = threading.Thread(target=self.loop, name="wandb-publisher", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.publisher.close()
        self.thread.join(timeout=5)

    def cycle(self):
        settings = self.store.settings()
        if not settings.get("api_key") or not settings.get("enabled", True):
            self.publisher.close()
            return
        observe(self.runtime, self.store)
        for scope in self.store.pending():
            if self.stop.is_set():
                break
            if scope["last_attempt_at"]:
                last = datetime.fromisoformat(scope["last_attempt_at"].replace("Z", "+00:00"))
                awaiting_ack = scope["error"] in {"REMOTE_ACK_PENDING", "REMOTE_DISPLAY_ACK_PENDING"}
                delay = 10 if awaiting_ack else min(300, 10 * 2 ** min(scope["failures"], 5))
                if (datetime.now(timezone.utc) - last).total_seconds() < delay:
                    continue
            export(self.store, scope, self.publisher)

    def loop(self):
        while not self.stop.wait(10):
            try:
                self.cycle()
            except Exception:
                # Publishing can never terminate the collector or experiment.
                atomic_error = {"code": "OBSERVATION_FAILED", "at": utc_now()}
                from ..storage import atomic_json
                atomic_json(self.store.root / "publisher-error.json", atomic_error)
