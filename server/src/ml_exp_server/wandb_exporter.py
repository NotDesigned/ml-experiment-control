"""Isolated official-cloud publisher; only remote evidence advances the cursor."""
from __future__ import annotations

import json
import subprocess
import os
import re
import select
import signal
from contextlib import redirect_stdout
import sys
import tempfile

from .tracking_store import digest, encoded
from . import wandb_display


def default_entity(key):
    import wandb
    try:
        return wandb.Api(api_key=key, overrides={"base_url": "https://api.wandb.ai"}, timeout=10).default_entity
    finally:
        # Configuration lookup owns no training/session handle in this process.
        wandb.teardown()


def history_record(event):
    payload = event["payload"]
    record = {"ml_expd/sequence": event["seq"], "ml_expd/event_id": event["event_id"],
              "ml_expd/kind": payload["kind"], "ml_expd/record": encoded(payload)}
    definitions = {}
    if payload["kind"] == "metrics":
        for item in payload["observations"]:
            context = {k: item.get(k) for k in ("name", "unit", "protocol_id", "checkpoint_id", "dataset_id", "variant_id")}
            key = "metrics/" + re.sub(r"[^A-Za-z0-9_.-]", "_", str(item["name"]))[:64] + "/" + digest(context)[:24]
            definitions[key] = context
            record[key + "/status"] = item["status"]
            if item["value"] is not None and item["status"] in {"VALID", "PARTIAL"}:
                record[key] = item["value"]
            for field in ("step", "epoch", "numerator", "denominator"):
                if field in item:
                    record[key + "/" + field] = item[field]
    elif payload["kind"] == "lifecycle":
        record.update({"ml_expd/" + key: value for key, value in payload["data"].items()})
    return record, definitions


def publish(job, sdk, sessions):
    route, scope = job["route"], job["scope"]
    identity = scope["wandb_id"]
    session = sessions.get(identity)
    if session is None:
        directory = tempfile.TemporaryDirectory(prefix="session-", dir=job["session_root"])
        try:
            handle = sdk.init(entity=route["entity"], project=route["project"], id=identity,
                name=job["config"]["ml_expd"]["run_id"] + "/" + scope["attempt"],
                group=scope["project"] + "/" + job["config"]["ml_expd"]["run_id"],
                job_type="preparation" if scope["attempt"] == "preparation" else "attempt",
                resume="allow", reinit="create_new", config=job["config"],
                settings=sdk.Settings(api_key=job["api_key"], base_url="https://api.wandb.ai",
                    root_dir=directory.name, disable_git=True, disable_code=True,
                    x_disable_stats=True, x_disable_meta=True, x_disable_machine_info=True,
                    console="off", quiet=True, init_timeout=30))
        except Exception:
            directory.cleanup()
            raise
        session = {"handle": handle, "directory": directory, "sent": {}, "definitions": {}}
        sessions[identity] = session
    handle = session["handle"]
    display = job.get("display", {"contract": "metric-display.v2", "through": scope["confirmed"], "series": {}})
    api = sdk.Api(api_key=job["api_key"], overrides={"base_url": "https://api.wandb.ai"}, timeout=15)
    remote = api.run(f'{route["entity"]}/{route["project"]}/{identity}')
    wandb_display.configure(handle, display)
    for key in remote.summary:
        if key.startswith("metrics/"):
            handle.define_metric(key, hidden=True, summary="none", overwrite=True)
    seen = {int(row["ml_expd/sequence"]): row["ml_expd/event_id"] for row in remote.scan_history(
        keys=["ml_expd/sequence", "ml_expd/event_id"], min_step=max(0, scope["confirmed"] - 1), page_size=128, use_cache=False)}
    if scope["confirmed"] and job.get("expected_event_id") is not None and seen.get(scope["confirmed"]) != job["expected_event_id"]:
        return {"error": "REMOTE_HISTORY_RECONCILE_REQUIRED"}
    maximum = max(max(seen, default=scope["confirmed"]), remote.lastHistoryStep + 1)
    confirmed = scope["confirmed"]
    for event in job["events"]:
        sequence, event_id = event["seq"], event["event_id"]
        if sequence in seen:
            if seen[sequence] != event_id:
                return {"error": "REMOTE_IDENTITY_CONFLICT"}
            confirmed = sequence
            session["sent"].pop(sequence, None)
            continue
        if sequence <= maximum:
            return {"error": "REMOTE_HISTORY_RECONCILE_REQUIRED"}
        if sequence in session["sent"]:
            if session["sent"][sequence] != event_id:
                return {"error": "REMOTE_IDENTITY_CONFLICT"}
            continue
        record, current = history_record(event)
        session["definitions"].update(current)
        for key in current:
            handle.define_metric(key, hidden=True, summary="none", overwrite=True)
            handle.define_metric(key + "/step", hidden=True, summary="none", overwrite=True)
        record.update(wandb_display.curve_record(event, display))
        # Explicit SDK steps default to commit=False. Without this, the final
        # event never reaches history and remote confirmation waits forever.
        handle.log(record, step=sequence - 1, commit=True)
        session["sent"][sequence] = event_id
    handle.summary["ml_expd/metric_definitions_json"] = encoded(session["definitions"])
    handle.summary["ml_expd/publication"] = "server-published; platform state is in ml_expd/state"
    through = job["events"][-1]["seq"] if job["events"] else scope["confirmed"]
    if confirmed < through:
        return {"confirmed": confirmed, "error": "REMOTE_ACK_PENDING"}
    wandb_display.summaries(handle, dict(remote.summary), display, terminal=job["terminal"],
                           coverage=job.get("coverage", "accepted metric observations"))
    if (remote.summary.get("ml_expd/display_version") != wandb_display.VERSION
            or remote.summary.get("ml_expd/display_through") != through):
        return {"confirmed": confirmed, "error": "REMOTE_DISPLAY_ACK_PENDING"}
    if job["terminal"]:
        handle.finish()
        session["directory"].cleanup()
        sessions.pop(identity)
    return {"confirmed": confirmed, "display_version": wandb_display.VERSION, "error": None}


class PublisherProcess:
    """One bounded SDK process; multiple active runs use official create_new."""
    def __init__(self):
        self.child = None
        self.directory = None
        self.key_hash = None

    def close(self):
        child, self.child = self.child, None
        if child is not None:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            child.stdin.close()
            child.stdout.close()
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None

    def call(self, job):
        key_hash = digest(job["api_key"])
        if self.child is None or self.child.poll() is not None or self.key_hash != key_hash:
            self.close()
            self.directory = tempfile.TemporaryDirectory(prefix="publisher-", dir=job["session_root"])
            environment = {key: value for key, value in os.environ.items() if not key.startswith("WANDB_")}
            environment.update(WANDB_CONFIG_DIR=self.directory.name, WANDB_CACHE_DIR=self.directory.name,
                               WANDB_BASE_URL="https://api.wandb.ai", WANDB_ERROR_REPORTING="false")
            self.child = subprocess.Popen([sys.executable, "-m", "ml_exp_server.wandb_exporter"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1, start_new_session=True, env=environment)
            self.key_hash = key_hash
        job = {**job, "session_root": self.directory.name}
        self.child.stdin.write(encoded(job) + "\n")
        self.child.stdin.flush()
        if not select.select([self.child.stdout], [], [], 90)[0]:
            self.close()
            raise TimeoutError("publisher did not respond")
        line = self.child.stdout.readline()
        return json.loads(line)


def export(store, scope, publisher):
    binding = store.binding(scope["project"], scope["run"])
    key = store.settings().get("api_key")
    if not key or not store.settings().get("enabled", True):
        store.outcome(scope["id"], error="CREDENTIALS_NOT_CONFIGURED" if not key else "DISABLED_BY_SERVER")
        return
    events = store.events(scope["id"], after=scope["confirmed"])
    lifecycle = store.latest_lifecycle(scope["id"])
    terminal = (lifecycle.get("state") in {"SUCCEEDED", "FAILED", "CANCELLED", "PREEMPTED", "TIMEOUT"}
                or lifecycle.get("phase") in {"READY", "DATA_FAILED", "RESULTS_UPLOADED", "RESULTS_UPLOAD_FAILED"})
    through = events[-1]["seq"] if events else scope["confirmed"]
    try:
        job = {**binding, "scope": scope, "api_key": key, "events": events,
               "display": store.display(scope["id"], through),
               "expected_event_id": store.event_id(scope["id"], scope["confirmed"]),
               "coverage": "worker JSONL; accepted observations" if binding["route"].get("record_stream") else "observed-legacy; training history incomplete",
               "terminal": terminal and through == store.total(scope["id"]), "session_root": str(store.root)}
        # No credential enters argv or log output. SDK failure leaves the durable
        # queue intact and never blocks the scheduler's separate executors.
        outcome = publisher.call(job)
        confirmed = outcome.get("confirmed")
        if confirmed is not None and (type(confirmed) is not int or not scope["confirmed"] <= confirmed <= through):
            raise ValueError("invalid remote confirmation")
        version = outcome.get("display_version")
        if version is not None and (version != wandb_display.VERSION or confirmed != through or outcome.get("error")):
            raise ValueError("invalid display confirmation")
        store.outcome(scope["id"], confirmed=confirmed, error=outcome.get("error"), display_version=version)
    except Exception:
        publisher.close()
        store.outcome(scope["id"], error="PUBLISHER_UNAVAILABLE")


def main():
    import wandb
    sessions = {}
    for line in sys.stdin:
        try:
            with redirect_stdout(sys.stderr):
                result = publish(json.loads(line), wandb, sessions)
        except Exception:
            result = {"error": "WANDB_REQUEST_FAILED"}
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
