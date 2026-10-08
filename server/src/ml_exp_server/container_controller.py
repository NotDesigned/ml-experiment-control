"""Generic controller for imported source plus a fixed OCI image and argv.

Only scheduler-neutral metadata is interpreted on the daemon host. Project code
is executed by the selected backend inside the published runtime image.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shlex
import sys

import yaml

from experiment_control.backends import BackendServices, build_registry
from experiment_control.manifest import ExperimentStateStore, atomic_write, utc_now
from experiment_control.outbox import execute_cancel_outbox
from experiment_control.project import SourceBundle
from experiment_control.run_manifest import build_run_manifest
from experiment_control.runner import SubprocessRunner

from .schemas import ServerConfig
from .projects.source_revisions import resolve_source_tree, _tree_digest
from .storage import read_json
from .worker_contract import managed_io
from .workers.worker_launcher import CONTRACT as LAUNCHER_CONTRACT
from .workers.worker_http import https_connection, relay_environment
from urllib.parse import urlsplit


def parse_metric(_campaign, line: str) -> dict | None:
    try:
        data = json.loads(line)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    if "name" in data:
        result = {key: data[key] for key in ("name", "value", "unit", "status", "error", "step", "epoch",
                                            "timestamp", "checkpoint_id", "dataset_id", "protocol_id", "variant_id",
                                            "numerator", "denominator") if key in data
                  and (data[key] is None or isinstance(data[key], (str, int, float, bool)))}
        result.setdefault("name", None)
        if any(key in data and key not in result for key in ("value", "unit", "status", "step", "epoch", "protocol_id", "checkpoint_id", "dataset_id", "variant_id", "numerator", "denominator")):
            result.update(status="FAILED", error="INVALID_METRIC_FIELD")
        for key, value in list(result.items()):
            if isinstance(value, float) and not math.isfinite(value):
                result[key] = None
                result.update(status="FAILED", error="NONFINITE_VALUE")
        return result
    result = {key: value for key, value in data.items()
              if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)}
    if "global_step" in result and "step" not in result:
        result["step"] = result.pop("global_step")
    return result or None


class Controller:
    def __init__(self, campaign: dict, run_id: str, attempt_id: str, *, runner=None):
        self.campaign = campaign
        self.run = next(item for item in campaign["runs"] if item["run_id"] == run_id)
        runtime = self.run["container"]
        runtime_record = read_json(Path(campaign["source_store"]) / "runtime-bundles" / campaign["project"]
                                   / (runtime["runtime_id"] + ".json"), {})
        spec = runtime_record.get("spec", {})
        if (runtime_record.get("status") != "READY" or runtime_record.get("project") != campaign["project"]
                or runtime_record.get("image") != runtime["image"]
                or any(spec.get(key) != runtime.get(key) for key in ("source_id", "entrypoint", "workdir", "packaging_revision", "dockerfile", "requirements"))
                or runtime_record.get("worker_contract") != runtime.get("worker_contract")
                or runtime.get("launcher_contract") != (LAUNCHER_CONTRACT if LAUNCHER_CONTRACT in runtime_record.get("capabilities", []) else None)
                or self.run["source_id"] != runtime["source_id"]
                or self.run["image_id"] != runtime["image"].split("@", 1)[1]):
            raise ValueError("controller runtime does not match its immutable definition")
        self.attempt_id = attempt_id
        self.root = Path(campaign["local_root"]) / campaign["campaign"] / run_id
        self.store = ExperimentStateStore(self.root)
        self.attempt = self.root / "attempts" / attempt_id
        self.runner = runner or SubprocessRunner()
        self.backend = build_registry(BackendServices(
            run_command=self.runner.run,
            local_run_dir=lambda _campaign, _run: self.attempt,
            backend_record=lambda _campaign, _run: self.store.load_backend(attempt_id) or {
                "attempt_id": attempt_id, "backend_job_id": None},
            summarize_run=self.summarize,
            parse_metric=parse_metric,
            parse_checkpoint=lambda _campaign, _line: None,
            atomic_write=atomic_write, utc_now=utc_now,
            collection_includes=lambda _campaign: tuple(
                f"/attempts/{attempt_id}/outputs/{pattern}" for pattern in self.run["outputs"]
            ),
            dispatch_command=self.dispatch_command,
            oci_pull_environment=self.oci_pull_environment,
            run_manifest_path=lambda _campaign, _run: self.store.manifest_path,
        )).get(self.run["backend"]["kind"])

    def source(self, source_root: str | None = None) -> Path:
        source_id = self.run["source_id"]
        config = ServerConfig(project_registry_root=self.campaign["source_store"])
        canonical = resolve_source_tree(config, self.campaign["project"], source_id)
        if source_root is not None and Path(source_root) != canonical:
            raise ValueError("controller source path does not match the frozen source")
        return canonical

    def environment(self, attempt_id: str) -> dict:
        output = self.run["storage"]["run_dir"] + f"/attempts/{attempt_id}/outputs"
        environment = {**self.run.get("env", {}),
                       "OUTPUT_DIR": output, "PROJECT_NAME": self.campaign["project"],
                       "RUN_ID": self.run["run_id"], "ATTEMPT_ID": attempt_id,
                       "SOURCE_ID": self.run["source_id"]}
        environment.update(relay_environment(self.run["backend"].get("api_relay")))
        if managed_io(self.run["container"]):
            environment["INPUTS_DIR"] = "/inputs"
        if self.run.get("data_preparation"):
            environment["DATA_DIR"] = (self.run["storage"]["project_data_root"] + "/data-preparations/"
                                       + self.run["data_preparation"]["preparation_id"] + "/tree")
        if self.run.get("checkpoint_persistence"):
            environment["STATE_DIR"] = output.rsplit("/", 1)[0] + "/state"
        if self.run.get("resume_from"):
            environment["RESUME_DIR"] = "/inputs/resume"
        return environment

    def duration_seconds(self) -> int:
        duration = self.run["resources"]["max_time"].split(":")
        return sum(int(value) * multiplier for value, multiplier in zip(duration, (3600, 60, 1)))

    def manifest_launcher(self) -> bool:
        return bool(self.campaign.get("artifact_store") and self.run["container"].get("launcher_contract") == LAUNCHER_CONTRACT)

    def command(self, attempt_id: str) -> list[str]:
        if self.manifest_launcher():
            return ["ml-exp-worker"]
        runtime = self.run["container"]
        environment = self.environment(attempt_id)
        output = environment["OUTPUT_DIR"]
        return ["env", *[f"{key}={value}" for key, value in sorted(environment.items())],
                "/bin/sh", "-c", 'mkdir -p "$1" && cd "$2" && shift 2 && exec "$@"',
                "ml-expd", output, runtime["workdir"],
                *(["timeout", "--signal=TERM", "--kill-after=30s", str(self.duration_seconds()) + "s", "python3", "/usr/local/lib/ml-expd/worker.py"] if self.campaign.get("artifact_store") else []),
                *runtime["entrypoint"], *self.run.get("arguments", [])]

    def dispatch_command(self, manifest):
        command = list(manifest["command"])
        if self.campaign.get("artifact_store"):
            from .results.artifact_store import ArtifactStore
            transfer = ArtifactStore(Path(self.campaign["artifact_store"]), Path(self.campaign["source_store"]))
            url, token, limit = transfer.issue(self.campaign["project"], self.run["run_id"],
                                               manifest["attempt_id"], self.root, self.run["outputs"],
                                               **({"record_stream": True} if self.run.get("tracking", {}).get("enabled") and self.run.get("tracking", {}).get("record_stream") else {}),
                                               **({"checkpoint_upload": True} if self.run.get("checkpoint_upload") else {}),
                                               **({"checkpoint_state": self.state_context(manifest["attempt_id"])} if self.run.get("checkpoint_persistence") else {}))
            dispatch = {"ML_EXPD_UPLOAD_URL": url, "ML_EXPD_UPLOAD_LIMIT": str(limit),
                        "ML_EXPD_OUTPUT_PATTERNS": json.dumps(self.run["outputs"])}
            relay = relay_environment(self.run["backend"].get("api_relay"))
            dispatch.update(relay)
            if self.run.get("tracking", {}).get("enabled") and self.run.get("tracking", {}).get("record_stream"):
                prefix = transfer.config["public_transfer_base"].rstrip("/").rsplit("/", 1)[0]
                dispatch["ML_EXPD_RECORD_URL"] = prefix + "/record-transfers/" + "/".join([self.campaign["project"], self.run["run_id"], manifest["attempt_id"]])
            if self.run.get("inputs"):
                prefix = transfer.config["public_transfer_base"].rstrip("/").rsplit("/", 1)[0]
                base = prefix + "/asset-transfers/" + "/".join([self.campaign["project"], self.run["run_id"], manifest["attempt_id"]])
                inputs = [{**item, "url": base + "/" + item["asset_id"]} for item in self.run["inputs"]]
                dispatch["ML_EXPD_INPUT_ASSETS"] = json.dumps(inputs)
            if self.run.get("checkpoint_upload"):
                prefix = transfer.config["public_transfer_base"].rstrip("/").rsplit("/", 1)[0]
                url = prefix + "/snapshot-transfers/" + "/".join([self.campaign["project"], self.run["run_id"], manifest["attempt_id"]])
                dispatch.update(ML_EXPD_SNAPSHOT_URL=url,
                                ML_EXPD_SNAPSHOT_INTERVAL=str(self.run["checkpoint_upload"]["interval_seconds"]))
            if self.run.get("data_preparation"):
                dispatch["ML_EXPD_DATA_PREPARATION"] = json.dumps(self.run["data_preparation"])
                if relay:
                    dispatch["ML_EXPD_DATA_PREPARATION_CACHED"] = "1"
            if self.run.get("checkpoint_persistence"):
                prefix = transfer.config["public_transfer_base"].rstrip("/").rsplit("/", 1)[0]
                state_url = prefix + "/checkpoint-transfers/" + "/".join([self.campaign["project"], self.run["run_id"], manifest["attempt_id"]])
                dispatch.update(ML_EXPD_CHECKPOINT_STATE=json.dumps(self.state_context(manifest["attempt_id"])),
                                ML_EXPD_CHECKPOINT_STATE_URL=state_url,
                                ML_EXPD_CHECKPOINT_STATE_INTERVAL=str(self.run["checkpoint_persistence"]["interval_seconds"]))
            if self.run.get("resume_from"):
                dispatch["ML_EXPD_CHECKPOINT_RESTORE"] = json.dumps(self.run["resume_from"])
            if self.manifest_launcher():
                launch_url = transfer.seal_launch(self.campaign["project"], self.run["run_id"], manifest["attempt_id"], {
                    "contract": LAUNCHER_CONTRACT,
                    "identity": {"project": self.campaign["project"], "run_id": self.run["run_id"], "attempt_id": manifest["attempt_id"]},
                    "environment": {**self.environment(manifest["attempt_id"]), **dispatch},
                    "workdir": self.run["container"]["workdir"],
                    "argv": [*self.run["container"]["entrypoint"], *self.run.get("arguments", [])],
                    "timeout_seconds": self.duration_seconds(),
                })
                return ["env", *[f"{key}={value}" for key, value in sorted(relay.items())],
                        "ML_EXPD_BOOTSTRAP_TOKEN=" + token, "ml-exp-worker", "--manifest-url", launch_url]
            command = ["env", "ML_EXPD_UPLOAD_TOKEN=" + token, *[f"{key}={value}" for key, value in sorted(dispatch.items())], *command]
        return command

    def state_context(self, attempt_id):
        return {"project": self.campaign["project"], "run_id": self.run["run_id"], "attempt_id": attempt_id,
                "source_id": self.run["source_id"], "image_id": self.run["image_id"],
                "storage_scope": self.run["checkpoint_persistence"]["storage_scope"],
                "state_root": self.run["storage"]["run_dir"] + f"/attempts/{attempt_id}/state"}

    def oci_pull_environment(self):
        path = self.campaign.get("registry_pull")
        if not path:
            return {}
        credential = json.loads(Path(path).read_text())
        registry = self.run["backend"].get("oci_image", "").split("/", 1)[0]
        if registry != credential["registry"]:
            return {}
        return {"APPTAINER_DOCKER_USERNAME": credential["username"],
                "APPTAINER_DOCKER_PASSWORD": credential["password"]}

    def prepare(self, campaign_id: str | None = None) -> dict:
        source = self.source()
        metadata = json.loads((source.parent / "source.json").read_text())
        resolved = {key: self.run[key] for key in ("container", "arguments", "env", "outputs")}
        resolved.update({key: self.run[key] for key in ("inputs", "checkpoint_upload", "data_preparation", "checkpoint_persistence", "resume_from", "tracking", "parameters") if key in self.run})
        manifest = build_run_manifest(
            project=self.campaign["project"], run_id=self.run["run_id"], created_at=utc_now(),
            config_path="container_execution", resolved_config=resolved,
            source_id=self.run["source_id"], runtime_tree_id=self.run["source_id"],
            git_commit=metadata.get("observation", {}).get("commit"),
            campaign_id=campaign_id or self.campaign.get("campaign_id"),
            campaign=self.campaign["campaign"], image_id=self.run["image_id"],
            run_dir=self.run["storage"]["run_dir"], max_infra_retries=self.run.get("max_infra_retries", 1),
            backend=self.run["backend"], resources=self.run["resources"], storage=self.run["storage"],
            command=self.command("{attempt_id}"),
            execution={"source_mount": "/workspace", "workdir": self.run["container"]["workdir"],
                       **({"managed_io": True} if managed_io(self.run["container"]) and self.run["backend"]["kind"] == "slurm" else {})},
            assets=[{"kind": "source", "identity": self.run["source_id"]},
                    {"kind": "runtime_image", "identity": self.run["container"]["image"]},
                    *[{"kind": "data_asset", "identity": item["asset_id"], "mount_path": item["mount_path"]}
                      for item in self.run.get("inputs", [])],
                    *([{"kind": "data_preparation", "identity": self.run["data_preparation"]["preparation_id"]}]
                      if self.run.get("data_preparation") else []),
                    *([{"kind": "persistent_checkpoint", "identity": self.run["resume_from"]["checkpoint_id"], "mount_path": "/inputs/resume"}]
                      if self.run.get("resume_from") else [])],
            checkpoint=self.run.get("checkpoint", {}),
            evaluation=self.run.get("evaluation", {}),
        )
        if self.store.manifest_path.exists():
            frozen = self.store.load_manifest()
            manifest["campaign_id"] = frozen.get("campaign_id")
            manifest["git_commit"] = frozen.get("git_commit")
        frozen = self.store.ensure_manifest(manifest)
        attempt = {**frozen, "created_at": utc_now(), "attempt_id": self.attempt_id,
                   "command": self.command(self.attempt_id), "resume_from": self.run.get("resume_from", {}).get("checkpoint_id")}
        if not self.store.attempt_path(self.attempt_id).exists():
            self.store.create_attempt(attempt)
        self.store.initialize_attempt_records(self.attempt_id)
        return {"run_id": self.run["run_id"], "attempt_id": self.attempt_id,
                "manifest_path": str(self.store.manifest_path), "source_id": self.run["source_id"]}

    def stage(self, source_root: str | None = None):
        self.backend.validate(self.run)
        self.check_api_transport()
        source = self.source(source_root)
        staged = self.backend.stage(self.campaign, self.run, self.run["source_id"], SourceBundle(
            root=source, excludes=(), container_path="/workspace", required_paths=()))
        self.prepare_gateway_data(require_cached=False)
        return staged

    def check_api_transport(self):
        relay = relay_environment(self.run["backend"].get("api_relay"))
        if not relay:
            return
        target = json.loads(Path(self.campaign["artifact_store"]).read_text())["public_transfer_base"]
        code = (Path(__file__).parent / "workers" / "worker_http.py").read_text() + "\n" + (
            "import json,sys\nvalue=json.load(sys.stdin)\n"
            "connection=https_connection(urlsplit(value['target']),timeout=10,environment=value['environment'])\n"
            "try: connection.connect()\nfinally: connection.close()\n")
        self.runner.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                         self.run["backend"]["ssh_alias"], shlex.join(["python3", "-c", code])],
                        input_text=json.dumps({"target": target, "environment": relay}), timeout_seconds=30)

    def prepare_gateway_data(self, *, require_cached):
        if not self.run["backend"].get("api_relay") or not self.run.get("data_preparation"):
            return
        backend = self.run["backend"]
        cache = self.run["storage"]["project_data_root"] + "/data-preparations"
        code = ("import json,sys;from pathlib import Path;sys.path.insert(0,'/usr/local/lib/ml-expd');"
                "from data_preparation import prepare;prepare(json.load(sys.stdin),Path(sys.argv[1]),"
                "require_cached=sys.argv[2]=='1')")
        seconds = self.run["data_preparation"]["timeout_seconds"] + 60
        cache_dir = backend.get("apptainer_cache_dir", self.run["storage"]["project_data_root"] + "/apptainer/cache")
        temp_dir = backend.get("apptainer_tmp_dir", self.run["storage"]["project_data_root"] + "/apptainer/tmp")
        command = ["env", "APPTAINER_CACHEDIR=" + cache_dir, "APPTAINER_TMPDIR=" + temp_dir,
                   "timeout", "--signal=TERM", "--kill-after=30s", str(seconds) + "s",
                   "apptainer", "exec", *(["--unsquash"] if backend.get("apptainer_unsquash") else []),
                   "--bind", backend["mount_root"] + ":" + backend["mount_root"],
                   "--pwd", self.run["container"]["workdir"], backend["sif_path"],
                   "python3", "-c", code, cache, "1" if require_cached else "0"]
        self.runner.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", backend["ssh_alias"],
                         shlex.join(["mkdir", "-p", cache_dir, temp_dir]) + " && " + shlex.join(command)],
                        input_text=json.dumps(self.run["data_preparation"]),
                        timeout_seconds=seconds + 60)

    def submit(self, campaign_id: str | None):
        self.prepare(campaign_id)
        self.check_api_transport()
        self.prepare_gateway_data(require_cached=True)
        self.backend.preflight(self.run, scope="submit").require_ready()
        attempt = self.store.load_attempt(self.attempt_id)
        intent = self.store.begin_submission(
            project=self.campaign["project"], run_id=self.run["run_id"],
            attempt_id=self.attempt_id, backend=self.backend.kind,
            request=self.backend.submission_request(self.campaign, self.run, self.attempt_id))
        job_id = self.backend.recover_submission(self.run, intent, self.attempt_id)
        if job_id is None:
            job_id = self.backend.submit(self.campaign, self.run, attempt, dry_run=False, intent=intent)
        self.store.reconcile_submission(project=self.campaign["project"], run_id=self.run["run_id"],
                                        attempt_id=self.attempt_id, backend_job_id=job_id)
        return {"run_id": self.run["run_id"], "attempt_id": self.attempt_id, "backend_job_id": job_id}

    def status(self):
        record = self.store.load_backend(self.attempt_id)
        intent = self.store.read_submission(self.attempt_id)
        if intent and (not record or not record.get("backend_job_id")):
            recovered = self.backend.recover_submission(self.run, intent, self.attempt_id)
            if recovered is None:
                return {"run_id": self.run["run_id"], "attempt_id": self.attempt_id,
                        "backend": self.backend.kind, "backend_job_id": None,
                        "state": "SUBMITTING", "submission_recovery": "NOT_FOUND"}
            self.store.reconcile_submission(project=self.campaign["project"], run_id=self.run["run_id"],
                                            attempt_id=self.attempt_id, backend_job_id=recovered)
            record = self.store.load_backend(self.attempt_id)
        if not record or not record.get("backend_job_id"):
            return {"run_id": self.run["run_id"], "attempt_id": self.attempt_id,
                    "backend": self.backend.kind, "backend_job_id": None, "state": "NOT_SUBMITTED"}
        result = self.backend.status(self.campaign, self.run)
        previous = self.store.load_status_payload(self.attempt_id) or {}
        if (self.backend.kind == "sensecore" and previous.get("state") == "PREEMPTED"
                and result.get("raw_state") == "SUSPENDED" and result["state"] == "CANCELLED"):
            # Old adapters classified every provider stop as preemption. Keep
            # the immutable terminal lifecycle while exposing the correction.
            result["state"] = "PREEMPTED"
            result["detail"] = {"classification": "legacy_terminal_state",
                                "observed_normalized_state": "CANCELLED"}
        result.update(attempt_id=self.attempt_id, updated_at=utc_now())
        self.store.write_status_payload(self.attempt_id, result)
        return result

    def check_identity(self):
        record = self.store.read_submission(self.attempt_id)
        identity = self.backend.identity(self.campaign, self.run, self.attempt_id)
        if record or identity.ambiguous or identity.scheduler_job_ids:
            raise ValueError("Run/Attempt identity has already been consumed")
        if identity.available:
            return
        # A new Attempt legitimately shares its frozen Run manifest. Require
        # an exact remote digest and prior terminal scheduler evidence.
        attempts = self.root / "attempts"
        prior_terminal = False
        for path in attempts.iterdir() if attempts.is_dir() else ():
            if path.name == self.attempt_id or not re.fullmatch(r"attempt-[0-9]{3,}", path.name):
                continue
            previous = self.store.load_backend(path.name) or {}
            state = self.store.load_status_payload(path.name) or {}
            if previous.get("backend_job_id") and state.get("state") in {"FAILED", "PREEMPTED", "CANCELLED"}:
                prior_terminal = True
        if not (prior_terminal and identity.remote_manifest_exists and identity.remote_manifest_matches is True):
            raise ValueError("remote Run identity is unavailable or conflicts with the frozen retry")

    def summarize(self, _campaign, root: Path) -> dict:
        records = []
        for path in sorted(root.rglob("metrics.jsonl")):
            if path.is_symlink() or path.stat().st_size > 32 * 1024 * 1024:
                continue
            with path.open() as stream:
                for line in stream:
                    metric = parse_metric(None, line)
                    if metric:
                        records.append(metric)
        return {"latest_metric": records[-1] if records else None,
                "model_observed": bool(records)}

    def collect(self):
        result = self.backend.collect(self.campaign, self.run)
        uploaded = self.attempt / "uploaded_outputs"
        if uploaded.exists():
            result.update(self.summarize(self.campaign, uploaded))
            result["artifacts"] = {"files": sum(1 for p in uploaded.rglob("*") if p.is_file()),
                                   "storage": "s3", "download_available": True}
        alias = self.run.get("artifact_ssh")
        if alias and self.backend.kind == "sensecore":
            mirror = self.attempt / "collected_run" / "attempts" / self.attempt_id / "outputs"
            mirror.mkdir(parents=True, exist_ok=True)
            remote = self.run["storage"]["run_dir"] + f"/attempts/{self.attempt_id}/outputs/"
            self.runner.run(["rsync", "-a", "--safe-links", "-e", "ssh -o BatchMode=yes",
                             f"{alias}:{remote}", str(mirror) + "/"], timeout_seconds=120)
        result.update(project=self.campaign["project"], run_id=self.run["run_id"],
                      attempt_id=self.attempt_id, collected_at=utc_now())
        result["scheduler_state"] = (self.store.load_status_payload(self.attempt_id) or {}).get("state")
        atomic_write(self.attempt / "collection.json", result)
        logs = self.backend.logs(self.campaign, self.run, tail=1000)
        for key in ("stdout", "stderr", "lines"):
            if isinstance(logs.get(key), list):
                target = "stdout.log" if key == "lines" else key + ".log"
                (self.attempt / target).write_text("\n".join(logs[key]) + "\n")
        current = self.store.load_backend()
        if current and current.get("attempt_id") == self.attempt_id:
            atomic_write(self.root / "collection.json", result)
        return result

    def decide(self):
        state = self.store.load_status_payload(self.attempt_id) or {}
        terminal = state.get("state") in {"FAILED", "CANCELLED", "LOST"}
        maximum = self.store.load_manifest().get("resume_policy", {}).get("max_infra_retries", 0)
        result = {"run_id": self.run["run_id"], "attempt_id": self.attempt_id,
                  "action": "REVIEW_RETRY" if terminal else "WAIT",
                  "failure_class": state.get("failure_class") or "unknown",
                  "retries_allowed": maximum, "retries_used": int(self.attempt_id.split("-")[1]) - 1}
        atomic_write(self.attempt / "decision.json", result)
        return result


def cli(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("verb", choices=["submit", "stage", "preflight", "check-identity", "assets-verify", "status", "observe", "collect", "decide", "cancel"])
    parser.add_argument("--run", required=True)
    parser.add_argument("--attempt-id", default="attempt-001")
    parser.add_argument("--source-root")
    parser.add_argument("--source-id")
    parser.add_argument("--local-root")
    parser.add_argument("--campaign-id")
    parser.add_argument("--scope", default="submit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        campaign = yaml.safe_load(args.campaign.read_text())
        if args.local_root:
            campaign["local_root"] = args.local_root
        controller = Controller(campaign, args.run, args.attempt_id)
        if args.source_id and args.source_id != controller.run["source_id"]:
            raise ValueError("controller source identity mismatch")
        if args.verb == "submit":
            result = controller.prepare(args.campaign_id) if args.dry_run else controller.submit(args.campaign_id)
        elif args.verb == "stage":
            result = {"staged": controller.stage(args.source_root)}
        elif args.verb == "preflight":
            if args.scope != "observe":
                controller.check_api_transport()
            report = controller.backend.preflight(controller.run, scope=args.scope)
            report.require_ready()
            result = {"ready": True}
        elif args.verb == "assets-verify":
            controller.source(args.source_root)
            result = {"missing": [], "verification": "immutable-source-and-runtime"}
            if controller.run.get("inputs"):
                from .data.data_assets import AssetStore
                assets = AssetStore(Path(campaign["artifact_store"]), Path(campaign["source_store"]))
                for item in controller.run["inputs"]:
                    value = assets.read(campaign["project"], item["asset_id"])
                    if any(value[key] != item[key] for key in ("sha256", "archive_bytes", "files")):
                        raise ValueError("input asset differs from the frozen Run")
                result.update(verification="immutable-input-assets; backend-delivery-verified-by-worker", inputs=len(controller.run["inputs"]))
        elif args.verb == "check-identity":
            controller.check_identity()
            result = {"available": True}
        elif args.verb in {"status", "observe"}:
            result = controller.status()
            if args.verb == "observe" and result.get("backend_job_id"):
                controller.collect()
        elif args.verb == "collect":
            result = controller.collect()
        elif args.verb == "decide":
            result = controller.decide()
        else:
            record = controller.store.load_backend(controller.attempt_id)
            if not record or not record.get("backend_job_id"):
                raise ValueError("Attempt has no exact scheduler job")
            result = execute_cancel_outbox(
                run_dir=controller.root, project=campaign["project"], run_id=args.run,
                attempt_id=args.attempt_id, backend=controller.backend.kind,
                backend_job_id=record["backend_job_id"], status_call=controller.status,
                cancel_call=lambda: controller.backend.cancel(campaign, controller.run))
            controller.store.write_status_payload(controller.attempt_id, result)
        print(json.dumps([result], allow_nan=False))
    except Exception as exc:
        print("container controller failed: " + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    cli()
