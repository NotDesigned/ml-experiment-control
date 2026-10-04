"""Generic controller for imported source plus a fixed OCI image and argv.

Only scheduler-neutral metadata is interpreted on the daemon host. Project code
is executed by the selected backend inside the published runtime image.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
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
from .source_revisions import resolve_source_tree, _tree_digest
from .storage import read_json


def parse_metric(_campaign, line: str) -> dict | None:
    try:
        data = json.loads(line)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
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
                or any(spec.get(key) != runtime.get(key) for key in ("source_id", "entrypoint", "workdir", "packaging_revision"))
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
        )).get(self.run["backend"]["kind"])

    def source(self, source_root: str | None = None) -> Path:
        source_id = self.run["source_id"]
        config = ServerConfig(project_registry_root=self.campaign["source_store"])
        canonical = resolve_source_tree(config, self.campaign["project"], source_id)
        if source_root is not None and Path(source_root) != canonical:
            raise ValueError("controller source path does not match the frozen source")
        return canonical

    def command(self, attempt_id: str) -> list[str]:
        runtime = self.run["container"]
        output = self.run["storage"]["run_dir"] + f"/attempts/{attempt_id}/outputs"
        environment = {**self.run.get("env", {}),
                       "OUTPUT_DIR": output, "PROJECT_NAME": self.campaign["project"],
                       "RUN_ID": self.run["run_id"], "ATTEMPT_ID": attempt_id,
                       "SOURCE_ID": self.run["source_id"]}
        duration = self.run["resources"]["max_time"].split(":")
        seconds = sum(int(value) * multiplier for value, multiplier in zip(duration, (3600, 60, 1)))
        return ["env", *[f"{key}={value}" for key, value in sorted(environment.items())],
                "/bin/sh", "-c", 'mkdir -p "$1" && cd "$2" && shift 2 && exec "$@"',
                "ml-expd", output, runtime["workdir"],
                *(["timeout", "--signal=TERM", "--kill-after=30s", str(seconds) + "s", "python3", "/usr/local/lib/ml-expd/worker.py"] if self.campaign.get("artifact_store") else []),
                *runtime["entrypoint"], *self.run.get("arguments", [])]

    def dispatch_command(self, manifest):
        command = list(manifest["command"])
        if self.campaign.get("artifact_store"):
            from .artifact_store import ArtifactStore
            transfer = ArtifactStore(Path(self.campaign["artifact_store"]), Path(self.campaign["source_store"]))
            url, token, limit = transfer.issue(self.campaign["project"], self.run["run_id"],
                                               manifest["attempt_id"], self.root, self.run["outputs"])
            command = ["env", f"ML_EXPD_UPLOAD_URL={url}", f"ML_EXPD_UPLOAD_TOKEN={token}",
                       f"ML_EXPD_UPLOAD_LIMIT={limit}",
                       "ML_EXPD_OUTPUT_PATTERNS=" + json.dumps(self.run["outputs"]), *command]
        return command

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
            execution={"source_mount": "/workspace", "workdir": self.run["container"]["workdir"]},
            assets=[{"kind": "source", "identity": self.run["source_id"]},
                    {"kind": "runtime_image", "identity": self.run["container"]["image"]}],
            checkpoint=self.run.get("checkpoint", {}),
        )
        if self.store.manifest_path.exists():
            frozen = self.store.load_manifest()
            manifest["campaign_id"] = frozen.get("campaign_id")
            manifest["git_commit"] = frozen.get("git_commit")
        frozen = self.store.ensure_manifest(manifest)
        attempt = {**frozen, "created_at": utc_now(), "attempt_id": self.attempt_id,
                   "command": self.command(self.attempt_id), "resume_from": None}
        if not self.store.attempt_path(self.attempt_id).exists():
            self.store.create_attempt(attempt)
        self.store.initialize_attempt_records(self.attempt_id)
        return {"run_id": self.run["run_id"], "attempt_id": self.attempt_id,
                "manifest_path": str(self.store.manifest_path), "source_id": self.run["source_id"]}

    def stage(self, source_root: str | None = None):
        self.backend.validate(self.run)
        source = self.source(source_root)
        return self.backend.stage(self.campaign, self.run, self.run["source_id"], SourceBundle(
            root=source, excludes=(), container_path="/workspace", required_paths=()))

    def submit(self, campaign_id: str | None):
        self.prepare(campaign_id)
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
        if not record or not record.get("backend_job_id"):
            return {"run_id": self.run["run_id"], "attempt_id": self.attempt_id,
                    "backend": self.backend.kind, "backend_job_id": None, "state": "NOT_SUBMITTED"}
        result = self.backend.status(self.campaign, self.run)
        result.update(attempt_id=self.attempt_id, updated_at=utc_now())
        self.store.write_status_payload(self.attempt_id, result)
        return result

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
            report = controller.backend.preflight(controller.run, scope=args.scope)
            report.require_ready()
            result = {"ready": True}
        elif args.verb == "assets-verify":
            controller.source(args.source_root)
            result = {"missing": [], "verification": "immutable-source-and-runtime"}
        elif args.verb == "check-identity":
            record = controller.store.read_submission(args.attempt_id)
            identity = controller.backend.identity(campaign, controller.run, args.attempt_id)
            if record or not identity.available or identity.ambiguous:
                raise ValueError("Run/Attempt identity has already been consumed")
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
