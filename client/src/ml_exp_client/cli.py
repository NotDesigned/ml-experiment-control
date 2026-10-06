"""Command-line source/container experiment workflow over HTTP."""
from __future__ import annotations

import argparse
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import sys
import time
import tempfile
from urllib.parse import urlencode

from . import __version__
from .api import Client, ClientError, data_archive, download, save, segment, source_archive, upload_asset_parts
from .workflow import experiment, validate_dockerfile, report
from .data_delivery import deliver_data


def parser():
    value = argparse.ArgumentParser(prog="ml-exp", description=__doc__)
    value.add_argument("--version", action="version", version="%(prog)s " + __version__)
    value.add_argument("--url", default=os.environ.get("ML_EXPD_API_URL"))
    commands = value.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="write a small source project; no API connection")
    initialize.add_argument("directory", type=Path, help="new source directory")
    initialize.add_argument("--base-image", help="approved sha256-pinned base; see ml-exp check")
    workflow = commands.add_parser("experiment", help="validate one JSON config, build, prepare, optionally execute and download")
    workflow.add_argument("config", type=Path)
    workflow.add_argument("--state", type=Path, required=True)
    workflow.add_argument("--resume", action="store_true", help="continue the same identities; never replay uncertain scheduler requests")
    workflow.add_argument("--execute", action="store_true", help="authorize and execute the prepared submission within its GPU-hour budget")
    workflow.add_argument("--seconds", type=int, default=1800)
    workflow.add_argument("--download-to", type=Path)
    check = commands.add_parser("check", help="GET health, policy, executors and optionally schema")
    check.add_argument("--schema", type=Path)
    pack = commands.add_parser("pack", help="import source and package image; allocates no GPU")
    pack.add_argument("--project", required=True)
    pack.add_argument("--source", type=Path, required=True)
    pack.add_argument("--dockerfile", default="Dockerfile", help="path inside source; external FROM images must be digest-pinned (default: Dockerfile)")
    pack.add_argument("--entrypoint", nargs="+", default=["python", "train.py"])
    pack.add_argument("--state", type=Path, required=True)
    create = commands.add_parser("create", help="freeze a Run; does not submit it")
    create.add_argument("--runtime-state", type=Path, required=True)
    create.add_argument("--run", required=True)
    create.add_argument("--executor", required=True)
    create.add_argument("--arguments", type=json.loads, default=[], help="JSON argv array")
    create.add_argument("--gpus", type=int, default=1)
    create.add_argument("--cpus", type=int, default=8)
    create.add_argument("--memory-gb", type=int, default=32)
    create.add_argument("--max-time", default="00:05:00")
    create.add_argument("--inputs", type=json.loads, default=[], help='JSON array: [{"asset_id":"asset.…","mount_path":"/inputs/data"}]')
    create.add_argument("--checkpoint-interval", type=int, help="publish atomic checkpoint.ready.json every N seconds")
    create.add_argument("--data-preparation", type=json.loads, help='JSON: {"script":"download_data.py","arguments":[],"timeout_seconds":1200}')
    create.add_argument("--checkpoint-state-interval", type=int, help="register persistent STATE_DIR checkpoints every N seconds; no state upload")
    create.add_argument("--resume-from", type=json.loads, help='JSON: {"run_id":"trial","attempt_id":"attempt-001","checkpoint_id":"checkpoint.…"}')
    upload = commands.add_parser("asset-upload", help="upload a data directory independently from source")
    upload.add_argument("--project", required=True)
    upload.add_argument("--directory", type=Path, required=True)
    upload.add_argument("--state", type=Path, required=True)
    upload.add_argument("--resume", action="store_true", help="resume the exact saved data archive; completed parts are retained")
    upload.add_argument("--executor", help="also prepare data on this SenseCore executor's NAS using a CPU-only ACP job")
    upload.add_argument("--wait-seconds", type=int, default=1800)
    assets = commands.add_parser("assets", help="list immutable project data assets")
    assets.add_argument("--project", required=True)
    snapshots = commands.add_parser("snapshots", help="list published checkpoints, including during training")
    snapshots.add_argument("--project", required=True)
    snapshots.add_argument("--run", required=True)
    snapshots.add_argument("--attempt", required=True)
    checkpoints = commands.add_parser("checkpoints", help="list backend-resident recovery checkpoints; no state download")
    checkpoints.add_argument("--project", required=True)
    checkpoints.add_argument("--run", required=True)
    checkpoints.add_argument("--attempt", required=True)
    prepare = commands.add_parser("prepare", help="preflight and save Submission gates; no GPU submission")
    prepare.add_argument("--project", required=True)
    prepare.add_argument("--run", required=True)
    prepare.add_argument("--max-gpu-hours", type=float, required=True)
    prepare.add_argument("--state", type=Path, required=True)
    execute = commands.add_parser("execute", help="authorize and submit the exact saved Submission")
    execute.add_argument("--state", type=Path, required=True)
    execute.add_argument("--confirm", required=True, help="exact confirmation printed by prepare")
    submission = commands.add_parser("submission", help="inspect/poll an existing Submission")
    submission.add_argument("--id", required=True)
    submission.add_argument("--reconcile", action="store_true", help="observe uncertain submission; never replay submit")
    runtime = commands.add_parser("runtime", help="inspect/poll an existing Runtime")
    runtime.add_argument("--state", type=Path, required=True)
    runtime.add_argument("--reconcile", action="store_true")
    runtime.add_argument("--logs", action="store_true", help="read recent server-side build logs")
    watch = commands.add_parser("watch", help="observe a Run until scheduler completion")
    watch.add_argument("--project", required=True)
    watch.add_argument("--run", required=True)
    watch.add_argument("--seconds", type=float, default=1800)
    fetch = commands.add_parser("download", help="download exact-Attempt outputs and verify against archive")
    fetch.add_argument("--project", required=True)
    fetch.add_argument("--run", required=True)
    fetch.add_argument("--attempt", required=True)
    fetch.add_argument("--out", type=Path, required=True, help="new destination directory")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            args.directory.mkdir(parents=True, exist_ok=False)
            template = files("ml_exp_client").joinpath("templates", "train.py.txt")
            (args.directory / "train.py").write_bytes(template.read_bytes())
            base = args.base_image or "REPLACE_WITH_APPROVED_BASE@sha256:" + "0" * 64
            (args.directory / "Dockerfile").write_text(f"FROM {base}\nWORKDIR /workspace\n", encoding="utf-8")
            print(json.dumps({"source_dir": str(args.directory), "entrypoint": ["python", "train.py"]}, indent=2))
            return 0
        token = os.environ.get("ML_EXPD_API_TOKEN", "")
        if not token and os.environ.get("ML_EXPD_API_TOKEN_FILE"):
            token = Path(os.environ["ML_EXPD_API_TOKEN_FILE"]).read_text().strip()
        client = Client(args.url or "", token)
        health = client.negotiate()
        def wait(endpoint, **kwargs):
            if "execution-progress.v1" in health.get("capabilities", []):
                kwargs["observer"] = lambda value: report(client.call(endpoint + "/progress"))
            return client.wait(endpoint, **kwargs)
        if args.command == "check":
            result = {"health": health, "policy": client.call("/api/actions/policy"),
                      "executors": client.call("/api/executors")}
            if "environments.v1" in health.get("capabilities", []):
                result["environments"] = client.call("/api/environments")
            if "data-assets.v1" in health.get("capabilities", []):
                result["storage_limits"] = client.call("/api/storage-limits")
            if args.schema:
                save(args.schema, client.call(health["openapi_path"]))
        elif args.command == "experiment":
            result = experiment(client, health, args.config, args.state, resume=args.resume,
                                execute=args.execute, seconds=args.seconds, out=args.download_to)
        elif args.command == "pack":
            if args.state.exists():
                raise ClientError("state file exists; inspect it with runtime instead of replaying pack")
            if "dockerfile-build.v1" not in health.get("capabilities", []):
                raise ClientError("server does not advertise the Dockerfile build capability")
            validate_dockerfile(args.source, args.dockerfile)
            source = source_archive(args.source)
            query = urlencode({"project": args.project, "sha256": hashlib.sha256(source).hexdigest()})
            imported = client.call("/api/source-imports/archive?" + query, raw=source)
            save(args.state, imported)
            endpoint = "/api/projects/" + segment(args.project) + "/runtimes"
            definition = {"source_id": imported["source_id"], "entrypoint": args.entrypoint}
            definition["dockerfile"] = args.dockerfile
            result = client.call(endpoint + "/prepare", data=definition)
            save(args.state, result)  # Recovery identity is saved before starting packaging.
            endpoint += "/" + segment(result["runtime_id"])
            if result["status"] == "PREPARED":
                client.call(endpoint + "/execute", data={"confirmation": result["confirmation"]})
            result = wait(endpoint, pending=("PREPARED", "EXECUTING"))
            save(args.state, result)
            if result["status"] != "READY":
                raise ClientError("packaging requires inspection; use runtime --state FILE --reconcile")
        elif args.command == "runtime":
            saved = json.loads(args.state.read_text())
            endpoint = f"/api/projects/{segment(saved['project'])}/runtimes/{segment(saved['runtime_id'])}"
            result = client.call(endpoint)
            if args.reconcile and result["status"] in {"RECONCILE_REQUIRED", "EXECUTING"}:
                client.call(endpoint + "/reconcile", data={"confirmation": result["confirmation"]})
            result = wait(endpoint)
            save(args.state, result)
            if args.logs:
                result = {"runtime": result, "build_logs": client.call(endpoint + "/logs")}
        elif args.command == "create":
            saved = json.loads(args.runtime_state.read_text())
            definition = {
                "runtime_id": saved["runtime_id"], "run_id": args.run, "executor": args.executor,
                "arguments": args.arguments, "resources": {"gpus": args.gpus, "cpus": args.cpus,
                "memory_gb": args.memory_gb, "max_time": args.max_time}, "outputs": ["**/*"], "inputs": args.inputs}
            if args.checkpoint_interval is not None:
                definition["checkpoint_upload"] = {"interval_seconds": args.checkpoint_interval}
            if args.data_preparation is not None:
                if "data-preparation.v1" not in health.get("capabilities", []):
                    raise ClientError("server lacks data-preparation.v1; upgrade before using a download script")
                definition["data_preparation"] = args.data_preparation
            if args.checkpoint_state_interval is not None or args.resume_from is not None:
                if "persistent-checkpoints.v1" not in health.get("capabilities", []):
                    raise ClientError("server lacks persistent-checkpoints.v1")
                if args.checkpoint_state_interval is not None:
                    definition["checkpoint_persistence"] = {"interval_seconds": args.checkpoint_state_interval}
                if args.resume_from is not None:
                    definition["resume_from"] = args.resume_from
            result = client.call(f"/api/projects/{segment(saved['project'])}/runs", data=definition)
        elif args.command == "asset-upload":
            if args.state.exists() and not args.resume:
                raise ClientError("state file exists; inspect the saved asset ID instead of replaying upload")
            if args.executor:
                if "ccr-data-delivery.v1" not in health.get("capabilities", []):
                    raise ClientError("server lacks ccr-data-delivery.v1")
                catalogue = client.call("/api/executors")["executors"]
                if not any(item["id"] == args.executor and item.get("kind") == "sensecore" for item in catalogue):
                    raise ClientError("data preparation requires a published SenseCore executor")
            limits = client.call("/api/storage-limits")
            with tempfile.TemporaryFile() as stream:
                digest, length = data_archive(args.directory, stream)
                if limits["asset_archive_bytes"] is not None and length > limits["asset_archive_bytes"]:
                    raise ClientError("data archive exceeds server upload limit")
                if args.resume:
                    saved = json.loads(args.state.read_text())
                    if saved.get("project") != args.project or saved.get("asset_id") != "asset." + digest:
                        raise ClientError("resume directory differs from the saved upload archive")
                    if saved.get("status") == "READY" and not args.executor:
                        print(json.dumps(saved, indent=2))
                        return 0
                save(args.state, {"project": args.project, "asset_id": "asset." + digest, "status": "UPLOADING"})
                if "multipart-upload.v1" in health.get("capabilities", []):
                    result = upload_asset_parts(client, args.project, stream, digest, length, args.state)
                else:
                    query = urlencode({"project": args.project, "sha256": digest})
                    result = client.call("/api/assets/archive?" + query, raw=stream, length=length)
            save(args.state, result)
            if args.executor:
                result = {**result, "data_delivery": deliver_data(client, args.project, result["asset_id"], args.executor,
                          args.state.with_name(args.state.name + ".delivery.json"), args.wait_seconds)}
        elif args.command == "assets":
            result = client.call(f"/api/projects/{segment(args.project)}/assets")
        elif args.command == "snapshots":
            result = client.call(f"/api/runs/{segment(args.project)}/{segment(args.run)}/attempts/{segment(args.attempt)}/snapshots")
        elif args.command == "checkpoints":
            if "persistent-checkpoints.v1" not in health.get("capabilities", []):
                raise ClientError("server lacks persistent-checkpoints.v1")
            result = client.call(f"/api/runs/{segment(args.project)}/{segment(args.run)}/attempts/{segment(args.attempt)}/checkpoints")
        elif args.command == "prepare":
            if args.state.exists():
                raise ClientError("state file exists; inspect the saved Submission")
            result = client.call(f"/api/experiments/{segment(args.project)}/{segment(args.run)}/submissions/prepare",
                                 data={"max_gpu_hours": args.max_gpu_hours, "reason": "API client trial"})
            save(args.state, result)
        elif args.command in {"execute", "submission"}:
            saved = json.loads(args.state.read_text()) if args.command == "execute" else None
            sid = saved["submission_id"] if saved else args.id
            endpoint = "/api/submissions/" + segment(sid)
            result = client.call(endpoint)
            if args.command == "execute":
                if args.confirm != result["confirmation"] or args.confirm != saved["confirmation"]:
                    raise ClientError("confirmation does not match the saved Submission")
                if result["status"] == "PREPARED" and result["ready"]:
                    result = client.call(endpoint + "/authorize", data={"note": "explicit API client execution"})
                if result["status"] == "AUTHORIZED":
                    client.call(endpoint + "/execute", data={"confirmation": args.confirm})
            elif args.reconcile and result["status"] == "RECONCILE_REQUIRED":
                client.call(endpoint + "/reconcile", data={})
            result = wait(endpoint)
            if saved:
                save(args.state, result)
        elif args.command == "watch":
            endpoint = f"/api/runs/{segment(args.project)}/{segment(args.run)}"
            deadline = time.monotonic() + args.seconds
            while True:
                result = client.call(endpoint)
                state = result.get("scheduler_state")
                print("scheduler_state=" + str(state), file=sys.stderr)
                if state in {"SUCCEEDED", "FAILED", "CANCELLED", "PREEMPTED", "TIMEOUT"}:
                    result = {"run": result, "attempts": client.call(endpoint + "/attempts")}
                    break
                if time.monotonic() >= deadline:
                    raise ClientError("watch timed out; Run continues on server; use watch again")
                time.sleep(5)
        else:
            result = download(client, args.project, args.run, args.attempt, args.out)
        print(json.dumps(result, indent=2))
        if args.command == "execute" and result["status"] != "VERIFIED":
            return 2
        if args.command == "watch" and result["run"]["scheduler_state"] != "SUCCEEDED":
            return 2
        return 0
    except (ClientError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
