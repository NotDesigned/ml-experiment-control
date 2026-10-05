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
from .api import Client, ClientError, data_archive, download, save, segment, source_archive


def parser():
    value = argparse.ArgumentParser(prog="ml-exp", description=__doc__)
    value.add_argument("--version", action="version", version="%(prog)s " + __version__)
    value.add_argument("--url", default=os.environ.get("ML_EXPD_API_URL"))
    commands = value.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="write a small source project; no API connection")
    initialize.add_argument("directory", type=Path, help="new source directory")
    check = commands.add_parser("check", help="GET health, policy, executors and optionally schema")
    check.add_argument("--schema", type=Path)
    pack = commands.add_parser("pack", help="import source and package image; allocates no GPU")
    pack.add_argument("--project", required=True)
    pack.add_argument("--source", type=Path, required=True)
    environment = pack.add_mutually_exclusive_group(required=True)
    environment.add_argument("--image", help="immutable base image digest")
    environment.add_argument("--environment", help="ID from the server environment catalogue")
    environment.add_argument("--dockerfile", help="Dockerfile path inside the uploaded source; external FROM images must be digest-pinned")
    pack.add_argument("--requirements", help="pinned dependency file inside the uploaded source")
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
    upload = commands.add_parser("asset-upload", help="upload a data directory independently from source")
    upload.add_argument("--project", required=True)
    upload.add_argument("--directory", type=Path, required=True)
    upload.add_argument("--state", type=Path, required=True)
    assets = commands.add_parser("assets", help="list immutable project data assets")
    assets.add_argument("--project", required=True)
    snapshots = commands.add_parser("snapshots", help="list published checkpoints, including during training")
    snapshots.add_argument("--project", required=True)
    snapshots.add_argument("--run", required=True)
    snapshots.add_argument("--attempt", required=True)
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
            print(json.dumps({"source_dir": str(args.directory), "entrypoint": ["python", "train.py"]}, indent=2))
            return 0
        token = os.environ.get("ML_EXPD_API_TOKEN", "")
        if not token and os.environ.get("ML_EXPD_API_TOKEN_FILE"):
            token = Path(os.environ["ML_EXPD_API_TOKEN_FILE"]).read_text().strip()
        client = Client(args.url or "", token)
        health = client.negotiate()
        if args.command == "check":
            result = {"health": health, "policy": client.call("/api/actions/policy"),
                      "executors": client.call("/api/executors")}
            if "environments.v1" in health.get("capabilities", []):
                result["environments"] = client.call("/api/environments")
            if "data-assets.v1" in health.get("capabilities", []):
                result["storage_limits"] = client.call("/api/storage-limits")
            if args.schema:
                save(args.schema, client.call(health["openapi_path"]))
        elif args.command == "pack":
            if args.state.exists():
                raise ClientError("state file exists; inspect it with runtime instead of replaying pack")
            if (args.environment and "environments.v1" not in health.get("capabilities", [])
                    or args.requirements and "dependency-build.v1" not in health.get("capabilities", [])
                    or args.dockerfile and "dockerfile-build.v1" not in health.get("capabilities", [])):
                raise ClientError("server does not advertise the requested environment/dependency build capability")
            source = source_archive(args.source)
            query = urlencode({"project": args.project, "sha256": hashlib.sha256(source).hexdigest()})
            imported = client.call("/api/source-imports/archive?" + query, raw=source)
            save(args.state, imported)
            endpoint = "/api/projects/" + segment(args.project) + "/runtimes"
            definition = {"source_id": imported["source_id"], "entrypoint": args.entrypoint}
            if args.dockerfile:
                if args.requirements:
                    raise ClientError("install dependencies in the Dockerfile when using --dockerfile")
                definition["dockerfile"] = args.dockerfile
            else:
                definition["image" if args.image else "environment_id"] = args.image or args.environment
            if args.requirements:
                definition["requirements"] = args.requirements
            result = client.call(endpoint + "/prepare", data=definition)
            save(args.state, result)  # Recovery identity is saved before starting packaging.
            endpoint += "/" + segment(result["runtime_id"])
            if result["status"] == "PREPARED":
                client.call(endpoint + "/execute", data={"confirmation": result["confirmation"]})
            result = client.wait(endpoint, pending=("PREPARED", "EXECUTING"))
            save(args.state, result)
            if result["status"] != "READY":
                raise ClientError("packaging requires inspection; use runtime --state FILE --reconcile")
        elif args.command == "runtime":
            saved = json.loads(args.state.read_text())
            endpoint = f"/api/projects/{segment(saved['project'])}/runtimes/{segment(saved['runtime_id'])}"
            result = client.call(endpoint)
            if args.reconcile and result["status"] == "RECONCILE_REQUIRED":
                client.call(endpoint + "/reconcile", data={"confirmation": result["confirmation"]})
            result = client.wait(endpoint)
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
            result = client.call(f"/api/projects/{segment(saved['project'])}/runs", data=definition)
        elif args.command == "asset-upload":
            if args.state.exists():
                raise ClientError("state file exists; inspect the saved asset ID instead of replaying upload")
            limits = client.call("/api/storage-limits")
            with tempfile.TemporaryFile() as stream:
                digest, length = data_archive(args.directory, stream)
                if length > limits["asset_archive_bytes"]:
                    raise ClientError("data archive exceeds server upload limit")
                save(args.state, {"project": args.project, "asset_id": "asset." + digest, "status": "UPLOADING"})
                query = urlencode({"project": args.project, "sha256": digest})
                result = client.call("/api/assets/archive?" + query, raw=stream, length=length)
            save(args.state, result)
        elif args.command == "assets":
            result = client.call(f"/api/projects/{segment(args.project)}/assets")
        elif args.command == "snapshots":
            result = client.call(f"/api/runs/{segment(args.project)}/{segment(args.run)}/attempts/{segment(args.attempt)}/snapshots")
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
            result = client.wait(endpoint)
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
