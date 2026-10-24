"""Command-line source/container experiment workflow over HTTP."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from . import __version__
from .api import Client, ClientError, error_payload
from .commands import dispatch, initialize


def parser():
    value = argparse.ArgumentParser(prog="ml-exp", description=__doc__)
    value.add_argument("--version", action="version", version="%(prog)s " + __version__)
    value.add_argument("--url", default=os.environ.get("ML_EXPD_API_URL"),
                       help="API base URL including proxy prefix; defaults to ML_EXPD_API_URL")
    commands = value.add_subparsers(dest="command", required=True)
    debug = commands.add_parser("debug", help="request and manage SSH debugging machines (CCI/DSW)")
    debug_actions = debug.add_subparsers(dest="debug_action", required=True)
    debug_actions.add_parser("profiles", help="list administrator-approved machines and limits")
    debug_actions.add_parser("list", help="list debug sessions; no cloud refresh")
    debug_create = debug_actions.add_parser("create", help="create once under a stable session ID; uploads only a public key")
    debug_create.add_argument("--id", required=True)
    debug_create.add_argument("--profile", required=True)
    debug_create.add_argument("--ssh-key", type=Path, required=True)
    debug_create.add_argument("--duration-seconds", type=int, default=1800)
    for operation in ("status", "refresh", "stop", "delete"):
        action = debug_actions.add_parser(operation)
        action.add_argument("--id", required=True)
    initialize = commands.add_parser("init", help="write a small source project; no API connection")
    initialize.add_argument("directory", type=Path, help="new source directory")
    initialize.add_argument("--base-image", help="approved sha256-pinned base; see ml-exp check")
    workflow = commands.add_parser("experiment", help="validate one JSON config, build, prepare, optionally execute and download")
    workflow.add_argument("config", type=Path)
    workflow.add_argument("--state", type=Path, required=True)
    workflow.add_argument("--resume", action="store_true", help="continue the same identities; never replay uncertain scheduler requests")
    workflow.add_argument("--continue-preparation", action="store_true", help="with --resume, continue after saved dependencies have been inspected; never replay uncertain effects")
    workflow.add_argument("--execute", action="store_true", help="authorize and execute the prepared submission within its GPU-hour budget")
    workflow.add_argument("--seconds", type=int, default=1800)
    workflow.add_argument("--download-to", type=Path)
    workflow.add_argument("--keep-server-copy", action="store_true", help="retain server archives after verified download")
    check = commands.add_parser("check", help="GET health, policy, executors and optionally schema")
    check.add_argument("--schema", type=Path)
    check.add_argument("--training", action="store_true",
                       help="also require training configuration and a responding image builder; allocates no GPU")
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
    create.add_argument("--max-time", help="HH:MM:SS; omitted uses the server's backend default (DLC: 24 hours)")
    create.add_argument("--allocation-policy", choices=["guaranteed", "allow_idle", "idle_only"], help="DLC resource policy; default guaranteed")
    create.add_argument("--priority", type=int, choices=range(1, 10), help="DLC task priority; default 5")
    create.add_argument("--inputs", type=json.loads, default=[], help='JSON array: [{"asset_id":"asset.…","mount_path":"/inputs/data"}]')
    create.add_argument("--checkpoint-interval", type=int, help="publish atomic checkpoint.ready.json every N seconds")
    create.add_argument("--data-preparation", type=json.loads, help='JSON: {"script":"download_data.py","arguments":[],"timeout_seconds":1200}')
    create.add_argument("--checkpoint-state-interval", type=int, help="register persistent STATE_DIR checkpoints every N seconds; no state upload")
    create.add_argument("--resume-from", type=json.loads, help='JSON: {"run_id":"trial","attempt_id":"attempt-001","checkpoint_id":"checkpoint.…"}')
    create.add_argument("--wandb", type=json.loads, help='JSON: {"enabled":true,"project":"fineweb"}; default enabled if configured')
    create.add_argument("--parameters", type=json.loads, help='explicit research parameters, e.g. {"lr":0.001,"seed":42}')
    tracking = commands.add_parser("tracking", help="inspect or retry server-side W&B publication")
    tracking.add_argument("--project", required=True)
    tracking.add_argument("--run", required=True)
    tracking.add_argument("--retry", action="store_true")
    wb = commands.add_parser("wandb", help="configure server defaults; API key is read from a private file")
    wb.add_argument("--key-file", type=Path)
    wb.add_argument("--entity")
    wb.add_argument("--project", help="optional default W&B project; otherwise use the ML-Expd project name")
    wb.add_argument("--disable", action="store_true")
    wb.add_argument("--enable", action="store_true")
    wb.add_argument("--clear-credentials", action="store_true")
    wb.add_argument("--preparation", action=argparse.BooleanOptionalAction, default=None,
                    help="enable/disable preparation W&B runs independently; default disabled")
    dataset = commands.add_parser("asset-import-oss",help="register and verify a personal OSS dataset archive")
    dataset.add_argument("--project",required=True)
    dataset.add_argument("--executor",required=True)
    dataset.add_argument("--object-key",required=True)
    dataset.add_argument("--sha256",required=True)
    dataset.add_argument("--bytes",type=int,required=True)
    dataset = commands.add_parser("asset-download",help="download and verify a dataset without deleting OSS data")
    dataset.add_argument("--project",required=True)
    dataset.add_argument("--id",required=True)
    dataset.add_argument("--out",type=Path,required=True)
    upload = commands.add_parser("asset-upload", help="upload a data directory independently from source")
    upload.add_argument("--project", required=True)
    upload.add_argument("--directory", type=Path, required=True)
    upload.add_argument("--state", type=Path, required=True)
    upload.add_argument("--resume", action="store_true", help="resume the exact saved data archive; completed parts are retained")
    upload.add_argument("--executor", help="target executor: Slurm shared storage, SenseCore NAS through CCR, or DLC OSS")
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
    status = commands.add_parser("status", help="read separate preparation, scheduler, training and publication status")
    status.add_argument("--project", required=True)
    status.add_argument("--run", required=True)
    status.add_argument("--attempt", help="exact attempt-NNN; otherwise resolve current Attempt once")
    status.add_argument("--state", type=Path, help="optional saved experiment state for preparation/submission evidence")
    status.add_argument("--details", action="store_true", help="expand existing API evidence; never submit or recover work")
    logs = commands.add_parser("logs", help="read stored Attempt stdout/stderr; defaults to the current Attempt")
    metrics = commands.add_parser("metrics", help="read recorded Attempt metrics with their units, protocols and variants")
    for reading in (logs, metrics):
        reading.add_argument("--project", required=True)
        reading.add_argument("--run", required=True)
        reading.add_argument("--attempt", help="exact attempt-NNN; otherwise resolve the current Attempt once")
    logs.add_argument("--stream", choices=("both", "stdout", "stderr"), default="both")
    logs.add_argument("--lines", type=int, default=80)
    metrics.add_argument("--keys", help="comma-separated recorded metric names; server filtering preserves dimensions")
    metrics.add_argument("--max-points", type=int, default=2000)
    cancel = commands.add_parser("cancel", help="prepare cancellation of an exact Attempt; review before --execute")
    retry = commands.add_parser("retry", help="prepare a new Attempt in the same frozen Run; does not select a checkpoint")
    for operation in (cancel, retry):
        operation.add_argument("--project", help="required when first preparing the plan")
        operation.add_argument("--run", help="required when first preparing the plan")
        operation.add_argument("--attempt", help="exact source attempt-NNN; required when first preparing")
        operation.add_argument("--reason", help="operator reason saved in the plan")
        operation.add_argument("--state", type=Path, required=True, help="saved Action plan; existing state is observed without replay")
        operation.add_argument("--action-id", help="attach the exact original Action after a lost preparation reply, for review")
        confirmation = operation.add_mutually_exclusive_group()
        confirmation.add_argument("--execute", action="store_true", help="explicitly authorize and execute an already saved, reviewed plan")
        confirmation.add_argument("--reconcile", action="store_true", help="observe the original uncertain Action; never resubmit")
    retry.add_argument("--max-gpu-hours", type=float, help="finite positive budget; required when first preparing")
    retry.add_argument("--new-attempt", help="optional exact new attempt-NNN identity")
    deletion = commands.add_parser("delete", help="prepare permanent administrative deletion of one managed Run; review before --execute")
    deletion.add_argument("--project", help="required when first preparing the deletion plan")
    deletion.add_argument("--run", help="exact managed Run; required when first preparing")
    deletion.add_argument("--reason", help="required administrative deletion reason")
    deletion.add_argument("--state", type=Path, required=True, help="saved deletion Action; existing state is observed without replay")
    deletion.add_argument("--action-id", help="attach the exact original Action after a lost preparation reply, for review")
    deletion_confirmation = deletion.add_mutually_exclusive_group()
    deletion_confirmation.add_argument("--execute", action="store_true", help="explicitly authorize permanent deletion of the reviewed saved targets")
    deletion_confirmation.add_argument("--reconcile", action="store_true", help="resume observation of the original deletion Action")
    fetch = commands.add_parser("download", help="download exact-Attempt outputs and verify against archive")
    fetch.add_argument("--project", required=True)
    fetch.add_argument("--run", required=True)
    fetch.add_argument("--attempt", required=True)
    fetch.add_argument("--out", type=Path, required=True, help="new destination directory")
    fetch.add_argument("--keep-server-copy", action="store_true", help="retain server archive after verified download")
    ack = commands.add_parser("acknowledge", help="reverify saved results and retry download confirmation without redownloading")
    ack.add_argument("--directory", type=Path, required=True)
    retention = ack.add_mutually_exclusive_group()
    retention.add_argument("--keep-server-copy", dest="keep_server_copy", action="store_true", default=None)
    retention.add_argument("--release-server-copy", dest="keep_server_copy", action="store_false")
    ack.set_defaults(keep_server_copy=None)
    collect = commands.add_parser("collect", help="recover original Attempt outputs without rerunning training; may allocate a bounded zero-GPU CPU job")
    collect.add_argument("--project", required=True)
    collect.add_argument("--run", required=True)
    collect.add_argument("--attempt", required=True)
    collect.add_argument("--start", action="store_true", help="authorize independent result collection")
    collect.add_argument("--retry", action="store_true")
    collect.add_argument("--reconcile", action="store_true")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            print(json.dumps(initialize(args), indent=2))
            return 0
        token = os.environ.get("ML_EXPD_API_TOKEN", "")
        token_file = os.environ.get("ML_EXPD_API_TOKEN_FILE")
        if token and token_file:
            raise ClientError("set only one of ML_EXPD_API_TOKEN_FILE and ML_EXPD_API_TOKEN")
        if token_file:
            token = Path(token_file).read_text().strip()
        client = Client(args.url or "", token)
        health = client.negotiate()
        result = dispatch(args, client, health)
        print(json.dumps(result, indent=2))
        if args.command == "check" and "storage_limits_error" in result:
            return 2
        if args.command == "check" and args.training and result.get("readiness", {}).get("training") != "CONFIGURED":
            return 2
        if args.command == "execute" and result["status"] != "VERIFIED":
            return 2
        if args.command in {"cancel", "retry"} and args.execute and result["action"]["execution"]["status"] not in {"EXECUTING", "VERIFIED"}:
            return 2
        if args.command == "delete" and args.execute and not result["deleted"]:
            return 2
        if args.command == "watch" and result["run"]["scheduler_state"] != "SUCCEEDED":
            return 2
        return 0
    except ClientError as exc:
        print(json.dumps(error_payload(exc, locals().get("token", "")), ensure_ascii=False), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
