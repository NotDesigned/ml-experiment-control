"""Command-line source/container experiment workflow over HTTP."""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import sys
import warnings

from . import __version__
from .api import Client, ClientError, error_payload
from .commands import dispatch, initialize
from .cli_layout import COMMAND_PATHS, GROUPS, CommandParser, output_context
from .connection import (MAX_TOKEN_BYTES, config_path, normalize_url, public_config, read_config,
                         read_token_file, resolve_connection, validate_token, write_config)
from .observation import observation_reference, observation_server
from .presentation import status_summary, operation_summary, experiment_summary
from .updater import update_client


def parser():
    value = CommandParser(prog="ml-exp", description="Run experiments through a remote ML-Expd service.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Start:   ml-exp init my-study\n"
               "Connect: ml-exp config set --url https://YOUR-API\n"
               "Prepare: ml-exp run submit experiment.json --state run.json\n"
               "Observe: ml-exp run status --state run.json\n"
               "Run submit prepares only; add --execute after review to authorize training.\n"
               "Use COMMAND --help for its tasks and options. Older flat commands remain accepted.")
    value.add_argument("--version", action="version", version="%(prog)s " + __version__)
    value.add_argument("--url", help="API base URL; overrides ML_EXPD_API_URL and saved configuration")
    value.add_argument("--config", dest="client_config", type=Path,
                       help="private client configuration; defaults to ML_EXPD_CLIENT_CONFIG or ~/.config/ml-expd/client.json")
    commands = value.add_subparsers(dest="group", required=True, parser_class=argparse.ArgumentParser)
    parents = {(): commands}
    for path, help_text in GROUPS.items():
        if len(path) > 1:
            continue
        group = parents[path[:-1]].add_parser(path[-1], help=help_text, description=help_text)
        parents[path] = group.add_subparsers(dest="_".join(path) + "_action", required=True,
                                             parser_class=argparse.ArgumentParser)

    def command(name, **kwargs):
        path = COMMAND_PATHS.get(name, (name,))
        if path[:-1] not in parents:
            parent = path[:-1]
            group = parents[parent[:-1]].add_parser(parent[-1], help=GROUPS[parent], description=GROUPS[parent])
            parents[parent] = group.add_subparsers(dest="_".join(parent) + "_action", required=True,
                                                   parser_class=argparse.ArgumentParser)
        leaf = parents[path[:-1]].add_parser(path[-1], **kwargs)
        leaf.set_defaults(command=name)
        return leaf
    config = command("config", help="save default API URL and private authentication; no API connection")
    config_actions = config.add_subparsers(dest="config_action", required=True)
    config_actions.add_parser("show", help="show saved defaults without displaying the token")
    config_actions.add_parser("clear", help="remove saved defaults; referenced token files are retained")
    config_set = config_actions.add_parser("set", help="save defaults; prompt securely for a token when needed")
    config_set.add_argument("--url", dest="config_url", help="API base URL including any reverse-proxy prefix")
    credential = config_set.add_mutually_exclusive_group()
    credential.add_argument("--token-file", type=Path, help="save a reference to an existing private token file")
    credential.add_argument("--token-stdin", action="store_true", help="read and save the token from standard input")
    credential.add_argument("--prompt-token", action="store_true", help="replace the saved token using a hidden prompt")
    update = command("update", help="update the standalone client from a verified official release; no service API connection")
    update.add_argument("--check", action="store_true", help="check the latest client version without installing")
    update.add_argument("--version", dest="release_version", help="choose an official stable release, for example 1.5.1")
    debug = command("debug", help="request and manage SSH debugging machines (CCI/DSW)")
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
    initialize = command("init", help="write a small source project; no API connection")
    initialize.add_argument("directory", type=Path, help="new source directory")
    initialize.add_argument("--base-image", help="approved sha256-pinned base; see ml-exp env list")
    workflow = command("experiment", help="prepare an experiment from one JSON config; training starts only with --execute", description="Upload source, build the environment and prepare the experiment. Without --execute this command does not authorize GPU training. Review the saved plan before explicitly adding --execute.")
    workflow.add_argument("config", type=Path)
    workflow.add_argument("--state", type=Path, required=True)
    workflow.add_argument("--resume", action="store_true", help="continue the same identities; never replay uncertain scheduler requests")
    workflow.add_argument("--continue-preparation", action="store_true", help="with --resume, continue after saved dependencies have been inspected; never replay uncertain effects")
    workflow.add_argument("--execute", action="store_true", help="authorize and execute the prepared submission within its GPU-hour budget")
    workflow.add_argument("--seconds", type=int, default=1800)
    workflow.add_argument("--download-to", type=Path)
    workflow.add_argument("--keep-server-copy", action="store_true", help="retain server archives after verified download")
    workflow.add_argument("--json", action="store_true", help="print machine-readable results and progress")
    status = command("status", help="read separate preparation, scheduler, training and publication status")
    status.add_argument("--project")
    status.add_argument("--run")
    status.add_argument("--attempt", help="exact attempt-NNN; otherwise resolve current Attempt once")
    status.add_argument("--state", type=Path, help="optional saved experiment state for preparation/submission evidence")
    status.add_argument("--details", action="store_true", help="expand evidence in --json output; never submit or recover work")
    status.add_argument("--json", action="store_true", help="print the structured status; combine with --details for all evidence")
    status.add_argument("--watch", action="store_true", help="poll the same exact Attempt without submitting or restarting it")
    status.add_argument("--seconds", type=float, default=1800, help="polling budget with --watch; in-flight HTTP reads keep their request timeout")
    run_list = command("run-list", help="list indexed runs; does not refresh or submit backend jobs")
    run_list.add_argument("--project", help="show only one project's runs")
    command("env-list", help="list approved base environments and their immutable image references")
    command("project-list", help="list remote projects and indexed run counts")
    logs = command("logs", help="read stored Attempt stdout/stderr; defaults to the current Attempt")
    metrics = command("metrics", help="read recorded Attempt metrics with their units, protocols and variants")
    for reading in (logs, metrics):
        reading.add_argument("--project")
        reading.add_argument("--run")
        reading.add_argument("--attempt", help="exact attempt-NNN; otherwise resolve the current Attempt once")
        reading.add_argument("--state", type=Path, help="read the original identities from saved experiment or operation state")
    logs.add_argument("--stream", choices=("both", "stdout", "stderr"), default="both")
    logs.add_argument("--lines", type=int, default=80)
    metrics.add_argument("--name", action="append", help="metric name to include; repeat for multiple names")
    metrics.add_argument("--keys", help="legacy comma-separated filter; prefer --name on current servers")
    metrics.add_argument("--max-points", type=int, default=2000)
    fetch = command("download", help="download exact-Attempt outputs and verify against archive")
    fetch.add_argument("--project")
    fetch.add_argument("--run")
    fetch.add_argument("--attempt", help="exact Attempt; required unless saved state binds one")
    fetch.add_argument("--state", type=Path, help="download the exact Attempt saved by experiment or an operation")
    fetch.add_argument("--out", type=Path, required=True, help="new destination directory")
    fetch.add_argument("--keep-server-copy", action="store_true", help="retain server archive after verified download")
    checkpoints = command("checkpoints", help="list backend-resident recovery checkpoints; no state download")
    checkpoints.add_argument("--project", required=True)
    checkpoints.add_argument("--run", required=True)
    checkpoints.add_argument("--attempt", required=True)
    collect = command("collect", help="recover original Attempt outputs without rerunning training; may allocate a bounded zero-GPU CPU job")
    collect.add_argument("--project", required=True)
    collect.add_argument("--run", required=True)
    collect.add_argument("--attempt", required=True)
    collect.add_argument("--start", action="store_true", help="authorize independent result collection")
    collect.add_argument("--retry", action="store_true")
    collect.add_argument("--reconcile", action="store_true")
    ack = command("acknowledge", help="reverify saved results and retry download confirmation without redownloading")
    ack.add_argument("--directory", type=Path, required=True)
    retention = ack.add_mutually_exclusive_group()
    retention.add_argument("--keep-server-copy", dest="keep_server_copy", action="store_true", default=None)
    retention.add_argument("--release-server-copy", dest="keep_server_copy", action="store_false")
    ack.set_defaults(keep_server_copy=None)
    cancel = command("cancel", help="prepare cancellation of an exact Attempt; review before --execute")
    retry = command("retry", help="prepare a new Attempt in the same frozen Run; does not select a checkpoint")
    for operation in (cancel, retry):
        operation.add_argument("--project", help="required when first preparing the plan")
        operation.add_argument("--run", help="required when first preparing the plan")
        operation.add_argument("--attempt", help="exact source attempt-NNN; required when first preparing")
        operation.add_argument("--reason", help="operator reason saved in the plan")
        operation.add_argument("--state", type=Path, required=True, help="saved Action plan; existing state is observed without replay")
        operation.add_argument("--from-state", type=Path, help="use the exact Attempt from experiment or earlier operation state; --state is a separate new plan")
        operation.add_argument("--json", action="store_true", help="print the complete reviewed Action and evidence")
        operation.add_argument("--action-id", help="attach the exact original Action after a lost preparation reply, for review")
        confirmation = operation.add_mutually_exclusive_group()
        confirmation.add_argument("--execute", action="store_true", help="explicitly authorize and execute an already saved, reviewed plan")
        confirmation.add_argument("--reconcile", action="store_true", help="observe the original uncertain Action; never resubmit")
    retry.add_argument("--max-gpu-hours", type=float, help="finite positive budget; required when first preparing")
    retry.add_argument("--new-attempt", help="optional exact new attempt-NNN identity")
    check = command("check", help="check connection, configured backends and training readiness")
    check.add_argument("--schema", type=Path)
    check.add_argument("--json", action="store_true", help="print full machine-readable evidence instead of the short summary")
    check.add_argument("--training", action="store_true",
                       help="also require training configuration and a responding image builder; allocates no GPU")
    source = command("source-import", help="create/reuse a remote project and upload immutable source; no image build or GPU submission")
    source.add_argument("--project", required=True)
    source.add_argument("--directory", type=Path, required=True)
    pack = command("pack", help="build an environment from source and a Dockerfile; no training GPU")
    pack.add_argument("--project", required=True)
    pack.add_argument("--source", type=Path, required=True)
    pack.add_argument("--dockerfile", default="Dockerfile", help="path inside source; external FROM images must be digest-pinned (default: Dockerfile)")
    pack.add_argument("--entrypoint", nargs="+", default=["python", "train.py"])
    pack.add_argument("--state", type=Path, required=True)
    tracking = command("tracking", help="inspect or retry server-side W&B publication")
    tracking.add_argument("--project", required=True)
    tracking.add_argument("--run", required=True)
    tracking.add_argument("--retry", action="store_true")
    wb = command("wandb", help="configure server defaults; API key is read from a private file")
    wb.add_argument("--key-file", type=Path)
    wb.add_argument("--entity")
    wb.add_argument("--project", help="optional default W&B project; otherwise use the ML-Expd project name")
    wb.add_argument("--disable", action="store_true")
    wb.add_argument("--enable", action="store_true")
    wb.add_argument("--clear-credentials", action="store_true")
    wb.add_argument("--preparation", action=argparse.BooleanOptionalAction, default=None,
                    help="enable/disable preparation W&B runs independently; default disabled")
    dataset = command("asset-import-oss",help="register and verify a personal OSS dataset archive")
    dataset.add_argument("--project",required=True)
    dataset.add_argument("--executor",required=True)
    dataset.add_argument("--object-key",required=True)
    dataset.add_argument("--sha256",required=True)
    dataset.add_argument("--bytes",type=int,required=True)
    dataset = command("asset-download",help="download and verify a dataset without deleting OSS data")
    dataset.add_argument("--project",required=True)
    dataset.add_argument("--id",required=True)
    dataset.add_argument("--out",type=Path,required=True)
    upload = command("asset-upload", help="upload a data directory independently from source")
    upload.add_argument("--project", required=True)
    upload.add_argument("--directory", type=Path, required=True)
    upload.add_argument("--state", type=Path, required=True)
    upload.add_argument("--resume", action="store_true", help="resume the exact saved data archive; completed parts are retained")
    upload.add_argument("--executor", help="target executor: Slurm shared storage, SenseCore NAS through CCR, or DLC OSS")
    upload.add_argument("--wait-seconds", type=int, default=1800)
    assets = command("assets", help="list immutable project data assets")
    assets.add_argument("--project", required=True)
    runtime = command("runtime", help="inspect a saved environment build and optionally read its logs")
    runtime.add_argument("--state", type=Path, required=True)
    runtime.add_argument("--reconcile", action="store_true")
    runtime.add_argument("--logs", action="store_true", help="read recent server-side build logs")
    deletion = command("delete", help="prepare permanent administrative deletion of one managed Run; review before --execute")
    deletion.add_argument("--project", help="required when first preparing the deletion plan")
    deletion.add_argument("--run", help="exact managed Run; required when first preparing")
    deletion.add_argument("--reason", help="required administrative deletion reason")
    deletion.add_argument("--state", type=Path, required=True, help="saved deletion Action; existing state is observed without replay")
    deletion.add_argument("--action-id", help="attach the exact original Action after a lost preparation reply, for review")
    deletion_confirmation = deletion.add_mutually_exclusive_group()
    deletion_confirmation.add_argument("--execute", action="store_true", help="explicitly authorize permanent deletion of the reviewed saved targets")
    deletion_confirmation.add_argument("--reconcile", action="store_true", help="resume observation of the original deletion Action")
    create = command("create", help="freeze a Run; does not submit it")
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
    prepare = command("prepare", help="preflight and save Submission gates; no GPU submission")
    prepare.add_argument("--project", required=True)
    prepare.add_argument("--run", required=True)
    prepare.add_argument("--max-gpu-hours", type=float, required=True)
    prepare.add_argument("--state", type=Path, required=True)
    execute = command("execute", help="authorize and submit the exact saved Submission")
    execute.add_argument("--state", type=Path, required=True)
    execute.add_argument("--confirm", required=True, help="exact confirmation printed by prepare")
    submission = command("submission", help="inspect/poll an existing Submission")
    submission.add_argument("--id", required=True)
    submission.add_argument("--reconcile", action="store_true", help="observe uncertain submission; never replay submit")
    watch = command("watch", help="observe a Run until scheduler completion")
    watch.add_argument("--project", required=True)
    watch.add_argument("--run", required=True)
    watch.add_argument("--seconds", type=float, default=1800)
    snapshots = command("snapshots", help="list published checkpoints, including during training")
    snapshots.add_argument("--project", required=True)
    snapshots.add_argument("--run", required=True)
    snapshots.add_argument("--attempt", required=True)
    return value


def check_summary(result):
    """Present configuration evidence without implying backend admission."""
    health, policy = result["health"], result["policy"]
    readiness = result.get("readiness", {})
    enabled = policy.get("allow_source_imports", health.get("source_imports"))
    source = ("DISABLED" if enabled is False else "AVAILABLE"
              if "source-import.v1" in health.get("capabilities", []) else "UNKNOWN")
    lines = ["API: READY", "Source import: " + source,
             "Training: " + readiness.get("training", "UNKNOWN")]
    components = readiness.get("components", {})
    components = components if isinstance(components, dict) else {}
    labels = {"executors": "Executors", "image_builder": "Image builder",
              "artifact_store": "Result store", "project_writes": "Project writes",
              "source_imports": "Source imports", "scheduler_mutations": "Scheduler writes"}
    for key in readiness.get("issues", []):
        if key in labels:
            component = components.get(key)
            status = component.get("status") if isinstance(component, dict) else None
            if not isinstance(status, str) or status not in {"NOT_CONFIGURED", "DISABLED", "UNAVAILABLE", "INVALID", "UNKNOWN"}:
                status = "NOT_READY"
            lines.append("  " + labels[key] + ": " + status)
    if "storage_limits_error" in result:
        lines.append("Storage limits: ERROR (use --json for diagnostics)")
    if readiness.get("training") == "CONFIGURED":
        lines.append("Configuration ready; backend access and resource admission are not verified.")
    else:
        lines.append("Next: ask the service operator to configure a backend and result store, then a builder, and enable training after verification.")
        lines.append("Setup: https://github.com/NotDesigned/ml-experiment-control/blob/main/docs/operator-guide.md#enable-training-after-source-import")
    lines.append("Use --json for executors, base images, limits and full evidence.")
    return "\n".join(lines)


def configure(args):
    """Update only local connection defaults; never authenticate or dispatch work."""
    path = config_path(args.client_config)
    if args.config_action == "clear":
        path.unlink(missing_ok=True)
        return {"config_file": str(path), "cleared": True}
    previous = read_config(path)
    if args.config_action == "show":
        return public_config(path, previous)
    url = args.config_url or args.url or previous.get("url")
    if not url:
        raise ClientError("set an API URL with ml-exp config set --url URL")
    url = normalize_url(url)
    if args.token_file is not None:
        token_path = args.token_file.expanduser().resolve()
        read_token_file(token_path)
        auth = {"token_file": str(token_path)}
    elif args.token_stdin:
        supplied = sys.stdin.read(MAX_TOKEN_BYTES + 3)
        if len(supplied) > MAX_TOKEN_BYTES + 2:
            raise ClientError("token input is too large; configuration unchanged")
        auth = {"token": validate_token(supplied.strip())}
    elif (args.prompt_token or previous.get("url") != url
          or not (previous.get("token") or previous.get("token_file"))):
        if not sys.stdin.isatty():
            raise ClientError("token input needs a terminal; use --token-file PATH or --token-stdin")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                token = getpass.getpass("API token (hidden): ")
        except (EOFError, KeyboardInterrupt, getpass.GetPassWarning):
            raise ClientError("secure token entry unavailable or cancelled; configuration unchanged") from None
        auth = {"token": validate_token(token)}
    else:
        auth = {key: previous[key] for key in ("token", "token_file") if key in previous}
    saved = {"url": url, **auth}
    write_config(path, saved)
    return {"saved": True, **public_config(path, saved)}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            print(json.dumps(initialize(args), indent=2))
            return 0
        if args.command == "config":
            print(json.dumps(configure(args), indent=2))
            return 0
        if args.command == "update":
            print(json.dumps(update_client(args.release_version, check=args.check), indent=2))
            return 0
        settings = read_config(config_path(args.client_config))
        args.url = args.url or os.environ.get("ML_EXPD_API_URL") or settings.get("url")
        # Validate a saved server before sending the token, including negotiate.
        reference_path = (getattr(args, "from_state", None)
                          or (getattr(args, "state", None) if args.command in
                              {"status", "logs", "metrics", "download", "retry", "cancel"} else None))
        if reference_path is not None and reference_path.exists():
            saved_server = (observation_server(reference_path) if args.command in {"retry", "cancel"}
                            and getattr(args, "from_state", None) is None
                            else observation_reference(reference_path).get("server"))
            if saved_server:
                if args.url and args.url.rstrip("/") != saved_server.rstrip("/"):
                    raise ClientError("saved state belongs to a different server; no request was sent")
                args.url = saved_server
        args.url, token = resolve_connection(args.url, settings)
        client = Client(args.url or "", token)
        health = client.negotiate()
        result = output_context(dispatch(args, client, health), args.client_config)
        if args.command == "check" and not args.json:
            output = check_summary(result)
        elif args.command == "status" and not args.json:
            output = status_summary(result)
        elif args.command in {"cancel", "retry"} and not args.json:
            output = operation_summary(result)
        elif args.command == "experiment" and not args.json:
            context = str(args.client_config.expanduser().resolve()) if args.client_config is not None else None
            output = experiment_summary(result, state_file=str(args.state), executed=args.execute, client_config=context)
        else:
            output = json.dumps(result, indent=2)
        print(output)
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
