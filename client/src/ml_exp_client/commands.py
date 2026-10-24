"""Concrete HTTP command operations; argparse and presentation stay in cli."""
from __future__ import annotations

import hashlib
from importlib.resources import files
import json
import re
import sys
import tempfile
import time
from urllib.parse import urlencode

from .api import ClientError, error_payload, segment
from .archives import data_archive, source_archive
from .artifacts import acknowledge, download
from .assets import deliver_data, upload_asset_parts, download_asset
from .config import validate_dockerfile
from .operations import training_operation
from .run_deletion import run_deletion
from .state import save
from .status import read_attempt, run_status
from .workflow import experiment, report


def initialize(args):
    args.directory.mkdir(parents=True, exist_ok=False)
    template = files("ml_exp_client").joinpath("templates", "train.py.txt")
    (args.directory / "train.py").write_bytes(template.read_bytes())
    # The exact stdlib module also works without installing the client
    # in a training image. Keep one canonical implementation.
    helper = files("ml_exp_client").joinpath("metrics.py")
    (args.directory / "ml_exp_metrics.py").write_bytes(helper.read_bytes())
    base = args.base_image or "REPLACE_WITH_APPROVED_BASE@sha256:" + "0" * 64
    (args.directory / "Dockerfile").write_text(f"FROM {base}\nWORKDIR /workspace\n", encoding="utf-8")
    return {"source_dir": str(args.directory), "entrypoint": ["python", "train.py"]}


def dispatch(args, client, health):
    if args.command in {"asset-import-oss","asset-download"}:
        if "oss-datasets.v1" not in health.get("capabilities",[]):
            raise ClientError("server lacks oss-datasets.v1")
        endpoint=f"/api/projects/{segment(args.project)}/assets"
        if args.command=="asset-import-oss":
            result=client.call(endpoint+"/import-oss",data={"executor":args.executor,"object_key":args.object_key,
                                                          "sha256":args.sha256,"bytes":args.bytes},timeout=1200)
        else:
            result=download_asset(client,args.project,args.id,args.out)
    elif args.command == "debug":
        from .debug_sessions import debug_session
        result = debug_session(args, client, health)
    elif args.command == "check":
        result = check(args, client, health)
    elif args.command == "source-import":
        if "source-import.v1" not in health.get("capabilities", []):
            raise ClientError("server does not advertise source import")
        if health.get("source_imports") is False:
            raise ClientError("source imports are disabled by the server")
        result = import_source(client, args.project, args.directory)
    elif args.command == "collect":
        if "result-collection.v1" not in health.get("capabilities", []):
            raise ClientError("server lacks result-collection.v1")
        endpoint = f"/api/runs/{segment(args.project)}/{segment(args.run)}/attempts/{segment(args.attempt)}/collection"
        result = (client.call(endpoint, data={"retry": args.retry, "reconcile": args.reconcile})
                  if args.start or args.retry or args.reconcile else client.call(endpoint))
    elif args.command == "status":
        result = run_status(client, health, args.project, args.run, args.attempt,
                            state_path=args.state, details=args.details)
    elif args.command in {"logs", "metrics"}:
        options = ({"stream": args.stream, "lines": args.lines} if args.command == "logs"
                   else {"keys": args.keys, "names": getattr(args, "name", None), "max_points": args.max_points})
        result = read_attempt(client, args.command, args.project, args.run, args.attempt, **options)
    elif args.command in {"cancel", "retry"}:
        result = training_operation(client, args.command, args.state, project=args.project,
            run=args.run, attempt=args.attempt, reason=args.reason, execute=args.execute,
            reconcile=args.reconcile, action_id=args.action_id,
            max_gpu_hours=getattr(args, "max_gpu_hours", None),
            new_attempt=getattr(args, "new_attempt", None))
    elif args.command == "delete":
        if "resource-deletion.v1" not in health.get("capabilities", []):
            raise ClientError("server lacks resource-deletion.v1")
        result = run_deletion(client, args.state, project=args.project, run=args.run,
            reason=args.reason, execute=args.execute, reconcile=args.reconcile, action_id=args.action_id)
    elif args.command == "experiment":
        result = experiment(client, health, args.config, args.state, resume=args.resume,
                            execute=args.execute, seconds=args.seconds, out=args.download_to,
                            continue_preparation=args.continue_preparation, keep_server_copy=args.keep_server_copy)
    elif args.command == "acknowledge":
        result = acknowledge(client, args.directory, keep_server_copy=args.keep_server_copy)
    elif args.command in {"tracking", "wandb"}:
        result = wandb(args, client, health)
    elif args.command == "pack":
        result = pack(args, client, health)
    elif args.command == "runtime":
        result = runtime(args, client, health)
    elif args.command == "create":
        result = create_run(args, client, health)
    elif args.command == "asset-upload":
        result = upload_asset(args, client, health)
    elif args.command == "assets":
        result = client.call(f"/api/projects/{segment(args.project)}/assets")
    elif args.command == "snapshots":
        result = client.call(f"/api/runs/{segment(args.project)}/{segment(args.run)}/attempts/{segment(args.attempt)}/snapshots")
    elif args.command == "checkpoints":
        if "persistent-checkpoints.v1" not in health.get("capabilities", []):
            raise ClientError("server lacks persistent-checkpoints.v1")
        result = client.call(f"/api/runs/{segment(args.project)}/{segment(args.run)}/attempts/{segment(args.attempt)}/checkpoints")
    elif args.command == "prepare":
        result = prepare(args, client, health)
    elif args.command in {"execute", "submission"}:
        result = submission(args, client, health)
    elif args.command == "watch":
        result = watch(args, client, health)
    else:
        result = download(client, args.project, args.run, args.attempt, args.out, keep_server_copy=args.keep_server_copy)
    return result


def check(args, client, health):
    result = {"health": health, "policy": client.call("/api/actions/policy"),
              "executors": client.call("/api/executors")}
    if "deployment-readiness.v1" in health.get("capabilities", []):
        result["readiness"] = client.call("/api/readiness")
    elif "deployment" in health:
        result["readiness"] = health["deployment"]
    if "readiness" in result:
        readiness = result["readiness"]
        if (not isinstance(readiness, dict) or readiness.get("api") != "READY"
                or readiness.get("training") not in {"CONFIGURED", "INCOMPLETE"}):
            raise ClientError("invalid deployment readiness response")
    if "environments.v1" in health.get("capabilities", []):
        result["environments"] = client.call("/api/environments")
    if "data-assets.v1" in health.get("capabilities", []):
        try:
            result["storage_limits"] = client.call("/api/storage-limits")
        except ClientError as error:
            if error.status == 404 and error.details == "data asset storage is not configured":
                # Older servers advertise all implemented routes, including
                # optional unconfigured storage. Other 404/5xx remain failures.
                result["storage_limits"] = {"status": "NOT_CONFIGURED"}
            else:
                result["storage_limits_error"] = error_payload(error, getattr(client, "token", ""))
    if getattr(args, "training", False) and "readiness" not in result:
        result["readiness"] = {"api": "READY", "training": "UNKNOWN",
                               "issues": ["server_does_not_report_training_readiness"]}
    if args.schema:
        save(args.schema, client.call(health["openapi_path"]))
    return result


def prepare(args, client, health):
    if args.state.exists():
        raise ClientError("state file exists; inspect the saved Submission")
    result = client.call(f"/api/experiments/{segment(args.project)}/{segment(args.run)}/submissions/prepare",
                         data={"max_gpu_hours": args.max_gpu_hours, "reason": "API client trial"})
    save(args.state, result)
    return result


def _wait(client, health, endpoint, **kwargs):
    if "execution-progress.v1" in health.get("capabilities", []):
        kwargs["observer"] = lambda value: report(client.call(endpoint + "/progress"))
    return client.wait(endpoint, **kwargs)


def _runtime_state(path):
    saved = json.loads(path.read_text())
    if (not isinstance(saved, dict) or not isinstance(saved.get("project"), str)
            or not isinstance(saved.get("runtime_id"), str)
            or not re.fullmatch(r"runtime\.[a-f0-9]{64}", saved["runtime_id"])):
        raise ClientError("runtime state needs a project and exact runtime.<sha256> identity; use the saved pack state")
    segment(saved["project"])
    return saved


def wandb(args, client, health):
    if "wandb-sync.v1" not in health.get("capabilities", []):
        raise ClientError("server lacks wandb-sync.v1")
    if args.command == "tracking":
        endpoint = f"/api/runs/{segment(args.project)}/{segment(args.run)}/tracking"
        result = client.call(endpoint + "/retry", data={}) if args.retry else client.call(endpoint)
    elif args.key_file or args.entity or args.project or args.disable or args.enable or args.clear_credentials or args.preparation is not None:
        if args.enable and args.disable:
            raise ClientError("choose --enable or --disable")
        definition = {}
        if args.preparation is not None:
            if "wandb-preparation-sync.v1" not in health.get("capabilities", []):
                raise ClientError("server lacks wandb-preparation-sync.v1")
            definition["preparation_enabled"] = args.preparation
        if args.enable or args.disable:
            definition["enabled"] = args.enable
        for key in ("entity", "project"):
            if getattr(args, key) is not None:
                definition[key] = getattr(args, key)
        if args.clear_credentials:
            definition["clear_credentials"] = True
        if args.key_file:
            definition["api_key"] = args.key_file.read_text().strip()
        result = client.call("/api/tracking/wandb", data=definition, method="PUT")
    else:
        result = client.call("/api/tracking/wandb")
    return result


def import_source(client, project, directory):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", project):
        raise ClientError("invalid project identity")
    source = source_archive(directory)
    query = urlencode({"project": project, "sha256": hashlib.sha256(source).hexdigest()})
    return client.call("/api/source-imports/archive?" + query, raw=source)


def pack(args, client, health):
    if args.state.exists():
        raise ClientError("state file exists; inspect it with runtime instead of replaying pack")
    if "dockerfile-build.v1" not in health.get("capabilities", []):
        raise ClientError("server does not advertise the Dockerfile build capability")
    validate_dockerfile(args.source, args.dockerfile)
    imported = import_source(client, args.project, args.source)
    save(args.state, imported)
    endpoint = "/api/projects/" + segment(args.project) + "/runtimes"
    definition = {"source_id": imported["source_id"], "entrypoint": args.entrypoint}
    definition["dockerfile"] = args.dockerfile
    result = client.call(endpoint + "/prepare", data=definition)
    save(args.state, result)  # Recovery identity is saved before starting packaging.
    endpoint += "/" + segment(result["runtime_id"])
    if result["status"] == "PREPARED":
        client.call(endpoint + "/execute", data={"confirmation": result["confirmation"]})
    result = _wait(client, health, endpoint, pending=("PREPARED", "EXECUTING"))
    save(args.state, result)
    if result["status"] != "READY":
        raise ClientError("packaging requires inspection; use runtime --state FILE --reconcile")
    return result


def runtime(args, client, health):
    saved = _runtime_state(args.state)
    endpoint = f"/api/projects/{segment(saved['project'])}/runtimes/{segment(saved['runtime_id'])}"
    result = client.call(endpoint)
    if args.reconcile and result["status"] in {"RECONCILE_REQUIRED", "EXECUTING"}:
        client.call(endpoint + "/reconcile", data={"confirmation": result["confirmation"]})
    result = _wait(client, health, endpoint)
    save(args.state, result)
    if args.logs:
        result = {"runtime": result, "build_logs": client.call(endpoint + "/logs")}
    return result


def create_run(args, client, health):
    saved = _runtime_state(args.runtime_state)
    definition = {
        "runtime_id": saved["runtime_id"], "run_id": args.run, "executor": args.executor,
        "arguments": args.arguments, "resources": {"gpus": args.gpus, "cpus": args.cpus,
        "memory_gb": args.memory_gb}, "outputs": ["**/*"], "inputs": args.inputs}
    if args.max_time is not None:
        definition["resources"]["max_time"] = args.max_time
    for key in ("allocation_policy", "priority"):
        if getattr(args, key) is not None:
            if "pai-idle-resources.v1" not in health.get("capabilities", []):
                raise ClientError("server lacks pai-idle-resources.v1")
            definition["resources"][key] = getattr(args, key)
    if args.wandb is not None:
        definition["wandb"] = args.wandb
    if args.parameters is not None:
        definition["parameters"] = args.parameters
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
    return result


def upload_asset(args, client, health):
    if args.state.exists() and not args.resume:
        raise ClientError("state file exists; inspect the saved asset ID instead of replaying upload")
    targeted = "target-backed-data-upload.v1" in health.get("capabilities", [])
    if targeted and not args.executor:
        raise ClientError("asset-upload requires --executor to select persistent backend storage")
    kind = None
    if args.executor:
        catalogue = client.call("/api/executors")["executors"]
        matches = [item for item in catalogue if item["id"] == args.executor]
        if len(matches) != 1 or matches[0].get("kind") not in {"slurm", "sensecore", "pai-dlc"}:
            raise ClientError("data upload requires a published Slurm, SenseCore or DLC executor")
        kind = matches[0]["kind"]
        if kind == "pai-dlc" and "oss-datasets.v1" not in health.get("capabilities",[]):
            raise ClientError("server lacks oss-datasets.v1")
        if kind == "pai-dlc" and not matches[0].get("capabilities",{}).get("data",{}).get("assets"):
            raise ClientError("DLC OSS dataset storage is not enabled")
        if kind == "sensecore" and "ccr-data-delivery.v1" not in health.get("capabilities", []):
            raise ClientError("server lacks ccr-data-delivery.v1")
    limits = client.call("/api/storage-limits")
    with tempfile.TemporaryFile() as stream:
        digest, length = data_archive(args.directory, stream)
        if limits["asset_archive_bytes"] is not None and length > limits["asset_archive_bytes"]:
            raise ClientError("data archive exceeds server upload limit")
        if args.resume:
            saved = json.loads(args.state.read_text())
            if saved.get("project") != args.project or saved.get("asset_id") != "asset." + digest:
                raise ClientError("resume directory differs from the saved upload archive")
            if targeted and (saved.get("target", {}).get("executor", saved.get("executor", args.executor)) != args.executor):
                raise ClientError("resume upload belongs to another executor")
            if saved.get("status") == "READY" and not args.executor:
                return saved
        save(args.state, {**(saved if args.resume else {}), "project": args.project, "asset_id": "asset." + digest, "status": "UPLOADING",
                          **({"target": {"executor": args.executor}} if targeted else {})})
        if "multipart-upload.v1" in health.get("capabilities", []):
            result = upload_asset_parts(client, args.project, stream, digest, length, args.state,
                                        target={"executor": args.executor} if targeted else None)
        else:
            query = urlencode({"project": args.project, "sha256": digest})
            result = client.call("/api/assets/archive?" + query, raw=stream, length=length)
    saved_upload = json.loads(args.state.read_text())
    save(args.state, {**saved_upload, **result})
    if args.executor and kind == "sensecore":
        result = {**result, "data_delivery": deliver_data(client, args.project, result["asset_id"], args.executor,
                  args.state.with_name(args.state.name + ".delivery.json"), args.wait_seconds)}
    return result


def submission(args, client, health):
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
    result = _wait(client, health, endpoint)
    if saved:
        save(args.state, result)
    return result


def watch(args, client, health):
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
    return result
