"""Official W&B SDK effects only. Rust owns routing, records and decisions.

Private bounded frames use stdin/stdout; credentials never enter argv/logs.
No imports from the Python server or user experiment are needed.
"""
import contextlib
import itertools
import json
import struct
import sys
import tempfile

BOUND = 8 * 1024 * 1024
sessions = {}


class RequestFailed(Exception):
    def __init__(self, code):
        self.code = code


def request(stage, operation):
    try:
        return operation()
    except (RequestFailed, ImportError):
        raise
    except Exception as exc:
        # Never transport provider messages, URLs, credentials or tracebacks.
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        if status == 401 or type(exc).__name__ == 'AuthenticationError':
            code = 'WANDB_AUTHENTICATION_FAILED'
        elif status == 403 or type(exc).__name__ == 'PermissionError':
            code = 'WANDB_PERMISSION_DENIED'
        elif stage == 'REMOTE_READ' and type(exc).__name__ == 'CommError' and 'Could not find run' in str(exc):
            code = 'WANDB_REMOTE_RUN_MISSING'
        else:
            code = 'WANDB_' + stage + '_FAILED'
        raise RequestFailed(code) from None


def exact(stream, length):
    chunks = bytearray()
    while len(chunks) < length:
        chunk = stream.read(length - len(chunks))
        if not chunk:
            raise EOFError()
        chunks.extend(chunk)
    return bytes(chunks)


def session(sdk, job):
    identity = job["wandb_id"]
    if identity in sessions:
        return sessions[identity][0]
    if len(sessions) >= 64:
        raise ValueError("session limit")
    directory = tempfile.TemporaryDirectory(prefix="ml-exp-wandb-")
    route = job["route"]
    scope = job["scope"]
    try:
        handle = request('INIT', lambda: sdk.init(
            entity=route["entity"], project=route["project"], id=identity,
            name=scope["run"] + "/" + scope["attempt"],
            group=scope["project"] + "/" + scope["run"],
            job_type="attempt", resume=job["resume"], reinit="create_new",
            config={**job["config"], **({"ml_expd_identity_json": job["config_identity"]}
                    if "config_identity" in job else {})}, settings=sdk.Settings(
                api_key=job["api_key"], base_url="https://api.wandb.ai",
                root_dir=directory.name, disable_git=True, disable_code=True,
                x_disable_stats=True, x_disable_meta=True,
                x_disable_machine_info=True, console="off", quiet=True,
                init_timeout=30)))
        if (handle.id, handle.entity, handle.project) != (
                identity, route["entity"], route["project"]):
            raise ValueError("SDK identity differs")
    except Exception:
        directory.cleanup()
        raise
    sessions[identity] = (handle, directory)
    return handle


def dispatch(job):
    import wandb as sdk
    api = sdk.Api(api_key=job["api_key"],
                  overrides={"base_url": "https://api.wandb.ai"}, timeout=15)
    operation = job["operation"]
    if operation == "default_entity":
        return {"entity": request('DEFAULT_ENTITY', lambda: api.default_entity)}
    if operation == "observe":
        route, identity = job["route"], job["wandb_id"]
        def observe_remote():
            return request('REMOTE_READ', lambda: api.run(f'{route["entity"]}/{route["project"]}/{identity}'))
        try:
            remote = observe_remote()
        except RequestFailed as exc:
            if exc.code != 'WANDB_REMOTE_RUN_MISSING' or not job['open_session'] or job['resume'] != 'allow':
                raise
            # Atomic create-if-absent for a new unreserved scope. Existing or
            # uncertain scopes are read first and never recreated implicitly.
            session(sdk, {**job, 'resume': 'never'})
            remote = observe_remote()
        if (remote.id, remote.entity, remote.project) != (
                identity, route["entity"], route["project"]):
            raise ValueError("remote identity differs")
        history = []
        rows = request('HISTORY_READ', lambda: list(itertools.islice(remote.scan_history(
                keys=["ml_expd/sequence", "ml_expd/event_id"],
                min_step=max(0, job["confirmed"] - 1),
                max_step=job["through"], page_size=128, use_cache=False), 258)))
        for row in rows:
            if len(history) >= 257:
                raise ValueError("history bound")
            sequence = row["ml_expd/sequence"]
            if type(sequence) is not int:
                raise ValueError("history sequence")
            history.append({"sequence": sequence,
                            "event_id": row["ml_expd/event_id"]})
        summary_keys = list(remote.summary.keys())
        if len(summary_keys) > 8192:
            raise ValueError("summary bound")
        return {"wandb_id": identity, "route": route, "state": remote.state,
                "config": remote.config.get("ml_expd"),
                "config_identity": remote.config.get("ml_expd_identity_json"),
                "summary_keys": summary_keys, "evidence": {
                    "history": history,
                    "last_history_step": remote.lastHistoryStep,
                    "display_version": remote.summary.get("ml_expd/display_version"),
                    "display_through": remote.summary.get("ml_expd/display_through")}}
    if operation == "apply":
        handle = session(sdk, job)
        for definition in job["definitions"]:
            handle.define_metric(definition["name"], **definition["options"])
        for effect in job["effects"]:
            if effect["operation"] == "history":
                request('WRITE', lambda: handle.log(effect["record"], step=effect["sequence"] - 1,
                           commit=True))
            elif effect["operation"] == "display":
                for key in effect["patch"]["delete"]:
                    if key in handle.summary:
                        del handle.summary[key]
                for key, value in effect["patch"]["set"].items():
                    handle.summary[key] = value
            elif effect["operation"] == "finish":
                handle.finish(exit_code=effect["exit_code"])
                _, directory = sessions.pop(job["wandb_id"])
                directory.cleanup()
            else:
                raise ValueError("unknown SDK operation")
        # This is only a transport receipt, never a remote history ACK.
        return {"applied": True}
    raise ValueError("unknown SDK request")


def main():
    incoming, outgoing = sys.stdin.buffer, sys.stdout.buffer
    while True:
        try:
            header = exact(incoming, 4)
        except EOFError:
            break
        length, = struct.unpack(">I", header)
        if not 0 < length <= BOUND:
            raise ValueError("frame bound")
        raw = exact(incoming, length)
        try:
            job = json.loads(raw)
            with contextlib.redirect_stdout(sys.stderr):
                stage = 'WRITE' if job.get('operation') == 'apply' else 'REMOTE_READ'
                result = request(stage, lambda: dispatch(job))
        except ImportError:
            result = {"error": "WANDB_SDK_NOT_INSTALLED"}
        except RequestFailed as exc:
            result = {"error": exc.code}
        except Exception:
            result = {"error": "WANDB_REQUEST_FAILED"}
        encoded = json.dumps(result, allow_nan=False, separators=(",", ":")).encode()
        if not 0 < len(encoded) <= BOUND:
            raise ValueError("response bound")
        outgoing.write(struct.pack(">I", len(encoded)) + encoded)
        outgoing.flush()


if __name__ == "__main__":
    main()
