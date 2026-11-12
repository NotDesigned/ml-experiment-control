"""Official W&B SDK effects only. Rust owns routing, records and decisions.

Private bounded frames use stdin/stdout; credentials never enter argv/logs.
No imports from the Python server or user experiment are needed.
"""
import contextlib
import json
import struct
import sys
import tempfile

BOUND = 8 * 1024 * 1024
# Each open W&B Run is a stream inside wandb-core on a small shared host.
SESSION_LIMIT = 16
sessions = {}
# The next history step W&B holds for a resumed Run. After a resume the SDK's
# own `step` can trail it; W&B then drops every row logged below it, so a
# publisher starting there never advances.
floors = {}


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


def close(identity):
    """Flush and end one session. It is forgotten even if the SDK fails."""
    handle, directory = sessions.pop(identity)
    floors.pop(identity, None)
    try:
        handle.finish(exit_code=0)
    finally:
        directory.cleanup()


def session(sdk, api, job):
    identity = job["wandb_id"]
    if identity in sessions:
        sessions[identity] = sessions.pop(identity)
        return sessions[identity][0]
    if len(sessions) >= SESSION_LIMIT:
        # Finishing flushes, so the least recently used Run only resumes later.
        with contextlib.suppress(Exception):
            close(next(iter(sessions)))
    route = job["route"]
    scope = job["scope"]
    floor = 0
    if job["resume"] == "must":
        # One read per opened session: a Run this scope created before and a
        # person deleted is never recreated under the same name.
        remote = request('REMOTE_READ', lambda: api.run(
            f'{route["entity"]}/{route["project"]}/{identity}'))
        if (remote.id, remote.entity, remote.project) != (
                identity, route["entity"], route["project"]):
            raise ValueError("remote identity differs")
        last = getattr(remote, "lastHistoryStep", None)
        if isinstance(last, int) and last >= 0:
            floor = last + 1
    directory = tempfile.TemporaryDirectory(prefix="ml-exp-wandb-")
    try:
        handle = request('INIT', lambda: sdk.init(
            entity=route["entity"], project=route["project"], id=identity,
            name=scope["run"] + "/" + scope["attempt"],
            group=scope["project"] + "/" + scope["run"],
            job_type="attempt", resume=job["resume"], reinit="create_new",
            config=job["config"], settings=sdk.Settings(
                api_key=job["api_key"], base_url="https://api.wandb.ai",
                root_dir=directory.name, disable_git=True, disable_code=True,
                x_disable_stats=True, x_disable_meta=True,
                x_disable_machine_info=True, console="off", quiet=True,
                init_timeout=30, x_file_stream_transmit_interval=5)))
        if (handle.id, handle.entity, handle.project) != (
                identity, route["entity"], route["project"]):
            raise ValueError("SDK identity differs")
    except Exception:
        directory.cleanup()
        raise
    sessions[identity] = (handle, directory)
    floors[identity] = floor
    return handle


def dispatch(job):
    operation = job["operation"]
    if operation == "shutdown":
        # One failure must not cost the other Runs their flush.
        for identity in list(sessions):
            with contextlib.suppress(Exception):
                close(identity)
        return {"shutdown": True}
    if operation == "close":
        if job["wandb_id"] in sessions:
            request('WRITE', lambda: close(job["wandb_id"]))
        return {"closed": True}
    import wandb as sdk
    api = sdk.Api(api_key=job["api_key"],
                  overrides={"base_url": "https://api.wandb.ai"}, timeout=15)
    if operation == "default_entity":
        return {"entity": request('DEFAULT_ENTITY', lambda: api.default_entity)}
    if operation == "apply":
        handle = session(sdk, api, job)
        for definition in job["definitions"]:
            handle.define_metric(definition["name"], **definition["options"])
        if job.get("display") is not None:
            patch = job["display"]
            for key in patch["delete"]:
                if key in handle.summary.keys():
                    del handle.summary[key]
            for key, value in patch["set"].items():
                handle.summary[key] = value
        step = start = max(handle.step, floors.get(job["wandb_id"], 0))
        for record in job["history"]:
            request('WRITE', lambda: handle.log(record, step=step, commit=True))
            step += 1
        # The SDK's own acknowledgement: wandb-core has taken every row above.
        # It is not a read of W&B, which may still miss or repeat a row.
        if job["history"] and handle.step != step:
            raise RequestFailed('WANDB_WRITE_FAILED')
        return {"applied": True, "rows": step - start}
    if operation == "finish":
        handle = session(sdk, api, job)
        # Forget it first: a failed finish must not leave a dead handle that
        # every later request for this Run would be sent to.
        _, directory = sessions.pop(job["wandb_id"])
        floors.pop(job["wandb_id"], None)
        try:
            request('WRITE', lambda: handle.finish(exit_code=job["exit_code"]))
        finally:
            directory.cleanup()
        return {"finished": True}
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
                stage = 'WRITE' if job.get('operation') in ('apply', 'finish') else 'REMOTE_READ'
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
