"""Short cross-process guards around Run identity creation and Action claims."""
from contextlib import contextmanager, nullcontext
from functools import wraps
import yaml

from ..deletion_state import RunFence
from ..application_errors import ApplicationError
from ..schemas import OperationScopeType


def request_run(method):
    @wraps(method)
    def wrapped(self, project, request, *args, **kwargs):
        run = getattr(request, "run", request).run_id
        with RunFence(self.runtime.config.project_registry_root_path()).guard(project, run):
            return method(self, project, request, *args, **kwargs)
    return wrapped


def require_action_runs(service, project, runs):
    deletion = getattr(service, "deletion_service", None)
    if deletion is not None:
        for run in runs:
            require_live(deletion.fence, project, run)


def require_live(fence, project, run):
    try:
        fence.require_live(project, run)
    except ValueError:
        raise ApplicationError("invalid Run identity", status_code=422, code="INVALID_RUN_IDENTITY") from None


def action_guard(service, snapshot):
    deletion = getattr(service, "deletion_service", None)
    scope = snapshot.get("scope", {})
    if deletion is None or snapshot.get("operation") == "DELETE_RUN_RESOURCES":
        return nullcontext()
    return _action_guard(deletion.fence, scope, snapshot)


@contextmanager
def _action_guard(fence, scope, snapshot):
    # Campaign writes may introduce Run identities too. Their preparation and
    # claims must use the same lock order as Run-scoped Actions.
    with fence.shared():
        runs = set()
        if scope.get("scope_type") in {OperationScopeType.RUN.value, OperationScopeType.ATTEMPT.value}:
            run = scope["object_id"]
            if scope["scope_type"] == OperationScopeType.ATTEMPT.value:
                run = run.rsplit("::", 1)[0]
            runs.add(run)
        if snapshot.get("run_id"):
            runs.add(snapshot["run_id"])
        for run in runs:
            fence.require_observed_live(scope["project"], run)
        if snapshot.get("operation") == "WRITE_CAMPAIGN":
            proposed = yaml.safe_load(snapshot["proposed_content"])
            for item in [*proposed.get("runs", []), *proposed.get("run_refs", [])]:
                require_live(fence, scope["project"], item.get("run_id") if isinstance(item, dict) else None)
            for item in proposed.get("runs", []):
                reference = item.get("resume_from") if isinstance(item, dict) else None
                if isinstance(reference, dict) and "run_id" in reference:
                    require_live(fence, scope["project"], reference["run_id"])
        yield
