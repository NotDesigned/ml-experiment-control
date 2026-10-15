"""Short cross-process guards around Run identity creation and Action claims."""
from contextlib import contextmanager, nullcontext
from functools import wraps
import yaml

from ..deletion_state import RunFence
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
            deletion.fence.require_live(project, run)


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
            runs.add(scope["object_id"].split("::", 1)[0])
        if snapshot.get("run_id"):
            runs.add(snapshot["run_id"])
        if snapshot.get("operation") == "WRITE_CAMPAIGN":
            proposed = yaml.safe_load(snapshot["proposed_content"])
            runs.update(item["run_id"] for item in [*proposed.get("runs", []), *proposed.get("run_refs", [])])
            for item in proposed.get("runs", []):
                reference = item.get("resume_from")
                if isinstance(reference, dict) and isinstance(reference.get("run_id"), str):
                    runs.add(reference["run_id"])
        for run in runs:
            fence.require_live(scope["project"], run)
        yield
