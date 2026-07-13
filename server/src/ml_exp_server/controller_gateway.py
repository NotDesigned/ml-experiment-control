"""Single compatibility boundary for invoking a Project controller.

``ml-expd`` owns orchestration and durable server state. Project-specific
materialization still lives behind the project's declared ``experimentctl``
command until projects expose native ``experiment_control`` adapters. All
command construction and subprocess execution is kept here so that legacy
transport does not leak through the daemon.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from experiment_control.runner import CommandRunner as CoreCommandRunner
from experiment_control.runner import SubprocessRunner

from .schemas import ResearchProject


_SECRET = re.compile(
    r"(?i)(?:^|[_-])(?:secret|token|password|credential|access[_-]?key|"
    r"api[_-]?key|proxy|authorization|cookie)(?:$|[_-])"
)


def redact(value: Any) -> Any:
    """Remove common credential shapes from controller results."""
    if isinstance(value, dict):
        return {
            str(key): ("[REDACTED]" if _SECRET.search(str(key)) else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"(?i)(https?://)[^/@\s]+@", r"\1[REDACTED]@", value)
    return value


@dataclass(frozen=True)
class ControllerCall:
    project: str
    run_id: str
    verb: str
    argv: list[str]
    cwd: Path


class CommandRunner:
    """Adapt the core command runner to the daemon's JSON controller protocol."""

    def __init__(self, runner: CoreCommandRunner | None = None) -> None:
        self.runner = runner or SubprocessRunner()

    def __call__(self, command: list[str], *, cwd: Path, timeout: int) -> dict[str, Any]:
        try:
            result = self.runner.run(
                command, cwd=cwd, check=False, timeout_seconds=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": None, "timeout": True, "stdout": "", "stderr": str(exc),
            }
        payload: Any = None
        if result.stdout.strip():
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                pass
        return {
            "returncode": result.returncode,
            "timeout": False,
            "payload": redact(payload),
            "stdout": redact(result.stdout[-2000:]),
            "stderr": redact(result.stderr[-2000:]),
        }


class ProjectControllerGateway:
    """Resolve and invoke the controller declared by one research Project."""

    def __init__(self, runner: Callable[..., dict[str, Any]] | None = None) -> None:
        self.runner = runner or CommandRunner()

    def build(
        self,
        project: ResearchProject,
        campaign: Path,
        verb: str,
        run_id: str,
        *,
        attempt_id: str | None = None,
        dry_run: bool = False,
        extra: Iterable[str] = (),
    ) -> ControllerCall:
        controller = project.controller
        if controller is None:
            raise ValueError(f"project {project.project} has no controller config")
        project_base = (project.base_dir or Path(".")).resolve()
        workdir = Path(controller.workdir)
        if not workdir.is_absolute():
            workdir = project_base / workdir
        workdir = workdir.resolve()
        tool = Path(controller.experimentctl)
        if not tool.is_absolute():
            tool = workdir / tool
        argv = [
            controller.python, str(tool), str(campaign), verb, "--run", run_id,
        ]
        if attempt_id is not None:
            argv.extend(["--attempt-id", attempt_id])
        if dry_run:
            argv.append("--dry-run")
        argv.extend(str(item) for item in extra)
        return ControllerCall(
            project=project.project,
            run_id=run_id,
            verb=verb,
            argv=argv,
            cwd=workdir,
        )

    def execute(self, call: ControllerCall, *, timeout: int) -> dict[str, Any]:
        return self.runner(call.argv, cwd=call.cwd, timeout=timeout)

    def execute_command(
        self, command: list[str], *, cwd: Path, timeout: int,
    ) -> dict[str, Any]:
        """Execute an immutable command stored in an approved action plan."""
        return self.runner(command, cwd=cwd, timeout=timeout)
