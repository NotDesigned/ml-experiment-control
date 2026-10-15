"""Injectable command execution used by scheduler adapters."""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class CommandResult:
    """Portable, immutable result returned by a command runner."""

    args: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    def check_returncode(self) -> None:
        """Raise the same error callers expect from ``subprocess.run``."""
        if self.returncode:
            raise subprocess.CalledProcessError(
                self.returncode, list(self.args), output=self.stdout, stderr=self.stderr
            )


class CommandRunner(Protocol):
    """Boundary for external commands and hermetic test fakes."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
        input_text: str | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandResult: ...


def run_process(command, *, timeout=None, check=False, capture_output=True, input=None, **options):
    """Run a command in an isolated group; failures terminate its descendants.

    SSH and transfer commands can spawn children that retain files or pipes.
    A timeout or caller interruption therefore kills the entire process group,
    then reaps the direct child without waiting for descendant-owned pipe EOF.
    """
    with subprocess.Popen(command, stdin=subprocess.PIPE if input is not None else None,
                          stdout=subprocess.PIPE if capture_output else None,
                          stderr=subprocess.PIPE if capture_output else None,
                          start_new_session=True, **options) as process:
        try:
            stdout, stderr = process.communicate(input=input, timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        if check:
            result.check_returncode()
        return result


class SubprocessRunner:
    """Production runner with shell expansion deliberately disabled."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
        input_text: str | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        arguments = tuple(str(arg) for arg in command)
        try:
            completed = run_process(
                list(command), cwd=cwd, check=False, input=input_text,
                text=True, encoding="utf-8", errors="replace",
                capture_output=True, timeout=timeout_seconds,
            )
            result = CommandResult(
                args=arguments, returncode=completed.returncode,
                stdout=completed.stdout, stderr=completed.stderr,
            )
        except OSError as error:
            result = CommandResult(
                args=arguments, returncode=127,
                stderr=f"command execution failed: {type(error).__name__}",
            )
        if check:
            result.check_returncode()
        return result
