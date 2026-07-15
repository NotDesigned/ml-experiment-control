"""Injectable command execution used by scheduler adapters."""

from __future__ import annotations

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
            completed = subprocess.run(
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
