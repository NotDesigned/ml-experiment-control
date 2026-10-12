"""Bounded server subprocesses: timeout also terminates transfer descendants."""
import os
from pathlib import Path
import signal
import subprocess
from typing import Sequence

from experiment_control.runner import CommandResult


def run_process(command, *, timeout, check=False, capture_output=True, input=None, **options):
    with subprocess.Popen(command, stdin=subprocess.PIPE if input is not None else None,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          start_new_session=True, **options) as process:
        try:
            stdout, stderr = process.communicate(input=input, timeout=timeout)
        except BaseException:
            # The controller, rsync and ssh share this group. Terminating only
            # the direct child can leave descendants writing large temp files.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            # Do not wait for EOF from descendant-owned pipes after timeout.
            # Reap the direct child; Popen's context closes our pipe ends.
            process.wait()
            raise
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        if check:
            result.check_returncode()
        return result


class SubprocessRunner:
    def run(self, command: Sequence[str], *, cwd: Path | None = None, check: bool = True,
            input_text: str | None = None, timeout_seconds: float | None = None) -> CommandResult:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        try:
            completed = run_process(list(command), cwd=cwd, check=False, input=input_text,
                                    text=True, encoding="utf-8", errors="replace", timeout=timeout_seconds)
            result = CommandResult(tuple(command), completed.returncode, completed.stdout, completed.stderr)
        except OSError as error:
            result = CommandResult(tuple(command), 127, stderr=f"command execution failed: {type(error).__name__}")
        if check:
            result.check_returncode()
        return result
