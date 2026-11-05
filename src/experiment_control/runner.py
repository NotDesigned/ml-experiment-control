"""Command execution in an isolated process group."""

from __future__ import annotations

import os
import signal
import subprocess


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
