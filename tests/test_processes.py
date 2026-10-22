"""Real subprocess timeouts terminate the rsync/ssh-shaped process tree."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from experiment_control import runner as processes
from experiment_control.runner import SubprocessRunner, run_process


def test_runner_input_check_and_safe_launch_failure(tmp_path):
    runner = SubprocessRunner()
    result = runner.run([sys.executable, "-c", "import sys; print(sys.stdin.read()); print('err',file=sys.stderr)"],
                        input_text="hello", cwd=tmp_path)
    assert result.returncode == 0 and result.stdout == "hello\n" and result.stderr == "err\n"
    with pytest.raises(subprocess.CalledProcessError): runner.run([sys.executable, "-c", "exit(3)"])
    result = runner.run([str(tmp_path / "missing-secret-token")], check=False)
    assert result.returncode == 127 and "secret-token" not in result.stderr
    with pytest.raises(ValueError): runner.run([sys.executable], timeout_seconds=0)
    with pytest.raises(subprocess.CalledProcessError): run_process([sys.executable, "-c", "exit(2)"], check=True, timeout=5)


def test_timeout_kills_grandchild_that_ignores_term_and_closes_its_descriptors(tmp_path):
    identity = tmp_path / "child.pid"
    child = "import os,signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(" + repr(str(identity)) + ").write_text(str(os.getpid())); time.sleep(60)"
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); time.sleep(60)"
    with pytest.raises(subprocess.TimeoutExpired):
        run_process([sys.executable, "-c", parent], timeout=1)
    pid = int(identity.read_text())
    # A reparented zombie may await init's reaper; it cannot retain write FDs.
    status = Path(f"/proc/{pid}/stat")
    def alive():
        try:
            return status.read_text().split()[2] != "Z"
        except (FileNotFoundError, ProcessLookupError):
            return False
    # SIGKILL delivery and the parent's wait are separate kernel events. The
    # grandchild can briefly be runnable before its pending kill is processed.
    deadline = time.monotonic() + 2
    while alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not alive()
    descriptors = Path(f"/proc/{pid}/fd")
    try:
        retained = list(descriptors.iterdir())
    except (FileNotFoundError, ProcessLookupError):
        retained = []
    except PermissionError:
        # Reparented zombies may deny fd enumeration to an unprivileged CI
        # user. Require positive exit evidence; a live process must still fail.
        assert not alive()
        retained = []
    assert not retained


def test_timeout_with_already_gone_process_group_preserves_original_timeout(monkeypatch):
    class Gone:
        pid = 1
        calls = 0
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def wait(self): return 0
        def communicate(self, **kwargs):
            self.calls += 1
            if self.calls == 1: raise subprocess.TimeoutExpired("child", 1)
            return b"", b""
    monkeypatch.setattr(processes.subprocess, "Popen", lambda *a, **kw: Gone())
    monkeypatch.setattr(processes.os, "killpg", lambda *a: (_ for _ in ()).throw(ProcessLookupError()))
    with pytest.raises(subprocess.TimeoutExpired): run_process(["child"], timeout=1)


def test_process_capture_can_be_disabled_without_pipe_or_result_confusion():
    result = run_process([sys.executable, "-c", "pass"], timeout=5, capture_output=False)
    assert result.returncode == 0
    assert result.stdout is None and result.stderr is None


def test_runner_normalizes_argument_types_in_returned_identity(tmp_path):
    result = SubprocessRunner().run([Path(sys.executable), "-c", "pass"])
    assert result.args == (sys.executable, "-c", "pass")
