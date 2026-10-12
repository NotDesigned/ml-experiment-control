"""Real subprocess timeouts terminate the rsync/ssh-shaped process tree."""
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ml_exp_server import processes
from ml_exp_server.processes import SubprocessRunner, run_process


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
    assert not status.exists() or status.read_text().split()[2] == "Z"
    descriptors = Path(f"/proc/{pid}/fd")
    assert not descriptors.exists() or not list(descriptors.iterdir())


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
