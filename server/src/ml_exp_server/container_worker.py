"""Stdlib-only image launcher: run frozen argv, then upload this Attempt's outputs."""
from __future__ import annotations

import fnmatch
import http.client
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.parse import urlsplit


def archive_outputs(root: Path, stream, limit: int, patterns: list[str]) -> int:
    total = 0
    count = 0
    # Descriptor-relative traversal prevents project-created symlinks escaping outputs.
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for directory, _, names, fd in os.fwalk(root, follow_symlinks=False):
            for name in sorted(names):
                path = Path(directory, name).relative_to(root).as_posix()
                if any(part.startswith('.') for part in Path(path).parts):
                    continue
                if not any(fnmatch.fnmatch(path, p) or (p.startswith('**/') and fnmatch.fnmatch(path, p[3:])) for p in patterns):
                    continue
                if not stat.S_ISREG(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode):
                    continue
                opened = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                try:
                    info = os.fstat(opened)
                    if not stat.S_ISREG(info.st_mode):
                        continue
                    count += 1
                    total += info.st_size
                    if count > 20000 or total > limit:
                        raise ValueError('artifact limit exceeded')
                    member = tarfile.TarInfo(path)
                    member.size, member.mode = info.st_size, 0o400
                    with os.fdopen(os.dup(opened), 'rb') as source:
                        archive.addfile(member, source)
                finally:
                    os.close(opened)
    return count


def upload(url: str, token: str, stream, length: int):
    target = urlsplit(url)
    if target.scheme != 'https' or target.username or target.password or target.query or target.fragment:
        raise ValueError('artifact transfer requires a fixed HTTPS endpoint')
    for attempt in range(3):
        connection = http.client.HTTPSConnection(target.hostname, target.port or 443, timeout=300)
        try:
            connection.putrequest('PUT', target.path)
            connection.putheader('Authorization', 'Bearer ' + token)
            connection.putheader('Content-Type', 'application/x-tar')
            connection.putheader('Content-Length', str(length))
            connection.endheaders()
            stream.seek(0)
            while chunk := stream.read(1024 * 1024):
                connection.send(chunk)
            response = connection.getresponse()
            response.read(4096)
            if response.status == 200:
                return
            if response.status < 500:
                raise ValueError('artifact transfer was rejected')
        except (OSError, http.client.HTTPException):
            if attempt == 2:
                raise
        finally:
            connection.close()
        time.sleep(2 ** attempt)
    raise ValueError('artifact transfer failed')


def main(argv=None):
    url = os.environ.pop('ML_EXPD_UPLOAD_URL')
    token = os.environ.pop('ML_EXPD_UPLOAD_TOKEN')
    limit = int(os.environ.pop('ML_EXPD_UPLOAD_LIMIT'))
    patterns = json.loads(os.environ.pop('ML_EXPD_OUTPUT_PATTERNS', '["**/*"]'))
    root = Path(os.environ['OUTPUT_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(argv or sys.argv[1:], start_new_session=True)
    def forward(signum, _frame):
        if child.poll() is None:
            os.killpg(child.pid, signum)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    code = child.wait()
    try:
        with tempfile.TemporaryFile(dir=root.parent) as stream:
            archive_outputs(root, stream, limit - 1024 * 1024, patterns)
            length = stream.tell()
            if length > limit:
                raise ValueError('artifact archive limit exceeded')
            upload(url, token, stream, length)
        print('ML_EXPD_ARTIFACT_UPLOAD=COMPLETE', flush=True)
    except Exception:
        print('ML_EXPD_ARTIFACT_UPLOAD=FAILED', file=sys.stderr, flush=True)
        return code if code > 0 else 74
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    raise SystemExit(main())
