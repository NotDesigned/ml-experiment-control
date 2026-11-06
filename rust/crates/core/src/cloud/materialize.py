"""CPU data preparation inside an existing Runtime image.

The worker owns downloads, archive validation, extraction and the READY callback.
Only its temporary JSON configuration and verified local archive live here.
"""
import json
import os
import stat
import subprocess
import sys
import tempfile


def main():
    worker, raw, channel, storage = sys.argv[1:]
    config = json.loads(raw)
    directory = "/ml-expd/data"
    archive = directory + "/archive.tar"
    os.makedirs(directory, mode=0o700, exist_ok=True)
    info = os.lstat(directory)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise RuntimeError("DATA_STAGING_DIRECTORY_UNAVAILABLE")
    if os.path.lexists(archive):
        raise RuntimeError("DATA_STAGING_ARCHIVE_EXISTS")
    try:
        with tempfile.TemporaryDirectory(prefix="ml-expd-data-") as scratch:
            path = os.path.join(scratch, "import.json")
            with open(path, "x", encoding="utf-8") as stream:
                os.chmod(path, 0o600)
                stream.write(raw)
            try:
                subprocess.run([worker, "data-import", "--config", path, "--bundle-output", archive], check=True)
            except (OSError, subprocess.CalledProcessError):
                raise RuntimeError("DATA_IMPORT_FAILED") from None
            config["source"] = {"kind": "file", "path": archive}
            command = [worker, "data-copy", "--config-json", json.dumps(config, separators=(",", ":")), "--channel-json", channel, "--storage-json", storage]
            try:
                subprocess.run(command, check=True)
            except (OSError, subprocess.CalledProcessError):
                raise RuntimeError("DATA_COPY_FAILED") from None
    finally:
        # A failed import may also have left a partial archive. The preflight
        # above established that this invocation owns the staging path.
        if os.path.lexists(archive):
            os.unlink(archive)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        code = str(error)
        if code not in {"DATA_IMPORT_FAILED", "DATA_COPY_FAILED", "DATA_STAGING_DIRECTORY_UNAVAILABLE", "DATA_STAGING_ARCHIVE_EXISTS"}:
            code = "DATA_MATERIALIZATION_FAILED"
        print("native data materialization failed: " + code, file=sys.stderr)
        sys.exit(1)
