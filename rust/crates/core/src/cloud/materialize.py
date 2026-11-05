"""Compatibility launcher for immutable worker17 images (CPU data preparation).

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
    downloaded = False
    try:
        with tempfile.TemporaryDirectory(prefix="ml-expd-data-") as scratch:
            path = os.path.join(scratch, "import.json")
            with open(path, "x", encoding="utf-8") as stream:
                os.chmod(path, 0o600)
                stream.write(raw)
            subprocess.run([worker, "data-import", "--config", path, "--bundle-output", archive], check=True)
            downloaded = True
            config["source"] = {"kind": "file", "path": archive}
            command = [worker, "data-copy", "--config-json", json.dumps(config, separators=(",", ":")), "--channel-json", channel]
            if storage != "null":
                command += ["--storage-json", storage]
            subprocess.run(command, check=True)
    finally:
        if downloaded:
            os.unlink(archive)


if __name__ == "__main__":
    main()
