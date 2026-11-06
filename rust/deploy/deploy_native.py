#!/usr/bin/env python3
"""Install one verified native release bundle on the server, or roll it back.

Without --apply this only verifies the bundle and reports what would change.
Standard library only; it runs on the host's system Python. Settings are read
from ROOT/deployment.json (see deployment.example.json). It never touches
Runs, Attempts, provider jobs or already-published Runtimes. Only
--first-install creates a registry, by running the server's own --initialize.
"""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

RELEASES = "https://github.com/NotDesigned/ml-experiment-control/releases/download"
TARGET = "x86_64-unknown-linux-gnu"
# Only these are copied, verified and restored. Any other file in bin/ is left as it is.
BINARIES = ("ml-exp", "ml-expd-rs", "ml-exp-worker-rs")
PACKAGES = ("experiment_control", "ml_exp_server", "ml_exp_worker")
SETTINGS = ("api_unit", "builder_unit", "server_config", "client_config", "cloud_config",
            "builder_state", "python", "python_site_packages", "publisher_locks")


class Refused(Exception):
    """Nothing was changed, or the previous installation was restored."""


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*argv, **kwargs):
    return subprocess.run(argv, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=300, **kwargs).stdout.strip()


def service_account(settings):
    """Root must not own the stores: run the server the way its unit does."""
    if os.geteuid() != 0:
        return {"cwd": "/"}
    name = command("systemctl", "show", settings["api_unit"], "-p", "User", "--value")
    if not name:
        raise Refused(f"{settings['api_unit']} names no service account; install the unit first")
    account = pwd.getpwnam(name)

    def become():
        os.setgroups([])
        os.setgid(account.pw_gid)
        os.setuid(account.pw_uid)
        os.umask(0o077)
    return {"cwd": "/", "preexec_fn": become}


def version_of(path):
    name, _, version = command(str(path), "--version").partition(" ")
    if name != Path(path).name:
        raise Refused(f"{path} reports another program: {name}")
    return version


def parsed(version):
    return tuple(int(part) for part in version.split("."))


def replace_file(source, destination, mode):
    temporary = destination.with_name(f".{destination.name}.deploy")
    shutil.copyfile(source, temporary)
    temporary.chmod(mode)
    os.replace(temporary, destination)


def download(url, destination):
    with urllib.request.urlopen(url, timeout=120) as response, open(destination, "wb") as stream:
        shutil.copyfileobj(response, stream)


def fetch_bundle(version, directory):
    base = f"{RELEASES}/native-v{version}"
    name = f"ml-expd-native-{version}-{TARGET}.tar.gz"
    download(f"{base}/SHA256SUMS", directory / "SHA256SUMS")
    expected = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        value, _, asset = line.partition("  ")
        expected[asset] = value
    if name not in expected:
        raise Refused("release checksums do not list the server bundle")
    download(f"{base}/{name}", directory / name)
    return directory / name, expected[name]


def extract(bundle, destination):
    """Unpack regular files only; a bundle cannot escape or plant links."""
    with tarfile.open(bundle, "r:gz") as tar:
        for member in tar.getmembers():
            target = destination.joinpath(member.name).resolve()
            if not member.isfile() or destination.resolve() not in target.parents:
                raise Refused(f"unsafe bundle member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, open(target, "wb") as stream:
                shutil.copyfileobj(source, stream)
            target.chmod(0o755 if member.mode & 0o100 else 0o644)


def verify_bundle(root):
    """Return (version, source commit) after checking every hash in the bundle."""
    try:
        manifest = json.loads((root / "native-release.json").read_text())
        info = json.loads((root / "release-info.json").read_text())
    except (OSError, ValueError) as error:
        raise Refused("bundle lacks server deployment metadata") from error
    version = manifest["version"]
    if info["version"] != version:
        raise Refused("bundle metadata versions differ")
    assets = {a["executable"]: a["sha256"] for a in manifest["assets"] if a["target"] == TARGET}
    for name in BINARIES:
        path = root / "bin" / name
        if sha256(path) != assets.get(name) or version_of(path) != version:
            raise Refused(f"bundled {name} does not match its release manifest")
    python = root / "python"
    found = {str(p.relative_to(python)) for p in python.rglob("*") if p.is_file()}
    if found != set(info["python"]):
        raise Refused("bundled Python files differ from release metadata")
    for name, value in info["python"].items():
        if sha256(python / name) != value:
            raise Refused(f"bundled Python file changed: {name}")
    return version, info["source_commit"]


def active_builds(state):
    """A builder holds <identity>.lock exclusively for the whole of a build."""
    active = []
    for path in sorted(Path(state).glob("*.lock")):
        with open(path, "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                active.append(path.name)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)
    return active


def health(settings):
    client = json.loads(Path(settings["client_config"]).read_text())
    token = Path(client["token_file"]).read_text().strip()
    request = urllib.request.Request(client["url"] + "/api/health", headers={
        "Authorization": "Bearer " + token})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def wait_healthy(settings, version, timeout):
    deadline = time.monotonic() + timeout
    last = "no response"
    while time.monotonic() < deadline:
        try:
            value = health(settings)
            if value.get("status") == "ok" and value.get("server_version") == version:
                return
            last = f"status={value.get('status')} version={value.get('server_version')}"
        except (OSError, ValueError, urllib.error.URLError) as error:
            last = type(error).__name__
        time.sleep(1)
    raise Refused(f"API did not report healthy {version}: {last}")


@contextlib.contextmanager
def publishers_stopped(settings):
    """Stop both services and prove the API released its publisher locks.

    A failure inside the block leaves the services running again; the block
    itself restores any file it had already replaced.
    """
    locks = []
    try:
        command("systemctl", "stop", settings["api_unit"])
        status = command("systemctl", "show", settings["api_unit"], "-p", "ExecMainStatus", "--value")
        if status != "0":
            raise Refused(f"API exited with status {status}; investigate before installing")
        command("systemctl", "stop", settings["builder_unit"])
        for path in settings["publisher_locks"]:
            try:
                descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            except FileNotFoundError:
                continue  # No publisher has run here, so none can still hold it.
            locks.append(descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    except BaseException:
        start(settings)
        raise
    finally:
        for descriptor in locks:
            os.close(descriptor)


def start(settings):
    command("systemctl", "start", settings["builder_unit"])
    command("systemctl", "start", settings["api_unit"])


def install_tree(source, destination):
    """Overwrite this release's files; other installed package data remains."""
    destination.mkdir(parents=True, exist_ok=True)
    wanted = set()
    for path in sorted(source.rglob("*")):
        if path.is_file():
            relative = path.relative_to(source)
            wanted.add(relative)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            replace_file(path, target, 0o644)
    for path in destination.rglob("*.py"):
        if path.relative_to(destination) not in wanted:
            path.unlink()


def install(root, settings, source):
    for name in BINARIES:
        replace_file(source / "bin" / name, root / "bin" / name, 0o755)
    site = Path(settings["python_site_packages"])
    for package in PACKAGES:
        install_tree(source / "python" / package, site / package)


def replace_config(temporary, path):
    """The service account refuses a private file it does not own, so keep the owner."""
    before = path.stat()
    os.chown(temporary, before.st_uid, before.st_gid)
    temporary.chmod(before.st_mode & 0o777)
    os.replace(temporary, path)


def pin_worker(settings, digest):
    """New builds embed this worker; existing Runtimes keep their frozen hash."""
    path = Path(settings["cloud_config"])
    config = json.loads(path.read_text())
    if config.get("builds", {}).get("worker_sha256") in (None, digest):
        return
    config["builds"]["worker_sha256"] = digest
    temporary = path.with_name(f".{path.name}.deploy")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    replace_config(temporary, path)


def save_previous(root, settings):
    previous = root / "previous"
    temporary = root / ".previous.deploy"
    shutil.rmtree(temporary, ignore_errors=True)
    (temporary / "bin").mkdir(parents=True)
    for name in BINARIES:
        shutil.copy2(root / "bin" / name, temporary / "bin" / name)
    site = Path(settings["python_site_packages"])
    ignore = shutil.ignore_patterns("__pycache__")
    for package in PACKAGES:
        shutil.copytree(site / package, temporary / "python" / package, ignore=ignore)
    shutil.copy2(settings["cloud_config"], temporary / "cloud_config.json")
    shutil.rmtree(previous, ignore_errors=True)
    os.replace(temporary, previous)


def restore_previous(root, settings):
    previous = root / "previous"
    install(root, settings, previous)
    path = Path(settings["cloud_config"])
    temporary = path.with_name(f".{path.name}.deploy")
    shutil.copyfile(previous / "cloud_config.json", temporary)
    replace_config(temporary, path)
    return version_of(root / "bin/ml-expd-rs")


def record(root, version, commit, action):
    value = {"version": version, "source_commit": commit, "action": action,
             "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "sha256": {n: sha256(root / "bin" / n) for n in BINARIES}}
    temporary = root / ".DEPLOYED.json.deploy"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, root / "DEPLOYED.json")
    return value


def rollback(root, settings, timeout):
    if not (root / "previous/bin/ml-expd-rs").is_file():
        raise Refused("no previous installation is retained")
    with publishers_stopped(settings):
        version = restore_previous(root, settings)
    start(settings)
    wait_healthy(settings, version, timeout)
    return record(root, version, None, "rollback")


def verified(args, work):
    """Unpack a bundle and check every hash; returns (unpacked, version, commit)."""
    if args.bundle:
        bundle, expected = args.bundle, args.sha256
    else:
        bundle, expected = fetch_bundle(args.version, work)
    if sha256(bundle) != expected:
        raise Refused("bundle checksum differs")
    unpacked = work / "bundle"
    extract(bundle, unpacked)
    version, commit = verify_bundle(unpacked)
    if args.version and version != args.version:
        raise Refused(f"bundle contains {version}, not {args.version}")
    return unpacked, version, commit


def preflight(settings, unpacked, work):
    """Run the new programs against the live configuration; returns the worker's hash."""
    for directory in (work, unpacked, unpacked / "bin"):
        directory.chmod(0o755)  # The service account runs the unpacked server.
    command(str(unpacked / "bin/ml-expd-rs"), "--config", settings["server_config"], "--check",
            **service_account(settings))
    command(settings["python"], "-c", "import ml_exp_server.image_builder",
            cwd=str(work), env={**os.environ, "PYTHONPATH": str(unpacked / "python")})
    return "sha256:" + sha256(unpacked / "bin/ml-exp-worker-rs")


def first_install(root, settings, args):
    """Install into a ROOT that has never held a release; there is nothing to roll back to."""
    for name in ("bin/ml-expd-rs", "previous", "DEPLOYED.json"):
        if (root / name).exists():
            raise Refused(f"{root / name} exists; --first-install needs a fresh ROOT")
    with tempfile.TemporaryDirectory(dir=root, prefix=".deploy-") as temporary:
        work = Path(temporary)
        unpacked, version, commit = verified(args, work)
        worker = preflight(settings, unpacked, work)
        json.loads(Path(settings["cloud_config"]).read_text())
        plan = {"current": None, "version": version, "source_commit": commit, "worker_sha256": worker}
        if not args.apply:
            return {**plan, "applied": False}
        # Initialize first: a refusal such as an occupied Run root leaves ROOT
        # empty, so correcting the cause and repeating this command is enough.
        command(str(unpacked / "bin/ml-expd-rs"), "--config", settings["server_config"], "--initialize",
                **service_account(settings))
        (root / "bin").mkdir(exist_ok=True)
        (root / "bin").chmod(0o755)
        install(root, settings, unpacked)
        pin_worker(settings, worker)
    try:
        start(settings)
        wait_healthy(settings, version, args.health_timeout)
    except (Refused, subprocess.SubprocessError) as error:
        for unit in (settings["api_unit"], settings["builder_unit"]):
            with contextlib.suppress(subprocess.SubprocessError):
                command("systemctl", "stop", unit)
        raise Refused(f"{error}; {version} stays installed and its registry initialized, the units "
                      "are stopped. Correct the cause, start both units, then repeat with --apply "
                      "and without --first-install to record it") from error
    return {**plan, **record(root, version, commit, "first install"), "applied": True}


def deploy(root, settings, args):
    with tempfile.TemporaryDirectory(dir=root, prefix=".deploy-") as temporary:
        work = Path(temporary)
        unpacked, version, commit = verified(args, work)
        if not (root / "bin/ml-expd-rs").exists():
            raise Refused(f"{root} holds no installed server; a fresh host uses --first-install")
        current = version_of(root / "bin/ml-expd-rs")
        if parsed(version) < parsed(current):
            raise Refused(f"refusing downgrade from {current}; use --rollback")
        worker = preflight(settings, unpacked, work)
        active = active_builds(settings["builder_state"])
        if active:
            raise Refused(f"builds are active: {', '.join(active)}")
        plan = {"current": current, "version": version, "source_commit": commit,
                "worker_sha256": worker}
        if not args.apply:
            return {**plan, "applied": False}
        save_previous(root, settings)
        with publishers_stopped(settings):
            try:
                install(root, settings, unpacked)
                pin_worker(settings, plan["worker_sha256"])
            except OSError:
                restore_previous(root, settings)
                raise
        try:
            start(settings)
            wait_healthy(settings, version, args.health_timeout)
        except (Refused, subprocess.SubprocessError) as error:
            with contextlib.suppress(subprocess.SubprocessError):
                command("systemctl", "stop", settings["api_unit"])
            with contextlib.suppress(subprocess.SubprocessError):
                command("systemctl", "stop", settings["builder_unit"])
            restored = restore_previous(root, settings)
            start(settings)
            wait_healthy(settings, restored, args.health_timeout)
            record(root, restored, None, "automatic rollback")
            raise Refused(f"{error}; restored {restored}") from error
        return {**plan, **record(root, version, commit, "deploy"), "applied": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--version", help="download this published release")
    parser.add_argument("--bundle", type=Path, help="local bundle; requires --sha256")
    parser.add_argument("--sha256")
    parser.add_argument("--apply", action="store_true", help="install instead of only verifying")
    parser.add_argument("--rollback", action="store_true", help="restore the retained previous install")
    parser.add_argument("--first-install", action="store_true",
                        help="install into a fresh ROOT and initialize a new registry")
    parser.add_argument("--health-timeout", type=int, default=120)
    args = parser.parse_args()
    if args.rollback == bool(args.version or args.bundle) or bool(args.bundle) != bool(args.sha256):
        parser.error("use --version, --bundle with --sha256, or --rollback")
    if args.rollback and args.first_install:
        parser.error("--first-install installs a release; it has nothing to roll back")
    root = args.root.resolve()
    try:
        settings = json.loads((root / "deployment.json").read_text())
        missing = [key for key in SETTINGS if key not in settings]
        if missing:
            raise Refused(f"deployment.json lacks {', '.join(missing)}")
        with open(root / ".deploy.lock", "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise Refused("another deployment is running") from None
            if args.rollback:
                result = rollback(root, settings, args.health_timeout)
            elif args.first_install:
                result = first_install(root, settings, args)
            else:
                result = deploy(root, settings, args)
    except (Refused, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        output = getattr(error, "stdout", None)
        print(f"deployment refused: {error}" + (f"\n{output}" if output else ""), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
