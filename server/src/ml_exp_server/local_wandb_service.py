"""Foreground Docker wrapper for a daemon-owned local W&B service."""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import subprocess
import sys
from pathlib import Path


def container_name(data_dir: Path, port: int) -> str:
    digest = hashlib.sha256(str(data_dir.resolve()).encode()).hexdigest()[:10]
    return f"ml-expd-wandb-{port}-{digest}"


def docker_command(
    docker: str, *, bind_host: str, port: int, data_dir: Path, image: str,
) -> list[str]:
    return [
        docker, "run", "--rm", "--init",
        "--name", container_name(data_dir, port),
        "--publish", f"{bind_host}:{port}:8080",
        "--volume", f"{(data_dir.resolve() / 'server')}:/vol",
        "--env", "LOCAL_USERNAME=ml-expd",
        image,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bind-host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--container-uid", type=int, default=999)
    parser.add_argument("--docker", default="/usr/bin/docker")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(data_dir, 0o700)
    name = container_name(data_dir, args.port)
    environment = {"PATH": "/usr/bin:/bin"}
    if os.environ.get("DOCKER_HOST"):
        environment["DOCKER_HOST"] = os.environ["DOCKER_HOST"]
    server_dir = data_dir / "server"
    server_dir.mkdir(mode=0o755, exist_ok=True)
    initialized = data_dir / ".server-volume-initialized"
    desired_marker = f"wandb uid={args.container_uid}\n"
    try:
        current_marker = initialized.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        current_marker = ""
    if current_marker != desired_marker:
        prepared = subprocess.run(
            [
                args.docker, "run", "--rm", "--user", "0",
                "--entrypoint", "sh", "--volume", f"{server_dir}:/vol",
                args.image, "-c",
                f"chown -R {args.container_uid}:0 /vol && chmod 0700 /vol",
            ],
            env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=60, check=False,
        )
        if prepared.returncode != 0:
            return 2
        initialized.write_text(desired_marker, encoding="utf-8")
        os.chmod(initialized, 0o600)
    # The workspace collector lease guarantees there is only one legitimate
    # owner. Recover an exact-name container left behind when the prior daemon
    # was killed before its signal handler ran; the bind-mounted data remains.
    subprocess.run(
        [args.docker, "rm", "--force", name], env=environment,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=15, check=False,
    )
    process = subprocess.Popen(
        docker_command(
            args.docker, bind_host=args.bind_host, port=args.port,
            data_dir=data_dir, image=args.image,
        ),
        env=environment,
    )
    stopping = False

    def stop(_signum, _frame) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        subprocess.run(
            [args.docker, "stop", "--time", "10", name],
            env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=15, check=False,
        )

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        return process.wait()
    finally:
        if process.poll() is None:
            stop(signal.SIGTERM, None)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
