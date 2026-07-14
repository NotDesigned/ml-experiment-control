"""Command line entry point for the independent ``ml-expd`` process."""

from __future__ import annotations

import argparse
import getpass
import ipaddress
import json
import os
from pathlib import Path
import sys
from typing import Sequence


def _require_loopback_host(host: str) -> str:
    """Keep the unauthenticated control plane local to the daemon host."""

    value = host.strip().strip("[]")
    if value.lower() == "localhost":
        return host
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise SystemExit(
            "ml-expd has no HTTP authentication; --host must be a loopback address"
        ) from exc
    if not address.is_loopback:
        raise SystemExit(
            "ml-expd has no HTTP authentication; refusing a non-loopback --host"
        )
    return host


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-expd",
        description="Run the ML experiment control-plane daemon.",
    )
    parser.add_argument("--config", required=True, type=Path, help="server workspace YAML")
    subcommands = parser.add_subparsers(dest="command")
    credential = subcommands.add_parser(
        "credential", help="manage daemon-host publisher credentials",
    )
    credential_kind = credential.add_subparsers(dest="credential_kind", required=True)
    wandb = credential_kind.add_parser("wandb", help="manage a W&B API key")
    wandb_action = wandb.add_subparsers(dest="credential_action", required=True)
    set_command = wandb_action.add_parser("set")
    set_command.add_argument("reference")
    set_command.add_argument(
        "--stdin", action="store_true",
        help="read the API key from stdin instead of a hidden prompt",
    )
    for action in ("status", "clear"):
        command = wandb_action.add_parser(action)
        command.add_argument("reference")
    subcommands.add_parser(
        "doctor",
        help="read-only checklist of config, action gates, and W&B credential state",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="serve the persisted read model without live backend polling",
    )
    parser.add_argument("--log-level", default="info")
    return parser


def _credential_command(args: argparse.Namespace) -> int:
    from .credentials import CredentialError, CredentialStore
    from .project_config import load_server_config

    config = load_server_config(args.config)
    store = CredentialStore(Path(config.observability.credential_root))
    try:
        if args.credential_action == "set":
            secret = sys.stdin.read() if args.stdin else getpass.getpass("W&B API key: ")
            store.set_wandb_api_key(args.reference, secret)
            result = {"configured": True, "reference": args.reference}
        elif args.credential_action == "clear":
            removed = store.clear_wandb_api_key(args.reference)
            result = {
                "configured": False, "reference": args.reference, "removed": removed,
            }
        else:
            status = store.status(args.reference)
            result = {
                "configured": status.configured, "reference": status.reference,
            }
    except CredentialError as exc:
        print(f"credential error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _doctor_command(args: argparse.Namespace) -> int:
    from .credentials import CredentialError, CredentialStore
    from .project_config import ConfigError, load_server_config
    from .project_registry import ProjectRegistry, ProjectRegistryError

    checks: list[tuple[str, bool | None, str, str | None]] = []
    config = None
    try:
        config = load_server_config(args.config)
        checks.append(("server config", True, str(args.config), None))
    except ConfigError as exc:
        checks.append((
            "server config", False, str(exc),
            "check --config path and schema_version",
        ))

    if config is not None:
        runtime = config.action_runtime
        for flag in (
            "allow_project_writes", "allow_scheduler_mutations",
            "allow_observability_mutations",
        ):
            if getattr(runtime, flag):
                checks.append((f"action_runtime.{flag}", True, "enabled", None))
            else:
                checks.append((
                    f"action_runtime.{flag}", None, "disabled (default)",
                    f"set action_runtime.{flag}: true to allow this Action class",
                ))

        store = CredentialStore(Path(config.observability.credential_root))

        def credential_check(label: str, reference: str | None) -> None:
            if not reference:
                checks.append((
                    label, False, "credential reference not set",
                    "set the corresponding W&B credential reference in observability config",
                ))
                return
            try:
                status = store.status(reference)
            except CredentialError as exc:
                checks.append((label, False, str(exc), "use a valid credential reference"))
                return
            if status.configured:
                checks.append((label, True, status.reference, None))
            else:
                checks.append((
                    label, False, f"{status.reference} not set",
                    f"ml-expd --config ... credential wandb set {status.reference} --stdin",
                ))

        local = config.observability.local_wandb
        if not local.enabled:
            checks.append((
                "local W&B", None, "disabled",
                "set observability.local_wandb.enabled: true to publish locally",
            ))
        else:
            if local.managed:
                docker = Path(local.docker_executable)
                if docker.is_file() and os.access(docker, os.X_OK):
                    checks.append(("local W&B docker", True, str(docker), None))
                else:
                    checks.append((
                        "local W&B docker", False, f"{docker} is not an executable file",
                        "install Docker or point docker_executable at a working binary",
                    ))
            if not local.publisher_entity:
                checks.append((
                    "local W&B entity", False, "publisher_entity not set",
                    "set observability.local_wandb.publisher_entity",
                ))
            credential_check("local W&B credential", local.publisher_credential_ref)

        cloud = config.observability.wandb_cloud
        if not cloud.enabled:
            checks.append((
                "W&B cloud", None, "disabled",
                "set observability.wandb_cloud.enabled: true to publish to W&B Cloud",
            ))
        else:
            if not cloud.entity:
                checks.append((
                    "W&B cloud entity", False, "entity not set",
                    "set observability.wandb_cloud.entity",
                ))
            credential_check("W&B cloud credential", cloud.default_credential_ref)

        registry_root = config.project_registry_root_path()
        registry_path = registry_root / "registry.json"
        if not registry_path.is_file():
            checks.append((
                "projects", None,
                f"registry not initialized; {len(config.projects)} configured for bootstrap",
                "start ml-expd once to initialize the durable Project registry",
            ))
        else:
            try:
                records = ProjectRegistry.read_records(registry_root)
                checks.append(("projects", True, f"{len(records)} registered", None))
            except ProjectRegistryError as exc:
                checks.append(("projects", False, str(exc), "repair the Project registry"))

    for name, ok, detail, hint in checks:
        symbol = "✓" if ok is True else ("·" if ok is None else "✗")
        print(f"{symbol} {name}: {detail}")
        if hint:
            print(f"    → {hint}")
    return 0 if all(ok is not False for _, ok, _, _ in checks) else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "credential":
        return _credential_command(args)
    if args.command == "doctor":
        return _doctor_command(args)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - installation contract
        raise SystemExit("install the ml-experiment-server package to run ml-expd") from exc

    from .api.app import create_app_from_config_file

    host = _require_loopback_host(args.host)
    app = create_app_from_config_file(args.config, poll=False if args.snapshot else None)
    uvicorn.run(app, host=host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
