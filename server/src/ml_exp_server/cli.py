"""Command line entry point for the independent ``ml-expd`` process."""

from __future__ import annotations

import argparse
import getpass
import ipaddress
import json
import os
from pathlib import Path
import ssl
import sys
from typing import Sequence


def _require_loopback_host(host: str) -> str:
    """Keep the unauthenticated control plane local to the daemon host."""

    value = host.strip().strip("[]")
    if value.lower() == "localhost":
        return "localhost"
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
    return str(address)


def _validate_bind_host(host: str, *, authenticated: bool, tls: bool = False) -> str:
    """Permit remote binds only with native bearer authentication and TLS."""

    value = host.strip().strip("[]")
    if not value:
        raise SystemExit("ml-expd --host must not be empty")
    if value.lower() == "localhost":
        return "localhost"
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        address = None
    if address is not None and address.is_loopback:
        return str(address)
    if not authenticated:
        raise SystemExit(
            "ml-expd has no HTTP authentication; refusing a non-loopback --host"
        )
    if not tls:
        raise SystemExit(
            "ml-expd refuses a non-loopback bearer bind without --ssl-certfile "
            "and --ssl-keyfile"
        )
    return str(address) if address is not None else value


def _validate_tls_cert_chain(
    certfile: Path | None, keyfile: Path | None,
) -> tuple[Path | None, Path | None]:
    """Fail before runtime initialization when a TLS pair cannot be loaded."""

    if bool(certfile) != bool(keyfile):
        raise SystemExit("--ssl-certfile and --ssl-keyfile must be provided together")
    if certfile is None or keyfile is None:
        return None, None
    certificate = certfile.expanduser().resolve()
    private_key = keyfile.expanduser().resolve()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        context.load_cert_chain(
            certfile=str(certificate), keyfile=str(private_key),
        )
    except (OSError, ssl.SSLError) as exc:
        raise SystemExit(
            "ml-expd TLS certificate/key cannot be loaded: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return certificate, private_key


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
    doctor = subcommands.add_parser(
        "doctor",
        help="read-only checklist of config, action gates, and W&B credential state",
    )
    doctor.add_argument(
        "--json", action="store_true", dest="json_output",
        help="emit a stable machine-readable diagnostic report",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--ssl-certfile", type=Path)
    parser.add_argument("--ssl-keyfile", type=Path)
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
    from .http_auth import HttpAuthError, load_bearer_token
    from .project_config import ConfigError, load_research_project, load_server_config
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
        token_path = config.http_auth.token_path()
        if token_path is None:
            checks.append((
                "HTTP authentication", None, "disabled; loopback bind only",
                "configure http_auth.bearer_token_file before binding a shared host",
            ))
        else:
            try:
                load_bearer_token(token_path)
                checks.append(("HTTP authentication", True, "bearer token ready", None))
            except HttpAuthError as exc:
                checks.append((
                    "HTTP authentication", False, str(exc),
                    "create an owner-only token file with mode 0600",
                ))
        tls_pair = False
        try:
            certificate, private_key = _validate_tls_cert_chain(
                args.ssl_certfile, args.ssl_keyfile,
            )
            tls_pair = certificate is not None and private_key is not None
            checks.append((
                "TLS certificate", True if tls_pair else None,
                "certificate/key load successfully" if tls_pair else "not configured",
                None if tls_pair else "TLS is required for a non-loopback bind",
            ))
        except SystemExit as exc:
            checks.append((
                "TLS certificate", False, str(exc),
                "configure a readable, matching certificate and private key",
            ))
        try:
            bind = _validate_bind_host(
                args.host, authenticated=config.http_auth.enabled, tls=tls_pair,
            )
            checks.append((
                "HTTP bind policy", True,
                f"{bind}:{args.port} ({'TLS' if tls_pair else 'loopback plaintext'})",
                None,
            ))
        except SystemExit as exc:
            checks.append((
                "HTTP bind policy", False, str(exc),
                "use loopback, or configure bearer authentication and TLS",
            ))
        runtime = config.action_runtime
        for flag in (
            "allow_project_writes", "allow_source_imports", "allow_scheduler_mutations",
            "allow_observability_mutations", "allow_local_evidence_rebuild",
        ):
            if getattr(runtime, flag):
                checks.append((f"action_runtime.{flag}", True, "enabled", None))
            else:
                checks.append((
                    f"action_runtime.{flag}", None, "disabled (default)",
                    f"set action_runtime.{flag}: true to allow this Action class",
                ))
        import_roots = config.project_import_root_paths()
        if not import_roots:
            checks.append((
                "project_import_roots", None, "disabled (default)",
                "configure project_import_roots to enable zero-config discovery",
            ))
        else:
            root_errors: list[str] = []
            for root in import_roots:
                if not root.is_dir():
                    root_errors.append(f"{root}: directory does not exist")
                elif not os.access(root, os.R_OK | os.X_OK):
                    root_errors.append(f"{root}: not readable/searchable")
                elif runtime.allow_project_writes and not os.access(
                    root, os.W_OK | os.X_OK,
                ):
                    root_errors.append(f"{root}: not writable/searchable")
            checks.append((
                "project_import_roots", not root_errors,
                "; ".join(root_errors) if root_errors else (
                    f"{len(import_roots)} canonical root(s) accessible"
                ),
                (
                    "create/fix root permissions and, for systemd ProtectSystem, "
                    "add matching ReadWritePaths"
                ) if root_errors else None,
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
            if not local.publisher_entity and not local.publisher_credential_ref:
                checks.append((
                    "local W&B publisher", None, "disabled (dashboard-only)",
                    "set publisher_entity and publisher_credential_ref to mirror Attempts",
                ))
            else:
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
                errors = []
                for record in records:
                    if str(record.state.value) != "ACTIVE":
                        continue
                    try:
                        project = load_research_project(Path(record.project_file))
                        if project.project != record.project:
                            errors.append(f"{record.project}: manifest identity drift")
                    except (ConfigError, OSError, UnicodeDecodeError) as exc:
                        errors.append(f"{record.project}: {exc}")
                if errors:
                    checks.append((
                        "projects", False, "; ".join(errors)[:1000],
                        "repair registered Project manifests before starting ml-expd",
                    ))
                else:
                    checks.append(("projects", True, f"{len(records)} registered", None))
            except ProjectRegistryError as exc:
                checks.append(("projects", False, str(exc), "repair the Project registry"))

    failed = any(ok is False for _, ok, _, _ in checks)
    if getattr(args, "json_output", False):
        print(json.dumps({
            "status": "FAIL" if failed else "PASS",
            "checks": [
                {
                    "name": name,
                    "status": "PASS" if ok is True else (
                        "INFO" if ok is None else "FAIL"
                    ),
                    "detail": detail,
                    "hint": hint,
                }
                for name, ok, detail, hint in checks
            ],
        }, ensure_ascii=False, sort_keys=True))
        return 1 if failed else 0
    for name, ok, detail, hint in checks:
        symbol = "✓" if ok is True else ("·" if ok is None else "✗")
        print(f"{symbol} {name}: {detail}")
        if hint:
            print(f"    → {hint}")
    return 1 if failed else 0


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

    from .api.app import create_app
    from .project_config import load_server_config

    config = load_server_config(args.config)
    certificate, private_key = _validate_tls_cert_chain(
        args.ssl_certfile, args.ssl_keyfile,
    )
    tls = certificate is not None and private_key is not None
    host = _validate_bind_host(
        args.host, authenticated=config.http_auth.enabled, tls=tls,
    )
    app = create_app(config, poll=False if args.snapshot else None)
    uvicorn.run(
        app, host=host, port=args.port, log_level=args.log_level,
        ssl_certfile=str(certificate) if certificate else None,
        ssl_keyfile=str(private_key) if private_key else None,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
