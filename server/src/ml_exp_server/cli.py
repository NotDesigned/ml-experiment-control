"""Command line entry point for the independent ``ml-expd`` process."""

from __future__ import annotations

import argparse
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
    doctor = subcommands.add_parser(
        "doctor",
        help=(
            "read-only checklist of config, backend availability, and action gates"
        ),
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




def _doctor_command(args: argparse.Namespace) -> int:
    from experiment_control.backends import BackendServices, build_registry
    from experiment_control.runner import SubprocessRunner

    from .http_auth import HttpAuthError, load_bearer_token
    from .projects.project_config import ConfigError, load_research_project, load_server_config
    from .projects.project_registry import ProjectRegistry, ProjectRegistryError

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
            "allow_local_evidence_rebuild",
        ):
            if getattr(runtime, flag):
                checks.append((f"action_runtime.{flag}", True, "enabled", None))
            else:
                checks.append((
                    f"action_runtime.{flag}", None, "disabled (default)",
                    f"set action_runtime.{flag}: true to allow this Action class",
                ))

        def unavailable(*_args, **_kwargs):
            raise RuntimeError(  # pragma: no cover - Backend availability invariant
                "backend Doctor probe used a non-probe service"
            )

        runner = SubprocessRunner()
        services = BackendServices(
            run_command=runner.run,
            local_run_dir=unavailable,
            backend_record=unavailable,
            summarize_run=unavailable,
            parse_metric=unavailable,
            parse_checkpoint=unavailable,
            atomic_write=unavailable,
            utc_now=unavailable,
        )
        registry = build_registry(services)
        for kind in sorted(registry.kinds):
            report = registry.get(kind).availability()
            failures = [
                check.name for check in report.checks if check.status == "FAIL"
            ]
            if not failures:
                checks.append((
                    f"backend.{kind}", True,
                    f"{len(report.checks)} host/API probe(s) passed", None,
                ))
                continue
            required = runtime.allow_scheduler_mutations
            checks.append((
                f"backend.{kind}", False if required else None,
                "unavailable: " + ", ".join(failures),
                (
                    "install/configure this backend before allowing scheduler mutations"
                    if required else
                    "backend is currently optional because scheduler mutations are disabled"
                ),
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
    if args.command == "doctor":
        return _doctor_command(args)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - installation contract
        raise SystemExit("install the ml-experiment-server package to run ml-expd") from exc

    from .api.app import create_app
    from .projects.project_config import load_server_config

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
