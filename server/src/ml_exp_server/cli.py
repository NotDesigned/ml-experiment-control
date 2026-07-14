"""Command line entry point for the independent ``ml-expd`` process."""

from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import sys
from typing import Sequence


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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "credential":
        return _credential_command(args)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - installation contract
        raise SystemExit("install the ml-experiment-server package to run ml-expd") from exc

    from .api.app import create_app_from_config_file

    app = create_app_from_config_file(args.config, poll=False if args.snapshot else None)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
