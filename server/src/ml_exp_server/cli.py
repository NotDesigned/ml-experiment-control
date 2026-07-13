"""Command line entry point for the independent ``ml-expd`` process."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-expd",
        description="Run the ML experiment control-plane daemon.",
    )
    parser.add_argument("--config", required=True, type=Path, help="server workspace YAML")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="serve the persisted read model without live backend polling",
    )
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
