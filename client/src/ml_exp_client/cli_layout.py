"""One public command tree, with old flat argv normalized before parsing."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys


LEGACY_PATHS = {
    "experiment": ("run", "submit"),
    **{name: ("run", name) for name in ("status", "logs", "metrics", "download", "cancel", "retry",
                                       "checkpoints", "collect", "delete", "acknowledge")},
    **{name: ("run", "advanced", name) for name in ("create", "prepare", "execute", "submission", "watch", "snapshots")},
    "pack": ("env", "build"), "runtime": ("env", "status"),
    "asset-upload": ("data", "upload"), "assets": ("data", "list"),
    "asset-import-oss": ("data", "import-oss"), "asset-download": ("data", "download"),
    "source-import": ("project", "create"),
    "tracking": ("tracking", "status"), "wandb": ("tracking", "config"),
}
COMMAND_PATHS = {**LEGACY_PATHS, "run-list": ("run", "list"), "env-list": ("env", "list"),
                 "project-list": ("project", "list")}
GROUPS = {
    ("run",): "Prepare, run and inspect experiments; recover outputs",
    ("env",): "List environments and build or inspect Dockerfile images",
    ("data",): "Upload, list, import or download datasets",
    ("project",): "Create remote projects from source and list projects",
    ("tracking",): "Inspect W&B synchronization or configure its defaults",
    ("run", "advanced"): "Individual workflow steps; explicit gates and confirmation still apply",
}


def normalize_argv(arguments):
    """Rewrite only the command position, leaving option values and payloads intact."""
    args = list(arguments)
    if not args:
        return ["--help"]
    index = 0
    while index < len(args):
        item = args[index]
        if item in {"--url", "--config"}:
            index += 2
        elif item.startswith(("--url=", "--config=")):
            index += 1
        else:
            break
    if index >= len(args):
        return args
    command, rest = args[index], args[index + 1:]
    # `tracking` was a flat command. Its new bare/help spelling is the group.
    legacy = command in LEGACY_PATHS and (command != "tracking" or
        bool(rest) and rest[0] not in {"status", "config", "--help", "-h"})
    if legacy:
        args[index:index + 1] = LEGACY_PATHS[command]
    tail = tuple(args[index:])
    if tail in GROUPS or tail in {("debug",), ("config",)}:
        args.append("--help")
    return args


class CommandParser(argparse.ArgumentParser):
    """Only the root parser normalizes; each leaf is registered exactly once."""

    def parse_args(self, args=None, namespace=None):
        return super().parse_args(normalize_argv(sys.argv[1:] if args is None else args), namespace)


def output_context(value, client_config):
    """Add a selected config only to known presentation commands, never stored data."""
    if client_config is None or not isinstance(value, dict):
        return value
    path = str(Path(client_config).expanduser().resolve())

    def bind(argv):
        if not isinstance(argv, list) or not argv or argv[0] != "ml-exp" or not all(isinstance(arg, str) for arg in argv):
            return argv
        if argv[1:3] == ["--config", path]:
            return argv
        return [argv[0], "--config", path, *argv[1:]]

    def acknowledgement(proof):
        if not isinstance(proof, dict) or not isinstance(proof.get("acknowledgement"), dict):
            return proof
        record = proof["acknowledgement"]
        command = record.get("retry_command")
        if not isinstance(command, str):
            return proof
        try:
            argv = shlex.split(command)
        except ValueError:
            return proof
        if argv[:3] != ["ml-exp", "run", "acknowledge"]:
            return proof
        return {**proof, "acknowledgement": {**record, "retry_command": shlex.join(bind(argv))}}

    view = dict(value)
    if "execute_argv" in view:
        view["execute_argv"] = bind(view["execute_argv"])
    if isinstance(view.get("next_actions"), list):
        view["next_actions"] = [{**item, "argv": bind(item["argv"])} if isinstance(item, dict) and "argv" in item else item
                                for item in view["next_actions"]]
    view = acknowledgement(view)
    if isinstance(view.get("download"), dict):
        view["download"] = acknowledgement(view["download"])
    return view
