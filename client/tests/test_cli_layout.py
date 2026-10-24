"""Public menus share the existing guarded handlers with hidden flat compatibility."""
from copy import deepcopy
import json
from pathlib import Path
import shlex

import pytest

from ml_exp_client.cli import main, parser
from ml_exp_client.cli_layout import LEGACY_PATHS, normalize_argv, output_context
from ml_exp_client.presentation import experiment_summary
from ml_exp_client.state import save


EXAMPLES = {
    "experiment": ["experiment.json", "--state", "run.json"],
    "status": ["--state", "run.json"],
    "logs": ["--state", "run.json"],
    "metrics": ["--state", "run.json"],
    "download": ["--state", "run.json", "--out", "results"],
    "cancel": ["--state", "cancel.json"],
    "retry": ["--state", "retry.json", "--max-gpu-hours", "2"],
    "checkpoints": ["--project", "demo", "--run", "trial", "--attempt", "attempt-001"],
    "collect": ["--project", "demo", "--run", "trial", "--attempt", "attempt-001"],
    "delete": ["--state", "delete.json"],
    "acknowledge": ["--directory", "results"],
    "create": ["--runtime-state", "env.json", "--run", "trial", "--executor", "gpu"],
    "prepare": ["--project", "demo", "--run", "trial", "--max-gpu-hours", "2", "--state", "submission.json"],
    "execute": ["--state", "submission.json", "--confirm", "EXECUTE exact-action"],
    "submission": ["--id", "action-" + "a" * 16],
    "watch": ["--project", "demo", "--run", "trial"],
    "snapshots": ["--project", "demo", "--run", "trial", "--attempt", "attempt-001"],
    "pack": ["--project", "demo", "--source", "code", "--state", "env.json"],
    "runtime": ["--state", "env.json"],
    "asset-upload": ["--project", "demo", "--directory", "data", "--state", "data.json"],
    "assets": ["--project", "demo"],
    "asset-import-oss": ["--project", "demo", "--executor", "gpu", "--object-key", "data.tar", "--sha256", "a" * 64, "--bytes", "10"],
    "asset-download": ["--project", "demo", "--id", "asset." + "a" * 64, "--out", "data.tar"],
    "source-import": ["--project", "demo", "--directory", "code"],
    "tracking": ["--project", "demo", "--run", "trial"],
    "wandb": ["--project", "demo", "--disable"],
}


@pytest.mark.parametrize("legacy", EXAMPLES)
def test_every_published_flat_command_resolves_to_one_public_leaf(legacy):
    suffix = EXAMPLES[legacy]
    old = parser().parse_args([legacy, *suffix])
    new = parser().parse_args([*LEGACY_PATHS[legacy], *suffix])
    assert vars(old) == vars(new) and old.command == legacy
    if hasattr(new, "execute"):
        assert new.execute is False
    assert set(EXAMPLES) == set(LEGACY_PATHS)


@pytest.mark.parametrize("prefix", [[], ["--url", "https://api.example", "--config", "client.json"],
                                   ["--config=client.json", "--url=https://api.example"]])
def test_normalization_only_rewrites_command_position_and_preserves_option_values(prefix):
    argv = [*prefix, "pack", "--project", "status", "--source", "experiment", "--state", "runtime",
            "--entrypoint", "python", "tracking"]
    result = normalize_argv(argv)
    assert result == [*prefix, "env", "build", *argv[len(prefix) + 1:]]
    assert argv[len(prefix)] == "pack"
    args = parser().parse_args(argv)
    assert args.project == "status" and args.source == Path("experiment")
    assert args.entrypoint == ["python", "tracking"]


@pytest.mark.parametrize("argv", [[], ["run"], ["run", "advanced"], ["env"], ["data"], ["project"],
                                  ["tracking"], ["debug"], ["config"]])
def test_bare_menu_is_help_and_never_reads_config_or_connects(argv, monkeypatch, capsys):
    monkeypatch.setattr("ml_exp_client.cli.read_config", lambda *args: pytest.fail("help is offline"))
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *args: pytest.fail("help must not connect"))
    with pytest.raises(SystemExit) as result:
        main(argv)
    assert result.value.code == 0
    assert "usage: ml-exp" in capsys.readouterr().out


def test_top_help_has_ten_goal_entries_and_hides_flat_compatibility(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    text = capsys.readouterr().out
    assert all(name in text for name in ("run", "env", "data", "project", "tracking", "debug", "config", "update", "init", "check"))
    assert all(name not in text for name in ("source-import", "asset-upload", "wandb", "snapshots", "acknowledge", "==SUPPRESS=="))
    assert "init my-study" in text and "run submit" in text and "--execute" in text


def test_run_help_puts_common_tasks_first_and_advanced_last(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    text = capsys.readouterr().out
    names = ("submit", "status", "list", "logs", "metrics", "download", "checkpoints", "collect",
             "acknowledge", "cancel", "retry", "delete", "advanced")
    positions = [text.index("    " + name + " ") for name in names]
    assert positions == sorted(positions)


@pytest.mark.parametrize("argv", [["run", "submit"], ["env", "build"], ["project", "create"],
                                  ["run", "retry"], ["run", "advanced", "execute"]])
def test_real_leaf_missing_required_values_is_error_not_menu_help(argv, monkeypatch):
    monkeypatch.setattr("ml_exp_client.cli.read_config", lambda *args: pytest.fail("parse errors must be offline"))
    with pytest.raises(SystemExit) as result:
        main(argv)
    assert result.value.code == 2


def test_submit_help_and_parser_preserve_review_before_gpu(capsys):
    with pytest.raises(SystemExit):
        main(["run", "submit", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "Without --execute" in text and "does not authorize GPU training" in text
    args = parser().parse_args(["run", "submit", "experiment.json", "--state", "run.json"])
    assert args.command == "experiment" and args.execute is False and args.resume is False
    args = parser().parse_args(["run", "submit", "experiment.json", "--state", "run.json", "--execute"])
    assert args.execute is True


@pytest.mark.parametrize("argv", [["run", "retry"], ["run", "cancel"], ["run", "delete"]])
def test_new_action_paths_preserve_mutually_exclusive_execution_guards(argv):
    with pytest.raises(SystemExit) as result:
        parser().parse_args([*argv, "--state", "plan.json", "--execute", "--reconcile"])
    assert result.value.code == 2


@pytest.mark.parametrize("argv,path", [
    (["run", "list"], "/api/terminal/snapshot"),
    (["run", "list", "--project", "demo"], "/api/terminal/snapshot?project=demo"),
    (["env", "list"], "/api/environments"),
    (["project", "list"], "/api/projects"),
])
def test_catalogue_commands_are_only_get_requests(argv, path, monkeypatch, capsys):
    calls = []
    class API:
        def negotiate(self):
            return {"capabilities": ["environments.v1"]}
        def call(self, endpoint, **kwargs):
            calls.append((endpoint, kwargs))
            return {"catalogue": []}
    monkeypatch.setenv("ML_EXPD_API_URL", "https://api.example")
    monkeypatch.setenv("ML_EXPD_API_TOKEN", "local-fixture-token")
    monkeypatch.delenv("ML_EXPD_API_TOKEN_FILE", raising=False)
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *args: API())
    assert main(argv) == 0
    assert calls == [(path, {})] and json.loads(capsys.readouterr().out) == {"catalogue": []}


def test_missing_env_catalogue_capability_fails_without_issuing_get(monkeypatch, capsys):
    class API:
        def negotiate(self): return {"capabilities": []}
        def call(self, *args, **kwargs): pytest.fail("unsupported catalogue must not be queried")
    monkeypatch.setenv("ML_EXPD_API_URL", "https://api.example")
    monkeypatch.setenv("ML_EXPD_API_TOKEN", "fixture-token")
    monkeypatch.delenv("ML_EXPD_API_TOKEN_FILE", raising=False)
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *args: API())
    assert main(["env", "list"]) == 2
    assert "server lacks environments.v1" in capsys.readouterr().err


def test_custom_config_context_only_changes_known_output_commands_and_roundtrips(tmp_path):
    config = tmp_path / "client's custom \"connection\".json"
    value = {"execute_argv": ["ml-exp", "--url", "https://api.example", "run", "retry", "--state", "plan.json", "--execute"],
        "next_actions": [{"argv": ["ml-exp", "run", "status", "--state", "run.json"]}, {"kind": "INSPECT"}],
        "action": {"argv": ["python", "train.py"], "execute_argv": ["scientific", "payload"]},
        "manifest": {"argv": ["python", "train.py"]},
        "acknowledgement": {"retry_command": shlex.join(["ml-exp", "run", "acknowledge", "--directory", "results with 'quotes'"])}}
    original = deepcopy(value)
    view = output_context(value, config)
    assert value == original and view["action"] == original["action"] and view["manifest"] == original["manifest"]
    assert output_context(view, config) == view
    for argv in [view["execute_argv"], view["next_actions"][0]["argv"], shlex.split(view["acknowledgement"]["retry_command"])]:
        parsed = parser().parse_args(argv[1:])
        assert parsed.client_config == config
    assert parser().parse_args(view["execute_argv"][1:]).execute is True
    text = experiment_summary({"project": "demo", "run_id": "trial"}, state_file="run 'state'.json",
                              executed=False, client_config=str(config))
    inspect = next(line.removeprefix("Inspect: ") for line in text.splitlines() if line.startswith("Inspect: "))
    parsed = parser().parse_args(shlex.split(inspect)[1:])
    assert parsed.client_config == config and parsed.state == Path("run 'state'.json") and parsed.command == "status"


def test_saved_operation_config_copy_keeps_private_config_plan_and_action_unchanged(tmp_path, monkeypatch, capsys):
    config = tmp_path / "private 'connection' settings.json"
    save(config, {"url": "https://api.example", "token": "private-fixture-token"})
    plan = tmp_path / "reviewed 'retry' plan.json"
    action = {"action_id": "action-" + "a" * 16, "operation": "RETRY_ATTEMPT", "run_id": "trial", "attempt_id": "attempt-002",
        "scope": {"project": "demo", "scope_type": "attempt", "object_id": "trial::attempt-001"},
        "intent_digest": "sha256:" + "b" * 64, "gate_bundle_digest": "sha256:" + "c" * 64,
        "ready": True, "preflight_summary": {"requested_gpu_hours": 1}, "execution": {"status": "PREPARED"}}
    saved = {"server": "https://api.example", "contract": "training-operation.v1", "project": "demo", "run_id": "trial",
        "operation": "retry", "source_attempt_id": "attempt-001", "request": {"max_gpu_hours": 2},
        "plan": action, "action": action, "authorization_requested": False, "execution_requested": False}
    save(plan, saved)
    before, private_before = plan.read_bytes(), config.read_bytes()
    calls = []
    class API:
        def __init__(self, url, token):
            assert (url, token) == ("https://api.example", "private-fixture-token")
            self.base = url
        def negotiate(self): return {"capabilities": []}
        def call(self, path, **kwargs):
            calls.append((path, kwargs))
            assert path == "/api/actions/" + action["action_id"] and not kwargs
            return deepcopy(action)
    for key in ("ML_EXPD_API_TOKEN", "ML_EXPD_API_TOKEN_FILE", "ML_EXPD_API_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("ml_exp_client.cli.Client", API)
    assert main(["--config", str(config), "run", "retry", "--state", str(plan), "--json"]) == 0
    view = json.loads(capsys.readouterr().out)
    copied = parser().parse_args(view["execute_argv"][1:])
    assert copied.client_config == config and copied.state == plan and copied.execute
    assert calls == [("/api/actions/" + action["action_id"], {})]
    assert plan.read_bytes() == before and config.read_bytes() == private_before and view["action"] == action


@pytest.mark.parametrize("value", [None, [], {"next_actions": [None, {"argv": "not a command"}]},
    {"execute_argv": ["python", "train.py"]}, {"acknowledgement": {"retry_command": "malformed 'quote"}},
    {"acknowledgement": {"retry_command": "other program"}}])
def test_config_context_preserves_unrecognized_or_malformed_command_fields(value):
    assert output_context(value, None) is value
    assert output_context(value, "custom.json") == value
