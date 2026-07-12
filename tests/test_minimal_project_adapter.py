import importlib.util
import json
from pathlib import Path

from experiment_control.project import ProjectRegistry


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "minimal_project_adapter.py"


def load_adapter_class():
    spec = importlib.util.spec_from_file_location("minimal_project_adapter", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MinimalProjectAdapter


def test_minimal_adapter_covers_config_command_metrics_and_assets(tmp_path):
    adapter = load_adapter_class()()
    config = tmp_path / "tiny.json"
    config.write_text(json.dumps({"dataset": "tiny", "steps": 10}))

    adapter.validate_run({"config": str(config)})
    assert adapter.resolve_config(str(config), ["steps=20"])["steps"] == "20"
    assert adapter.command({"config": str(config), "config_overrides": ["steps=20"]}) == [
        "python", "train.py", "--config", str(config), "--override", "steps=20"
    ]
    requirement = adapter.plan_assets(str(config), [])[0]
    data_root = tmp_path / "project-data"
    assert adapter.asset_probes(
        [requirement], {"DATA_ROOT": str(data_root)},
    )[0].path == str(data_root / "tiny")
    assert adapter.parse_metric("step=10 val_loss=1.25") == {
        "step": 10, "val_loss": 1.25,
    }
    checkpoint = tmp_path / "checkpoint"
    assert adapter.parse_checkpoint(f"checkpoint={checkpoint} step=10 bytes=42") == {
        "path": str(checkpoint), "step": 10, "bytes": 42,
    }
    assert ProjectRegistry(adapter).get("minimal") is adapter


def test_minimal_adapter_summarizes_durable_eval_records(tmp_path):
    adapter = load_adapter_class()()
    (tmp_path / "eval_metrics.jsonl").write_text(
        '{"step": 5, "val_loss": 2.0}\n{"step": 10, "val_loss": 1.5}\n'
    )
    assert adapter.summarize(tmp_path) == {
        "latest_eval": {"step": 10, "val_loss": 1.5}
    }
