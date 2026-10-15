"""Read-only interpretation of historical project-specific run layouts.

Managed Runs freeze generic configuration and metric schemas in their manifest.
These guesses are retained only to display identified historical records;
they never write manifests or participate in preparation or scheduling.
"""
from pathlib import Path
import re

_CONFIG_EXCERPT_KEYS = (
    "seed", "seeds", "max_length", "global_batch_size", "epochs",
    "grad_accum_steps",
    "use_sentence_plan", "sentence_encoder_type", "sentence_encoder_grad",
    "plan_aux_passes", "plan_aux_token_context",
    "depth", "device_batch_size",
)

def source_parameters(run_dir: Path) -> dict[str, int]:
    """Read a tiny allowlist of immutable constants from a harness snapshot."""
    path = run_dir / "source" / "train.py"
    if not path.is_file():
        return {}
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    parameters = {}
    for authored, canonical in (("DEPTH", "depth"),
                                ("DEVICE_BATCH_SIZE", "device_batch_size")):
        match = re.search(rf"^\s*{authored}\s*=\s*(\d+)\b", source, re.MULTILINE)
        if match:
            parameters[canonical] = int(match.group(1))
    return parameters


def config_excerpt(resolved: dict) -> dict:
    return {key: resolved[key] for key in _CONFIG_EXCERPT_KEYS if key in resolved}


def applies(manifest: dict) -> bool:
    """Select only legacy readers; managed container manifests are generic."""
    if manifest.get("config_path") == "container_execution":
        return False
    resolved = manifest.get("resolved_config")
    return manifest.get("identity_version") != 2 or (
        isinstance(resolved, dict) and any(key in resolved for key in (
            "use_sentence_plan", "sentence_encoder_type", "sentence_encoder_grad",
            "plan_aux_passes", "plan_aux_token_context",
        ))
    )
