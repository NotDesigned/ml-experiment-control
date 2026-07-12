"""One canonical scientific run-manifest constructor shared by control and runtime."""

from __future__ import annotations

from typing import Any


def build_run_manifest(
    *, project: str, run_id: str, created_at: str, config_path: str,
    resolved_config: dict[str, Any], source_id: str, runtime_tree_id: str,
    git_commit: str | None, campaign_id: str | None, campaign: str | None,
    image_id: str, run_dir: str, max_infra_retries: int,
    backend: dict[str, Any], resources: dict[str, Any],
    storage: dict[str, Any], command: list[str], execution: dict[str, Any],
    config_overrides: list[str] | None = None,
    assets: list[dict[str, Any]] | None = None,
    checkpoint: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    research_contract: dict[str, Any] | None = None,
    research_role: str | None = None,
) -> dict[str, Any]:
    """Build the platform-neutral immutable identity written as manifest.yaml."""
    # ``resume`` selects an attempt's starting checkpoint; it does not change
    # the scientific run being continued.  Keeping it in the immutable run
    # config would make attempt-002 conflict with attempt-001 solely because it
    # resumes from the checkpoint produced by attempt-001.
    scientific_config = {
        key: value for key, value in resolved_config.items() if key != "resume"
    }
    manifest = {
        "schema_version": 1,
        "identity_version": 2,
        "project": project,
        "run_id": run_id,
        "created_at": created_at,
        "config_path": config_path,
        "resolved_config": scientific_config,
        "source_id": source_id,
        "runtime_tree_id": runtime_tree_id,
        "git_commit": git_commit,
        "campaign_id": campaign_id,
        "campaign": campaign,
        "image_id": image_id,
        "seed": scientific_config.get("seed"),
        "backend": backend,
        "resources": resources,
        "storage": storage,
        "command": command,
        "execution": execution,
        "config_overrides": list(config_overrides or []),
        "assets": list(assets or []),
        "checkpoint": dict(checkpoint or {}),
        "evaluation": dict(evaluation or {}),
        "resume_policy": {
            "enabled": True,
            "max_infra_retries": max_infra_retries,
        },
    }
    if research_contract is not None:
        manifest["research_contract"] = research_contract
        manifest["research_role"] = research_role
    return manifest


def comparable_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return only scientific run identity, excluding attempt operations."""
    comparable = {key: value for key, value in manifest.items() if key != "created_at"}
    resolved = comparable.get("resolved_config")
    if isinstance(resolved, dict):
        comparable["resolved_config"] = {
            key: value for key, value in resolved.items() if key != "resume"
        }
    return comparable
