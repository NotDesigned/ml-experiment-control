"""Parse one run directory into a RunIndexRow. Pure functions, no I/O beyond reads.

Layout compatibility: two generations of run directories exist.
  new (canonical): manifest.yaml, attempts/*/attempt.yaml
  old (pre-refactor): control_manifest.yaml, attempts/*/control_attempt.yaml,
      science manifest mirrored under collected_run/manifest.yaml
Root-level status/backend/collection/decision json are mirrors of the current
attempt in both generations.

Evidence layers are kept separate on purpose: a root status.json with a fresh
mtime can still carry a stale RUNNING scheduler snapshot while collection.json
holds newer science metrics. Never collapse them into one state.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from ..schemas import (
    AttemptSummary,
    CampaignBinding,
    CampaignRelationship,
    EvidenceLayer,
    EvidenceLayers,
    RunIndexRow,
)

# collection.json keys that are operational rather than scientific metrics.
_COLLECTION_NON_METRIC_KEYS = {
    "attempt_id", "backend", "collected_from", "project", "run_dir", "run_id",
    "state", "runtime_state", "scheduler_state", "worker_state", "process_state",
    "model_state", "evidence_outcome", "evidence_unavailable_reason",
    "metric_evidence", "evidence_conflicts", "artifacts", "warnings",
    "latest_completed_checkpoint",
}

_EVAL_METRIC_KEYS = (
    "g_ppl", "oracle_plan_ppl", "shuffled_plan_ppl", "plan_ppl_gap",
    "token_recon_ppl", "generation_mean_entropy", "generation_nonempty_fraction",
    "val_bpb",
)


def _canonical_eval_variant_id(
    variants: list[dict[str, Any]], contract: Optional[dict[str, Any]],
) -> Optional[str]:
    """Resolve a flat eval view only when its variant identity is unambiguous."""
    declared = None
    if isinstance(contract, dict):
        declared = contract.get("canonical_eval_variant_id")
        evaluation = contract.get("evaluation")
        if declared is None and isinstance(evaluation, dict):
            declared = evaluation.get("canonical_variant_id")
    names = {str(item.get("variant")) for item in variants}
    if isinstance(declared, str) and declared in names:
        return declared
    if len(variants) == 1:
        return str(variants[0].get("variant"))
    return None

_ROLE_PATTERN = re.compile(r"^[a-z]+-([a-z]\d+)-")

_PROVENANCE_KEYS = ("git_commit", "source_id", "image_id", "config_path", "seed",
                    "campaign_id", "runtime_tree_id", "created_at")
_CONFIG_EXCERPT_KEYS = (
    "seed", "max_length", "global_batch_size", "epochs", "grad_accum_steps",
    "use_sentence_plan", "sentence_encoder_type", "sentence_encoder_grad",
    "plan_aux_passes", "plan_aux_token_context",
    "use_wandb", "wandb_base_url", "wandb_project", "wandb_entity",
    "wandb_run_id", "wandb_run_name", "wandb_url",
    "depth", "device_batch_size",
)

_CHECKPOINT_KEYS = (
    "latest_completed_checkpoint",
    "latest_completed_checkpoint_step",
    "checkpoint_exposure",
    "checkpoint_exposure_minutes",
)
_WANDB_INIT_PATTERN = re.compile(
    r"wandb initialized:\s*(https?://[^\s)>\]\"']+)", re.IGNORECASE
)
_WANDB_LOG_SCAN_LIMIT = 8 * 1024 * 1024


def parse_iso_ts(value: Any) -> Optional[float]:
    """Parse an ISO-8601 timestamp (as written in status.json/events.jsonl) to unix seconds."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _wandb_url_in_text(value: str) -> Optional[str]:
    match = _WANDB_INIT_PATTERN.search(value)
    return match.group(1) if match else None


def _wandb_url_in_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return _wandb_url_in_text(handle.read(_WANDB_LOG_SCAN_LIMIT))
    except OSError:
        return None


def _wandb_provenance(
    run_dir: Path,
    attempt_dir: Optional[Path],
    manifest: dict[str, Any],
    collection: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Resolve requested W&B identity and observed runtime URL from durable evidence."""
    resolved = manifest.get("resolved_config")
    resolved = resolved if isinstance(resolved, dict) else {}
    requested = _truthy(resolved.get("use_wandb"))
    result: dict[str, Any] = {
        "requested": requested,
        "enabled": requested,
        "initialized": False,
        "entity": resolved.get("wandb_entity"),
        "project": resolved.get("wandb_project"),
        "run_id": resolved.get("wandb_run_id") or manifest.get("run_id"),
        "name": resolved.get("wandb_run_name") or manifest.get("run_id"),
    }

    attempt_collection = (
        _load_json(attempt_dir / "collection.json") if attempt_dir is not None else {}
    )
    for source, payload in (
        ("attempt.collection.wandb", attempt_collection.get("wandb")),
        ("run.collection.wandb", collection.get("wandb")),
    ):
        observed = payload if isinstance(payload, dict) else {}
        url = observed.get("url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            result.update({
                "initialized": bool(observed.get("initialized", True)),
                "url": url,
                "evidence_source": observed.get("evidence_source") or source,
            })
            return result

    roots = [path for path in (attempt_dir, run_dir) if path is not None]
    structured_candidates = []
    log_candidates = []
    for root in roots:
        structured_candidates.extend((
            root / "wandb.json",
            root / "collected_run" / "wandb.json",
        ))
        log_candidates.extend((
            root / "stdout.log", root / "stderr.log",
            root / "collected_run" / "stdout.log",
            root / "collected_run" / "stderr.log",
        ))

    for path in structured_candidates:
        observed = _load_json(path)
        url = observed.get("url") or observed.get("run_url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            result.update({
                "initialized": bool(observed.get("initialized", True)),
                "url": url,
                "evidence_source": str(path),
            })
            for key in ("entity", "project", "run_id", "name"):
                if observed.get(key) is not None:
                    result[key] = observed[key]
            return result

    for path in log_candidates:
        url = _wandb_url_in_file(path)
        if url:
            result.update({
                "initialized": True,
                "url": url,
                "evidence_source": str(path),
            })
            return result

    process = collection.get("process_evidence")
    process = process if isinstance(process, dict) else {}
    for stream in ("stdout_tail", "stderr_tail"):
        lines = process.get(stream)
        if not isinstance(lines, list):
            continue
        url = _wandb_url_in_text("\n".join(str(item) for item in lines))
        if url:
            result.update({
                "initialized": True,
                "url": url,
                "evidence_source": f"collection.process_evidence.{stream}",
            })
            return result

    return result if requested else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping unparsable lines (partial writes on live runs)."""
    if not path.is_file():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


@dataclass(frozen=True)
class EvidenceSource:
    """One ordered filesystem source for execution evidence."""

    root: Path
    attempt_id: Optional[str]
    kind: str


def preferred_attempt_id(run_dir: Path) -> Optional[str]:
    """Resolve the current Attempt without inferring it from scientific metrics.

    Root status/collection/decision files are controller-maintained mirrors of
    the current Attempt. If none names an existing Attempt, use the last
    append-only attempt directory as a deterministic fallback.
    """
    attempts_dir = run_dir / "attempts"
    for mirror in (run_dir / "status.json", run_dir / "collection.json",
                   run_dir / "decision.json"):
        attempt_id = _load_json(mirror).get("attempt_id")
        if isinstance(attempt_id, str) and (attempts_dir / attempt_id).is_dir():
            return attempt_id
    if not attempts_dir.is_dir():
        return None
    attempts = sorted(path.name for path in attempts_dir.iterdir() if path.is_dir())
    return attempts[-1] if attempts else None


def evidence_sources(run_dir: Path, *, attempt_id: Optional[str] = None,
                     exact_attempt: bool = False) -> list[EvidenceSource]:
    """Return evidence roots in precedence order.

    Exact Attempt queries never fall back to another Attempt or a Run mirror.
    Run aggregation first reads the selected Attempt and uses root mirrors only
    for legacy/incomplete layouts.
    """
    selected = attempt_id or preferred_attempt_id(run_dir)
    sources: list[EvidenceSource] = []
    if selected:
        attempt_root = run_dir / "attempts" / selected
        if attempt_root.is_dir():
            sources.extend([
                EvidenceSource(attempt_root / "collected_run", selected,
                               "attempt_collected"),
                EvidenceSource(attempt_root, selected, "attempt_local"),
            ])
    if exact_attempt:
        return sources

    mirror_attempt = _load_json(run_dir / "collection.json").get("attempt_id")
    if not isinstance(mirror_attempt, str):
        mirror_attempt = selected
    sources.extend([
        EvidenceSource(run_dir / "collected_run", mirror_attempt, "run_mirror"),
        EvidenceSource(run_dir, mirror_attempt, "run_root"),
    ])
    unique: list[EvidenceSource] = []
    seen: set[Path] = set()
    for source in sources:
        if source.root in seen:
            continue
        seen.add(source.root)
        unique.append(source)
    return unique


def train_metric_records(
    run_dir: Path, *, attempt_id: Optional[str] = None, exact_attempt: bool = False,
) -> tuple[list[dict[str, Any]], Optional[Path], Optional[str]]:
    """Read training metrics from one coherent Attempt-scoped source."""
    for source in evidence_sources(
        run_dir, attempt_id=attempt_id, exact_attempt=exact_attempt,
    ):
        for filename in ("train_metrics.jsonl", "metrics.jsonl"):
            path = source.root / filename
            records = read_jsonl(path)
            if records:
                return records, path, source.attempt_id
    return [], None, attempt_id


def evaluation_variants(
    run_dir: Path, *, attempt_id: Optional[str] = None, exact_attempt: bool = False,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Read unique evaluation variants from one coherent evidence source.

    Collectors may place variants both directly under collected_run/ and under
    train_sampling_eval/. For duplicate names, the greatest numeric step wins.
    Evidence is never merged across different Attempts.
    """
    for source in evidence_sources(
        run_dir, attempt_id=attempt_id, exact_attempt=exact_attempt,
    ):
        candidates = list((source.root / "train_sampling_eval").glob("*/metrics.jsonl"))
        candidates.extend(source.root.glob("*/metrics.jsonl"))
        by_name: dict[str, dict[str, Any]] = {}
        for path in sorted(set(candidates)):
            records = read_jsonl(path)
            if not records:
                continue
            latest = records[-1]
            step = latest.get("step")
            score = float(step) if isinstance(step, (int, float)) else float("-inf")
            current = by_name.get(path.parent.name)
            if current is not None and score < current["_score"]:
                continue
            by_name[path.parent.name] = {
                "variant": path.parent.name,
                "latest": latest,
                "records": len(records),
                "source": str(path),
                "_score": score,
            }
        if by_name:
            variants = []
            for name in sorted(by_name):
                item = by_name[name]
                item.pop("_score", None)
                variants.append(item)
            return variants, source.attempt_id
    return [], attempt_id


def is_run_dir(path: Path) -> bool:
    """A run directory carries a manifest in either generation's spelling."""
    return (
        (path / "manifest.yaml").is_file()
        or (path / "manifest.json").is_file()
        or (path / "control_manifest.yaml").is_file()
        or (path / "collected_run" / "manifest.yaml").is_file()
    )


def _science_manifest(run_dir: Path) -> dict[str, Any]:
    """Prefer the science manifest; older dirs mirror it under collected_run/."""
    for candidate in (run_dir / "manifest.yaml", run_dir / "manifest.json",
                      run_dir / "collected_run" / "manifest.yaml",
                      run_dir / "control_manifest.yaml"):
        data = _load_json(candidate) if candidate.suffix == ".json" else _load_yaml(candidate)
        if data:
            return data
    return {}


def _infer_role(run_id: str) -> Optional[str]:
    match = _ROLE_PATTERN.match(run_id)
    return match.group(1) if match else None


def _mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _autoresearch_source_parameters(run_dir: Path) -> dict[str, int]:
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


def _scan_attempts(run_dir: Path) -> list[AttemptSummary]:
    attempts_dir = run_dir / "attempts"
    if not attempts_dir.is_dir():
        return []
    summaries = []
    for attempt_dir in sorted(attempts_dir.iterdir()):
        if not attempt_dir.is_dir():
            continue
        status = _load_json(attempt_dir / "status.json") or _load_json(attempt_dir / "attempt.json")
        backend = _load_json(attempt_dir / "backend.json")
        submission = _load_json(attempt_dir / "submission.json")
        summaries.append(AttemptSummary(
            attempt_id=attempt_dir.name,
            state=status.get("state"),
            backend=backend.get("backend") or status.get("backend")
            or ("local-cuda" if submission.get("gpu") is not None else None),
            backend_job_id=backend.get("backend_job_id") or status.get("backend_job_id"),
            decision=_observation_decision(_load_json(attempt_dir / "decision.json")),
            # ``job.sbatch`` is produced while preparing an Attempt and is
            # therefore not evidence that the scheduler mutation happened.
            # Only the durable submission receipt crosses that boundary.
            has_submission=(attempt_dir / "submission.json").is_file(),
        ))
    return summaries


def _observation_decision(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep operational decision metadata, not a hypothesis verdict."""
    return {
        key: payload[key]
        for key in ("action", "reason", "failure_class")
        if payload.get(key) is not None
    }


def _decision_history(run_dir: Path) -> list[dict[str, Any]]:
    """Return durable decision snapshots with honest source timestamps.

    Newer producers may append decision events or write attempt-local decision
    files. Older runs only have the root mirror, so that remains a single
    snapshot rather than being presented as a fabricated lifecycle timeline.
    """
    history: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_payloads: set[str] = set()

    def add(decision: dict[str, Any], *, source: Path, attempt_id: Any = None,
            timestamp: Optional[float] = None, mirror: bool = False) -> None:
        decision = _observation_decision(decision)
        if not decision:
            return
        payload_identity = json.dumps(decision, sort_keys=True, default=str)
        if mirror and payload_identity in seen_payloads:
            return
        identity = json.dumps({
            "attempt_id": attempt_id,
            "decision": decision,
            "timestamp": timestamp,
        }, sort_keys=True, default=str)
        if identity in seen:
            return
        seen.add(identity)
        seen_payloads.add(payload_identity)
        history.append({
            "ts": timestamp if timestamp is not None else _mtime(source),
            "attempt_id": attempt_id,
            "source": str(source),
            "action": decision.get("action"),
            "reason": decision.get("reason"),
            "failure_class": decision.get("failure_class"),
        })

    events_path = run_dir / "events.jsonl"
    for event in read_jsonl(events_path):
        event_name = str(event.get("event") or event.get("event_type") or "")
        if "decision" not in event_name.lower():
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        decision = payload.get("decision", payload)
        if isinstance(decision, dict):
            add(
                decision,
                source=events_path,
                attempt_id=event.get("attempt_id"),
                timestamp=parse_iso_ts(event.get("timestamp")),
            )

    attempts_dir = run_dir / "attempts"
    if attempts_dir.is_dir():
        for attempt_dir in sorted(attempts_dir.iterdir()):
            path = attempt_dir / "decision.json"
            add(_load_json(path), source=path, attempt_id=attempt_dir.name)

    root_path = run_dir / "decision.json"
    add(_load_json(root_path), source=root_path, mirror=True)
    history.sort(key=lambda item: item.get("ts") or 0)
    return history


def _latest_train_record(
    run_dir: Path,
) -> tuple[dict[str, Any], Optional[Path], Optional[str]]:
    records, path, attempt_id = train_metric_records(run_dir)
    return (records[-1] if records else {}), path, attempt_id


def _eval_layer(run_dir: Path) -> EvidenceLayer:
    """Latest record across every eval variant directory."""
    layer = EvidenceLayer()
    best_ts: Optional[float] = None
    variants, attempt_id = evaluation_variants(run_dir)
    layer.attempt_id = attempt_id
    for variant in variants:
        metrics_path = Path(variant["source"])
        ts = _mtime(metrics_path)
        latest = variant["latest"]
        layer.detail[variant["variant"]] = latest
        if ts is not None and (best_ts is None or ts > best_ts):
            best_ts = ts
            layer.source = str(metrics_path)
            layer.state = f"step {latest.get('step')}" if latest.get("step") is not None else "present"
    layer.as_of = best_ts
    return layer


def scan_run_dir(run_dir: Path, project: str, *, campaign: Optional[str] = None,
                 now: Optional[float] = None) -> RunIndexRow:
    """Build the read model for one run directory.

    Args:
        run_dir: the run directory (either file-name generation).
        project: project name the run root belongs to.
        campaign: campaign name; defaults to the parent directory name, which is
            the canonical `<local_root>/<campaign>/<run_id>` layout.
        now: injected clock for freshness (tests); defaults to time.time().
    """
    from .freshness import apply_freshness  # local import to avoid cycle

    manifest = _science_manifest(run_dir)
    status = _load_json(run_dir / "status.json")
    collection = _load_json(run_dir / "collection.json")
    decision = _observation_decision(_load_json(run_dir / "decision.json"))
    collected_status = _load_json(run_dir / "collected_run" / "status.json")
    run_id = manifest.get("run_id") or status.get("run_id") or run_dir.name
    selected_attempt = preferred_attempt_id(run_dir)
    attempt_dir = run_dir / "attempts" / selected_attempt if selected_attempt else None
    attempt_record = _load_json(attempt_dir / "attempt.json") if attempt_dir else {}
    attempt_summary = _load_json(attempt_dir / "summary.json") if attempt_dir else {}

    if manifest.get("campaign"):
        campaign_name = manifest["campaign"]
        campaign_source = "manifest"
    elif campaign:
        campaign_name = campaign
        campaign_source = "argument"
    else:
        campaign_name = run_dir.parent.name
        campaign_source = "directory"

    binding_issues: list[CampaignRelationship] = []
    manifest_project = manifest.get("project")
    if manifest_project and manifest_project != project:
        binding_issues.append(CampaignRelationship.PROJECT_MISMATCH)
    if campaign_source != "manifest" or not manifest_project:
        binding_issues.append(CampaignRelationship.LEGACY_INFERRED)
    initial_relationship = (
        CampaignRelationship.PROJECT_MISMATCH
        if CampaignRelationship.PROJECT_MISMATCH in binding_issues
        else CampaignRelationship.LEGACY_INFERRED
        if binding_issues
        else CampaignRelationship.UNRESOLVED
    )
    campaign_binding = CampaignBinding(
        relationship=initial_relationship,
        issues=binding_issues,
        origin_project=manifest_project,
        origin_campaign=manifest.get("campaign"),
        origin_revision=manifest.get("campaign_id"),
    )

    role = manifest.get("research_role")
    role_source = "manifest" if role else None
    if role is None:
        role = _infer_role(run_id)
        role_source = "heuristic" if role else None

    evidence = EvidenceLayers()

    # Scheduler layer: root status.json (mirror of current attempt).
    if status:
        evidence.scheduler = EvidenceLayer(
            state=status.get("state"),
            attempt_id=status.get("attempt_id"),
            as_of=parse_iso_ts(status.get("updated_at")) or _mtime(run_dir / "status.json"),
            source=str(run_dir / "status.json"),
            detail={k: status[k] for k in ("raw_state", "backend", "backend_job_id",
                                           "partition", "elapsed", "exit_code")
                    if status.get(k) is not None},
        )

    # Worker layer: remote-side status mirrored into collected_run/, or the
    # newer collection.json worker_state field.
    if collection.get("worker_state") is not None:
        evidence.worker = EvidenceLayer(
            state=collection.get("worker_state"),
            attempt_id=collection.get("attempt_id"),
            as_of=_mtime(run_dir / "collection.json"),
            source=str(run_dir / "collection.json"),
        )
    elif collected_status:
        evidence.worker = EvidenceLayer(
            state=collected_status.get("state"),
            attempt_id=collected_status.get("attempt_id"),
            as_of=parse_iso_ts(collected_status.get("updated_at"))
            or _mtime(run_dir / "collected_run" / "status.json"),
            source=str(run_dir / "collected_run" / "status.json"),
            detail={k: collected_status[k] for k in ("attempt_id",) if collected_status.get(k)},
        )

    # Process layer: only the newer collection format separates it.
    if collection.get("process_state") is not None:
        evidence.process = EvidenceLayer(
            state=collection.get("process_state"),
            attempt_id=collection.get("attempt_id"),
            as_of=_mtime(run_dir / "collection.json"),
            source=str(run_dir / "collection.json"),
        )
    elif attempt_record:
        evidence.worker = EvidenceLayer(
            state="LOCAL_GPU",
            attempt_id=selected_attempt,
            as_of=parse_iso_ts(attempt_record.get("finished_at")
                               or attempt_record.get("started_at")),
            source=str(attempt_dir / "attempt.json"),
        )
        evidence.process = EvidenceLayer(
            state=attempt_record.get("state"),
            attempt_id=selected_attempt,
            as_of=parse_iso_ts(attempt_record.get("finished_at")
                               or attempt_record.get("started_at")),
            source=str(attempt_dir / "attempt.json"),
            detail={key: attempt_record[key] for key in
                    ("return_code", "failure_class", "launch_error")
                    if attempt_record.get(key) is not None},
        )

    # Model layer: last train_metrics record; its own timestamp is the evidence time.
    train_record, train_path, train_attempt_id = _latest_train_record(run_dir)
    if train_record:
        ts = train_record.get("timestamp")
        evidence.model = EvidenceLayer(
            state=f"step {train_record.get('step')}" if train_record.get("step") is not None else "present",
            attempt_id=train_attempt_id,
            as_of=float(ts) if isinstance(ts, (int, float)) else (_mtime(train_path) if train_path else None),
            source=str(train_path) if train_path else None,
            detail={k: v for k, v in train_record.items() if k != "timestamp"},
        )
    elif collection.get("model_state") is not None:
        evidence.model = EvidenceLayer(
            state=collection.get("model_state"),
            attempt_id=collection.get("attempt_id"),
            as_of=_mtime(run_dir / "collection.json"),
            source=str(run_dir / "collection.json"),
        )

    evidence.evaluation = _eval_layer(run_dir)
    harness_metrics = status.get("metrics") if isinstance(status.get("metrics"), dict) else {}
    if not harness_metrics and isinstance(attempt_summary.get("metrics"), dict):
        harness_metrics = attempt_summary["metrics"]
    if harness_metrics.get("val_bpb") is not None:
        evidence.evaluation = EvidenceLayer(
            state="OBSERVED",
            attempt_id=selected_attempt,
            as_of=parse_iso_ts(attempt_summary.get("collected_at"))
            or _mtime(attempt_dir / "summary.json"),
            source=str(attempt_dir / "summary.json"),
            detail={"val_bpb": harness_metrics.get("val_bpb")},
        )

    latest_metrics: dict[str, Any] = {}
    for key, value in collection.items():
        if key in _COLLECTION_NON_METRIC_KEYS or not isinstance(value, (int, float, str, bool)):
            continue
        if key not in _EVAL_METRIC_KEYS:
            latest_metrics[key] = value
    # train_metrics.jsonl is fresher than collection.json when both exist.
    for key, value in train_record.items():
        if key != "timestamp":
            latest_metrics[key] = value
    for key, value in harness_metrics.items():
        if key not in _EVAL_METRIC_KEYS and isinstance(value, (int, float, str, bool)):
            latest_metrics[key] = value

    provenance = {k: manifest[k] for k in _PROVENANCE_KEYS if manifest.get(k) is not None}
    source_identity = manifest.get("source")
    if isinstance(source_identity, dict):
        provenance.update({
            "git_commit": source_identity.get("git_commit"),
            "source_id": source_identity.get("source_sha256"),
            "config_path": "train.py",
            "source_origin": source_identity.get("origin"),
            "train_py_sha256": source_identity.get("train_py_sha256"),
        })
        provenance = {key: value for key, value in provenance.items() if value is not None}
        source_parameters = _autoresearch_source_parameters(run_dir)
        if source_parameters:
            provenance["resolved_config_excerpt"] = source_parameters
    resolved = manifest.get("resolved_config")
    if isinstance(resolved, dict):
        provenance["resolved_config_excerpt"] = {
            k: resolved[k] for k in _CONFIG_EXCERPT_KEYS if k in resolved
        }
    wandb = _wandb_provenance(run_dir, attempt_dir, manifest, collection)
    if wandb is not None:
        provenance["wandb"] = wandb

    contract = manifest.get("research_contract")
    if not isinstance(contract, dict):
        contract = None
    eval_variants, _ = evaluation_variants(run_dir)
    canonical_eval_variant_id = _canonical_eval_variant_id(eval_variants, contract)
    eval_metrics: dict[str, Any] = {}
    if canonical_eval_variant_id is not None:
        canonical = next(
            item for item in eval_variants
            if item.get("variant") == canonical_eval_variant_id
        )
        latest = canonical.get("latest")
        if isinstance(latest, dict):
            eval_metrics = {
                key: latest[key] for key in _EVAL_METRIC_KEYS if latest.get(key) is not None
            }
    elif not eval_variants and not collection.get("evidence_conflicts"):
        # Legacy collectors sometimes provide one unnamed eval record only.
        eval_metrics = {
            key: collection[key] for key in _EVAL_METRIC_KEYS
            if isinstance(collection.get(key), (int, float, str, bool))
        }
        if harness_metrics.get("val_bpb") is not None:
            eval_metrics["val_bpb"] = harness_metrics["val_bpb"]
    conflicts = list(collection.get("evidence_conflicts") or [])
    warnings = [
        warning for warning in list(collection.get("warnings") or [])
        if warning not in conflicts
    ]
    if eval_variants and canonical_eval_variant_id is None:
        warnings.append(
            "multiple eval variants present; flat eval_metrics suppressed until "
            "research_contract declares canonical_eval_variant_id"
        )
    checkpoint = {
        key: collection[key] for key in _CHECKPOINT_KEYS
        if collection.get(key) is not None
    }
    artifacts = collection.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    if attempt_summary:
        artifacts.setdefault("summary", {
            "records": 1,
            "nonempty_records": 1 if harness_metrics else 0,
        })
        integrity = attempt_summary.get("integrity")
        if isinstance(integrity, dict):
            artifacts["integrity"] = {
                "records": len(integrity),
                "nonempty_records": sum(value is True for value in integrity.values()),
            }
    if not manifest:
        warnings.append("no readable manifest found (manifest.yaml / manifest.json / control_manifest.yaml)")
    if CampaignRelationship.PROJECT_MISMATCH in binding_issues:
        warnings.append(
            f"manifest project {manifest_project!r} differs from configured project {project!r}"
        )

    row = RunIndexRow(
        project=project,
        campaign=campaign_name,
        campaign_source=campaign_source,
        campaign_binding=campaign_binding,
        run_id=run_id,
        role=role,
        role_source=role_source,
        run_dir=str(run_dir),
        scheduler_state=status.get("state"),
        evidence=evidence,
        latest_metrics=latest_metrics,
        eval_metrics=eval_metrics,
        eval_variants=eval_variants,
        canonical_eval_variant_id=canonical_eval_variant_id,
        decision=decision,
        decision_history=_decision_history(run_dir),
        research_contract=contract,
        research_contract_source=("manifest" if contract is not None else None),
        checkpoint=checkpoint,
        artifacts=artifacts,
        provenance=provenance,
        attempts=_scan_attempts(run_dir),
        warnings=warnings,
        evidence_conflicts=conflicts,
        scanned_at=now if now is not None else time.time(),
    )
    apply_freshness(row, now=row.scanned_at)
    return row


def discover_run_dirs(root: Path) -> list[Path]:
    """Find run directories below a root, including instance-scoped layouts.

    Controllers may insert durable namespaces such as
    ``state/<instance>/<campaign>/<run>`` below the Project run root.  Stop
    descending as soon as a Run is found so mirrored ``collected_run`` trees
    cannot become duplicate Runs.
    """
    if not root.is_dir():
        return []
    found: list[Path] = []
    for directory, children, _ in os.walk(root):
        path = Path(directory)
        children.sort()
        if is_run_dir(path):
            found.append(path)
            children[:] = []
    return sorted(found)
