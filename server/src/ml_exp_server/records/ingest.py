"""Import verified output JSONL before the archive becomes releasable."""
from __future__ import annotations

import json
from .store import RecordStore, digest
from ..tracking.metric_contract import observations


def import_metrics(registry_root, project, run, attempt, tree, manifest):
    store = RecordStore(registry_root)
    binding = store.binding(project, run)
    if binding is None:
        definition = manifest.get("resolved_config", manifest)
        store.bind(project, run, {"enabled": False, "reason": "HISTORICAL_RUN_NOT_BOUND"},
                   {"ml_expd": {"evaluation": definition.get("evaluation", {})}})
        binding = store.binding(project, run)
    contract = binding["config"].get("ml_expd", {}).get("evaluation", {})
    scope = store.scope(project, run, attempt)
    count, rejected, batch = 0, 0, []
    for name in ("metrics.jsonl", "train_metrics.jsonl"):
        path = tree / name
        if not path.is_file():
            continue
        with path.open("rb") as source:
            number = 0
            while line := source.readline(65537):
                number += 1
                if len(line) > 65536:
                    # Consume one malformed logical record without allocating it.
                    while not line.endswith(b"\n") and (line := source.readline(65537)):
                        pass
                    record = None
                else:
                    try:
                        record = json.loads(line)
                    except (ValueError, UnicodeError):
                        record = None
                if not isinstance(record, dict):
                    rejected += 1
                    payload = {"kind": "diagnostic", "data": {"diagnostic": "INVALID_METRIC_JSONL"}}
                    event_id = "invalid." + digest([name, number, line.hex()])
                else:
                    count += 1
                    payload = {"kind": "metrics", "observations": observations(record,
                               contract.get("metrics_schema", {}), protocol_id=contract.get("protocol_id"),
                               source="exact-attempt-worker")}
                    event_id = "archive." + digest(payload)
                batch.append((event_id, payload))
                if len(batch) == 64:
                    store.append(scope, batch)
                    batch.clear()
    if batch:
        store.append(scope, batch)
    return {"status": "COMPLETE", "records": count, "invalid_lines": rejected}
