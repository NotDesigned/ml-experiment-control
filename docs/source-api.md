# API workflow reference

Use the [quickstart](api-quickstart.md) for a complete client example.
`GET /api/v2/openapi.json` is the installed request/response schema. All paths
below are relative to the API base; send Bearer auth and protocol 2 headers
([HTTP contract](http_contract.md)). `P`, `R`, `A`, `S` denote project, Run,
Attempt and Submission IDs returned by the service, not filesystem paths.

## Discover and import

| Method/path | Purpose |
|---|---|
| `GET /api/health` | Version, protocol, capabilities and enabled policy |
| `GET /api/executors` | Configured backend/resource choices |
| `GET /api/environments` | Approved base-image catalogue |
| `GET /api/storage-limits` | Upload limits, part policy and available storage |
| `POST /api/source-imports/archive?project=P&sha256=HASH` | Raw tar/tar.gz source upload |
| `POST /api/source-imports/git` | HTTPS repository plus exact 40-character commit |
| `GET /api/projects/P/sources/SOURCE_ID` | Immutable source metadata |

Source defaults are 64 MiB archive, 256 MiB expanded and 20,000 entries. Imports
reject links, escaping paths and credential-looking filenames; this is not a
content scan proving there are no secrets. Import creates a managed Project when
needed; it cannot silently replace an existing custom controller.

## Build a Runtime

```json
{
  "source_id": "source.ACTUAL_HASH",
  "dockerfile": "Dockerfile",
  "entrypoint": ["python3", "train.py"],
  "workdir": "/workspace"
}
```

| Method/path | Purpose |
|---|---|
| `POST /api/projects/P/runtimes/prepare` | Validate/freeze the definition above |
| `POST /api/projects/P/runtimes/RUNTIME_ID/execute` | Build using returned `confirmation` |
| `GET /api/projects/P/runtimes/RUNTIME_ID` | Durable state and receipt |
| `GET …/runtimes/RUNTIME_ID/logs` | Bound build log |
| `GET …/runtimes/RUNTIME_ID/progress` | Phases/activity/diagnostics |
| `POST …/runtimes/RUNTIME_ID/reconcile` | Look up publication without rebuilding |

Execute returns HTTP 202 after durable claim. Poll for READY. New builds are
Dockerfile-only; `image`, `environment_id` and `requirements` selectors are not
accepted. Image identity binds source, Dockerfile, managed worker and transport
recipe. Old READY images remain usable; an obsolete unbuilt definition needs
fresh preparation. See [builds](builds.md) for admission and uncertainty.

## Freeze a Run and submit

```json
{
  "run_id": "trial-001",
  "runtime_id": "runtime.ACTUAL_HASH",
  "executor": "wyd-l40s",
  "arguments": ["--epochs", "1"],
  "resources": {"gpus": 1, "cpus": 8, "memory_gb": 32, "max_time": "00:10:00"},
  "outputs": ["metrics.jsonl", "summary.json", "weights/**"]
}
```

`POST /api/projects/P/runs` freezes that definition. Optional fields include
`env`, `inputs`, `metrics_schema`, `evaluation`, `data_preparation`,
`checkpoint_persistence`, `resume_from` and `checkpoint_upload`. Read their
contracts in [data](data.md), [metrics](metrics.md) and [checkpoints](persistent-checkpoints.md).
Run IDs cannot be rebound; changes require a new Run. A retry gets a new Attempt.

| Method/path | Purpose |
|---|---|
| `POST /api/experiments/P/R/submissions/prepare` | Preflight/budget checks; body `max_gpu_hours`, `reason` |
| `GET /api/experiments/P/R/submissions` | Existing submission records |
| `POST /api/submissions/S/authorize` | Authorize prepared gates; body `note` |
| `POST /api/submissions/S/execute` | Use exact returned `confirmation` |
| `GET /api/submissions/S` | Durable state and next action |
| `GET /api/submissions/S/progress` | Preparation and current scheduler evidence |
| `POST /api/submissions/S/reconcile` | Recover the exact uncertain job without resubmitting |

Submission VERIFIED proves exact job visibility, not training completion. The
Action/outbox is recorded before the scheduler mutation. HTTP disconnect does
not stop execution. Unknown effects require reconciliation.
`GET /api/operations` describes generic operations; `/api/operations/direct`
prepares operations such as `attempt.cancel` or `attempt.retry`. Scope/object
IDs and parameters follow that catalogue/schema. Apply returned Action gates
rather than guessing a mutation body. Retry does not automatically pick a newer
checkpoint; the Run's frozen restore reference remains unchanged.

## Observe and download

| Method/path | Purpose |
|---|---|
| `GET /api/runs/P/R` | Scheduler state, bound evidence and provenance |
| `GET /api/runs/P/R/attempts` | Exact executions and current Attempt |
| `GET /api/runs/P/R/metrics` | Metric observations and contexts |
| `GET /api/attempts/P/R::A/logs` | Bounded stored Attempt logs |
| `GET /api/runs/P/R/attempts/A/checkpoints` | Registered persistent checkpoints |
| `GET /api/runs/P/R/attempts/A/files` | Collected/uploaded file inventory |
| `GET …/attempts/A/artifacts/download` | Short-lived signed URL and verified inventory |
| `GET …/attempts/A/artifacts/archive` | Authenticated proxy archive stream |
| `GET …/attempts/A/files/outputs/PATH` | Exact file; ranges and ETags |

The signed URL request requires API auth; the object GET carries no API token.
Check archive/file hashes and never save signed URLs. An empty file list or 404
does not establish that NAS contains no output. It may mean upload never finished.
Artifacts never fall back to another Attempt. See [recovery](recovery.md).

Internal `/launch-transfers`, `/asset-transfers`, `/checkpoint-transfers` and
`/attempt-uploads` routes use narrow worker capabilities. They are not substitutes
for client control API credentials. Research-question and tracking-vendor APIs
are absent; evaluation is an ordinary Run with checkpoint/data/protocol inputs.
