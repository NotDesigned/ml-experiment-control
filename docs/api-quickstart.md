# An experiment from an external computer

Install only this repository's standalone `client` package in Python 3.10+.
The client needs the HTTPS API URL and a private Bearer token file; it needs no
Docker, SSH, SenseCore credentials or backend tools.

```bash
python3 -m venv .client-venv
. .client-venv/bin/activate
python -m pip install ./client
export ML_EXPD_API_URL='https://api.example.org/ml-expd'
export ML_EXPD_API_TOKEN_FILE="$HOME/.config/ml-expd/client.token"
ml-exp check --schema openapi.json
```

`check` returns API capabilities, resource policy, executors, approved base images
and storage limits. Executors describe supported GPU configurations, not currently
free capacity. Copy an approved image digest into Dockerfile's `FROM`.
Keep source, data and client state/results in separate directories:

```text
study/
  source/Dockerfile
  source/train.py
  data/tokens.bin
  experiment.json
```

`ml-exp init source --base-image <approved-digest>` writes a small starter.
Install dependencies in Dockerfile before copying frequently changed code.
The server additionally copies complete frozen source to `/workspace` and installs
its managed worker. The final image needs Python 3.10+, `/bin/sh`, GNU `timeout`
and permissions to create `/inputs` and `/outputs`.
See [Dockerfile rules](sensecore-user-workflow.md).

Write `experiment.json`; paths are relative to this config file:

```json
{
  "project": "my-study",
  "run_id": "trial-001",
  "source": "./source",
  "dockerfile": "Dockerfile",
  "entrypoint": ["python3", "train.py"],
  "executor": "wyd-l40s",
  "arguments": ["--steps", "20"],
  "resources": {"gpus": 1, "cpus": 8, "memory_gb": 32, "max_time": "00:10:00"},
  "inputs": [{"directory": "./data", "mount_path": "/inputs/data"}],
  "checkpoint_upload": {"interval_seconds": 60},
  "max_gpu_hours": 0.2
}
```

The program reads `/inputs/data`, writes under `OUTPUT_DIR`, and emits metrics to
stdout or `metrics.jsonl`. For safe checkpoint upload, the trainer must atomically
publish `checkpoint.ready.json` with exact paths, sizes and SHA256 hashes after
completing a checkpoint. Setting the interval alone does not make partial files
safe. Follow the [checkpoint example](sensecore-user-workflow.md).

Prepare without allocating GPU resources:

```bash
ml-exp experiment experiment.json --state trial.state.json
```

This validates local inputs and executor selection, imports source, builds or
reuses its Dockerfile Runtime, uploads separate dataset assets, freezes a Run and
prepares its Submission gates. Inspect the returned IDs, gates and budget.
New builds accept Dockerfile only. The ordinary image/environment/requirements
selectors have been removed. `/api/environments` remains a base-image catalogue;
existing READY Runtime IDs and immutable Runs remain usable.

Execute the prepared experiment and collect verified outputs:

```bash
ml-exp experiment experiment.json --state trial.state.json --resume \
  --execute --seconds 1800 --download-to results/trial-001
```

`--execute` explicitly authorizes scheduling within the configured GPU-hour
budget. State is saved before effectful requests. Runtime READY means an image
exists; Submission VERIFIED means the exact job is visible. Only scheduler
SUCCEEDED and verified artifacts complete training. Download checks archive
SHA256/size and every recorded file hash. Signed links are fetched without the
API token and are not saved in state.

After a disconnect, use the same command with `--resume`. It observes the same
Runtime/Run/Submission; uncertain build or scheduler requests are never replayed.
Changed configuration/source requires a new Run ID and state file. Keep state
outside source. Downloads require a new destination directory; after interruption,
inspect existing output or choose a fresh destination rather than overwriting it.

## Waiting and reuse

- `GET /api/projects/P/runtimes/R/progress`: current build phase, log tail,
  phase start, real last progress time, and a diagnostic after 120 seconds without
  recorded progress. Build/push and remote digest verification are separate.
- `GET /api/submissions/S/progress`: source validation, backend staging/image
  conversion, scheduler submission, exact job verification, and indexed scheduler
  state/queue reason with its observation timestamp and stale flag.

Cold OCI→SIF conversion on WYD may take minutes. Phase timeouts are displayed.
No-progress warnings do not prove failure. Queue reasons belong to the exact
Attempt; start-time estimates remain null without reliable backend evidence.
Observing or diagnosing a job never launches a replacement.

The same frozen source/Dockerfile/worker recipe reuses a READY Runtime. CCR-backed
BuildKit cache reuses dependency layers across source changes. The control server
still removes its transient builder/cache volume. WYD reuses a verified,
digest-bound SIF. New worker recipes require new images and preserve old history.
First-use conversion and queueing still take time; cache does not guarantee capacity.

## Lower-level commands and recovery

`pack` → `create` → `prepare` → `execute` → `watch` → `download` remain available.
`pack --project my-study --source ./source --state runtime.json` uses Dockerfile
by default. `create --runtime-state runtime.json` reuses that code/environment
image for a new Run. `runtime --state runtime.json --logs` reads logs;
`runtime --state runtime.json --reconcile` looks up a published receipt without
building. `submission --id <saved-id> --reconcile` observes the exact job without
resubmitting. A retry requires a terminal prior job and a new authorized Attempt.

Data uses 16 MiB parts numbered **0 through part_count-1**. Storage limits also
return `upload_part_number_base: 0`. `asset-upload --resume` continues an unchanged
directory. Archive, expanded-data and final-output limits default to 4 GiB;
source is separate: 64 MiB archive/256 MiB expanded. Read actual deployment limits
through `check` or `GET /api/storage-limits`.
See [multipart uploads](multipart-uploads.md) and [storage lifecycle](storage-lifecycle.md).

Project-defined metric names, units, completeness and frozen scoring protocols:
[metrics contract](metrics.md).
