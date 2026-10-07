# Optional W&B publication

ML-Expd publishes to W&B official Cloud from the API server. Training containers
need no W&B SDK/key or direct connection to W&B. WYD workers use the existing
private API relay. Scheduler execution, recovery, outputs and scientific decisions
remain platform/experiment responsibilities.

## Configure once, choose a project per Run

After configuring the usual API URL/token, install the current standalone client.
Keep the W&B API key in a private file outside uploaded source/data. Configure it
without putting its contents in shell arguments or chat:

```sh
chmod 600 /private/path/wandb.key
ml-exp wandb --key-file /private/path/wandb.key
ml-exp wandb
```

Entity is optional. The server queries the key's W&B default entity and exposes
`resolved_entity`. To choose a team, add `--entity TEAM`; to set a default project,
add `--project PROJECT`. Configuration responses never return the API key.
`ml-exp wandb --disable` pauses publication; `--enable` resumes it.
`--clear-credentials` deletes the stored key. Configuration updates retain omitted
defaults/key; API `entity:null` requests default-entity resolution again, and
`project:null` restores the ML-Expd project-name fallback.
The settings API is `GET/PUT /api/tracking/wandb`; writes use the normal API
bearer credential. Its `api_key` field is write-only.

In an experiment JSON or `POST /api/projects/P/runs`:

```json
{
  "wandb": {"enabled": true, "project": "fineweb-lr-search"},
  "parameters": {"lr": 0.001, "wd": 0.1, "seed": 42}
}
```

`parameters` is an optional finite JSON object (16 KiB) of experiment-authored
research parameters. It records configuration; it does not change argv or choose
scientific metrics. The low-level client also accepts `create --wandb JSON
--parameters JSON`.

Omitting `wandb` requests publication by default. Project precedence is request
project, server default project, then ML-Expd project name. Entity precedence is
request entity, then configured/resolved default entity. A request entity override
must be accessible with the stored credential; permission failures remain visible
in publication status.

`{"wandb":{"enabled":false}}` disables this Run. Missing credentials, unresolved
default entity, or a disabled server yields effective `enabled:false` and a reason;
it does not block training. The preparation/Run response contains the effective
route. Preparation freezes it on acceptance. Duplicate requests reuse the same
route; changing defaults affects future Runs. Retry Attempts inherit their Run's
target. Changing a Run's destination or parameters requires a new Run identity.

## What is published

Each Attempt has a stable W&B run, grouped by ML-Expd project/Run. Preparation has
its own record, so build/data failures can be queried without inventing an Attempt.
Publication includes source/image/runtime/data identities, explicit parameters,
frozen metric schema/protocol, restore lineage, observed scheduler/worker stages,
exit information, metrics, checkpoint manifests and verified output receipts.
Full recovery checkpoints stay on datapool/NAS; weights/data/raw logs are not
copied to W&B by this integration. Private commands, SSH configuration, transfer
credentials, signed URLs and object-store keys are excluded.

For complete live metric history, rebuild a Dockerfile Runtime with
`experiment-records.v1` and append newline-terminated objects to
`OUTPUT_DIR/metrics.jsonl`. The worker reads bounded batches every five seconds
and posts using its exact-Attempt write capability. Retry retains the same IDs and
cursor. A partial final line is not consumed. A failed transfer does not stop the
training process. Final uploaded JSONL can fill delivery gaps.

Metrics keep their name, unit, step/epoch, variant, checkpoint, dataset, protocol,
status and numerator/denominator. W&B curve keys contain a readable name plus a
context hash; `ml_expd/metric_definitions` maps each key to its exact meaning.
Each curve uses its own scientific step axis. W&B history step is a separate
monotonic publication sequence. Failed/nonfinite observations retain diagnostics
and have no fabricated zero. No best-model selection or unit conversion is done.

Older immutable workers are unchanged. Their saved metrics can be published, but
collector observations may be sampled: they are explicitly labelled incomplete.
Stdout-only metrics use this compatibility observation path. Rebuild/use JSONL to
obtain the new live stream. Existing Runs are never automatically opted in.

## Confirm, retry and backfill

```sh
ml-exp tracking --project P --run R
ml-exp tracking --project P --run R --retry
```

`GET /api/runs/P/R/tracking` returns the frozen route, W&B URLs, confirmed/pending
record counts, last attempt/confirmation times and a fixed diagnostic code.
`POST …/tracking/retry` clears the retry delay/error; it does not submit a GPU job.
To explicitly publish a historical/disabled Run's saved evidence:

```http
POST /api/runs/P/R/tracking/backfill
Content-Type: application/json

{"project":"historical-study"}
```

Backfill is a separate, immutable publication binding; it never rewrites old
Run/Attempt definitions. Repeating it reuses the original target and IDs.
It preserves saved evidence only and cannot reconstruct missed training history.

A private SQLite outbox records complete accepted batches before acknowledging
workers. The publisher uses stable IDs and checks W&B history before advancing
its confirmed cursor. SDK enqueue is not confirmation. Network/credential errors
retain pending records and back off (up to five minutes). If remote history has
an identity conflict or a hole behind its latest step, reconciliation is required;
no older record is silently discarded or blindly appended. `REMOTE_ACK_PENDING`
means W&B has not yet confirmed the record. Normal observation/publication polls
are ten seconds, plus backend collector/API latency; this is not a latency SLA.
W&B session status may lag during a publisher outage; ML-Expd state/phase and
publication diagnostics remain authoritative.

Operators must back up `project_registry_root/tracking/` with the service state.
It contains the root-private settings/key and outbox. Do not restore an old outbox
on top of active records. The publisher SDK runs in a separate bounded process
with host/git/code telemetry disabled; it never reports the API server's GPU/CPU
usage as a training measurement.
