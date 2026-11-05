# Attach metrics from local training

`ml-exp run attach` uploads an existing append-only JSONL file through the native
client to the daemon journal. The daemon's existing W&B publisher handles cloud
publication; the client needs only its normal ML-Expd URL and API token.
Attachment never starts, restarts, stops or signals the training process.

The project must already be configured on the daemon with one canonical Run
root and discovery enabled. No cloud backend or scheduler credentials are
required. Use a new ML-Expd Run ID for each local execution.

Create `local.json`:

```json
{
  "run_id": "local-seed42",
  "seed": 42,
  "parameters": {"lr": 0.001},
  "evaluation": {"protocol_id": "validation-v1"},
  "metrics_schema": {
    "train/loss": {"unit": "nats/token"}
  },
  "wandb": {"enabled": true, "project": "experiments"}
}
```

The optional `execution_id` is 32 lowercase hexadecimal characters. When
omitted, the client generates it once and saves it before registration. Keep the
same definition and private upload state when resuming. The effective W&B route
and scientific definition are frozen on registration. Missing server W&B
credentials disable publication with a reason; journal ingestion still works.

Write one explicit observation per JSONL line:

```json
{"name":"train/loss","value":4.2,"unit":"nats/token","step":100,"epoch":0.5}
```

Names, values, units, step, epoch and supplied scientific context are preserved.
W&B uses exactly the supplied name, without added prefixes or suffixes. A scalar
can omit step and epoch. The existing metric schema and non-finite-value rules
apply; invalid JSON produces a diagnostic and increases `invalid_lines`.
Wide logs such as `{"global_step":100,"train/loss":4.2}` require an explicit
caller-written converter. The platform does not guess which fields are metrics
or invent units.

```bash
# Import all current bytes, then return.
ml-exp run attach --project demo --file local.json \
  --metrics metrics.jsonl --state attach.json

# Import history, then follow newly appended bytes.
ml-exp run attach --project demo --file local.json \
  --metrics metrics.jsonl --state attach.json --watch

# After the producer has stopped writing, explicitly drain and report completion.
ml-exp run attach --project demo --file local.json \
  --metrics metrics.jsonl --state attach.json --finish succeeded

ml-exp run local-status --project demo --run local-seed42
ml-exp run tracking --project demo --run local-seed42
```

`failed` and `cancelled` are also accepted finish values. Completion is the
client's report, not a daemon process check. The daemon reports process liveness
as `UNKNOWN`; an ingress timestamp proves only that metrics arrived. Local Runs
have `execution_kind=local` and no Submission or provider Job ID. The ordinary
Run/Attempt metric reads expose the accepted journal observations.

Each upload carries at most 64 KiB of raw input, including partial lines. A
durable packet intent, journal append, exact receipt and cursor are committed in
that order. A lost response leaves the exact pending packet in client state.
Rerun the same command to reconcile it; HTTP mutations are not automatically
retried. Replayed accepted packets return the original receipt and add no events.
Different bytes at an occupied offset are rejected.

On resumption the client checks the full accepted file prefix and pending bytes.
While following, it detects a replaced file or an observed truncation. The input
must remain append-only: this is not a watcher for arbitrary in-place edits.
One uploader holds the state lock. Keep the state file after finishing; do not
start a second uploader with a fresh state for the same execution.

## Existing direct W&B SDK writers

This initial attachment interface does not adopt an existing external W&B Run.
If training already writes directly through the SDK, attach with
`wandb.enabled=false` to ingest into ML-Expd without creating a duplicate W&B Run
or a second writer. A verified same-ID handover is a separate operation and is
not provided by `attach`. Do not stop a live training process to switch tracking.

## HTTP contract

| Method and path | Body or result |
|---|---|
| `POST /api/projects/{project}/local-runs` | The definition above, with explicit `execution_id`; identical registration reuses the Run |
| `GET /api/local-runs/{project}/{run}` | Local execution, durable cursor, last ingress time and W&B scope progress |
| `POST /api/local-runs/{project}/{run}/metrics` | `execution_id`, byte `offset`, `previous_chain_sha256`, `bytes` as a JSON byte array and optional uppercase `finish` |

The initial chain is 64 zeroes. Each exact packet ACK supplies the next chain
and offset. The official client owns packet encoding and recovery state. The
capability is `native-local-metrics.v1`; all endpoints require the usual bearer
authentication and protocol 2 header.
