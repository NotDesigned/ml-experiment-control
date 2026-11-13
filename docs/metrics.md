# Metrics: curves, results and provenance

`METRICS_PATH` is `OUTPUT_DIR/metrics.jsonl`, so it is already part of the results. Do not copy
it into `OUTPUT_DIR` yourself: copying a file onto itself fails.

Experiment code defines and computes metrics. ML-Expd stores arbitrary named
measurements with units and context, validates their declared schema and exposes
them through the API. Optional [W&B](wandb.md) and [MLflow](mlflow.md) publication
use the same accepted records.
There is no required accuracy, perplexity or project-specific metric vocabulary.

For existing local training, see [local attachment](#attach-metrics-from-local-training).
To add evaluations computed after a cloud Run ended, see
[derived metrics](#add-derived-metrics-to-an-ended-run).
To remove metric names from what W&B shows for a finished Attempt, see
[W&B rebuilds](wandb.md#rebuild-a-finished-wb-run-without-some-metrics);
records here never change.

## Record and query

Write one JSON record per line to the file named by `METRICS_PATH`. On every backend
this is `OUTPUT_DIR/metrics.jsonl`, separate from `STATE_DIR`, so it is also part of
the result archive. Any language works; this example uses only the Python standard
library:

```python
import json, os

metrics = open(os.environ["METRICS_PATH"], "a")

def write(**record):
    metrics.write(json.dumps(record) + "\n")
    metrics.flush()

write(name="train_loss", value=loss.item(), unit="nats/token", step=step,
      dataset_id="training-v1")
write(name="validation_loss", value=total_nll / target_tokens, unit="nats/token",
      step=step, dataset_id="validation-v1",
      numerator=total_nll, denominator=target_tokens)

# Your code selects and evaluates the checkpoint to report:
write(name="validation_loss", value=selected_loss, unit="nats/token",
      step=selected_step, dataset_id="validation-v1",
      checkpoint_id=selected_checkpoint,
      numerator=selected_nll, denominator=selected_target_tokens, summary=True)
```

A record without `summary` is a history point. `summary:true` declares an explicit
comparison result; it neither chooses a checkpoint nor adds a curve point. Both use
one append-only JSONL file. Do not replace it during final export. Write and flush
complete lines. The image needs no client package, API token, W&B SDK/key or direct
W&B connection.

Query a Run, repeating `--name` to select more metrics:

```bash
ml-exp run metrics --project my-study --run trial-001 \
  --name train_loss --name validation_loss --max-points 2000
```

## Record format

One line is one JSON object, for example:

```json
{"name":"validation_loss","value":2.5,"unit":"nats/token","step":100,"dataset_id":"validation-v1","variant_id":"baseline","checkpoint_id":"step-0100","numerator":25000,"denominator":10000,"summary":true}
```

| Fields | Contract |
|---|---|
| `name`, `value`, `unit` | Metric name, JSON number or null, explicit unit; booleans are not values |
| `step`, `epoch`, `timestamp` | Optional measurement position/time |
| `dataset_id`, `variant_id`, `checkpoint_id` | Optional context; separate variants never overwrite one another |
| `protocol_id` | Inherits the frozen protocol if omitted; explicit mismatch fails |
| `numerator`, `denominator` | Optional aggregation counts, retained with the measurement |
| `summary` | Optional boolean declaring a comparison result |
| `status`, `error` | Optional `VALID` (default), `PARTIAL`, `FAILED` or `MISSING`, plus a diagnostic |

Names/units are nonempty printable strings, at most 128 characters. One encoded
line is limited to 64 KiB. Convert tensors with `.item()` before writing.
Lines are UTF-8 JSON. A byte order mark or any other encoding makes that line an
`INVALID_METRIC_JSONL` diagnostic; write files with `encoding="utf-8"`.
Missing values, NaN and infinity never become zero; nonfinite values remain
failed evidence. A failed value keeps at most 128 characters of its text in
`invalid_value`: a string as written, `true` or `false`, `nan`, `inf` or `-inf`,
and a list or object as sorted compact JSON with nonfinite numbers as those
words. Lists and objects that differ only in key order are therefore one event. Different units/protocols cannot silently overwrite one another.
Metric checkpoint/data references are authored labels; checkpoint file/hash
verification is a separate registration contract.

## Declare units and required contexts

Set `metrics_schema` and `evaluation` in the experiment file's `run` object.

```json
{
  "metrics_schema": {
    "schema_version": 1,
    "definitions": {
      "train_loss": {"unit": "nats/token"},
      "validation_loss": {
        "unit": "nats/token",
        "required": true,
        "required_when": {"dataset_id": "validation-v1"},
        "aggregation": "ratio_of_sums",
        "numerator_unit": "nats",
        "denominator_unit": "target token"
      }
    }
  },
  "evaluation": {
    "benchmark": "validation-v1",
    "tokenizer": "sha256:ACTUAL_HASH",
    "context_tokens": 1024,
    "target_positions": "target tokens",
    "loss_mask": "target-only"
  }
}
```

`required_when` matches exact `dataset_id`, `variant_id` and/or `checkpoint_id`
values. Here validation loss is required only in observed validation contexts,
not at every training step. Matching requirements must be satisfied together in
the same protocol/checkpoint/dataset/variant/epoch/step context. A value from
another context cannot fill a gap. Entirely unobserved contexts do not prove
that evaluation ran. Without `required_when`, requirements apply to every
observed context.

Empty definitions allow arbitrary names with explicit units. A nonempty schema
marks undeclared names; at most 256 definitions are accepted. Evaluation metadata
is finite JSON, at most 16 KiB. Its protocol hash binds schema, evaluation, source,
image, command/environment, inputs and restore/preparation context. Defaults
change future Runs only. Claims such as `D_FT=0` or a tokenizer hash are recorded
metadata, not scientific verification by the platform.

## Aggregation belongs to experiment code

For mean token loss, retain summed NLL and the effective target-token count:

```text
nats/token = total_nll_nats / effective_target_tokens
bits/raw byte = total_nll_nats / (ln(2) × scored_raw_bytes)
```

These denominators differ. Neither averaging batch means nor applying a fixed
unit conversion necessarily produces the intended result. `ratio_of_sums`
requires finite counts and a positive denominator; the platform does not recompute
`value` or check arithmetic equality. Experiment code defines masks, weighting,
invalid-draw handling, selection and success criteria. Distributed jobs should
emit their aggregate from one reporting rank. Store large per-task results as
Parquet/JSONL artifacts so they can be regrouped without rerunning the model.

## Compact API and validation

```http
GET /api/attempts/P/R::attempt-001/metrics?name=validation_loss&max_points=2000
```

Repeat `name` for multiple exact metric names; a comma is part of a name, not a
separator. Responses contain identity fields, `protocol_id`, `records`,
`validation`, `total_records`, `downsampled`, `missing_names` and
`source_attempt_id`. They omit server paths.

`total_records` counts matching normalized records. `validation` checks complete
history before name filtering/downsampling, including required metrics, failed
records and conflicts. Use full exported JSONL for analysis of every point.
`summary:true` is distinct from history even at the same step. Missing observations
are `NOT_OBSERVED`; incomplete contexts are `PARTIAL`; invalid values, units,
protocols or conflicting rewrites are `FAILED`. Identical observations deduplicate.

## Delivery and retention

Workers read complete newline-terminated records roughly every five seconds,
in batches of at most 64. Retry preserves event IDs and cursor without stopping
training. Stdout-only observations may be sampled and cannot reconstruct a full
curve; complete records are the live delivery path.

### Cloud Runs: the rest of the file from the result archive

A cloud worker sends `metrics.jsonl` to the server 128 KiB at a time, about every
five seconds, and keeps sending for at most 60 seconds after training exits. A Run
that wrote faster than that ends with only the start of its file in the journal.
The whole file is in the result archive, and the server imports the rest from there:

- Once the archive is published and verified, a background pass on the server,
  every 30 seconds, reads the archived root `metrics.jsonl` from the object store
  and continues from where the live stream stopped, with the same import state.
  Before `run download` releases an archive, the server does the same first. The
  worker sends nothing again and no job is started.
- The archived file's bytes before that point must hash to what the stream
  delivered. If they differ, for example because the file was rewritten, nothing
  is imported and the fill is refused with `METRICS_PREFIX_MISMATCH`.
- Records are validated against the frozen schema and deduplicated as live ones
  are, so a record already delivered is not added again, and a repeat after a
  restart adds nothing.
- When it is done, the journal ends with `METRICS_DRAINED`, as when the stream
  drains, with `"delivery": "result-archive"`, and `run history` reports
  `COMPLETE`. `run history` and `run tracking` show the fill as `metric_gap_fill`.

Runs that ended with a gap before this existed are filled by the first pass after
the server starts, as long as their archive is still kept. `metric_gap_fill.status`
is `FILLED` or `REFUSED`, and a refusal names its reason in `diagnostic`. Either is
final; a failed object-store read records nothing and is tried again.

| `diagnostic` | Meaning |
| --- | --- |
| `METRICS_PREFIX_MISMATCH` | The archived file does not start with what was streamed |
| `METRICS_NOT_ARCHIVED` | The archive holds no root `metrics.jsonl` |
| `METRICS_FILE_TOO_LARGE` | The archived file is over 256 MiB |
| `METRICS_ARCHIVE_RELEASED` | The archive was released before the fill ran |
| `METRICS_ARCHIVE_DIGEST_MISMATCH` | The archived bytes differ from the archive's receipt |
| `METRICS_ARCHIVE_LAYOUT_UNRECOGNIZED` | The file is not where the archive's receipt places it |
| `METRICS_IMPORTER_MISMATCH` | The Run's frozen metric schema no longer matches the stream's |

Only a Run whose training succeeded uploads a result archive, so only such a Run is
filled; `train_metrics.jsonl` is not used. If the W&B publisher already finished the
Attempt's W&B Run, the filled records stay in the journal and are not added to that
W&B Run; MLflow receives them.

Accepted metrics survive result archive ACK/release. The canonical journal and
independent publication cursors remain in `tracking/outbox.sqlite`;
this is not a second copy of weights or datasets. See [W&B](wandb.md) for display,
publication confirmation and the distinction between curves and final results, and
[MLflow](mlflow.md) for how the same records reach an MLflow server.

## Add derived metrics to an ended Run

Metrics computed after a cloud Run ended, such as an evaluation of its saved
checkpoints, can be added to one of its Attempts. They are kept apart from what
training reported, and the W&B publisher adds them to the Attempt's existing W&B Run.

```bash
ml-exp run append-derived --project demo --run train \
  --metrics derived.jsonl --confirmation 'APPEND demo/train/attempt-001'
```

The Attempt is chosen as `run status`, `run logs` and `run diagnose` choose it:
`--state`, or `--project` and `--run` with an optional `--attempt`, which defaults
to the Run's current Attempt. A wrong confirmation is refused before anything is
sent, and the message gives the exact one.

The file uses the [record format](#record-format), one record per line, each with
an explicit `name`, `value` and `unit`. A `step` or `epoch` may be earlier than the
last training step. Records are validated against the Run's frozen schema as
worker records are, but a line that is not a complete record refuses the whole
file (`DERIVED_METRICS_INVALID`), so nothing is half added. Blank lines are skipped.

- Only an Attempt whose journal recorded the end of its Run (`METRICS_DRAINED` with
  `SUCCEEDED`, `FAILED`, `CANCELLED` or `TIMEOUT`) takes a batch; any other Attempt
  is refused with `DERIVED_RUN_NOT_ENDED`.
- One file holds at most 5000 records and 2 MiB; a larger one is refused with
  `DERIVED_BATCH_TOO_LARGE`. Split it into several files.
- A name used by a training-time metric of the same Attempt is refused with
  `DERIVED_METRIC_NAME_CONFLICT`, so a derived record never changes a training
  curve or its summary. Use a distinct name such as `eval/accuracy`.

One file is one derived batch, identified by the SHA256 of its bytes and stored
with its submission time. Sending the same file again stores and publishes
nothing and returns the same result, so a lost response is safe to repeat. A
different file is a new batch. Records never replace or deduplicate against
existing ones.

Derived records do not enter the training journal, the result archive, or
`run metrics` reads and their validation. Each is stored in its own table with
kind `derived_metrics`, its batch SHA256 and submission time, and an event ID
`derived.<sha256>.<n>`, continuing the Attempt's publication sequence.
`ml-exp run tracking` lists each batch with its SHA256, submission time, record
count and sequence range. On W&B each row carries `ml_expd/source=derived` and
`ml_expd/derived_sha256`; see [W&B](wandb.md#derived-metrics-after-a-run-ended).

| Method and path | Body or result |
|---|---|
| `POST /api/runs/{project}/{run}/attempts/{attempt}/derived-metrics` | `sha256` (lowercase hex of the UTF-8 text), `records` (the JSONL text) and `confirmation` (`APPEND {project}/{run}/{attempt}`); returns the `batch` with `sha256`, `submitted_at`, `records`, `first_sequence` and `last_sequence` |

The server checks the SHA256 against the text (`DERIVED_METRICS_DIGEST_MISMATCH`).
This is available for cloud Runs.

## Attach metrics from local training

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
    "schema_version": 1,
    "definitions": {
      "train/loss": {"unit": "nats/token"}
    }
  },
  "wandb": {"enabled": true, "project": "experiments"}
}
```

The optional `execution_id` is 32 lowercase hexadecimal characters. When
omitted, the client generates it once and saves it before registration. Keep the
same definition and private upload state when resuming. The effective W&B route
and scientific definition are frozen on registration. Missing server W&B
credentials disable publication with a reason; journal ingestion still works.

`run_id` names the ML-Expd record and `execution_id` identifies this local
execution. The first Attempt is `attempt-001`; omit an Attempt or scheduler ID
from the definition. `metrics_schema`, `evaluation`, `parameters`, `seed` and
`wandb` are optional. Units can be supplied on each observation without a schema.

For an existing experiment, keep its source ID, initial-config SHA256 and data
fingerprints in `parameters`, and its scoring protocol in `evaluation`. These
are caller-provided provenance: registration preserves them without uploading
source, datasets or weights. For example, the following fields can be added to
`parameters`, replacing each placeholder with the original recorded value:

```json
{
  "source_id": "source.<original source digest>",
  "config_initial_sha256": "<original config.initial.json SHA256>",
  "data_fingerprints": {
    "train": "<original train fingerprint>",
    "validation": "<original validation fingerprint>",
    "canary": "<original canary fingerprint>"
  }
}
```

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

The saved state binds the API URL, project, complete registration definition,
execution ID, accepted byte offset, accepted-prefix SHA256 and packet chain.
It also retains any packet whose acknowledgement was lost. Resume with the same
`--state`, `--file` definition and append-only metrics content. A new pathname
with the same accepted prefix can resume after restarting the uploader; changing
the accepted bytes or replacing the file while it is being followed is refused.

### Existing direct W&B SDK writers

This initial attachment interface does not adopt an existing external W&B Run.
If training already writes directly through the SDK, attach with
`wandb.enabled=false` to ingest into ML-Expd without creating a duplicate W&B Run
or a second writer. A verified same-ID handover is a separate operation and is
not provided by `attach`. Do not stop a live training process to switch tracking.

### Local attachment HTTP contract

| Method and path | Body or result |
|---|---|
| `POST /api/projects/{project}/local-runs` | The definition above, with explicit `execution_id`; identical registration reuses the Run |
| `GET /api/local-runs/{project}/{run}` | Local execution, durable cursor, last ingress time and W&B scope progress |
| `POST /api/local-runs/{project}/{run}/metrics` | `execution_id`, byte `offset`, `previous_chain_sha256`, `bytes` as a JSON byte array and optional uppercase `finish` |

The initial chain is 64 zeroes. Each exact packet ACK supplies the next chain
and offset. The official client owns packet encoding and recovery state. The
capability is `native-local-metrics.v1`; all endpoints require the usual bearer
authentication.
