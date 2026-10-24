# Metrics: curves, results and provenance

Experiment code defines and computes metrics. ML-Expd stores arbitrary named
measurements with units and context, validates their declared schema and exposes
them through the API. Optional W&B publication uses the same accepted records.
There is no required accuracy, perplexity or project-specific metric vocabulary.

## Record and query

`ml-exp init source` includes the stdlib-only `ml_exp_metrics.py` helper. Use it
inside your training/evaluation program:

```python
from ml_exp_metrics import MetricWriter

metrics = MetricWriter()  # OUTPUT_DIR/metrics.jsonl; separate from STATE_DIR
metrics.log("train_loss", loss.item(), unit="nats/token", step=step,
            dataset_id="training-v1")
metrics.log("validation_loss", total_nll / target_tokens, unit="nats/token",
            step=step, dataset_id="validation-v1",
            numerator=total_nll, denominator=target_tokens)

# Your code selects and evaluates the checkpoint to report:
metrics.summary("validation_loss", selected_loss, unit="nats/token",
                step=selected_step, dataset_id="validation-v1",
                checkpoint_id=selected_checkpoint,
                numerator=selected_nll, denominator=selected_target_tokens)
```

`log()` appends a history point. `summary()` appends an explicit comparison
result with `summary:true`; it neither chooses a checkpoint nor adds a curve
point. Both use one append-only JSONL file. Do not replace it during final export.
Each call writes and flushes a complete line. The image needs no client package,
API token, W&B SDK/key or direct W&B connection. An installed client also exports
`from ml_exp_client import MetricWriter`.

Query an exact Attempt, repeating `--name` to select more metrics:

```bash
ml-exp run metrics --project my-study --run trial-001 --attempt attempt-001 \
  --name train_loss --name validation_loss --max-points 2000
```

## Record format

One line is one JSON object, for example:

```json
{"name":"validation_loss","value":2.5,"unit":"nats/token","step":100,"dataset_id":"validation-v1","variant_id":"baseline","checkpoint_id":"step-0100","numerator":25000,"denominator":10000,"summary":true}
```

| Fields | Contract |
|---|---|
| `name`, `value`, `unit` | Metric name, Python `int`/`float` or null, explicit unit; booleans are not values |
| `step`, `epoch`, `timestamp` | Optional measurement position/time |
| `dataset_id`, `variant_id`, `checkpoint_id` | Optional context; separate variants never overwrite one another |
| `protocol_id` | Inherits the frozen protocol if omitted; explicit mismatch fails |
| `numerator`, `denominator` | Optional aggregation counts, retained with the measurement |
| `summary` | Optional boolean declaring a comparison result |
| `status`, `error` | Optional `VALID` (default), `PARTIAL`, `FAILED` or `MISSING`, plus a diagnostic |

Names/units are nonempty printable strings, at most 128 characters. One encoded
line is limited to 64 KiB. Convert tensors with `.item()` before writing.
Missing values, NaN and infinity never become zero; nonfinite values remain
failed evidence. Different units/protocols cannot silently overwrite one another.
Metric checkpoint/data references are authored labels; checkpoint file/hash
verification is a separate registration contract.

## Declare units and required contexts

Set `metrics_schema` and `evaluation` in the experiment JSON. A project can also
set future defaults through `GET/PUT /api/projects/P/metrics-schema`.

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
observed context, preserving old schema behavior.

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

Current clients use `metric-records.v1`:

```http
GET /api/attempts/P/R::attempt-001/metrics?format=records&name=validation_loss&max_points=2000
```

Repeat `name` for multiple exact metric names; a comma is part of a name, not a
separator. The Run-level `/api/runs/P/R/metrics` route supports the same query,
resolving the current Attempt. Responses contain identity fields, `protocol_id`,
`records`, `validation`, `total_records`, `downsampled`, `missing_names` and
`source_attempt_id`. They omit legacy duplicate projections and server paths.

`total_records` counts matching normalized records. `validation` checks complete
history before name filtering/downsampling, including required metrics, failed
records and conflicts. Use full exported JSONL for analysis of every point.
`summary:true` is distinct from history even at the same step. Missing observations
are `NOT_OBSERVED`; incomplete contexts are `PARTIAL`; invalid values, units,
protocols or conflicting rewrites are `FAILED`. Identical observations deduplicate.

Legacy HTTP callers that omit `format` retain their existing response. The API's
legacy `keys` parameter selects record fields; it is not a metric-name filter and
cannot be combined with `format=records`. Prefer CLI `--name`; it requires the
new capability instead of guessing legacy semantics.

## Delivery and retention

Workers read complete newline-terminated records roughly every five seconds,
in batches of at most 64. Retry preserves event IDs and cursor without stopping
training. A final uploaded root `metrics.jsonl` or `train_metrics.jsonl` can fill
delivery gaps; journal-write failure prevents archive publication until retry.
Stdout-only/old worker observations may be sampled and cannot reconstruct a full
curve. Use a Runtime with `experiment-records.v1` for live delivery.

Accepted metrics survive result archive ACK/release. The canonical journal and
independent publication cursors remain in the historical `tracking/outbox.sqlite`;
this is not a second copy of weights or datasets. See [W&B](wandb.md) for display,
publication confirmation and the distinction between curves and final results.
