# Metrics: write once, query or publish

Training code writes `OUTPUT_DIR/metrics.jsonl`. The managed worker sends complete
lines to the platform journal; the API reads that journal and optional W&B
publication uses the same observations. No W&B key, SDK or direct cloud connection
is needed in the training container.

## Training history and explicit results

`ml-exp init source` includes the stdlib-only `ml_exp_metrics.py`:

```python
from ml_exp_metrics import MetricWriter

metrics = MetricWriter()  # OUTPUT_DIR, independently of checkpoint STATE_DIR
metrics.log("train_loss", loss.item(), unit="nats/token", step=step,
            dataset_id="training-v1")
metrics.log("validation_loss", total_nll / tokens, unit="nats/token", step=step,
            dataset_id="validation-v1", numerator=total_nll, denominator=tokens)

# After the experiment has selected and evaluated a checkpoint:
metrics.summary("validation_loss", selected_loss, unit="nats/token",
                step=selected_step, checkpoint_id=selected_checkpoint,
                dataset_id="validation-v1",
                numerator=selected_total_nll, denominator=selected_tokens)
```

`log()` appends a history observation. `summary()` writes the same record format
with `summary:true`, explicitly selecting a result for comparison. It does not
choose the best checkpoint. Keep summary and history in the same file; do not
overwrite history during final export. An installed client also provides
`from ml_exp_client import MetricWriter`.

Each line is one JSON object:

```json
{"name":"validation_loss","value":2.5,"unit":"nats/token","step":100,"dataset_id":"validation-v1","variant_id":"baseline","checkpoint_id":"step-0100","numerator":25000,"denominator":10000,"summary":true}
```

| Fields | Meaning |
| --- | --- |
| `name`, `value`, `unit` | Arbitrary metric name, scalar measurement and explicit unit |
| `step`, `epoch`, `timestamp` | Optional measurement position/time; multiple positions provide a curve |
| `dataset_id`, `variant_id`, `checkpoint_id` | Optional scientific context; different variants remain separate |
| `protocol_id` | Omit to inherit the frozen protocol; an explicit mismatch fails validation |
| `numerator`, `denominator` | Optional aggregation counts, without unit conversion or recomputation |
| `summary` | Optional boolean; true explicitly declares a result, independently of history |
| `status`, `error` | Optional `VALID` (default), `PARTIAL`, `FAILED` or `MISSING`, and diagnostic |

Names and units are nonempty printable strings of at most 128 characters.
Nonfinite measurements remain failed evidence with a null value, never zero.
Distributed code must emit its aggregate from a reporting rank; the platform does
not combine losses from different ranks. Large task-level results belong in
Parquet/JSONL artifacts, with this interface holding aggregate measurements.

## Declare units and requirements

Use a Run's `metrics_schema`, or project defaults at
`GET/PUT /api/projects/P/metrics-schema`:

```json
{
  "metrics_schema": {
    "schema_version": 1,
    "definitions": {
      "train_loss": {"unit": "nats/token"},
      "validation_loss": {
        "unit": "nats/token", "required": true,
        "required_when": {"dataset_id": "validation-v1"},
        "aggregation": "ratio_of_sums",
        "numerator_unit": "nats", "denominator_unit": "target token"
      }
    }
  },
  "evaluation": {
    "benchmark": "fineweb-validation-v1",
    "tokenizer": "sha256:ACTUAL_HASH", "context_tokens": 1024,
    "target_positions": "target tokens", "loss_mask": "target-only"
  }
}
```

Empty definitions impose no metric vocabulary. There are at most 256 definitions
and 16 KiB of finite evaluation metadata. The frozen protocol hash binds schema,
evaluation, source, image, argv/environment, inputs and any restore reference.
Changing defaults affects future Runs only. Authored claims such as `D_FT=0` are
metadata; the platform does not prove them scientifically.

`required_when` limits a requirement to exact matching `dataset_id`, `variant_id`
and/or `checkpoint_id` values. In the example, ordinary training steps do not
need validation loss. Matching evaluation contexts must still contain the required
valid metrics together: another step/variant/checkpoint cannot fill a gap.
An entirely unobserved target context is not evidence that evaluation ran.
Omitting the selector preserves the original rule: required metrics apply to
every observed context. Old schemas and protocol hashes are not rewritten.

`ratio_of_sums` validates finite counts and a positive denominator. Training code
computes the value; the platform does not check arithmetic equality or convert
nats/token to bits/raw byte. Metric meaning, aggregation, selection and scientific
success criteria belong to experiment code.

## One client read format

```sh
ml-exp metrics --project P --run R --attempt attempt-001 \
  --name train_loss --name validation_loss --max-points 2000
```

Current clients negotiate `metric-records.v1` and request
`GET /api/attempts/P/R::attempt-001/metrics?format=records`. Repeat the `name` query
parameter to filter by metric name, including names containing commas. The Run
endpoint `/api/runs/P/R/metrics` supports the same format for the current Attempt.

The response has `project`, `run_id`, `attempt_id`, `source_attempt_id`,
`protocol_id`, `records`, `validation`, `total_records`, `downsampled` and
`missing_names`. Each record retains scientific context, validity and errors.
No duplicate points/keys projection or server filesystem paths is included.
`total_records` counts matching normalized records; `validation` checks the
complete history before name filtering and downsampling. Download full JSONL for
complete analysis. `summary:true` observations remain distinct from history at
the same step.

For old callers, omitting `format` retains the legacy response and `keys` field
filter. Current clients use that path only when the server lacks the capability;
`--name` requires the new capability rather than guessing legacy semantics.

## Validation and delivery

Missing observations are `NOT_OBSERVED`; missing requirements or partial records
are `PARTIAL`; invalid values, units, protocols or conflicting rewrites are
`FAILED`. Identical observations deduplicate. References on a metric are labels;
checkpoint file/hash verification is a separate registration process.

The worker reads append-only complete lines roughly every five seconds, in
batches of at most 64 records. A line is limited to 64 KiB. Network failures retain
the cursor and event IDs for retry without stopping training. Stdout observations
are a sampled legacy path; private files or a final-only copy do not provide live
history. Old immutable worker images require rebuilding for live record delivery.

Before publishing an uploaded result archive, the platform imports root
`metrics.jsonl` and `train_metrics.jsonl`, deduplicates live observations and fills
missing final points. Journal write failure prevents publication and preserves
the upload for retry. Archive release does not delete metrics: the historical
`projects/tracking/outbox.sqlite` contains the canonical journal as well as
independent W&B confirmation cursors. Previously lost history cannot be invented.

See [W&B publication](wandb.md) for curves, explicit results and remote confirmation.
