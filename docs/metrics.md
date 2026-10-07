# Project-defined metrics

The platform preserves numerical evidence and provenance. Experiment code owns
metric meaning, aggregation, model selection and success criteria. There are no
default accuracy/perplexity requirements or project-specific metric vocabulary.

## Declare units and requirements

Set project defaults with `GET/PUT /api/projects/P/metrics-schema`, or put an
override in the Run/experiment configuration:

```json
{
  "metrics_schema": {
    "schema_version": 1,
    "definitions": {
      "validation_loss": {
        "unit": "nats/token", "required": true,
        "aggregation": "ratio_of_sums",
        "numerator_unit": "nats", "denominator_unit": "target token"
      },
      "encoded_loss": {"unit": "bits/raw byte"},
      "valid_fraction": {"unit": "fraction"}
    }
  },
  "evaluation": {
    "benchmark": "programs-v1", "D_FT": 0,
    "tokenizer": "sha256:ACTUAL_HASH", "context_tokens": 1024,
    "target_positions": "first 256 raw bytes", "loss_mask": "target-only",
    "draw_weighting": "prior draws", "scoring_config": "sha256:ACTUAL_HASH"
  }
}
```

Names are arbitrary printable strings; units must be nonempty. At most 256
definitions and 16-KiB finite evaluation JSON are accepted. Empty definitions
impose no required vocabulary. Updating defaults affects future Runs only.

The frozen protocol hash includes schema, evaluation metadata and execution
provenance: source, image, argv/environment, inputs and restore reference when
configured. Changed tokenizer/context/mask/scoring needs a new identity. Metadata
such as `D_FT=0` is an authored claim; the platform does not prove it scientifically.

## Record and query

Write newline-terminated records to `OUTPUT_DIR/metrics.jsonl`. New opted-in
[W&B workers](wandb.md) deliver this file live to the server before result upload.
Named JSON records on stdout remain a sampled compatibility observation path:

```json
{"name":"validation_loss","value":2.5,"unit":"nats/token","step":100,"numerator":25000,"denominator":10000,"checkpoint_id":"step-0100","dataset_id":"fineweb-v1","variant_id":"baseline"}
```

The client starter (`ml-exp init source`) includes the standalone, stdlib-only
`ml_exp_metrics.py`. It is copied from the client's canonical metrics module:

```python
from ml_exp_metrics import MetricWriter

metrics = MetricWriter()  # resolves OUTPUT_DIR, not STATE_DIR
metrics.log("train_loss", loss.item(), unit="nats/token", step=step,
            dataset_id="training-v1", variant_id="baseline")
metrics.log("validation_loss", total_nll / tokens, unit="nats/token", step=step,
            numerator=total_nll, denominator=tokens,
            checkpoint_id=checkpoint_hash, dataset_id="validation-v1")
```

An installed client also exposes `from ml_exp_client import MetricWriter`.
Each call writes a complete line and flushes before returning, using append mode
so a reporting restart and final result cannot erase earlier points. There is no
network call or API/W&B credential in the helper. Names and units must be explicit;
nonfinite values remain invalid evidence, never zero. The server owns schema and
protocol validation. For distributed training, write the experiment's aggregate
from its reporting rank; the platform does not combine rank losses.

Keep the live file directly in OUTPUT_DIR, including when checkpoint recovery
uses STATE_DIR. Writing `updates.jsonl` in a private experiment directory, printing
to stdout, or copying a final-only file at exit does not provide a complete live
stream. Do not overwrite OUTPUT_DIR/metrics.jsonl during final export. The worker
reads incremental complete lines roughly every five seconds; network retry and
W&B confirmation add delay. Two distinct steps allow a curve, not just a scalar.
Older immutable Runtime images require a new build for the record-stream worker;
upgrading the API does not change them. Existing output JSONL can be delivered
after the run, but this is not real-time history delivery.

Query `/api/runs/P/R/metrics` or exact Attempt
`/api/attempts/P/R::attempt-001/metrics`. Responses retain records, context,
status/errors, completeness and source. `max_points` bounds/downsamples the
response; download full JSONL for complete analysis. Uploaded exact-Attempt
records take priority over sampled collector observations. Old wide numeric
records can inherit frozen schema units; otherwise they remain UNDECLARED.

Identity includes name, unit, protocol, checkpoint, dataset, variant, epoch and
step. Different variants/checkpoints remain separate. Conflicting values in the
same context are marked CONFLICTING rather than silently overwritten. A unit
that differs from the declared schema is FAILED/UNIT_MISMATCH; different units
remain distinct records, not interchangeable values. Invalid explicit
protocol/variant IDs fail validation.

Missing observations are NOT_OBSERVED, not zero. Partial/missing required records
produce PARTIAL; nonfinite values and failures/conflicts produce FAILED. NaN and
Infinity retain diagnostics with a null value. Completeness requires all declared
required metrics to be valid in the same context, not merely somewhere in a Run.

## Aggregation belongs to code

`ratio_of_sums` requires finite numerator and positive denominator. Code computes
the reported value; the platform preserves counts without converting units or
recalculating scientific results. Nats/token uses target tokens as denominator;
bits/raw byte uses raw bytes and the logarithm-base conversion. A fixed multiplier
cannot generally convert one into the other.

Keep task-level NLL, token/byte counts, repeated/invalid draws and weighting in
Parquet/JSONL artifacts for regrouping/bootstrap. Evaluation is an ordinary Run
with a checkpoint input and scoring protocol. Keep selection validation and
independent benchmark results under distinct dataset/protocol identities.
The platform does not choose the best model from arbitrary names containing `loss`.
