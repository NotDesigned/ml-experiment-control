# Project-defined metrics

The platform records evidence. Experiment code defines scientific metrics,
aggregation, model selection and success criteria. There are no default accuracy,
perplexity or ELF evaluation requirements. A single training or evaluation Run
can be submitted directly; Campaign is optional grouping. Research-question
APIs and operations have been removed. Legacy question files are left untouched
and are not loaded.

## Declare a contract

Add `metrics_schema` and optional `evaluation` to an experiment configuration
(or the `POST /api/projects/P/runs` body):

```json
{
  "metrics_schema": {
    "schema_version": 1,
    "definitions": {
      "fineweb_validation_loss": {
        "unit": "nats/token", "required": true,
        "aggregation": "ratio_of_sums",
        "numerator_unit": "nats", "denominator_unit": "target token"
      },
      "encoded_loss": {"unit": "bits/raw byte"},
      "valid_draw_fraction": {"unit": "fraction"}
    }
  },
  "evaluation": {
    "benchmark": "programs-v1", "D_FT": 0,
    "tokenizer": "sha256:YOUR_TOKENIZER_HASH",
    "context_tokens": 1024, "target_positions": "first 256 raw bytes",
    "loss_mask": "target-only", "draw_weighting": "prior draws",
    "scoring_config": "sha256:YOUR_CONFIG_HASH"
  }
}
```

`GET/PUT /api/projects/P/metrics-schema` reads/sets project defaults. Updating
these defaults affects future Runs. Existing Run definitions and manifests are
never rewritten. A per-Run schema overrides the default. Empty definitions impose
no required metric vocabulary.

Run creation freezes the schema and evaluation metadata together with actual
source ID, image digest, entrypoint/arguments/environment and data bindings
(including archive/file hashes). Its canonical SHA256 `protocol_id` is returned
and transported in the Run manifest. Checkpoint assets can be bound as inputs;
their IDs and hashes then participate in this identity. Put tokenizer, context,
loss mask, benchmark/config versions and protocol choices in `evaluation`.
Changing any of this context creates a new identity. Metadata is JSON, at most
16 KiB; there are at most 256 metric definitions. The platform does not verify
that a scientific statement such as `D_FT=0` is true: code and artifacts supply
that evidence.

## Produce and query

Write JSONL to `OUTPUT_DIR/metrics.jsonl` or emit individual records to stdout:

```json
{"name":"fineweb_validation_loss","value":2.5,"unit":"nats/token","step":100,"numerator":25000,"denominator":10000,"checkpoint_id":"step-0100","dataset_id":"fineweb-v1"}
```

Units can be inherited from the frozen schema for old wide records such as
`{"step":100,"fineweb_validation_loss":2.5}`. Without a schema, typed records
still need explicit units; old untyped numeric data remains readable with status
`UNDECLARED`, rather than being silently assigned a unit.

Query `/api/runs/P/R/metrics` or the exact Attempt resource
`/api/attempts/P/R::attempt-001/metrics`. Existing `points` remain available for
legacy chart clients; `metrics` returns records, contexts, required/missing
metrics and completeness. `evaluation` returns the frozen provenance contract.
`max_points` limits/downsamples the response; full uploaded JSONL is preserved
as an artifact and takes priority over sampled worker observations. A sampled
response is not a certificate that the entire benchmark completed.

A result is complete only when all declared required metrics are valid in the
same protocol/checkpoint/dataset/variant/epoch/step context. Separate contexts
remain separate; there is no cross-context best-model selection. Rewrites with
conflicting values or units are marked `CONFLICTING`, never silently corrected.
A mismatched explicit protocol ID fails validation. Missing observations are
`NOT_OBSERVED`; declared missing or partial records yield `PARTIAL`; nonfinite
values, producer failures and conflicts yield `FAILED`. Each record retains its
status/errors and source file. NaN/Infinity become a failed observation with a
null value, never zero; invalid-value diagnostics are retained.

`ratio_of_sums` requires finite numerator/denominator and a positive denominator.
The evaluator computes the value; the platform preserves counts and does not
convert units or recalculate it. In particular, nats/token and bits/raw byte have
different denominators. Preserve invalid/repeated draw counts and weighting in
metrics/artifacts according to your protocol.

For an evaluation Run, reuse the same normal Run API with a checkpoint input and
an evaluation entrypoint. Keep per-task/task-ID/NLL/token/byte tables in
Parquet/JSONL artifacts, not thousands of metric names. FineWeb validation and a
program benchmark should use distinct dataset/protocol identities. No automatic
selection is performed merely because both names contain `loss`.

Historical ELF JSONL/artifacts remain available. The generic read model no
longer derives `plan_ppl_gap`, promotes an alias to a scientific metric, or
requires a four-mode evaluation family.
