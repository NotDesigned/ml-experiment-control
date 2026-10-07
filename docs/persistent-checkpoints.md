# Persistent checkpoints and restore

Complete recovery state can include model, optimizer, scheduler, RNG and data
cursor. Keep it on backend storage; export selected weights/logs as artifacts.
The platform hashes files but never interprets or unpickles training state.
Persistence requires a Runtime with `persistent-checkpoints.v1`.

## Enable and write

```json
{
  "checkpoint_persistence": {"interval_seconds": 60},
  "outputs": ["weights/**", "metrics.jsonl", "summary.json"]
}
```

| Variable | Purpose |
|---|---|
| `DATA_DIR` | Optional verified download-script dataset cache |
| `STATE_DIR` | This Attempt's persistent complete recovery generations |
| `OUTPUT_DIR` / `/outputs` | Exported declared results |
| `RESUME_DIR` / `/inputs/resume` | Exact verified prior generation |

`STATE_DIR` is a sibling of this Attempt's outputs within the configured shared
mount. WYD uses shared datapool; SenseCore uses NAS. New Attempts do not overwrite
the restore source. Old frozen paths/images are not migrated by profile changes.

Write all recovery files below `STATE_DIR/checkpoints/UNIQUE_GENERATION/`.
Finish/flush the files, then atomically replace `STATE_DIR/checkpoint.ready.json`:

```json
{
  "step": 4000,
  "files": [
    {"path":"checkpoints/step-004000/state.pt","bytes":12345,"sha256":"ACTUAL_64_HEX_HASH"},
    {"path":"checkpoints/step-004000/config.json","bytes":456,"sha256":"ACTUAL_64_HEX_HASH"}
  ]
}
```

The list must exactly cover one generation of regular files. Links, special
files, traversal and changing bytes are rejected. Never modify/reuse a published
generation. The worker verifies size/SHA/stability and seals permissions 0444/0555.
This is permission protection, not a kernel read-only mount or off-site backup.

## Register and restore

Only metadata is uploaded. The checkpoint ID binds project/Run/Attempt,
source/image/storage context and sorted file hashes. Registration is idempotent
and reports REGISTERED / worker-sha256: the worker verified bytes, not an
independent server read of NAS. The platform exports acknowledged references
and restore provenance in `checkpoints.json`.

```bash
ml-exp checkpoints --project my-study --run training --attempt attempt-001
```

Queries: `GET /api/runs/P/R/attempts/A/checkpoints` and `/checkpoints/CHECKPOINT_ID`.
Create a **new Run** with its exact reference:

```json
{
  "resume_from": {
    "run_id":"training", "attempt_id":"attempt-001",
    "checkpoint_id":"checkpoint.ACTUAL_64_HEX_HASH"
  }
}
```

`resume_from` implicitly enables persistence if absent. The worker rehashes the
actual generation before training and exposes it through RESUME_DIR. Your code
restores all scientific state. Missing/corrupt files block training without a
latest/older fallback. References must belong to the same project/storage scope;
GPU profiles may differ when storage is shared, but WYD/SenseCore are different
scopes. Cross-backend use requires explicit state export/data asset delivery.

`resume_from` is frozen in the Run. Retrying the same Run does **not** automatically
select a checkpoint produced by its failed Attempt. A changed restore reference
currently requires a new Run. Automatic recovery is [not implemented](recovery.md).

## Limits and failures

Backend state bypasses API archive byte limits but remains subject to backend
disk/quota. Metadata is limited to 256 KiB/20,000 files; combined input/preparation/
restore manifests must fit 32 KiB. Keep manifests small enough for this tighter bound.
`checkpoint_upload` is a separate optional full-state backup path, governed by
advertised archive/part/storage limits. With persistence it archives the STATE_DIR
ready generation; without persistence it uses the historical output-ready format.

Registration failures retry while the worker remains alive. No acknowledged
metadata means no registered API reference yet. Result-upload failure does not
delete persistent state, but a REGISTERED checkpoint alone is not a downloadable
archive. See [output recovery](recovery.md#output-recovery). Retain referenced
generations deliberately; no automatic state garbage collection exists.
