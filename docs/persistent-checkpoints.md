# Persistent checkpoints and restore

Complete recovery state can include model, optimizer, scheduler, RNG and data
cursor. Keep it on backend storage; export selected weights/logs as artifacts.
The platform hashes files but never interprets or unpickles training state.

## Enable and write

| Variable | Purpose |
|---|---|
| `DATA_DIR` | Optional verified download-script dataset cache |
| `STATE_DIR` | This Attempt's persistent complete recovery generations |
| `OUTPUT_DIR` / `/outputs` | Exported declared results |
| `RESUME_DIR` / `/inputs/resume` | Exact verified prior generation |

`STATE_DIR` is a sibling of this Attempt's outputs within the configured shared
mount: Slurm uses shared storage, SenseCore uses NAS, and PAI DLC uses its
approved personal CPFS/NAS workspace. New Attempts do not overwrite the restore
source. Old frozen paths/images are not migrated by profile changes.

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
ml-exp run checkpoints --project my-study --run training --attempt attempt-001
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

The worker rehashes the
actual generation before training and exposes it through RESUME_DIR. Your code
restores all scientific state. Missing/corrupt files block training without a
latest/older fallback. References must belong to the same project/storage scope;
GPU profiles may differ when storage is shared, but Slurm, SenseCore and DLC
are different scopes. DLC scope also binds the approved user and storage
endpoint. Cross-backend use requires explicit state export/data asset delivery.

The initial `resume_from` is frozen in the Run. Native cloud retry can explicitly
select a checkpoint produced by a terminal Attempt of that same Run:

```bash
ml-exp run retry ACTION --run training --attempt attempt-001 \
  --checkpoint checkpoint.ACTUAL_64_HEX_HASH --max-gpu-hours 0.5
```

This creates the next Attempt with its selected restore reference, leaving the
original Run and Attempt unchanged. It does not submit training or choose the
latest checkpoint automatically. The parent can have completed successfully;
its checkpoint must match the same source, image and storage scope. Changed
scientific configuration requires a new Run. See [recovery](recovery.md).

## Limits and failures

Backend state bypasses API archive byte limits but remains subject to backend
disk/quota. Checkpoint metadata is limited to 256 KiB/20,000 files. The complete native
worker package, including input definitions and restore metadata, must fit
64 KiB; its serialized size determines whether preparation can proceed.
`checkpoint_upload` is a separate optional full-state backup path on supported
Slurm/SenseCore executors, governed by archive/part/storage limits. With persistence
it archives the STATE_DIR ready generation; without persistence it uses the
historical output-ready format. **DLC supports persistent metadata registration
and restore, but rejects full `checkpoint_upload`.** Export selected weights as
normal outputs instead; do not infer support from a worker capability alone.

Registration failures retry while the worker remains alive. No acknowledged
metadata means no registered API reference yet. Result-upload failure does not
delete persistent state, but a REGISTERED checkpoint alone is not a downloadable
archive. See [output recovery](recovery.md#output-recovery). Retain referenced
generations deliberately; no automatic state garbage collection exists.
On supported Slurm/SenseCore backends an administrator can explicitly
[delete a completed managed Run](resource-deletion.md), including its persistent
generations. Frozen restore consumers block deletion. DLC remote Run deletion
is unsupported; shared uploaded snapshot assets have a separate lifecycle.

Checkpoint/snapshot queries read registered metadata and validate its frozen
Run/Attempt identity without downloading or expanding result archives. A released
archive or unavailable object store does not prevent metadata queries. Retrieving
actual files still requires a supported file/archive export or backend access.
