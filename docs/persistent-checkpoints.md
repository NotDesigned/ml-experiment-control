# Persistent checkpoints and restore

Complete recovery state can include model, optimizer, scheduler, RNG and data
cursor. Keep it on backend storage in `STATE_DIR`. Write what you want to take
away, such as selected weights, into `OUTPUT_DIR`. The platform hashes files but
never interprets or unpickles training state.

## Enable and write

These variables are the same on Slurm, SenseCore and PAI DLC:

| Variable | Purpose |
|---|---|
| `DATA_DIR` | This Project's data directory on backend storage, or with `data_preparation` the verified tree that script wrote. Do not create it on a cloud backend with no dataset yet. See [data](data.md#paths-in-the-job) |
| `DATASETS_JSON` | Declared `data` inputs: each name maps to its verified directory. Set only when the Run declares `data` |
| `STATE_DIR` | This Attempt's persistent complete recovery generations |
| `OUTPUT_DIR` | Results. Everything below it is archived on collection |
| `METRICS_PATH` | [Metric records](metrics.md), at `OUTPUT_DIR/metrics.jsonl` |
| `RESUME_DIR` | Exact verified prior generation. Set only on a restoring Attempt |
| `RESUME_CHECKPOINT_ID`, `RESUME_STEP` | That generation's checkpoint ID and step. Set only on a restoring Attempt |

The platform owns these names, so a Run's `env` cannot set them.

`STATE_DIR` and `OUTPUT_DIR` are the sibling directories `state` and `outputs` of
one Attempt directory within the configured shared mount: Slurm uses shared
storage, SenseCore uses NAS, and PAI DLC uses its approved personal CPFS/NAS
workspace. New Attempts do not overwrite the restore source. A profile change
does not move a Run's frozen paths or image.

A result archive holds everything below `OUTPUT_DIR`, including a directory you
happen to call `checkpoints` or `outputs` and any hidden file, and nothing below
`STATE_DIR`. See [what an archive can leave out](recovery.md#what-an-archive-can-leave-out).

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
generation. The worker verifies size/SHA/stability and seals files 0400 and directories
0500. This is permission protection, which does not stop root, not a kernel read-only
mount or off-site backup.
On cloud backends the worker does this once for each ready marker and does not hash
the same marker again, so damage after sealing is found when a restore hashes the
generation, not by a later report.

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
8 MiB; its serialized size determines whether preparation can proceed.
The platform exports nothing from `STATE_DIR`. To take recovery state off backend
storage, write a copy into `OUTPUT_DIR`, where archive and part limits apply.

Registration failures retry while the worker remains alive. No acknowledged
metadata means no registered API reference yet. Result-upload failure does not
delete persistent state, but a REGISTERED checkpoint alone is not a downloadable
archive. See [output recovery](recovery.md#output-recovery). Retain referenced
generations deliberately; no automatic state garbage collection exists.
An administrator can explicitly [remove](resource-deletion.md) a terminal Attempt's
`outputs`, its `state` or one generation, on Slurm, SenseCore and PAI DLC. On the cloud
backends this runs as a zero-GPU job and needs the backend's `collector`. A generation
that a frozen `resume_from` names blocks the plan. Real provider cleanup has not been
accepted on any backend. Dataset assets have a separate lifecycle.

Checkpoint/snapshot queries read registered metadata and validate its frozen
Run/Attempt identity without downloading or expanding result archives. A released
archive or unavailable object store does not prevent metadata queries. Retrieving
actual files still requires a supported file/archive export or backend access.
