# Persistent training state and exact checkpoint recovery

Complete recovery state belongs on backend storage, separate from exported
results. It may include model, optimizer, scheduler, RNG and data cursor; the
platform never interprets or unpickles it. Evaluation weights and configuration
are normal output artifacts. Scientific checkpoint selection stays in project
code.

Build a new Dockerfile Runtime advertising `persistent-checkpoints.v1` and create
a Run with:

```json
{
  "checkpoint_persistence": {"interval_seconds": 60},
  "outputs": ["weights/**", "metrics.jsonl", "summary.json"]
}
```

The server adds `checkpoints.json` to exported outputs. Persistence is opt-in;
historical Runtime images and frozen Runs do not change. `resume_from` implicitly
enables persistence with a 60-second interval if not explicitly configured.

## Three separate directories

| Environment variable | Purpose | Lifetime |
| --- | --- | --- |
| `DATA_DIR` | Verified data produced by `data_preparation` | Backend project cache |
| `STATE_DIR` | Complete recovery generations | Backend Run/Attempt directory |
| `OUTPUT_DIR` (`/outputs`) | Selected weights, logs, metrics, small receipts | Exported exact-Attempt archive |
| `RESUME_DIR` (`/inputs/resume`) | One verified, previously registered generation | Read input for the new job |

`STATE_DIR` is `<project-root>/runs/<run>/attempts/<attempt>/state`, a sibling of
`outputs`. New jobs get their own state directory; they do not overwrite their
restore source. Both directories are within the configured persistent mount.
WYD's example profile uses `/datapool`; SenseCore uses its configured NAS at
`/data`. Local `/data` on a WYD node is not automatically equivalent to the
cluster shared filesystem. Operators must check mount accessibility on every
target node. Profile changes affect new Runs; old Runs keep their frozen paths.

## Write a complete immutable generation

Application code writes **all** required recovery files beneath
`STATE_DIR/checkpoints/<generation>/`. Each generation name is unique within its
Attempt. Finish writing files, flush/fsync them and their containing directories
as needed by the filesystem, then atomically replace
`STATE_DIR/checkpoint.ready.json` with:

```json
{
  "step": 4000,
  "files": [
    {"path": "checkpoints/step-004000/state.pt", "bytes": 12345, "sha256": "ACTUAL_64_HEX_SHA256"},
    {"path": "checkpoints/step-004000/config.json", "bytes": 456, "sha256": "ACTUAL_64_HEX_SHA256"}
  ]
}
```

The file list must cover the entire generation; paths must be regular files in
one generation, without links, devices or traversal. `step` is a nonnegative
integer. A partial generation without a ready marker is never registered. Do not
modify, append files to, or reuse a published generation. A newer checkpoint uses
a new generation and a new ready marker.

The worker verifies every file's size and streaming SHA256, rejects changes
during verification, and sets generation files/directories to 0444/0555. This is
permission protection, not a kernel-enforced read-only mount; root code can
change permissions. Future restore always checks the actual files again. Durable
storage behavior under host/storage power loss remains the backend's guarantee;
NAS/shared storage is not an off-site backup.

## Registration and recovery over the API

Only the small manifest is transmitted to ML-Expd. The server derives storage
location, storage scope and source/image provenance from the exact Attempt's
frozen definition and scoped capability; the worker cannot submit an arbitrary
storage path. `checkpoint_id` hashes that context and the sorted manifest.
Repeated registration is idempotent; a generation cannot acquire a different
manifest. The receipt states `REGISTERED` / `verification: worker-sha256`, meaning
the worker checked the bytes. The API host does not claim to have independently
read SenseCore NAS.

```bash
ml-exp checkpoints --project my-project --run training --attempt attempt-001
```

HTTP queries:

- `GET /api/runs/P/R/attempts/A/checkpoints`
- `GET /api/runs/P/R/attempts/A/checkpoints/CHECKPOINT_ID`

Create a new Run with the chosen exact reference:

```json
{
  "resume_from": {
    "run_id": "training",
    "attempt_id": "attempt-001",
    "checkpoint_id": "checkpoint.ACTUAL_64_HEX_SHA256"
  }
}
```

Or add `--resume-from '{"run_id":"training","attempt_id":"attempt-001","checkpoint_id":"checkpoint.…"}'`
to `ml-exp create`. Single-config `ml-exp experiment` also accepts
`checkpoint_persistence` and `resume_from`.

The new Run freezes the receipt and restore provenance, including in its metric
protocol when one is declared. Before training, the worker verifies the identity
and every actual file, then exposes just that generation at `/inputs/resume`.
Application code loads `RESUME_DIR/state.pt`, etc. Missing, changed, incomplete or
inaccessible files prevent training. There is no automatic "latest checkpoint"
fallback, automatic code compatibility inference, or automatic scheduler replay.

References stay within the project and the same configured storage scope. A GPU
profile change is allowed when both profiles use the same storage namespace;
WYD and SenseCore are different scopes. `/inputs/resume` cannot simultaneously
be an uploaded asset mount.

## Uploads and failures

Backend-resident state has no 4 GiB API byte limit: backend free space/quota still
apply. Metadata is bounded at 256 KiB / 20,000 files; the combined frozen
input/preparation/restore manifests must fit the 32 KiB scheduler command budget.
Source, uploaded data assets, final artifacts and uploaded checkpoint archives
retain their advertised deployment limits. Desktop-staged input assets have
no default byte cap; final outputs and uploaded checkpoint archives may still
have private operator quotas. Read `/api/health` before packing.

`checkpoint_upload` remains a separate, explicit backup option. With persistence
enabled it archives the complete generation described by the **state** ready
marker; without persistence its historical `/outputs` behavior stays unchanged.
Registration does not wait for backup success. Backups exceeding the archive
limit fail; they do not invalidate a separately registered local checkpoint.
For cross-backend recovery, explicitly export/upload a complete state asset and
use the existing `inputs` route. This feature does not implement larger uploaded
archives or cross-backend storage copying.

Registration failures are logged and retried while the worker runs; bytes remain
on backend storage. Until a metadata registration is acknowledged, clients have
no registered API reference to it. `checkpoints.json` reports acknowledged
checkpoints and the restore source; a checkpoint can be recoverable even if final
artifact upload fails. Uploaded-result failure still remains a distinct worker
failure. No automatic state garbage collection is enabled: operators must retain
referenced generations and manage backend quotas deliberately.
