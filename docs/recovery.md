# Diagnose and recover

Preserve project, Run, Attempt and any preparation, Runtime, Submission, upload
or delivery ID. Keep timestamps/timezones and sanitized HTTP error bodies.
Recovery observes original identities before making another request with effects.
Never infer permission to resubmit from a timeout or an empty log.

## Identify the failed stage

```bash
ml-exp run status --state experiment.state.json
ml-exp run status --state experiment.state.json --watch --seconds 60
ml-exp run status --state experiment.state.json --details --json
ml-exp run logs --state experiment.state.json
ml-exp run metrics --state experiment.state.json
```

`run status` defaults to a short human-readable summary; `--json` keeps its machine
interface. Use `--details --json` for all existing evidence; `--details` alone
does not expand the text summary. Other low-level reads retain JSON output.
Saved state supplies the server and exact identities without local source or
repeated project/Run/Attempt arguments. Without state, supply these identities
explicitly. Before an Attempt is bound, only status can inspect preparation.

| Evidence | Meaning |
| --- | --- |
| Runtime READY | A verified image was published |
| Submission VERIFIED | The exact backend job was observed |
| Scheduler RUNNING | Scheduler state, not evidence that user training started |
| Recent worker step/log | Training liveness evidence at that observation time |
| Checkpoint REGISTERED | Worker verified a complete persistent generation, not an API archive |
| Checkpoint in `archives` | OSS's size and CRC-64 of every part matched the worker's, and the worker's file SHA256 values the registered ones; see [state purge](#checkpoint-copies-and-the-state-purge) |
| Process exit 0 | Entry process succeeded; publication/scientific validity remain separate |
| AVAILABLE + SHA256_VERIFIED | Result archive is ready for verified client download |
| Download ACK accepted | Exact verified local receipt and retention choice were recorded |

A worker's captured log is one file of at most 8 MiB. Past that it keeps its first
4 MiB, the line `[ml-expd: output dropped here; this log keeps its start and its
end]`, and the last 2 to 4 MiB of output. Script preparation logs work the same
way; the Slurm gateway's copy of them has a 1 MiB limit. The end is written to the
file as the job runs, so an OOM or a preemption leaves it there; a node lost
outright can lose the last few seconds. This is a marker, not a failure.
`stdout_dropped_bytes` and `stderr_dropped_bytes` in the process evidence count
what went, and the log hashes describe the file as kept. A worker killed before
its receipt leaves the seam line and no count. A log changed other than through
its pipe still fails with `LOG_IDENTITY_OR_CONTENT_CHANGED`, and the launch command
then exits non-zero, on Slurm and cloud alike.

`ml-exp run diagnose --state experiment.state.json` is the explicit exception for
PAI: it asks the provider for the job's `ReasonCode`, `ReasonMessage`, pods, events
and a bounded log tail (`--lines`, up to 2000). Use it when status and `run logs` are
empty because the job failed before a worker report, such as during image pull or
input delivery. It is read-only at the provider and stores nothing; provider logs
are redacted but remain user text, and a reason does not establish the root cause.

Status/metrics/log reads do not submit, collect, retry or refresh a provider.
`--state` binds observation to saved dependencies and the original Attempt.
`status --watch --seconds N` polls that Attempt for a bounded client interval;
it does not switch to a later Attempt or authorize a restart.
The interval is checked between polls; an in-flight HTTP read keeps its own
request timeout and can finish after the polling interval.
Absent/stale evidence remains unknown. Poll timestamps are not training heartbeats;
the last reported worker phase retains its own observation time and age.
FAILED/SUSPENDED alone proves neither timeout nor preemption.

For pre-submission failures, read:

```text
GET /api/actions/ACTION_ID/diagnostics
GET /api/actions/ACTION_ID/diagnostics?refresh=true
```

The bounded redacted response can include staging stdout/stderr, exit code,
timeout and image phases. Slurm refresh reads the frozen preparation receipt;
it never reruns conversion or the Action. A late remote
READY event is separate evidence and does not rewrite the Action's saved failure.
A new Action still rechecks the actual SIF. A missing receipt remains UNKNOWN.

Slurm image preparation occurs before scheduler submission. It distinguishes lock
wait, cache verification, OCI pull/conversion, SIF verification and publication.
The remote limit is 1080 seconds (configurable down to 30), below the 1200-second
Action limit; lock wait is at most 300 seconds. Cache reuse verifies the SIF hash.
See [builds](builds.md) for independent builder/registry budgets.

## Uncertain build, submit or data delivery

```bash
ml-exp env show RUNTIME --project PROJECT
ml-exp env progress RUNTIME --project PROJECT
ml-exp env logs RUNTIME --project PROJECT
ml-exp env reconcile RUNTIME --project PROJECT
ml-exp run submission ACTION
ml-exp run reconcile ACTION
```

Reconcile checks the frozen image receipt, exact job or exact data-copy identity.
It does not rebuild or resubmit. Follow `safe_to_retry`, `resolution` and
`next_action`. A not-yet-published Runtime with an obsolete recipe may need fresh
preparation after effects are reconciled; an existing READY image remains usable.

A lane-capacity `503 EXECUTION_CAPACITY_UNAVAILABLE` admits no new effect for that
request. It does not settle an earlier uncertain request. Keep the state and
original ID. [Client errors](http_contract.md#structured-errors-and-execution-capacity)
preserve bounded status/code/body/retry hints without credentials.

Multipart transport is separately idempotent: GET the same session, resend missing
identical parts and complete the same archive identity. Never discard a complete
upload merely because publication returned 507; resolve storage capacity first.

### Interrupted dataset publication

Repeat the same `data upload`, or let the next Run's import reach `Complete`. Nothing
needs clearing first.

| What was left | What the next `Complete` does |
| --- | --- |
| A tree with no receipt that passes verification | Publishes it. Only the receipt was lost. |
| A tree with no receipt that fails verification | Proves the parts against the archive hash, removes the tree and rebuilds. If the parts do not prove the archive, it fails and the tree stays. The parts stay until READY is saved. |
| A tree with a receipt that fails verification | Fails, every time, and changes nothing. A published tree is never removed or rebuilt, and its parts are gone. Clearing it is an operator decision. |

Script [data preparation](data.md) caches do not get this. A script is arbitrary code
and may have had effects outside its tree, so running it again would replay an
uncertain execution. Change its arguments or environment to get a new plan.

On Slurm a `Complete` has 1200 seconds over SSH. Past that the API call fails, and the
helper keeps going: it ignores the hangup from the lost session and holds the asset's
lock until it finishes. A repeat `data upload` during that time fails after 10 seconds
of waiting on the lock, and succeeds once the helper has saved READY. If the helper
really died, the repeat finds what it left and rebuilds. A dataset that needs longer
than 1200 seconds therefore takes more than one repeat, not a bigger limit. TERM or INT
still cancels the helper, and it removes its own tree.

Each asset has its own lock, `PROJECT/ASSET.lock` beside its directory, so a long import
does not hold up another asset. Runs of the same asset wait on each other only while
its tree is being built; hashing a tree that already has a receipt holds no lock. Two
assets publishing at once can both pass the capacity check and then run out of room,
which fails the job and removes the tree it was building.

This relies on flock excluding across nodes on CPFS and NAS, which has not been tested.
If it does not, the result is a failed job and not bad data: a tree reaches READY only
after its bytes are hashed against the definition and it is confirmed complete at its
final name, and every Run hashes it again at job start. The one state an operator has
to clear is READY with a damaged or missing tree.

## Cancel or retry an exact Attempt

Every execution of a Run is an **Attempt**: the first is `attempt-001`, and each
retry adds the next one (`attempt-002`, ...). The platform keeps them apart: each has
its own scheduler job, `RUN/attempt-NNN/` storage, checkpoints, results, W&B/MLflow
run and purge. Retries are rare, so commands act on the Run and its **current**
Attempt by default, and human output names an Attempt only once a Run has more than
one. `--attempt attempt-NNN` reads or acts on an earlier one, and `--json` always
carries `attempt_id`. A confirmation that removes or changes recorded data, such as
`PURGE P/R/attempt-001`, still names the Attempt; a wrong one prints the exact
string.

```bash
ml-exp cancel --run RUN
# Continue the Run from a checkpoint it registered; without --checkpoint, from its start.
ml-exp retry --run RUN --checkpoint checkpoint.ACTUAL_HASH
```

The same steps one at a time, with the original native Submission (`ACTION`):

```bash
ml-exp run cancel ACTION --confirmation 'CANCEL ACTION'
ml-exp run retry ACTION --run RUN --checkpoint checkpoint.ACTUAL_HASH \
  --max-gpu-hours 0.5 --reason 'Resume this checkpoint'
```

Retry checks that the exact parent job is terminal, including SUCCEEDED, and
creates the next Attempt's metadata. It does not submit training. The checkpoint
must be one the retried Attempt registered, in the same project, source, image and
backend storage scope. Its Attempt need not be given: the server finds it in the
checkpoint registry and freezes the complete reference, the same one, with the same
request identity, as when `--attempt` names it. A named `--attempt` (the source
Attempt, not the new one) must match, and a checkpoint of another Attempt is refused
with `RETRY_SOURCE_NOT_THIS_ATTEMPT`. The new Attempt uses this explicit checkpoint;
it need not reuse the original Run's initial `resume_from`. No latest-checkpoint
selection is performed.

To restart without a checkpoint, replace `--checkpoint ...` with `--from-start`.
The Run's other scientific configuration remains fixed. Changed source, image,
parameters or resources require a new Run.

Review the created Attempt, then prepare and submit it explicitly:

```bash
ml-exp run prepare --project PROJECT --run RUN --max-gpu-hours 0.5
# Use the NEW_ACTION and exact confirmation returned by preparation.
ml-exp run execute NEW_ACTION --confirmation 'EXECUTE NEW_ACTION' --reason 'Resume reviewed Attempt'
```

A timeout does not establish that submission failed. Read/reconcile the same
Action before another request. Repeating the same retry request reuses the
already-created Attempt; it never allocates another Attempt or submits a job.

## Output recovery

### Native CPU sync recovery

For native PAI submissions, inspect an existing CPU sync without
refreshing the provider or creating a job:

```bash
ml-exp run sync ACTION --request-id ORIGINAL_32_HEX_REQUEST --status
```

The GET `/api/submissions/ACTION/sync/REQUEST` reads retained original/retry
evidence. If the original CPU job is confirmed FAILED and training is terminal,
one explicit retry is supported under the same request:

```bash
ml-exp run sync ACTION --request-id ORIGINAL_32_HEX_REQUEST \
  --retry-failed-job ORIGINAL_FAILED_CPU_JOB \
  --confirmation 'RETRY SYNC ACTION ORIGINAL_32_HEX_REQUEST ORIGINAL_FAILED_CPU_JOB'
```

This POST targets `/api/submissions/ACTION/sync/REQUEST/retry`. It refreshes the
exact parent/failed job evidence, then persists one deterministic CPU child
before effects. The child keeps the original frozen image, worker, full `--sync`
command and output directory, with 0 GPUs and the backend's `result_job` limits.
It does not replay training. The failed CPU job and its outbox remain intact.
Repeating the same request after a lost reply or restart observes the same
child. Unknown creates are reconciled, never blindly repeated. A second failed
child is preserved; no further child is automatically allocated.

### Collect and download

Training exit, checkpoint persistence and archive publication are independent.
Read the saved native Submission before collecting its existing outputs:

```bash
ml-exp run artifacts ACTION
ml-exp run files ACTION
ml-exp run collect ACTION
```

Slurm collects through its configured SSH/shared-storage path. PAI
can create a zero-GPU CPU synchronization job under a retained request identity;
use `run sync` above when you need to retain that identity explicitly. Collection
does not rerun training or require a replacement training Run. Missing backend
files cannot be reconstructed by collecting, and original FAILED scheduler
history stays FAILED. Delayed offline logs remain pending, not proof of success.

The CPU job takes the backend's `result_job` limits, by default 2 CPUs, 4 GiB and
6 hours. Set it in the backend's entry in `cloud_config`, for example
`"result_job": {"max_time_seconds": 43200}`, with any of `cpus`, `memory_gb` and
`max_time_seconds`. PAI allows up to its `collector` capacity and 24 hours. Storage maintenance jobs use the same limits. An interrupted archive restarts
from nothing, so the time limit must cover packing, hashing and uploading the whole
result. A job keeps the limits it was first recorded with: to give more time after a
limit was hit, raise it and start a new `run sync --request-id`, or retry a failed one.

### What an archive can leave out

A finished job's results leave even if some paths cannot. These are skipped and
recorded, and the archive holds everything else:

| `reason` | Skipped entry |
| --- | --- |
| `symbolic_link` | A link. It is never followed, so its target is not archived |
| `special_file` | A FIFO, socket or device |
| `unsupported_path` | A name that is not UTF-8 or holds a control character or backslash, a path over 4096 bytes, or a directory nested over 128 levels |
| `file_limit` | Anything past the 20,000 entries a receipt can list, in sort order. Long paths can reach the limit sooner, because the saved record has a fixed size |

`run files` reports the list as `skipped`, `run download` as `skipped` and `run file`
as `archive_skipped`. **This list is the only record of what is missing.** Nothing else
compares the archive with the original directory, and the file hashes prove only
the files that are listed. Read it before releasing an archive or removing `outputs`.
A row naming a directory covers what lies beneath it, and `.` is `OUTPUT_DIR` itself.
It holds up to 1024 rows; `unlisted` counts the entries beyond those. Files listed in
`artifacts.required_files` are archived whatever the limit, and a required file that
is absent, or is a link, is the one thing that fails the result. Paths there are
relative to `OUTPUT_DIR`, so one starting with `outputs/` is refused when the Run is
created. An empty directory is not recorded. A failed read, or a file that changes
while it is packed, fails the archive so that it can be collected again.

## Download confirmation and server storage

```bash
# Download a tar, verify its inventory and hashes, then permit archive release.
ml-exp run download ACTION --output results.tar
# Retain the generated server archive instead.
ml-exp run download ACTION --output results.tar --keep
# One file: verify it without ACK or archive release.
ml-exp run file ACTION --path models/final.pt --output final.pt
```

If ACK fails after the tar was saved, repeat `run download` with the same Action,
output file and retention choice. The client re-verifies the complete existing
tar and sends its proof without redownloading or overwriting it. Changed bytes
prevent ACK. The native endpoint is `POST /api/submissions/ACTION/acknowledge`;
GET, partial download or a failed checksum never authorizes release.

Release-enabled ACK permits immediate removal of the generated transfer
archive; there is no 24-hour delay. It does not remove backend checkpoints or their
OSS copies,
original outputs or dataset assets. Physical object-store cleanup can lag. The
platform has no timer that expires an archive, but `--keep` is not a permanent
backup guarantee: a lifecycle rule on the bucket would act outside the platform.
Released archives return `410 ARTIFACT_RELEASED`. Redownload
requires an explicit new export from retained backend outputs, not training.
See [administrative deletion](resource-deletion.md) for separate state removal.

## Restore unfinished training and remaining limits

Choose a complete registered generation. Use the native retry command above
for another Attempt of the same Run, or create a new Run with explicit
[resume_from](persistent-checkpoints.md) when its configuration changes. The worker rehashes files; experiment
code restores optimizer, scheduler, RNG and data cursor. Registration alone does
not prove a successful restore or make state downloadable.

There is no automatic preemption recovery. The
collector never launches replacement training. Worker restore, inside-job data,
training and result return share one worker duration. A hard kill can interrupt
publication; offline recovery can only return bytes that were actually persisted.

Interrupted server preparations become RECONCILE_REQUIRED at startup. Observe
the original Runtime/copy/script dependencies, then explicitly continue once
verified READY. Continuation advances preparation, never GPU authorization.
Changing the selected executor configuration blocks the old frozen selection.

A job that fails during launch can leave no report, and that absence proves no
particular cause. For errno 28, check the
actual mounted path, inode/file-count and quota; freeing API archives is not
proof that NAS or CPFS writes work.


## Checkpoint copies and the state purge

On PAI DLC a Run that copies checkpoints keeps each registered checkpoint on OSS, and
the verified copy is its authority; see
[object-store copies](persistent-checkpoints.md#object-store-copies-on-pai-dlc). A
retry with `--checkpoint`, or a new Run with `resume_from`, restores from the original
Attempt's CPFS files while they are there and from the verified copy after its `state/`
is purged. The restore fails, and training does not start, when neither holds the
registered bytes.

An ended Attempt's `state/` is removed from CPFS when all of these hold:

| Condition | `blocked_by` when it does not |
|---|---|
| The Run copies checkpoints | `CHECKPOINT_ARCHIVE_NOT_ENABLED` |
| The scheduler job is proven to have ended | `ATTEMPT_NOT_TERMINAL` |
| No sync job of the Attempt is still running or unobserved | `SYNC_IN_PROGRESS` |
| Its worker reported its end | `WORKER_NOT_FINISHED` |
| With `PURGE UNFINISHED` only: a sync delivered the state after training ended | `SYNC_REQUIRED` |
| The last ready marker verified | `LAST_CHECKPOINT_UNVERIFIED` |
| A completed training's results are published | `RESULTS_NOT_PUBLISHED` |
| Every registered checkpoint has a verified OSS copy | `CHECKPOINT_NOT_ARCHIVED` |

The server checks these about once a minute and starts the purge
`purge_delay_seconds` (default 900) after it first saw all of them hold; 0 starts it at
once. When one stops holding before then, the delay starts again when it holds. The
automatic purge is on by default and removes everything in `state/`, including files
that are not registered checkpoints, such as snapshots. The purge is the reviewed `state` maintenance plan run as a zero-GPU CPU job on the
backend's `collector`, with the Attempt's own worker and capability; it is bounded to
that Attempt's `state/`, deletes through the directory it inspected and is idempotent.
`outputs/`, checkpoint records and their OSS copies stay. Its plan, trigger, times, the
checkpoints it relied on and the job's receipt are kept beside the Attempt and shown by
`ml-exp run checkpoints` as `state_purge`. The plan's identity decides the job, so a
restart or a repeated request observes the same job. The record is written before the
job is asked for; each pass asks again for a job that was never sent and observes one
that was. After a job proven to have failed, for instance one whose receipt did not
arrive, a further job runs the same plan: it resumes the targets the first one recorded,
or returns the receipt it already has. The automatic purge runs at most 3 jobs per plan,
the explicit command up to 10.

To purge now, without the delay:

```bash
ml-exp run purge-state --project my-study --run training \
  --confirmation 'PURGE my-study/training/attempt-001'
```

Without `--attempt` it purges the Run's current Attempt; a wrong confirmation prints
the exact one.

It answers `BLOCKED` with `blocked_by` and changes nothing while a condition fails,
`PURGING` once the job is asked for, and `PURGED` with the receipt. Repeat it to observe.
A worker that was preempted never reports its end, and it may have written a generation
it never registered. Run `ml-exp run sync` first. Its job sends two packets built from
storage as it is now, which registers the last ready marker, copies every registered
checkpoint, and then tells the server so; the server keeps that only once the training
job has ended. Then confirm with `'PURGE UNFINISHED my-study/training/attempt-001'` to
accept that anything still unregistered is removed. The automatic purge never purges an
unfinished worker's state. The endpoint is
`POST /api/runs/P/R/attempts/A/purge-state` with `{"confirmation": ...}`.

A registered checkpoint whose CPFS files were removed before its copy was made cannot
be copied. Remove it with `admin plan --kind checkpoint` ([deletion](resource-deletion.md))
to let the purge proceed. Runs that do not copy checkpoints, including all Runs created
before this release, are never purged; use [administrative deletion](resource-deletion.md)
for them. Slurm keeps its shared storage and has no purge.

## Read-only collector diagnostics

For an existing terminal PAI synchronization job:

```bash
ml-exp run sync ACTION --request-id ORIGINAL_REQUEST --diagnose
ml-exp run sync ACTION --request-id ORIGINAL_REQUEST --diagnose --status
```

The first command creates or observes one deterministic 0GPU/2CPU/4GiB/300s
job under the original synchronization lineage. It keeps the original image,
mounts personal CPFS read-only, and never runs the worker, trains, packages,
uploads, removes a file or creates a missing lock. It reads bounded Attempt
metadata, checks required output file sizes without reading weights, probes
existing locks without blocking, and performs only a scoped GET of accepted
exchange/artifact stages. Authentication contents and raw exception text are
not returned. The original failed collector and any explicit retry remain intact.

The `--status` form reads retained evidence only. Offline logs may arrive later;
repeat the first command to observe the same diagnostic handle. A pending or
failed observation does not start another diagnostic or collector. The scoped
worker GET endpoint cannot authenticate ordinary client or administrative calls.
See [dataset diagnostics](data.md) for failed PAI cache preparation.
