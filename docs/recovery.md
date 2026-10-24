# Diagnose and recover

Preserve project, Run, Attempt and any preparation, Runtime, Submission, upload
or delivery ID. Keep timestamps/timezones and sanitized HTTP error bodies.
Recovery observes original identities before making another request with effects.
Never infer permission to resubmit from a timeout or an empty log.

## Identify the failed stage

```bash
ml-exp status --project PROJECT --run RUN --state experiment.state.json
ml-exp status --project PROJECT --run RUN --attempt attempt-001 --details
ml-exp logs --project PROJECT --run RUN --attempt attempt-001
ml-exp metrics --project PROJECT --run RUN --attempt attempt-001
```

| Evidence | Meaning |
| --- | --- |
| Runtime READY | A verified image was published |
| Submission VERIFIED | The exact backend job was observed |
| Scheduler RUNNING | Scheduler state, not evidence that user training started |
| Recent worker step/log | Training liveness evidence at that observation time |
| Checkpoint REGISTERED | Worker verified a complete persistent generation, not an API archive |
| Process exit 0 | Entry process succeeded; publication/scientific validity remain separate |
| AVAILABLE + SHA256_VERIFIED | Result archive is ready for verified client download |
| Download ACK accepted | Exact verified local receipt and retention choice were recorded |

Status/metrics/log reads do not submit, collect, retry or refresh a provider.
`--state` binds observation to saved dependencies and the original Attempt.
Absent/stale evidence remains unknown. Poll timestamps are not training heartbeats;
FAILED/SUSPENDED alone proves neither timeout nor preemption.

For pre-submission failures, read:

```text
GET /api/actions/ACTION_ID/diagnostics
GET /api/actions/ACTION_ID/diagnostics?refresh=true
```

The bounded redacted response can include staging stdout/stderr, exit code,
timeout and image phases. Slurm refresh reads the frozen preparation receipt;
it never reruns conversion, the Action or a project controller. A late remote
READY event is separate evidence and does not rewrite the Action's saved failure.
A new Action still rechecks the actual SIF. Old/missing receipts remain UNKNOWN.

Slurm image preparation occurs before scheduler submission. It distinguishes lock
wait, cache verification, OCI pull/conversion, SIF verification and publication.
The remote limit is 1080 seconds (configurable down to 30), below the 1200-second
Action limit; lock wait is at most 300 seconds. Cache reuse verifies the SIF hash.
See [builds](builds.md) for independent builder/registry budgets.

## Uncertain build, submit or data delivery

```bash
ml-exp runtime --state runtime.state.json --logs
ml-exp runtime --state runtime.state.json --reconcile
ml-exp submission --id SUBMISSION_ID --reconcile
```

Reconcile checks the frozen image receipt, exact job or exact data-copy identity.
It does not rebuild or resubmit. Follow `safe_to_retry`, `resolution` and
`next_action`. A not-yet-published Runtime with an obsolete recipe may need fresh
preparation after effects are reconciled; an existing READY image remains usable.
`--resume` observes saved work, not automatic replacement training.

A lane-capacity `503 EXECUTION_CAPACITY_UNAVAILABLE` admits no new effect for that
request. It does not settle an earlier uncertain request. Keep the state and
original ID. [Client errors](http_contract.md#structured-errors-and-execution-capacity)
preserve bounded status/code/body/retry hints without credentials.

Multipart transport is separately idempotent: GET the same session, resend missing
identical parts and complete the same archive identity. Never discard a complete
upload merely because publication returned 507; resolve storage capacity first.

## Cancel or retry an exact Attempt

```bash
ml-exp cancel --project PROJECT --run RUN --attempt attempt-001 --state cancel.state.json
# Review the returned Action, then explicitly execute.
ml-exp cancel --state cancel.state.json --execute

ml-exp retry --project PROJECT --run RUN --attempt attempt-001 \
  --max-gpu-hours 0.2 --state retry.state.json
ml-exp retry --state retry.state.json --execute
```

Preparation and execution are separate invocations. Each operation keeps its own
state file. Execution rechecks exact scope, digests, gates and the saved budget.
The client writes dispatch markers before HTTP and never automatically replays
an uncertain authorization or execution. Repeat with only `--state` to observe;
use the corresponding `--reconcile` to reconcile the original Action.

A lost preparation reply leaves the original request and recovery URL. Find the
Action with the exact saved intent, then attach it using `--action-id ACTION_ID`.
Attachment is observation/review, not execution. Changed requests need a new
reviewed operation only after the prior outcome is resolved.

Retry creates a new Attempt within the same frozen Run; it does not choose a
checkpoint. To change `resume_from`, create a new Run with an explicit registered
checkpoint reference. Reconcile the prior job before replacement training.

## Output recovery

Training exit, checkpoint persistence and archive publication are independent.
Check the exact Attempt's receipt, file list and collection before using compute.
A published archive can be downloaded without a recovery job.

```bash
ml-exp collect --project PROJECT --run RUN --attempt attempt-001
ml-exp collect --project PROJECT --run RUN --attempt attempt-001 --start
ml-exp download --project PROJECT --run RUN --attempt attempt-001 --out results
```

`GET/POST /api/runs/P/R/attempts/A/collection` separates training, collection and
verification. POST accepts `{}`, `{"retry":true}` or `{"reconcile":true}`. It keeps
the original Attempt, frozen outputs and durable operation; recovery does not
require a new training Run or input-data delivery. The service bounds requests
to three recovery requests, and uncertain CPU creates require exact reconciliation.
Recovering files never changes original FAILED scheduler history to SUCCEEDED.

| Backend | Independent recovery support |
| --- | --- |
| Slurm | Verify terminal job; inventory/reserve capacity; copy the exact shared outputs through SSH |
| SenseCore | Bounded RESERVED debug ACP, 2CPU/4GiB/0GPU, pinned existing Python image; read NAS without running training |
| PAI DLC | Publish already available local outputs; remote CPU/offline recovery is unsupported |

SenseCore's result worker has a 900-second budget and a 1500-second queue/observation
deadline. Temporary archives use its container `/tmp`; NAS outputs and state are
read-only inputs. Missing backend files cannot be recovered by the platform.
Registered generations, archive identity and file hashes are checked before
publication. An existing sealed receipt cannot be replaced by another digest.

Result publication reserves its copy/build/validation/object-storage peak using
the shared capacity policy. Insufficient capacity returns 507 and retains the
original upload. Newly copied caches are evicted only after independent published
object and local file verification. State/status queries never download weights.

```text
GET /api/runs/P/R/attempts/A/collection/diagnostics
GET /api/runs/P/R/attempts/A/collection/diagnostics?refresh=true
```

Publication diagnostics distinguish missing parts from all-parts-present capacity
rejection, including required/observed free bytes. Refresh queries the exact saved
SenseCore CPU job's workers, events and offline logs; it never creates a job.
Provider stdout/stderr can be combined and delayed. Empty accepted logs remain
PENDING; unavailable exit codes stay null. Results cache only for that job identity.

`ML_EXPD_RESULT_RECOVERY_DIAGNOSTIC` reports phase, safe exception class, errno and
optional hashed path reference. It excludes raw paths/errors. NAS ENOSPC, worker
`/tmp` capacity and API result-store capacity are different failures.

## Resume publication or a download acknowledgement

`experiment --download-to` waits for the original Attempt's collection to become
AVAILABLE, SHA256_VERIFIED and downloadable after training is terminal. It never
POSTs collection while waiting. A timeout saves the last snapshot and IDs; resume
observes the same publication. FAILED/RECONCILE_REQUIRED needs explicit inspection.
A released archive is not silently recreated.

After successful download, an ACK error leaves usable local files and
`acknowledgement.status=PENDING` in `verification.json` and saved workflow state.
Resume with the original directory, or run `ml-exp acknowledge --directory results`.
It verifies exact Attempt/archive/file proof and rehashes local bytes before
retrying only ACK. It neither redownloads nor starts collection, and preserves
the saved KEEP/release choice. Changed/corrupt proof prevents acknowledgement.

## Download confirmation and server storage

```bash
# Default: verify locally, then permit immediate archive release.
ml-exp download --project PROJECT --run RUN --attempt attempt-001 --out results
# Retain the server archive instead.
ml-exp download --project PROJECT --run RUN --attempt attempt-001 --out results --keep-server-copy
ml-exp acknowledge --directory results
# Explicitly change retention while the archive still exists.
ml-exp acknowledge --directory results --release-server-copy
```

ACK requires normal API authentication and exact archive SHA/bytes plus every
file's SHA/bytes. The endpoint is
`POST /api/runs/P/R/attempts/A/artifacts/ack`; `release` defaults to true. A GET,
partial transfer or checksum failure never creates this proof.

Accepted release-enabled ACK is immediately eligible for maintenance, with no
24-hour waiting delay. When a frozen durable OSS archive plan is required,
release also waits for its VERIFIED copy. Physical object-store garbage collection
may lag. Separate configured local-payload expiry can affect archives regardless
of ACK; consult current storage policy rather than treating KEEP as a permanent
backup guarantee. Old clients without ACK do not authorize ACK-based release.

Maintenance removes exactly bound published objects and matching verified local
delivery caches; it retains metadata, metrics, receipts and persistent backend
state. Changed files, extra files, symlinks and original worker outputs are not
silently removed by cache cleanup. Released archives return
`410 ARTIFACT_RELEASED`, not missing-upload 404. Local cleanup retries do not
repeat object deletion. Dataset originals, backend checkpoint state and durable
OSS archives have their own lifecycle; see [administrative deletion](resource-deletion.md).

## Restore unfinished training and remaining limits

Choose a complete registered generation and create a new Run with explicit
[resume_from](persistent-checkpoints.md). The worker rehashes files; experiment
code restores optimizer, scheduler, RNG and data cursor. Registration alone does
not prove a successful restore or make state downloadable.

There is no automatic preemption recovery. ACP uses `backoff_limit: 0`; the
collector never launches replacement training. Worker restore, inside-job data,
training and result return share one worker duration. A hard kill can interrupt
publication; offline recovery can only return bytes that were actually persisted.
Detailed diagnostics require the corresponding frozen image capabilities.

Interrupted server preparations become RECONCILE_REQUIRED at startup. Observe
the original Runtime/copy/script dependencies, then explicitly continue once
verified READY. Continuation advances preparation, never GPU authorization.
Changing the selected executor configuration blocks the old frozen selection.

New `launch-diagnostics.v1` images report FETCH_MANIFEST, CREATE_OUTPUT,
START_WORKER and child exit using exact-Attempt capability before writable NAS
is required. Missing best-effort reports prove no particular cause. Exit 78 can
be launcher setup, filesystem failure or a child result. For errno 28, check the
actual mounted path, inode/file-count and quota; freeing API archives is not
proof that NAS or CPFS writes work.
