# Diagnose and recover without repeating training

## Identify the failed stage

Save project, Run, Attempt, Runtime, Submission, upload and delivery IDs as
applicable, plus timestamp/timezone and sanitized HTTP error body. Do not include
tokens, launch commands or signed URLs. Query exact IDs, not an arbitrary latest.

| Evidence | What it proves | Next check |
|---|---|---|
| Runtime READY | Verified image was published | Run/submission gates |
| Submission VERIFIED | Exact backend job was observed | Scheduler/worker/training evidence |
| Scheduler RUNNING | Scheduler lifecycle state | Data readiness, child process, fresh logs/step |
| Checkpoint REGISTERED | Worker verified one complete persistent generation | Actual restore validation or export |
| Artifact receipt / COMPLETE upload | Complete result archive published | Signed download and file hashes |
| Archive 404 | No published archive receipt; files may already exist | Exact-Attempt files and collection status |

Polling time is not a process heartbeat. Submission preparation history is not
current training progress. Preserve each evidence layer's Attempt binding,
observation time and stale status. Queue reasons can be reported; reliable ETA
may remain null. Unknown exit cause must not be called preemption or timeout.

## Uncertain build, submit or data delivery

```bash
ml-exp runtime --state runtime.state.json --logs
ml-exp runtime --state runtime.state.json --reconcile
ml-exp submission --id SUBMISSION_ID --reconcile
```

Reconcile uses the frozen receipt/exact scheduler identity. It does not rebuild
or resubmit. Follow returned next_action/safe_to_retry and inspect actual effects
before explicitly preparing another operation. Old unbuilt recipes and stale
unexecuted gates need fresh preparation. Completed READY images remain usable.
Client `--resume` observes saved identities; it is not automatic GPU recovery.

Archive transfer is independently resumable. Confirm committed parts first;
retry missing identical bytes and completion. Preserve incomplete sessions and
their expiry/whole-archive hash when diagnosing a released worker.

## Output recovery

Training completion, checkpoint persistence and result publication are separate.
A worker can finish training/register final state, then disappear during upload.
This leaves NAS bytes and an incomplete upload but no downloadable archive.

First check whether the exact Attempt already has a receipt. If so, download and
verify it without any compute job. If not, an operator can use a bounded zero-GPU
task on storage accessible to that backend to read the exact registered generation
and declared outputs, verify hashes, reconstruct the archive and publish through
the original scoped transfer. Identical archive identity can resume prior parts.
Never overwrite a sealed receipt with another digest or rewrite FAILED history
as SUCCEEDED merely because results were recovered.

Use the original Attempt's collection endpoint; no new training Run or input-data
delivery is required:

```bash
ml-exp collect --project PROJECT --run RUN --attempt attempt-001
ml-exp collect --project PROJECT --run RUN --attempt attempt-001 --start
# Repeat the query above to observe completion; then use ordinary download.
ml-exp download --project PROJECT --run RUN --attempt attempt-001 --out results
```

`GET/POST /api/runs/P/R/attempts/A/collection` exposes training, recovery and
archive verification separately. POST accepts `{}`, `{"retry":true}` or
`{"reconcile":true}`. Duplicate requests observe the same durable operation.
An uncertain create requires reconciliation of its exact CPU job; it is never
silently resubmitted. Retries retain the original Run/Attempt and stop after three
recovery requests. Original scheduler FAILED history remains unchanged.

WYD reads the frozen shared output directory through its configured SSH endpoint.
SenseCore uses the configured debug pool with RESERVED quota, a verified
2CPU/4GiB/0GPU spec and an existing pinned Python image. The copy worker is bounded
to 900 seconds, with a 1500-second queue/observation deadline; it never runs the
training entrypoint. Registered checkpoints and unfinished archive identities
are checked before publishing. Recoveries run serially and do not retain another
expanded output cache after publication. File listings use receipt metadata
without restoring whole archives; normal signed downloads go to object storage.
Artifact completion reserves space for validation and the published object in
addition to already saved parts: object storage can share the API host's disk.
Insufficient space still rejects completion and preserves the session.

New workers persist and register `training-result.v1` before result upload.
When a terminal Attempt has a successful process result but no archive, the
collector requests one independent recovery. Failed recoveries require explicit
retry/reconcile. Old workers have no such evidence and are not automatically
enrolled; their training status is UNKNOWN until interpreted by the experiment's
own protocol. These are process and transport statuses, not scientific success
criteria. Existing collected files can also be downloaded without an archive,
provided the complete file list includes SHA256 values and every download matches.

## Restore unfinished training

Choose a registered complete checkpoint from the exact source Attempt and create
a new Run with `resume_from` ([checkpoint contract](persistent-checkpoints.md)).
Code owns optimizer/RNG/data-cursor restoration; the worker verifies files before
starting it. Reconcile the prior job first if its outcome is uncertain. Never
create a replacement while the old job might still run.

## Known limitations

Remaining limits:

- SenseCore offline queries are bounded to 200 provider records and report
  truncation. Accepted requests can still return no hits, which does not establish
  an exit cause.
- Worker preparation, restore, training and artifact return share one duration.
  A hard kill can interrupt return; independent recovery handles saved outputs,
  but cannot recover bytes never persisted.
- Detailed managed phase/liveness/transfer evidence depends on image capability;
  a server upgrade cannot retrofit it into a frozen historical image.

Server 0.3.13 fixes the old delivery-worker-identity barrier by reusing sealed
identical-data/NAS receipts ([data readiness](data.md)). Read `operation_status`
when effective readiness comes from another delivery; the uncertain copy still
needs separate inspection and is never automatically replayed. Builder registry
authentication must be visible inside its service sandbox ([deployment](operator-guide.md)).

## Automatic preemption recovery: design only

Automatic recovery is not implemented/enabled. ACP requests use `backoff_limit: 0`;
the collector never launches replacements. Manual retry allowance is not an
automatic policy, and FAILED/SUSPENDED alone does not prove preemption.

A future opt-in implementation needs one restart owner and a new Attempt with
an immutable execution overlay: prior exact job/failure evidence, compatible
registered checkpoint, chosen non-debug placement and remaining budget. It must
reserve cumulative GPU time transactionally before create, retain reservations
for uncertain submissions, enforce finite retry/deadline limits and reconcile
the existing outbox after interruption. Cancellation disables pending recovery.
Code/data/hash failures, user cancellation and output-return failures must not
automatically restart training. No latest-checkpoint fallback or infinite retry.

## Server-owned experiment preparation

An interrupted `EXECUTING` preparation becomes `RECONCILE_REQUIRED` at daemon
startup. Original Runtime/copy request markers and dependency IDs stay intact.
Observe those exact dependencies; after READY verification, explicitly continue
the preparation. This only advances preparation, never authorizes/replays GPU
submission. Changed executor configuration blocks the frozen selection.
