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
| Empty files / archive 404 | No collected/published result available | Upload state and backend storage |

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

**There is no general public CPU output-recovery API yet.** The 2026-10-07
SenseCore recovery was an audited one-off operator tool in private `.ops/`:
NAS verification, original upload 43/250→250/250, exact final-weight/download hashes,
zero GPU. It does not establish a reusable client endpoint. Do not rerun training
or require training-data delivery just to retrieve already completed results.

## Restore unfinished training

Choose a registered complete checkpoint from the exact source Attempt and create
a new Run with `resume_from` ([checkpoint contract](persistent-checkpoints.md)).
Code owns optimizer/RNG/data-cursor restoration; the worker verifies files before
starting it. Reconcile the prior job first if its outcome is uncertain. Never
create a replacement while the old job might still run.

## Known limitations

These issues were observed on 2026-10-07 in server 0.3.12 and are **not fixed by a
documentation rewrite**. Remove this section's items only after implementation
and validation:

- SenseCore offline logs forward requested tail as page_size. The provider
  accepts at most 200; larger requests return 400. A bounded accepted request can
  still return no hits, so absence of logs does not establish an exit cause.
- Worker preparation, restore, training and artifact return share one duration.
  A hard kill can interrupt return; separate recovery is not automated.
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
