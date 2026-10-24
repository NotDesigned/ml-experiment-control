# Simplification: current design and boundaries

The three structural steps below were introduced in server 0.3.34 and client
0.1.17. The subsequent [client/server refactor](client-server-refactor.md)
separates their internal responsibilities without adding another workflow engine.
Source/image/data identities, Run/Attempt history, budgets, explicit execution,
reconciliation, metric provenance and verified download acknowledgements remain.

## 1. One client workflow

The CLI exposes `experiment`, `status`, `logs`, `metrics`, `cancel`, `retry`,
`download`, and explicit checkpoint references. A single experiment
configuration describes source, Dockerfile or reusable Runtime, inputs, resources
and metrics. Campaign, Action and Submission remain implementation records;
advanced integrators can inspect them when diagnosing an issue.

`SavedPlan` shares durable save, observe and single-request dispatch between the
experiment workflow and training operations. Each caller keeps its own effect
markers, validators and original identities. `status` returns a compact summary
of preparation, scheduler, training, publication and verification; `--details`
expands evidence. It never equates scheduler success with downloadable results
or hides a failed result collection behind training success.

Regression checks cover saved identity/lost replies, observation and delayed
publication without repeated effects. Real training acceptance is separate from
these injected HTTP/backend checks.

## 2. A small backend boundary

The existing core BackendRegistry owns capabilities, stage, submit, status and
cancel. The server's explicit backend mapping owns asset/script preparation,
terminal verification and result collection. This separates provider mechanics
from experiment orchestration without a second scheduler or plugin framework.
Preparation retains immutable data/runtime receipts; submission consumes them.

SenseCore dataset copy, download-script preparation and result collection share
CPU-profile/resource validation and exact name/confirmed UID checks. Each
operation keeps its own intent, retry policy and receipt. Experiment preparation
and result collection dispatch through the server mapping instead of branching
on provider details. Confirmed UID mismatch refuses a replacement; uncertain
creation is observed under its original identity rather than replayed.

An injected third adapter tests this server boundary without client changes; it
does not make an additional production executor available. Executor definitions
and core scheduler support are still needed for a real provider. Real GPU
checkpoint restoration remains a separate acceptance test and must not be
inferred from CPU preparation success.

## 3. One storage lifecycle policy

`StorageService` composes existing object/upload/lifecycle services and their
shared capacity policy. Canonical manifests, receipts, checkpoint registrations
and metric records remain separate from disposable delivery caches. Maintenance
prunes expired uploads before ACK releases and verified cache cleanup, without
introducing per-provider deletion rules or a second metadata store.

The API archive is released only after durable verified client acknowledgement
with release enabled. Reproducible local caches may be removed after independent
object verification. NAS/shared-storage training state and datasets require their own
explicit retention policy. Downloading a URL does not fabricate an acknowledgement.

Synthetic regression checks cover interrupted publication, concurrent capacity
claims, repeated release and cache regeneration while preserving receipts and
metrics. Historical frozen manifests retain their compatibility paths.

## Remaining limits

There is no automatic GPU replacement, general workflow engine, dynamic backend
plugin system or second metadata database. Orchestration waits still occupy two
bounded threads. Dependency continuations are a later option if measured queue
pressure warrants them. Image/data/queue timing and real checkpoint restoration
remain acceptance work, rather than evidence inferred from structural refactoring.
