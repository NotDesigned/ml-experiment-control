# Simplification plan

This is a proposed next sequence, not a declaration of implemented APIs. Keep
source/image/data identities, Run/Attempt history, budgets, explicit execution,
reconciliation, metric provenance and verified download acknowledgements.

## 1. One client workflow

Expose `experiment`, `status`, `logs`, `metrics`, `cancel`, `retry`, `download` and
checkpoint references as the normal client vocabulary. A single experiment
configuration describes source, Dockerfile or reusable Runtime, inputs, resources
and metrics. Campaign, Action and Submission remain internal records; advanced
integrators can expand them when diagnosing an issue.

First extract the duplicated prepare/observe/save-plan/authorize/execute code in
client workflow and operations into one small helper. Preserve their distinct
effect markers and original identities. Then make the default status a compact
summary of preparation, scheduler, training, publication and verification, with
evidence accessible on request. Do not equate scheduler success with downloadable
results, or hide a failed result collection behind training success.

Acceptance: build once, submit several Runs, resume after lost replies, observe
training, and download after delayed publication without repeated effects.

## 2. A small backend boundary

Use the existing scheduler core and server storage helpers; do not add a plugin
framework. Group backend operations around capabilities, prepare, submit,
observe, cancel and collect. Preparation returns immutable data/runtime receipts;
submission consumes them. Provider names, cluster specs, NAS mounts, Slurm and
OCI-to-SIF mechanics stay behind this boundary.

First remove repeated CPU-profile validation and exact-job reconciliation from
SenseCore dataset copy, download-script preparation and result collection. Share
only the verified profile/job identity primitive; each operation keeps its own
intent, retry policy and receipt. Next move direct provider branching out of the
orchestrator into these backend helpers.

Acceptance: the same experiment configuration works on WYD and SenseCore except
for executor selection; a third fake adapter passes contract tests without edits
to the client. Real GPU checkpoint restoration remains a separate acceptance
test and must not be inferred from CPU preparation success.

## 3. One storage lifecycle policy

Keep canonical manifests, receipts, checkpoint registrations and metric records
separate from disposable delivery caches. Consolidate capacity admission,
verified cache release and expired-upload maintenance behind the existing storage
services, instead of adding per-provider cleanup rules.

The API archive is released only after durable verified client acknowledgement
with release enabled. Reproducible local caches may be removed after independent
object verification. NAS/datapool training state and datasets require their own
explicit retention policy. Downloading a URL does not fabricate an acknowledgement.

Acceptance: repeated download/resume/release, interrupted publication, concurrent
transfers and cache regeneration preserve original receipts and metrics while
respecting bounded disk admission. Test with synthetic objects before research
archives. Do not remove historical compatibility until its frozen manifests are
independently accounted for.

## Keep this round limited

Do not add automatic GPU replacement, a general workflow engine, dynamic backend
plugins or a second metadata database. Current orchestration waits still occupy
two bounded threads; replace them with dependency continuations only when measured
queue pressure justifies that complexity. Optimize measured image/data/queue time
before broad module rewrites.
