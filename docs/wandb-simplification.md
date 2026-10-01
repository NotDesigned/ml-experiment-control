# W&B design for metrics and experiment comparison

Status: proposed target design, based on the operator's 2026-10-02 choice of
metrics visualization and experiment comparison. This document does not enable
publication, delete the existing archive, or claim that the old bridge has been
removed. Action storage simplification is a separate implemented change.

## Responsibility

The experiment project writes scientific metrics and chooses their meaning.
W&B stores and visualizes those metrics. ml-expd owns immutable experiment
identity, approved execution, scheduler observations, and links to evidence.
It should not reconstruct a second experiment history by parsing every log line.

```text
Project training/evaluation code
    |-- canonical metrics/checkpoints/logs --> existing project evidence
    `-- explicit safe metric/config fields --> official W&B SDK
                 |-- online --> one configured W&B endpoint
                 `-- offline .wandb files --> official wandb sync --> same endpoint

ml-expd: Attempt identity + tracking reference + open-dashboard link
```

Use the existing official SDK instrumentation in project repositories. ELF
already calls wandb.init/log/finish and supports an authored run ID and resume
policy. That is the preferred writer, not a second daemon uploader of the same
metrics. No training code or publishing credentials are changed by this design
review.

## Minimal contract

- One W&B run per immutable Attempt; keep the existing hash of workspace,
  project, Run, and Attempt as its stable ID. A new Attempt has a new W&B run;
  group related Attempts by the logical Run/experiment identity.
- Configure one endpoint, entity, and project. Cloud or an externally managed
  server is a deployment choice, not two simultaneously mirrored targets.
- Record explicit reproducibility fields: source revision, dataset/version,
  model variant, seed, relevant hyperparameters, and experiment tags.
- Record named metrics with their real independent axes: global step/tokens/
  epochs for training, checkpoint step for evaluation. Do not use outbox row IDs
  as scientific steps. Related metrics need consistent names and units.
- Store only run ID, endpoint/project identity, mode, observed synchronization
  outcome, and a validated credential-free dashboard URL in ml-expd's read
  model. Remote metric delivery is not proof of scheduler/job success.
- One producer owns a W&B run. Separate training/evaluation producers must not
  concurrently write unrelated steps under the same ID without an explicit
  aggregation contract. Retain only the configured primary-rank writer for
  distributed training; log aggregated metrics at the project layer.

## Online and offline use

Offline mode is the lowest-coupling default for remote jobs: write native SDK
files within the immutable Attempt directory; no upload credential is needed
on the compute job. After collection, sync explicitly selected completed SDK
runs with the official CLI from the host that owns the credential. Do not parse
or edit the binary .wandb format or maintain a second per-record retry queue.
One bounded completion hook/operator command is sufficient initially. A failed
sync remains visible and can be retried with the SDK's supported behavior; do
not infer remote exactly-once delivery from a local marker or process exit alone.

Offline-first means dashboard curves become available after synchronization.
For experiments that need live curves and have reliable access, use SDK online
mode instead. SDK initialization/publication failures must be surfaced as
tracking errors without invalidating canonical training output. Automatically
switching between modes midway through a run is out of scope initially.

The host must actually receive native SDK files, not just JSONL metrics or log
tails. Confirm ProjectAdapter evidence collection includes the Attempt's W&B
output before removing the old bridge. Existing JSONL-only history is not input
to wandb sync. If a specific historical comparison is needed, use an explicit
one-shot project exporter that reads those metrics, writes a fresh SDK offline
run, and then syncs it. Do not retain a generic daemon backfill Action engine for
this optional operation.

## Data boundary

Only pass approved config/metric fields to the SDK. ELF's current initialization
collects every public config attribute and must be narrowed before deployment;
its silent log-error handling also needs visible tracking status. Configure the
SDK with console capture disabled and automatic code/git/system-stat collection
restricted to the agreed scope. Keep full logs, generated samples, credentials,
checkpoints, and datasets in their existing evidence stores by default. Filtering
must happen before SDK persistence, including offline files.

No API key goes in a manifest, Action plan, command argument, or saved config
payload. Retain the existing protected credential mechanism for the chosen
uploader until all consumers have been audited; removing a publisher does not
authorize deleting its credential store.

## Code retirement after a producer/consumer contract test

| Current responsibility | Target | Candidate removal |
| --- | --- | --- |
| Local dashboard process management | Separate deployment/operations | observability.py and local_wandb_service.py: 492 lines |
| Log scanning, private archive, record cursors/outbox and target coordination | Project evidence + native SDK storage/sync | observability_archive.py, observability_store.py, observability_runtime.py: 1,818 lines |
| Isolated daemon record publisher | Existing project SDK instrumentation | wandb_publisher.py: 486 lines |
| Two-target status and historical backfill API | One tracking reference/read model | simplify runtime, schemas, routes, operations and Action policy bindings |

These are whole-file candidate counts at dda6acc (2,796 lines), not a promised
net deletion. The replacement contract and one-shot utilities need code too.
The old archive also supports independent log retention, so preserve existing
archive bytes and verify that original evidence remains available before
retiring its writer. Shared storage/credential helpers must remain if they have
other consumers.

## Cutover and acceptance

1. Prove one small project run writes only approved fields using the installed
   SDK version, the real scientific step axis, and the expected Attempt ID.
2. Collect those native SDK files through the actual ProjectAdapter; verify an
   offline run, a resumed run, and a new Attempt remain distinct and comparable.
3. Test the chosen endpoint with the user's publishing configuration; confirm
   chart values/axes and grouping, not merely a successful HTTP request.
4. Update client/API capability contracts before removing old observability and
   backfill routes. No backend claims support for a deleted operation.
5. Remove the superseded daemon bridge and its tests, preserving historical
   archives and a rollback environment. Remove a dependency only after its last
   consumer is gone.

Today's inventory: both existing publishers are disabled, with zero publication
outbox rows and zero targets. This reduces cutover complexity but does not prove
there are no clients depending on the current API or archived logs.

Official implementation references: [W&B SDK run lifecycle](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py),
[SDK settings](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py),
[official sync CLI](https://github.com/wandb/wandb/blob/main/wandb/cli/cli.py).
Behavior must be checked against the installed version (0.28.0 at this review),
not assumed from the development branch. Native offline writing was verified in
the previous change; actual remote sync and project artifact transport still
need their own acceptance checks.
