# Native experiment tracking

W&B is for numeric metric visualization and experiment comparison. Project
training/evaluation code writes through the official SDK; ml-expd reads the
Attempt's tracking reference. The daemon does not parse logs into a second metric
history, host a W&B service, run a publisher loop, or maintain a dual-target outbox.

## Producer contract

ELF's `elf_experiments.tracking` is the reference producer. Tracking remains
opt-in (`use_wandb` / `USE_WANDB`). Obtain the workspace identity from authenticated
`GET /api/tracking`, then author the following non-secret environment for a new
Run through its normal reviewed configuration flow:

```yaml
env:
  USE_WANDB: 'true'
  WANDB_WORKSPACE_ID: '<workspace identity from /api/tracking>'
  WANDB_MODE: offline
  WANDB_PROJECT: ELF
```

The controller already supplies `PROJECT_NAME`, `RUN_ID`, `ATTEMPT_ID` and
`SOURCE_ID`. Missing identity produces a visible tracking error; it never falls
back to a shared Run ID. The run ID is SHA-256 of the four identity strings
(workspace, project, Run, Attempt), separated by NUL, truncated to 32 hex digits.
Related Attempts share a group derived from the first three identity strings.
This is an SDK integration for newly authored source revisions. Existing
immutable images, Run manifests and approved controller snapshots are not
rewritten, and old JSONL-only experiments do not acquire native files retroactively.

ELF uses W&B 0.28.x. Rank zero writes an explicit whitelist of scientific
configuration and finite numeric metrics. Training uses `global_step`; evaluation
uses `checkpoint_step`. SDK record order is not the scientific x-axis. Console,
code, git, system statistics, automatic metadata and job collection are disabled.
Generated text/tables, credentials, full logs, checkpoints and datasets are not
passed to the SDK. Canonical metrics are written independently and remain valid
if tracking initialization/logging/finish fails. `tracking.json` records the
bounded identity/mode/status, never the exception message.

Offline is the default. Each SDK session writes native files under the Run output directory,
`tracking/<attempt>/wandb/offline-run-*/run-<id>.wandb`. Restarting the same Attempt
produces segments with the same ID; retrying as a new Attempt produces a new ID.
WYD collection includes those exact native files plus `tracking.json`; actual
rsync filters are tested against real SDK output. Local jobs retain native files in their local output; a host must map or collect
that directory into its canonical Attempt evidence before using the daemon read API. SenseCore's log-only collector cannot transport binary
SDK files: use an explicit external artifact transfer before offline sync, or
an explicitly configured online producer. No binary transport is claimed for it.

## One destination and explicit sync

Replace the retired `observability:` configuration with `tracking:`. The old
configuration is rejected, rather than silently enabling or ignoring a target.

```yaml
tracking:
  credential_root: /var/lib/ml-expd/credentials
  credential_ref: wandb-cloud-default
  entity: my-team
  project: ELF
  api_url: https://api.wandb.ai
  dashboard_url: https://wandb.ai
```

Cloud and an externally managed W&B server are alternative destinations. HTTPS
is required except on loopback. Provision the API key with the existing local
`credential wandb set` command; it is never placed in a manifest or argv.
An absent entity/project leaves sync unconfigured; offline recording still works.

```sh
ml-expd --config server.yaml tracking-sync PROJECT RUN ATTEMPT
```

This command requires an indexed terminal Attempt and finished native tracking
evidence. It stages only native segments outside `collected_run` so subsequent
rsync `--delete` cannot remove SDK sync markers. It invokes official `wandb sync`
with an exact ID/entity/project, `--append`, `--skip-console` and
`--no-sync-tensorboard`; a private environment supplies the credential only to
the child. One Attempt lock and a five-minute total wait bound apply. The CLI's
own `.synced` markers handle retries; there is no per-record application queue.
Changed native segments or a different destination fail closed. Failed sync is
recorded as `SYNC_FAILED`, without changing job state. `CLI_COMPLETED` means the
CLI exited successfully and wrote its native marker; chart values and remote
persistence still require endpoint acceptance. No background upload or historical
backfill runs automatically. Preserve the native segments and markers together.

## Read API and migration

- `/api/health` advertises `tracking.v1`, no longer `observability.v1`.
- `/api/tracking` returns non-secret destination readiness and producer identity.
- `/api/tracking/attempts/{project}/{run}/{attempt}` returns bounded evidence state,
  native segment count and observed sync outcome. Unknown Attempts return 404.
- Terminal snapshots include `tracking.attempts`. URLs must match the configured
  destination and exact identity; credential-bearing or arbitrary URLs are omitted.
- Protocol-v1 observability read routes remain small `RETIRED` tombstones with
  empty target lists. Old health counters remain empty. Clients should discover
  capabilities and use tracking fields. There is no backfill operation in the
  catalog; old backfill Actions remain readable but cannot execute/reconcile.
- Submission `wandb_cloud_sync: true` is rejected. A false legacy field is accepted
  for protocol-v1 clients. Scheduler authorization and reconciliation are separate.

Preserve historical archive files, the old observability SQLite database and
credential store. Retiring their writer does not delete them. They are not
current tracking state. Rollback requires restoring the old config/runtime as
well as the original project controller environment; do not replace current
Action data with a pre-upgrade copy.
