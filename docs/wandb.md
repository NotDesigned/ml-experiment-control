# Optional W&B publication

ML-Expd publishes accepted experiment records to W&B official Cloud from the
server. The training container needs no W&B SDK/key or W&B network access;
Slurm workers write to shared storage and the server reads them. Metrics remain stored
and queryable in ML-Expd without W&B.

## Enable the integration

The operator points `tracking_config` in the deployment file at a private JSON file:

```json
{
  "schema_version": 1,
  "python": "/absolute/path/to/interpreter-with-wandb-sdk",
  "api_key_file": "/etc/ml-exp/wandb.key"
}
```

Optional `entity` and `project` set server defaults, and `enabled: false` pauses
publishing. The client and worker do not need the SDK, and the key is not a
training-process environment variable. See the
[operator guide](operator-guide.md#optional-wb). The native client has no command
that changes these settings.

## Choose publication per experiment

In an experiment JSON or `POST /api/projects/P/runs` request:

```json
{
  "wandb": {
    "enabled": true,
    "preparation_enabled": false,
    "project": "fineweb-lr-search"
  },
  "parameters": {"lr": 0.001, "wd": 0.1, "seed": 42}
}
```

| Option/default | Meaning |
|---|---|
| Training `enabled:true`, including omitted `wandb` | Request publication; effective enablement still needs server settings/key/entity |
| `enabled:false` | Disable training publication for this Run |
| Preparation omitted | Inherit the server default, initially false |
| `preparation_enabled:true` | Publish a separate preparation W&B run for build/data progress |
| Project omitted | Use server default project, then ML-Expd project name |
| Entity omitted | Use configured/resolved default entity |

Training and preparation switches are independent, subject to the same server
master switch and credentials.

Without credentials, a usable entity, or an enabled server, the effective route
is disabled with a reason and training continues. With a configured route but no
SDK, settings/tracking instead report:

```json
{"publisher":{"available":false,"error":"WANDB_SDK_NOT_INSTALLED"}}
```

Publication remains pending; platform records and frozen routes remain intact.
Installing the extra restores the dependency, not cloud permission/connectivity.
The preparation response freezes the effective routes. Later defaults affect new
Runs only; retry Attempts inherit the Run's destination. A different destination
or different research parameters requires a new Run. A requested entity override
must be accessible to the server's stored credential.

`parameters` is finite JSON up to 16 KiB. It records research configuration but
does not change command arguments or define scientific success.

## Curves and chosen results

Enabling W&B does not extract loss automatically. Training/evaluation code must
write [metric records](metrics.md#record-and-query) to `OUTPUT_DIR/metrics.jsonl`:

- Records without `summary` are history points. Multiple measured steps/epochs form a curve.
- `summary:true` declares the result selected by experiment code. It does not add a curve point or select the best checkpoint.

Each Attempt has a stable W&B run grouped by ML-Expd project/Run.
For new publications (native 0.1.25+), **the client owns metric names**:

| W&B field | Content |
|---|---|
| Exact supplied `name`, such as `train/loss` | History and the current/final summary under the same key |
| `ml_expd/state` and lifecycle metadata | Observed platform state and stages |
| `ml_expd/display_definitions_json` | Units, protocol, dataset, variant, checkpoint, counts and summary-selection source |

The service adds no prefix, hash, variant label or `/by_epoch` suffix to scientific
metric names. Use distinct names when curves should be separate; reusing a name
intentionally shares one W&B metric. Protocol/dataset/variant provenance remains
in accepted records and does not rename the metric. Across Runs, choose
compatible units/protocols for comparisons. Two observations with the same key
inside a single history row cannot be represented without overwriting, and
publication reports `DISPLAY_NAME_CONFLICT` instead.

Charts use explicit training `step`, otherwise `epoch`, when supplied. These
coordinates are stored as hidden `ml_expd/axes/...` helper fields; a metric with
no scientific coordinate uses W&B's default history position. There is no
automatic second epoch curve or two-point visibility gate. W&B's internal
`_step` is a publication position, independent of training progress.

The same exact name appears in the live and final summary. An explicit summary
wins; otherwise the latest accepted observation is used, never minimum loss or
automatic model selection. When several contexts intentionally share a name,
the most recent explicit summary (otherwise observation) in journal order wins.
Invalid explicit results retain their diagnostic and remove that metric's
summary value instead of falling back to unrelated history. Unrelated
client-owned summary keys are not removed. Failed Attempts can retain valid
measurements; filter `ml_expd/state` when comparing successful runs.

Finished W&B Runs are not reopened. Publication does not rename or replay accepted
points, reset confirmation cursors, or resend uncertain events. Publications use
exact names. Checkpoint changes remain per-point
provenance. Numerator/denominator are metadata, not extra result metrics.

## What reaches W&B

Publication contains source/image/data identities, explicit parameters, frozen
schema/protocol, restore lineage, lifecycle/exit evidence, metric records,
checkpoint manifests and verified output receipts. It does not upload complete
weights, datasets, recovery state or raw logs. Private commands, SSH settings,
transfer credentials, signed URLs and object keys are excluded. The SDK disables
host/git/code telemetry so API-server hardware is not reported as training usage.

Saved JSONL can fill gaps; sampled stdout observations remain labelled incomplete.
Missing history is not invented.

## Inspect and recover publication

```bash
ml-exp run tracking --project my-study --run trial-001
```

The status API returns the frozen route, W&B URLs, confirmed/pending counts,
timestamps and a fixed diagnostic. It makes no cloud request. SDK enqueue is not confirmation: the publisher advances only
after observing matching remote event identities. History and derived display
have separate confirmation states:

| Diagnostic | Meaning |
|---|---|
| `WANDB_SDK_NOT_INSTALLED` | Operator must install the optional SDK |
| `REMOTE_ACK_PENDING` | History is not remotely confirmed yet |
| `REMOTE_DISPLAY_ACK_PENDING` | History arrived; derived display is not confirmed yet |
| Identity/reconciliation error | Inspect remote history; do not blindly replay it |

Worker acceptance, W&B confirmation and artifact upload are independent. Normal
observation/publication polls are about ten seconds plus collection/network
latency; error backoff reaches five minutes. W&B's green session indicator tracks
the server publisher session and can lag during outages; it is not direct proof
that GPU training is alive. Use platform state, logs and steps together.

The native API has no backfill route for disabled Runs, and
publication cannot recreate observations that were never saved. Publication cursors
live in the registry's `tracking/` directory; keep it with service state.


## Reserved history gaps (native 0.1.21+)

Normal publication stops at an unacknowledged send and flushes SDK sessions on
orderly API shutdown. An uncertain send is never automatically replayed.

For a verified remote gap below later published events, an operator can stop the
API and run the existing configuration with:

```bash
ml-expd-rs --config /private/server.json --repair-tracking PROJECT RUN ATTEMPT
```

The command takes the existing publisher leadership lock, compares two remote
identity scans with original journal events and send reservations, then records
one repair attempt before appending only those missing events to the original
W&B Run. Physical rows advance; values, units, scientific axes and event IDs stay
unchanged. Confirmation still requires matching remote history. Repeating an
uncertain repair only observes; it does not send again or clear reservations.
Restart the API to resume incremental publication. Existing W&B panels are not
recreated. Keep the pre-repair event list and actual remote confirmation evidence.

A saved repair adds `history_repair` to publication state. After a repair use a
forward fix rather than a program downgrade or a database restore. This operation does not restart training or recover
records that are missing from the canonical journal.

Remote SDK pages may repeat the same physical row. Native 0.1.22 compares the
complete row (including values and axes) before collapsing such page repeats;
conflicting content or the same sequence at different physical rows is rejected.
This does not resubmit history or relax the exact-prefix acknowledgement.

Explicit history recovery can also repair an absent reserved tail. It still requires review of the exact original events and stable remote absence; the normal publisher never resends unknown reservations. The recovery confirms the appended range using normal remote evidence, preserves scientific step axes, and a repeated or uncertain repair only observes.
