# Optional W&B publication

ML-Expd mirrors accepted experiment records to W&B official Cloud from the
server. The mirror is best effort: W&B may rarely miss or repeat a row. The
platform's own journal, its result archives and [MLflow](mlflow.md) are the
authoritative record; use them when an exact count matters. The training container needs no W&B SDK/key or W&B network access;
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
write [metric records](metrics.md#record-and-query) to `METRICS_PATH`:

- Records without `summary` are history points. Multiple measured steps/epochs form a curve.
- `summary:true` declares the result selected by experiment code. It does not add a curve point or select the best checkpoint.

Each Attempt has a stable W&B run grouped by ML-Expd project/Run.
**The client owns metric names**:

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

Finished W&B Runs are not reopened, with one exception: a
[derived batch](#derived-metrics-after-a-run-ended) added after the Run ended.
To remove metric names from a finished W&B Run, publish a new one with
[`run rebuild-wandb`](#rebuild-a-finished-wb-run-without-some-metrics).
Publication does not rename accepted points or reset
its cursor. Publications use exact names. Checkpoint changes remain per-point
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

## Inspect publication

```bash
ml-exp run tracking --project my-study --run trial-001
```

The status API returns the frozen route, W&B URLs, sent/pending counts,
timestamps and a fixed diagnostic. It makes no cloud request.

**How rows are sent.** The publisher sends each Attempt's journal rows to its W&B
Run in order through the official SDK, in batches of up to 8192 rows. A batch
counts as sent when the SDK has logged every row and its own synchronous step
acknowledgement shows them all taken. W&B is not read back. `confirmed` is then
the cursor: how many rows the SDK accepted, and `pending` is the rest of `total`.
`last_confirmed_at` is when the cursor last moved.

**Errors.** A failed send leaves the cursor where it was, and the same rows are
sent again on the next attempt, with backoff up to five minutes. Rows the SDK took
before the failure can then appear twice in W&B. `failures` counts consecutive
errors and is reset by the next accepted batch.

**Restarts.** The cursor is saved with the journal, so after a restart publication
continues from it and accepted rows are not sent again. Rows the SDK accepted but
had not yet uploaded are lost if its process is killed. An orderly API stop
flushes the SDK sessions. This needs `KillMode=mixed` in the service unit, as in
[`ml-expd.service`](../rust/deploy/ml-expd.service): under systemd's default the
SDK child and `wandb-core` also receive SIGTERM and die at once, losing what they
hold. The deploy tool does not change units; add it to an existing one by hand.

Nothing compares W&B with the journal afterwards, and nothing repairs a missing
or repeated W&B row. `ml-expd-rs --repair-tracking` is retired: it prints an
explanation and changes nothing. To replace a W&B Run wholesale, use
[`run rebuild-wandb`](#rebuild-a-finished-wb-run-without-some-metrics).

| `status` | Meaning |
|---|---|
| `PENDING` | Rows are waiting to be sent |
| `SYNCED` | Every row and the summary were accepted by the SDK; the W&B Run is open |
| `FINISHED` | The W&B Run was finished: `finished` for a Run that succeeded, `failed` otherwise |
| `RETRY_PENDING` | The last attempt failed; `error` says why and the same rows are sent again |
| `REMOTE_RUN_MISSING` | A person deleted the W&B Run; it is never recreated and the scope stops |
| `LOCAL_ONLY` | W&B publication is disabled for this Run |

| `error` | Meaning |
|---|---|
| `WANDB_SDK_NOT_INSTALLED` | Operator must install the optional SDK |
| `WANDB_AUTHENTICATION_FAILED`, `WANDB_PERMISSION_DENIED` | The server's key cannot write to the route |
| `WANDB_INIT_FAILED`, `WANDB_WRITE_FAILED`, `WANDB_REMOTE_READ_FAILED` | An SDK call failed; retried |
| `WANDB_REMOTE_RUN_MISSING` | See `REMOTE_RUN_MISSING` |
| `DISPLAY_NAME_CONFLICT` | Two observations with one name in one row |
| `JOURNAL_HISTORY_PROJECTION_INVALID`, `JOURNAL_METRIC_DEFINITIONS_INVALID`, `JOURNAL_AUTHORITY_UNAVAILABLE` | A journal record could not be published |
| `PUBLICATION_STATE_INVALID` | The saved publication state cannot be used |
| `PUBLISHER_UNAVAILABLE` | The SDK process or the local store failed; retried |

States saved by an earlier release, which waited to read rows back from W&B
(`ACK_PENDING`, `RECOVERY_SETTLING`, `REPAIR_ACK_PENDING` and their reservations,
repair records and duplicate counts), load unchanged and move on by themselves on
the first visit. Rows that release had handed to the SDK, from the cursor up to
the first one it had not, count as sent; later ones are sent again. That earlier
release cannot load a state this one has saved, so after a downgrade it reports
`PUBLICATION_RECONCILE_REQUIRED` for that scope instead of publishing.

Worker acceptance, W&B publication and artifact upload are independent. W&B's
green session indicator tracks the server publisher session and can lag during
outages; it is not direct proof that GPU training is alive. Use platform state,
logs and steps together.

The native API has no backfill route for disabled Runs, and
publication cannot recreate observations that were never saved. Publication cursors
live in the registry's `tracking/` directory; keep it with service state.

At most 16 W&B Runs stay open in the SDK. Opening another finishes the least
recently used one, which flushes it, and it resumes when it has more to send, so
W&B may briefly show a running Run as `finished`. The publisher turns to the next
Run after each batch, so one large backlog cannot hold back a live Run, and a Run
with nothing new makes no W&B request. Each journal record is one W&B row with its
own `ml_expd/sequence` and `ml_expd/event_id`, so a repeated row can be recognised
and filtered by those fields.

## Derived metrics after a Run ended

A [derived batch](metrics.md#add-derived-metrics-to-an-ended-run) added to an
Attempt whose W&B Run is already finished resumes that same W&B Run; no new Run is
created. Its rows continue the Attempt's `ml_expd/sequence` after the last training
row, and each carries `ml_expd/source=derived`, `ml_expd/kind=derived_metrics` and
`ml_expd/derived_sha256`. Charts use the record's own `step` or `epoch`, so a step
before the last training step is drawn where it belongs. The summary shows each
derived name's latest value like any other metric.

The rows go through the same publication as training rows: sent in order from
the cursor and retried on error. When every row and the summary are sent, the W&B
Run is finished again: `finished` for a Run that succeeded, `failed` otherwise. The same file sent again changes nothing,
so the W&B Run is not reopened for it. Training rows are not resent or changed.

## Rebuild a finished W&B Run without some metrics

W&B cannot delete one metric key from a Run. To remove metric names from an
Attempt's finished W&B Run, the platform publishes a new W&B Run for the
Attempt from its own records, leaving out every observation with a listed name,
and binds the Attempt to the new W&B id. Rows are sent the same best-effort way. It never deletes or changes a W&B Run:
check the new Run, then delete the old one yourself in W&B.

```bash
printf 'loss/old\neval/old_accuracy\n' > names.txt
ml-exp run rebuild-wandb --project demo --run train --attempt attempt-001 \
  --exclude-names names.txt --confirmation 'REBUILD demo/train/attempt-001'
```

The Attempt is chosen as for `run append-derived`; `--attempt` defaults to the
Run's current Attempt. A wrong confirmation is refused before anything is sent,
and the message gives the exact one.

**The list.** One full metric name per line, compared exactly; `*` and other
characters have no special meaning. Surrounding spaces and blank lines are
ignored and a repeated name counts once. At most 5000 names and 256 KiB; a larger
list is refused with `WANDB_REBUILD_LIST_TOO_LARGE`, and one with no name or a
control character with `WANDB_REBUILD_NAMES_INVALID`. The file is sent and stored
verbatim with its SHA256.

**When.** Only an Attempt whose Run ended (its journal recorded the end, or the
provider proved the job stopped) and whose W&B publication has sent every
record, derived batches included, and finished the W&B Run. Otherwise nothing changes:
`WANDB_REBUILD_PUBLICATION_NOT_FINISHED` while publication is still running or a
derived batch is not yet on W&B, `WANDB_REBUILD_RUN_NOT_ENDED` for a Run that has
not ended. This is available for cloud Runs.

**What the new Run holds.** Every row of the training journal and the Attempt's
derived batches, in sequence order, as the publisher always publishes them, minus
the listed observations: the same `ml_expd/sequence`, `ml_expd/event_id`, steps,
summary rules and configuration, and the same display name `RUN/attempt-NNN` and
group, so name filters find it. The old Run keeps its name. A row whose only
metrics were all excluded is still published, with only its `ml_expd/*` metadata
and its `ml_expd/record` showing no observation, so every sequence is sent to the
new Run. Excluded names get no chart, no
summary value and no metric definition. The new Run ends `finished` for a Run that
succeeded and `failed` otherwise.

**The new id.** It is made from the old W&B id and the list's SHA256, so a server
restart in the middle resumes the same new Run from its cursor, as for any
publication. Nothing is sent to the old
W&B id after the rebuild starts.

**Repeating and chaining.** The list of the latest rebuild again returns that
rebuild (`status: ALREADY_REQUESTED`) and starts nothing, so a lost response is safe
to repeat. A different list, once the previous rebuild's Run is finished, starts
another rebuild from the full records to another new id made from the current
one; lists do not accumulate, so each list names everything to leave out. While a
rebuild is still publishing, a different list is refused with
`WANDB_REBUILD_PUBLICATION_NOT_FINISHED`.

**Afterwards.** The Attempt publishes as before, to the new id: a later
[derived batch](#derived-metrics-after-a-run-ended) resumes the new Run, and the
latest list still applies to it. The training journal, derived batches, results
and `run metrics` reads are unchanged; only what W&B shows differs.

**Audit.** `ml-exp run tracking` shows each scope's current `wandb_id` and a
`wandb_rebuilds` list, oldest first. Each entry has `ordinal`, `sha256`,
`previous_wandb_id`, `wandb_id`, `requested_at`, `finished_at` (the first time the
new Run was finished), `status` (`PUBLISHING` or `FINISHED`), `names`, `excluded`
(each excluded name with its observation count), `excluded_observations`,
`unmatched_names` (listed names no record has), `emptied_rows` and
`previous_wandb_run_deleted: false`. The verbatim list and the old publication
state are kept in `tracking/outbox.sqlite`.

| Method and path | Body or result |
|---|---|
| `POST /api/runs/{project}/{run}/attempts/{attempt}/wandb-rebuilds` | `sha256` (lowercase hex of the UTF-8 list), `names` (the list text) and `confirmation` (`REBUILD {project}/{run}/{attempt}`); returns `status` (`STARTED` or `ALREADY_REQUESTED`), the `rebuild` above and `wandb` with both ids and URLs |

The server checks the SHA256 against the text (`WANDB_REBUILD_DIGEST_MISMATCH`).
A server older than this feature cannot load a rebuilt scope's publication state,
so after a downgrade it reports `PUBLICATION_RECONCILE_REQUIRED` for that scope
rather than publishing the excluded names.
