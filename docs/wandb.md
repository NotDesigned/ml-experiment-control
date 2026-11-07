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
Publication does not rename or replay accepted
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

`pending` is how many records are not yet confirmed on W&B, out of `total`, and
`last_confirmed_at` is when that last moved. `failures` counts real errors, not
waiting for W&B to show what was sent.

Worker acceptance, W&B confirmation and artifact upload are independent. Error
backoff reaches five minutes. W&B's green session indicator tracks
the server publisher session and can lag during outages; it is not direct proof
that GPU training is alive. Use platform state, logs and steps together.

The native API has no backfill route for disabled Runs, and
publication cannot recreate observations that were never saved. Publication cursors
live in the registry's `tracking/` directory; keep it with service state.


## Throughput

Publication is designed to keep up with at least 100 records per second across
all Runs, with a lag near the time W&B takes to show a batch. This rate is derived,
not yet measured against W&B. Each journal record stays one W&B row with its own
`ml_expd/sequence` and `_step`, so a row still proves exactly one record.

- A Run sends up to 4096 rows at once and waits until W&B shows them, confirming
  by range as they appear. The official SDK accepted about 2000 rows a second on
  the 2 GB production host, offline, so the limit is the time rows take to become
  visible: 4096 rows over about 30 seconds is about 130 records a second. Rows
  are sent every 5 seconds per open W&B Run; a waiting Run is read every 5 seconds.
- All Runs with a backlog share 16384 rows in flight equally, never less than 128
  each, and the publisher turns to the next Run after each batch, so one large
  backlog cannot hold back a live Run. A Run with nothing new makes no W&B request.
- At most 16 W&B Runs stay open in the SDK. Opening another finishes the least
  recently used one, which flushes it, and it resumes when it has more to send, so
  W&B may briefly show a running Run as `finished`.

## Restarts and reserved history gaps

Every row is reserved durably before it is sent, and an uncertain row is never
blindly sent again. An orderly API stop flushes the SDK sessions. This needs
`KillMode=mixed` in the service unit, as in
[`ml-expd.service`](../rust/deploy/ml-expd.service): under systemd's default the
SDK child and `wandb-core` also receive SIGTERM and die at once, losing what they
hold. The deploy tool does not change units; add it to an existing one by hand.

A crash, a kill or a lost SDK child can still lose rows, so the publisher recovers
them by itself. Rows no running SDK child holds show `RECOVERY_SETTLING` while
it keeps reading W&B: anything that arrives is confirmed, and each change restarts
a three-minute delay, because a dying `wandb-core` or W&B itself may still deliver
them. Then the two identical scans are compared with the original journal events
and send reservations, a repair record is saved (`history_repair`, with `source`,
`attempts` and time; older repairs are summarized in `repairs`), and only rows
proven absent are appended to the original W&B Run. Status is `REPAIR_ACK_PENDING`
until W&B shows them, and publication then continues. A repair that also goes
missing is retried the same way, each time waiting twice as long, up to about three
hours, and it does not give up. Losing the SDK child or a failed read restarts the delay.
A session that shows no progress for ten minutes is closed, which flushes it,
and is then treated the same way.

A successful recovery ends like any other publication: once all records and the
display are confirmed, the W&B Run is resumed if needed and finished, so a
`crashed` Run becomes `finished`, or `failed` for a Run that did not succeed.

An operator can run the same repair at once, without the delay, with the API stopped:

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

After a repair use a
forward fix rather than a program downgrade or a database restore. This operation does not restart training or recover
records that are missing from the canonical journal.

Remote SDK pages may repeat the same physical row. The service compares a digest of the
complete row (including values and axes) before collapsing such page repeats;
conflicting content or the same sequence at different physical rows is rejected.
This does not resubmit history or relax the exact-prefix acknowledgement.

Both repairs cover a reserved hole or an absent reserved tail. They preserve
scientific step axes and confirm by normal remote evidence.

What only a real W&B run can show: how long rows take to become visible, and so
the real rate; that a resumed `crashed` Run ends `finished`; and whether three
minutes always covers late delivery. A row delivered after its repair appears twice,
which W&B cannot undo. Publication continues, and the state counts it as
`duplicate_rows`. Only rows from the confirmed position on are read, so a repeat below
that position goes uncounted. Two different events at one sequence still stop the Run
with `WANDB_HISTORY_READ_FAILED`.

Sixteen Runs keep a session open at once. A seventeenth closes the oldest inside its own
call, and a slow close can lose the SDK child. Recovery is the same as above, so it costs
delay and not records.

## Derived metrics after a Run ended

A [derived batch](metrics.md#add-derived-metrics-to-an-ended-run) added to an
Attempt whose W&B Run is already finished resumes that same W&B Run; no new Run is
created. Its rows continue the Attempt's `ml_expd/sequence` after the last training
row, and each carries `ml_expd/source=derived`, `ml_expd/kind=derived_metrics` and
`ml_expd/derived_sha256`. Charts use the record's own `step` or `epoch`, so a step
before the last training step is drawn where it belongs. The summary shows each
derived name's latest value like any other metric.

The rows go through the same publication as training rows: each is reserved before
it is sent, confirmed only by reading it back from W&B history by its sequence and
event ID, and never sent again unless it has been seen to be absent. When every
row and the display are confirmed, the W&B Run is finished again: `finished` for
a Run that succeeded, `failed` otherwise. The same file sent again changes nothing,
so the W&B Run is not reopened for it. Training rows are not resent or changed.

## Rebuild a finished W&B Run without some metrics

W&B cannot delete one metric key from a Run. To remove metric names from an
Attempt's finished W&B Run, the platform publishes a new W&B Run for the
Attempt from its own records, leaving out every observation with a listed name,
and binds the Attempt to the new W&B id. It never deletes or changes a W&B Run:
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
provider proved the job stopped) and whose W&B Run holds every record, derived
batches included, and is finished. Otherwise nothing changes:
`WANDB_REBUILD_PUBLICATION_NOT_FINISHED` while publication is still running or a
derived batch is not yet on W&B, `WANDB_REBUILD_RUN_NOT_ENDED` for a Run that has
not ended. This is available for cloud Runs.

**What the new Run holds.** Every row of the training journal and the Attempt's
derived batches, in sequence order, as the publisher always publishes them, minus
the listed observations: the same `ml_expd/sequence`, `ml_expd/event_id`, steps,
summary rules and configuration, and the same display name `RUN/attempt-NNN` and
group, so name filters find it. The old Run keeps its name. A row whose only
metrics were all excluded is still published, with only its `ml_expd/*` metadata
and its `ml_expd/record` showing no observation, so every sequence is on the new Run and
it is proven complete exactly as any other Run. Excluded names get no chart, no
summary value and no metric definition. The new Run ends `finished` for a Run that
succeeded and `failed` otherwise.

**The new id.** It is made from the old W&B id and the list's SHA256, so a server
restart in the middle resumes the same new Run: rows already sent are read back
before anything is sent again, as for any publication. Nothing is sent to the old
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
