# Optional MLflow publication

ML-Expd can publish accepted metric records to an MLflow tracking server, beside
or instead of [W&B](wandb.md). The server does this itself over MLflow's REST API;
training containers need no MLflow client, credential or network access to MLflow.
Records stay stored and queryable in ML-Expd whether or not MLflow is reachable.

## Configure

Add an `mlflow` block to the private `tracking_config` file the deployment points
at. Existing files without it behave exactly as before.

```json
{
  "schema_version": 1,
  "python": "/absolute/path/to/interpreter-with-wandb-sdk",
  "api_key_file": "/etc/ml-exp/wandb.key",
  "mlflow": {
    "url": "https://mlflow.example.org:5000",
    "username": "ml-expd",
    "password_file": "/etc/ml-exp/mlflow.password",
    "ca_file": "/etc/ml-exp/mlflow-ca.pem",
    "experiment_prefix": "ml-expd/",
    "projects": ["fineweb", "vision"]
  }
}
```

For MLflow without W&B, leave the W&B fields out:
`{"schema_version": 1, "mlflow": {...}}`.

| Field | Meaning |
|---|---|
| `url` | Server address, `https://host[:port][/prefix]`. Plain `http` with credentials is refused unless the host is loopback; the URL may not hold a user, query or fragment |
| `username`, `password_file` | HTTP Basic credentials; both or neither. The password file must be private (mode `0600`, owned by the service user) |
| `ca_file` | Optional PEM certificate added to the trusted roots, for a private CA |
| `experiment_prefix` | Optional text put before the project name to form the experiment name |
| `projects` | Optional list: only these ML-Expd projects publish to MLflow |
| `enabled` | Default `true`; `false` stops the publisher and disables MLflow for new Runs |
| `timeout_seconds` | Per-request timeout, 1 to 300, default 60 |

`ml-expd-rs --config /etc/ml-exp/server.json --check` validates the block and reads the password and CA
files; it makes no MLflow request. The server reads them at start; restart it after
changing them.

## Frozen per Run

When a Run is created, its tracking route records the MLflow decision beside the
W&B one, in `tracking_route.mlflow`:

```json
{"enabled": true, "reason": null, "experiment": "ml-expd/fineweb"}
```

`reason` is `DISABLED_BY_SERVER` or `PROJECT_NOT_SELECTED` when `enabled` is false.
The decision and the experiment name never change for that Run: later settings
affect new Runs only, and retry Attempts inherit their Run's route. Runs created
before MLflow was configured have no `mlflow` key and are never published to MLflow.
The W&B request options (`wandb.enabled` and so on) do not affect MLflow, so a Run
can publish to W&B, to MLflow, to both, or to neither. Preparation Runs are not
published to MLflow. The server address and credentials are not frozen: publication
always uses the current `url`.

## What MLflow shows

| ML-Expd | MLflow |
|---|---|
| Project | Experiment named `experiment_prefix` + project, created if missing |
| Attempt | One run named `RUN/attempt-NNN`, tags `ml_expd.project`, `ml_expd.run`, `ml_expd.attempt`, `ml_expd.scope` (`PROJECT/RUN/attempt-NNN`), and `ml_expd.source_id`, `ml_expd.image_id`, `ml_expd.backend` when known |
| Metric record | Metric with the record's exact `name`, its `value`, `step` = the record's `step`, otherwise `epoch`, otherwise 0 (a fractional value is rounded down), and `timestamp` = the run's start time in milliseconds plus the record's journal sequence |
| `summary: true` record | Metric `summary/<name>` |
| Run `parameters` | Params: nested objects become dotted keys (`opt.beta1`), strings stay as they are, other values are their compact JSON; `seed` becomes `ml_expd.seed` |
| Derived batch | More metrics in the same MLflow run |
| Attempt lifecycle | Run status `RUNNING` while active, then `FINISHED` or `FAILED` |

Only observations with status `VALID` or `PARTIAL` and a finite value are points,
as on W&B. The timestamp is a publication coordinate that makes every point
unique and every replay identical; it is not when the record was measured. Use
`step` as the x-axis.

**Terminal status.** The rule is the W&B publisher's: once every journal record is
committed, an Attempt whose journal ends with `METRICS_DRAINED` and state
`SUCCEEDED` sets its run `FINISHED`; one that ended in any other state, or that the
provider proved stopped without `METRICS_DRAINED` (preempted, killed, timed out),
sets it `FAILED`. A [derived batch](metrics.md#add-derived-metrics-to-an-ended-run)
added later is written to the same run, which MLflow accepts while it is finished,
and the status is set again.

**What does not fit is skipped and named.** A metric or param name MLflow would
refuse (characters other than letters, digits, `_ - . /` and spaces, a path that
normalizes differently such as `a//b` or `a/`, or more than 250 characters), a param
value over 6000 characters, and params past the first 1000 are not sent. The state
counts skipped points in `skipped_points`, lists up to 20 skipped metric names in
`skipped_names`, and lists skipped params in `skipped_params`. The journal keeps
every record.

## Delivery

One publisher thread, separate from the W&B publisher, visits each scope with work.
It finds the experiment by name (creating it if missing), then the Attempt's run by
its `ml_expd.scope` tag, creating it only when none exists, and saves the run id.
Params are sent once. Records then go in `runs/log-batch` calls of at most 1000
points and under 1 MB, in journal order.

- A 200 from `runs/log-batch` means MLflow committed the whole batch, and the
  cursor (`confirmed`) moves past it.
- After an uncertain outcome, such as a timeout or a lost connection, the cursor
  stays and the same batch is sent again. Its points are identical, and MLflow's
  SQL store keeps one row per identical (key, timestamp, step, value) point, so the
  replay adds nothing. No `ml_expd/sequence` metrics are needed.
- A lost `runs/create` response is safe for the same reason in another form: the
  next visit finds the run by its tag and does not create a second one.
- A scope with nothing new makes no MLflow request.

A scope with a backlog sends up to 8 batches, then the publisher turns to the next
Run. Errors back off up to five minutes per scope. Measured against MLflow 3.18 with
the SQL store, one server accepted about 450 to 500 points per second.

If a person deletes the MLflow run, it is never recreated: the scope stops with
`MLFLOW_RUN_MISSING`. A deleted experiment is reported as
`MLFLOW_EXPERIMENT_DELETED` and retried; restore it in MLflow to continue.

## Status

```bash
ml-exp run tracking --project my-study --run trial-001
```

`GET /api/runs/P/R/tracking` returns, for each scope of a Run whose route publishes
to MLflow, an `mlflow` object beside the W&B columns. It makes no MLflow request.

| Field | Meaning |
|---|---|
| `status` | `MLFLOW_PENDING`, `MLFLOW_ACK_PENDING`, `MLFLOW_SYNCED`, `MLFLOW_FINISHED`, `MLFLOW_RETRY_PENDING` or `MLFLOW_STOPPED` |
| `error` | The last failure's code, absent after a success |
| `confirmed`, `total`, `pending` | Records committed on MLflow, journal records, and the difference |
| `experiment`, `experiment_id`, `run_id`, `url` | Where the Attempt is published |
| `terminal_status` | `FINISHED` or `FAILED` once set |
| `failures`, `last_attempt_at`, `last_confirmed_at` | Failure count since the last success and times |
| `skipped_points`, `skipped_names`, `skipped_params` | What did not fit, as above |

| Status or error | Meaning |
|---|---|
| `MLFLOW_PENDING` | Nothing sent yet |
| `MLFLOW_ACK_PENDING` | Records are waiting or being sent |
| `MLFLOW_SYNCED` | Every record is committed; the Attempt has not ended |
| `MLFLOW_FINISHED` | Every record is committed and the run status is set |
| `MLFLOW_RETRY_PENDING` | The last visit failed; see `error` |
| `MLFLOW_STOPPED` | The run was deleted in MLflow; nothing more is sent |
| `MLFLOW_UNREACHABLE` | Connection failed or timed out; the batch is sent again |
| `MLFLOW_AUTH_FAILED` | MLflow answered 401 or 403; check `username` and `password_file` |
| `MLFLOW_REQUEST_REJECTED` | MLflow refused a request (4xx) |
| `MLFLOW_SERVER_ERROR` | MLflow answered 5xx |
| `MLFLOW_RESPONSE_INVALID` | A response was not the expected JSON |
| `MLFLOW_RUN_MISSING` | The Attempt's run is gone or deleted |
| `MLFLOW_EXPERIMENT_DELETED` | The project's experiment is deleted |
| `MLFLOW_RECONCILE_REQUIRED` | Stored state disagrees with the journal, or two MLflow runs carry the Attempt's tag |
| `JOURNAL_AUTHORITY_UNAVAILABLE` | The Run's journal authority could not be read |

W&B status and MLflow status are independent: each destination has its own cursor,
state and diagnostics, and one failing never holds back the other.

## Stored state

Each scope's MLflow state is one JSON document in the `mlflow_publications` table of
`tracking/outbox.sqlite`, beside the W&B state; the W&B columns of `scopes` are not
used. Keep `tracking/` with the service state. An older server ignores the table.
The publisher takes its own lock, `tracking/.mlflow-publisher.lock`.

## Limits and what is not covered

- MLflow's per-request limits are respected: at most 1000 metrics and 100 params per
  `runs/log-batch`, under 1 MB, keys up to 250 characters. Values are doubles.
- Find-before-create relies on there being one publisher; two servers publishing the
  same registry to one MLflow could create two runs, which is then reported as
  `MLFLOW_RECONCILE_REQUIRED` until one is deleted.
- MLflow publication does not backfill Runs created before it was configured, upload
  artifacts or models, or reflect [W&B rebuild](wandb.md#rebuild-a-finished-wb-run-without-some-metrics)
  exclusions for points it already sent. Records not yet sent are read from the same
  view the W&B publisher uses, so a rebuild's exclusions apply to them.
- There is no command that changes MLflow settings, and no repair command: replays
  are safe, so none is needed.
