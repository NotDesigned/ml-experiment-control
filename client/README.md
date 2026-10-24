# ML-Expd client

`ml-experiment-client` is independently installable, with no runtime dependency
outside stdlib Python. It provides `ml-exp` and `python -m ml_exp_client`.
The server/core, local Docker, SSH and backend credentials are unnecessary.

Follow the [quickstart](../docs/api-quickstart.md) for installation, API/token
configuration and an experiment. The operator supplies connection information.
Only the API URL and one token source are needed; see
[connection configuration](../docs/configuration.md#client-connect-to-an-existing-service).

Run `ml-exp check` for a basic API connection. Use `ml-exp check --training` to
also require training configuration and a responding private image builder;
it lists missing/disabled/unavailable components and exits 2 when incomplete.
Neither command builds an image or allocates a GPU. Backend authentication and
resource admission are still checked by the experiment's normal preflight.
The default `check` output is a short text summary; use `check --json` for the
full evidence in scripts, executor/base choices and limits. See the
[two-machine deployment example](../docs/two-machine-quickstart.md) for a fresh
basic API, private CA and independent client.

Upload source without requiring a builder or Dockerfile:

```bash
ml-exp source-import --project my-study --directory ./source
```

This returns the immutable source ID/tree digest and import metadata. Existing
archive limits and credential-path checks apply. It never builds an image or
submits a job. `pack` and the one-config `experiment` still upload automatically.

## Python use

```python
import os
from pathlib import Path
from ml_exp_client import Client, download

token = Path(os.environ["ML_EXPD_API_TOKEN_FILE"]).read_text().strip()
client = Client(os.environ["ML_EXPD_API_URL"], token)
health = client.negotiate()
executors = client.call("/api/executors")
# After checking the exact Attempt:
# download(client, "my-study", "trial-001", "attempt-001", Path("new-results"))
```

Exports: `Client`, `ClientError`, `source_archive`, `download`, `acknowledge`, `MetricWriter`. `call(path)` is
GET; `data=...` is JSON POST; `raw=...` is raw POST. Paths begin `/api/` and are
relative to the API base. Auth/protocol headers and proxy prefixes are handled.

CLI state preserves IDs. Ordinary requests default to 60 seconds; multipart
PUT/completion use 1200 seconds. Only archive transfer retries transient failures.
Uncertain scheduling is reconciled without replay. Signed object downloads carry
no API Authorization and are hash-checked into a new directory.
On `artifact-retention.v1` servers, verified downloads acknowledge durable local
files and authorize release of the server archive without a waiting period. Pass `--keep-server-copy`
(Python: `keep_server_copy=True`) to retain it. If acknowledgement is PENDING,
run `ml-exp acknowledge --directory results` to recheck saved files and retry
confirmation without redownloading. Old clients/servers do not authorize cleanup.
Checkpoint storage and experiment metadata remain separate. Keep your local
results; see [retention and recovery](../docs/recovery.md#download-confirmation-and-server-storage).
See [development](../docs/development.md) for independent build/tests.

## Data upload

New 1.0 servers require an executor for independent data upload:

```bash
ml-exp asset-upload --project my-study --executor slurm-gpu \
  --directory ./data --state data.state.json
# Repeat with --resume after an interrupted transfer, keeping the same directory.
```

The one-config `experiment` command selects its executor before uploading local
inputs. Slurm stores data on shared storage; SenseCore streams a data image to CCR
and prepares NAS before GPU submission. DLC uses approved OSS datasets and CPFS.
Slurm/SenseCore target-backed datasets have no client archive-download route;
OSS datasets support signed, hash-verified downloads without deletion. Keep the
original data. See [data and limits](../docs/data.md).

## Observe an experiment

```bash
ml-exp status --project my-study --run trial-001 --state trial.state.json
ml-exp status --project my-study --run trial-001 --attempt attempt-001 --details
ml-exp logs --project my-study --run trial-001 --attempt attempt-001
ml-exp metrics --project my-study --run trial-001 --attempt attempt-001
```

`status` is read-only and defaults to a compact JSON summary: preparation,
scheduler, training, publication and verification stay separate. `--state`
adds the original experiment's preparation/submission observations, including
before the Run exists. Without it, preparation can remain `NOT_OBSERVED`.
Omitting `--attempt` resolves the current Attempt once; a saved submission pins
its original Attempt. `--details` expands existing evidence without refreshing
providers or starting collection. Scheduler success does not imply completed
training or a downloadable archive. A failed collection stays visible even
when training completed.

Optional W&B publication: [configuration and per-Run projects](../docs/wandb.md).

## Delete a managed Run

Administrative deletion includes all Attempts, experiment records, persistent
checkpoints and retained result objects, including unacknowledged/`KEEP` archives.
Prepare and review the exact targets before explicitly executing:

```bash
ml-exp delete --project my-study --run trial-001 \
  --reason 'Remove this completed experiment' --state trial-001.deletion.json
ml-exp delete --state trial-001.deletion.json --execute
```

The server must support `resource-deletion.v1` and enable administrative deletion.
Active or uncertain Attempts and referenced checkpoints block deletion. Local
downloads, shared assets (including uploaded checkpoint snapshots), original
Action records, object receipts and W&B cloud records are retained. See the
[deletion contract and interruption recovery](../docs/resource-deletion.md).

## Training metrics

`ml-exp init source` includes a standalone `ml_exp_metrics.py` and a training
example that appends an explicit-unit metric at each step. The helper is the
same stdlib-only module exported as `ml_exp_client.MetricWriter`; training images
need no client installation, API token or W&B SDK/key. Copy the generated helper
alongside your training entry when adapting an existing project.

```python
from ml_exp_metrics import MetricWriter

metrics = MetricWriter()  # OUTPUT_DIR/metrics.jsonl, independently of STATE_DIR
metrics.log("train_loss", loss.item(), unit="nats/token", step=step,
            dataset_id="fineweb-train-v1")
metrics.log("validation_loss", total_nll / target_tokens, unit="nats/token",
            step=step, numerator=total_nll, denominator=target_tokens,
            dataset_id="fineweb-validation-v1", checkpoint_id=checkpoint_hash)
```

Declare these names/units in your Run's schema. Each call appends and flushes;
never overwrite the file with a final-only result. Keep recovery state in
`STATE_DIR` and stream metrics directly to `OUTPUT_DIR`, even when final weights
are exported later. A reporting rank should write your distributed aggregate.
Use a new Dockerfile Runtime with `experiment-records.v1` for live delivery.
See [metric meaning, aggregation and provenance](../docs/metrics.md).
