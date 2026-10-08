# ML-Expd client

`ml-experiment-client` is independently installable, with no runtime dependency
outside stdlib Python. It provides `ml-exp` and `python -m ml_exp_client`.
The server/core, local Docker, SSH and backend credentials are unnecessary.

Follow the [quickstart](../docs/api-quickstart.md) for installation, API/token
configuration and an experiment. The operator supplies connection information.

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
files and release the server archive after 24 hours. Pass `--keep-server-copy`
(Python: `keep_server_copy=True`) to retain it. If acknowledgement is PENDING,
run `ml-exp acknowledge --directory results` to recheck saved files and retry
confirmation without redownloading. Old clients/servers do not authorize cleanup.
Checkpoint storage and experiment metadata remain separate. Keep your local
results; see [retention and recovery](../docs/recovery.md#download-confirmation-and-server-storage).
See [development](../docs/development.md) for independent build/tests.

Optional W&B publication: [configuration and per-Run projects](../docs/wandb.md).

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
