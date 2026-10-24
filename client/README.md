# ML-Expd client

`ml-experiment-client` provides `ml-exp` and `python -m ml_exp_client` on Python
3.10+. It has no runtime dependencies. Client machines need the API base URL and
a private token; they need no server package, local Docker, SSH or cloud credentials.

Follow the [client quickstart](../docs/api-quickstart.md) for the complete path:
install → connect → upload code/data → prepare → authorize/submit → observe →
download and verify. Connection precedence and private CA settings are in
[configuration](../docs/configuration.md#client-connect-to-an-existing-service).

## Command map

| Goal | Commands |
|---|---|
| Check connectivity and configured choices | `check`, `check --json`, `check --training` |
| Generate a training starter | `init` |
| Prepare and run one experiment JSON | `experiment`, then resume its saved state |
| Reuse a built Runtime across Runs | `pack`, `create`, `prepare`, `execute` |
| Upload source independently of a builder | `source-import` |
| Manage data | `asset-upload`, `assets`, `asset-import-oss`, `asset-download` |
| Observe the exact execution | `status`, `submission`, `logs`, `metrics`, `checkpoints`, `snapshots` |
| Retrieve outputs | `download`, `acknowledge`, `collect` |
| Review a state-changing operation | `cancel`, `retry`, `delete` |
| Configure or inspect W&B publication | `wandb`, `tracking` |
| Request a temporary debugging machine | `debug` |

Use `ml-exp COMMAND --help` for flags. `check` is a short text summary;
`check --json` provides full machine-readable evidence. `check --training` checks
training configuration and builder connectivity without allocating a GPU; it is
not a backend resource-admission guarantee.

`status`, `cancel` and `retry` print short human-readable summaries by default;
use `--json` for their structured results. `status --details --json` includes the
full existing evidence. Other low-level commands keep their JSON output.

Once an experiment has saved its identities, observing or downloading it does
not require the original source directory or repeating project/Run/Attempt IDs:

```bash
ml-exp status --state trial.state.json
ml-exp status --state trial.state.json --watch --seconds 60
ml-exp logs --state trial.state.json --lines 80
ml-exp metrics --state trial.state.json --name loss
ml-exp download --state trial.state.json --out results/trial-001
```

Observation stays bound to the saved Attempt. `--watch --seconds` limits client
observation only; it neither restarts training nor switches to a later retry.
Before submission, `status --state` can inspect preparation without an Attempt;
logs, metrics and download require a bound Attempt.

Keep source, data, saved client state and downloaded results in separate
directories. State files preserve original identities after a lost reply; do not
replace them to retry an uncertain request. The [recovery guide](../docs/recovery.md)
explains continuation, reconciliation, result collection and reviewed cancel/retry.
For example, `ml-exp retry --from-state trial.state.json --state retry.state.json
--max-gpu-hours 0.2` creates a separate plan for review; only a subsequent
`ml-exp retry --state retry.state.json --execute` authorizes execution. Retry
keeps the frozen Run's original checkpoint reference; it does not choose the
latest checkpoint. An uncertain operation must be reconciled before another
execution is considered.

## Training metrics

`ml-exp init source` includes the same stdlib-only `MetricWriter` exposed by the
client package. Training code appends to `OUTPUT_DIR/metrics.jsonl` using `log()`
for curve points and `summary()` for an explicitly selected comparison result.
Neither requires client installation, network access or a W&B key in the image.
See the [metric contract](../docs/metrics.md) and [W&B display](../docs/wandb.md).

## Python API

```python
import os
from pathlib import Path
from ml_exp_client import Client, download

token = Path(os.environ["ML_EXPD_API_TOKEN_FILE"]).read_text().strip()
client = Client(os.environ["ML_EXPD_API_URL"], token)
health = client.negotiate()
executors = client.call("/api/executors")
# After confirming the exact Attempt and result availability:
# download(client, "my-study", "trial-001", "attempt-001", Path("new-results"))
```

Public exports: `Client`, `ClientError`, `source_archive`, `download`,
`acknowledge`, `MetricWriter`. `call(path)` performs GET; `data=...` sends JSON
POST; `raw=...` sends raw POST. Paths start `/api/` relative to the configured base.
The client handles protocol headers and proxy prefixes. Signed object downloads
carry no API Authorization and are checked against archive and file hashes.

Verified downloads acknowledge local files and normally authorize release of the
server archive. Use `--keep-server-copy` (`keep_server_copy=True` in Python) to
retain it. If ACK is pending, `ml-exp acknowledge --directory RESULTS` verifies
saved files and retries only confirmation. This does not delete backend
checkpoints, datasets or experiment records. See [retention](../docs/recovery.md#download-confirmation-and-server-storage).
