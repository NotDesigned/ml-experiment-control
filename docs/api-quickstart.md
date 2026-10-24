# Run an experiment through the API

This is the complete path for a researcher using an existing service. Operators
should start with [deployment](operator-guide.md); custom API clients should use
this workflow before the [resource reference](source-api.md).

## Install and connect

Ask the operator for an HTTPS API base URL (including any proxy prefix), a private
Bearer token, a reviewed client wheel and an enabled executor/base-image choice.
Use Python 3.10+; no local Docker, SSH or backend credentials are required.

```bash
python3 -m venv .client-venv
. .client-venv/bin/activate
python -m pip install ./ml_experiment_client-*.whl
export ML_EXPD_API_URL='https://api.example.org/ml-expd'
export ML_EXPD_API_TOKEN_FILE="$HOME/.config/ml-expd/client.token"
ml-exp check --schema openapi.json
ml-exp check --json
ml-exp check --training
```

Provision the private token file first, outside source and data. Configure exactly
one token source: `ML_EXPD_API_TOKEN_FILE` or `ML_EXPD_API_TOKEN`.
[Configuration](configuration.md) also covers private CAs. From a checkout,
`python -m pip install ./client` installs only the client.

`check` prints a short summary; `--json` includes full capabilities, executors,
base digests and limits. `--training` also checks training configuration and the
builder, returning 2 when incomplete. It does not allocate a GPU or prove that
resources are immediately available. Select actual executor IDs and immutable
base-image digests from your service.

## First experiment

Use a new working directory and keep each kind of file separate:

```text
study/
  source/                 # Dockerfile, train.py, ml_exp_metrics.py
  data/                   # optional dataset, separate upload
  experiment.json
  trial.state.json        # created by client; preserve for resume
  results/                # downloads; never include in source
```

Generate a starter after replacing the base-image placeholder:

```bash
ml-exp init source --base-image 'REGISTRY/BASE@sha256:ACTUAL_64_HEX_DIGEST'
```

The starter writes `loss` curve points and `summary.json` using stdlib Python.
It tests the platform path, not CUDA or training throughput. Keep its generated
`ml_exp_metrics.py` helper when replacing `train.py`; add dependencies to the
Dockerfile. The server adds the managed launcher and copies source to
`/workspace`. See [Dockerfile rules](builds.md#dockerfile-contract).

Create `experiment.json`, replacing `slurm-gpu` with a published executor:

```json
{
  "project": "my-study",
  "run_id": "trial-001",
  "source": "./source",
  "dockerfile": "Dockerfile",
  "entrypoint": ["python3", "train.py"],
  "arguments": ["--steps", "20"],
  "executor": "slurm-gpu",
  "resources": {"gpus": 1, "cpus": 8, "memory_gb": 32, "max_time": "00:10:00"},
  "max_gpu_hours": 0.2,
  "metrics_schema": {"definitions": {"loss": {"unit": "dimensionless"}}},
  "outputs": ["metrics.jsonl", "summary.json"],
  "required_outputs": ["summary.json"]
}
```

Local paths are relative to the JSON file. A fixed SenseCore worker specification
must support the requested GPU count; its CPU/memory allocation can exceed the
requested minimum. Optional `executor_selector` and capability `requirements`
are described in [backend preparation](backends.md).

Prepare first, inspect the returned executor, dependencies, gates and budget,
then authorize execution with the same saved state:

```bash
ml-exp experiment experiment.json --state trial.state.json
ml-exp experiment experiment.json --state trial.state.json --resume \
  --execute --seconds 1800 --download-to results/trial-001
```

The first command uploads source/data and requests server preparation: build or
reuse a Runtime, prepare data, freeze the Run and review its Submission. It does
not submit GPU training. Preparation can run a bounded zero-GPU data-copy job.
`--execute` authorizes and submits the reviewed experiment; `--seconds` limits each
client wait stage, while `resources.max_time` bounds the backend job. Closing the
client does not cancel a submitted job.

W&B training publication is requested by default but is effectively disabled
without server credentials/configuration. Preparation publication defaults off.
You can explicitly set `"wandb": {"enabled": false}`; see [W&B](wandb.md).

## Observe the exact execution

Use another terminal with the same connection settings:

```bash
ml-exp status --project my-study --run trial-001 --state trial.state.json
ml-exp status --project my-study --run trial-001 --attempt attempt-001 --details
ml-exp logs --project my-study --run trial-001 --attempt attempt-001 --stream both --lines 80
ml-exp metrics --project my-study --run trial-001 --attempt attempt-001 --name loss --max-points 2000
```

Replace `attempt-001` with the returned ID. `status --state` includes the original
preparation/submission, even before a Run exists. Without it, preparation can be
`NOT_OBSERVED`. An explicit Attempt prevents accidentally reading a later retry.
`--details` expands existing evidence; it neither refreshes providers nor starts
collection. Metrics use normalized records, including units and provenance.

| Observation | What it establishes |
|---|---|
| Runtime `READY` | An image was published |
| Submission `VERIFIED` | The exact scheduler job is visible |
| Scheduler `RUNNING` | The job has started; training may still be preparing data |
| Training `COMPLETED`, exit code 0 | Worker training completion evidence exists |
| Collection `AVAILABLE`, `SHA256_VERIFIED`, `download_available=true` | Results can be downloaded |

`ml-exp watch --project my-study --run trial-001 --seconds 1800` waits for a
scheduler terminal state; it does not wait for result publication. Logs may be
delayed or unavailable. Missing evidence remains unknown rather than implying
success. See [recovery](recovery.md) for diagnostics and collection.

## Download and acknowledge

The executing `experiment` command above waits for the original Attempt's result
collection and downloads when ready. To download separately, choose a new path:

```bash
ml-exp download --project my-study --run trial-001 \
  --attempt attempt-001 --out results/trial-001-copy
```

The client verifies archive and file hashes, writes `verification.json`, then
acknowledges the local copy. By default ACK authorizes server archive release
without a waiting period, subject to configured archival policy. Use
`--keep-server-copy` to retain that archive. ACK does not delete data assets,
persistent checkpoints or the metric journal; keep your local results safe.

If download succeeded but ACK is pending, preserve the directory and retry only
confirmation:

```bash
ml-exp acknowledge --directory results/trial-001
```

With `--download-to`, a timeout preserves the exact Attempt and destination.
Repeat the original `experiment` command with `--resume --execute` to keep
observing; ACK-only recovery also works with `--resume` without `--execute`.
A failed or uncertain collection needs diagnosis, not another training Attempt.
A `RELEASED` archive is no longer available through the same download route.

## Add data, checkpoints and scientific metrics

| Need | Experiment configuration / convention | Reference |
|---|---|---|
| Upload local training data | `"inputs": [{"directory":"./data","mount_path":"/inputs/data"}]` | [Data](data.md) |
| Reuse an asset | Replace `directory` with its returned `asset_id` | [Data](data.md) |
| Download on the backend | `data_preparation` script writes persistent `DATA_DIR` | [Data preparation](data.md) |
| Keep complete recovery state | `checkpoint_persistence`; atomically publish a ready generation in `STATE_DIR` | [Checkpoints](persistent-checkpoints.md) |
| Restore an exact checkpoint | New Run with `resume_from` referencing Run/Attempt/checkpoint | [Restore](persistent-checkpoints.md#register-and-restore) |
| Emit curves and chosen results | `MetricWriter.log()` and `.summary()` to `OUTPUT_DIR/metrics.jsonl` | [Metrics](metrics.md) |
| Require exported files | `required_outputs`; every pattern must match a regular file | [Result recovery](recovery.md) |

Slurm data is staged on shared storage accessible to offline compute nodes.
SenseCore uses a CCR data image and a zero-GPU NAS-copy job. PAI can use approved
OSS datasets and CPFS; query capabilities rather than assuming native OSS mounts.
Data and recovery state are separate from exported `OUTPUT_DIR` results.
For independent transfers use `asset-upload`; for source-only service testing,
`source-import` requires neither Dockerfile nor builder.

## Resume safely or reuse a Runtime

Keep the original config, source and state unchanged when using `--resume`.
Lost responses are observed through the saved identities; uncertain scheduler
requests are not replayed. If preparation is `RECONCILE_REQUIRED`, inspect its
Runtime/data dependency before using explicit `--resume --continue-preparation`.
This is separate from authorization to train. Changed code or experiment config
needs a new Run ID and state file.

For lower-level reuse, the sequence is `pack → create → prepare → execute`.
`prepare` returns review gates and an exact confirmation; `execute --state …
--confirm …` authorizes/submits that saved Submission. See command help and the
[resource reference](source-api.md), rather than recreating a Runtime per Run.
Cancel/retry use separately reviewed plans. Retry adds an Attempt to the frozen
Run; it does not automatically choose a new checkpoint. See
[safe operation recovery](recovery.md#cancel-or-retry-an-exact-attempt) and
[administrative deletion](resource-deletion.md).
