# Run an experiment from another computer

## Connect

The operator supplies an HTTPS API base URL, private Bearer token, client wheel
and enabled executors/base images. You need Python 3.10+; CI uses 3.12. Local
Docker, SSH and cloud credentials are unnecessary.

```bash
python3 -m venv .client-venv
. .client-venv/bin/activate
python -m pip install ./ml_experiment_client-*.whl
export ML_EXPD_API_URL='https://api.example.org/ml-expd'
export ML_EXPD_API_TOKEN_FILE="$HOME/.config/ml-expd/client.token"
ml-exp check --schema openapi.json
```

Provision that private token file first. Keep it outside source and experiment
JSON. From a checkout, `python -m pip install ./client` installs only the client.
`check` returns capabilities, policy, executors, base digests and storage limits.
Choose actual IDs/digests from those responses; executors are not free-GPU counts.

## First experiment

Keep source, data, client state and results separate:

```text
study/
  source/Dockerfile
  source/train.py
  experiment.json
  trial.state.json       # created by client, outside source
  results/               # downloads, outside source
```

Generate a starter, replacing the placeholder with an approved digest:

```bash
ml-exp init source --base-image 'REGISTRY/BASE@sha256:ACTUAL_64_HEX_DIGEST'
```

The starter uses stdlib Python. It verifies the workflow, not CUDA availability
or GPU throughput. Replace it with training code and dependency installation
afterwards. See [Dockerfile rules](builds.md#dockerfile-contract).

Write `experiment.json`:

```json
{
  "project": "my-study",
  "run_id": "trial-001",
  "source": "./source",
  "dockerfile": "Dockerfile",
  "entrypoint": ["python3", "train.py"],
  "executor": "wyd-l40s",
  "arguments": ["--steps", "20"],
  "resources": {"gpus": 1, "cpus": 8, "memory_gb": 32, "max_time": "00:10:00"},
  "max_gpu_hours": 0.2
}
```

Replace the executor with one published by your server. SenseCore GPU count
must match the fixed spec; actual CPU/memory may exceed requested minimums.
Local paths are relative to the config file. Instead of `executor`, use
`"executor_selector": {"backend": "slurm"}` or an ordered `candidates` list.
Optional `requirements` can demand persistent checkpoints or reliable exit codes;
matching rejects unsupported capabilities before upload/build. See [backend
contracts and preparation](backends.md).

```bash
ml-exp experiment experiment.json --state trial.state.json
ml-exp experiment experiment.json --state trial.state.json --resume \
  --execute --seconds 1800 --download-to results/trial-001
```

The first command uploads source/data and submits one preparation request. The
server matches the executor, builds/reuses the image, prepares data, freezes the
Run and prepares gates. It does not submit GPU training. Adding desktop SenseCore inputs
can launch a bounded zero-GPU data-copy job during preparation. `--execute`
authorizes scheduling within the budget. `--seconds` is the client's wait limit;
`resources.max_time` controls worker/job duration. Disconnecting does not cancel.

## Add data and outputs

Code is packaged at `/workspace`. Write exports to `OUTPUT_DIR` (`/outputs`).
Add data when needed:

```json
{"inputs": [{"directory": "./data", "mount_path": "/inputs/data"}]}
```

Use `asset_id` instead of `directory` to reuse an upload. A source download
script can instead prepare persistent `DATA_DIR`. See [data](data.md).
Use `STATE_DIR` for complete recovery state ([checkpoints](persistent-checkpoints.md)),
and explicit units for [metrics](metrics.md). These are optional additions.

## Observe and retrieve

```bash
ml-exp submission --id SUBMISSION_ID
ml-exp watch --project my-study --run trial-001 --seconds 1800
ml-exp download --project my-study --run trial-001 \
  --attempt attempt-001 --out results/trial-001-copy
```

Use the actual Attempt ID and a new destination directory. Download verifies the
archive and recorded file hashes. Failed jobs can still have useful outputs.
Runtime READY means image publication; Submission VERIFIED means exact scheduler
job visibility; neither proves training or result publication succeeded.

After disconnecting, use the same state with `--resume`. It observes saved
identities without replaying uncertain build/scheduler requests. Changed code
or config needs a new Run ID and state file. See [recovery](recovery.md) for
`RECONCILE_REQUIRED`, empty results and stalled preparation.

Lower-level commands are `pack → create → prepare → execute → watch → download`.
Use `ml-exp COMMAND --help` and the [API reference](source-api.md) when reusing
one Runtime across multiple Runs or integrating a custom client.

## Interrupted preparation

Client state records `preparation_id`, dependency IDs and the submitted intent.
`--resume` observes the same preparation. If it reports `RECONCILE_REQUIRED`, inspect
its Runtime/data delivery first; an explicit `--resume --continue-preparation`
advances independently READY dependencies without replaying uncertain effects.
`--execute` remains separate authorization for GPU training.
