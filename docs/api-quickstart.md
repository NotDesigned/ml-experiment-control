# Run an experiment through the API

This is the path for a researcher using an existing service. Operators start with
the [operator guide](operator-guide.md). The [native usage guide](../rust/README.md)
has further examples, and `ml-exp <group> --help` gives exact flags.

## Install and connect

Ask the operator for an HTTPS API base URL (including any proxy prefix), a private
API token and an approved Project. You need no Python, Docker, SSH or backend
credentials on this machine.

Download `install-native.sh` from the
[latest release](https://github.com/NotDesigned/ml-experiment-control/releases/latest)
and replace `X.Y.Z` with that release's version. On Windows use
`./install-native.ps1 -Version X.Y.Z`.

```sh
bash install-native.sh --version X.Y.Z
ml-exp --url https://api.example/ml-expd config set --token-file /private/api-token \
  --project my-study --executor pai-h200
ml-exp backend    # the executors, and what each allows
```

The installer places `ml-exp` in `~/.local/bin` unless you pass `--prefix`. The saved
Project and executor are only defaults; see [configuration](configuration.md).

## Run a repository

The repository needs a Dockerfile whose external `FROM` lines are pinned by digest;
`ml-exp get /api/environments` lists approved bases, and the
[Dockerfile rules](builds.md#dockerfile-contract) say what else is required. The code
reads where to write from its environment:

```python
import json, os

with open(os.environ["METRICS_PATH"], "a") as metrics:
    metrics.write(json.dumps({"name": "loss", "value": 0.05, "unit": "dimensionless", "step": 1}) + "\n")
with open(os.path.join(os.environ["OUTPUT_DIR"], "summary.json"), "w") as summary:
    json.dump({"final_loss": 0.05}, summary)
```

Then one command does the rest:

```sh
ml-exp submit ~/my-repo --gpus 1 --time 2h -- python3 train.py
```

```text
Run my-repo-3fa81c2e
  executor  pai-h200
  command   python3 train.py
  image     registry.example/base@sha256:...
  source    42 files from /home/me/my-repo
  resources 1 GPU, 8 CPU, 32 GB
  time      up to 7200 seconds
  spend     up to 2 GPU hours
Proceed? [y/N]
```

After you answer, it uploads the source, builds the image, prepares and sends the job,
then shows the backend's status until the job ends. Anything already done is skipped.
Ctrl-C stops the watching, and a job already sent keeps running. The command fails if
the job does not end in success. Afterwards:

```sh
ml-exp status     # the latest submission from this machine
ml-exp logs
ml-exp results    # download and verify the results
ml-exp cancel     # ask, then stop it
ml-exp retry      # ask, then run it again; --checkpoint resumes from one
```

**Running it again.** The job's identity comes from the directory name, the code and the
settings, so repeating the same command goes back to the same job and sends nothing new.
This is what makes it safe to repeat after a lost connection. To run again, change the
code or settings, or pass `--seed` or `--run`. A failed job is run again with `ml-exp retry`.

**What is uploaded.** In a git repository, the files git tracks and the files it does not
ignore, with nested repositories following their own rules. Elsewhere, every file except
`.git`, `.venv` and `__pycache__`. The question shows how many. Keep data and outputs out;
the limits are 20,000 files, 256 MiB and 64 MiB compressed.

**Preparation needs the client.** Closing it during a build pauses the work, and repeating
the command resumes it. Use `--detach` to return right after the job is sent, and `--yes`
to skip the question in a script; without a terminal `--yes` is required.

### Options

| Need | Option |
| --- | --- |
| Another backend | `--executor pai-h200`. A prefix of fixed shapes, such as `sensecore-n6ls`, takes the smallest that fits |
| CPUs and memory | `--cpus`, `--memory`. Sized from `--gpus` unless given. A fixed shape always runs whole |
| Another Dockerfile | `--dockerfile docker/train.Dockerfile` |
| Environment variables | `--env KEY=VALUE`. Platform names are refused |
| An uploaded dataset | `--data train=asset.HASH`, from [`ml-exp data upload`](data.md); read its path from `DATASETS_JSON` |
| Everything else | `--file experiment.json` |

`--file` takes the experiment file below, with everything `submit` derives left out, for
metrics, W&B, scheduling and required files. Options win over the file, and the file over
saved settings. It cannot set `request_id`, `source_id`, `base_image` or `runtime_id`; to
fix those, use the steps below.

## Each step separately

`submit` runs existing operations in order. Run them yourself to reuse a Runtime, to use a
fixed source, or to look at each step. Keep each kind of file separate:

```text
study/
  source/                 # Dockerfile, train.py
  data/                   # optional dataset, separate upload
  experiment.json
  trial.state.json        # created by the client; preserve for recovery
  results/                # downloads; never include in source
```

Upload the source and keep the returned source ID:

```sh
ml-exp source import --project my-study --directory ./source
```

Create `experiment.json` using an executor from `ml-exp backend`, the source ID and
the digest of your Dockerfile's final `FROM`:

```json
{
  "request_id": "11111111111111111111111111111111",
  "runtime": {
    "source_id": "source.REPLACE_WITH_64_HEX",
    "executor": "slurm-gpu",
    "base_image": "registry.example/base@sha256:REPLACE_WITH_64_HEX",
    "dockerfile": "Dockerfile"
  },
  "run": {
    "run_id": "trial-001",
    "executor": "slurm-gpu",
    "command": ["python3", "train.py"],
    "resources": {"gpus": 1, "cpus": 8, "memory_gb": 32},
    "walltime_seconds": 600,
    "seed": 42,
    "metrics_schema": {"definitions": {"loss": {"unit": "dimensionless"}}},
    "artifacts": {"required_files": ["summary.json"]}
  },
  "approval": {"max_gpu_hours": 0.2, "reason": "bounded validation"}
}
```

`request_id` is 32 lowercase hex characters and is a durable recovery slot;
changing the payload under the same ID is a conflict. For an existing Runtime,
omit `runtime` and set `run.runtime_id`. The executor decides whether the requested
resources fit. Slurm rejects the cloud-only `scheduling` option.

Preparation, authorization and dispatch are separate commands:

```sh
ml-exp run start --project my-study --file experiment.json --state trial.state.json
ml-exp run continue PREPARATION_ID --project my-study --confirmation 'PREPARE PREPARATION_ID'
ml-exp run authorize SUBMISSION_ID --reason 'bounded validation'
ml-exp run execute SUBMISSION_ID --confirmation 'CONFIRMATION_RETURNED_BY_THE_API'
```

`run start` saves the recovery identity before sending anything. `run continue`
builds or reconciles the environment, stages data and prepares a Submission. Repeat
it for the same preparation until a Submission is returned. It does not authorize
training. Use the IDs and confirmations the API returns. Closing the client does
not cancel a submitted job.

W&B training publication is requested by default but stays off unless the operator
configured it. Set `"wandb": {"enabled": false}` in `run` to opt out; see [W&B](wandb.md).

## Observe the exact execution

```sh
ml-exp run status --state trial.state.json
ml-exp run status --state trial.state.json --watch --seconds 60
ml-exp run status --state trial.state.json --details --json
ml-exp run logs --state trial.state.json --stream both --lines 80
ml-exp run metrics --state trial.state.json --name loss --max-points 2000
```

These reads use the original identities from state. Without state, pass `--project`,
`--run` and `--attempt`. They do not contact the provider or submit work: the
optional background observer or an explicit `ml-exp run reconcile SUBMISSION_ID`
refreshes the original job. HTTP 503 `CLOUD_RUN_BUSY` means repeat the same query.

| Observation | What it establishes |
|---|---|
| Runtime `READY` | An image was published |
| Submission `VERIFIED` | The exact scheduler job is visible |
| Scheduler `RUNNING` | The job has started; training may still be preparing data |
| Training `COMPLETED`, exit code 0 | Worker training completion evidence exists |
| Collection `AVAILABLE`, `SHA256_VERIFIED` | Results can be downloaded |

`--watch` observes the same Attempt for a bounded period. It never submits or
restarts a job. Logs may be delayed or unavailable; missing evidence remains
unknown. See [recovery](recovery.md).

## Download and acknowledge

```sh
ml-exp run files SUBMISSION_ID
ml-exp run file SUBMISSION_ID --path summary.json --output results/summary.json
ml-exp run download SUBMISSION_ID --output results/trial-001.tar
```

`run files` and `run file` send no ACK and authorize no release. `run download`
verifies the archive and every member, fsyncs the local file, then acknowledges; by
default that releases the generated transfer copy. Use `--keep` to retain it. ACK
does not delete datasets, persistent checkpoints or the metric journal. If the ACK
is lost, repeat the same command: it reconciles without downloading again.

When results are not published yet, use `ml-exp run artifacts SUBMISSION_ID` and
`ml-exp run collect SUBMISSION_ID`, or `run sync` for cloud Runs; see
[recovery](recovery.md). A failed or uncertain collection needs diagnosis, not
another training Attempt.

## Add data, checkpoints and scientific metrics

| Need | Run field or convention | Reference |
|---|---|---|
| Upload local training data | `ml-exp data upload --project P --executor E --directory ./data` | [Data](data.md) |
| Use an uploaded asset | `"data": [{"name": "training", "asset_id": "asset.HASH"}]`; read the path from `DATASETS_JSON` | [Data](data.md) |
| Download on the backend | `data_preparation` script writes `DATA_DIR` | [Native usage guide](../rust/README.md#one-experiment-configuration) |
| Keep complete recovery state | Atomically publish a ready generation in `STATE_DIR`; register with `ml-exp run checkpoint SUBMISSION_ID` | [Checkpoints](persistent-checkpoints.md) |
| Restore an exact checkpoint | New Run with `resume_from`, or `ml-exp run retry` with `--checkpoint` | [Restore](persistent-checkpoints.md#register-and-restore) |
| Emit curves and chosen results | JSONL records in `METRICS_PATH` | [Metrics](metrics.md) |
| Require exported files | `artifacts.required_files` | [Result recovery](recovery.md) |
| Run a parameter matrix | `ml-exp run batch` | [Batches](../rust/README.md#batches) |

## Recover safely

Keep the original experiment file and state. Lost responses are observed through
saved identities: use `ml-exp run preparation --project P --run R` or
`ml-exp run submission SUBMISSION_ID`, then `run reconcile`. Uncertain scheduler
requests are not replayed. Changed code or configuration needs a new Run ID,
request ID and state file.

Cancel and retry are explicit:

```sh
ml-exp run cancel SUBMISSION_ID --confirmation 'CONFIRMATION_RETURNED_BY_THE_API'
ml-exp run retry SUBMISSION_ID --run trial-001 --attempt attempt-001 --from-start
```

Use `--checkpoint CHECKPOINT_ID` instead of `--from-start` to restore saved state.
A retry creates new Attempt metadata and does not submit training or choose a
checkpoint for you. An expired authorization gate needs `ml-exp run renew`. See
[recovery](recovery.md#cancel-or-retry-an-exact-attempt) and
[administrative deletion](resource-deletion.md).
