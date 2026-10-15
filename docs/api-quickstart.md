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
  source/ml_exp_metrics.py
  experiment.json
  trial.state.json       # created by client, outside source
  results/               # downloads, outside source
```

Generate a starter, replacing the placeholder with an approved digest:

```bash
ml-exp init source --base-image 'REGISTRY/BASE@sha256:ACTUAL_64_HEX_DIGEST'
```

The starter uses stdlib Python and appends an explicit-unit metric after every
smoke step, immediately visible in OUTPUT_DIR/metrics.jsonl. Keep its generated
ml_exp_metrics.py helper when replacing train.py with your training code.
It verifies the workflow, not CUDA availability
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
  "metrics_schema": {"definitions": {"loss": {"unit": "dimensionless"}}},
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
authorizes scheduling within the budget. `--seconds` limits each client wait stage;
`resources.max_time` controls worker/job duration. Disconnecting does not cancel.

With `--download-to`, scheduler completion is followed by observation of the
original Attempt's collection. Download starts only when it is `AVAILABLE`,
`verification=SHA256_VERIFIED` and `download_available=true`. This wait can include
the server's automatic result collection; the client does not request collection
or create another Attempt. `FAILED` or `RECONCILE_REQUIRED` requires inspection;
`RELEASED` cannot be downloaded again. A publication timeout preserves the exact
Attempt and destination in the state file. Repeat the same command with
`--resume --execute` to continue observing it. Servers without
`result-collection.v1` retain the older direct-download behavior.

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

`outputs` selects files to export; it does not promise that every matching file
exists. Use `required_outputs` for the files your experiment must publish, for
example `{"required_outputs": ["summary.json", "weights/*.pt", "updates.jsonl"]}`.
Each pattern must match at least one regular file. These patterns are frozen
with the Run, included in the export selection, and checked before publication.
The platform does not prescribe model filenames or scientific success criteria.

For an original job's persisted provider evidence, query
`GET /api/runs/{project}/{run}/attempts/{attempt}/diagnostics`; add `?refresh=true`
to capture current provider events and logs. No compute is submitted. An
`ExitedWithCode` event is accepted only for the exact job UID and identifies its
worker/container. Empty delayed logs remain pending, and later log expiry does
not erase previously captured evidence. Missing historical stderr cannot be
reconstructed from an exit code.

A byte-verified archive can still lack intended scientific outputs. If an
unacknowledged archive was published from an incomplete mirror and the complete
outputs are now collected, an operator can call
`POST /api/runs/{project}/{run}/attempts/{attempt}/collection/repair` with
`{"expected_sha256": "CURRENT_ARCHIVE_SHA256"}`. Repair only adds files, verifies
every old file is unchanged, preserves the old receipt/object, and switches the
download to the expanded archive. It rejects acknowledged/retained/released
archives and active Attempts; it never launches training. This request may take
time to package and validate; after a lost response, query collection before
retrying the same digest.

If preparation's response is lost, list
`GET /api/experiments/{project}/{run}/submissions` and query the original
`GET /api/submissions/{submission_id}`. A valid prepared review returns
`AUTHORIZE`; an expired review returns `REPREPARE`. Reprepare the same Run to
obtain current gates instead of executing expired approval. Never create a
second Run or replay an uncertain scheduler submission.

## Observe and retrieve

```bash
ml-exp status --project my-study --run trial-001 --state trial.state.json
ml-exp status --project my-study --run trial-001 --attempt attempt-001 --details
ml-exp submission --id SUBMISSION_ID
ml-exp watch --project my-study --run trial-001 --seconds 1800
ml-exp logs --project my-study --run trial-001 --attempt attempt-001 --stream both --lines 80
ml-exp metrics --project my-study --run trial-001 --attempt attempt-001 --keys loss --max-points 2000
ml-exp download --project my-study --run trial-001 \
  --attempt attempt-001 --out results/trial-001-copy
```

`status` returns a compact summary of preparation, scheduler, training,
publication and verification. `--state` observes the original saved preparation
and submission, including while the Run has not yet been created. A saved
submission pins its exact Attempt; otherwise omitting `--attempt` resolves the
current Attempt once. Without experiment state, preparation can remain
`NOT_OBSERVED`. `--details` adds existing API evidence; it neither refreshes
provider logs nor starts result collection. Training stays `UNKNOWN` when there
is no completion evidence, even if the scheduler reports `SUCCEEDED`. Publication
is downloadable only when collection and hash verification are both complete.
A failed collection is shown separately from successful training.

Use the actual Attempt ID and a new destination directory. Download verifies the
archive and recorded file hashes. Failed jobs can still have useful outputs.
Runtime READY means image publication; Submission VERIFIED means exact scheduler
job visibility; neither proves training or result publication succeeded.
`watch` stops at a scheduler terminal state and does not wait for publication.
`logs` and `metrics` are reads; omitting `--attempt` resolves the current Attempt
once. Metrics retain their recorded units, protocols and variants.

After disconnecting, use the same state with `--resume`. It observes saved
identities without replaying uncertain build/scheduler requests. New state files
also bind the server and Run; historical experiment state remains readable.
Authorization and scheduling each save a durable request marker before dispatch.
A lost reply is inspected through the original Submission instead of repeating
the request. Changed code
or config needs a new Run ID and state file. See [recovery](recovery.md) for
`RECONCILE_REQUIRED`, empty results and stalled preparation.

A completed download with a `PENDING` acknowledgement can be resumed with the
same experiment state and `--resume`, even without `--execute`. The client verifies
the saved local archive and files, then retries only ACK, preserving the saved
KEEP/release choice. It does not download again. Keep the results directory and
`verification.json`; `ml-exp acknowledge --directory results/trial-001` performs
the same local verification and ACK retry directly.

ACK releases only the downloaded archive. Experiment records and backend
checkpoint state remain. To remove an entire completed managed Run, use the
separately enabled, reviewed [administrative deletion](resource-deletion.md)
workflow; it can also remove result objects without a download ACK.

Cancellation and retry use separately saved plans. The first command prepares an
exact-Attempt Action for review; only the second explicit invocation executes it:

```bash
ml-exp cancel --project my-study --run trial-001 --attempt attempt-001 --state cancel.state.json
# Inspect the returned Action, scope, gates and next_action first.
ml-exp cancel --state cancel.state.json --execute

ml-exp retry --project my-study --run trial-001 --attempt attempt-001 \
  --max-gpu-hours 0.2 --state retry.state.json
# Inspect the new Attempt and budget first.
ml-exp retry --state retry.state.json --execute
```

Repeating either command with only `--state` observes its original Action. Use
`--reconcile` for an uncertain original Action. Retry creates a new Attempt in the
same frozen Run and does not select a checkpoint. To change the frozen checkpoint,
create a new Run with an explicit `resume_from` reference. See [safe operation
recovery](recovery.md#cancel-or-retry-an-exact-attempt).

Lower-level commands are `pack → create → prepare → execute → watch → download`.
Use `ml-exp COMMAND --help` and the [API reference](source-api.md) when reusing
one Runtime across multiple Runs or integrating a custom client.
The [HTTP contract](http_contract.md#structured-errors-and-execution-capacity)
describes bounded execution queues and safe structured SDK errors.

## Interrupted preparation

Client state records `preparation_id`, dependency IDs and the submitted intent.
`--resume` observes the same preparation. If it reports `RECONCILE_REQUIRED`, inspect
its Runtime/data delivery first; an explicit `--resume --continue-preparation`
advances independently READY dependencies without replaying uncertain effects.
`--execute` remains separate authorization for GPU training.
