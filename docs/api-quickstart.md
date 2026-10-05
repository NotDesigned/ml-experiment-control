# Use an existing ml-expd API

You need Python 3.10 or later and an HTTPS connection. Your computer does not
need Docker, Rust, uv, SSH, SCO, Apptainer or a `ProjectAdapter`. All control
operations below use HTTP; the backend owns the platform credentials.

Ask the operator for:

| Item | Why it is needed |
| --- | --- |
| API base URL, including any proxy prefix | For example `https://api.example.org/ml-expd` |
| Bearer token | This grants access to the configured control plane; keep it private |
| Approved base image as `repository@sha256:<64 hex>` | Contains Python, your dependencies, `/bin/sh` and GNU `timeout` |
| Executor ID and allowed resources/budget | Profiles describe WYD or SenseCore infrastructure |
| Dataset location and access convention | Large data is provisioned by the operator, outside source upload |

There is no API for listing approved base images or building an environment
from `requirements.txt`. A client submits source and argv, not a Dockerfile.
The server packages source into the supplied environment with a fixed recipe.
If a dependency is missing, first ask the operator for a new environment image.

## Connect and inspect

Download [api_client.py](../examples/api_client.py) and the source files in
[api_project](../examples/api_project/train.py), or clone this repository. The
client uses only the Python standard library. Commands below assume the
repository root. If only downloading files, preserve the shown relative paths.
Obtain the revision containing these examples from the operator while the
source API changes are still under review.

```bash
export ML_EXPD_API_URL='https://api.example.org/ml-expd'
export ML_EXPD_API_TOKEN_FILE="$HOME/.config/ml-expd/client.token"
python3 examples/api_client.py check --schema openapi.json
```

Put the token received from the operator into that local file with mode `0600`;
do not generate a new token locally and expect the server to accept it.
Alternatively set `ML_EXPD_API_TOKEN` in your process environment. The client
never prints either value and refuses remote plaintext HTTP and redirects.
Health, policy, executor discovery and schema retrieval are read-only.

Every API request carries Bearer auth and `X-ML-Expd-Client-Protocol: 2`.
The client checks health's supported protocol range. Optional capabilities such
as `async-actions.v1` differ between protocol-2 deployments: it handles either
an immediate submission result or `EXECUTING` followed by polling. A profile in
`/api/executors` is a configuration entry, not proof that its scheduler is live.
Submission preparation supplies the actual preflight gates.

## Import source and package a Runtime

The supplied program writes three numeric metrics and a summary. It is an
output/transport smoke, not a model-quality or GPU benchmark. Replace it with
your training code once the workflow works. It needs only Python; the common
managed Run resource schema still requires at least one GPU on these executors.

```bash
export ML_EXPD_BASE_IMAGE='registry.example.org/team/python@sha256:<operator-provided-digest>'
export PROJECT="hello-$(date -u +%Y%m%dT%H%M%SZ)"
python3 examples/api_client.py pack \
  --project "$PROJECT" --source examples/api_project \
  --image "$ML_EXPD_BASE_IMAGE" --state runtime.json
```

Replace the image placeholder with an actual approved digest. `pack` uploads a
tar.gz and its SHA256, prepares an immutable Runtime with `python train.py`,
executes packaging and waits for `READY`. It allocates no GPU, but consumes
builder/registry resources. `runtime.json` is saved before packaging begins.
The server's `source_id` hashes source content, so it is different from the
uploaded tar's hash. Use a clean source folder: links, escaping paths and
credential-looking paths are rejected. This path policy is not a content
secret scanner. Defaults are 64 MiB uploaded / 256 MiB expanded / 20,000 entries;
do not upload a virtualenv, model cache or large dataset as source.

For a public Git repository, the equivalent import endpoint is
`POST /api/source-imports/git` with `project`, HTTPS `url`, and a full exact
40-character `commit`. Private repositories use archive upload; the API does
not accept client Git credentials.

## Freeze, review and submit a Run

Select an ID from `check`'s executor list. Here `wyd-l40s` is illustrative.

```bash
python3 examples/api_client.py create --runtime-state runtime.json \
  --run trial-wyd --executor wyd-l40s \
  --arguments '["--steps","3"]' --max-time 00:05:00
python3 examples/api_client.py prepare --project "$PROJECT" --run trial-wyd \
  --max-gpu-hours 0.1 --state submission-wyd.json
```

These commands freeze the definition and prepare review gates; they do not
submit a GPU job. Inspect `ready`, `gates`, the actual resources and
`gate_expires_at` in the printed result. SenseCore fixed allocations can be
larger than your requested CPU/memory minimum; the frozen Run reports them.
The operator's resource policy also applies. A five-minute one-GPU Run needs
at least 1 × 5 / 60 GPU-hours; `0.1` is the client budget in this example.

The next command authorizes and executes the exact saved Submission, and **can
allocate paid GPU resources**. Supply the exact confirmation printed by prepare:

```bash
python3 examples/api_client.py execute --state submission-wyd.json \
  --confirm 'EXECUTE <submission_id-from-prepare>'
python3 examples/api_client.py watch --project "$PROJECT" --run trial-wyd
```

`VERIFIED` means the exact scheduler job ID was observed. It does not mean
training finished. `watch` prints scheduler state and returns the Attempt list
when the Run reaches a terminal state. Inspect that state: `FAILED` is also
terminal. For logs, use the scoped identity from the list:

```bash
curl --fail --silent --show-error \
  -H "Authorization: Bearer $ML_EXPD_API_TOKEN" \
  -H 'X-ML-Expd-Client-Protocol: 2' \
  "$ML_EXPD_API_URL/api/attempts/$PROJECT/trial-wyd%3A%3Aattempt-001/logs?stream=both&lines=80"
```

The curl example uses an environment token; if using a token file, load it
into `ML_EXPD_API_TOKEN` in your shell without printing it. Run-level metrics:
`GET /api/runs/{project}/{run}/metrics`. Scheduler state, metrics, logs and
artifact availability are independent evidence; the API does not judge whether
an experiment supports a hypothesis.

## Retrieve and verify results

Use the exact Attempt you inspected, not an assumed latest Attempt:

```bash
python3 examples/api_client.py download --project "$PROJECT" --run trial-wyd \
  --attempt attempt-001 --out results-wyd
```

The destination must be new. It contains `outputs/metrics.jsonl`,
`outputs/summary.json`, `artifacts.tar` and `verification.json`. The client
streams individual output files, compares sizes, verifies the tar against its
receipt SHA256, and verifies that every archived file matches the individually
downloaded bytes. It does not extract untrusted tar paths. File ETags are cache
identities, not content SHA256; the archive ETag carries the receipt SHA256.
Partial downloads remain for diagnosis if verification fails; choose a new
destination to try retrieval again. This client expects S3-uploaded outputs,
not a legacy Run with only rsync evidence.

A terminal scheduler state can precede final collection. If files or the
archive are unavailable, inspect the exact Attempt and wait for collection;
do not start another experiment to repair a download. Writes under
`OUTPUT_DIR` are uploaded after the program exits. A hard kill or upload failure
can prevent results from arriving. There is no periodic live checkpoint upload.

## Repeat on SenseCore

Reuse the same Runtime, create a **different** Run ID, select the SenseCore
executor returned by `check`, then repeat prepare/execute/watch/download:

```bash
python3 examples/api_client.py create --runtime-state runtime.json \
  --run trial-sensecore --executor sensecore-1gpu \
  --arguments '["--steps","3"]' --max-time 00:05:00
python3 examples/api_client.py prepare --project "$PROJECT" --run trial-sensecore \
  --max-gpu-hours 0.1 --state submission-sensecore.json
```

Each backend pulls/converts the same derived image digest and runs the same
argv at `/workspace`. Backend profiles handle scheduler/storage differences.
Changing source, entrypoint, arguments or resource allocation requires a new
immutable definition. A retry uses another Attempt of the existing Run;
the retry endpoint prepares an Action and still requires authorization/execution.

## Recover without duplicate submission

The client never automatically retries a mutation. A connection timeout or
interrupted client process is not proof that the server did nothing.

| Situation | Next operation |
| --- | --- |
| Runtime packaging still `EXECUTING` | `runtime --state runtime.json` (GET polling) |
| Runtime `RECONCILE_REQUIRED` | Inspect builder, then `runtime --state runtime.json --reconcile` (receipt lookup) |
| Submission still `EXECUTING` | `submission --id <id>` (GET polling) |
| Submission `RECONCILE_REQUIRED` | `submission --id <id> --reconcile` (status observation only) |
| Execute response was lost | Inspect the saved Submission ID; never blindly prepare another submit |
| Prepare response was lost before saving its ID | List `/api/experiments/{project}/{run}/submissions` and inspect existing entries |
| Source imported but packaging prepare failed | State contains `source_id`; query source and prepare that Runtime explicitly |
| Program failed | Read exact Attempt logs/artifacts; choose an explicit retry or new definition |
| HTTP 401 / 426 / 409 | Check token / protocol / gate or identity conflict, respectively |

Check `resolution`, `safe_to_retry`, and `next_action` when the server advertises
`action-resolution.v1`. An uncertain mutation must first be reconciled. Poll
timeouts only stop the client; they do not cancel the server's work.

The [source API contract](source-api.md) describes all underlying endpoints;
the [HTTP contract](http_contract.md) explains authenticated schema retrieval
and the current limitations of the stock browser Swagger page.
