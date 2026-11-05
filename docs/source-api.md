# API resource and diagnostic reference

For a complete install-to-results example use the [client quickstart](api-quickstart.md).
This page is a lookup table for custom clients, Runtime reuse and diagnosis;
request/response bodies are defined by the installed `GET /api/v2/openapi.json`.
Paths are relative to the API base. `P`, `R`, `A`, `S` denote returned project,
Run, Attempt and Submission IDs, not filesystem paths.

Send the API Bearer credential and `X-ML-Expd-Client-Protocol: 2` on control
requests. `GET /api/health` is the protocol-discovery exception; see the
[HTTP contract](http_contract.md) for auth, errors and asynchronous capacity.

## Discover, import and build

| Method/path | Resource or purpose |
|---|---|
| `GET /api/health` | Installed version, protocol, capabilities and policy |
| `GET /api/readiness` | Per-backend submission availability and evidence; no provider request |
| `GET /api/executors` | Configured resource/backend choices |
| `GET /api/backends/EXECUTOR_ID/capacity` | Cached capacity with its original observation time |
| `POST /api/backends/EXECUTOR_ID/capacity/refresh` | Read provider evidence into the cache |
| `GET /api/projects` | Registered projects |
| `POST /api/source-imports/archive` | Raw source archive |
| `GET /api/projects/P/source-imports/ARCHIVE_SHA256` | Reconcile a lost publication response |
| `GET /api/projects/P/sources/SOURCE_ID` | Immutable source identity |
| `POST /api/projects/P/runtimes` / `GET` | Freeze source and Dockerfile; list Runtimes |
| `GET /api/projects/P/runtimes/RUNTIME_ID` | Runtime state/receipt |
| `POST …/runtimes/RUNTIME_ID/execute` | Build using returned confirmation |
| `GET …/runtimes/RUNTIME_ID/logs` / `/progress` | Build diagnostics |
| `POST …/runtimes/RUNTIME_ID/reconcile` / `/materialize` | Observe publication without rebuilding; produce a Slurm SIF |

New builds use Dockerfile only, with pinned external bases. Old READY images
retain their workers; an obsolete unbuilt recipe requires new preparation.
Source admission rejects links, unsafe paths and credential-looking filenames;
it is not proof that file contents contain no secrets. See [builds](builds.md).

## Data and server preparation

| Method/path | Resource or purpose |
|---|---|
| `GET /api/projects/P/datasets/E` | Retained dataset identities and readiness |
| `POST /api/projects/P/datasets/E` | Begin one target-backed upload |
| `GET /api/projects/P/datasets/E/A` | Read one asset's status |
| `POST /api/projects/P/datasets/E/import-oss` | Import an approved plain tar from OSS |
| `POST /api/projects/P/experiment-preparations` | Combined Runtime/data/Run/Submission preparation |
| `GET /api/projects/P/experiment-preparations/PREPARATION_ID` | Dependencies, phase and next action |
| `POST …/experiment-preparations/PREPARATION_ID/continue` | Explicitly advance inspected dependencies |

Use the exact [multipart paths and resume rules](data.md#native-http-endpoints),
including zero-based part numbers. Completion establishes destination durability,
not GPU training. SenseCore data-image publication and NAS delivery are distinct.
Slurm/SenseCore target-backed assets do not provide client archive downloads;
approved OSS datasets have a separate signed-download route. Do not use output
routes to retrieve datasets.

## Freeze, review and submit

| Method/path | Resource or purpose |
|---|---|
| `POST /api/projects/P/runs` | Freeze a Run referencing a READY Runtime |
| `POST /api/runs/P/R/submissions` | Preflight and GPU-hour budget review |
| `GET /api/runs/P/R/preparation` | Find the original preparation after a lost reply |
| `GET /api/runs/P/R/submission` | Find the original Submission after a lost reply |
| `POST /api/submissions/S/authorize` | Authorize current prepared gates |
| `POST /api/submissions/S/execute` | Submit using returned confirmation |
| `GET /api/submissions/S` | Durable state and next action |
| `POST /api/submissions/S/reconcile` | Resolve an uncertain original submission |
| `POST /api/submissions/S/refresh` / `/history` | Sample evidence; ingest the next metrics page |
| `POST /api/submissions/S/cancel` / `/retry` / `/renew` | Explicit cancel, new Attempt or replacement gate |

Prepare/build/execute may return HTTP 202: poll the returned identity. An expired
review requires reprepare, not bypassing gates. Unknown effects require
reconciliation, not resubmission. Retry creates a new Attempt of the same Run;
changing source, parameters or restore reference requires a new Run.

## Read evidence and retrieve results

| Method/path | Resource or purpose |
|---|---|
| `GET /api/runs/P/R` / `/attempts` | Run evidence and exact executions |
| `GET /api/attempts/P/R::A/logs` | Stored bounded stdout/stderr |
| `GET /api/attempts/P/R::A/metrics?format=records&name=loss` | Compact normalized [metrics](metrics.md); repeat `name` |
| `GET /api/runs/P/R/attempts/A/checkpoints` / `/checkpoints/CHECKPOINT_ID` | Registered checkpoint metadata |
| `POST /api/submissions/S/checkpoint` | Verify and register a complete backend checkpoint |
| `GET /api/runs/P/R/attempts/A/collection` | Training, collection and verification states |
| `POST /api/submissions/S/collect` / `/sync` | Explicit collection; a cloud sync may use a zero-GPU job |
| `GET /api/submissions/S/artifacts` / `/artifacts/SHA256` | Result bundle metadata and download |
| `POST /api/submissions/S/download-ticket` | Short-lived signed part URL |
| `POST /api/submissions/S/acknowledge` | Confirm verified local files and release |
| `GET /api/runs/P/R/attempts/A/files` / `/artifacts/download` | Historical archive inventory and ticket |
| `GET /api/runs/P/R/tracking` | Frozen [W&B](wandb.md) destination and publication status |

The signed object GET carries no API token. Validate archive and file hashes,
then follow the ticket's acknowledgement contract. Signed URLs are temporary;
do not treat them as artifact identities. Download ACK can release the server
archive; it does not delete checkpoints or the metric journal.

A scheduler terminal state does not prove training/result success. Empty delayed
logs, missing files or a 404 are not proof that backend storage is empty.
Provider diagnostics are bound to the original job; refreshing submits no training.
For incomplete archives, repair and other recovery options see [recovery](recovery.md).
Internal launch/data/checkpoint/record transfer routes use narrow worker
capabilities, not the administrative client token. Scientific evaluation uses an
ordinary Run with checkpoint/data/protocol inputs, not a project-specific metric
or research-question prerequisite.
