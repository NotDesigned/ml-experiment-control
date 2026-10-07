# Architecture

## Packages and execution locations

| Package | Source | Responsibility |
|---|---|---|
| `ml-experiment-client` | `client/src/ml_exp_client/` | Dependency-free HTTP client and `ml-exp` CLI |
| `ml-experiment-server` | `server/src/ml_exp_server/` | API, definitions, Actions, collection, builder and worker sources |
| `ml-experiment-control` | `src/experiment_control/` | Backend interface and durable controller primitives |

The server imports the core; the core never imports the server. Clients install
neither. `rust/src/main.rs` supplies `experiment-redact`, packaged with the core.

```mermaid
flowchart LR
    Client[Client computer] --> API[API daemon]
    API --> Controller[Controller and backend library]
    Controller --> WYD[WYD Slurm and Apptainer]
    Controller --> SC[SenseCore REST and ACP]
    API --> Builder[Private image builder]
    Builder --> Desktop[Desktop Docker and BuildKit]
    Desktop --> Registry[Container registry]
    Registry --> WYD
    Registry --> SC
    WYD --> Store[Artifact storage]
    SC --> Store
    Store --> Client
```

The daemon freezes metadata and controls schedulers; it does not run training
code. A private builder owns Docker and publication. In SSH desktop mode, actual
build contexts/layers live on the desktop. Training runs inside backend containers.

## Server code map

| Concern | Main modules under `server/src/ml_exp_server/` |
|---|---|
| HTTP and auth | `api/`, `http_auth.py`, `api_contract.py` |
| Composition | `runtime.py`, `api/app.py` |
| Immutable definitions | `source_imports.py`, `source_revisions.py`, `container_execution.py`, `authored_runs.py` |
| Checked mutations | `application.py`, `submissions.py`, `actions/`, `controller_gateway.py` |
| Capability matching/preparation | `executor_capabilities.py`, `experiment_preparation.py` |
| Dispatch | `container_controller.py` → core `backends/` |
| Observation | `collectord.py`, `ingest/`, `terminal_snapshot.py`, `execution_progress.py`, `input_progress.py` |
| Images | `image_builder.py`, `dockerfile_build.py`, `image_build_context.py` |
| Data | `data_assets.py`, `multipart_upload.py`, `remote_data_uploads.py`, `desktop_upload.py`, `data_delivery.py`, `data_image_build.py` |
| Results/state | `artifact_store.py`, `artifacts.py`, `checkpoint_registry.py`, `metric_contract.py` |

`runtime.py` initializes daemon services; the public **Runtime resource** denotes
a code/environment image. `controller_gateway.py` is the project-controller
subprocess boundary. Managed projects use the generic container controller;
library consumers can supply adapters. A single managed Run currently generates
an internal Campaign file, without requiring clients to design a Campaign.

## Worker code executes on compute

These files are injected into new images:

- `worker_launcher.py`: verify a digest-bound startup manifest.
- `managed_worker.py`: inputs, restore, user child, registration and output return.
- `data_input.py`: verified data cache; `data_preparation.py`: optional download script.
- `persistent_state.py`: verify/register/restore complete generations.
- `worker_artifacts.py`: declared-output archive and multipart transfer.
- `data_copy_worker.py`: separate zero-GPU NAS dataset copy.

Worker fingerprints participate in image identity. Updating server source does
not change workers in existing images. Backend implementations are core
`backends/wyd.py`, `sensecore.py`, `sensecore_rest.py` and development `local.py`.

## Identities and state

| Entity | Meaning | Durable evidence |
|---|---|---|
| Source | Immutable code snapshot | Frozen tree and metadata |
| Runtime | Source/Dockerfile/worker and published image | Definition and build receipt |
| Asset | Immutable input bytes | Archive/file hashes and locator |
| Preparation | One server-owned request and saved dependencies | Hash-bound intent and durable progress journal |
| Run | Image, data, argv, resources and metric protocol | Immutable manifest |
| Attempt | One execution | Intent, exact scheduler ID and observations |
| Checkpoint | Complete backend-resident generation | NAS/datapool bytes and registered manifest |
| Artifact | Published exact-Attempt output archive | Object, hashes and receipt |

Actions/events use transactional SQLite. The read index uses another SQLite
database. Runtime/delivery/checkpoint/upload metadata use files and durable
journals/locks as needed. There is no global transaction across all stores.
Reconciliation verifies exact effects; it does not replay an uncertain create.
The collector observes/collects and never starts replacement jobs.

Programs live in `/root/ml-expd`; config in `/etc/ml-expd`; state in
`/var/lib/ml-expd` and configured project/builder roots. Backend storage and
registry assets have separate lifetimes. `.ops/` is private operational evidence,
not public functionality. See [deployment](operator-guide.md) and [recovery](recovery.md).
