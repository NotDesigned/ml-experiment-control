# Rust implementation and acceptance status

Development branch: `rewrite/rust`. Python reference baseline:
`e725faec689fb8395e692e27d00a8896fa7fe5a5` (1.5.1). Production still uses Python
1.5.1; the Rust workspace is `0.1.0-dev.1`. The planned native release is 0.1.0.
There is no production replacement or provider acceptance claim in this document.

The implementation pass connects the main research workflows before the next
round of debugging and provider tests. “Connected” means the CLI/API/service/
worker code exists. It does not mean an actual provider run has passed or that
all historical Python state can be migrated.

## Architecture

```mermaid
flowchart TD
    CLI[Native client] --> HTTP[Authenticated server API]
    HTTP --> CORE[Pure validation, identities and state transitions]
    HTTP --> LEDGER[Durable intent, receipt and observation stores]
    HTTP --> SLURM[SSH Slurm gateway]
    HTTP --> PAI[PAI REST: DLC / DSW]
    HTTP --> SC[SenseCore REST: ACP / CCI]
    HTTP --> BUILD[Optional Python image effects]
    BUILD --> ACR[Tokyo ACR; same-digest CCR mirror]
    SLURM --> WORKER[Native worker]
    PAI --> WORKER
    SC --> WORKER
    WORKER --> STATE[Backend checkpoint / input cache]
    WORKER --> OBJECT[Private result parts]
    WORKER --> HTTP
    HTTP --> JOURNAL[Canonical metrics and events]
    JOURNAL --> WB[Optional official W&B SDK process]
    OBJECT --> CLI
```

| Package | Owns |
| --- | --- |
| `ml-exp-core` | Pure identity, validation, budgets, transitions, reconciliation and plans |
| `ml-exp-io` | Confined files, hashes, bounded processes, streaming and sockets |
| `ml-exp-client` | Private URL/token settings, commands, saved identities and verified transfers |
| `ml-exp-server` | HTTP, durable storage, provider effects and background observation |
| `ml-exp-worker` | Process supervision, Slurm helpers, inputs, checkpoints, metrics and outputs |
| `experiment-redact` | Shared credential redaction |

The core reads no environment, clock, files or network. Effects return explicit
receipts to the core. Image build/copy wrappers may remain Python, as permitted
by the complexity-first migration policy. W&B retains the official SDK process.
Research code may use Python/PyTorch or another image language.

## Connected workflows

| Workflow | Main implementation | Verification boundary |
| --- | --- | --- |
| Configure client, inspect status/logs/metrics, watch exact Attempt | `client/{config,session,watch,main}.rs` | Local HTTP and binary tests |
| Register approved Project; upload immutable source | `server/{projects,sources}`, `client/{projects,sources}.rs` | Private temporary stores and HTTP |
| Dockerfile prepare/build/reconcile/log/progress | `core/cloud/build.rs`, `server/cloud_runs/service/builds.rs`, `builds/native_image.py` | Frozen recipe and late-publication bridge tests; actual build pending |
| OCI-to-SIF on Slurm gateway | `worker/materialize.rs` | Fake apptainer; exact reuse/capacity/tampering checks |
| Upload/download cloud datasets | `server/objects.rs`, `service/datasets.rs`, `client/datasets.rs` | Bounded parts and immutable receipts; actual OSS transfer pending |
| Download into CPFS / NAS input cache | `worker/{data_import,cloud_data}.rs` | Import/hash/cache tests; provider storage pending |
| CCR data image and CPU ACP copy before training | `service/data_delivery.rs`, `builds/native_data.py`, `worker/data_copy_cli.rs` | Stage control/hash/capability tests; actual copy pending |
| One experiment configuration | `core/preparation.rs`, `client/preparations.rs`, `service/preparations.rs` | Recovery slot, conflict, early preflight and HTTP tests |
| Matrix preparation and bounded dispatch | `core/preparation.rs`, `client/batch.rs` | Stable identities, exact terminal slot accounting; actual batch pending |
| Slurm / DLC / ACP submit, reconcile, refresh and cancel | `server/submissions`, `server/backends`, `server/cloud_runs` | Local scheduler/HTTP fixtures; actual provider run pending |
| Logs/history/checkpoints over gateway or scoped HTTPS | `io/observations.rs`, `worker/cloud_exchange.rs`, `service/exchange.rs` | Lost ACK, sequence, variant and frozen identity tests |
| Explicit new Attempt from selected persistent checkpoint | `core/cloud/retry.rs`, `service/retry.rs` | Immutable restore and cumulative budget checks; actual restore pending |
| CPU result collection, direct download, verified ACK/release | `worker/cloud_artifacts.rs`, `service/artifacts.rs`, `client/artifacts.rs` | Archive/member integrity and durable cursor tests; provider collection pending |
| DSW / CCI debug lifecycle | `core/cloud/debug.rs`, `service/debug.rs`, `client/debug.rs` | Native compile/identity checks; actual machine lifecycle pending |
| Backend catalogue/cached capacity | `core/capacity.rs`, `service/capacity.rs`, `capacity_http.rs` | Local GET is effect-free; actual provider capacity query pending |
| Optional W&B publication | `core/tracking`, `server/tracking` | Fake SDK and durable history/display ACK tests; official-cloud acceptance pending |
| Scoped administrative maintenance and record retirement | `core/maintenance.rs`, `server/administration.rs`, `worker/maintenance.rs` | Confined plans and worker receipts; actual provider cleanup pending |
| Standalone verified client update | `core/distribution.rs`, `client/update.rs` | Manifest/version/hash logic; native release assets not yet published |

Paths in the table are relative to `rust/crates`, except the two image bridge
paths under `server/src/ml_exp_server`. See [the native usage guide](../rust/README.md)
for commands and configuration shapes.

SenseCore can select the compatible cluster with the highest SPOT allowance
from an operator allowlist when creating a new Run. The selected profile is then
frozen. Reconcile, cancel and explicit retry never select a different pool.
Training pools exclude debug; data-copy profiles explicitly use debug CPU pools.
Quota observations do not establish the number of idle GPUs.

## Storage and transfer contracts

- Dataset/result parts are 16 MiB, numbered from zero. Part count, metadata and
  storage budgets remain bounded; the old 4 GiB payload ceiling is removed.
- The API keeps source/control metadata and receipts. Cloud payloads stream to
  approved OSS parts; Slurm payloads go to the shared gateway. The API does not
  retain full dataset or weight archives.
- Recovery states stay on backend persistent storage. A complete ready marker,
  generation contents and hashes bind each registered checkpoint.
- Training completion, result publication, verification and release are distinct.
  Result ACK follows full client verification and local fsync. A lost ACK is
  reconciled with the same saved identity and file.
- Administrative deletion is a separate explicit plan. It cannot impersonate a
  verified download or cause a removed result to be collected again.
- PAI dataset loading uses signed object URLs and a verified personal CPFS cache.
  This does not enable the provider's previously blocked native OSS filesystem mount.

Metrics preserve arbitrary names and explicit units; protocol, checkpoint and
variant are identity fields. Required metrics come from each Run's schema. Step
series and final summaries have distinct W&B projections. Scientific aggregation
and model selection remain research-code responsibilities.

## Deliberate limits and remaining acceptance

The native registry uses a separate marker. Normal startup and GET never
initialize or repair it. Do not point the preview binary at production state.
Before cutover, historical Project/Run/Attempt identities, source/runtime hashes,
checkpoint reads, result tombstones and W&B cursors need migration acceptance.
The old working environment remains the sole production environment meanwhile.

Campaign authoring is intentionally absent; a Run does not need a research
question or Campaign. Batch commands perform explicit passes, not an unattended
queue. An expired authorization gate blocks dispatch until an explicit `run renew`
creates a fresh PREPARED plan. Renewal is idempotent, preserves the Run/Attempt
and budget, and refuses any possible prior execution. No automatic checkpoint selection or unbounded restart exists.
Native debug supports CCI create/stop and DSW create/stop/delete. Slurm dataset
export is outside the OSS-specific dataset download command.

An offline `ml-expd-state-audit` now fingerprints existing manifests, checkpoint
and result receipts, source/runtime/preparation control receipts, tombstones,
lifecycle records, Action plans and SQLite rows.
It compares metadata without initializing a native marker, making a database
copy or reading scientific payloads. It preserves metric receive order and
includes event IDs and W&B cursors; equality is not production cutover authority.
Historical-state transformation and provider acceptance remain separate work.

Registered checkpoint GETs can also read historical Run/Attempt receipts without
a native Submission. They validate the original source/image and issued storage
context, preserve withdrawn/deleted scopes, and create no locks or seals. A
native-created Run never falls back to this reader if its native authority is
missing or corrupt. Reading a historical checkpoint does not authorize replaying
its old Action or automatically restarting a job.

Next testing sequence: deterministic regressions → isolated client/server on CPU
DSW → actual storage/build/debug lifecycle → bounded training/restore/collection
on each enabled backend → historical-state migration and guarded production
cutover. Test failures keep their original evidence; missing or ambiguous effects
are reconciled under the original identity, never replayed with a new token.

## Inventory and evidence

[The inventory](../rust/compatibility/inventory.json) retains every baseline
Python module and its original hash: 220 platform modules / 41,359 lines, plus
238 test/tool/example files. Baseline module parity and connected native
workflows are tracked separately. A zero count of fully accepted Python module
replacements is not an estimate of how much Rust code is written.

Local/DSW logs live in ignored `.ops` directories and are not release artifacts.
[The usage guide](../rust/README.md#build-and-verification) lists reproducible
format, lint, workspace-test, binary-smoke and bridge-test commands. Passing these
checks establishes development evidence, not real GPU or production acceptance.
