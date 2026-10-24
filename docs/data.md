# Data inputs and transfers

Keep four things separate: source/Dockerfile, input datasets, persistent recovery
state and exported results. Uploading source is not a dataset channel. Result
ACKs do not delete datasets or persistent checkpoints.

## Upload once, select backend storage

Create the project through source import or experiment preparation, then select
an executor from `/api/executors`:

```bash
ml-exp asset-upload --project study --executor slurm-gpu \
  --directory ./data --state data.state.json
ml-exp asset-upload --project study --executor slurm-gpu \
  --directory ./data --state data.state.json --resume
ml-exp assets --project study
```

The client builds a deterministic tar.gz in a temporary file; it needs local
space for that archive. Resume requires unchanged bytes and the same state file.
The saved session retains its original storage target. Bind its asset ID:

```json
{"inputs":[{"asset_id":"asset.ACTUAL_HASH","mount_path":"/inputs/data"}]}
```

The one-config `experiment` workflow can upload an `inputs[].directory` and
prepare delivery itself. Assets identify bytes; they do not imply that every
backend shares the same files or authorize access to another storage namespace.

## Backend delivery

| Backend | New asset path | Before training |
| --- | --- | --- |
| Slurm | Client → bounded API part buffer → SSH receiver → shared storage | Gateway verifies archive/files; compute reuses verified shared data |
| SenseCore | Client → bounded API part buffer → CCR data-image layer → debug CPU ACP → NAS | CPU receipt must establish NAS readiness; training rehashes cached files |
| PAI DLC | Client → bounded API part buffer → approved personal OSS archive → CPFS cache | First fill and verification run inside the allocated DLC job |

The API's global **1 GiB buffer** accounts for in-flight multipart bytes,
sidecars and filesystem blocks. It seals a part and removes it only after a
matching durable destination ACK. Lost ACKs retain resumable bytes. Full capacity
returns `503 TRANSFER_BUFFER_BUSY` with `Retry-After`. A complete new dataset is
not retained on the API host or desktop. This limit does not cap source snapshots,
result storage, recovery scratch, databases or total server disk.

Slurm's `storage_root` must be shared with its compute nodes. Only the SSH gateway
needs external access; see [offline nodes](#slurm-without-compute-node-internet).
SenseCore streams fixed data layers through registry REST, without Docker or the
desktop builder. It requires an operator-provisioned Python/glibc/Bash base in
the same CCR repository. The data Runtime is separate from the training Runtime.
The copy uses a verified RESERVED 2CPU/4GiB/0GPU debug profile. GPU placement
excludes debug clusters. A published CCR image (`verification=PENDING_BACKEND`)
is not proof that the dataset is available on NAS.

Slurm/SenseCore target-backed datasets expose metadata and backend references;
their direct API archive/download returns `404 DIRECT_DOWNLOAD_UNAVAILABLE`.
OSS datasets support direct download below. Historical object-store/desktop
assets keep their original read/reuse paths; new uploads do not select those
legacy paths. If a configured finite local-payload lifetime expires historical
local data, it returns `410 ASSET_EXPIRED` while retaining its identity. Backend
shared storage, NAS, CCR and durable OSS originals have separate lifetimes.

### Delivery identity and reconciliation

```text
POST /api/projects/P/assets/ASSET_ID/deliveries/prepare  {"executor":"ID"}
POST /api/projects/P/data-deliveries/DELIVERY_ID/execute {"confirmation":"returned value"}
GET  /api/projects/P/data-deliveries/DELIVERY_ID
POST /api/projects/P/data-deliveries/DELIVERY_ID/reconcile
```

Asset publication and backend delivery READY are different. Reconciliation
observes the original publication/job and never repeats an uncertain create.
`nas-data-ready-reuse.v1` allows a sealed receipt for identical project, bytes,
file inventory and NAS workspace/mount/path to establish readiness. Workers
rehash actual files; receipt reuse is not an independent server read of NAS.

An older uncertain delivery can expose effective `status=READY`, `reused_from`
and `ready_delivery` while preserving its own `operation_status` and history.
This means another receipt proves data readiness, not that the original copy
succeeded. New Runs bind the canonical receipt. Different scopes or corrupt,
unsealed receipts cannot be reused. Permission sealing (0444/0555) is not a
kernel read-only mount or protection against privileged code.

## Download data on the backend instead

Include the downloader in frozen source:

```json
{"data_preparation":{"script":"download_data.py","interpreter":"python3","arguments":["--dataset","version-1"],"timeout_seconds":1200}}
```

It writes datasets beneath `DATA_DIR`, separate from `OUTPUT_DIR`. Successful
exit and verified regular-file SHA256 inventory are required before training.
The preparation identity binds script/source, image, arguments, environment and
inputs. An exclusive cache lock and atomic sealed receipt permit rehash-and-reuse;
partial/corrupt data fails without silently running the downloader again.
`data-preparation.json` is exported even with custom output patterns.

| Backend | Download script placement |
| --- | --- |
| Relay-configured Slurm | Connected SSH gateway, in the same SIF, before GPU submission |
| SenseCore with configured CPU preparation | RESERVED debug CPU ACP before GPU submission |
| PAI DLC | Inside the allocated job and CPFS workspace; consumes runtime/GPU budget |

Check the executor's declared `data.download_script` and preparation phase.
Internet reachability is not guaranteed by the existence of a downloader.
SenseCore uses an exact-job capability callback: it revalidates the sealed NAS
receipt and requires the confirmed original CPU UID, independently of log
indexing. A callback received before the create response stays pending. The
capability is removed before user code. A valid receipt can prove readiness even
if a later acknowledgement fails; the original CPU failure is still retained.
Historical inside-job/stdout modes remain frozen. Receipt bounds are 20,000 files
and 8 MiB; configured timeout, backend quota and Runtime capabilities also apply.

## Multipart HTTP contract

Data sessions live at `/api/projects/P/asset-uploads`:

1. POST the archive `sha256`, `bytes`, and `target: {"executor":"ID"}`. Supply
   `files: [{"path":"train.bin","bytes":N,"sha256":"HASH"}]` and
   `transport_sha256` for SenseCore. The official client computes these identities.
2. Save `upload_id`, `part_bytes`, `part_count`, confirmed `parts` and expiry.
   GET the same session before resuming or after losing a response.
3. PUT `/{upload_id}/parts/{number}?sha256=PART_HASH` with exact raw bytes.
4. POST `/{upload_id}/complete` to validate and publish; DELETE the session to
   abort unfinished staging, never published data.

Part numbering starts at **0**. Identical duplicates are idempotent; different
bytes cannot replace a confirmed part. Only the final part may be shorter.
Typical errors are 400 for bad hashes/bytes, 409 for conflicts/missing parts,
410 for expiry, 413 for configured limits, 507 for storage admission and typed
503 for buffer saturation or unconfirmed backend transport. Keep sanitized error
bodies and IDs; do not create a replacement upload after an uncertain completion.

Workers use `/api/attempt-uploads/P/R/A/{artifacts|checkpoint}` with the same
multipart lifecycle and narrow Attempt credentials. Old images keep their frozen
uploader. A COMPLETE published receipt is required for archive download.
Persistent checkpoint registration sends metadata rather than these state bytes.

## Limits and storage lifecycle

`/api/storage-limits` reports configured byte limits, part policy, inventory
bounds, session TTL and capacity. There is no fixed 4 GiB default. Null means no
configured cap, not infinite destination space. Compatibility integer fields can
be large; consult `configured_byte_limits` for actual policy.

The general default is 16 MiB parts, at most 65,536 parts, 20,000 files and a
24-hour unfinished-session TTL. SenseCore defaults to 1 MiB parts when
`upload_part_bytes` is absent. Negotiate rather than hardcode: 65,536 parts give
1 TiB at 16 MiB or 64 GiB at 1 MiB. Expanded bytes, individual files and source
archives have their own limits. OSS has the lower 10,000-part bound below.

Verified result-download ACK with release enabled makes that archive eligible
for the next maintenance cycle; a GET alone does not. Local result expiry and
required durable archival may also affect release. KEEP, unconfirmed downloads,
backend datasets and checkpoint state have distinct policies. Physical object-store
garbage collection can lag logical release. Metrics and receipts remain.
See [result retention](recovery.md#download-confirmation-and-server-storage).

## Slurm without compute-node Internet

Only the configured gateway must reach the API/registry and external dataset
sources. Uploaded datasets are verified on shared storage before GPU submission.
Relay-configured download scripts run there in the Runtime SIF; training reuses
the sealed cache and refuses to redownload missing data on an offline node.

The configured TLS-preserving TCP gateway relay carries launch requests, metrics,
checkpoint registration and results for offline compute. It is not a general
Internet proxy. Existing frozen low-level Runs retain their old transport paths;
an API upgrade cannot change those manifests. See [Slurm configuration](slurm.md).

## Personal OSS dataset originals and DLC caches

When `oss-datasets.v1` is enabled for the executor:

```bash
ml-exp asset-upload --project study --executor pai-h200 --directory ./data --state data.state.json
ml-exp asset-import-oss --project study --executor pai-h200 \
  --object-key users/USER/datasets/version-1.tar.gz --sha256 ARCHIVE_SHA256 --bytes ARCHIVE_BYTES
ml-exp asset-download --project study --id asset.SHA256 --out ./downloaded-data
```

Import uses `POST /api/projects/P/assets/import-oss` with `executor`, `object_key`,
`sha256`, `bytes`. Bucket, region and user are server configuration. Reads are
limited to approved personal `users/USER/datasets/` and public `public/datasets/`
prefixes; uploads write beneath `users/USER/datasets/ml-expd/PROJECT/` with distinct
keys. No role, Bucket ACL/Policy, public access or lifecycle policy is changed.

Completion/import streams the object through the API once to verify full archive
and file SHA256, using bounded memory rather than local payload storage. It is
not zero-copy registration. Upload snapshots remain observable during validation;
repeat completion reconciles the same object. Changed external objects fail hash
checks. Unknown initiations/completions require inspection, never blind overwrite.

DLC mounts approved personal CPFS at `/mnt/workspace`. Its first cache fill uses
the authenticated exact-Attempt API endpoint to relay bounded OSS chunks into a
verified CPFS cache. This is inside the allocated DLC job, not a pre-job CPU copy.
Dataset files remain outside outputs. Registration returns an ordinary ML-Expd
asset ID; it does not create a native PAI Dataset or mount OSS as a filesystem.
**Native OSS filesystem mounts remain disabled.**

`GET /api/projects/P/assets/ASSET/download` returns a private five-minute OSS GET
ticket. The client downloads directly, checks archive/file hashes and saves
`data/` and `verification.json`. It sends no result ACK and never deletes the
OSS original. Durable datasets are outside local expiry and Run deletion.

OSS allows at most 10,000 parts. Default `oss_dataset_part_bytes` is 16 MiB:
up to 156.25 GiB per archive, subject to smaller configured limits and capacity.
Non-final parts must be at least 100 KiB. `/api/storage-limits` is authoritative.
Personal storage budgets are soft accounting unless separately enforced.
The operator configures `oss_datasets.bucket`, `region`, `user` and
`credentials_file`; credentials never reach clients or training code. Public
CPFS models/datasets remain read-only.
