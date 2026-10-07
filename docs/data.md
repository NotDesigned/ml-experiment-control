# Data inputs and transfers

Source contains code/Dockerfile. Dataset assets are separate. Complete recovery
state belongs in persistent backend storage; selected weights and reports are
outputs. Do not include datasets or checkpoint archives in source uploads.

## Upload once, reference by identity

```bash
ml-exp asset-upload --project my-study --directory ./data --state data.state.json
ml-exp asset-upload --project my-study --directory ./data --state data.state.json --resume
ml-exp assets --project my-study
```

Create the project first by source import/Runtime preparation. Resume requires
unchanged data: the client recreates a deterministic tar.gz and checks its hash.
It streams to a local temporary file, so the client needs packing disk space.
Bind the returned asset to a Run with
`{"inputs":[{"asset_id":"asset.ACTUAL_HASH","mount_path":"/inputs/data"}]}`.
`ml-exp experiment` also accepts a local `directory` and uploads automatically.

An uploaded asset is READY only after the whole archive, extracted regular files
and publication have passed validation. On `desktop-data-upload.v1` deployments,
client data parts pass through bounded API memory into the desktop's private
staging helper. The API host retains metadata, not a full dataset copy. The
desktop helper uses separate code/data volumes and no public port. Other
deployments can use the server/object-store transfer path.

## Backend delivery

| Backend | Uploaded data path |
|---|---|
| WYD | Worker fetches/reuses a verified cache on shared `/datapool`, then exposes the configured `/inputs/...` path |
| SenseCore desktop assets | Build/push a separate data-only CCR image; zero-GPU ACP copies/verifies it into NAS before creating the GPU Run |
| Historical object-store assets | Worker fetches/verifies its cache through an exact-Attempt input capability |

SenseCore's data image is a delivery vehicle, not the training Runtime. CPU copy
uses the configured debug cluster, 2CPU/4GiB/0GPU spec and RESERVED quota in the
current deployment. GPU jobs exclude debug clusters. The common worker verifies
the NAS cache before training; dataset files are not copied into final outputs.
`ml-exp experiment` handles desktop delivery automatically, or use
`asset-upload … --executor SENSECORE_EXECUTOR`.

Delivery routes:

```text
POST /api/projects/P/assets/ASSET_ID/deliveries/prepare  {"executor":"ID"}
POST /api/projects/P/data-deliveries/DELIVERY_ID/execute {"confirmation":"returned value"}
GET  /api/projects/P/data-deliveries/DELIVERY_ID
POST /api/projects/P/data-deliveries/DELIVERY_ID/reconcile
```

Asset READY and backend delivery READY are separate. Reconciliation checks the
exact existing publication/job and never resubmits an uncertain copy.
With `nas-data-ready-reuse.v1`, preparation reuses a sealed READY receipt for
identical project/asset/archive/file hashes and NAS workspace/mount/path. Worker
upgrades and copy-placement settings do not invalidate verified dataset bytes.
The training worker still rehashes actual cached files before starting code;
receipt reuse is not a new independent server-side NAS verification.

GET of an older pending/uncertain request can report effective `status: READY`,
with `operation_status` retaining its original state, `reused_from` naming the
canonical delivery and `ready_delivery` carrying the verified receipt. Its
original file, journal, image and scheduler identity remain unchanged. This
means data is ready through another delivery, not that the uncertain copy
succeeded. New Runs bind the canonical receipt. Corrupt/unsealed receipts and
different bytes/NAS scopes are never reused. Failure `diagnostic` exposes the
phase, safe error code/class, bounded details and next action without raw errors.
Permission-based 0444/0555 protection is not a kernel read-only mount; root code
can change permissions. Reuse verifies bytes again and rejects corruption.

## Download data on the backend instead

Include a script in frozen source and configure:

```json
{
  "data_preparation": {
    "script": "download_data.py",
    "interpreter": "python3",
    "arguments": ["--dataset", "version-1"],
    "timeout_seconds": 1200
  }
}
```

The script writes only its dataset beneath supplied `DATA_DIR`. Success is
required before training starts. Platform identity includes script/source hash,
arguments, environment, image and inputs. It locks the persistent cache,
verifies regular-file SHA256 and publishes atomic readiness. Reuse rehashes;
partial/corrupt cache fails without silent redownload. Preparation metadata is
exported as `data-preparation.json`, even with custom output patterns.

This script runs inside the allocated training job and its wall-time/GPU budget;
it is not a separate CPU downloader. SenseCore compute connectivity may differ
from the debug cluster, so direct Internet download must be validated. Desktop
CCR delivery avoids making GPU training depend on a large external download.
The data-preparation receipt has 20,000-file/8-MiB bounds and requires a Runtime
with `data-preparation.v1`. Backend disk/quota still applies.

## Multipart HTTP contract

For client data, the root is `/api/projects/P/asset-uploads`:

1. POST `{"sha256":"whole archive hash","bytes":N}` returns session ID,
   negotiated `part_bytes`, `part_count`, confirmed `parts`, status and expiry.
2. GET `/{upload_id}` before resuming or after losing an acknowledgement.
3. PUT `/{upload_id}/parts/{number}?sha256=PART_HASH` sends exact raw part bytes.
4. POST `/{upload_id}/complete` validates/publishes the complete asset.
5. DELETE `/{upload_id}` aborts unfinished staging, not published assets.

Part numbers start at **0**. Identical duplicate parts are idempotent; different
bytes cannot replace a confirmed part. Only the final part may be shorter.
Missing parts/conflicts yield 409; wrong bytes/hash 400; size/part limits 413;
expiry 410; staging capacity failures 507. Desktop RPC transport failures can
return typed 503 errors. Keep response bodies and IDs for diagnosis, never tokens.

Workers use `/api/attempt-uploads/P/R/A/{artifacts|checkpoint}` with the same
create/read/parts/complete/abort sequence and narrow Attempt credentials.
New images use multipart; old images keep their original uploader. A COMPLETE
session/receipt is required for download. Persistent state registration is
different and uploads no state bytes.

## Limits and storage lifecycle

Read `/api/storage-limits`, including `configured_byte_limits`, part size/count,
file count, session TTL and actual storage capacity. No fixed 4-GiB default
remains in code. An operator can still set positive byte quotas; null means no
configured cap, not infinite storage. Some protocol-2 limit fields use a large
integer for compatibility; `configured_byte_limits` disambiguates them.

Defaults: 16-MiB parts, 65,536 parts, 24-hour unfinished-session TTL and 20,000
asset files. The part count implies at most 1 TiB at 16 MiB, or 64 GiB at the
1-MiB size used by the desktop deployment. Expanded bytes/single files have their
own optional quota and real disk constraints. Source retains its smaller limits.

Completed publication removes its own part files and retains the journal.
Published desktop archives, NAS datasets, registry images and outputs have no
automatic retention cleanup. Worker outputs still use server/object-store
storage; moving client data staging to the desktop did not move that path.
Direct signed output downloads bypass ML-Expd staging; a local object store can
still use the same physical server's disk/network. It is not an off-host backup.

## WYD preparation without compute-node internet

The one-config server workflow prepares desktop assets through SSH into shared
`/datapool` before GPU submission. API memory is bounded; archive staging and
unpacking use backend disk. The helper has a 1800-second total transfer/validation
budget, separate from the GPU walltime. A verified shared cache is frozen as a
required input. Corruption fails closed; uncertain transfers are never repeated
automatically. Existing low-level/historical Runs keep their original HTTP worker
path. See [backend preparation](backends.md).
