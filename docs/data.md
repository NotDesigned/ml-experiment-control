# Data inputs and transfers

Source contains code/Dockerfile. Dataset assets are separate. Complete recovery
state belongs in persistent backend storage; selected weights and reports are
outputs. Do not include datasets or checkpoint archives in source uploads.

## Upload once, select backend storage

```bash
ml-exp asset-upload --project my-study --executor wyd-l40s \
  --directory ./data --state data.state.json
ml-exp asset-upload --project my-study --executor wyd-l40s \
  --directory ./data --state data.state.json --resume
ml-exp assets --project my-study
```

Create the project first by source import/Runtime preparation. A new upload
requires `--executor`; use an ID from `/api/executors`. Resume uses the saved
session and its frozen storage target, not a newly resolved backend profile.
Data must be unchanged: the client recreates a deterministic tar.gz and checks
its hash. Packing uses a client-side temporary file, so the client needs that
disk space. Bind the returned asset with
`{"inputs":[{"asset_id":"asset.ACTUAL_HASH","mount_path":"/inputs/data"}]}`.
`ml-exp experiment` accepts a local `directory`, selects the executor first,
and performs the upload and required delivery automatically.

The API has one global **1 GiB disk transfer buffer**, shared by input and worker
multipart transfers. It accounts conservatively for part bytes, sidecars and
filesystem blocks. A part is sealed by SHA256 and fsync, forwarded to its frozen
destination, then removed only after a durable matching ACK. Lost ACKs retain
resumable bytes; explicit POST create/complete can confirm the original part
again. A full buffer returns HTTP 503 `TRANSFER_BUFFER_BUSY` with `Retry-After`.
It never stages the complete new dataset archive on the API host or desktop.

## Backend delivery

| Backend | New uploaded data path |
|---|---|
| WYD | Client → API part buffer → SSH receiver on shared `/datapool`; archive and files are verified there before use |
| SenseCore | Client → API part buffer → CCR data-image layer; zero-GPU debug ACP copies and verifies it on NAS before GPU submission |

WYD uses the executor's `storage_root` beneath shared datapool. Compute nodes
need no external data download. The immutable asset records whole-archive and
regular-file hashes plus its storage scope. Reuse requires the same bytes and
compatible backend storage; a missing or corrupt cache fails before training.

SenseCore's data image is a delivery vehicle, separate from the training Runtime.
The API streams it through registry HTTP requests using an operator-provisioned
base; Docker and the desktop builder are not involved. Asset publication records
`verification=PENDING_BACKEND`: a published CCR image alone does not prove NAS
readiness. The CPU copy uses the configured debug cluster, 2CPU/4GiB/0GPU and
RESERVED quota. GPU jobs exclude debug clusters. Training verifies the sealed
NAS cache again; dataset files are excluded from final outputs.

New target-backed assets expose metadata and backend references. Their API
`/archive` and `/download` return typed HTTP 404 `DIRECT_DOWNLOAD_UNAVAILABLE`;
there is no client dataset-download command. Keep the original data or use its
backend storage reference. Historical object-store and desktop assets retain
read/reuse behavior at their original locations; new uploads never select those
old storage paths.

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

New SenseCore Runs prepare scripts before GPU submission on the configured
debug CPU pool (2CPU/4GiB/0GPU, RESERVED quota). The wrapper rereads the sealed NAS
receipt and verifies file SHA256, source/script/image and frozen CPU-profile
identities. It reports readiness through a separate short-lived HTTPS capability,
without depending on provider stdout indexing. A callback arriving before the
create response is saved as pending; READY requires the original confirmed job
UID. The wrapper starts Python in isolated mode without site initialization,
then removes the capability before executing the user's download script. It
does not grant access to other Runs or the administrative API.
Data readiness is separate from the CPU job's final state: a verified receipt
can establish READY even if its acknowledgement later fails and the original
CPU job exits unsuccessfully. Its observed state, exit code and prior diagnostic
remain visible; the platform does not relabel that job as successful.

Historical Runs retain their frozen inside-job or stdout-receipt behavior.
Reconciliation observes the original job; it never silently creates a replacement.
WYD prepares scripts on its Internet-connected gateway before GPU submission.
External source connectivity still needs validation on the chosen preparation
node; a verified backend cache avoids repeating that download during training.
The data-preparation receipt has 20,000-file/8-MiB bounds and requires a Runtime
with `data-preparation.v1`. Backend disk/quota still applies.

## Multipart HTTP contract

For client data, the root is `/api/projects/P/asset-uploads`:

1. POST `{"sha256":"whole archive hash","bytes":N,"target":{"executor":"ID"},
   "files":[{"path":"train.bin","bytes":M,"sha256":"file hash"}],
   "transport_sha256":"payload layer hash"}` returns session ID, negotiated
   `part_bytes`, `part_count`, confirmed `parts`, status and expiry. The standard
   client computes the inventory and transport hash required by SenseCore;
   they are not arbitrary labels. See `client/archives.py::data_transport_identity` when integrating
   another client.
2. GET `/{upload_id}` before resuming or after losing an acknowledgement.
3. PUT `/{upload_id}/parts/{number}?sha256=PART_HASH` sends exact raw part bytes.
4. POST `/{upload_id}/complete` validates/publishes the complete asset.
5. DELETE `/{upload_id}` aborts unfinished staging, not published assets.

Part numbers start at **0**. Identical duplicate parts are idempotent; different
bytes cannot replace a confirmed part. Only the final part may be shorter.
Missing parts/conflicts yield 409; wrong bytes/hash 400; size/part limits 413;
expiry 410; disk admission failures 507; transfer-buffer saturation or backend
transport failures return typed 503 errors. Keep response bodies and IDs for
diagnosis, never tokens.

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

Default part policy is 16 MiB with at most 65,536 parts, a 24-hour unfinished
session TTL and 20,000 asset files. SenseCore registry upload defaults to 1 MiB
when `upload_part_bytes` is absent; configure it explicitly for consistent policy.
The negotiated session is authoritative. The part count implies a 1 TiB archive
at 16 MiB or 64 GiB at 1 MiB. Expanded bytes/single files have independent optional
quotas and backend disk constraints. Source retains its smaller limits.

The **1 GiB limit covers the transfer buffer only**. It does not bound total
server disk: source snapshots, journals, result packaging/recovery and the local
Garage object store still occupy separate space. Outputs continue to use
server/object-store storage. A verified client download with release enabled
permits archive release on the next maintenance cycle without a 24-hour delay;
a GET alone is not an ACK. KEEP/unacknowledged outputs remain, and Garage physical
reclamation is asynchronous. Metrics and persistent checkpoints survive release.

Confirmed parts are removed after their durable destination ACK; unconfirmed
sealed bytes are retained for reconciliation. Session journals preserve identity.
Maintenance prunes expired unfinished local multipart staging; it does not
pretend an uncertain remote write was acknowledged. Backend datasets, NAS
checkpoint generations and registry images have separate retention
and no automatic scientific-data deletion. Direct signed output downloads bypass
API staging, but a same-host Garage still consumes this server's disk.

## WYD without compute-node Internet

New uploads already reside on shared datapool before Run preparation. Dataset
verification uses the connected SSH gateway and backend disk, not a GPU worker's
Internet access. The preparation helper has its own bounded validation timeout,
separate from GPU walltime. Existing frozen low-level Runs retain their original
HTTP worker path; an API upgrade does not rewrite their inputs.

For relay-configured WYD executors, download scripts run on the connected gateway
in the same Runtime SIF before GPU allocation. Training verifies and reuses the
shared `data-preparations` cache; it never reruns a missing preparation script on
an offline node. API checkpoint registration, metrics and result upload can use
the fixed TLS-preserving gateway relay. See
[WYD offline compute nodes](backends.md#wyd-compute-nodes-without-internet-access).
