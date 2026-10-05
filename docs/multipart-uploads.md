# Resumable archive uploads

Server 0.2.1 and standalone client 0.1.1 retain protocol 2 and add the health
capability `multipart-upload.v1`. The default data archive, expanded data and
final artifact limits are **4 GiB (4,294,967,296 bytes)**. A checkpoint must fit
both the artifact and asset limits. The archive still has one whole SHA256;
parts improve transport reliability, not the total size allowance.

## Client data

```bash
ml-exp asset-upload --project my-training --directory ./data --state data.json
# If the command exits before completion, keep the same directory unchanged:
ml-exp asset-upload --project my-training --directory ./data --state data.json --resume
```

The archive is reproducible tar.gz, streamed to a local temporary file rather
than memory. Resume rebuilds its hash and rejects a changed archive. Upgraded
clients use multipart when advertised; legacy servers keep whole-body upload.
Only archive requests retry transient network/429/5xx failures. Runs, Actions
and scheduler submissions are never replayed by this mechanism.

All client requests require the existing Bearer and
`X-ML-Expd-Client-Protocol: 2`. For an existing project:

1. `POST /api/projects/{project}/asset-uploads` with JSON
   `{"sha256":"<whole-archive-sha256>","bytes":123456}`. The response includes
   `upload_id`, `part_bytes`, `part_count`, `parts`, `status` and `expires_at`.
   Repeating create with the same target/hash/size returns the same session.
2. `GET /api/projects/{project}/asset-uploads/{upload_id}` reads durable state.
   `parts` maps zero-based part numbers to `{sha256,bytes}` receipts.
3. `PUT /api/projects/{project}/asset-uploads/{upload_id}/parts/{number}?sha256=<part-sha>`
   sends raw bytes with Content-Length. Each part has the exact negotiated size;
   only the last may be smaller. Identical duplicate parts are idempotent;
   different bytes cannot replace a sealed part.
4. `POST /api/projects/{project}/asset-uploads/{upload_id}/complete` publishes
   the asset after all part and archive hashes, safe extraction, expanded-byte
   bounds and object-store publication pass. The response is the normal READY
   asset receipt. Completion is idempotent, including response-loss recovery.
5. `DELETE /api/projects/{project}/asset-uploads/{upload_id}` cancels staging.
   A published asset cannot be deleted through this route. Create can restart an
   aborted/expired session with the same bytes.

Missing parts or identity conflicts return 409; bad part bytes/hash return 400;
oversized archives/parts return 413; expired staging returns 410; insufficient
disk space returns 507. Unpublished parts are never listed as input assets.

## Workers

Newly built images set `ML_EXPD_MULTIPART_UPLOAD=1`; the launcher consumes this
flag before running the application. The same mechanism is used for the final
declared outputs and every atomic checkpoint archive, on both WYD and SenseCore.
It uses `/api/attempt-uploads/{project}/{run}/{attempt}/{kind}`, where `kind` is
`artifacts` or `checkpoint`, with the same create/read/parts/complete/DELETE
suffixes. Every request checks the existing exact-Attempt capability; it cannot
access control APIs, another Attempt, or checkpoint publication without opt-in.
Worker create/read omit the potentially large publication receipt; complete
returns `{status:"COMPLETED",sha256:...}`. The normal artifact and checkpoint
read APIs retain the complete receipts.

Worker retries reissue create to recover committed parts, then continue only
missing ones. Existing frozen images remain unchanged and use their legacy
whole-body upload with the new configured cap. To use multipart workers, build
a new Runtime and freeze a new Run against it; a running image is not modified.
New worker images require a server advertising this capability. Job wall-clock
and GPU budgets still include transfer time; a hard kill can leave an incomplete
session, not a recoverable published checkpoint.

## Storage and operator policy

Private artifact-store configuration keys are `max_archive_bytes`,
`max_asset_archive_bytes`, `max_asset_bytes`, `upload_part_bytes` (default 16 MiB,
allowed 1 KiB..64 MiB), and `upload_session_seconds` (default 24 hours, allowed
5 minutes..7 days). Uploads have at most 4,096 parts. `/api/storage-limits`
exposes byte limits, part size and session lifetime without cloud credentials.

Parts and their journal live under the project registry's `multipart-uploads`,
survive daemon restarts, and never contain API/worker/cloud credentials. Archive
validation uses a seekable view of the parts, avoiding a second assembled tar.
Publication still needs disk for expanded validation/cache and the object
store; free-space checks can reject an upload below 4 GiB when the disk is full.
Global storage quota, if unset, does not imply unlimited disk.

Successful publication deletes its own parts and keeps a small session journal
with the receipt. DELETE removes unfinished staging; create prunes expired
unfinished parts under nonblocking locks. It does not remove published objects,
historical results or recovery points. This feature retains existing API uploads
and local Garage; it does not expose public S3 writes or migrate storage off-host.
