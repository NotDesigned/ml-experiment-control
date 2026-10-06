# SenseCore: code, Dockerfile, data, execution and recovery

The standalone client only needs Python 3.10+, the ML-Expd HTTPS URL and Bearer
token. The server owns SenseCore/CCR/object-store credentials. All client
operations below use the API; no Docker or cloud tools are needed on the client.

## Build the code and environment

Upload source including a Dockerfile, then let the server build and publish it to
CCR. Use a base digest from `GET /api/environments`; the operator's existing base
image policy still applies to every external build stage. The build context is
the source directory. Code is additionally copied to `/workspace`, and the
managed launcher is installed by the server after the client's instructions.

```bash
ml-exp check --schema openapi.json
ml-exp pack --project my-training --source ./source --dockerfile Dockerfile \
  --entrypoint python3 train.py --state runtime.json
ml-exp runtime --state runtime.json --logs
```

New builds use Dockerfile only (`--dockerfile Dockerfile` is the default).
Install dependencies in the Dockerfile. External `FROM` images must use SHA256
digests; previous named/numeric stages are supported. This first recipe uses the
default Dockerfile frontend and supports normal `RUN`/`COPY` instructions;
custom syntax/escape directives, heredocs, `ADD`, `ONBUILD`, external `COPY
--from` images and build-argument base images are rejected. The final image must
provide Python 3.10+, `/bin/sh`, and permission to create `/inputs` and `/outputs`
aliases and write its configured NAS output directory. The example uses root.
Entrypoint and CMD are replaced by the fixed API entrypoint. Builds execute only
after operator opt-in `allow_dockerfile_builds: true` in the private builder.

The receipt binds the frozen source, original Dockerfile SHA, all external base
digests, generated Dockerfile SHA, launcher SHA and published OCI digest.
Existing READY source-copy/dependency images and old frozen Runs remain valid;
new Runtime preparation rejects those ordinary environment selectors.

Production can use disposable BuildKit: CCR keeps the published digest while
the API host deletes its local build cache and never loads the training image into
Docker. `registry_cache: true` keeps one reusable cache tag per project in CCR. WYD retains the required SIF and removes the temporary conversion cache.
See [storage lifecycle](storage-lifecycle.md).

## Upload data independently

```bash
ml-exp asset-upload --project my-training --directory ./data --state data.json
ml-exp assets --project my-training
```

Assets contain regular files, not links or hidden/credential-looking files.
The client streams a reproducible gzip-compressed tar without loading the dataset into memory.
The archive SHA256 is the upload's idempotency key and asset ID. An asset becomes
READY after extraction/checksums, object-store publication and local metadata
publication. Interrupted uploads can be checked by the saved ID; the same bytes
can be resumed with `asset-upload --resume` on supporting servers. Completed
16 MiB parts are retained and skipped after reconnecting. Dataset bytes do not enter the source bundle or
image. The service exposes actual archive/file/count limits and free local space
at `GET /api/storage-limits`; a null storage quota means no configured overall
quota, not infinite capacity. Data/archive/artifact defaults are 4 GiB and 20,000
files; source retains its separate 64 MiB upload and 256 MiB expanded limits.
See [resumable upload API](multipart-uploads.md) for part checksums, completion,
expiry, cancellation and worker checkpoint/final-output transfers.

## Create and run

For data already available online, a source download script can populate a
backend-local persistent `DATA_DIR` instead of uploading an archive. Both
backends verify and cache its files before training; see [script data](script-data.md).

```bash
ml-exp create --runtime-state runtime.json --run trial --executor sensecore-1gpu \
  --inputs '[{"asset_id":"asset.REPLACE_WITH_DATA_SHA256","mount_path":"/inputs/fineweb"}]' \
  --arguments '["--data","/inputs/fineweb"]' --checkpoint-interval 60 \
  --gpus 1 --cpus 8 --memory-gb 32 --max-time 00:05:00
ml-exp prepare --project my-training --run trial --max-gpu-hours 0.1 \
  --state submission.json
ml-exp execute --state submission.json --confirm 'COPY_THE_EXACT_PREPARE_CONFIRMATION'
ml-exp watch --project my-training --run trial
ml-exp download --project my-training --run trial --attempt attempt-001 --out ./result
```

On a server with a public object-storage endpoint, the current client obtains
a short-lived API link, downloads the archive once, then verifies and unpacks
results locally. The object request carries no API token. Legacy servers keep
the original proxy download.

Replace the asset placeholder with `data.json`'s actual `asset_id`. Use an unused
Run ID and a new client state/output directory. Preparing and creating do not
allocate GPUs. The normal budget, authorization and submission-outbox checks
remain in effect. An uncertain submission is observed/reconciled and never
blindly resubmitted. Published resource profiles describe capabilities, not
immediate availability.

The job mounts the configured NAS at `/data`. The managed worker downloads the
approved assets over HTTPS into a project cache on that NAS, verifies archive and
every file checksum, then exposes the declared `/inputs/<name>` paths. Cached
assets are reverified before reuse. The dataset download uses an exact-Attempt
capability scoped to its frozen input IDs; the user subprocess receives neither
that capability nor global API/cloud credentials. Code uses `/workspace`, inputs
use `/inputs`, and `/outputs` aliases the exact Attempt's `OUTPUT_DIR`.

Input files use 0444/directory 0555 permissions. This is **permission-based write
protection, not a kernel-enforced read-only mount**; root code can change its own
NAS permissions. The immutable server/object-store copy remains the canonical
version, and a modified cache causes later execution to fail verification.
Delivery currently occurs inside the allocated job, so its time counts against
the task's wall-clock/GPU budget. Logs report delivery completion/time/failure.
The aggregate frozen input manifest has a 32 KiB scheduler-command limit.

New Dockerfile runtimes use the same
managed data/checkpoint worker and support SenseCore and WYD.
WYD converts the pinned OCI image to SIF, downloads approved data into its shared
`/datapool` project cache, and binds per-Attempt directories at `/inputs` and
`/outputs` through Apptainer. Existing fixed images and frozen Runs keep their
original behavior; prepare/build a new Runtime to obtain the common contract.
API executor entries advertise `dockerfile_execution` and
`data_asset_transport`. Transfer time is charged on both backends; compression
helps compressible data, while already compressed datasets may still need a
larger wall-clock budget. A 221 MB zero-filled transfer probe does not benchmark
network throughput for real training data.

## Publish and reuse checkpoints

Write immutable checkpoint generations beneath `/outputs/checkpoints`. Once a
generation is fully written, atomically replace `/outputs/checkpoint.ready.json`:

```json
{"step":4,"files":[{"path":"checkpoints/step-0004/state.pt","bytes":12345,"sha256":"ACTUAL_FILE_SHA256"}]}
```

Do not mutate that generation after publishing its ready marker. The worker
checks paths, sizes, SHA256 and changes while copying. It polls only the small
marker when it is unchanged. A changed marker is uploaded as a new immutable
asset; failures are logged and retried, independently of final output upload.

```bash
ml-exp snapshots --project my-training --run trial --attempt attempt-001
```

Snapshots are available during training and survive job termination after a
successful upload. Before publishing, the worker also seeds a separate,
checksum-verified copy in that project's shared data-asset cache. A new Run on
the same backend reuses it after verifying every file, avoiding another download
of model weights over the control server's network. The object-store asset and
its upload receipt remain authoritative; a failed local cache does not prevent
publication. Another backend, a cleared cache, or old workers still download the
asset normally. Cache copies consume backend storage and share the dataset cache
retention policy; they do not keep training images on the control server.
Select a snapshot's `asset_id`, create a new Run using the same
runtime/data plus that asset at `/inputs/resume`, and pass its checkpoint path to
the program. Application code restores its own model, optimizer and RNG state.
ML-Expd does not interpret or unpickle training checkpoints on the server.
References are always scoped to the exact Run/Attempt; no latest-Attempt fallback.
Checkpoint publication is opt-in and bounded by the same configured archive cap.
SIGKILL before the first successful snapshot still loses unpublished progress.

The final output path keeps the existing immutable exact-Attempt archive receipt,
individual file/Range downloads and client SHA256/archive comparison. A training
exit of zero becomes a nonzero worker exit if its final upload fails.

## HTTP endpoints and executable example

| Operation | API |
| --- | --- |
| Storage limits | `GET /api/storage-limits` |
| Upload data tar | `POST /api/assets/archive?project=P&sha256=SHA` |
| Asset metadata/list | `GET /api/projects/P/assets[/ASSET_ID]` |
| Download data/checkpoint tar | `GET /api/projects/P/assets/ASSET_ID/archive` |
| Prepare Dockerfile runtime | `POST /api/projects/P/runtimes/prepare` with `dockerfile` |
| Build/poll/logs | Existing runtime execute/read plus `GET .../runtimes/ID/logs` |
| Bind inputs/start | Existing Run/Submissions APIs with `inputs`, `checkpoint_upload` |
| Live checkpoints | `GET /api/runs/P/R/attempts/A/snapshots` |
| Final artifacts | Existing exact-Attempt files/archive APIs |

Use `Authorization: Bearer TOKEN` and `X-ML-Expd-Client-Protocol: 2` on client
requests. Worker transfer routes are deliberately hidden from the public schema
and reject ordinary/global credentials; clients use the public asset API.

[`examples/sensecore-data`](../examples/sensecore-data) contains a real 65.3M
byte-level transformer and client Dockerfile. It uses a public FineWeb text sample,
CUDA/bfloat16, SGD, and atomic model/optimizer/RNG checkpoints. This is a bounded
workflow acceptance, not a quality or long-duration performance benchmark.
