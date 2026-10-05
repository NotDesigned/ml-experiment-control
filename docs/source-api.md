# Source-to-container API

Protocol 2 / server 0.2.0. A remote client needs HTTPS and the existing ml-expd
Bearer token; it does not need SSH, SCO, Apptainer, or a checkout on the daemon.
Start with the [runnable API quickstart](api-quickstart.md); operators use
[the deployment guide](operator-guide.md). Obtain the API URL from your operator.
Every normal API call sends `Authorization: Bearer …` and
`X-ML-Expd-Client-Protocol: 2`. `GET /api/health` bootstraps negotiation;
`GET /api/v2/openapi.json` is the authoritative schema.

## Import and run

1. Import a public HTTPS Git repository at its exact 40-character commit:
   `POST /api/source-imports/git` with
   `{"project":"my-study","url":"https://github.com/owner/project.git","commit":"40 hex characters"}`.
   For private/local code, upload a tar or tar.gz rooted at the source directory:
   `POST /api/source-imports/archive?project=my-study&sha256=<archive SHA256>`
   with the archive as the raw body. The daemon rejects links, escaping paths,
   credential-looking paths, and oversized archives. This is path filtering,
   not a scan proving the contents contain no secrets. Defaults: 64 MiB upload, 256 MiB expanded,
   20,000 entries. Import creates a daemon-managed project when its ID is new.
   An existing project with its own controller is retained and cannot be silently
   converted. The returned `source_id` identifies immutable source content;
   later imports preserve earlier versions.
2. `GET /api/executors` lists operator-defined WYD/SenseCore profiles.
   Clients select a profile ID and common resources. Backend addresses, NAS
   mount definitions, cluster partitions, and registry credentials stay on the
   daemon. SenseCore fixed worker allocations are reported in `capacity`;
   the Run records the actual allocation, which may exceed a minimum request.
3. `POST /api/projects/my-study/runtimes/prepare` with
   `{"source_id":"source.…","image":"registry/repository@sha256:64-hex-digest","entrypoint":["python","train.py"],"workdir":"/workspace"}`.
   The image is a prepared dependency environment, pinned by OCI digest. It
   must provide Python 3, `/bin/sh` and GNU `timeout` for the stdlib upload
   launcher. Alternatively select `environment_id` from `GET /api/environments`
   instead of `image`. Add `requirements`, a relative path inside the imported
   source, to install exact pinned Python dependencies. Prepare returns the
   generated Dockerfile and dependency-file SHA256 for review. The builder runs
   the reviewed installer inside a BuildKit container before copying source,
   preserves the base GPU framework, checks dependency consistency, and records
   actual versions at `/usr/local/share/ml-expd/environment.json` in the image.
   Requirements use `package==version` and optionally complete SHA256 hashes;
   URLs, includes, pip options, markers and source builds are unsupported.
   The daemon never executes a project Dockerfile or installs dependencies on
   the control-plane host. The original FROM/COPY-only recipe remains available.
   The server supplies the reviewed `packaging_revision`; it is part of the
   runtime identity, so a packaging repair creates a new runtime instead of
   replacing an existing frozen image.
4. Execute packaging with `POST /api/projects/my-study/runtimes/{runtime_id}/execute`
   and the exact `confirmation` returned by prepare. Poll the runtime until
   `READY`. Packaging does not allocate a GPU. `RECONCILE_REQUIRED` requires
   `/reconcile`, which reads the packaging receipt; it cannot submit a job.
   Run only after `READY`; retain the returned final immutable OCI digest.
5. `POST /api/projects/my-study/runs` with
   `{"run_id":"trial-001","runtime_id":"runtime.…","executor":"wyd-l40s","arguments":["--epochs","1"],"resources":{"gpus":1,"cpus":8,"memory_gb":32,"max_time":"00:10:00"},"outputs":["**/*"]}`.
   `run_id` cannot be rebound to another definition. Source, derived image,
   entrypoint/arguments, non-secret environment, resource allocation and output
   declarations are frozen together. A retry creates another Attempt using
   this same definition. New source or commands require a new Run.
6. Submit entirely over HTTP:
   `POST /api/experiments/my-study/trial-001/submissions/prepare` with
   `{"max_gpu_hours":1,"reason":"run the frozen trial"}`, then
   `/api/submissions/{submission_id}/authorize`, then `/execute`, using the
   returned confirmation strings. The existing policy gates still apply.
   Preparation validates identity, source, image/staging and resource budget.
   On servers advertising `async-actions.v1`, execution durably claims the
   Action and returns `EXECUTING` while the backend runs in the background.
   Older protocol-2 servers may return the terminal submission status directly. Poll `GET /api/submissions/{submission_id}` until
   `status` leaves `EXECUTING`; `VERIFIED` confirms the exact scheduler job was
   observed, and does not mean training has finished. The Action endpoint
   `GET /api/actions/{action_id}` exposes this state under `execution.status`.
   Execution persists a submission outbox before the scheduler call. Only
   `RECONCILE_REQUIRED` needs `/reconcile`; do not resubmit an uncertain Action.
   List/status/logs/cancel/retry remain the existing Run/Attempt/Action APIs.

The `OUTPUT_DIR`, `PROJECT_NAME`, `RUN_ID`, `ATTEMPT_ID`, `SOURCE_ID` variables
are supplied to the program. Write checkpoints, reports and numeric JSONL
metrics under `OUTPUT_DIR`. Paths are separate for each Attempt; a local
relative output directory outside it is not uploaded. `metrics.jsonl` supports
finite numeric fields (`global_step` is normalized to `step`). The daemon does
not infer the scientific meaning of those fields.
Once an Attempt uploads its outputs, the metrics API reads the complete
`metrics.jsonl` from that exact Attempt before falling back to sampled
collector observations. It never substitutes a previous Attempt's history.

## Same execution on both backends

SenseCore pulls the derived image by digest. WYD converts that same digest to
SIF through Apptainer and verifies an OCI/SIF receipt plus the actual SIF
checksum before reusing a cached conversion. Both use the source packaged at
`/workspace` inside the image; WYD does not copy or mount a second checkout
over it. Legacy projects with an authored SIF and source checkout retain their
existing source staging behavior.
A WYD environment that denies FUSE needs `apptainer_unsquash: true` in its
backend profile. The upload launcher is readable by the non-root compute user.
The profile chooses the scheduler and storage layout; the scientific command
is identical. Private registry access is an operator-owned prerequisite on
both backends. Remote schedulers must pass their live preflight.

The private image builder is a separate, root-owned Unix-socket worker. It
accepts only the daemon UID and a fixed image/source packaging operation; the
HTTP daemon has no Docker socket permission. Image build outcomes are durable
and keyed by source, base digest and upload-launcher revision. Interrupted
packaging can be reconciled without starting an experiment.

With `publisher: buildkit`, the builder uses BuildKit's image
exporter with Docker schema 2, disabled attestations, and existing registry
blob reuse. It avoids exporting and unpacking a second complete CUDA image.
Skopeo verifies the exact remote manifest and configuration digests against
BuildKit's publication metadata. The earlier archive/Skopeo publisher remains
available for rollback. A local `RepoDigests` entry alone is insufficient.

`action_runtime.stage_timeout_seconds` optionally gives environment staging
(including a first OCI-to-SIF conversion) its own timeout; the operator template uses
1200 seconds. Scheduler submission still uses `timeout_seconds` (300 seconds),
and actual GPU execution retains the frozen Run's `resources.max_time` limit.
WYD bootstrap creates its cache and sandbox directories before Apptainer runs,
and Slurm writes stdout/stderr directly into the exact Attempt directory so
startup errors survive. Managed terminal jobs get a final collection even if
they completed before the first polling cycle. API retries use a new Attempt
and accept an existing Run manifest only with an exact digest match, a prior
failed/cancelled/preempted scheduler Attempt, and no conflicting new job.

## Artifacts and object storage

The operator template uses a loopback S3 endpoint, for example Garage on
`127.0.0.1:3900`. An existing S3-compatible service is also supported. The bucket credentials remain on the
server. At actual dispatch, the controller injects a separate write-only
capability for one exact Run/Attempt, valid for seven days. It is absent from
Run manifests, authored campaigns, preparation responses and the user program's
environment. Worker transfer is a narrowly authenticated PUT route, not an
exemption from authentication on the control API.

The image launcher runs the fixed program, forwards termination signals and
then uploads the declared regular files as a tar. Hidden paths and symlinks
are excluded; the API also rejects credentials and escaping archive paths.
The default total archive limit is 2 GiB / 20,000 entries; an operator can
configure a smaller byte limit. The operator template uses 256 MiB. GNU timeout bounds the worker
including upload; a hard kill or lost network can prevent the final upload.
There is no periodic live checkpoint upload in this version. A failed upload
is visible as `ML_EXPD_ARTIFACT_UPLOAD=FAILED` and fails an otherwise successful
worker. Scientific program failure is preserved even when its artifacts upload.

The API validates tar paths, exact Attempt identity and declared outputs,
commits the archive to S3, publishes a local immutable download cache, then
writes a receipt. Same digest retries are idempotent; a different upload cannot
replace a sealed Attempt. Each Attempt has a different capability and key.
Lost local caches are rebuilt from S3 after size and SHA256 verification.

- `GET /api/runs/{project}/{run}/attempts/{attempt}/files`: list available files.
- `GET /api/runs/{project}/{run}/attempts/{attempt}/files/outputs/<path>`:
  stream one file, supporting byte ranges and ETags.
- `GET /api/runs/{project}/{run}/attempts/{attempt}/artifacts/archive`:
  download the uploaded tar directly from S3 through the authenticated API.

An empty file list means no collected/uploaded artifact is available; it does
not claim the remote run produced nothing. Legacy WYD collection remains
readable. Artifacts never fall back to a different Attempt or arbitrary host
path. No automatic artifact or history deletion is enabled.

A single-node object store on the daemon machine is not an off-host backup.
See [the operator guide](operator-guide.md) for component ownership and
sanitized configuration templates. Never publish credential files or dispatch scripts.

## W&B removal and compatibility

The current daemon and ELF code have no W&B SDK, sync, publication, tracking
API, credential-management command, or producer integration. Numeric metrics
and downloadable files are the retained observability surface. Old tracking
parameters are rejected. `/api/tracking` and `/api/observability` return 404.
Historical experiment data, old credentials, backups and rollback environments
are retained without being read by the new implementation. Already frozen old
images remain historical execution definitions; removing current support does
not rewrite their binaries or their old manifests.

Protocol 1 clients must upgrade to protocol 2 because the old health/tracking
contract was removed. Existing Action SQLite schema and scheduler outboxes are
unchanged. The root repository remains the reusable backend library; new
managed projects need no project-specific controller implementation.
