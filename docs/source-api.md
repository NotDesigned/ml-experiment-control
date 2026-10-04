# Source-to-container API

Protocol 2 / server 0.2.0. A remote client needs HTTPS and the existing ml-expd
Bearer token; it does not need SSH, SCO, Apptainer, or a checkout on the daemon.
API entry on this server: `https://api.adscn.dev/ml-expd`.
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
   credentials, and oversized archives. Defaults: 64 MiB upload, 256 MiB expanded,
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
   launcher. The daemon uses a fixed FROM/COPY recipe, adds the reviewed upload
   launcher, and publishes a derived image. It never executes a project
   Dockerfile or installs arbitrary dependencies on the host.
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
   `POST /api/experiments/my-study/trial-001/submissions/prepare`, then
   `/api/submissions/{submission_id}/authorize`, then `/execute`, using the
   returned confirmation strings. The existing policy gates still apply.
   Preparation validates identity, source, image/staging and resource budget.
   Execution persists a submission outbox before the scheduler call and verifies
   the exact job. An uncertain result requires `/reconcile`; do not resubmit.
   List/status/logs/cancel/retry remain the existing Run/Attempt/Action APIs.

The `OUTPUT_DIR`, `PROJECT_NAME`, `RUN_ID`, `ATTEMPT_ID`, `SOURCE_ID` variables
are supplied to the program. Write checkpoints, reports and numeric JSONL
metrics under `OUTPUT_DIR`. Paths are separate for each Attempt; a local
relative output directory outside it is not uploaded. `metrics.jsonl` supports
finite numeric fields (`global_step` is normalized to `step`). The daemon does
not infer the scientific meaning of those fields.

## Same execution on both backends

SenseCore pulls the derived image by digest. WYD converts that same digest to
SIF through Apptainer and verifies an OCI/SIF receipt plus the actual SIF
checksum before reusing a cached conversion. Both use the source packaged at
`/workspace` inside the image; WYD does not copy or mount a second checkout
over it. Legacy projects with an authored SIF and source checkout retain their
existing source staging behavior.
The profile chooses the scheduler and storage layout; the scientific command
is identical. Private registry access is an operator-owned prerequisite on
both backends. Remote schedulers must pass their live preflight.

The private image builder is a separate, root-owned Unix-socket worker. It
accepts only the daemon UID and a fixed image/source packaging operation; the
HTTP daemon has no Docker socket permission. Image build outcomes are durable
and keyed by source, base digest and upload-launcher revision. Interrupted
packaging can be reconciled without starting an experiment.

Publication uses Skopeo to normalize Docker's exported archive to Docker
schema 2 for registry compatibility. It checks the remote manifest digest and
retains the original image configuration digest; a local `RepoDigests` entry
alone is not a publication receipt. Repeated publication precomputes layer
digests to reuse existing registry blobs. The fixed source and command remain
unchanged by this representation conversion.

## Artifacts and object storage

This deployment uses [Garage S3](https://garagehq.deuxfleurs.fr/documentation/quick-start/),
listening only on `127.0.0.1:3900`. The S3 bucket credentials remain on the
server. At actual dispatch, the controller injects a separate write-only
capability for one exact Run/Attempt, valid for seven days. It is absent from
Run manifests, authored campaigns, preparation responses and the user program's
environment. Worker transfer is a narrowly authenticated PUT route, not an
exemption from authentication on the control API.

The image launcher runs the fixed program, forwards termination signals and
then uploads the declared regular files as a tar. Hidden paths and symlinks
are excluded; the API also rejects credentials and escaping archive paths.
The total archive limit is 2 GiB / 20,000 entries. GNU timeout bounds the worker
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

Garage here is a single-node persistent store on this machine, not an off-host
backup. Operator files: `/root/services/ml-expd-artifacts`; private S3 config:
`/etc/ml-expd/artifact-store.json`; private packaging config:
`/etc/ml-expd/image-builder.json`; backend profiles:
`/etc/ml-expd/executors.yaml`. Never publish credential files or dispatch scripts.

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
