# Runtime image builds

The default publisher builds on a CPU PAI DSW instance and pushes to the
configured registry, such as Docker Hub. A local Docker/BuildKit publisher
is also supported. Building and training are separate: compatible authenticated images
can be used by any configured training backend.

## Dockerfile contract

Clients upload code with a Dockerfile. The server validates it, appends frozen
source at `/ml-expd/source`, which is the working directory unless code is delivered
separately, plus the trusted
worker at `/usr/local/bin/ml-exp-worker-rs`, and publishes the resulting image. The client does not upload a prebuilt Docker image through this workflow.

```dockerfile
FROM REGISTRY/BASE@sha256:ACTUAL_64_HEX_DIGEST
WORKDIR /opt/app
# Only the files that decide the dependencies, so changing code builds nothing.
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project
```

Run it with `--env PYTHONPATH=src`, or however your code finds its packages: the
Dockerfile above installs what the code needs and not the code itself.

### Code delivered to the job

On a PAI executor, `ml-exp submit` reads the `COPY` and `ADD` sources of the
Dockerfile, and uploads those files, the Dockerfile and its `.dockerignore` variants as
the environment. Docker can only see what these instructions name, so a file the
Dockerfile uses cannot be missed. The image's identity depends only on those files, so
changing other code gives a new Run of the same Runtime, with no rebuild and no new
command.

The whole directory is uploaded as well, as the Run's code, frozen with the Run. It holds
the Dockerfile and the lock file too, so editing one of those changes both the Runtime and
the code source; only the reverse is free:

- The worker downloads it before training, into `ATTEMPT_ROOT/code`, and checks every
  file against the source's identity. A file that differs is named in the job's error, and
  training does not start. A repeat of the same Attempt reuses a tree it has proven, or
  refuses; it never overwrites one.
- That directory is the working directory and `SOURCE_DIR`. `/ml-expd/source` is not the
  code path in this case: it holds only the environment files.
- The directory is read-only and `PYTHONDONTWRITEBYTECODE=1` is set. Write to `OUTPUT_DIR`
  and `STATE_DIR`. `--env` cannot set these two names for such a Run.
- A retry or a checkpoint resume keeps the original code, in a new directory of its own.
- The code that goes with a job is at most 64 MiB of files and 20,000 entries. A larger
  directory is not refused: it stays in the image and any change builds it again, and the
  question says so. Move large files to data with [`ml-exp data upload`](data.md) to avoid
  that. A Run named by hand with `code_source_id` over the limit is refused.
- Choosing a Runtime by hand with `code_source_id` needs one built with the current
  worker, unless it can bootstrap (see [below](#the-worker-and-upgrades)).

The whole directory stays in the image, as `COPY . /workspace` asks, and nothing is
delivered separately, when the Dockerfile uses any of these, because what it reads cannot
then be known for certain: a source of `.`, a glob, a variable, a path outside the
directory, `ADD` from a URL, `RUN --mount`, a heredoc or `ONBUILD`. The same holds for a
source that is not among the uploaded files, for a Run with `data_preparation`, whose job
may start before any Attempt has code, and on a Slurm executor. `COPY --from` is another
stage and does not count. The question before submitting says which applies, and why.

A base image with its own `ONBUILD COPY` cannot be seen from here. It would see only the
environment files, so name what it needs in your Dockerfile, or avoid such a base.

`ml-exp status` shows the Runtime and the code source a Run was made from, under
`identity` with `--json`. To name a code source yourself, set `run.code_source_id` in the
experiment file of `ml-exp run start`.

Replace the placeholder with the image digest. Any registry/repository may be
used; ML-Expd does not filter image origins. Private images need credentials on
the builder, and the builder must be able to reach the registry. Every external FROM
must be pinned; named/numeric earlier stages are supported. The supported target
is linux/amd64. Current admission rejects custom syntax/escape directives,
heredocs, build-argument FROM and external-image COPY --from. ADD and ONBUILD
are supported; literal `<<` inside quoted strings or comments is permitted.
Dockerfile size is at most 256 KiB; `ml-expd-build-internal` is reserved.
## The worker and upgrades

On PAI, a server upgrade does not rebuild an environment. A Runtime's identity
names a constant, `worker-bootstrap.v1`, and not the worker in its image, so the same recipe
finds the same Runtime after any number of releases. The worker in the image is a bootstrap:

- A new Run freezes the server's current worker as `native_cloud.worker_sha256`, which is
  also where the Run shows what it ran with. A Runtime records the one it was built with.
  A retry keeps the Run's worker.
- `cloud-launch` first reads `worker_sha256` from the package, and nothing else in it, since
  a later server's package may not decode in this worker. If its own hash differs, it
  downloads that worker from the Attempt's `/worker` exchange endpoint with the Attempt's
  token, retrying a passing failure for at most ten minutes, checks the hash, seals it in a
  memory file, checks that again, and `exec`s it with the same arguments.
- The worker `exec`ed must have that hash, or the job stops. Any failure stops before
  training and the job's log says why; lines start `ML_EXPD_WORKER_BOOTSTRAP=`.
- A memory file needs no writable or executable directory and leaves nothing behind, and needs
  Linux and `/proc`. If that fails, as under a stricter `vm.memfd_noexec`, the worker goes in a
  private temporary directory instead: `$TMPDIR`, `/tmp` or `/var/tmp`, whichever can execute.
  That copy stays for the life of the job. It never goes in the Attempt's directory, which
  cannot exist before the Attempt starts, and on shared storage that may be mounted noexec.

The bootstrap contract is everything a later server asks of a worker in an image: `cloud-launch`
and its arguments, `worker_sha256` at the top of a package, the `/package` and `/worker`
endpoints, `data-import` and `data-copy` with their configuration, channels, storage layout and
the dataset exchange for PAI data jobs, and `script-prepare` with its plan. Changing any of these
means raising `BOOTSTRAP` in `rust/crates/core/src/cloud/build.rs`, which builds every environment
once more, and recognising the old value for stored Runtimes. A test pins these shapes.

A Runtime on the new identity keeps it even if its build is stuck in `RECONCILE_REQUIRED` with
nothing to prove the build left nothing behind, as with `PAI_DSW_CLEANUP_REQUIRED` or BuildKit.
No second build starts by itself. Its view gives an `operator_step`: reconcile again, and if that
does not settle it, confirm no builder is running and the `bundle-` tag is absent from the
registry, then remove the record named. The next attempt starts over.

A Runtime built before this keeps its identity and behaviour: its worker cannot bootstrap,
its Runs freeze nothing and run with the image's worker, and it cannot take delivered code
once that worker is not the current one. The same recipe gets a new identity, so **each
environment builds one more time** after this release, and then no more for an upgrade. A
preparation not yet built moves to the new identity; one built stays on the old.

Slurm keeps today's identity and builds per worker, since its jobs have no exchange to fetch
from and a SIF is materialized per Runtime. To follow later, a Slurm job would need a route to
the worker, such as the gateway placing it on shared storage, and materialization keyed
on the image rather than the worker.

See [platform checks](platform-checks.md) for the distinction between user
requirements, implementation limits and additional platform policies under review.

The final image does not need Python to train: jobs start the statically linked
worker, so training
code can use any language the image provides. The appended layer runs `chmod` through a shell, so the final stage needs
`/bin/sh` and `chmod`. Every backend needs access to its configured
mount/workdir.

`python3` must be on the image's `PATH` only for these optional jobs. The native code
checks no minimum version.

- A separate PAI CPU data materialization, which runs on a READY Runtime image.
  Inline PAI preparation in the training job uses only the worker.
- A read-only PAI diagnostic, which reuses the original job's image.
- A `data_preparation` script, unless its `interpreter` names something else.

Managed ENTRYPOINT/CMD replace client startup commands; configure scientific
argv with Runtime `entrypoint` and Run `arguments`. On Slurm, code is copied into the
image, never overlaid by a second checkout.

## Publication and reuse

The private image builder checks the caller's Unix peer UID. The daemon itself
has no Docker socket permission. BuildKit or the DSW Docker engine publishes images;
Skopeo verifies remote manifest/config digests against build metadata. Runs use
the verified digest, not a mutable cache tag or local RepoDigests assertion.

With `ephemeral_buildkit: true`, each build owns its container/cache volume,
pushes without loading the training image into Docker and cleans its exact
resources after success/failure. A lease records the local builder;
failed cleanup blocks later publication. No global prune is performed.
Each project has one inline cache tag; cache misses permit cold construction. Cached layers do not guarantee GPU capacity or fast cold pulls.
Slurm converts OCI to a digest/checksum-bound SIF on shared storage, retaining the
SIF and removing the converter's private temporary OCI cache.

## Diagnose and resume a failed build

Query the original Runtime's progress, logs and reconciliation endpoint.
Reconciliation checks the owned DSW instance and the published registry image,
and can accept a late completed image without triggering a second build.
Missing receipts alone do not establish a failed publication.
An empty or failed MCP status sample retries observation of the same CPU
build within its configured budget. The start command is sent once; a lost
start acknowledgement does not send it again. A confirmed worker failure or
budget exhaustion triggers cleanup, with its phase, last log offset and error
class retained in the build diagnostic.

If the original CPU instance was deleted and no image was published, use
`ml-exp env reconcile RUNTIME_ID --project PROJECT`. Reconciliation checks the
original instance, instance name, registry and bundle lock without building: a retry
needs the owned cleanup lease, exact absence from native Get/List and a registry 404.
Confirmed absence changes that same Runtime to PREPARED with `retry_safe=true`;
`ml-exp env execute RUNTIME_ID --project PROJECT --confirmation "BUILD RUNTIME_ID"`
can then rebuild it, using the exact confirmation returned by the API. An active
builder, uncertain create or failed observation stays RECONCILE_REQUIRED. Missing receipts do not overwrite the original build error.
Once the Runtime is READY, continue its original Preparation.

## PAI DSW publisher

Use a fixed approved CPU quota and Docker-enabled image with Python 3.11+.
Configure the private image builder, not a DLC training executor:

```json
{
  "publisher": "pai-dsw",
  "allow_dockerfile_builds": true,
  "timeout_seconds": 1200,
  "repository": "docker.io/YOUR_ACCOUNT/RUNTIME",
  "registry_auth_file": "/etc/ml-exp/build-docker/config.json",
  "pai_dsw": {
    "workspace_id": "WORKSPACE_ID",
    "quota_id": "QUOTA_ID",
    "image_id": "DOCKER_ENABLED_CPU_IMAGE_ID",
    "pai_bin": "/usr/local/bin/pai"
  }
}
```

Merge this with the common settings, replacing each example value.
[builder.example.json](../rust/deploy/builder.example.json) has the result:

```json
{
  "socket": "/run/ml-exp-builder/builder.sock",
  "client_uid": 10000,
  "client_gid": 10000,
  "state_root": "/var/lib/ml-exp/builder",
  "native_worker_bin": "/opt/ml-exp/bin/ml-exp-worker-rs",
  "native_source_root": "/var/lib/ml-exp/registry/source-revisions/sources"
}
```

`client_uid` and `client_gid` are the API service account's. `native_source_root` is
`source-revisions/sources` under the daemon's `registry_root`. The builder refuses a
configuration key it does not know, and requires `native_worker_bin` and
`native_source_root`. Its PAI credentials file is selected by `ML_EXP_PAI_CONFIG`, as
the [unit example](../rust/deploy/ml-expd-builder.service) shows.

Omit local BuildKit options in DSW mode. PAI API credentials remain private;
MCP upload/exec uses an isolated CLI profile. Signed REST creates, queries,
stops and deletes only the owned instance. No public Docker port is exposed.

Log into the target registry on the builder host using the configured auth file.
Docker Hub accepts both Skopeo's `docker.io` login and Docker's
`https://index.docker.io/v1/` login entry. Registry requests use Docker Hub's
distribution endpoint; published image references retain `docker.io/...`.
Any other registry uses the same path. One that issues tokens or redirects blobs
from another host, such as ACR, needs those exact hosts in `registry_auth_hosts`
and `registry_blob_hosts`.
Changing `repository` applies to new builds. Existing DSW publication
intents retain their recorded repository; READY images retain their digest.

Each build requests 4 CPU, 8 GiB and no GPU. Configure 20 minutes explicitly as
above; an omitted DSW build timeout currently falls back to 10 minutes. Its
cloud shutdown timer allows the build timeout plus 20 minutes, capped at four
hours, and is read back before upload. Stop timers do not guarantee deletion
after an API-host crash. Known owned instances are cleaned up; unknown creation
or cleanup retains its lease and requires reconciliation.

The context is capped at 300 MiB. Large MCP transfers use 512 KiB pieces and
SHA256-checked reassembly. The build starts once in the background; short polls
read bounded logs and an atomic receipt. Only selected registry credentials are
sent. The DSW uses Docker's classic builder, so BuildKit-only RUN/COPY options
are unsupported. Pinned base pulls have bounded download retries; Dockerfile
execution and pushes are not blindly replayed. Docker cache is temporary.

Optional CPFS workspace/role configuration follows [PAI personal storage](pai.md#approved-cpfs-personal-workspace).
Keep native OSS mounts unset: they remain provider-blocked. With a configured VPC,
`pai_dsw.default_route` selects shared `eth0` or approved private `eth1`; the actual
instance must match before credentials are uploaded. The builder does not
create a NAT, EIP, security group or quota.

The daemon asks for a same-digest copy of every Runtime built for a Slurm executor,
because a gateway may not reach the main registry, so configure `slurm_mirror` on the
builder. Its fields are `target_repository`, plus `source_repository` when the source
differs from the main `repository`, and optional `target_auth_hosts` and
`target_blob_hosts`. The builder copies with Skopeo and verifies the digest; repeated
requests reuse the copy. No API setting is involved, and image layers do not pass
through the API disk. Point it at a registry the
gateway can reach and whose login is in the helper's `registry_auth_file`.

## Local BuildKit storage

The local publisher uses `"publisher": "buildkit"` with the same `repository`,
`registry_auth_file`, `allow_dockerfile_builds` and `timeout_seconds` settings as
above, plus `docker` and `skopeo` executable paths (for example `/usr/bin/docker`,
`/usr/bin/skopeo`). Set `build_storage_path` to the local Docker state filesystem,
for example `/var/lib/docker`. The builder checks free bytes and inodes before downloading
layers. Build logs distinguish space exhaustion from registry or build errors.
