# Runtime image builds

The default publisher builds on a CPU PAI DSW instance and pushes to the
configured registry, such as Docker Hub. An optional ACR Git build provides
fallback when publishing to ACR. A local Docker/BuildKit publisher
is also supported. Building and training are separate: compatible authenticated images
can be used by any configured training backend.

## Dockerfile contract

Clients upload code with a Dockerfile. The server validates it, appends frozen
source at `/ml-expd/source`, which becomes the working directory, plus the trusted
worker at `/usr/local/bin/ml-exp-worker-rs`, and publishes the resulting image. The client does not upload a prebuilt Docker image through this workflow.
`GET /api/environments` lists approved pinned bases, not alternative builders.

```dockerfile
FROM REGISTRY/BASE@sha256:ACTUAL_64_HEX_DIGEST
WORKDIR /workspace
# Install frozen dependencies here, before frequently changing source.
# RUN python3 -m pip install ...
COPY . /workspace
```

Replace the placeholder with the image digest. Any registry/repository may be
used; ML-Expd does not filter image origins. Private images need credentials on
the builder, and the builder must be able to reach the registry. Every external FROM
must be pinned; named/numeric earlier stages are supported. The supported target
is linux/amd64. Current admission rejects custom syntax/escape directives,
heredocs, build-argument FROM and external-image COPY --from. ADD and ONBUILD
are supported; literal `<<` inside quoted strings or comments is permitted.
Dockerfile size is at most 256 KiB; `ml-expd-build-internal` is reserved.
See [platform checks](platform-checks.md) for the distinction between user
requirements, implementation limits and additional platform policies under review.

The final image does not need Python to train: jobs start the statically linked
worker, so training
code can use any language the image provides. The appended layer runs `chmod` through a shell, so the final stage needs
`/bin/sh` and `chmod`. SenseCore ACP wraps the job in
`timeout --signal=TERM --kill-after=30s`, so ACP images need a `timeout` accepting
those options, such as GNU coreutils. Slurm managed I/O binds `/inputs` and
`/outputs` into the image, and every backend needs access to its configured
mount/workdir.

`python3` must be on the image's `PATH` only for these optional jobs. The native code
checks no minimum version.

- A separate PAI CPU data materialization, which runs on a READY Runtime image.
  Inline PAI preparation in the training job uses only the worker.
- A read-only PAI diagnostic, which reuses the original job's image.
- A `data_preparation` script, unless its `interpreter` names something else.

Managed ENTRYPOINT/CMD replace client startup commands; configure scientific
argv with Runtime `entrypoint` and Run `arguments`. Code is copied into the
image, never overlaid by a second checkout on Slurm. The CPU data-copy base
additionally needs glibc/Bash; Alpine is unsuitable for that ACP payload.

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
DSW reconciliation checks the owned instance and published registry image;
ACR reconciliation can accept a late completed image without triggering a
second build. Missing receipts alone do not establish a failed publication.
An empty or failed MCP status sample retries observation of the same CPU
build within its configured budget. The start command is sent once; a lost
start acknowledgement does not send it again. A confirmed worker failure or
budget exhaustion triggers cleanup, with its phase, last log offset and error
class retained in the build diagnostic.

If the original CPU instance was deleted and no image was published, use
`ml-exp env reconcile RUNTIME_ID --project PROJECT`. Reconciliation checks the
original instance, instance name, registry and bundle lock without building.
Confirmed absence changes that same Runtime to PREPARED with `retry_safe=true`;
`ml-exp env execute RUNTIME_ID --project PROJECT --confirmation "BUILD RUNTIME_ID"`
can then rebuild it, using the exact confirmation returned by the API. An active
builder, uncertain create, failed observation, or pending ACR publication stays
RECONCILE_REQUIRED. Missing receipts do not overwrite the original build error.
Once the Runtime is READY, continue its original Preparation.

## PAI DSW and ACR fallback

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
Changing `repository` applies to new builds. Existing DSW/ACR publication
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

For an ACR Personal repository bound to a private build GitHub repository, add:

```json
{
  "acr_fallback": {
    "git_repository": "https://github.com/YOUR_ACCOUNT/YOUR_PRIVATE_BUILD_REPO",
    "wait_seconds": 2700,
    "prefer": false
  }
}
```

ACR must have the built-in rule `release-v$version` → `$version`, context `/`,
file `Dockerfile`. The builder host needs authenticated `git`/`gh`. Configure
exact registry authentication/blob host allowlists as required. ACR Personal
cannot use third-party private-base credentials through this Git build path;
choose a compatible pinned base.

The main `repository` must be the ACR repository the Git build publishes to. A saved
ACR intent is still observed at the repository it recorded if that setting changes.

Set `prefer: true` to build new runtime images directly through ACR while the
DSW Docker engine is unavailable. Existing DSW leases still require confirmed
absence before switching that bundle; an in-flight or unknown primary cannot be
overlapped. Existing ACR publication intents are observed without sending a new
tag. This affects runtime source builds only; data-image delivery keeps its CPU
DSW path and never sends datasets or weights to Git. Omit `prefer` or set it to
false to restore the DSW-first order. The frozen source, worker and Runtime
identity do not change with this operator choice.

Fallback is allowed only after confirmed primary-instance absence and an
authenticated registry absence check. It records one publication intent and
pushes the exact `release-vbundle-BUNDLE_SHA256` tag. It never updates `main` or
overwrites a tag. Git contexts reject empty directories, symlinks/special files,
files above 95 MiB and total contexts above 300 MiB. Put data/weights elsewhere.

`wait_seconds` accepts 0–2700 seconds: 45 minutes is publication observation,
not another DSW build timeout. `WAITING_ACR_BUILD` observes the registry;
`ACR_NATIVE_BUILD_PENDING` requires reconciliation after the wait. Detailed
provider build logs remain in the ACR console. A late exact image can be
accepted without another build. Expiry or an absent receipt alone grants no
retry. A DSW retry additionally requires its owned cleanup lease, exact absence
from native Get/List and registry 404, all rechecked under the bundle lock.

The daemon asks for a same-digest CCR copy of every Runtime built for a SenseCore
executor, so configure `sensecore_mirror` on the builder. Its fields are
`target_repository`, plus `source_repository` when the source differs from the main
`repository`, and optional `target_auth_hosts` and `target_blob_hosts`. The builder
copies with Skopeo and verifies the digest; repeated requests reuse the copy. No API
setting is involved, and image layers do not pass through the API disk.

## Local BuildKit storage

The local publisher uses `"publisher": "buildkit"` with the same `repository`,
`registry_auth_file`, `allow_dockerfile_builds` and `timeout_seconds` settings as
above, plus `docker` and `skopeo` executable paths (for example `/usr/bin/docker`,
`/usr/bin/skopeo`). Set `build_storage_path` to the local Docker state filesystem,
for example `/var/lib/docker`. The builder checks free bytes and inodes before downloading
layers. Build logs distinguish space exhaustion from registry or build errors.
