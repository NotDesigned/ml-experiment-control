# Runtime image builds

The operator chooses a private Docker/BuildKit publisher, local or reached over
SSH, or an optional PAI DSW publisher. Building and training are separate: any
training backend can consume a compatible, authenticated published image. The
client always supplies source plus a Dockerfile through the same API.

## Dockerfile contract

Clients upload code with a Dockerfile. The server validates it, appends frozen
source at `/workspace` plus the trusted worker, and publishes the resulting
image. The client does not upload a prebuilt Docker image through this workflow.
`GET /api/environments` lists approved pinned bases, not alternative builders.

```dockerfile
FROM REGISTRY/BASE@sha256:ACTUAL_64_HEX_DIGEST
WORKDIR /workspace
# Install frozen dependencies here, before frequently changing source.
# RUN python3 -m pip install ...
COPY . /workspace
```

Replace the placeholder with an operator-approved digest. Every external FROM
must be pinned; named/numeric earlier stages are supported. The supported target
is linux/amd64. Current admission rejects custom syntax/escape directives,
heredocs, ADD, ONBUILD, build-argument FROM and external-image COPY --from.
Dockerfile size is at most 256 KiB; `ml-expd-build-internal` is reserved.

The final image needs Python 3.10+, `/bin/sh`, GNU `timeout`, permission to create
`/inputs` and `/outputs`, and access to the configured backend mount/workdir.
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
resources after success/failure. A lease records owner and Docker endpoint;
failed cleanup blocks later publication. No global prune is performed.
`registry_cache: true` uses one inline cache tag per project; cache misses permit
cold construction. Cached layers do not guarantee GPU capacity or fast cold pulls.
Slurm converts OCI to a digest/checksum-bound SIF on shared storage, retaining the
SIF and removing the converter's private temporary OCI cache.

## Diagnose and resume a failed build

`/runtimes/{id}/progress` includes a structured `build_error` and diagnostic
code. Stage messages, including remote bytes/inodes and their required budget,
are written to build logs before BuildKit starts. `BUILDER_DOCKER_UNAVAILABLE`
means the configured Docker tunnel/engine could not be reached; it does not mean
that either the API disk or the desktop disk is full. Historical discarded
stderr cannot be reconstructed.

Keep the original Runtime and call its `/reconcile` with its saved confirmation.
A missing local receipt alone never permits a new build. For a current remote
BuildKit definition, reconciliation can additionally check under both private
locks that no lease/container remains and the authenticated registry returns
404 for the exact bundle tag. Only this positive, identity-bound observation
can grant `BUILD_PUBLICATION_ABSENT` with `retry_safe: true`. Registry/network
errors, busy builders, malformed observations and existing-but-unverified
manifests do not grant retry. Reconcile never builds or deletes an image.

An explicit `/execute` can then retry the same Runtime; the worker checks
publication absence again under its locks immediately before creating a builder.
Old observations cannot overwrite a newly published image. Prior error evidence
is kept in `build_error_history`; unknown failures have a structured error and
remain unsafe. When the Runtime is READY, continue the original Preparation.
This prepares a Submission; GPU authorization/execution remain separate.

## SSH desktop setup

On the API host, create/recreate a root-private socket directory:

```bash
install -d -m 0700 -o root -g root /run/ml-expd-desktop
```

Use `/etc/tmpfiles.d/ml-expd-desktop.conf` containing
`d /run/ml-expd-desktop 0700 root root -`. From WSL with Docker Desktop running:

```bash
ssh -NT -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R /run/ml-expd-desktop/docker.sock:/var/run/docker.sock server-alias
```

Keep SSH and Docker running. Confirm an abandoned socket has no listener before
removing that exact path; never unlink an active tunnel. No public Docker port,
Cloudflare proxy or local fallback is required.

Configure the root-owned builder with `docker_host` set to
`unix:///run/ml-expd-desktop/docker.sock`, `publisher: buildkit`,
`ephemeral_buildkit: true`, pinned `buildkit_image`, repository/auth and allowlist.
**Remove local `build_storage_path` in remote mode.** Provision the pinned tool
image on the desktop before probes, which use pull=never. New dataset uploads
bypass Docker and the desktop: Slurm receives parts over SSH, while SenseCore
receives a streamed CCR data image. The old desktop data helper is retained only
for historical assets/sessions; do not provision it for new input uploads.

Source contexts are bounded gzip archives, sent using Docker's archive API,
SHA/size checked and sealed on the desktop before BuildKit reads them across
the private bridge. This avoids WAN BuildKit filesystem-session transfer.
Scoped leases remove only that source context. Historical desktop dataset assets
remain separate and are not removed by build cleanup.
A broken tunnel fails closed; reconcile uncertain publication before retrying.

## PAI DSW and ACR fallback

Use a fixed approved CPU quota and Docker-enabled image with Python 3.11+.
Configure the private image builder, not a DLC training executor:

```json
{
  "publisher": "pai-dsw",
  "allow_dockerfile_builds": true,
  "timeout_seconds": 1200,
  "repository": "REGISTRY/NAMESPACE/RUNTIME",
  "registry_auth_file": "/etc/ml-expd/build-docker/config.json",
  "base_image_prefixes": ["REGISTRY/"],
  "pai_dsw": {
    "workspace_id": "WORKSPACE_ID",
    "quota_id": "QUOTA_ID",
    "image_id": "DOCKER_ENABLED_CPU_IMAGE_ID",
    "pai_bin": "/usr/local/bin/pai"
  }
}
```

Merge this with the common peer UID/GID, source/state root and socket settings
from [image-builder.json](../server/examples/image-builder.json). Omit desktop
Docker endpoint/BuildKit options in DSW mode. PAI API credentials remain private;
MCP upload/exec uses an isolated CLI profile. Signed REST creates, queries,
stops and deletes only the owned instance. No public Docker port is exposed.

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
    "wait_seconds": 2700
  }
}
```

ACR must have the built-in rule `release-v$version` → `$version`, context `/`,
file `Dockerfile`. The builder host needs authenticated `git`/`gh`. Configure
exact registry authentication/blob host allowlists as required. ACR Personal
cannot use third-party private-base credentials through this Git build path;
choose a compatible pinned base.

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

For SenseCore consuming a different private registry, configure matching
`sensecore_mirror` on the builder and
`container_execution.sensecore_image_mirror` on the API. Their fields are
`source_repository` and `target_repository`. Staging streams a digest-preserving
copy to CCR and verifies it before scheduling; repeated requests reuse it.
Image layers do not accumulate on the API disk. Old frozen image/delivery
references remain unchanged.

## Capacity and progress

Local builders can configure `build_storage_path` on the Docker-state filesystem.
Remote builders must measure bytes/inodes on a restricted desktop volume probe.
Default conservative budget: 4× compressed external base layers + 2× context
bytes + 2 GiB reserve; at least 100,000 free inodes. Config keys are
`build_expansion_factor`, `build_reserve_bytes`, `build_min_free_inodes`.
This is not measured unpacked size or a guarantee for arbitrary RUN commands.
Docker Desktop also needs enough physical space on its Windows backing drive.

Runtime `/logs` and `/progress` expose storage, source staging, build/push and
digest verification. Structured errors include BUILD_STORAGE_INSUFFICIENT,
BUILD_STORAGE_UNCHECKED, BUILD_DISK_EXHAUSTED and transport/session errors.
Check `build_error.retry_safe`, logs and exact receipt. A retry-safe capacity
failure still needs explicit operator action; ENOSPC/publication uncertainty
requires reconciliation. Old unbuilt recipes need fresh Runtime preparation.

## Optional verified CCR layer cache

`server/examples/ssh-builder/https_blobs.py` is a restricted GET/HEAD adapter for
the fixed CCR blob host/path. It streams certificate-verified HTTPS, stores no
signed URLs in logs and has no published port. It can use an owned compressed
blob cache volume (`CCR_BLOB_CACHE=/cache`); unverified partials are never hits.
Choose `buildkit_network`/credential-free `buildkit_http_proxy` to reach it.

For slow desktop-to-CCR transfers, an operator can use the separate bounded
relay `ccr_relay.py` and `prewarm_ccr.py`:

```bash
python3 server/examples/ssh-builder/prewarm_ccr.py \
  --config /etc/ml-expd/image-builder.json \
  --image REGISTRY/REPOSITORY@sha256:ACTUAL_CCR_DIGEST \
  --relay-url https://api.example.org/ml-expd-builder-relay \
  --relay-token-file /etc/ml-expd/builder-relay.token --max-cache-gib 16
```

The relay needs its own private token/TLS configuration and fixed restricted
routing; it is not an experiment API or general proxy. Prewarm downloads exact
ranges directly to desktop, verifies complete layer hashes, resumes verified
partial prefixes and renews short-lived upstream signatures. It does not build
or schedule. Use at least 256 MiB for the adapter during prewarm; inspect owned
cgroup OOM evidence on failure. Monitor its bounded cache and backing drive;
no automatic cache/registry deletion is implied.

Keep build-command timeout separate from RPC wait and worker duration. Set the
builder's `timeout_seconds` explicitly and inspect the actual configuration;
the private build RPC permits up to 7200 seconds. For optional DSW and ACR build
budgets, see [PAI configuration](pai.md). Remove obsolete owned cache blobs only
while idle; retain pinned
images, results and recovery evidence.
