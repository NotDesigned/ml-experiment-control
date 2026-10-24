# Dockerfile builds and the desktop builder

An optional [CPU DSW publisher](pai.md#dsw-builder-without-enterprise-acr) builds
through PAI and pushes to an existing OCI registry, without enterprise ACR.
It is opt-in; desktop configuration remains the default.

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
has no Docker socket permission. BuildKit publishes training Runtime images;
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

Keep build-command timeout separate from RPC wait and worker duration. The
current installation uses a 900-second build command and 1200-second private
RPC wait, but these are operator settings, not API guarantees. Check actual
configuration. Remove obsolete owned cache blobs only while idle; retain pinned
images, results and recovery evidence.
