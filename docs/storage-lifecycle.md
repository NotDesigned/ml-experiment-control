# Temporary images and direct result downloads

The registry owns published OCI images. Compute backends pull the pinned digest;
the API host does not need a permanent copy of each training environment.

## Builds and WYD conversion

Set `ephemeral_buildkit: true` and `buildkit_image` to an approved
`repository@sha256:...` in the private image-builder configuration. Each build
uses its own `docker-container` driver with `default-load=false`, pushes directly
to the registry, and verifies the remote manifest/config digest. The builder is
removed with its dedicated cache/state volume on success or failure. Only small
receipts/logs remain. Builds are serialized to limit transient disk pressure;
temporary layers still require free disk during the build. The pinned BuildKit
tool image itself remains a reusable service dependency.

Build tools use `state_root/tmp` for transient client configuration; a service
may keep the host `/tmp` read-only without preventing BuildKit bootstrap.

A private `ephemeral-builder.json` records the exact owner before creation.
The next builder startup/build cleans up that recorded builder after interruption.
Failed cleanup blocks publication and retains the recovery record. No default
Docker builder, shared cache, registry image or rollback image is pruned. The
legacy archive publisher with `cleanup_published_image: true` removes its exact
output tag after remote verification; it cannot eliminate shared base-layer
cache and is not the production path for large CUDA builds.

WYD converts OCI to SIF directly on its own node. Each conversion uses a private
temporary Apptainer OCI cache, removed on converter exit including failure.
The verified SIF and receipt stay in shared WYD storage for queued/running jobs
and retries. The API host never downloads this SIF or keeps a conversion image.
Old shared Apptainer caches are not automatically pruned. Abrupt host/power loss
may leave conversion temporary files for operator inspection.

## Downloads

Set `public_endpoint` to the public HTTPS **S3 endpoint** in the private
artifact-store configuration, with no added path prefix. `download_url_seconds`
defaults to 300 (allowed 30..3600). Keep the bucket private and preserve original
Host/path/query through the proxy because they form the S3 signature. For a
bucket `ml-expd-artifacts`, forward GET/HEAD `/ml-expd-artifacts/*` without
stripping its path. Do not log signed URL query strings.

- `GET /api/runs/{project}/{run}/attempts/{attempt}/artifacts/download` returns
  a signed GET URL, archive SHA256/size, expiration and exact file inventory.
- `GET /api/projects/{project}/assets/{asset}/download` returns a registered
  dataset/checkpoint archive link and per-file checksums.

The first route reads only the exact Attempt's receipt and never restores a
local expanded cache. Normal API Bearer authentication is required to get links.
The link itself is a temporary bearer capability for that one immutable object:
do not save it in state, reports or public logs. Sharing it grants access until
expiry. Link responses set `Cache-Control: no-store`.

Current `ml-exp download` obtains the link, fetches the archive once **without
API Authorization headers**, checks archive SHA256/size and writes only expected
regular files under a new output directory. New receipts verify file SHA256 as
well; historical receipts lacking file SHA are protected by the complete archive
hash. Redirects are refused. Only HTTP 404 falls back to legacy per-file plus
archive downloads; auth/conflict/network failures do not silently switch paths.
No cloud SDK is needed on the client.

Legacy download routes and local uploaded-output caches remain compatible.
Data uploads, worker inputs, outputs and checkpoints retain existing scoped API
transfers. This change does not add direct PUT or automatically delete results
and recovery copies.

With external S3, download bytes bypass the API host entirely. With local Garage,
bytes bypass ML-Expd and staging but still pass through this host's HTTPS proxy
and use this host's object-storage disk. Moving storage off-host is a separate
operator decision and migration.
