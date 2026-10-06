# Build images on a desktop through SSH

The API server keeps source snapshots, build contexts, logs and receipts. Its
private image builder sends the context through SSH to the desktop Docker
Engine. BuildKit unpacks and builds there, and pushes directly to the configured
registry. The API server independently verifies the published manifest and
config digests with Skopeo. Scheduling, datasets and artifact download remain
unchanged. No GPU job is needed to test this connection.

## Connect from WSL

Docker Desktop must be running and WSL must be able to use
`/var/run/docker.sock`. On the API server, provision a root-private directory:

```bash
install -d -m 0700 -o root -g root /run/ml-expd-desktop
```

To recreate it after reboot, put this in
`/etc/tmpfiles.d/ml-expd-desktop.conf`:

```text
d /run/ml-expd-desktop 0700 root root -
```

From WSL, use the server's existing SSH address or alias:

```bash
ssh -NT -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R /run/ml-expd-desktop/docker.sock:/var/run/docker.sock server-alias
```

Keep the session running. If a previous disconnected session left a socket,
the server administrator must verify no listener remains before removing that
exact path. Do not unlink an active tunnel. This connection needs no new public
Docker port or Cloudflare proxy. SSH forwarding must be allowed for the account.

## Configure the private builder

First install the existing administrator-approved, digest-pinned
`buildkit_image` on the desktop (`docker pull` from WSL). Storage probes use
`--pull=never`, so an offline desktop or missing tool image fails closed before
pulling large base layers.

Add to the existing administrator-only builder JSON:

```json
{
  "docker_host": "unix:///run/ml-expd-desktop/docker.sock",
  "publisher": "buildkit",
  "ephemeral_buildkit": true
}
```

Retain the existing pinned `buildkit_image`, repository, allowlist, credentials
and timeouts. **Remove `build_storage_path`**: it describes the API host's Docker
filesystem and cannot be used for the desktop. Remote mode requires ephemeral
BuildKit and cannot disable its remote storage preflight. Apply only while the
builder is idle, with no outstanding builder or probe lease; restart the private
builder after saving a configuration recovery point.

Each preflight runs a restricted, short-lived container using the pinned tool
image and an anonymous volume on the desktop's Docker data filesystem. It
measures available bytes and inodes there, retaining the compressed-layer budget
and reserved space described in [build storage](build-storage.md). This is the
Docker VM filesystem capacity. For Docker Desktop's growing VHD, also monitor
free space on the Windows drive storing it; the VM measurement alone cannot
prove that the Windows backing drive has sufficient physical space.

The context and metadata remain small local files. Training images are pushed
without loading them into either Docker Engine. Only the exact owned temporary
BuildKit container/state volume and storage-probe container/anonymous volume are
removed. Existing desktop images, containers and caches are untouched.

BuildKit containers do not automatically inherit Docker Desktop's daemon proxy.
If HTTP blob downloads need its proxy, the administrator can set
`buildkit_http_proxy: "http://http.docker.internal:3128"`. This endpoint must
contain no credentials. Configure the desktop proxy's routing as well: the
SenseCore CCR may redirect to HTTP on `aoss.cn-sh-01b.sensecoreapi-oss.cn`, which
can return an ICP filing block on mainland direct connections even when its
HTTPS URL succeeds. This is a separate registry/network issue; storage and SSH
availability do not prove a complete CUDA build.

An optional [restricted HTTPS adapter](../server/examples/ssh-builder/https_blobs.py)
can handle this specific CCR redirect. Run it using an administrator-pinned
Python image on a dedicated Docker network, with a read-only filesystem and no
published ports. It accepts only GET/HEAD under the configured CCR blob path,
uses certificate-verified HTTPS upstream, streams without writing blobs to disk,
and logs no signed URLs. Set `buildkit_network` to that network's name and
`buildkit_http_proxy` to `http://ADAPTER-CONTAINER:8080`. This is not a general
proxy and does not change immutable image identities. The small tool container
remains running; transient BuildKit caches are still removed after each build.

Recovery records include the Docker endpoint. Switching endpoints while a lease
exists is rejected instead of cleaning another engine. Recover the lease on the
original engine first. A broken SSH tunnel never falls back to local Docker and
never automatically replays a build; reconcile uncertain publication before an
explicit retry. Existing READY receipts remain reusable while the desktop is
offline. For rollback, restore the saved local-builder configuration and runtime
while idle; retain databases, immutable Runs, registry images and receipts.

Docker references: [SSH and Docker socket access](https://docs.docker.com/engine/security/protect-access/),
[BuildKit containers and state volumes](https://docs.docker.com/build/builders/drivers/docker-container/).
