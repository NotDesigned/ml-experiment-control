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

## Prewarm slow CCR base layers

Registry inline build cache does not eliminate the first download of a base
layer into each temporary builder. If the API host reaches CCR faster than the
desktop, the operator can retain **verified compressed base blobs on the
desktop**, while still deleting each temporary BuildKit container and its
unpacked layers. This does not retain training images on the API host or change
any pinned image digest.

Configure the restricted adapter with `CCR_BLOB_CACHE=/cache` and a dedicated
Docker volume mounted at `/cache`, writable by its existing non-root UID. Keep
the adapter's read-only root filesystem, private network, memory limit and lack
of published ports. Provision only this owned volume; do not change permissions
on another workload's volume. Save the old container's exact configuration
before replacing it, and apply only with an idle builder and no outstanding
lease. A mount alone does not populate the cache.

Run [the restricted range relay](../server/examples/ssh-builder/ccr_relay.py)
as an unprivileged systemd service on `127.0.0.1:8878`, with a separate random
private credential supplied through `LoadCredential`. Route only
`/ml-expd-builder-relay` through the existing HTTPS reverse proxy. The service
requires its dedicated Bearer credential and accepts signed URLs only for the
fixed CCR blob host/path, with exact ranges of at most 4 MiB and sixteen concurrent
upstream requests. It validates the response offset, total size and actual byte
count; upstream redirects are rejected. Signed URLs travel in POST bodies,
never access-log query strings. Do not enable request-body logging or reuse the
platform API token. Keep its credential out of public build contexts and logs.
This is an operator prewarming endpoint, not a client experiment API.

If the API host's outgoing path to the desktop is also slow, the same relay can
run on a better-connected auxiliary host without Docker or image storage. Use
an existing valid certificate and a separate free HTTPS port:

```bash
python3 ccr_relay.py --host :: --port 8443 \
  --token-file /run/credentials/ml-expd-ccr-relay.service/relay-token \
  --cert-file /run/credentials/ml-expd-ccr-relay.service/tls-cert \
  --key-file /run/credentials/ml-expd-ccr-relay.service/tls-key
```

Supply these files with systemd `LoadCredential`; never move existing service
certificates or print their keys. Non-loopback binding requires both certificate
and key. Clients retain normal certificate and hostname verification. The
listener supports IPv4/IPv6 and bounds idle TLS handshakes in request threads.
Keep the existing API-host relay as a fallback, and point prewarming's
`--relay-url` and its private token file at the chosen host. Check an actual
desktop-to-relay range download before claiming a speed improvement.

Certificates copied by `LoadCredential` refresh when the service restarts.
An operator may use `RuntimeMaxSec=1d` and `Restart=always` to pick up the host's
already-renewed certificate daily, without changing its existing renewal hooks.
This restarts only the owned relay; ranges are idempotent and retryable, and no
scheduler or GPU operation is involved.

From the API host, using its existing private registry authentication:

```bash
python3 server/examples/ssh-builder/prewarm_ccr.py \
  --config /etc/ml-expd/image-builder.json \
  --image registry.cn-sh-01.sensecore.cn/ccr-zhicheng-02/elf@sha256:ACTUAL_DIGEST \
  --relay-url https://api.example/ml-expd-builder-relay \
  --relay-token-file /etc/ml-expd/builder-relay.token \
  --max-cache-gib 16
```

The command accepts only a digest-pinned, single-platform CCR manifest. It
checks the manifest identity and passes a small private control document through
the SSH Docker endpoint. The desktop downloads exact ranges over verified HTTPS
with sixteen bounded concurrent 2 MiB requests (at most 32 MiB of range data in
one downloader batch). No blob bytes pass through SSH and no
layer is written to API-host disk. The desktop checks the
full SHA256 and byte count before atomically publishing a read-only blob.
Existing blobs are rehashed before reuse. Only complete content-addressed blobs
can be served; temporary files, symlinks and special files cannot be cache hits.
Conditional/range requests retain the original TLS upstream behavior. BuildKit
independently verifies downloaded layer digests.

The cache is operator-managed, with a default 16 GiB hard write budget and a
2 GiB free-space reserve; competing writes fail on its dedicated lock. Monitor
the desktop's Windows backing drive as well. There is no global prune or
automatic deletion: remove selected obsolete digest files only from this owned
cache, while idle. A writer stops after 120 seconds without input.
Unverified `.partial-<digest>` prefixes remain
private and can resume when the operator reruns the command. The writer rehashes
the entire prefix, downloads only the remaining ranges, and publishes only if
the whole SHA256 matches. A corrupt complete prefix is removed. Symlinks, special
files, hardlinks, oversized prefixes and changing offsets are rejected. Capacity
accounts for the already stored prefix; its bytes are not counted as new network
transfer. Older unnamed `.partial-*` files require operator inspection while
idle. Registry images and other Docker volumes remain untouched.

Use at least 256 MiB for the owned adapter container when prewarming. The 16-way
Python/SSL downloader shares its cgroup with the serving process; a real 4 GB
transfer exceeded the previous 128 MiB limit. The command checks the configured
memory budget before retrieving private credentials or starting any downloads.
It prints only a numeric child exit code on failure; inspect the owned cgroup's
`memory.events` to distinguish OOM from network or checksum failures.

CCR redirect signatures expire after 1200 seconds. The operator obtains a fresh
signed URL every 300 seconds and sends only that small control over the private
Docker connection. The desktop continues the same download; each range retry
uses the current URL. Logs report renewal without revealing the signature.
Renewal failure aborts rather than publishing an unverified blob. The complete
layer watchdog is 3600 seconds, outside the image-build budget.

This command does not build an image or submit a GPU job. Reconcile an existing
uncertain Runtime receipt before explicitly retrying it. Keep the SSH connection
running throughout prewarming and construction. The daemon's default private
builder response wait is 1200 seconds, longer than the production builder's
900-second command limit; this prevents the previous 600-second wait from marking
an ongoing build uncertain too early. It does not increase the build or GPU
budget, and a genuinely lost response still requires reconciliation.

Recovery records include the Docker endpoint. Switching endpoints while a lease
exists is rejected instead of cleaning another engine. Recover the lease on the
original engine first. A broken SSH tunnel never falls back to local Docker and
never automatically replays a build; reconcile uncertain publication before an
explicit retry. Existing READY receipts remain reusable while the desktop is
offline. For rollback, restore the saved local-builder configuration and runtime
while idle; retain databases, immutable Runs, registry images and receipts.

Docker references: [SSH and Docker socket access](https://docs.docker.com/engine/security/protect-access/),
[BuildKit containers and state volumes](https://docs.docker.com/build/builders/drivers/docker-container/).
