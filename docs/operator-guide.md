# Deploy and maintain the service

API users start with the [quickstart](api-quickstart.md). This page is for the
operator of a Linux deployment. Templates contain placeholders and disabled
mutations; copying them alone does not produce a runnable GPU service.

## Components and layout

| Component | Access it needs |
|---|---|
| Daemon | Dedicated UID, writable state, backend auth, server S3 credentials |
| Private builder | Docker/Buildx, Skopeo, registry push auth, private socket |
| Slurm | Daemon-owned SSH identity, Slurm/Apptainer, shared storage, registry pull auth |
| PAI DLC | OpenAPI credentials, workspace/quota, personal CPFS/role, registry pull auth |
| SenseCore | REST AK/SK, workspace/pools/specs, CCR pull authorization, mounted NAS |
| Object store/proxy | Private bucket, authenticated upload paths and HTTPS downloads |

Keep one daemon writer per workspace. For the consolidated deployment:

```text
/opt/ml-expd/                 checkout and installed .venv
  .ops/                       private operations evidence
  .recovery/previous/          one verified preceding program
/etc/ml-expd/                 private config/credentials
/var/lib/ml-expd/             Action DB, index, immutable Runs and metadata
/var/lib/ml-expd-image-builder/  configured builder receipts/progress/leases
/srv/ml-expd/projects/        registered external projects, when used
```

Templates may choose a different project/builder root. Make daemon config,
builder `source_root`/socket, service writable paths and actual ownership agree.
Do not relocate backend datasets/checkpoints when rearranging program files.
Code directories/modules are 0755/0644; entrypoints executable. Keep `.git`,
`.ops`, `.recovery` mode 0700. Services use `ProtectHome=tmpfs` with a read-only
bind of `/opt/ml-expd`, exposing code without opening other home directories.
The daemon must not have unrestricted Docker access.

## Install a reviewed commit

Clone into `/opt/ml-expd`, check out an exact reviewed commit, install uv and
Rust 1.85+ for core wheel construction, then build/install core and server:

```bash
uv build --package ml-experiment-control
uv build --package ml-experiment-server
uv venv /opt/ml-expd/.venv
uv pip install --python /opt/ml-expd/.venv/bin/python \
  dist/ml_experiment_control-*.whl dist/ml_experiment_server-*.whl
/opt/ml-expd/.venv/bin/ml-expd --help
```

Use these creation commands for a new installation, not over an active runtime.
Existing deployments need a staged, verified upgrade and program rollback plan.
Install the client separately on client machines. Ensure `experiment-redact`
is on the controller/service PATH. Remove installers/dev caches after verification.

## Configure

Start from [source-api.yaml](../server/examples/source-api.yaml). Provision a
daemon account and allow it to write the configured state/project roots. The
HTTP token must be regular, daemon-owned and mode 0600; create a cryptographically
random token of at least 32 non-whitespace characters without printing it to
shared output. Distribute it privately. There is no per-user/project RBAC API.

Configure these files; keep every actual secret outside Git:

| Template/config | Required choices |
|---|---|
| [Backend executor templates](configuration.md#enable-only-the-backends-you-use) | Choose Slurm, SenseCore or DLC; combine only the entries you need |
| [image-builder.json](../server/examples/image-builder.json) | Daemon peer UID/GID, matching source root/socket, publisher, repository and allowlist |
| [registry-pull.json](../server/examples/registry-pull.json) | Slurm pull-only credential, distinct from publisher auth |
| [artifact-store.json](../server/examples/artifact-store.json) | S3 endpoint/bucket/auth, worker callback base, quotas and part policy |
| Private backend credentials | Configure only selected backends: [Slurm](slurm.md), [SenseCore](sensecore-rest.md), [PAI](pai.md) |

Replace all example hosts/REPLACE values. Example artifact-store values are
explicit quotas, not immutable platform limits: remove them or use null only
after deciding capacity/retention policy. Byte quotas, if present, must be
positive integers. Set `public_endpoint` to a pathless public HTTPS S3 endpoint
and `download_url_seconds` (30–3600) for direct downloads. Preserve signed
Host/path/query through its proxy and do not log signed URL queries.

Provision publisher Docker auth as root-only
`/etc/ml-expd/build-docker/config.json` in a mode-0700 directory. Set builder
`registry_auth_file` to that file for Skopeo and its service environment
`DOCKER_CONFIG=/etc/ml-expd/build-docker` for Buildx pull/push. Keep `BUILDX_CONFIG`
under writable builder state. `ProtectHome=tmpfs` hides `/root/.docker`; a
successful root-shell login does not prove the service can authenticate. Verify
manifest reads and publication as the service inside its sandbox. Never copy
credentials into source/images or expose the whole home to fix authentication.

## Keep the API host temporary

For a host that must not retain datasets/weights, set
`"local_payload_retention_seconds": 86400` in its private artifact-store JSON.
This is an explicit destructive retention policy, disabled when omitted. It is
visible as `local_payload_retention` in `/api/storage-limits`.

- New inputs continue directly to Slurm shared storage or CCR/NAS. Their backend copies
  and persistent checkpoints are outside this policy.
- Historical local dataset archives expire after the configured age. Their
  extracted local files are removed too; immutable asset manifests remain.
- Published result archives expire after that age, including unacknowledged
  archives and `KEEP`. `KEEP` does not override this host maximum. Verified ACK
  with release still removes an archive earlier, with no grace period.
- Terminal Attempts' local binary weight/data caches are removed once locally
  unchanged for that age. Active/unknown Attempts, active/uncertain collection,
  source, logs, metrics, receipts and checkpoint registration are retained.

Expiry records `EXPIRING` before deleting an exact canonical object and `EXPIRED`
after confirmation. A crash retries that same deletion; it never fabricates a
client ACK. Download/restore returns HTTP410 `ARTIFACT_EXPIRED` or `ASSET_EXPIRED`,
instead of silently rebuilding a missing cache. Metadata remains queryable.
The collector runs cleanup; in-flight work and unavailable storage can defer it,
so this is not a hard filesystem quota. Garage physical GC is asynchronous.

Keep original datasets and long-lived model state on backend storage or on the
client. Download selected results before expiry. An expired uploaded legacy
asset must be uploaded to executor storage again if no verified backend copy
exists. Preparation first reuses and verifies an existing backend cache; local
expiry must not block that reuse. This policy deliberately does not
change NAS/shared-storage/CCR retention or require a GPU job to delete local bytes.

The approved base catalogue is an operator-owned YAML selected with
`container_execution.environments_file`:

```yaml
environments:
  torch-base:
    title: Verified training base
    image: registry.example.org/team/base@sha256:ACTUAL_64_HEX_DIGEST
    versions: {torch: OPERATOR_VERIFIED_VERSION, cuda: OPERATOR_VERIFIED_VERSION}
```

Catalogue metadata is operator-verified, not automatic package detection.
Use [builds](builds.md) to configure local or SSH desktop Runtime construction.
New input uploads always select backend storage; `data_upload_storage` and
`data_upload_socket` are legacy-session settings, not a way to select new uploads.
For optional provider-specific transfers, follow [SenseCore data publication](sensecore-rest.md#dataset-publication-and-cpu-copy)
and [PAI storage](pai.md). They are configured independently from Slurm.

## Backend preconditions

Follow [Slurm setup](slurm.md) for daemon-owned SSH, account/QOS/GRES, shared
filesystem scope, Apptainer and optional API relay. The filesystem path is site
configuration; it must be shared by staging and compute nodes.

SenseCore requires REST access, compatible SPOT GPU specs, CCR pull auth and NAS
permissions. Advertised fixed allocation must match actual specs. GPU jobs
exclude debug pools; data-copy jobs use the configured debug CPU pool. NAS need
not be accessible directly from the daemon when workers return over HTTPS.
Backend Internet access is separate from registry/NAS access.

PAI uses [DLC training](pai.md) with its own workspace/quota and approved personal
CPFS/role. DSW debugging/building and OSS transfers are separate opt-ins.

After provisioning, deliberately enable the required `action_runtime` source
imports/project writes/scheduler mutations and finite GPU budget policy. Do not
enable unrelated permissions just to bypass a failed gate.

Administrative Run deletion is separately disabled by default. Deliberately set
`action_runtime.allow_resource_deletions: true` only when the API bearer holder
may delete completed managed Runs, including persistent checkpoints and unacknowledged
or KEEP result objects. SenseCore deletion additionally requires scheduler
mutations and the bounded debug CPU profile. There is no separate RBAC role in
the current single-credential API. Each deletion still requires its reviewed
Action's explicit authorization and execution; see [resource deletion](resource-deletion.md).

## Start and verify

Review/adapt the [daemon](../server/examples/ml-expd.service) and
[builder](../server/examples/ml-expd-image-builder.service) units, daemon config,
PATH, writable state paths and credential permissions before installing them.
Daemon and builder use `/opt/ml-expd/.venv`. Keep the private socket inaccessible
to other users. Public HTTPS can proxy a loopback daemon; direct non-loopback
binding requires native TLS and Bearer auth ([HTTP contract](http_contract.md)).

Verify as the service UID and through the formal HTTPS endpoint:

1. Health/protocol/authentication and all placeholder replacements.
2. Catalogue/base digests/limits, REST/SSH read-only availability and mount scope.
3. A bounded source build through READY, remote digest and cleanup receipts.
4. Existing exact-Attempt artifacts and checkpoint metadata after upgrades.
5. Only with explicit resource authorization: a small real GPU run, checkpoint
   restore and downloaded file hashes on each intended backend.

A CPU starter, successful submit or queue entry does not validate GPU training.
Do not claim queue ETA/free cards without provider evidence.

## Maintenance and rollback

Back up live SQLite with its backup API, or take a stopped-service copy including
WAL. Keep an isolated restore check and one verified preceding program/config
recovery point. Never restore an old DB over newer production state. Refuse
program rollback if guard conditions or active/newer operations make it unsafe.
See [Action storage](action-storage.md).

Registry images, backend datasets/state and historical desktop archives require
explicit retention. No global Docker prune or automatic NAS cleanup exists.
Record exact commit/package versions and verification privately; these docs
are not a list of live jobs or historical test counts. Known defects and
unimplemented recovery capabilities are in [recovery](recovery.md).

## Storage maintenance boundaries

Managed status/observe/controller-collect operations copy only scheduler evidence,
logs and JSONL records (32 MiB per copied file), never declared model weights.
Use the exact Attempt's `/collection` operation to recover large results. Duplicate
requests share its durable operation; recoveries use one global worker lock.

Local uploads and recoveries share disk admission. Configure
`local_storage_headroom_bytes` in the private artifact-store JSON (default
1 GiB). `GET /api/storage-limits` reports `local_storage`: actual free bytes,
unreceived bytes promised to unexpired uploads, active operation reservations,
headroom and admissible bytes. Promises survive daemon restarts; crashed process
reservations are reaped using OS locks, without deleting their transfer data.
Peak estimates include copying, tar creation, validation and a same-host object
store. They are conservative admission checks, not a filesystem quota against
other programs consuming this disk. HTTP 507 `UPLOAD_STORAGE` includes required,
available and reserved bytes. Existing parts and scientific files remain intact.

After local recovery publishes an archive, the server reads the complete stored
object through a bounded checksum sink and only then evicts receipt-matched
delivery cache files. A missing/corrupt object or changed local file keeps the
copy; hidden/unknown files are retained. Cache verification failure is recorded
independently of successful publication. This does not acknowledge a client
download or release the server archive. Transfer timeouts terminate their process
group before removing the operation's own scratch directory.

Every collector cycle prunes expired unfinished multipart sessions and performs
verified-client-ACK archive releases before backend polling. The unfinished
upload TTL is unchanged; unexpired resumable parts are retained. Maintenance
failures appear as `storage_maintenance_error` in `GET /api/health`, while
scheduler polling continues. A download GET is not an ACK. Scientific receipts, registered metrics, datasets and backend
checkpoints survive archive release. Garage physical block reclamation is
asynchronous.

The shared multipart transfer buffer is capped at 1 GiB including conservative
filesystem overhead; saturation returns 503 and retains unconfirmed parts. It is
not a cap on source, result/recovery caches, journals or Garage objects. Review
`transfer_buffer` and `local_storage` independently in `/api/storage-limits`.

Treat API disk, desktop builder disk, Slurm shared storage and SenseCore NAS as distinct
capacity domains. Check each domain's actual write/quota/inode state. In
particular, freeing an API archive does not free NAS training-state generations.
