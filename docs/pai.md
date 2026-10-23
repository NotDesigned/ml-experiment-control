# Alibaba PAI: DLC training and DSW debugging/builds

PAI scheduling and image construction are separate adapters. DLC runs the
experiment; an optional Docker-enabled DSW builds its OCI image. The normal
source → Runtime → Run → Submission → Attempt workflow is unchanged. Configuring
one does not implicitly enable the other or allocate public compute.
DSW is not a training executor. Production experiments use DLC; DSW is reserved
for interactive debugging and the optional CPU image builder.

## DLC executor

The initial adapter supports one worker, 1–8 GPUs, a fixed region/workspace/quota,
digest-pinned images, status, cancellation, and current/historical pod logs.
It uses the official signed OpenAPI SDK with HTTPS and write retries disabled.
Provider restart/Spot are disabled. A lost submission response is reconciled by
its exact Run/Attempt/scope/submission tags; it never triggers a replacement job.
Provider reason text is reported when available; exit code and queue ETA remain
unknown without evidence.

Keep host credentials in `/etc/ml-expd/pai.json` (or set
`EXPERIMENTCTL_PAI_CONFIG`). Install the optional core `pai` dependencies when
using the core alone; the server includes the official SDK. The config format is:

```json
{
  "region": "REGION",
  "workspace_id": "WORKSPACE_ID",
  "access_key_id": "SET_PRIVATELY",
  "access_key_secret": "SET_PRIVATELY",
  "registry_auth_file": "/etc/ml-expd/pai-registry-auth.json"
}
```

An optional `security_token` supports temporary credentials. Never put credentials
in an experiment config, manifest or log. A private Docker auth file is injected
only into the host's CreateJob request as `JobSpecs[].ImageConfig`, for the exact
image registry. Rendering or freezing a Run contains no pull password. Restrict
credential files to the service identities that need them.

An operator executor profile can contain:

```yaml
executors:
  pai-h200:
    title: PAI H200
    capacity: {gpus: 8, cpus: 216, memory_gb: 1800}
    storage_root: /mnt/workspace/ml-expd
    backend:
      kind: pai-dlc
      region: REGION
      workspace_id: WORKSPACE_ID
      quota_id: QUOTA_ID
      gpu_type: H200
      gpu_type_from_quota: true
      time: '00:10:00'
```

Replace capacity with verified per-node limits. The server sets image, job name
and walltime when freezing a Run. `ResourceId` is mandatory; no public/pay-as-you-go
fallback is attempted. Clients may name the executor or select backend `pai-dlc`.
On Lingjun quotas that reject `ResourceConfig.GPUType`, set
`gpu_type_from_quota: true`. Submission preflight then requires the quota's
reported GPU type to equal the frozen executor's `gpu_type`, and omits only the
native per-pod GPU type selector. GPU count and fixed quota remain explicit.

`GET /api/executors/{id}/status` returns the quota's `ActualMinQuota` and
`AllocatedQuota`, converted to `gpus`, `cpus` and `memory_bytes`. The query verifies
region and workspace binding. Its 25-second cache uses the existing
OBSERVED/STALE/UNKNOWN semantics. Arithmetic unallocated quota is neither GPU
utilization nor an admission guarantee; pending queue count and ETA are unknown.

### Data and recovery limits

The operator can configure the approved personal CPFS directory directly:

```yaml
      cpfs_uri: bmcpfs://cpfs-FILESYSTEM-TARGET.REGION.cpfs.aliyuncs.com/
      workspace_user: YOUR_RAM_USER
      data_role_arn: acs:ram::ACCOUNT_ID:role/pai-data-YOUR_RAM_USER
      filesystem_type: cpfs
      storage_id: IMMUTABLE_CPFS_FILESYSTEM_ID
```

This generates `/users/YOUR_RAM_USER/` → `/mnt/workspace` RW,
`/public/models/` → `/mnt/models` RO, and `/public/datasets/` →
`/mnt/datasets` RO. No filesystem root, other user's directory or project is
mounted. VPC endpoints also require the matching `user_vpc` with `VpcId`,
`VSwitchId` and `SecurityGroupId`; VSC endpoint affinity must be verified by the
operator. Additional versioned Dataset mounts cannot shadow these paths.
The personal RAM role is injected using DLC's `EnableCredentialInject` and
`CredentialConfigItems` schema. Host access keys and registry passwords are
never put into the frozen Run.

Existing operator-configured native versioned PAI Dataset mounts remain valid:

```yaml
      data_sources:
        - DataSourceId: DATASET_ID
          DataSourceVersion: v1
          MountPath: /data
          MountAccess: RW
      filesystem_type: nas
      storage_id: IMMUTABLE_FILESYSTEM_ID
```

Only an explicit RW NAS/CPFS dataset or personal CPFS mount covering `storage_root` enables persistent
checkpoint and in-job download-script capabilities. Its storage identity includes
region, workspace, filesystem ID and project path; quota/GPU size does not change
that identity. Declaring an ID does not prove the remote mount exists.

Platform-uploaded data/checkpoint assets, pre-job CPU data preparation, offline
CPU result recovery and managed remote Run deletion are **not implemented** for
PAI. Unsupported requirements are refused. Successful managed workers can use
the existing result/metric transfer protocol; missing worker uploads do not
silently schedule a CPU recovery job. Do not equate this initial adapter with
full WYD/SenseCore feature parity.

## DSW builder without enterprise ACR

PAI's dedicated CreateImageBuild API requires enterprise ACR. The DSW route does
not call that API: it uses a private CPU DSW in a fixed quota, enables Docker,
builds the sealed context and pushes to the configured accessible OCI registry
(including an existing SenseCore CCR repository).

Set the private image-builder configuration to opt in:

```json
{
  "publisher": "pai-dsw",
  "allow_dockerfile_builds": true,
  "state_root": "/var/lib/ml-expd/image-builder",
  "source_root": "/var/lib/ml-expd/projects/source-revisions/sources",
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

Use a CPU image with Python 3.11+; Docker must be supported by its resource quota.
Omit desktop `docker_host`, `ephemeral_buildkit` and related Docker options in
this configuration. The builder uses PAI CLI's MCP upload/exec channel with a
private temporary CLI profile; creation/status/stop/deletion use signed REST.
It does not modify the operator's logged-in CLI profile or expose Docker TCP.

Optionally add an OSS mount to `pai_dsw`:

```json
{
  "oss_uri": "oss://BUCKET.oss-REGION-internal.aliyuncs.com/users/YOUR_RAM_USER/",
  "oss_mount_path": "/mnt/ml-expd-oss"
}
```

Confirm the bucket, region and account's directory with the operator first; the
builder does not infer directory ownership from its name. It passes this explicit
prefix to DSW's native `Datasets` configuration with `MountAccess: RW`. The default
mount path is `/mnt/ml-expd-oss`; custom paths must be below `/mnt` or `/data`.
Bucket-root mounts, traversal, signed URLs and cross-region endpoints are rejected
before creating a DSW. Omit both fields to retain the existing unmounted behavior.
This makes files available in the DSW, not automatically inside Docker build RUN
steps. Source context still uses MCP; this option does not persist Docker layers
or enable a direct client OSS-upload API. OSS is object storage and should not be
treated as a POSIX checkpoint filesystem.

### Approved CPFS personal workspace

For an existing CPFS AI Edition VPC mount target, add these fields to `pai_dsw`
after an administrator approves the account's personal directory:

```json
{
  "cpfs_uri": "bmcpfs://cpfs-FILESYSTEM-vpc-TARGET.REGION.cpfs.aliyuncs.com/",
  "workspace_user": "YOUR_RAM_USER",
  "user_vpc": {
    "VpcId": "vpc-REPLACE",
    "VSwitchId": "vsw-REPLACE",
    "SecurityGroupId": "sg-REPLACE"
  }
}
```

Use the actual active mount-target endpoint and matching VPC from the NAS/CPFS
API or console; do not guess an endpoint or infer permission from visibility.
The builder creates exactly these CPFS mounts:

| Container path | CPFS directory | Access |
| --- | --- | --- |
| `/mnt/workspace` | `/users/YOUR_RAM_USER/` | RW |
| `/mnt/models` | `/public/models/` | RO |
| `/mnt/datasets` | `/public/datasets/` | RO |

DSW automatically binds its workspace source to `/mnt/workspace`. The personal
CPFS dataset is therefore registered at `/mnt/ml-expd-personal` and that alias is
selected as `WorkspaceSource`; both paths reference the same approved personal
directory. Registering the dataset directly at `/mnt/workspace` creates duplicate
volume mounts on this platform and prevents startup. Public paths stay as shown.
For a VPC endpoint, use the matching user VPC. A native VSC endpoint can omit
`user_vpc` only after the official filesystem/HPN affinity API confirms it. No project or filesystem-root mount is created. Project
mounts require a separate approved configuration and are not part of this option.
Public access modes are fixed in code; they cannot be upgraded through config.
The optional OSS mount may coexist at a separate path, but cannot shadow these
mounts. The operator must bind `workspace_user` to the authenticated account.

The DSW workspace is persistent CPFS. Build control files and registry secrets
remain under the owned `/tmp/ml-expd-build-*` directory; Docker image layers still
use the engine's temporary storage. This does not create a cross-instance Docker
cache or change a client's Dockerfile `WORKDIR` or frozen source identity.

Each build requests 4 CPU, 8 GiB, 0 GPU. A cloud shutdown timer is created and read
back before upload; its 30-minute stop budget survives the API host exiting.
Known owned instances are stopped/deleted in cleanup. Uncertain creation or
cleanup retains a durable identity lease and requires operator reconciliation;
it never repeats CreateInstance or blindly rebuilds an existing bundle.
A timer stops an instance; it does not guarantee deletion after a host crash.

Only code/build context is transferred, bounded at 300 MiB. Larger files use
512 KiB MCP parts and SHA256-checked reassembly. Data and weights belong in
separate datasets, not the Docker context. Only the target registry's auth is
sent to the DSW. Its stdlib client calls the Docker Engine through the Unix proxy;
no Docker/buildx binary or privileged BuildKit container is uploaded.

This route uses the Docker classic builder: standard COPY/RUN work; BuildKit-only
RUN/COPY options are unsupported. Managed launcher executable mode is sealed in
the context, so the platform recipe needs no `COPY --chmod` extension. The recipe
hash distinguishes new bundles; frozen historical image identities are retained.

The remote build is launched once in the background. Short observations read a
separate atomic status/receipt and bounded log deltas, avoiding MCP gateway
timeouts on long calls. Startup and build each have a ten-minute bound; transfers
and cleanup also have individual bounds. The host independently checks the exact
registry tag's manifest digest before accepting the receipt. A publication and a
cleanup failure retain both facts; cleanup uncertainty prevents reporting a new
build as successful. No automatic recovery from an uncertain lease is claimed.

DSW has network and disk limits. In particular, Docker Hub access may fail; use
an accessible pinned base. Docker's cache disappears when a DSW restarts, even
with a persistent disk. This implementation starts a fresh instance per build;
it does not promise warm layers, arbitrary CUDA image capacity or fast cold pulls.
A small successful build is not a large-image or training acceptance test.

Official references: [Docker in DSW](https://help.aliyun.com/zh/pai/using-docker-in-dsw),
[DSW datasets and OSS mounts](https://help.aliyun.com/zh/pai/developer-reference/api-pai-dsw-2022-01-01-createinstance),
[shutdown timer](https://help.aliyun.com/zh/pai/developer-reference/api-pai-dsw-2022-01-01-createinstanceshutdowntimer),
[DLC CreateJob](https://help.aliyun.com/zh/pai/developer-reference/api-pai-dlc-2020-12-03-createjob).

### Personal role and approved OSS prefixes

Add `data_role_arn: "acs:ram::ACCOUNT_ID:role/pai-data-YOUR_RAM_USER"` alongside
`workspace_user`. The role must exactly match that user. The builder injects it
through native DSW `CredentialConfig`, separately from storage mount permissions.
It never gives registry credentials to another user or requests PUBLIC visibility.

To request native OSS mounts with that role, set:

```json
{
  "workspace_user": "YOUR_RAM_USER",
  "data_role_arn": "acs:ram::ACCOUNT_ID:role/pai-data-YOUR_RAM_USER",
  "oss_uri": "oss://BUCKET.oss-REGION-internal.aliyuncs.com/users/YOUR_RAM_USER/",
  "oss_mount_path": "/mnt/oss"
}
```

This mode requests personal `/mnt/oss` RW, `/mnt/oss-models` from
`public/models/` RO, and `/mnt/oss-datasets` from `public/datasets/` RO. Each native
mount carries the same personal `RoleArn`. No project path is requested. Omit
both OSS fields if provider mount initialization fails; there is no silent
fallback to an unscoped default role. SDK access using the instance role and
server-side result archival are independent of native OSS filesystem mounts.

Instance status alone may omit startup errors. The builder now saves sanitized
`GetLifecycle` failure reasons before deleting its owned instance, and exposes
them through the existing Runtime build diagnostics/logs.

### Private result archival

An optional `oss_archive` block in the private artifact-store configuration enables
archival of **newly sealed** result archives:

```json
{
  "oss_archive": {
    "bucket": "ai-oss-h200-1020",
    "region": "us-southeast-1",
    "user": "liangluocheng",
    "credentials_file": "/etc/ml-expd/oss-archive-credentials.json"
  }
}
```

The separate private credential file contains `workspace_user`, `region`,
`access_key_id`, `access_key_secret` and optional `security_token`. The service
user must be able to read it; keep the original builder-only credentials private.
Server archival uses the authenticated personal RAM user and its existing scoped
OSS permissions. Inside a DSW, native runtime credentials use the personal PAI
role instead. The server does not pretend to assume a service-only role.

The destination is immutable:
`users/USER/results/ml-expd/PROJECT/RUN/ATTEMPT/ARCHIVE_SHA256.tar`.
A bounded collection worker streams from Garage to OSS using the OSS SDK's
multipart uploader (8 MiB parts, one at a time), then downloads the object stream
to verify SHA256 and size. No second full local archive is created. The source
stream verifies the platform SHA256 before completing publication; its S3 client
uses required transport checks only because Garage's optional response-checksum
headers have failed real streaming verification. OSS transfer checks remain on.
An existing exact object is verified and reused after a lost response. A
conflicting object is never overwritten. Failures remain visible and retry after
60 seconds; uploading does not restart training.

`GET /api/runs/PROJECT/RUN/attempts/ATTEMPT/collection` includes `oss_archive`
state, progress and sanitized diagnostics. `VERIFIED` requires actual object-byte
proof, not metadata or ETag. An authenticated
`GET /api/runs/PROJECT/RUN/attempts/ATTEMPT/oss-archive/download` returns a private
five-minute signed download URL and the original file/hash receipt. It works
after the Garage copy is released. Download and validate the SHA256/file receipt
as usual; an OSS download does not delete the durable object or require an ACK.

A real client ACK remains recorded immediately. For a newly archive-enabled
receipt, ACK-triggered Garage deletion waits for OSS verification. The existing
24-hour local expiry policy is unchanged: an archive still failing when its
source expires reports `SOURCE_EXPIRED`, never a fabricated success. Monitor
archival failures before that deadline. Historical receipts are not silently
backfilled. Repairs retain the previous archive identity and create a new target.

The personal 4 TB budget is **soft**; neither this worker nor OSS automatically
rejects excess usage. This is result archival, not periodic backup of the entire
training state or an automatic synchronization of datasets/Git/environments.
Persistent recovery checkpoints remain on the backend filesystem unless included
in selected exported outputs. OSS copies are deliberate durable backups and are
outside automatic local retention, download ACK release, and Run deletion, as are
existing external W&B records; remove them separately under the personal prefix
when desired. No Bucket ACL/Policy or lifecycle deletion is changed.
