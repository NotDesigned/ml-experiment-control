# Alibaba PAI: DLC jobs and a CPU DSW builder

PAI scheduling and image construction are separate adapters. DLC runs the
experiment; an optional Docker-enabled DSW builds its OCI image. The normal
source → Runtime → Run → Submission → Attempt workflow is unchanged. Configuring
one does not implicitly enable the other or allocate public compute.

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
    storage_root: /data/ml-expd
    backend:
      kind: pai-dlc
      region: REGION
      workspace_id: WORKSPACE_ID
      quota_id: QUOTA_ID
      gpu_type: H200
      time: '00:10:00'
```

Replace capacity with verified per-node limits. The server sets image, job name
and walltime when freezing a Run. `ResourceId` is mandatory; no public/pay-as-you-go
fallback is attempted. Clients may name the executor or select backend `pai-dlc`.

`GET /api/executors/{id}/status` returns the quota's `ActualMinQuota` and
`AllocatedQuota`, converted to `gpus`, `cpus` and `memory_bytes`. The query verifies
region and workspace binding. Its 25-second cache uses the existing
OBSERVED/STALE/UNKNOWN semantics. Arithmetic unallocated quota is neither GPU
utilization nor an admission guarantee; pending queue count and ETA are unknown.

### Data and recovery limits

The first adapter uses operator-configured native versioned PAI Dataset mounts:

```yaml
      data_sources:
        - DataSourceId: DATASET_ID
          DataSourceVersion: v1
          MountPath: /data
          MountAccess: RW
      filesystem_type: nas
      storage_id: IMMUTABLE_FILESYSTEM_ID
```

Only an explicit RW NAS/CPFS dataset covering `storage_root` enables persistent
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
[shutdown timer](https://help.aliyun.com/zh/pai/developer-reference/api-pai-dsw-2022-01-01-createinstanceshutdowntimer),
[DLC CreateJob](https://help.aliyun.com/zh/pai/developer-reference/api-pai-dlc-2020-12-03-createjob).
