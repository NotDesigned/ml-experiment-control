# Alibaba PAI: DLC training and DSW debugging/builds

DLC runs experiments. DSW supplies optional temporary debug machines and CPU
image builders. These are independent configurations: enabling a DLC executor
does not enable DSW, allocate public compute, or create storage or IAM roles.

## DLC executor

Use [executors/pai-dlc.yaml](../server/examples/executors/pai-dlc.yaml) as one
entry in the operator's executor catalogue. Set its region, workspace, quota,
verified single-node capacity, GPU type and approved persistent storage. The
adapter supports one worker with 1–8 GPUs. CPU and memory are requested per Run
within that capacity; provider role/quota limits also apply. It does not
implement multi-node distributed training.

The server includes the signed OpenAPI dependencies. Core-library-only users
need its `pai` extra. Store credentials using
[pai.json](../server/examples/pai.json), selected by `EXPERIMENTCTL_PAI_CONFIG`:

| Setting | Purpose |
| --- | --- |
| `region`, `workspace_id` | Exact approved PAI scope |
| `access_key_id`, `access_key_secret` | Private host credentials |
| `security_token` | Optional temporary-credential token |
| `registry_auth_file` | Exact-host Docker pull credentials |

Keep these outside source, experiment JSON and Git. The native CreateJob request
receives credentials for the selected image registry; frozen Runs do not contain
pull passwords. Runtime-role credentials are separate from host API credentials.

`ResourceId` is mandatory: there is no silent public or pay-as-you-go fallback.
On quotas that reject `ResourceConfig.GPUType`, use
`gpu_type_from_quota: true`. Preflight verifies the quota's reported GPU type
against the executor and then omits the per-pod GPU selector. The fixed quota and
requested GPU count remain explicit.

### Duration and scheduling

New DLC Runs default to `resources.max_time: "24:00:00"`; a shorter explicit
value is preserved. Values above 24 hours are rejected before image construction
or submission. The provider receives `JobMaxRunningTimeMinutes` and the worker
receives the same effective timeout. Preparation budgets must cover GPU count
multiplied by duration: 8 GPUs for the default day require 192 GPU-hours.
Existing frozen Runs/preparations keep their original duration.

For longer training, save complete recovery state on CPFS and explicitly submit
a subsequent Run with `resume_from`. Provider restart and paid Spot are disabled;
there is no automatic continuation or unbounded cumulative budget. Client wait
timeouts do not terminate the remote job.

## DLC scheduling options

New Runs default to priority **5** and guaranteed quota only. A Run can choose:

```json
{"resources":{"gpus":8,"cpus":64,"memory_gb":512,"max_time":"08:00:00",
              "allocation_policy":"allow_idle","priority":5}}
```

| `allocation_policy` | `Settings.EnableTideResource` | `Settings.OversoldType` and `JobSpecs[].OversoldType` |
| --- | --- | --- |
| `guaranteed` (default) | false | `ForbiddenQuotaOverSold` |
| `allow_idle` | true | `AcceptQuotaOverSold` |
| `idle_only` | true | `ForceQuotaOverSold` |

Both job-level permission and policy are set. A node policy alone does not enable
idle resources. `UseOversoldResource` reports actual allocation, not permission.
Priority is an integer from 1 to 9, with larger values scheduled at higher
priority. The catalogue advertises these options; clients check
`pai-idle-resources.v1` before sending them. Other backends reject these options.

Each Run freezes its choice. Old Runs lacking the permission marker retain their
native fields when read or retried; create a new Run to use the corrected policy.
Idle permission does not bypass role limits, increase the per-job GPU ceiling,
guarantee admission, or enable Spot. Borrowed resources can be reclaimed.

### Resource and queue evidence

`GET /api/executors/{id}/status` returns the bound quota's `ActualMinQuota` and
`AllocatedQuota` as GPU, CPU and memory-byte totals. It checks region/workspace
binding and uses the shared 25-second OBSERVED/STALE/UNKNOWN cache. Arithmetic
unallocated quota is not physical GPU utilization or an admission guarantee.

For queued jobs the adapter also queries `ListQuotaWorkloads` for the exact
workspace, quota and job. Submission progress returns bounded, redacted evidence
under `queue`; Run evidence exposes it in `evidence.scheduler.detail.queue`:

```json
{
  "code": "roleMaximumResource",
  "provider_code": "CheckSelfQuotaFailed",
  "code_type": "ConfigRule",
  "reason": "Current Limit is GPU: 8 limited by Role PAI.AlgoDeveloper.",
  "blocker": {
    "CodeType": "ConfigRule",
    "Code": "roleMaximumResource",
    "Reason": "Current Limit is GPU: 8 limited by Role PAI.AlgoDeveloper."
  },
  "position": 1,
  "availability": "OBSERVED"
}
```

Missing workload evidence is `NOT_OBSERVED`; query errors are `UNAVAILABLE` with
`query_error`. They do not replace the known scheduler state or authorize a new
submission. Queue position is not an ETA. Status and current/historical pod logs
are supported; exit/preemption reasons remain unknown when the provider does not
supply reliable evidence. A lost submission response is reconciled using the
original exact identity, with no automatic create retry.

### Approved CPFS personal workspace

Configure these backend fields after the administrator approves the personal
path, role and endpoint:

```yaml
cpfs_uri: bmcpfs://cpfs-FILESYSTEM-TARGET.REGION.cpfs.aliyuncs.com/
workspace_user: YOUR_RAM_USER
data_role_arn: acs:ram::ACCOUNT_ID:role/pai-data-YOUR_RAM_USER
filesystem_type: cpfs
storage_id: IMMUTABLE_CPFS_FILESYSTEM_ID
```

| Container path | CPFS directory | Access |
| --- | --- | --- |
| `/mnt/workspace` | `/users/YOUR_RAM_USER/` | RW |
| `/mnt/models` | `/public/models/` | RO |
| `/mnt/datasets` | `/public/datasets/` | RO |

The platform mounts no root, other user's directory or project directory. Set
matching `user_vpc` values (`VpcId`, `VSwitchId`, `SecurityGroupId`) when required
by the mount target; verify filesystem/quota affinity rather than guessing an
endpoint. Direct personal storage identities bind the endpoint and approved
user, so identical container paths do not make another user's state reusable.

DLC injects the approved personal role through `EnableCredentialInject` and
`CredentialConfigItems`. Existing versioned PAI Dataset mounts can instead be
configured through `data_sources`; an explicit RW NAS/CPFS mount covering
`storage_root` is required for persistent checkpoints and in-job data scripts.
Additional mounts must not shadow the personal/public paths above. A declared
storage ID is configuration, not proof that the remote mount works.

### Data and recovery limits

Configure [OSS dataset assets](data.md#personal-oss-dataset-originals-and-dlc-caches)
for versioned originals, upload/import and signed downloads. DLC first delivery
uses the exact-Attempt API transfer into a verified personal CPFS cache. This
happens inside the allocated job and consumes its time/GPU budget; subsequent
jobs can reuse verified cache bytes. An alternative is an experiment-authored
download script writing to persistent `DATA_DIR` before training. Successful
image pulls do not prove general Internet connectivity for that script.

Native OSS filesystem mounts remain **blocked and disabled** because provider
mount authorization fails before user code starts. Do not configure `oss_uri`
or OSS `Datasets` mounts as a workaround, change the role, or make the bucket
public. CPFS mounts and OSS SDK/object transfers are independent and remain the
supported paths. No native-mount acceptance is claimed.

PAI does not implement periodic full checkpoint-upload snapshots, pre-job CPU
data preparation, offline CPU result recovery or managed remote Run deletion.
It does support backend-resident checkpoint registration/restoration and normal
worker metric/result transfer. Unsupported requirements are rejected before
work; a missing result upload does not silently launch a CPU recovery job.

## DSW builder without enterprise ACR

The optional builder starts a private 4-CPU/8-GiB/zero-GPU DSW, builds a sealed
source context and publishes to the configured OCI registry. It does not call
PAI's enterprise-ACR image-build API. See [DSW and ACR builder setup](builds.md#pai-dsw-and-acr-fallback)
for configuration, registry authentication, timeouts and reconciliation.

DSW's hard cap is **4 hours**. Debug leases can be shorter; the recommended build
configuration explicitly sets **20 minutes**, with a verified cloud shutdown
budget of build timeout plus 20 minutes, capped at four hours. ACR fallback
publication observation is separately bounded at **45 minutes**. These values
are different from the DLC job duration. DSW is not used for training.

For a CPFS-backed DSW, the personal dataset is registered at
`/mnt/ml-expd-personal` and selected as `WorkspaceSource`; DSW also binds that
same personal directory at `/mnt/workspace`. Registering it directly at the
latter path can create duplicate mounts. Public mounts stay read-only. Registry
secrets and build controls use the owned temporary directory; a persistent CPFS
workspace is not a persistent Docker layer cache.

### Tokyo registry and native build fallback

A shared private registry can serve all backends. One supported configuration is
DSW → Tokyo ACR, with Git-triggered ACR construction as fallback, PAI/Slurm direct
pulls, and digest-preserving on-demand CCR delivery for SenseCore. These are
operator choices, not requirements for an unrelated Slurm installation.
Follow [build publication and fallback](builds.md#pai-dsw-and-acr-fallback);
existing Runtime/Run image references are never rewritten when defaults change.

## Private result archival

An optional `oss_archive` block in the private artifact-store configuration
archives newly sealed result objects:

```json
{
  "oss_archive": {
    "bucket": "YOUR_APPROVED_ARCHIVE_BUCKET",
    "region": "YOUR_REGION",
    "user": "YOUR_RAM_USER",
    "credentials_file": "/etc/ml-expd/oss-archive-credentials.json"
  }
}
```

The private credential file contains `workspace_user`, `region`,
`access_key_id`, `access_key_secret` and optional `security_token`. Server SDK
access uses the approved personal RAM user's existing permissions; it does not
assume the DSW service role. Keep builder-only credentials separate.

The immutable destination is
`users/USER/results/ml-expd/PROJECT/RUN/ATTEMPT/ARCHIVE_SHA256.tar`. Collection
streams into OSS multipart storage and checks actual object SHA256/size without
creating another full local archive. Lost responses reuse the exact verified
object; conflicts are never overwritten. Failures remain diagnostic and retry
after 60 seconds, without restarting training.

The Attempt's `/collection` response includes `oss_archive` status and progress.
Authenticated `/api/runs/PROJECT/RUN/attempts/ATTEMPT/oss-archive/download` returns
a five-minute signed URL and the file/hash receipt. It remains available after
the temporary platform archive is released. Downloading does not delete OSS.

For new archive-enabled receipts, an ACK is recorded immediately but temporary
archive release waits for OSS verification. An operator's local expiry deadline
still applies: failed archival can become `SOURCE_EXPIRED`. Historical receipts
are not automatically backfilled. OSS copies are outside client-ACK release,
local expiry and managed Run deletion. Delete them separately under the approved
personal prefix. A personal storage budget is advisory unless independently
enforced; this integration does not change bucket ACL/policy or lifecycle rules.
It does not synchronize whole checkpoints, datasets, Git trees or environments.

For optional SSH machines see [debug sessions](debug-sessions.md); for storage
transfer configuration see [data](data.md). Provider reference:
[DLC CreateJob](https://help.aliyun.com/zh/pai/developer-reference/api-pai-dlc-2020-12-03-createjob).
