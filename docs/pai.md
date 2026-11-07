# Alibaba PAI: DLC training and DSW debugging/builds

DLC runs experiments. DSW supplies optional temporary debug machines and CPU
image builders. These are independent configurations: enabling a DLC executor
does not enable DSW, allocate public compute, or create storage or IAM roles.

## DLC executor

Copy one entry of `backends` from [the PAI example](../rust/deploy/pai.example.json)
into the file named by `cloud_config` in the
[deployment file](../rust/README.md#deploying-a-native-service). Set its `executor`,
and in `profile` its `region`, `workspace_id`, `quota_id`, verified capacity
(`max_gpus`, `max_cpus`, `max_memory_gb`), `gpu_type` and approved `storage`, plus
`run_root` and `data_root`. The adapter supports one worker with 1–8 GPUs. CPU and
memory are requested per Run within that capacity; provider role/quota limits also
apply. It does not implement multi-node distributed training.

Store credentials in the file named by `credentials_file`, with the shape shown in
the [deployment guide](../rust/README.md#deploying-a-native-service). The image
builder reads its own PAI file, selected by `ML_EXP_PAI_CONFIG`.

| Setting | Purpose |
| --- | --- |
| `access_key_id`, `access_key_secret` | Private host credentials |
| `security_token` | Optional temporary-credential token |
| `registry_auth_file` | Backend setting: exact-host Docker pull credentials |

Keep these outside source, experiment JSON and Git. The native CreateJob request
receives credentials for the selected image registry; frozen Runs do not contain
pull passwords. Runtime-role credentials are separate from host API credentials.

`ResourceId` is mandatory: there is no silent public or pay-as-you-go fallback.
On quotas that reject `ResourceConfig.GPUType`, omit `gpu_type` from the profile,
which also omits the per-pod GPU selector. The fixed quota and
requested GPU count remain explicit.

### Duration and scheduling

New DLC Runs default to `resources.max_time: "24:00:00"`; a shorter explicit
value is preserved. Values above 24 hours are rejected before image construction
or submission. The provider receives `JobMaxRunningTimeMinutes` and the worker
receives the same effective timeout. Preparation budgets must cover GPU count
multiplied by duration: 8 GPUs for the default day require 192 GPU-hours.
A frozen Run or preparation keeps its duration.

For longer training, save complete recovery state on CPFS and explicitly submit
a subsequent Run with `resume_from`. Provider restart and paid Spot are disabled, and
[idle resources can be reclaimed earlier](#dlc-scheduling-options);
there is no automatic continuation or unbounded cumulative budget. Client wait
timeouts do not terminate the remote job.

## DLC scheduling options

A job with GPUs from `ml-exp submit` defaults to `allow_idle` and priority **5**. The
question before it is sent names the allocation and says when the job may be preempted.
Choose another with `--allocation`, or in the file `submit --file` reads:

```bash
ml-exp submit . --gpus 8 --time 8h --allocation guaranteed -- python train.py
```

```json
{"run":{"scheduling":{"allocation":"guaranteed","priority":5}}}
```

| `allocation` | `Settings.EnableTideResource` | `Settings.OversoldType` and `JobSpecs[].OversoldType` |
| --- | --- | --- |
| `guaranteed` | false | `ForbiddenQuotaOverSold` |
| `allow_idle` | true | `AcceptQuotaOverSold` |
| `idle_only` | true | `ForceQuotaOverSold` |

Both job-level permission and policy are set. A node policy alone does not enable
idle resources. `UseOversoldResource` reports actual allocation, not permission.
Priority is an integer from 1 to 9, with larger values scheduled at higher
priority. The catalogue advertises these options. SenseCore accepts only `guaranteed`,
and Slurm refuses `scheduling` altogether.

**A job on idle resources can be preempted when they are reclaimed.** Nothing restarts
it: provider restart is off, and running again is your explicit `ml-exp retry`. A long
Run should write checkpoints to `STATE_DIR` often enough to lose little. PAI reports no
preempted state: such a job reads as `FAILED` or `CANCELLED`, the same as a failure or
a cancellation, and this service cannot tell them apart. `ml-exp submit` says so when
a job that could use idle resources ends that way. The provider's own reason may or may
not say more.

`max_gpu_hours` is unchanged by the allocation. It bounds GPU count times duration for
one Attempt, whichever resources it runs on, and it is not a bill. A retry is a new
Attempt that asks and spends again.

Each Run freezes its choice, in the Run itself, and a retry keeps it; create a new Run
to change it. With `submit`, a different `--allocation` is a different job. Leaving the
option out keeps the identity a repeat of an older command had, so:

- A Run, Attempt or preparation stored with no `scheduling` keeps meaning what it meant,
  guaranteed, and a retry of it stays guaranteed. Repeating the command that saved such
  a preparation finds it and shows `guaranteed`.
- The default applies when `ml-exp submit` creates a new preparation, which then holds
  `allow_idle` explicitly. Changing the default later changes no existing Run.

Only `ml-exp submit` applies this default, because only it asks first. A job with no
GPUs, such as on a CPU executor, stays guaranteed, and so do the platform's own copy,
collection, sync, script-preparation and diagnostic jobs. `run start`, `run create`,
`run batch`, the HTTP API and clients older than this release freeze what they are
given, and without `scheduling` that is guaranteed.

The backend's `allocation` setting is what a Run with no `scheduling` asks for, stored
ones included. Leave it at `guaranteed`: changing it changes what old Runs and their
retries ask for, and an Action already prepared under the old value then refuses to
run. It does not set the default for new Runs.

Idle permission does not bypass role limits, increase the per-job GPU ceiling,
guarantee admission, or enable Spot.

### Resource and queue evidence

`GET /api/backends/{executor}/capacity` returns the bound quota's `ActualMinQuota` and
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

Configure this block in the backend's `profile` after the administrator approves the
personal path, role and endpoint:

```json
{
  "storage": {
    "cpfs_uri": "bmcpfs://cpfs-FILESYSTEM-TARGET.REGION.cpfs.aliyuncs.com/",
    "workspace_user": "YOUR_RAM_USER",
    "role_arn": "acs:ram::ACCOUNT_ID:role/pai-data-YOUR_RAM_USER"
  }
}
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
`CredentialConfigItems`. A job's mounts are exactly the three above: the profile has
no other mount setting and refuses unknown fields. A declared `storage` is
configuration, not proof that the remote mount works.

### Data and recovery limits

Configure [OSS dataset assets](data.md#pai-preparation-using-an-existing-image)
for versioned originals, [direct upload](data.md#direct-upload), import and signed
downloads. DLC first delivery
uses the exact-Attempt API transfer into a verified personal CPFS cache. This
happens inside the allocated job and consumes its time/GPU budget, unless you
publish the asset first with a CPU job; subsequent jobs can reuse verified cache
bytes. An alternative is an experiment-authored
download script writing to persistent `DATA_DIR` before training. Successful
image pulls do not prove general Internet connectivity for that script.

Native OSS filesystem mounts remain **blocked and disabled** because provider
mount authorization fails before user code starts. Neither the backend profile nor
the DSW builder has a setting for one. Do not add OSS `Datasets` mounts by hand as a
workaround, change the role, or make the bucket public. CPFS mounts and OSS object
transfers are independent and remain the supported paths. No native-mount acceptance
is claimed.

PAI does not upload periodic full checkpoints. It supports backend-resident checkpoint
registration/restoration and normal worker metric/result transfer. With a `collector`
configured, three zero-GPU CPU jobs run on explicit request: `data publish` prepares a
dataset before training ([data](data.md#pai-preparation-using-an-existing-image)),
`run sync` recovers results after training ([recovery](recovery.md#native-cpu-sync-recovery))
and `admin execute` removes an Attempt's bytes ([deletion](resource-deletion.md)). A
missing result upload never starts a CPU job by itself.

## DSW builder without enterprise ACR

The optional builder starts a private 4-CPU/8-GiB/zero-GPU DSW, builds a sealed
source context and publishes to the configured OCI registry. It does not call
PAI's enterprise-ACR image-build API. See [DSW builder setup](builds.md#pai-dsw-publisher)
for configuration, registry authentication, timeouts and reconciliation.

DSW's hard cap is **4 hours**. Debug leases can be shorter; the recommended build
configuration explicitly sets **20 minutes**, with a verified cloud shutdown
budget of build timeout plus 20 minutes, capped at four hours. These values
are different from the DLC job duration. DSW is not used for training.

For a CPFS-backed DSW, the personal dataset is registered at
`/mnt/ml-expd-personal` and selected as `WorkspaceSource`; DSW also binds that
same personal directory at `/mnt/workspace`. Registering it directly at the
latter path can create duplicate mounts. Public mounts stay read-only. Registry
secrets and build controls use the owned temporary directory; a persistent CPFS
workspace is not a persistent Docker layer cache.

### Shared registry

A shared private registry can serve all backends. The default publisher pushes
from a CPU DSW to the configured registry, such as Docker Hub. PAI/Slurm pull directly;
SenseCore uses digest-preserving on-demand CCR delivery. These are operator
choices, not requirements for an unrelated Slurm installation.
Follow [build publication](builds.md#pai-dsw-publisher);
existing Runtime/Run image references are never rewritten when defaults change.

## Result objects

A result archive is uploaded in parts below the `prefix` of the backend's `artifacts`
block, at `PREFIX/results/PROJECT/RUN/ATTEMPT/ARCHIVE_SHA256/parts/NUMBER`. A
release-enabled ACK deletes those parts and `--keep` leaves them. There is no second
long-term copy, and no timer expires them. After a release, what remains is your
download and the original `OUTPUT_DIR` on CPFS. See [recovery](recovery.md#collect-and-download).

For optional SSH machines see [debug sessions](debug-sessions.md); for storage
transfer configuration see [data](data.md). Provider reference:
[DLC CreateJob](https://help.aliyun.com/zh/pai/developer-reference/api-pai-dlc-2020-12-03-createjob).
