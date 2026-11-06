# SenseCore REST and ACP

GPU training and zero-GPU data-image copy use signed HTTPS REST requests.
No SCO executable or fallback is used. Client Bearer auth, REST AK/SK and
registry credentials are independent.

## Private setup and request scope

Copy [the SenseCore example](../rust/deploy/sensecore.example.json) into the file named
by `cloud_config` in the [deployment file](../rust/README.md#deploying-a-native-service).
Its `credentials_file` is a private file owned by the service account:

```json
{"provider":"sensecore","access_key_id":"YOUR_AK","access_key_secret":"YOUR_SK"}
```

Subscription, resource group, region, zones and workspace are in `profile.scope`. Do
not include credentials in Git, images, configuration or Runs. Configuration is HMAC-SHA256 signed
over `X-Date`. Requests verify TLS, reject redirects/non-SenseCore endpoints
and bypass ambient proxies. Change keys deliberately and restart the service
afterward. Errors expose bounded status/reason, not account payloads or keys.

| Operation | Service |
|---|---|
| Account identity | IAM `/iam/idp/v1/me` |
| Workspace/NAS/log-station discovery | RM `/rmh/v1/resources` |
| Pool bindings and specifications | Workspace `workspaceAEC2Bindings`, AEC2 `resourceSpecs` |
| Create/read/reconcile/stop | ACP `trainingJobs`, `trainingJobs:batchStop` |
| Worker status | ACP `trainingJobs/J/workers` |
| Active logs | Monitor signed livestream token/poll |
| Historical logs | Monitor `logStream/products/product.lepton-acp-new/logs` |

Workspace and pool zones are resolved separately. Jobs retain exact account,
workspace, pool, pinned image, NAS subdirectory and submission-token-qualified
resource name. Complete pagination/owner checks are required before declaring
absence. A write timeout or invalid acknowledgement requires exact reconciliation;
it never authorizes a second create.

## Cluster choice

Reported allowance is not a promise of immediate allocation.

New GPU Runs default to `pool_selection: highest_spot`. Selection fetches ACTIVE
workspace bindings, excludes debug pools and requires the same zone, known VPC
and compatible GPU specification. Optional `allowed_clusters` restricts scope.
Rank by summed `spot_status[].spot_quota.device`, breaking ties by cluster name.
This is **SPOT allowance, not free cards or an estimated start time**. Invalid
allowance is excluded; query/permission failures block selection. Freeze the
chosen pool and timestamped evidence in the new Run; subsequent operations use
that placement. `pool_selection: fixed` supports explicit non-debug placement.

Data delivery is separate: a zero-GPU ACP in the configured debug pool uses
RESERVED quota and a verified 2CPU/4GiB/0GPU spec to copy a data image into NAS.
There is no CCI dependency in this path. GPU training rejects debug placement.
See [data delivery](data.md).

## Container startup

ACP pulls the pinned `roles[].image_path` and mounts NAS before executing its
`startup_script` inside the container. Source is already in `/workspace`.
New images with `launcher-manifest.v1` use a short entrypoint:

```bash
env ML_EXPD_BOOTSTRAP_TOKEN=<exact-attempt-capability> \
  ml-exp-worker --manifest-url https://API/api/launch-transfers/PROJECT/RUN/ATTEMPT/SHA256
```

The sealed canonical JSON carries exact identity, workdir, scientific argv/env,
inputs, restore/state manifests, transfer endpoints and worker duration. The
capability is separate from JSON and cannot control jobs or read other Attempts.
The launcher fetches at most 256 KiB over HTTPS within 30 seconds, verifies
canonical hash/schema/identity, removes its bootstrap token and starts the
managed worker. The worker removes its upload capability before user code.

Inside the worker: verify/restore checkpoint → prepare/verify data → start
training → register complete state generations → archive/return declared outputs.
Args are passed as an array, without `eval`. The wrapper creates OUTPUT_DIR,
selects workdir and forwards termination to its owned process group.

Historical images keep a longer startup command with quoted environment layers
and inline JSON for input/restore/state/output manifests, followed by GNU
`timeout`, `worker.py` and the frozen training argv. Those layers transport hashes
and configuration; they do not build the image. Rebuild to obtain the new
launcher; a server upgrade cannot change an immutable image.

## Timeouts and failure evidence

| Limit | Source/scope |
|---|---|
| Worker duration | Run `resources.max_time`, default `00:10:00`; GNU TERM, then KILL after 30s |
| Startup manifest | 30-second launcher fetch before worker timer |
| Data script | `data_preparation.timeout_seconds`, default 1200; separate CPU preparation when declared `before_job`, otherwise inside worker duration |
| REST | Backend `timeout_seconds`, default 30, range 1–120 |
| Controller/build | Separate host operation budgets, not training duration; see [builds](builds.md) |

Worker duration covers restore, inside-job data preparation, training and result
return. Before-job debug CPU preparation has its own bounded operation budget.
It excludes cloud queue/image pull/NAS setup. ACP create currently contains no
matching cloud wall-time field; Slurm also applies the frozen duration to Slurm.
Each Attempt gets its own worker timer, not a cumulative retry budget.

Terminal jobs use offline logs; unavailable live logs can fall back to historical
queries. Queries bind exact worker/container and time scope, then redact text
with `experiment-redact`. Historical logs are not live heartbeats. SUSPENDED
normalizes to CANCELLED with unknown cause unless an exact cancellation marker
exists; neither FAILED nor SUSPENDING alone proves preemption. Empty logs or an
HTTP error do not establish an exit code. Offline pages are capped at the
provider's 200-record limit; larger requested tails report truncation.

ACP requests currently set `backoff_limit: 0`. There is no automatic restart;
training restart remains unimplemented. Independent result collection can launch
a bounded zero-GPU CPU job; it does not restart training. See
[recovery](recovery.md) for collection and manual checkpoint restoration.

## Dataset publication and CPU copy

SenseCore needs a `data_copier` profile on the backend, a zero-GPU profile in a debug
CPU pool, and `builds.data_base_image` in the cloud configuration. Both are in
[the example](../rust/deploy/sensecore.example.json).

Publish an ACP-compatible linux/amd64 glibc/Bash base first; `data_base_image` must be
its immutable digest. The image builder publishes the data image from a CPU PAI DSW
and copies it to CCR by digest through `sensecore_mirror`; see [builds](builds.md).
Include only the required CCR host entry in its registry auth and do not copy unrelated
registry logins. `registry_blob_hosts` contains only the actual CCR blob redirect
hostnames verified by the operator, without scheme, paths or wildcards; redirects use
TLS and receive no registry credentials.
Workspace, storage mount and dataset root are derived from the chosen executor.
Verify the CPU spec is exactly 2 CPUs, 4 GiB and 0 GPUs in the selected debug pool.
Dataset uploads stream directly to backend storage. Runtime builds use
the separately configured builder, which can be local Docker or PAI DSW;
see [builds](builds.md).
