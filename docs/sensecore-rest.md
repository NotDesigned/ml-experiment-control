# SenseCore REST and ACP

GPU training and zero-GPU data-image copy use signed HTTPS REST requests.
No SCO executable or fallback is used. Client Bearer auth, REST AK/SK and
registry credentials are independent.

## Private setup and request scope

Provision a daemon/root-readable private JSON, normally
`/etc/ml-expd/sensecore-rest.json`:

```json
{
  "access_key_id": "YOUR_AK",
  "access_key_secret": "YOUR_SK",
  "subscription_name": "YOUR_EXISTING_SUBSCRIPTION",
  "resource_group_name": "default"
}
```

`EXPERIMENTCTL_SENSECORE_REST_CONFIG` overrides its path. Do not include it in
Git, images, executor definitions or Runs. Configuration is HMAC-SHA256 signed
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
launcher; a server upgrade cannot change an immutable old image.

## Timeouts and failure evidence

| Limit | Source/scope |
|---|---|
| Worker duration | Run `resources.max_time`, default `00:10:00`; GNU TERM, then KILL after 30s |
| Startup manifest | 30-second launcher fetch before worker timer |
| Data script | `data_preparation.timeout_seconds`, default 1200; inside worker duration |
| REST | Generic 30s; create 120s, configurable 10–600 via `EXPERIMENTCTL_SENSECORE_CREATE_TIMEOUT_SECONDS` |
| Controller/build | Separate host operation budgets, not training duration; see [builds](builds.md) |

Worker duration covers restore, data preparation, training and result return.
It excludes cloud queue/image pull/NAS setup. ACP create currently contains no
matching cloud wall-time field; WYD also applies the frozen duration to Slurm.
Each Attempt gets its own worker timer, not a cumulative retry budget.

Terminal jobs use offline logs; unavailable live logs can fall back to historical
queries. Queries bind exact worker/container and time scope, then redact text
with `experiment-redact`. Historical logs are not live heartbeats. SUSPENDED
normalizes to CANCELLED with unknown cause unless an exact cancellation marker
exists; neither FAILED nor SUSPENDING alone proves preemption. Empty logs or an
HTTP error do not establish an exit code. The offline page-size defect is listed
in [known limitations](recovery.md#known-limitations).

ACP requests currently set `backoff_limit: 0`. There is no automatic restart;
the collector only observes. See [recovery](recovery.md) for manual restoration
and the explicitly unimplemented automatic-recovery design.
