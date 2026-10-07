# SenseCore REST backend

Both GPU training and CPU-only data-image delivery use signed HTTPS requests.
SCO is not installed or invoked by this adapter. Configuration follows the
HMAC `X-Date` flow used by SLAI-sensecore-tool, with a daemon-owned client rather
than importing its UI or account configuration.

Create `/etc/ml-expd/sensecore-rest.json`, readable only by the daemon and root:

```json
{
  "access_key_id": "YOUR_AK",
  "access_key_secret": "YOUR_SK",
  "subscription_name": "YOUR_EXISTING_SUBSCRIPTION",
  "resource_group_name": "default"
}
```

`EXPERIMENTCTL_SENSECORE_REST_CONFIG` overrides the private file location.
Do not put this file or its values in an executor, Run manifest, image,
client configuration or Git. Restart the daemon after an intentional key change.
The API/client Bearer token and registry credentials are independent.

| Operation | REST service and path |
| --- | --- |
| Authentication | IAM `GET /iam/idp/v1/me` |
| Workspace/storage/log station discovery | RM `GET /rmh/v1/resources` |
| Pool bindings | Workspace `GET …/workspaceAEC2Bindings` |
| Resource specifications | AEC2 `GET …/aec2s/P/resourceSpecs` |
| Exact job identity/reconciliation | ACP `GET …/trainingJobs?filter=name=…` and `GET …/trainingJobs/J` |
| Submit | ACP `POST …/trainingJobs?training_job_name=J` |
| Stop | ACP `POST …/trainingJobs:batchStop`, containing only the exact job name |
| Workers | ACP `GET …/trainingJobs/J/workers` |
| Live logs | Monitor log-livestream token and bounded HTTPS log polling |
| Offline logs | Monitor `POST …/logStream/products/product.lepton-acp-new/logs` |

The workspace zone and pool zone are resolved separately. Jobs retain their
existing workspace, pool, immutable image digest, AFS volume/subdirectory and
submission-token-qualified name. Every observed exact job must belong to the
configured account. Complete pagination is required before absence can be
reported. Foreign owners, malformed responses and scope drift fail closed.

New GPU ACP requests reject debug clusters. CPU-only data-image delivery keeps
the configured debug pool, reserved quota and verified 2CPU/4GiB/0GPU specification.
The shared create translator accepts debug only through the explicit `cpu_copy`
path and rejects GPU/resource/spec mismatches. Existing frozen records remain
readable and are not rewritten or restarted.

New GPU Runs default to `pool_selection: highest_spot`; profiles can set an
optional `allowed_clusters` list. Creating a new Run fetches fresh ACTIVE workspace
bindings, excludes debug pools, and requires the same pool zone, known VPC and
requested GPU worker specification. Candidates are ranked by the sum of
`spot_status[].spot_quota.device`, with cluster name breaking ties. This field
reports SPOT allowance, not immediately free GPUs or a start-time guarantee.
Missing/malformed allowance is excluded; missing permission fails the selection.
The chosen pool and timestamped candidate evidence are frozen in that Run.
Repeating the same Run request reuses this choice without another selection;
changing a definition conflicts. Describe, logs, cancel and reconciliation
continue to use the frozen pool. Explicit `pool_selection: fixed` non-debug
profiles and existing historical fixed Runs remain supported.

Requests verify TLS, bypass ambient proxy settings, reject redirects and reject
endpoints outside SenseCore domains. There is no write retry or CLI fallback.
A timeout, invalid create acknowledgment or server-side write failure requires
exact reconciliation; it cannot cause an automatic second job.
`EXPERIMENTCTL_SENSECORE_CREATE_TIMEOUT_SECONDS` defaults to 120 (10–600).

Public errors include only the HTTP status and bounded provider reason, never
raw responses, commands, signed URLs or account keys. Log text still passes
through the `experiment-redact` Rust redactor; this is a local text
utility and sends no cloud requests. A provider permission failure is distinct
from an explicitly expired log token. Missing logs do not prove training failed.
Invoke the redactor without a subcommand; use `EXPERIMENTCTL_REDACTOR_BIN` when
overriding its executable. Legacy names, parsing modes and timeout fallback
variables have been removed.

Terminal jobs use offline logs. Active jobs fall back to offline logs only when
the live stream is unavailable. Queries are scoped to the owned job's exact
worker/container, workspace and creation-time interval; returned identities and
timestamps are checked, then text is redacted. The response identifies historical
logs, last log time, unavailability and whether the bounded page is truncated.
It does not promise a complete log export. Historical logs can supply metrics
and checkpoint evidence but never count as a fresh process heartbeat.
`SUSPENDED` means stopped; it is normalized to `CANCELLED`, preserving raw state
and an unknown-cause stop reason unless an exact cancellation marker exists.
`SUSPENDING` without that marker remains `UNKNOWN`, not proof of preemption.
No exit code or preemption cause is invented from an empty log response.
Historical `PREEMPTED` terminal lifecycles stay immutable; a subsequent
`SUSPENDED` observation reports the corrected normalized state in `detail`
without replacing the earlier lifecycle or inferring a new preemption.

The migration does not rewrite historical Runs, Attempts, Actions or upload
sessions, and does not add a cloud-side retry policy (`backoff_limit: 0`). A
transport-only rollout is checked using live read-only queries and isolated
create/stop contract tests; a new GPU training run requires its own acceptance
scope.

Primary contracts: [ACP create](https://www.sensecore.cn/help/docs/API/acp/training-job-service-create-training-job),
[ACP list](https://www.sensecore.cn/help/docs/API/acp/training-job-service-list-training-jobs),
[Workers](https://www.sensecore.cn/help/docs/API/acp/training-job-service-list-workers),
[Log token](https://www.sensecore.cn/help/docs/developer-tools/cms/openapi/sign-log-livestream-token),
[Offline logs](https://www.sensecore.cn/help/docs/developer-tools/cms/openapi/logstream-logs),
[Pool bindings](https://www.sensecore.cn/help/docs/API/aec2/workspace-service-list-a-i-elastic-compute-clusters-details),
[ACP lifecycle](https://www.sensecore.cn/help/docs/cloud-foundation/compute/acp/acpIntroduction/acpLifeCycle).

## New worker entrypoint

New Runtime receipts advertise `launcher-manifest.v1`. Their ACP startup script
contains one entrypoint instead of nested `env` blocks and inline JSON:

```sh
env ML_EXPD_BOOTSTRAP_TOKEN=<exact-attempt-capability> \
  ml-exp-worker --manifest-url https://API/api/launch-transfers/PROJECT/RUN/ATTEMPT/SHA256
```

The server seals canonical JSON in the private exact-Attempt transfer record.
It includes identity, workdir, training argv, environment, input/checkpoint
manifests, transfer endpoints and worker duration, and excludes the transfer
credential. GET requires that Attempt's unexpired capability and exact digest;
the capability cannot query projects, control jobs or access another Attempt.
The configured public service prefix (such as `/ml-expd`) is retained in URLs.
Responses are `no-store`. The launcher accepts only the fixed HTTPS path,
performs a bounded 30-second, 256-KiB fetch without redirects or ambient proxies,
checks canonical encoding/SHA256/identity/schema and only then starts the worker.
Failures stop before training and print fixed diagnostics without response bodies,
URLs or credentials. The bootstrap token is removed, and the existing worker
removes its upload capability before launching project code.

The entrypoint creates OUTPUT_DIR, selects the frozen workdir and invokes the
same managed worker with the same whole-worker GNU timeout internally. Training
argv is passed directly, without shell parsing or `eval`. TERM/INT forwards to
the owned process group. Run/Attempt manifests retain scientific argv/config and
the credential-free `ml-exp-worker` command; the sealed launch manifest provides
the additional dispatch configuration.

Both WYD and SenseCore use this entrypoint when their immutable image receipt
advertises the capability and artifact transfer is configured. Changing the
worker/recipe digest gives new builds a new Runtime identity. Existing READY
images, frozen Runs and past Attempts keep their old command path; they cannot
gain this executable without rebuilding. Rebuild the Runtime to use this capability.

## Timeout sources

| Limit | Source and scope |
| --- | --- |
| Worker duration | Client `resources.max_time`, default `00:10:00`, frozen in Run; a client `02:00:00` becomes 7200 seconds |
| Termination grace | GNU `timeout --signal=TERM --kill-after=30s`; TERM first, then KILL if still running |
| Launch configuration fetch | New launcher's own 30-second bounded HTTPS fetch, before the worker timer starts |
| Data-preparation script | Client `data_preparation.timeout_seconds`, default 1200; still inside worker duration |
| SenseCore REST request | Generic 30 seconds; create 120 seconds by default, override `EXPERIMENTCTL_SENSECORE_CREATE_TIMEOUT_SECONDS` (10–600); live log token/poll 20 seconds |
| Scheduler-control process | Controller's default 300 seconds, a daemon-side operation timeout rather than training duration |
| Builder | Separate build-command budget; the current deployment uses 900 seconds and private builder wait 1200 seconds |

The worker timer starts inside the already running container. It covers restore,
data preparation/delivery, training, checkpoint registration and final result
upload; it excludes cloud queue, image pulling and NAS setup. WYD also freezes
the same duration into Slurm `--time`; ACP currently has no corresponding cloud
wall-time field in our create document. It is not a SenseCore fixed two-hour
limit. Every new Attempt currently gets its own timer. Zero duration is rejected
for new API submissions rather than silently disabling GNU timeout. Automatic
recovery needs a separate cumulative budget and deadline; see
[recovery preparation](sensecore-recovery.md).

## Historical startup command

`roles[].image_path`, `resource_pool`, scheduling and NAS `mount` are separate
REST fields. ACP pulls the pinned image and mounts NAS before executing
`startup_script` **inside that container**. The script does not build or convert
an image. Imported source is already copied into the image at `/workspace`.

The controller assembles the script as quoted argument arrays, adding these
layers only when enabled by the frozen Run:

| Layer | Contents and purpose |
| --- | --- |
| Job identity | `BACKEND_JOB_ID`, tying observations to the exact ACP job |
| Checkpoint restore | `ML_EXPD_CHECKPOINT_RESTORE`: checkpoint identity, storage location, per-file SHA256 and size; the worker verifies these before training |
| Persistent state | `ML_EXPD_CHECKPOINT_STATE` and its URL/interval: exact Run/Attempt/storage scope, metadata registration; full state stays in NAS |
| Script data preparation | `ML_EXPD_DATA_PREPARATION`: script identity, argv and timeout; preparation failure prevents training |
| Input assets | `ML_EXPD_INPUT_ASSETS`: IDs, mount paths, delivery policy, download URLs and file/archive hashes; verified shared-cache reuse or delivery to `/inputs` |
| Output return | Upload URL, limit, output patterns and a write capability scoped to this Attempt; optional snapshot URL/interval enables full checkpoint backup |
| Experiment environment | Authored environment plus `OUTPUT_DIR`, Run/Attempt/source IDs, `INPUTS_DIR`; optional `DATA_DIR`, `STATE_DIR` and `RESUME_DIR` |
| Shell wrapper | Creates output directory, changes to the image workdir, then `exec`s the remaining argv without `eval` |
| Time limit | `timeout --signal=TERM --kill-after=30s <seconds>s` wraps the whole worker, including data preparation and result return |
| Worker and training | `python3 /usr/local/lib/ml-expd/worker.py` followed by the image entrypoint and frozen training arguments |

The shell wrapper is:

```sh
/bin/sh -c 'mkdir -p "$1" && cd "$2" && shift 2 && exec "$@"' \
  ml-expd <output-directory> <image-workdir> <timeout-and-worker-argv...>
```

Here `ml-expd` is the shell's `$0` label, not a second daemon. `$1`/`$2` are
the directories; `shift 2` removes them before executing the worker. Multiple
`env` prefixes come from composing independent dispatch layers. Long strings
mostly contain JSON file manifests and hashes, not elaborate shell operations.

The worker removes private transfer capabilities from its environment before
starting the training child. It restores state, prepares/validates inputs,
then starts the authored entrypoint. While training runs it can periodically
verify complete checkpoint generations and register their metadata. Finally it
archives the declared outputs and sends them to the exact Attempt's upload
endpoint. Data and complete persistent state are outside the output directory.
Training still owns optimizer/RNG/data-cursor recovery and the scientific meaning
of metrics. This launcher does not automatically restart a failed job.
