# SSH Slurm and Apptainer

This backend connects to an existing Slurm cluster through an operator-configured
SSH alias. The adapter has no site name or fixed shared filesystem path. The
current implementation supports one node, with a typed `gpu:MODEL:COUNT` GRES.

## Executor and credentials

Copy [the Slurm example](../rust/deploy/slurm.example.json) to the file named by
`submission_config` in the [deployment file](../rust/README.md#deploying-a-native-service).
Change the backend ID, SSH alias, partition, account, QOS and GPU model to values
assigned by your cluster administrator.

| Setting | Meaning |
|---|---|
| `profile.target.backend_id` | Executor ID, unique across Slurm and cloud |
| `profile.ssh_alias` | SSH config alias usable by the daemon account without interactive login |
| `profile.partition`, `account`, `qos` | Existing Slurm placement and authorization |
| `profile.gpu_type` | GPU model in the typed GRES, for example `a100`; the Run's GPU count sets the count |
| `profile.mount_root` | Absolute shared filesystem root mounted into Apptainer |
| `profile.run_root` | ML-Expd Run storage below that shared root |
| `profile.data_root` | Dataset root inside the shared filesystem |
| `profile.max_gpus`, `max_cpus`, `max_memory_gb`, `max_walltime_seconds` | Finite limits for one Run |
| `ssh.config_file`, `ssh.known_hosts_file` | Pinned private SSH files on the API host |
| `ssh.helper_path`, `ssh.helper_config_path` | `ml-exp-worker-rs` and its private configuration on the gateway |
| `delivery.root` | Optional. Shared directory where the server places workers and code; see [below](#delivered-worker-and-code) |

For example, provision this under the daemon's own SSH configuration:

```sshconfig
Host hpc-login
    HostName login.cluster.example
    User REPLACE_CLUSTER_USER
    IdentityFile /etc/ml-exp/ssh/id_ed25519
    UserKnownHostsFile /etc/ml-exp/ssh/known_hosts
    BatchMode yes
    StrictHostKeyChecking yes
```

Keep the private key outside Git, source and images. Verify the server's host key
through your cluster administrator. The daemon needs local `ssh`. The gateway needs
`ml-exp-worker-rs` once, to initialize its stores, then Slurm tools, Apptainer, `sh`
and coreutils (`sha256sum`, `cp`, `mv`). With `delivery.root` the server keeps that
helper current itself. The gateway and compute nodes must see
the same `run_root`. Remote paths are not paths on the API server.

Private images use `registry_auth_file` in the worker's materialization configuration
on the gateway; prefer a scoped Docker auth file. Slurm converts the digest-pinned OCI image to SIF before scheduling. Conversion and source staging
have their own timeouts and receipts; CPU/GPU resources and walltime are then
passed to Slurm. See [builds](builds.md) and [preparation diagnostics](recovery.md).

## Data and compute connectivity

Uploaded parts pass through the bounded API transfer buffer and go over SSH to
shared storage. The remote receiver verifies the complete archive and files
before admitting a Run. Complete checkpoints stay on shared storage; the API
registers their exact identity. Result collection reads the frozen output path.

An Attempt lives at `profile.run_root/PROJECT/RUN/attempts/ATTEMPT`. Slurm uses the same path
variables as every other backend, documented in
[persistent checkpoints](persistent-checkpoints.md#enable-and-write):

| Path below the Attempt | Variable | Created by |
|---|---|---|
| `state` | `STATE_DIR` | Gateway staging, before the job is queued |
| `outputs` | `OUTPUT_DIR` | The worker, which refuses one that already exists |
| `outputs/metrics.jsonl` | `METRICS_PATH` | Training code |
| `worker` | None | The worker, for its own ledger and receipt |

`DATASETS_JSON` holds each receipt root, and the job binds that root read-only at its
own path over the read-write `mount_root` bind. The job script exports only Apptainer's
cache and temporary paths: the worker replaces the environment before starting
training, so anything else would stop there. Variables come from the sealed launch
manifest.

Compute nodes [need no Internet access](backends.md#slurm-compute-nodes-without-internet-access).
The SSH gateway needs access to image and data sources. A pre-job download script
runs there when `script_preparation` is on. Do not rely on an offline compute node
fetching data.

The [allocation status endpoint](backends.md#slurm-allocation-status) exposes
Slurm scheduler allocation/queue evidence. It does not measure GPU utilization
or guarantee immediate allocation.

## Delivered worker and code

Without `delivery.root`, a job runs the worker built into its image, the code is in the
image too, and every code change builds the image and its SIF again. Set it to a
directory under `profile.mount_root`, apart from `run_root` and `data_root`, for
example the root that holds the helper's stores:

```json
"delivery": {"root": "/shared/researcher/ml-exp"}
```

It takes effect when the server also keeps a worker store (`builds.worker_store`,
which the deploy tool sets). The server then works as it does on PAI:

- A new Run freezes the server's worker, as `native_slurm_delivery.worker_sha256`. A
  retry or a resume keeps it, and the job runs exactly that file, not the image's.
- The executor lists `capabilities.code_delivery`, so `ml-exp submit` splits the
  directory as [on PAI](builds.md#code-delivered-to-the-job): the files the Dockerfile's
  `COPY` and `ADD` name build the environment, and the whole directory is the Run's
  `code_source_id`. The same fallbacks keep everything in the image: `COPY .`, a
  Dockerfile that cannot be read for certain, more than 64 MiB or 20,000 entries,
  or `data_preparation`.
- When only the code changes, the Run names the same Runtime: nothing is built and
  no SIF is converted.

Before the one submission, while the Action is still `PREPARED`, the server places
what the Attempt froze, over the same SSH channel. This is idempotent preparation and
asks the scheduler for nothing:

| What | Where | How it is proven |
|---|---|---|
| Worker | `ROOT/bin/workers/SHA256`, mode 0555 | A fixed `sh` command checks `sha256sum`; only if the file is absent the server sends it from its store, and it is renamed into place only after its hash matches |
| Gateway helper | `ssh.helper_path` | At each submission and dataset upload, made a verified copy of the server's own worker when it differs, through a copy beside it and one rename |
| Code | `ROOT/code/SOURCE_ID/code`, sealed read-only, with `.code-manifest.json` beside it | `ml-exp-worker-rs code-receive` checks every file and the tree's source identity; only if absent does the server send the listing and content, and the listing is published after the tree is proven |

Each copy is named by its content and never replaced or removed, so any number of Runs
and Attempts reuse it; do not delete `ROOT/bin/workers` or `ROOT/code`. A copy that differs
from its name stops the submission with `SLURM_DELIVERY_REFUSED`, before the scheduler, and
stays for an operator to look at: the server log names the file. Move it aside and execute
again. A transfer that cannot be confirmed answers `SLURM_DELIVERY_UNCONFIRMED`; the Action
is unchanged, so execute again.

The gateway checks the code's names and sizes before `sbatch`. On the compute node the job
script checks the worker's hash before Apptainer starts (`ML_EXPD_WORKER=VERIFIED`), and the
worker checks every code file against the frozen source before training
(`ML_EXPD_CODE=VERIFIED`); a difference is named in the job's error and training does not
start. The job works in the code directory, which is also `SOURCE_DIR`, with
`PYTHONDONTWRITEBYTECODE=1`. Write to `OUTPUT_DIR` and `STATE_DIR`. The shared root must be
executable inside the container, because the worker runs from it.

A Run frozen before delivery, or on a backend without it, keeps running the worker in its
image, and its documents and identity are unchanged.

## Storage identity

Changing an SSH alias or root can change the storage identity even when the physical filesystem
is the same, so do not rename frozen Runs, Attempts or checkpoint scopes.
