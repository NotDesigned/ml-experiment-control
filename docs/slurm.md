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
`ml-exp-worker-rs`, Slurm tools and Apptainer. The gateway and compute nodes must see
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

Compute nodes [need no Internet access](backends.md#slurm-compute-nodes-without-internet-access).
The SSH gateway needs access to image and data sources. A pre-job download script
runs there when `script_preparation` is on. Do not rely on an offline compute node
fetching data.

The [allocation status endpoint](backends.md#slurm-allocation-status) exposes
Slurm scheduler allocation/queue evidence. It does not measure GPU utilization
or guarantee immediate allocation.

## Storage identity

Changing an SSH alias or root can change the storage identity even when the physical filesystem
is the same, so do not rename frozen Runs, Attempts or checkpoint scopes.
