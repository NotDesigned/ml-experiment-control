# SSH Slurm and Apptainer

> **Configuration names on this page are partly retained Python reference**
> (`container_execution`, `server/examples`, `EXPERIMENTCTL_*`, YAML executors). The
> provider behavior still applies. Configure a native Slurm backend from the
> [native deployment guide](../rust/README.md#deploying-a-native-service) and
> [its example](../rust/deploy/slurm.example.json).

This backend connects to an existing Slurm cluster through an operator-configured
SSH alias. The adapter has no site name or fixed shared filesystem path. The
current implementation supports one node, with a typed `gpu:MODEL:COUNT` GRES.

## Executor and credentials

Use [executors/slurm.yaml](https://github.com/NotDesigned/ml-experiment-control/blob/v1.5.1/server/examples/executors/slurm.yaml) as the entire
`container_execution.profiles_file`, or copy its entry into a combined executor
file. Change the ID/title, SSH alias, partition, account, QOS and GPU model to
values assigned by your cluster administrator.

| Setting | Meaning |
|---|---|
| `backend.ssh_alias` | SSH config alias usable by the daemon account without interactive login |
| `backend.partition`, `account`, `qos` | Existing Slurm placement and authorization |
| `backend.gres` | Typed GPU request, for example `gpu:a100:1`; Run GPU count determines the requested count |
| `backend.mount_root` | Absolute shared filesystem root mounted into Apptainer |
| `storage_root` | ML-Expd project/run/data storage below that shared root |
| `data_root` | Dataset root inside the shared filesystem |
| `backend.apptainer_unsquash` | Use extracted SIF execution when the cluster cannot use FUSE |

For example, provision this under the daemon's own SSH configuration:

```sshconfig
Host hpc-login
    HostName login.cluster.example
    User REPLACE_CLUSTER_USER
    IdentityFile /etc/ml-expd/ssh/id_ed25519
    UserKnownHostsFile /etc/ml-expd/ssh/known_hosts
    BatchMode yes
    StrictHostKeyChecking yes
```

Keep the private key outside Git, source and images. Verify the server's host key
through your cluster administrator. The daemon needs local `ssh` and `rsync`;
the remote staging host needs Python 3, rsync, Slurm tools and Apptainer. Staging
and compute nodes must see the same `storage_root`. Remote paths are not paths
on the API server.

Private images use `container_execution.registry_pull_file`; prefer a scoped
[Docker auth file](https://github.com/NotDesigned/ml-experiment-control/blob/v1.5.1/server/examples/registry-auth.json). Slurm converts the
digest-pinned OCI image to SIF before scheduling. Conversion and source staging
have their own timeouts and receipts; CPU/GPU resources and walltime are then
passed to Slurm. See [builds](builds.md) and [preparation diagnostics](recovery.md).

## Data and compute connectivity

Uploaded parts pass through the bounded API transfer buffer and go over SSH to
shared storage. The remote receiver verifies the complete archive and files
before admitting a Run. Complete checkpoints stay on shared storage; the API
registers their exact identity. Result collection reads the frozen output path.

Compute nodes may lack Internet access. Set `compute_internet: false` and configure
the [fixed-destination API relay](backends.md#slurm-compute-nodes-without-internet-access)
on a gateway reachable by compute nodes. That gateway needs access to the API
and image/data sources; it can be any suitable host. A pre-job download script
uses the connected gateway when the relay path is configured. Do not rely on
an offline compute node fetching data or posting checkpoints directly.

The [allocation status endpoint](backends.md#slurm-allocation-status) exposes
Slurm scheduler allocation/queue evidence. It does not measure GPU utilization
or guarantee immediate allocation.

## Earlier state

The daemon starts from a brand-new registry. Records from earlier versions,
including `remote_storage: wyd` locations and `wyd:` preparation markers, stay on
disk unread: nothing accepts or rewrites them, and an uncertain old transfer is
never replayed. Only `kind: slurm` configures this backend. There is no `wyd`
alias and no Python `backends.wyd` import entry point.

Executor IDs, SSH aliases and storage roots can keep their old names. Changing an
SSH alias or root can change the storage identity even when the physical filesystem
is the same, so do not rename frozen Runs, Attempts or checkpoint scopes.
