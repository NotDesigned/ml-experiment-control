# SSH debug sessions

A debug session requests a temporary **SenseCore CCI** or **PAI DSW** machine.
It is independent of a scientific Run/Attempt. Formal training still uses
SenseCore ACP, PAI DLC, or Slurm.

## Client workflow

The client uses its normal API settings. No cloud access key is sent to it.
Generate a local SSH key if you need one; keep its private half on your machine.
Create a request file with a profile from `debug list`, a new 32-character
hexadecimal request ID and your public key, as in the
[native usage guide](../rust/README.md#temporary-ssh-machines).

```sh
ml-exp debug list
ml-exp debug create --file ./debug.json
ml-exp debug refresh "$debug_identity"
ml-exp debug show "$debug_identity"
```

CPU, memory, GPU, image, approved storage and connection method are bounded by the
server operator's profile. This API does not build an image or submit training.

`refresh` checks the provider and `show` reads saved status. An SSH connection is
returned only after the provider reports Running.

A configured route does not prove reachability from your network. `intranet`
requires a route into the provider's VPC. The API returns
`host_key_verified=false`: verify the machine's host fingerprint through a
trusted channel, or explicitly use your usual SSH first-connection trust
policy. The client never disables SSH host-key checking.

```bash
ml-exp debug stop "$debug_identity" --confirmation "STOP $debug_identity"
ml-exp debug refresh "$debug_identity"
# DSW additionally supports deletion:
ml-exp debug delete "$debug_identity" --confirmation "DELETE $debug_identity"
ml-exp debug refresh "$debug_identity"
```

CCI supports create/query/stop; DSW also supports delete. Each profile in the
catalog reports `delete_supported`. Requesting CCI deletion returns
`CLOUD_CONFIGURATION_INVALID` without a provider mutation.

Stop/delete requests can remain `STOPPING`/`DELETING` while the provider acts.
The server continues requested cleanup about every 30 seconds, at most 16 machines
per pass. A permission denial
(HTTP 403) pauses automatic retries; fix the permission and explicitly refresh. DSW deletion is complete
only after the machine is absent. Stopping CCI releases compute and removes its
session-owned DNAT rule; the stopped CCI and internal Service metadata remain.
There is no automatic restart. An expired or deleted identity cannot be reused
for a new machine; choose a new request ID.

Save work under the approved persistent mount: DSW `/mnt/workspace`, CCI `/data`.
The container's own filesystem is temporary. Stop/delete never deletes the
mounted CPFS/NAS contents. They do remove ephemeral instance storage.

## Lifetimes and uncertainty

Each request states its `lease_seconds`, and `max_active` limits concurrently
active machines. Profiles are disabled until the operator configures them.
DSW always has a four-hour hard ceiling, even if the shared debug-session maximum
is configured higher. A shorter operator lease still takes precedence.

- DSW: a **native shutdown timer** is installed and read back. Its absolute
  deadline never advances on refresh. On expiry the server also stops the instance.
  It deletes an instance only on an explicit `delete`.
- CCI: the injected SSH process has an **absolute expiry deadline**, and the
  server stops the instance and removes its own port rule on expiry. This is
  **not a provider-side compute shutdown timer**: if ML-Expd is unavailable,
  SSH expires but compute may remain allocated until server recovery or manual
  cleanup. Closing the shell or stopping sshd does not release cloud resources.

A repeated `create` with the identical request ID and request returns the
existing record. A changed request under the same ID is rejected. If a create
response is lost, `RECONCILE_REQUIRED` retains ownership and consumes capacity;
refresh queries the original name/identity and **never repeats that create**.
A definitive rejected create is `REJECTED`, with a sanitized provider code.
Do not start replacements while the original allocation is uncertain.

## HTTP API

All routes use the normal bearer authentication.

| Method | Path | Meaning |
| --- | --- | --- |
| GET | `/api/debug` | Configured profiles and saved machines |
| POST | `/api/debug` | Create once |
| GET | `/api/debug/{identity}` | Saved observation |
| POST | `/api/debug/{identity}/refresh` | Reconcile the same machine |
| POST | `/api/debug/{identity}/stop` | Stop compute |
| POST | `/api/debug/{identity}/delete` | Delete a DSW instance; unsupported for CCI |

```json
{
  "profile": "pai-debug",
  "request_id": "REPLACE_WITH_32_HEX_CHARACTERS",
  "ssh_public_key": "ssh-ed25519 REPLACE_WITH_REAL_PUBLIC_KEY",
  "lease_seconds": 900,
  "gpus": 0,
  "cpus": 2,
  "memory_gb": 4
}
```

`machine_status` and `machine_observed_at` retain the last provider observation,
including a stopped machine whose deletion is blocked by a permission error.
The workflow `status` can therefore be `RECONCILE_REQUIRED` while the last
confirmed machine status is `STOPPED`.

The server accepts one Ed25519 or RSA (at least 2048 bits) public key. Private
keys, authorized_keys options, additional shell commands and caller-supplied
mounts/cloud credentials are not request fields. Profile details are frozen in
the private record, so changing the catalog cannot redirect cleanup.

## Operator configuration

Configure machines in the cloud configuration's `debug` block: `enabled`,
`max_active` and `profiles`. Each profile holds a `credentials_file` and a nested
`profile`. PAI/SenseCore credential files and signed REST transports are the same
kind the backends use; no `sco` command is invoked. A machine's record keeps its
frozen profile for cleanup. Unknown fields are refused at every level.

```json
{
  "enabled": true,
  "max_active": 2,
  "profiles": {
    "pai-debug": {
      "credentials_file": "/etc/ml-exp/pai-credentials.json",
      "profile": {
        "provider": "pai_dsw",
        "profile": {"region": "...", "workspace_id": "...", "quota_id": "...",
                    "max_gpus": 1, "max_cpus": 8, "max_memory_gb": 32,
                    "storage": {"cpfs_uri": "bmcpfs://cpfs-...", "workspace_user": "...",
                                "role_arn": "acs:ram::..."},
                    "vpc": {"VpcId": "...", "VSwitchId": "...", "SecurityGroupId": "..."}},
        "image_id": "...",
        "access": {"kind": "intranet"}
      }
    }
  }
}
```

DSW (`provider: pai_dsw`) fields:

- `profile`: the [PAI backend profile](pai.md), with `storage` and `vpc` both
  required here. `storage` is the same approved personal CPFS and role as the DSW builder.
- `image_id`: an approved image with sshd.
- `access`: `{"kind":"intranet"}`, or
  `{"kind":"internet","nat_gateway_id":"ngw-...","eip_id":"eip-..."}` naming an
  existing NAT gateway and EIP.

The personal CPFS path maps to `/mnt/workspace`, public models/datasets mount
read-only, and visibility is PRIVATE. The server does not create a new NAT,
EIP, security group, project mount or OSS filesystem mount. Native PAI assigns
the public port.

CCI (`provider: sensecore_cci`) fields:

- `profile`: the [SenseCore backend profile](sensecore-rest.md). Its `purpose` must not
  be `training`, `mount_path` must be `/data`, and `storage_subdir` must be one
  leading-slash personal directory, such as `/alice`.
- `image`: an image with `/usr/sbin/sshd`, `ssh-keygen` and `timeout` already
  installed. No network package installation happens at startup.
- `network`: an existing EIP in the chosen pool's VPC and zone, with all of
  `eip_name`, `eip_id` (the path of that EIP in the profile's scope),
  `nat_gateway_id`, `external_ip`, `external_port` (at least 1024) and `tenant_id`.

The selected native resource specification must match the profile's CPU,
memory and GPU count. CCI debug uses RESERVED quota. A new, session-owned DNAT
rule uses the profile's `external_port`; no existing rule is migrated. Its UID,
creator, network and CCI target are checked before bind/unbind/delete of that
port rule. CCI app/Service deletion is intentionally not provided. Shared
EIPs/NAT and unrelated instances/rules are never removed.
