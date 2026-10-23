# SSH debug sessions

A debug session requests a temporary **SenseCore CCI** or **PAI DSW** machine.
It is independent of a scientific Run/Attempt. Formal training still uses
SenseCore ACP, PAI DLC, or WYD Slurm.

## Client workflow

Use the normal `ML_EXPD_API_URL` and `ML_EXPD_API_TOKEN_FILE` settings. No cloud
access key is sent to the client. Generate a local SSH key if you need one;
keep its private half on your machine.

```bash
ml-exp debug profiles
ml-exp debug create --id debug-myproject-001 --profile pai-cpu \
  --ssh-key ~/.ssh/id_ed25519.pub --duration-seconds 1800
ml-exp debug refresh --id debug-myproject-001
ml-exp debug status --id debug-myproject-001
ml-exp debug list
```

Select a profile returned by `profiles`; CPU, memory, GPU, image, approved storage
and connection method are controlled by the server operator. The initial
profiles can be CPU-only. This API does not build an image or submit training.

`create` may return `STARTING`. Use `refresh` to observe the original instance
and finish its access setup. `status`/`list` read the last saved observation;
they do not make cloud requests. When the status is `RUNNING`, a configured
route appears in `connection.ssh_command`. Add `-i ~/.ssh/your_private_key` if
that key is not already in your SSH agent.

A configured route does not prove reachability from your network. `intranet`
requires a route into the provider's VPC. The API returns
`host_key_verified=false`: verify the machine's host fingerprint through a
trusted channel, or explicitly use your usual SSH first-connection trust
policy. The client never disables SSH host-key checking.

```bash
ml-exp debug stop --id debug-myproject-001
ml-exp debug refresh --id debug-myproject-001
# DSW additionally supports deletion:
ml-exp debug delete --id debug-myproject-001
ml-exp debug refresh --id debug-myproject-001
```

CCI supports create/query/stop; DSW also supports delete. The profile catalog
returns `operations` and `expiry_action`. Requesting CCI deletion returns
`DEBUG_DELETE_UNSUPPORTED` without a provider mutation.

Stop/delete requests can remain `STOPPING`/`DELETING` while the provider acts.
The server continues requested cleanup every 20 seconds. A permission denial
(HTTP 403) pauses automatic retries; fix the permission and explicitly refresh. DSW deletion is complete
only after the machine is absent. Stopping CCI releases compute and removes its
session-owned DNAT rule; the stopped CCI and internal Service metadata remain.
There is no automatic restart. An expired or deleted identity cannot be reused
for a new machine; choose a new session ID.

Save work under the approved persistent mount: DSW `/mnt/workspace`, CCI `/data`.
The container's own filesystem is temporary. Stop/delete never deletes the
mounted CPFS/NAS contents. They do remove ephemeral instance storage.

## Lifetimes and uncertainty

Default duration is 30 minutes. The operator default maximum is two hours, with
a maximum of two concurrently active sessions. Profiles are disabled until the
operator configures them, and scheduler mutations must also be enabled.

- DSW: a **native shutdown timer** is installed and read back. Its absolute
  deadline never advances on refresh. Server expiry cleanup also deletes the
  stopped instance.
- CCI: the injected SSH process has an **absolute expiry deadline**, and the
  server stops the instance and removes its own port rule on expiry. This is
  **not a provider-side compute shutdown timer**: if ML-Expd is unavailable,
  SSH expires but compute may remain allocated until server recovery or manual
  cleanup. Closing the shell or stopping sshd does not release cloud resources.

A repeated `create` with the identical session ID and request returns the
existing record. A changed request under the same ID is rejected. If a create
response is lost, `RECONCILE_REQUIRED` retains ownership and consumes capacity;
refresh queries the original name/identity and **never repeats that create**.
A definitive rejected create is `REJECTED`, with a sanitized provider code.
Do not start replacements while the original allocation is uncertain.

## HTTP API

All routes use the normal bearer authentication and protocol-2 header.

| Method | Path | Meaning |
| --- | --- | --- |
| GET | `/api/debug-session-profiles` | Configured profiles and limits |
| POST | `/api/debug-sessions` | Create once |
| GET | `/api/debug-sessions` | Saved session list |
| GET | `/api/debug-sessions/{id}` | Saved observation |
| POST | `/api/debug-sessions/{id}/refresh` | Reconcile and finish the same session |
| POST | `/api/debug-sessions/{id}/stop` | Stop compute |
| DELETE | `/api/debug-sessions/{id}` | Delete a DSW instance; unsupported for CCI |

```json
{
  "session_id": "debug-myproject-001",
  "profile": "pai-cpu",
  "ssh_public_key": "ssh-ed25519 REPLACE_WITH_REAL_PUBLIC_KEY",
  "duration_seconds": 1800
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

Set `debug_sessions.profiles_file` to a private JSON file readable by the API
service. Optional settings: `max_active`, `max_duration_seconds`. Use a file
such as `/etc/ml-expd/debug-session-profiles.json` with mode `0640`, owned by
root and the API service's group. Existing PAI/SenseCore credential files and
signed REST transports are reused; no `sco` command is invoked.

Profiles can set `enabled: false` and a short `unavailable_reason` to remain
visible in the catalog while refusing new allocations. Existing records retain
their frozen profile for cleanup.

The profile file is `{"profiles": {"PROFILE_ID": PROFILE}}`. Each profile has
`provider` (`pai-dsw` or `sensecore-cci`), `description`, `cpus`, `memory_gib`,
`gpus` and an operator-owned `native` object.

DSW native fields:

- `workspace_id`, `quota_id`, an approved `image_id` with sshd.
- `cpfs_uri`, `workspace_user`, `data_role_arn` using the same approved personal
  storage/role policy as the DSW builder.
- `user_vpc`: `VpcId`, `VSwitchId`, `SecurityGroupId`.
- `ssh_access_type`: `intranet` or `internet`; internet additionally requires
  existing `ssh_nat_gateway_id` and `ssh_eip_id`.

The personal CPFS path maps to `/mnt/workspace`, public models/datasets mount
read-only, and visibility is PRIVATE. The server does not create a new NAT,
EIP, security group, project mount or OSS filesystem mount. Native PAI assigns
the public port; the literal string `Auto` must not be sent as `ExternalPort`.

CCI native fields:

- `workspace`, `aec2`, `worker_spec`, `image` with `/usr/sbin/sshd`, `ssh-keygen`
  and `timeout` already installed. No network package installation at startup.
- `mounts`: approved personal AFS mount entries with exactly `type: PV_AFS`,
  `mount_path: /data`, `id`, and a **leading-slash** personal `subdir` (for
  example `/alice`). Root and traversal paths are rejected.
- `eip_name`, `external_ip`: an existing EIP in the chosen pool's VPC and zone.

The selected native resource specification must match the profile's CPU,
memory and GPU count. CCI debug uses RESERVED quota. A new, session-owned DNAT
rule is assigned an unused high port; no existing rule is migrated. Its UID,
creator, network and CCI target are checked before bind/unbind/delete of that
port rule. CCI app/Service deletion is intentionally not provided. Shared
EIPs/NAT and unrelated instances/rules are never removed.
