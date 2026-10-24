# Connection and backend configuration

## Client: connect to an existing service

Install only `ml-experiment-client` on the researcher's computer. Python 3.10+
is required; Docker, SSH and cloud SDKs are not. The operator supplies the API
base URL, client wheel and a private API token.

| Setting | Purpose |
|---|---|
| Saved `url` | Default HTTPS base URL including any reverse-proxy prefix |
| Saved `token` or `token_file` | Private token or reference to a private token file, outside uploaded source |

```bash
ml-exp config set --url 'https://api.example.org/ml-expd'
# Enter the token at the hidden prompt once.
ml-exp check
ml-exp run status --project my-project --run trial-001
```

Defaults are saved in `~/.config/ml-expd/client.json`, or
`$XDG_CONFIG_HOME/ml-expd/client.json` when set. On POSIX the file is private
(0600) and new configuration directories are 0700. The token is stored as plain
text in that private file; keep it outside Git and uploaded source. No API request
is made by `config set`, `show` or `clear`.

To retain an existing token file instead of storing another copy:

```bash
ml-exp config set --url 'https://api.example.org/ml-expd' --token-file /private/client.token
ml-exp config show           # URL, file path and whether auth is configured; never the token
ml-exp config set --prompt-token  # rotate the saved token using the hidden prompt
ml-exp config clear          # remove defaults; referenced token files are retained
```

For noninteractive setup use `--token-stdin < /private/client.token`; there is
no token command-line argument. `--config PATH` or `ML_EXPD_CLIENT_CONFIG`
selects another private configuration. CLI `--url` overrides `ML_EXPD_API_URL`,
which overrides the saved URL. Environment authentication overrides saved
authentication. Saved credentials remain bound to their saved URL: choosing a
different server requires explicit credentials for that server, before any HTTP
request. Saved experiment states retain their original server binding as well.

`ml-exp update --check` checks the official GitHub release for a newer client;
`ml-exp update` downloads and verifies the standalone client wheel against the
release SHA256SUMS, then updates the current Python environment without installing
server packages or dependencies. `ml-exp update --version 1.5.1` chooses a stable
release. It never uses the experiment API token, changes connection defaults, or
automatically downgrades. An environment containing the server must be upgraded
through the operator's deployment process; client self-update does not replace a
server installation. A compatible `pip` or `uv` must already be available.

`ml-exp check` verifies the basic API and reports optional deployment components.
An API without a builder or result store is a valid minimal deployment. Before
using it for new training, run `ml-exp check --training`: it also requires the
training configuration and a responding private image builder, and returns 2
with the missing/disabled/unavailable component names when incomplete. This is
a read-only check; it does not build an image or reserve a GPU. Backend credentials,
storage access and resource admission still need the experiment's normal preflight.
Use `check --json` for the full machine-readable response; the default output is
a short human-readable summary. The [two-machine example](two-machine-quickstart.md)
includes private CA trust and source import without a builder.
If training is incomplete, the operator follows
[the ordered setup checklist](operator-guide.md#enable-training-after-source-import).
The researcher then uses the normal experiment workflow; no additional client
credential or command family is needed.

Provision the token file privately before running this command. On POSIX systems
give its directory mode 0700 and file mode 0600. `ML_EXPD_API_TOKEN` is an alternative
for process environments; setting both nonempty token sources is an error, so an
old shell variable cannot silently shadow a new token file. Environment variables
remain supported for CI and temporary overrides; they do not change the saved
file. Remote HTTP, URL-embedded credentials and redirects
are rejected. Local loopback HTTP is permitted for development.

Project/Run IDs, source, Dockerfile, executor, resources, inputs, metrics and the
GPU-hour budget belong in the [experiment JSON](api-quickstart.md#first-experiment).
Keep client state and downloads outside source. State binds recovery operations
to the original server and experiment identities; it contains no API token.
`--seconds` bounds client waiting, `resources.max_time` bounds the remote job.
An expired client wait does not stop a remote task.

W&B is optional. Its API key is configured on the server; each experiment chooses
whether and where to publish. The operator also installs the server's `wandb`
extra; base-server metrics need no W&B SDK. The service uses a shared API bearer and
instance-wide W&B credentials, without per-user/project RBAC. Distribute access
accordingly; see [W&B](wandb.md) and [deletion permissions](resource-deletion.md).

## Connection errors

API connection failures include a stable `error_code` in the existing stderr
JSON. The client classifies typed errors without printing the raw network/TLS
exception, certificate names, private endpoints or request credentials.

| Error code | Action |
|---|---|
| `TLS_CERTIFICATE_VERIFICATION_FAILED` | Check the server certificate's validity and chain. For an approved private CA, set `SSL_CERT_FILE` to the public CA certificate supplied by the operator. |
| `TLS_HOSTNAME_MISMATCH` | Use an API hostname/IP present in the server certificate SAN, or have the operator issue the correct certificate. |
| `TLS_HANDSHAKE_FAILED` | Ask the operator to check that this is the HTTPS endpoint and that the TLS/proxy configuration is compatible. |
| `CONNECTION_REFUSED` | Check the API address/port, running service and firewall or approved network path. |
| `DNS_LOOKUP_FAILED` | Check the hostname in `ML_EXPD_API_URL`, DNS and network access. |
| `CONNECTION_TIMEOUT` | Check routing, VPN/proxy and service availability. A timeout does not prove a write was rejected. |
| `CONNECTION_FAILED` | The cause could not be classified safely; check the address and network with the operator. |

TLS errors have `retryable=false`: fix the trust, hostname or TLS configuration
before retrying. Do not disable certificate/hostname verification or send private
keys. `retryable=true` for other connection failures is a transport hint, not
permission to replay a write. The HTTP transport does not automatically replay
requests; inspect saved identities and reconcile uncertain operations first.

The [two-machine guide](two-machine-quickstart.md#3-configure-https) shows a
separate private CA, a matching server SAN and client trust configuration.

## Server: common services

Start from [source-api.yaml](../server/examples/source-api.yaml). The operator
configures an API token, writable metadata/state roots, result object storage and
a private image-builder socket. [Deployment](operator-guide.md) describes service
accounts, installation and permission gates.

For API-only installation, use [minimal-api.yaml](../server/examples/minimal-api.yaml).
It permits managed project creation and source import but keeps scheduler writes
off. Add an executor, builder and object store independently when enabling training;
these are not provisioned or inferred from cloud credentials.

Training backend, image builder and storage are separate choices. A Slurm service
can use a local/SSH builder without PAI credentials. A DLC service can consume an
image from the same builder without enabling DSW. ACR, CCR and S3/OSS credentials
are required only for the configured repositories and transfer paths.

## Enable only the backends you use

Copy one of these complete executor files to `/etc/ml-expd/executors.yaml` and
set `container_execution.profiles_file` to that file. Replace its placeholders
before enabling scheduler mutations.

| Backend | Executor template | Additional private configuration | Setup |
|---|---|---|---|
| SSH Slurm/Apptainer | [slurm.yaml](../server/examples/executors/slurm.yaml) | Service-account SSH config/key/known_hosts; registry pull auth for private images | [Slurm](slurm.md) |
| SenseCore ACP | [sensecore.yaml](../server/examples/executors/sensecore.yaml) | [sensecore-rest.json](../server/examples/sensecore-rest.json), CCR authorization, NAS; data registry/debug CPU copy settings when uploading datasets | [SenseCore](sensecore-rest.md) |
| PAI DLC | [pai-dlc.yaml](../server/examples/executors/pai-dlc.yaml) | [pai.json](../server/examples/pai.json), Docker registry auth, approved personal CPFS and runtime role | [PAI](pai.md) |

Only executors present in that file are published to clients. For multiple
backends, put their uniquely named entries under the same `executors:` mapping;
there is no automatic directory loading or provider auto-discovery. Executor IDs
are operator-chosen names, not built-in cluster names. To stop offering a profile
for new submissions, remove its entry; existing frozen Runs/Attempts retain their
original backend definition.

Backend credentials are read when the selected adapter needs them. A configured
Slurm executor does not require SenseCore/PAI credentials, and neither cloud
backend requires a Slurm login. Start with `ml-exp check` to inspect published
capabilities; then follow the chosen backend's preflight and bounded acceptance.
Reading the catalogue does not authenticate or allocate a cloud job.

The [Docker auth template](../server/examples/registry-auth.json) uses `auths` and
base64-encoded `username:password`. Base64 is not encryption. Restrict it to exact
registry hosts and keep actual values outside Git. The older
[single-registry pull format](../server/examples/registry-pull.json) remains
accepted by Slurm; PAI requires Docker `auths` format. Publisher auth can remain
separate from pull-only credentials.

Optional capabilities have independent configuration: [image builds](builds.md),
[OSS dataset/archive settings](pai.md), [W&B](wandb.md) and
[temporary debug machines](debug-sessions.md). Enabling a training executor does
not enable a debug machine, DSW builder or another backend.
