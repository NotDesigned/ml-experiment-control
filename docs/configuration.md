# Connection and backend configuration

## Client: connect to an existing service

Install only `ml-experiment-client` on the researcher's computer. Python 3.10+
is required; Docker, SSH and cloud SDKs are not. The operator supplies the API
base URL, client wheel and a private API token.

| Setting | Purpose |
|---|---|
| `ML_EXPD_API_URL` | HTTPS base URL including any reverse-proxy prefix, for example `https://api.example.org/ml-expd` |
| `ML_EXPD_API_TOKEN_FILE` | Path to the private token file, outside uploaded source |

```bash
export ML_EXPD_API_URL='https://api.example.org/ml-expd'
export ML_EXPD_API_TOKEN_FILE="$HOME/.config/ml-expd/client.token"
ml-exp check
```

Provision the token file privately before running this command. On POSIX systems
give its directory mode 0700 and file mode 0600. `ML_EXPD_API_TOKEN` is an alternative
for process environments; setting both nonempty token sources is an error, so an
old shell variable cannot silently shadow a new token file. `--url` overrides the
URL environment variable. Remote HTTP, URL-embedded credentials and redirects
are rejected. Local loopback HTTP is permitted for development.

Project/Run IDs, source, Dockerfile, executor, resources, inputs, metrics and the
GPU-hour budget belong in the [experiment JSON](api-quickstart.md#first-experiment).
Keep client state and downloads outside source. State binds recovery operations
to the original server and experiment identities; it contains no API token.
`--seconds` bounds client waiting, `resources.max_time` bounds the remote job.
An expired client wait does not stop a remote task.

W&B is optional. Its API key is configured on the server; each experiment chooses
whether and where to publish. The current service uses a shared API bearer and
instance-wide W&B credentials, without per-user/project RBAC. Distribute access
accordingly; see [W&B](wandb.md) and [deletion permissions](resource-deletion.md).

## Server: common services

Start from [source-api.yaml](../server/examples/source-api.yaml). The operator
configures an API token, writable metadata/state roots, result object storage and
a private image-builder socket. [Deployment](operator-guide.md) describes service
accounts, installation and permission gates.

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
