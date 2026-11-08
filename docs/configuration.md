# Connection and backend configuration

## Client: connect to an existing service

Install the standalone `ml-exp` client on the researcher's computer; see the
[quickstart](api-quickstart.md#install-and-connect). It needs no Python, Docker,
SSH or cloud SDK. The operator supplies the API base URL and a private API token.

```sh
ml-exp --url https://api.example/ml-expd config set --token-file /private/client.token
ml-exp config show           # URL, token file path and whether auth is configured
ml-exp check
```

`config set` saves the URL and a reference to the token file; the token itself is
not copied or passed on a command line. `config show` never prints the token and
neither command contacts the API. On POSIX systems give the token file's directory
mode 0700 and the file mode 0600. Windows credential files must allow only the
current user, SYSTEM and local Administrators.

Settings are saved in `ml-exp/client.json` under `$XDG_CONFIG_HOME`, or under
`.config` in your home directory (`USERPROFILE` on Windows). Keep this file, token
files, client state and downloads outside uploaded source.

| Setting | Selected by | Precedence |
|---|---|---|
| Configuration file | `--config PATH`, `ML_EXPD_CLIENT_CONFIG` | Flag, then environment, then the default path |
| API URL | `--url`, `ML_EXPD_API_URL`, saved `url` | In that order |
| Token | `ML_EXPD_API_TOKEN` or `ML_EXPD_API_TOKEN_FILE`, saved token file | Environment overrides saved; setting both variables is an error |

Saved credentials are bound to their normalized URL: selecting a different URL
without explicit environment credentials fails before any request. A saved state
file binds recovery to its original server and contains no API token. URLs must be
HTTPS, or HTTP to loopback; embedded credentials, queries and fragments are
rejected. The client follows no redirects, ignores ambient proxy settings and does
not retry requests automatically.

`ml-exp update --check` looks for a newer stable native release. `ml-exp update`
installs the highest one after checking hash, platform and version, and refuses an
automatic downgrade. `ml-exp update --version X.Y.Z` selects an exact version. An
update never changes connection defaults or a server installation. A client from an
earlier release cannot update itself to this one, so use the installer once; see the
[native usage guide](../rust/README.md#client-configuration).

`ml-exp check` validates the authenticated connection and returns the
server's version, capabilities and source limits. `ml-exp backend` lists executors.
For per-backend submission availability read `ml-exp get /api/readiness`. It makes
no provider request and starts no job. `training=INCOMPLETE` there is acceptance
evidence, not a blocker; inspect `submission_available_backends`. If no backend is
available, the operator follows
[the setup checklist](operator-guide.md#enable-training-after-source-import).

Project/Run IDs, source, executor, resources, inputs, metrics and the GPU-hour
budget belong in the [experiment file](api-quickstart.md#first-experiment).
`--seconds` bounds client waiting, `walltime_seconds` bounds the remote job. An
expired client wait does not stop a remote task.

W&B is optional and configured on the server; each experiment chooses whether and
where to publish. The service uses one shared API bearer and instance-wide W&B
credentials, without per-user or per-project RBAC. Distribute access accordingly;
see [W&B](wandb.md) and [administrative deletion](resource-deletion.md).

## Connection errors

Errors are one sanitized line on stderr with a nonzero exit code. They do not print
tokens, signed URLs or raw TLS or network exceptions.

| Message begins with | Action |
|---|---|
| `invalid API URL` | Use HTTPS, or HTTP to loopback, without userinfo, query or fragment |
| `invalid API token` | Check the token file's contents |
| `cannot read or write private client configuration` | Check the path, ownership and permissions of the settings or token file |
| `saved credentials belong to a different API URL` | Run `config set` for this URL, or supply explicit environment credentials |
| `HTTP request failed` | Check the address, DNS, certificate, routing and service with the operator. TLS, DNS, refused and timed-out connections all report this |
| `daemon did not provide a valid health document` | The address is not an ML-Expd API of this release, or a proxy altered it |

The client trusts the public WebPKI roots built into it. A certificate from a
private CA is not accepted, and there is no option to disable verification; give
the API a publicly trusted certificate.

A transport failure does not prove that a write was rejected, and it is not
permission to replay one. Inspect saved identities and reconcile uncertain
operations first; see [recovery](recovery.md).

## Server: common services

The service reads one private deployment file, which refers to independently
approved configuration files; see the
[native deployment guide](../rust/README.md#deploying-a-native-service) and the
[operator guide](operator-guide.md). A minimal API needs only a token, read
configuration and approved Projects, and can import source without a backend.

Training backend, image builder and storage are separate choices. A Slurm service
can use a local BuildKit publisher without PAI credentials. A DLC service can use
an image from the same builder without enabling DSW. Registry and object-store
credentials are required only for the configured repositories and transfer paths.

## Enable only the backends you use

| Backend | Deployment reference | Example | Provider notes |
|---|---|---|---|
| SSH Slurm/Apptainer | `submission_config` | [slurm.example.json](../rust/deploy/slurm.example.json) | [Slurm](slurm.md) |
| PAI DLC | `cloud_config` | [pai.example.json](../rust/deploy/pai.example.json) | [PAI](pai.md) |

Omitted backends stay disabled, and the examples start with `enabled`,
`allow_run_creation` and polling off. Several cloud backends go in one cloud
file's `backends` list. Executor IDs must be unique across Slurm
and cloud; they are operator-chosen names, not built-in cluster names. Removing an
entry stops offering it for new work, and existing frozen Runs and Attempts keep
their original definition. A cloud file that sets `allow_run_creation` must also set
`exchange`; `--check` refuses it otherwise. Credential file shapes and permissions
are in the [native deployment guide](../rust/README.md#deploying-a-native-service).

`ml-exp backend` shows what the service publishes without contacting a provider.
Capacity and quota do not guarantee an allocation.

Private images use Docker `auths` credentials, set per backend with
`registry_auth_file` on PAI and separately on the builder. Base64 in that format is
not encryption. Restrict it to exact registry hosts and keep values outside Git.

### Checkpoint copies and the state purge

A PAI cloud file with `artifacts` and `exchange` makes new Runs copy their registered
checkpoints to OSS and lets the server purge ended Attempts' `state/`
([recovery](recovery.md#checkpoint-copies-and-the-state-purge)). The optional
`checkpoint_archive` block tunes it:

```json
"checkpoint_archive": {"purge_delay_seconds": 900, "automatic_purge": true, "upload_bytes_per_second": 67108864}
```

| Key | Default | Meaning |
|---|---|---|
| `purge_delay_seconds` | `900` | Seconds from the moment every purge condition holds to the automatic purge. `0` purges at once. `run purge-state` never waits |
| `automatic_purge` | `true` | The server starts the purge job itself. `false` leaves only `ml-exp run purge-state`. Either needs the backend's `collector` and `enabled` |
| `upload_bytes_per_second` | `67108864` (64 MiB/s) | The most bytes a second one Attempt's worker sends to OSS, beside training and after. `0` is no limit |

Whether a Run copies checkpoints is frozen in the Run when it is created and does not
change with this file. Omitting `artifacts` stops copies for new Runs only. A Run also
copies only when the worker it freezes declares the capability, which the release's own
worker does; `deploy_native.py` pins `builds.worker_sha256` to that worker when it
installs the server, in the same step.

The bucket under `artifacts` must not have versioning enabled or suspended. Parts are
written with `x-oss-forbid-overwrite`, which OSS ignores on a versioned bucket, so a
copy could then be replaced after it was verified.

Rolling the server back to a release before this one is unsafe once Runs that copy
exist: that release refuses their packages, which carry `checkpoint_archive`, and
cannot purge them. `deploy_native.py --rollback` refuses it and names their
preparations.

Optional capabilities have independent configuration: [image builds](builds.md),
[W&B](wandb.md) and [temporary debug machines](debug-sessions.md). Enabling a
training executor does not enable a debug machine, builder or another backend.
