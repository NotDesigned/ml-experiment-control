# Deploy a source/container API service

This guide is for the operator of a new daemon. Existing-API users start at
[the quickstart](api-quickstart.md). These steps are deployment instructions,
not a request to modify a running production workspace.

## Components and ownership

| Component | Responsibility | Privileges/credentials |
| --- | --- | --- |
| `ml-expd` | HTTP, source import, immutable definitions, Actions, collection | Dedicated service UID; workspace writes; native SSH/SCO and server S3 credentials |
| Image builder | Generated Dockerfile, optional pinned dependency installation and registry publication | Root-owned Unix socket worker; Docker/Buildx and registry push access |
| WYD | Slurm allocation and Apptainer conversion/execution | Daemon user's SSH config/keys; registry pull credential |
| SenseCore | SCO worker execution | Daemon user's SCO login; platform registry pull authorization |
| S3-compatible storage | Sealed per-Attempt archives | Bucket access on server only |
| HTTPS reverse proxy | Public API and exact-Attempt upload path | TLS; forwards Authorization and protocol headers |

Use one daemon per workspace. Lifespan acquires its lease before constructing
writable stores; a second process is not a supported standby writer. Keep
source, Action state, SQLite indexes and canonical Run/Attempt metadata in
daemon-owned directories. Large datasets/checkpoints belong in backend storage.
Do not put the HTTP daemon in the Docker group. Keep code directories traversable
and code readable by its UID; keep token/credential files private.

## Install a reviewed revision

On Linux, clone a revision containing server 0.2.0 / protocol 2. Pin the commit;
do not assume an older `main` or release provides the source API. A source
installation requires uv and Rust 1.85+ to build the packaged SCO sanitizer:

```bash
git clone https://github.com/NotDesigned/ml-experiment-control.git
cd ml-experiment-control
git checkout <reviewed-commit>
uv sync --locked --all-packages
uv run --package ml-experiment-server ml-expd --help
```

For an installed deployment, build the core and server distributions, then
install those two wheels into a dedicated runtime environment. The independent
client wheel is installed on client machines, not required by the daemon.
For the paths used by these templates:

```bash
uv build --package ml-experiment-control
uv build --package ml-experiment-server
uv venv /opt/ml-expd/venv
uv pip install --python /opt/ml-expd/venv/bin/python \
  dist/ml_experiment_control-*.whl dist/ml_experiment_server-*.whl
```

Its `ml-expd` and `experiment-safe-sco` entry points must be on the service PATH.
Do not leave a runtime venv inside a root-only checkout inaccessible to its UID.
Back up the existing workspace/config and retain the previous runtime before
an upgrade. Stop the sole daemon for workspace copies; never restore over a
running service. See [Action storage](action-storage.md) for migration/rollback.

Canonical Run/Attempt control metadata belongs under `run_root` (by default a
`runs` sibling of `index_db`), outside imported source. Existing authored
`research_project.yaml` `run_roots` remain compatibility read roots. An upgrade
does not copy old Run trees automatically. For migration, stop the sole daemon,
copy each `<authored-run-root>/<campaign>/<run>` to
`<run_root>/<project>/<campaign>/<run>`, preserving ownership/relative paths,
then restart and verify. During a staged migration the daemon-owned copy wins;
an unmigrated Run continues to write retries beside its existing Attempts, so
one immutable Run is not split between roots. Backend data is referenced, not
relocated by this metadata migration.

## Configure a new workspace

[source-api.yaml](../server/examples/source-api.yaml) is the full shape with
mutations initially disabled. [ml-expd.yaml](../server/examples/ml-expd.yaml)
is the smaller loopback read-only scaffold. Neither is a ready remote deployment.

Provision a dedicated `ml-expd` account and directories, for example:

- `/var/lib/ml-expd` and `/srv/ml-expd/projects`: owned/writable by the daemon;
- `/etc/ml-expd`: traversable by its UID, configuration files readable by it;
- `/var/lib/ml-expd-builder`: private to root;
- `/run/ml-expd-builder`: root-owned, daemon primary group allowed to traverse;
- installed Python/runtime code: directories `0755`, code `0644`, entry points `0755`.

Copy the templates into `/etc/ml-expd`, replace every `REPLACE_*`/example host
and confirm backend paths. Generate a server token as the daemon account:

```bash
umask 077
python3 -c 'import secrets; print(secrets.token_urlsafe(48))' > /etc/ml-expd/http.token
```

The account must have permission to create this file, or provision it as root
and then set its owner to the daemon UID. Token files must be regular,
daemon-owned and mode `0600` or stricter, at least 32 non-whitespace characters.
Artifact/registry credential files should also be daemon-readable only.
Distribute the token privately to permitted clients. It is a shared control
credential; this version has no per-user token issuance or project RBAC API.

### Executor profiles

Edit [executors.yaml](../server/examples/executors.yaml). Profile IDs are the
client-facing selection; credential/config paths stay on the server.

For WYD, configure SSH under the **daemon account**, verify host keys, and
provision persistent `storage_root` beneath `mount_root`. The remote host needs
Slurm, rsync and Apptainer; its partition/GRES, account/QOS and mount must pass
live run-specific preflight. Set `apptainer_unsquash` when FUSE is unavailable.
Private images need pull authorization on the conversion host. A first OCI to
SIF conversion can use the longer `stage_timeout_seconds`; this does not change
the GPU job's frozen wall time.

For SenseCore, install SCO and authenticate under the daemon account. Replace
workspace, AEC2, worker specification and volume mount with existing resources.
Set `gpus` and `capacity` to the worker's **actual** fixed allocation. Requests
must match the profile GPU count and cannot exceed capacity. NAS need not be
reachable from the daemon when S3 return is configured, but the worker needs
its volume mounted and permission/network access to pull the private image.
The implementation currently accepts `quota_type: spot`.

### Private image publication

Edit [image-builder.json](../server/examples/image-builder.json). Set
`client_uid`/`client_gid` to the daemon's numeric identity (`id -u/-g ml-expd`).
Set `source_root` to `<project_registry_root>/source-revisions/sources`.
Keep a narrow base-image allowlist and a writable target registry repository.
Provision Docker with Buildx on the builder host, Skopeo, and root registry
authentication. Buildx uses Docker's native credential configuration; Skopeo
uses `registry_auth_file`. Both must authorize the same publication workflow.
The pull-only [registry-pull.json](../server/examples/registry-pull.json) is a
separate credential used for WYD OCI conversion; it does not configure SCO's
platform-side registry credentials.

Start the root-owned builder with the installed runtime:

```bash
/opt/ml-expd/venv/bin/python -m ml_exp_server.image_builder \
  --config /etc/ml-expd/image-builder.json
```

The worker creates a `0660` Unix socket, checks the caller's peer UID and only
accepts the reviewed fixed packaging operation. It never executes a project
Dockerfile. `publisher: buildkit` reuses registry blobs and emits Docker schema
2 without attestations; Skopeo verifies the remote manifest/config digests.
`publisher: archive` retains the earlier Docker-archive/Skopeo path. Choose and
verify the supported toolchain before enabling imports, rather than treating a
local image tag or RepoDigests entry as a publication receipt.

To enable Python dependency builds, explicitly set
`allow_dependency_builds: true` with `publisher: buildkit` in its JSON config.
Only this recipe enables build-container networking to PyPI; source-only
packaging retains `--network=none`. The reviewed installer uses binary wheels,
base-framework constraints, `pip check`, and a version manifest. Requirements
cannot supply indexes, URLs, includes or commands. Registry credentials stay
with the publisher and are not passed into installation commands.

Set `container_execution.environments_file` to an operator-owned YAML file:

```yaml
environments:
  torch-example:
    title: Approved PyTorch environment
    image: registry.example.org/team/base@sha256:<64 hex characters>
    versions: {torch: "operator-verified version", cuda: "operator-verified version"}
    validation: {wyd-l40s: "not-tested", sensecore-1gpu: "not-tested"}
```

The API exposes only approved public catalogue fields. Selecting an ID freezes
its current image digest; later catalogue edits do not rebind a Runtime.
Record actual GPU validation separately from successful image publication.

Run the builder under its own systemd unit with its writable state and runtime
directory. Do not apply the daemon's restrictive UID or sandbox blindly to
this root/Docker worker. Runtime/source permissions and allowlist entries are
part of the operator's packaging trust boundary.

### Object store and public return path

Provision a bucket in an S3-compatible service, then edit
[artifact-store.json](../server/examples/artifact-store.json). A loopback Garage
instance is one supported deployment; an existing S3 service is another.
The daemon needs boto3 (included in the server distribution) and bucket read/
write access. Keep these credentials out of source, Runs, argv and responses.

`public_transfer_base` must be reachable over **HTTPS from compute workers**,
including the reverse-proxy prefix and `/api/artifact-transfers`. Server S3
credentials are not sent to the worker. Dispatch issues a seven-day write-only
capability for one exact Attempt. The worker PUT route validates that capability
instead of the control-plane token. Configure the proxy to pass the PUT body
and allow the configured size (the template uses 256 MiB). Source uploads have
their own smaller 64 MiB default. Set request/upload timeouts for real network
conditions. A single-node object store is not an off-host backup.

## Serve HTTPS and enable capabilities deliberately

Run the daemon as its dedicated account on loopback, using the installed runtime:

```bash
/opt/ml-expd/venv/bin/ml-expd --config /etc/ml-expd/ml-expd.yaml \
  --host 127.0.0.1 --port 8765
```

A systemd unit should specify `User`, `Group`, a readable `WorkingDirectory`,
the absolute `ExecStart`, PATH for backend tools, `Restart=on-failure` and the
workspace's writable paths. Keep native SSH/SCO credentials in that account's
home. Start the builder before testing packaging. The daemon's own collector
runs immediately and polls every 20 seconds by default. `--snapshot` disables
live collection and is not the normal remote experiment mode.

Ready-to-edit unit templates are [ml-expd.service](../server/examples/ml-expd.service)
and [ml-expd-image-builder.service](../server/examples/ml-expd-image-builder.service).
They assume the account, credentials and source directory have already been
provisioned. Install them into `/etc/systemd/system`, copy `source-api.yaml` as
`/etc/ml-expd/ml-expd.yaml`, update numeric builder UID/GID, then reload systemd
and enable/start the builder and daemon. Adapt tool paths and resource limits
to your host. These are examples for a new deployment, not drop-ins to overwrite
an existing service.

A minimal reverse-proxy location is:

```nginx
location /ml-expd/ {
    client_max_body_size 256m;
    proxy_read_timeout 1300s;
    proxy_send_timeout 1300s;
    proxy_pass http://127.0.0.1:8765/;
    proxy_set_header Host $host;
    proxy_set_header Authorization $http_authorization;
    proxy_set_header X-ML-Expd-Client-Protocol $http_x_ml_expd_client_protocol;
    proxy_buffering off;
}
```

This strips `/ml-expd/` while the client retains that prefix in its base URL.
Install TLS on the public listener. Native remote binding instead requires
both `--ssl-certfile` and `--ssl-keyfile` plus Bearer auth; plaintext remote
binding is refused. Never remove auth to make the stock Swagger UI work; fetch
the authenticated schema as described in [the HTTP contract](http_contract.md).

With mutations still disabled, run the read-only operator checklist. Run the
`ml-exp` check from a separate machine/environment with the client installed:

```bash
/opt/ml-expd/venv/bin/ml-expd --config /etc/ml-expd/ml-expd.yaml doctor --json
ml-exp check --schema openapi.json
```

Doctor verifies generic host availability and policy; it does not validate all
profile details or prove a specific Run can submit. After configuration and
connectivity checks, enable `allow_source_imports` and `allow_project_writes`
to test packaging and Run creation. Enable `allow_scheduler_mutations` only
when the budget/profile policy is ready for actual execution. Each submitted
Run still has prepare/authorize/execute gates. Restart after config changes;
profile changes do not rebind existing immutable Runs.

## Acceptance and maintenance

Use a **new** project/Run definition for acceptance. Run the quickstart first,
then a small real training task on each intended executor. Verify independently:

1. Source import and Runtime `READY` with a remotely verified image digest.
2. Submission gates and `VERIFIED` with the exact backend job ID.
3. Exact Attempt reaches `SUCCEEDED`; inspect logs and numeric metrics.
4. Artifacts arrive through the public HTTPS return path and S3.
5. Download files and tar, verify receipt SHA256 and contents, check your
   program's Run/Attempt/source identity and scientific outputs.

Injected local tests establish API/client and recovery behavior, not scheduler,
GPU or off-host network availability. Do not report submission preflight alone
as an executed experiment. Preserve old Attempts, histories and rollback
runtimes; no automatic artifact deletion is implemented. Monitor health and
Action resolution; reconcile uncertain effects with observation, never blind
replay. Stop the sole daemon for consistent workspace backups and test restores
in an isolated location. W&B is absent from the current implementation.
