# Deploy and maintain the service

API users start with the [quickstart](api-quickstart.md). This page is for the
operator of a Linux x86_64 deployment and summarizes the order of work. The
[native deployment guide](../rust/README.md#deploying-a-native-service) has the
configuration shapes. Templates contain placeholders and disabled mutations;
copying them alone does not produce a runnable GPU service.

## Components and layout

| Component | Access it needs |
|---|---|
| `ml-expd-rs` daemon | Dedicated service account, writable registry and Run roots, API token; selected backend/store credentials for training |
| Reverse proxy | Terminates public HTTPS and forwards to the loopback daemon |
| Optional image builder | Private socket, registry push auth; a separately configured PAI CPU DSW or local Docker/BuildKit. Separate Python process; see [below](#image-builder) |
| Slurm | Pinned SSH to a networked gateway, `ml-exp-worker-rs` helper, Slurm/Apptainer, shared storage, registry pull auth |
| PAI DLC | Access key file, workspace/quota, personal CPFS/role, registry pull auth, private OSS prefix |
| SenseCore | Access key file, workspace/pools/specs, CCR pull authorization, NAS; separate debug CPU copier and collector |
| Optional W&B | Separate interpreter with the official SDK, private API key file |

Keep one daemon writer per registry. The examples use this layout:

```text
/etc/ml-expd/native/          private deployment file, reads, backends, credentials
/var/lib/ml-expd-native/      registry root
/srv/ml-expd-native/          approved Project files and Run roots
```

Adjust the [API](../rust/deploy/ml-expd.service) and
[builder](../rust/deploy/ml-expd-builder.service) systemd examples so their binary,
writable and read-only paths match. They install nothing by themselves. Do not relocate
backend datasets or checkpoints when rearranging program files.

## Install a native release

From the [latest release](https://github.com/NotDesigned/ml-experiment-control/releases/latest),
download `SHA256SUMS` and `ml-expd-native-X.Y.Z-x86_64-unknown-linux-gnu.tar.gz`,
verify the archive's hash, and extract it. The bundle contains `bin/` (`ml-exp`,
`ml-expd-rs`, `ml-exp-worker-rs`), `deploy/` examples and
the upgrade tool, `python/` (the image bridge sources), `README.md`, `MIGRATION.md`,
`native-release.json` and `release-info.json` (source commit and bridge hashes). It
contains no secrets and creates no service. No compiler or Python is needed to run
these binaries.

The server and client target requires glibc 2.34 or newer. The worker is statically
linked with musl so it does not depend on the training image's glibc. Install
clients separately on client machines; see the [quickstart](api-quickstart.md).

## Configure

Copy [server.example.json](../rust/deploy/server.example.json),
[reads.example.json](../rust/deploy/reads.example.json) and
[project.example.json](../rust/deploy/project.example.json) to the configured
paths, and create the empty Run directory. All paths are absolute. Replace every
placeholder. This minimum serves the API and source import with no backend
credential, builder or object store.

Create the API token as a regular file owned by the service account, mode 0600 or
0400, with at least 32 non-whitespace random characters, without printing it to
shared output. Distribute it privately. There is no per-user or per-project RBAC.
All native credential files follow the same ownership and mode rule; do not relax
it for files copied from an older deployment.

The daemon binds loopback only and refuses anything else. Terminate TLS in a
reverse proxy that preserves `Authorization`, protocol headers and the API prefix,
using a certificate the [client trusts](configuration.md#connection-errors).

Run these as the service account that will own the stores:

```sh
ml-expd-rs --config /etc/ml-expd/native/server.json --check
ml-expd-rs --config /etc/ml-expd/native/server.json --initialize
ml-expd-rs --config /etc/ml-expd/native/server.json
```

`--check` reads local configuration, credentials and Project definitions. It sends
no provider request, and `provider_connectivity_checked=false` is deliberate.
`--initialize` creates all selected stores in a completely new registry, with no
provider jobs. It refuses an existing or partial registry and never repairs, resets
or deletes one. **Never initialize a registry over existing state.** Ordinary
startup does not initialize missing stores.

A `registry_root` holding the retired `.rust-preview-workspace.json` marker refuses
every command with `registry_predates_clean_restart`. Set `registry_root` to a new
path and run `--initialize`. Nothing in the old registry is read or changed.
`--initialize` also refuses a Project run root that is not empty, with
`project_run_root_not_empty`, because a new registry would discover what it holds.

## Upgrade a release

`deploy/deploy_native.py` in the bundle upgrades an installed server. It needs only
the system Python and reads `ROOT/deployment.json`
([example](../rust/deploy/deployment.example.json)), which names the units, private
configuration files, builder state and publisher locks. It expects `ROOT/bin/` for
the executables and installs the bridge into the builder's `python_site_packages`.

```sh
python3 deploy/deploy_native.py --root /opt/ml-expd --version X.Y.Z          # verify only
python3 deploy/deploy_native.py --root /opt/ml-expd --version X.Y.Z --apply
python3 deploy/deploy_native.py --root /opt/ml-expd --rollback
```

Always run the copy from the bundle being installed: a copy from a release that
still shipped `ml-expd-state-audit` refuses newer bundles, which no longer contain
it. The retired executable left in `ROOT/bin/` is not touched and can be deleted.

Without `--apply` it changes nothing. It verifies every hash in the bundle, runs the
new `ml-expd-rs --check` against the live configuration, imports the new bridge with
the builder's interpreter, and refuses a downgrade or an active build (a builder
holds `<identity>.lock` for the whole build). Also confirm that no transfer or
submission you care about is mid-request; the tool does not inspect those.

With `--apply` it keeps the installed files in `ROOT/previous/`, stops the API and
requires exit status 0, stops the builder, takes the publisher locks, replaces the
files, sets `builds.worker_sha256` to the new worker, and starts both services. If
the API does not report the new version on `/api/health`, it restores the previous
files and worker hash and restarts them. On success it writes `ROOT/DEPLOYED.json`
with the version, source commit and hashes; read that file to learn what is running.

The tool replaces programs only. It does not touch registries, Runs, Attempts,
provider jobs or remote Slurm helpers, and rollback restores no database. New builds
embed the new worker, so they get new Runtime identities; existing Runtimes and images
keep their frozen worker. An unpublished Runtime prepared with the old worker needs
fresh preparation. One deployment runs at a time (`ROOT/.deploy.lock`).

## Enable training after source import

A successful source import completes code registration. The **service operator**
prepares the rest in this order; researchers only need the API connection and an
experiment file.

1. **Choose one backend and its persistent storage.** Add it through
   [the matching example](configuration.md#enable-only-the-backends-you-use), with
   credentials, shared data/Run paths and pull authorization. Follow its
   [Slurm](slurm.md), [SenseCore](sensecore-rest.md) or [PAI](pai.md) preconditions.
   Confirm the compute nodes can access those paths; API access alone does not
   prove this. Slurm helpers must be installed and their stores initialized once on
   the gateway; enabling an executor installs nothing remote.
2. **Configure results and worker callbacks.** Cloud backends need `exchange` with
   a public HTTPS URL that workers can reach and a private signing key of at least
   32 random bytes, plus `artifacts` for the private object prefix. Keep persistent
   training state separate from temporary result parts.
3. **Prepare the image builder and registry.** See [image builder](#image-builder).
   Slurm may instead use an existing trusted Runtime receipt or the cloud builder's
   `slurm` materializer map. Verify that a bounded build reaches READY and that the
   selected backend can read its digest.
4. **Enable training deliberately.** Set finite resource limits, then `enabled` and
   `allow_run_creation`, and optionally `polling`. Cloud Run creation also needs
   `exchange`, because every worker fetches its launch package through it; `--check`
   refuses the combination without it. Run `--check` again, restart, and read
   `ml-exp get /api/readiness` from a client.
5. **Validate one bounded experiment.** The researcher uses the
   [experiment flow](api-quickstart.md#first-experiment), then checks progress, a
   checkpoint restore and downloaded result hashes with the operator. Expand to
   larger or batch workloads after that acceptance.

Client users do not need backend SDKs, cloud credentials or server settings.

## Image builder

Builds are optional and run in a separate private process, the narrow Python bridge
kept in `server/src/ml_exp_server`. The bundle carries its sources, but its
interpreter and third-party dependencies are outside the bundle, and it is the only
part of this deployment that needs them. It is started as
`python -m ml_exp_server.image_builder --config CONFIG`; see the
[unit example](../rust/deploy/ml-expd-builder.service) and the
[bridge notes](rust-rewrite.md#latest-implementation-completion-pass).

In the cloud configuration, `builds` sets the builder `socket`, its `peer_uid`, and
the `worker_sha256` of the exact native worker. Configure the publisher itself
following [builds](builds.md). The default builds on a CPU PAI DSW and pushes to the
configured registry; local BuildKit is the alternative. The daemon has no Docker
socket access. Build push credentials are separate from training pull credentials:
a builder login does not authorize a backend to pull.

After changing the worker or bridge, exercise a real build through the running
daemon. Updating a source checkout does not update what the service loads.

## Optional W&B

Point `tracking_config` in the deployment file at a private JSON file with
`schema_version` 1, `python` (an interpreter that has the official `wandb` SDK),
`api_key_file`, and optional `entity`, `project` and `enabled`. Platform metrics
work without it. See [W&B](wandb.md).

## Verify

Verify as the service account and through the public HTTPS endpoint:

1. `--check`, then `ml-exp check` for health, protocol and authentication.
2. `ml-exp backend` for the published catalogue.
3. A bounded source build through READY and its remote digest.
4. Only with explicit resource authorization: a small real run, checkpoint restore
   and downloaded file hashes on each intended backend.

A CPU run, successful submit or queue entry does not validate GPU training. Do not
claim queue ETA or free cards without provider evidence. Current acceptance limits
are in the [acceptance status](rust-rewrite.md).

`GET /api/readiness` reports `api`, `training`, `checks`, `backends`
and `submission_available_backends`. It sends no provider request. Configuration
permission, cached provider connectivity and recorded GPU acceptance are separate
fields. `training=INCOMPLETE` remains until acceptance evidence exists; that does
not by itself block submission. `GET /api/observer` reports local background observer
state.

## Existing state

The server serves only its native namespace. Older namespaces are never read or
changed. A deployment or tracking file with a key the release does not know is
refused by `--check` and at startup.

## Project registration and lifecycle

This is the operator registry for indexing and polling, not a prerequisite
research workflow. The daemon's workspace-local registry selects projects to index
and poll. Each record binds a stable identity to an absolute JSON Project file on the
daemon host (see the [example](../rust/deploy/project.example.json)), lifecycle,
source, timestamps and operator reason. A research question is not required or
loaded. Use `ml-exp project` to register and change state; `project register` takes
a daemon-host path and uploads nothing. Source import uploads to an approved Project.

The authored file owns title, Run roots and optional Campaign catalogue. Registry removal does not delete that file, source, Runs, Attempts,
datasets or artifacts, and never stops scheduler jobs.

### States

| State | Live indexing/collection | Allowed next states |
|---|---|---|
| ACTIVE | Enabled, subject to policy | PAUSED, ARCHIVED |
| PAUSED | Disabled | ACTIVE, ARCHIVED |
| ARCHIVED | Disabled; retained as history | PAUSED |

Archive requires a reason. Restore first returns to PAUSED, then explicit resume
enables ACTIVE collection. Pausing/archiving does not cancel a running job; issue
an exact-Attempt cancel separately when intended. Each workspace has its own
registry; do not run two live collectors against the same project without a
controller designed for it.

### Routes

Send normal auth/protocol headers and use the installed OpenAPI request bodies:

```text
GET  /api/project-lifecycle
POST /api/project-lifecycle/register
POST /api/project-lifecycle/PROJECT/pause
POST /api/project-lifecycle/PROJECT/resume
POST /api/project-lifecycle/PROJECT/archive
POST /api/project-lifecycle/PROJECT/restore
POST /api/project-lifecycle/PROJECT/unregister
POST /api/project-lifecycle/unregister-all
```

These operations are separately policy-gated. Unregister only changes registry
membership; it is not asset cleanup. Registration checks configured allowed
roots, regular files and protected paths. Project imports and reviewed writes
have their own durable transaction/recovery rules; do not edit managed immutable
Run files to bypass them. Registration executes no project code.

See the [API resource reference](http_contract.md#discover-import-and-build) for
managed imports.

## Maintenance and recovery

The registry's SQLite stores include `actions/` and `tracking/outbox.sqlite`. Back
up live SQLite with its backup API, or take a stopped-service copy including WAL.
Verify backups in isolation and record the matching program version and
configuration digests. Never restore an old database over newer production state:
recovery must account for frozen worker contracts and state written by newer
versions, and an old release is not an automatic rollback target.

Registry images and backend datasets and state need explicit retention. There is no
global Docker prune or automatic NAS cleanup. Reviewed removal of outputs, persistent
state or checkpoints uses [administrative maintenance](resource-deletion.md), enabled
with `administration: true`. Record exact versions and verification privately; these
docs are not a list of live jobs. Known defects and unimplemented recovery
capabilities are in [recovery](recovery.md).

Treat API disk, DSW build storage, Slurm shared storage and SenseCore NAS as distinct
capacity domains, and check each one's actual write, quota and inode state. Freeing
an API-side transfer does not free NAS training-state generations.
