# Deploy and maintain the service

API users start with the [quickstart](api-quickstart.md). This page is for the
operator of a Linux deployment. Templates contain placeholders and disabled
mutations; copying them alone does not produce a runnable GPU service.

## Components and layout

| Component | Access it needs |
|---|---|
| Daemon | Dedicated UID, writable state, backend auth, server S3 credentials |
| Private builder | Docker/Buildx, Skopeo, registry push auth, private socket |
| WYD | Daemon-owned SSH identity, Slurm/Apptainer, shared datapool, registry pull auth |
| SenseCore | REST AK/SK, workspace/pools/specs, CCR pull authorization, mounted NAS |
| Object store/proxy | Private bucket, authenticated upload paths and HTTPS downloads |

Keep one daemon writer per workspace. For the consolidated deployment:

```text
/root/ml-expd/                 checkout and installed .venv
  .ops/                       private operations evidence
  .recovery/previous/          one verified preceding program
/etc/ml-expd/                 private config/credentials
/var/lib/ml-expd/             Action DB, index, immutable Runs and metadata
/var/lib/ml-expd-image-builder/  configured builder receipts/progress/leases
/srv/ml-expd/projects/        registered external projects, when used
```

Templates may choose a different project/builder root. Make daemon config,
builder `source_root`/socket, service writable paths and actual ownership agree.
Do not relocate backend datasets/checkpoints when rearranging program files.
Code directories/modules are 0755/0644; entrypoints executable. Keep `.git`,
`.ops`, `.recovery` mode 0700. Services use `ProtectHome=tmpfs` with a read-only
bind of `/root/ml-expd`, exposing code without opening other home directories.
The daemon must not have unrestricted Docker access.

## Install a reviewed commit

Clone into `/root/ml-expd`, check out an exact reviewed commit, install uv and
Rust 1.85+ for core wheel construction, then build/install core and server:

```bash
uv build --package ml-experiment-control
uv build --package ml-experiment-server
uv venv /root/ml-expd/.venv
uv pip install --python /root/ml-expd/.venv/bin/python \
  dist/ml_experiment_control-*.whl dist/ml_experiment_server-*.whl
/root/ml-expd/.venv/bin/ml-expd --help
```

Use these creation commands for a new installation, not over an active runtime.
Existing deployments need a staged, verified upgrade and program rollback plan.
Install the client separately on client machines. Ensure `experiment-redact`
is on the controller/service PATH. Remove installers/dev caches after verification.

## Configure

Start from [source-api.yaml](../server/examples/source-api.yaml). Provision a
daemon account and allow it to write the configured state/project roots. The
HTTP token must be regular, daemon-owned and mode 0600; create a cryptographically
random token of at least 32 non-whitespace characters without printing it to
shared output. Distribute it privately. There is no per-user/project RBAC API.

Configure these files; keep every actual secret outside Git:

| Template/config | Required choices |
|---|---|
| [executors.yaml](../server/examples/executors.yaml) | Existing scheduler IDs, resource specs and shared mount roots |
| [image-builder.json](../server/examples/image-builder.json) | Daemon peer UID/GID, matching source root/socket, publisher, repository and allowlist |
| [registry-pull.json](../server/examples/registry-pull.json) | WYD pull-only credential, distinct from publisher auth |
| [artifact-store.json](../server/examples/artifact-store.json) | S3 endpoint/bucket/auth, worker callback base, quotas and part policy |
| Private SenseCore REST JSON | See [SenseCore setup](sensecore-rest.md) |

Replace all example hosts/REPLACE values. Example artifact-store values are
explicit quotas, not immutable platform limits: remove them or use null only
after deciding capacity/retention policy. Byte quotas, if present, must be
positive integers. Set `public_endpoint` to a pathless public HTTPS S3 endpoint
and `download_url_seconds` (30–3600) for direct downloads. Preserve signed
Host/path/query through its proxy and do not log signed URL queries.

Provision publisher Docker auth as root-only
`/etc/ml-expd/build-docker/config.json` in a mode-0700 directory. Set builder
`registry_auth_file` to that file for Skopeo and its service environment
`DOCKER_CONFIG=/etc/ml-expd/build-docker` for Buildx pull/push. Keep `BUILDX_CONFIG`
under writable builder state. `ProtectHome=tmpfs` hides `/root/.docker`; a
successful root-shell login does not prove the service can authenticate. Verify
manifest reads and publication as the service inside its sandbox. Never copy
credentials into source/images or expose the whole home to fix authentication.

The approved base catalogue is an operator-owned YAML selected with
`container_execution.environments_file`:

```yaml
environments:
  torch-base:
    title: Verified training base
    image: registry.example.org/team/base@sha256:ACTUAL_64_HEX_DIGEST
    versions: {torch: OPERATOR_VERIFIED_VERSION, cuda: OPERATOR_VERIFIED_VERSION}
```

Catalogue metadata is operator-verified, not automatic package detection.
Use [builds](builds.md) to configure local or SSH desktop construction and its
data staging helper. For desktop data, artifact-store configuration additionally
sets `data_upload_storage: desktop-builder`, private `data_upload_socket`, and
the bounded CPU `data_delivery` profile described in [data](data.md).

## Backend preconditions

WYD requires SSH/known-hosts under the daemon account, Slurm, rsync, Apptainer,
valid account/QOS/GRES and storage_root beneath the mounted shared `/datapool`.
Verify access on target nodes; a node-local `/data` is not equivalent. Private
images need conversion-host pull auth. Use `apptainer_unsquash` where FUSE is
unavailable. Staging/conversion has a separate controller timeout.

SenseCore requires REST access, compatible SPOT GPU specs, CCR pull auth and NAS
permissions. Advertised fixed allocation must match actual specs. GPU jobs
exclude debug pools; data-copy jobs use the configured debug CPU pool. NAS need
not be accessible directly from the daemon when workers return over HTTPS.
Backend Internet access is separate from registry/NAS access.

After provisioning, deliberately enable the required `action_runtime` source
imports/project writes/scheduler mutations and finite GPU budget policy. Do not
enable unrelated permissions just to bypass a failed gate.

## Start and verify

Review/adapt the [daemon](../server/examples/ml-expd.service) and
[builder](../server/examples/ml-expd-image-builder.service) units, daemon config,
PATH, writable state paths and credential permissions before installing them.
Daemon and builder use `/root/ml-expd/.venv`. Keep the private socket inaccessible
to other users. Public HTTPS can proxy a loopback daemon; direct non-loopback
binding requires native TLS and Bearer auth ([HTTP contract](http_contract.md)).

Verify as the service UID and through the formal HTTPS endpoint:

1. Health/protocol/authentication and all placeholder replacements.
2. Catalogue/base digests/limits, REST/SSH read-only availability and mount scope.
3. A bounded source build through READY, remote digest and cleanup receipts.
4. Existing exact-Attempt artifacts and checkpoint metadata after upgrades.
5. Only with explicit resource authorization: a small real GPU run, checkpoint
   restore and downloaded file hashes on each intended backend.

A CPU starter, successful submit or queue entry does not validate GPU training.
Do not claim queue ETA/free cards without provider evidence.

## Maintenance and rollback

Back up live SQLite with its backup API, or take a stopped-service copy including
WAL. Keep an isolated restore check and one verified preceding program/config
recovery point. Never restore an old DB over newer production state. Refuse
program rollback if guard conditions or active/newer operations make it unsafe.
See [Action storage](action-storage.md).

Registry images, backend datasets/state and published desktop archives require
explicit retention. No global Docker prune or automatic NAS cleanup exists.
Record exact commit/package versions and verification privately; these docs
are not a list of live jobs or historical test counts. Known defects and
unimplemented recovery capabilities are in [recovery](recovery.md).
