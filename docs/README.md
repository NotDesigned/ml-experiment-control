# Documentation

The default implementation on `main` is native Rust. See the
[latest release](https://github.com/NotDesigned/ml-experiment-control/releases/latest)
for the current version. Use the installed `--help` and `/api/v2/openapi.json` for
exact contracts; `/api/health` identifies what a particular deployment provides.
Credentials, site inventories and deployment evidence belong in private operator
records.

## Use an existing service

1. [Client quickstart](api-quickstart.md): the complete install-to-results path.
2. [Native usage guide](../rust/README.md): further commands, configuration shapes and batches.
3. [Client commands](cli_reference.md): command groups and recovery examples.
4. [Data](data.md): target-backed uploads, resume, download scripts and limits.
5. [Persistent checkpoints](persistent-checkpoints.md): complete state and explicit restore.
6. [Metrics](metrics.md): curves, comparison results, units and protocol identity.
7. [W&B](wandb.md): optional server publication and per-Run destinations.
8. [Local metrics](local-metrics.md): attach an existing local JSONL stream through the native client.
9. [Recovery](recovery.md): failed stages, uncertain requests and output collection.
10. [Debug sessions](debug-sessions.md): temporary CCI/DSW SSH machines.
11. [Administrative maintenance](resource-deletion.md): reviewed removal of outputs, state and Run registrations.

No research-question entity is required. Scientific evaluation is an ordinary Run
with declared inputs and metrics.

## Provide the service

- [Operator guide](operator-guide.md): install, configure and verify a release.
- [Native deployment guide](../rust/README.md#deploying-a-native-service): configuration shapes and credentials.
- [Configuration](configuration.md): client settings and independently enabled backends.
- [Slurm](slurm.md): SSH scheduling, shared storage and offline compute nodes.
- [PAI](pai.md): DLC training, CPFS/OSS and an optional DSW builder.
- [SenseCore](sensecore-rest.md): REST, cluster choice and job diagnostics.
- [Builds](builds.md): Dockerfile rules, publishers, cache and capacity.
- [Project lifecycle](project_lifecycle.md): registration, pause and archive.

## Integrate or develop

- [API resource reference](source-api.md): resources, identities and diagnostic endpoints.
- [HTTP contract](http_contract.md): auth, protocol 2 and errors.
- [Backend capabilities](backends.md): matching and server preparation contracts.
- [Architecture](architecture.md): crates, module boundaries and execution locations.
- [Compatibility and upgrades](architecture.md#compatibility-and-upgrades).
- [Acceptance status](rust-rewrite.md): what has and has not passed on real providers.
- [Platform checks](platform-checks.md): requirements, limits and policies.
- [Development](development.md): tests, CI and releases.

## Retained Python reference

These describe retained Python packages and migration; their commands and
configuration are not the native interface.

- [Two-machine example](two-machine-quickstart.md): retired Python API and client install.
- [Core integration](library-integration.md) and [downstream contract](downstream_contract.md): Python core library.
- [Action storage](action-storage.md): storage notes and offline migration commands.
- [Client package README](../client/README.md): retained Python client.
