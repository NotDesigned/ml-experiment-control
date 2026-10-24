# Documentation

Start with the guide for your role. These documents describe the repository's
implemented behavior; `/api/health` and the installed `/api/v2/openapi.json`
identify what a particular deployment provides. Credentials, site inventories
and deployment evidence belong in private operator records.

## Use an existing service

1. [Client quickstart](api-quickstart.md): the complete install-to-results path.
2. [Data](data.md): target-backed uploads, resume, download scripts and limits.
3. [Persistent checkpoints](persistent-checkpoints.md): complete state and explicit restore.
4. [Metrics](metrics.md): curves, comparison results, units and protocol identity.
5. [W&B](wandb.md): optional server publication and per-Run destinations.
6. [Recovery](recovery.md): failed stages, uncertain requests and output collection.
7. [Debug sessions](debug-sessions.md): temporary CCI/DSW SSH machines.
8. [Administrative deletion](resource-deletion.md): reviewed Run and storage removal.

The [client package README](../client/README.md) provides a command map and Python
API. No research-question entity is required; Campaign is optional grouping.
Scientific evaluation is an ordinary Run with declared inputs and metrics.

## Provide the service

- [Operator guide](operator-guide.md): install, configure and verify a release.
- [Configuration](configuration.md): client settings, credentials and independently enabled backends.
- [Two-machine example](two-machine-quickstart.md): a basic HTTPS API and separate client without a training backend.
- [Slurm](slurm.md): SSH scheduling, shared storage and offline compute nodes.
- [PAI](pai.md): DLC training, CPFS/OSS and an optional DSW builder.
- [SenseCore](sensecore-rest.md): REST, cluster choice and job diagnostics.
- [Builds](builds.md): Dockerfile rules, publishers, cache and capacity.
- [Action storage](action-storage.md): persistence, backup and offline migration.
- [Project lifecycle](project_lifecycle.md): registration, pause and archive.

## Integrate or develop

- [API resource reference](source-api.md): low-level resources, identities and diagnostic endpoints.
- [HTTP contract](http_contract.md): auth, protocol 2, asynchronous operations and errors.
- [Backend capabilities](backends.md): matching and server preparation contracts.
- [Architecture](architecture.md): packages, module boundaries and execution locations.
- [Development](development.md): tests, coverage, CI and offline bundles.
- [Core integration](library-integration.md) and [downstream contract](downstream_contract.md).
- [Compatibility and upgrades](architecture.md#compatibility-and-upgrades).
- [Generated redactor CLI reference](cli_reference.md).
