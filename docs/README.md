# Documentation

These pages describe current code, not deployment history. Check `/api/health`
and the installed OpenAPI schema for your server. Site credentials, inventories
and acceptance reports belong in private operator configuration/evidence.

## Client: use an existing API

1. [Quickstart](api-quickstart.md): connect, build, submit and download.
2. [Data](data.md): upload/resume, NAS delivery, download scripts and limits.
3. [Checkpoints](persistent-checkpoints.md): persist complete state and restore it.
4. [Metrics](metrics.md): units, completeness and scoring provenance.
5. [W&B](wandb.md): optional per-Run routing and confirmed publication.
6. [Recovery](recovery.md): diagnose a failed stage without duplicating work.

## Operator: provide the service

- [Deployment](operator-guide.md): installation, configuration and verification.
- [Builds](builds.md): Dockerfile admission, desktop builder, cache and capacity.
- [SenseCore](sensecore-rest.md): REST, cluster selection, startup and timeouts.
- [Action storage](action-storage.md): database, backup and offline migration.
- [Project lifecycle](project_lifecycle.md): registration, pause and archive.

## Developer and API integrator

- [Architecture](architecture.md): packages, execution locations and ownership.
- [Backend capabilities and preparation](backends.md): matching, server workflow and data diagnostics.
- [API workflow](source-api.md): source, Runtime, Run, Submission and result routes.
- [HTTP contract](http_contract.md): authentication, protocol, async execution and SSE.
- [Development](development.md): checks, CI and documentation maintenance.
- [Core integration](library-integration.md) and [public contract](downstream_contract.md).
- [Generated redactor CLI reference](cli_reference.md).

Start with one experiment. Data preparation, checkpoint persistence and custom
metrics are optional additions. No research-question entity is required;
Campaign is optional user-facing grouping. Designs and one-off operator tools
are explicitly distinguished from implemented public APIs in recovery.
