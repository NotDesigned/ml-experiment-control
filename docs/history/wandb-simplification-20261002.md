# W&B simplification

Status: implemented, 2026-10-02, following the operator's choice of metrics
visualization and experiment comparison. Current contract and operator steps:
[Native experiment tracking](observability.md).

Removed the daemon-owned local dashboard process manager, sanitized log mirror,
source cursors, dual-target publication outbox/coordinator, SDK record publisher,
and historical backfill Action. Existing archives and credentials are retained.
Small protocol-v1 read tombstones support client migration without advertising
the retired capability. Scheduling and Action state remain independent.

ELF writes safe metric/config fields directly with W&B 0.28.x. ml-expd reads
Attempt identity/status and offers one explicit native `wandb sync` command.
Actual offline SDK tests cover true scientific axes, resumed sessions, new
Attempt identity, rejected extra fields, and actual WYD rsync collection filters.
A failed SDK init does not invalidate canonical metrics. No remote scheduler
submission or historical cloud publication is part of validation.

Production cloud publication was disabled before this change and no entity was
configured. Actual endpoint charts, grouping and values require a supplied
entity/project and a separately selected synthetic run. Until that acceptance,
report offline/local verification only. SenseCore native binary transfer is not
implemented by its log-only collector; see the current contract's explicit limit.

Official implementation references: [SDK lifecycle](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py),
[SDK settings](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py),
[sync CLI](https://github.com/wandb/wandb/blob/main/wandb/cli/cli.py).
Behavior is verified against installed 0.28.0, not assumed from the development branch.
