# Public core contract

The stable root exports are listed in `experiment_control.__all__`. Hosts
may import those names from their defining modules. The backend composition
interfaces below are also supported in `backends`, `backends.base` and
`backends.services`. Private names beginning
with `_` and unexported implementation helpers are not compatibility promises.

## Supported boundaries

| Surface | Purpose |
|---|---|
| `Backend`, `BackendRegistry`, `build_registry`, `BackendServices` | Compose `local`, `slurm` and `sensecore` backends |
| `ProjectAdapter`, `ProjectRegistry`, asset/source types | Host-owned configuration, science and filesystem probes |
| `RunSpec`, `AttemptManifest`, submission/status/log/record types | Typed JSON/YAML-compatible mappings |
| `ExperimentStateStore`, lifecycle/state and manifest helpers | Immutable definitions, events and durable intents |
| `CommandRunner`, `CommandResult`, `SubprocessRunner` | Inject command execution |
| Preflight, identity, checkpoint and cancel-outbox helpers | Check ownership and reconcile effects |

Mapping contracts are `TypedDict` structures, not runtime model objects.
Campaign/project summaries remain open to host-owned fields. Backend reports
must expose sanitized evidence, not raw account responses or commands.

Every backend provides read-only `availability()` for local tools/generic
authentication. Run-specific checks belong in `preflight(run, scope=...)`.
An observation-only check must not require staging/submission or create jobs.

## Exact submission and cancellation

Call `begin_submission` with `backend.submission_request` before a non-dry-run
`submit`. Pass its unchanged intent to recovery and submit. The persisted random
128-bit token is bound to the Attempt: Slurm Comment, exact ACP resource name
or local process claim. Recover first; reconcile an existing exact job without
creating another. Missing, foreign or ambiguous evidence fails closed.

Uncertain creates require status-only reconciliation. A legacy pending intent
without a token cannot be silently upgraded into ownership proof. Legitimate
retry needs a terminal prior execution and a new unused Attempt. Cancellation
must use the durable outbox and exact owned scheduler/process identity.

Status can contain `reason`, `detail`, `observed_at` and `observation_source`.
Populate them from evidence. Slurm `%R` supplies a pending reason only for a
pending/configuring job; a running node list is not a queue reason. Historical
logs, poll timestamps and scheduler RUNNING are not fresh training heartbeats.

## Redactor and collection

`experiment-redact` reads log text from stdin, without a subcommand. Its
generated [CLI reference](cli_reference.md) is authoritative.
`EXPERIMENTCTL_REDACTOR_BIN` selects another reviewed executable. Legacy names,
mode arguments and fallback variables are unsupported. Redactor failure must
never echo raw input.

`BackendServices.collection_includes` optionally supplies host collection
patterns; managed Runs declare `outputs`. Backend collection has no required
project-specific scientific metric names. Scientific parsing and completeness
rules belong to the host or [per-project/per-Run metrics schema](metrics.md).
No current W&B SDK, publishing or history synchronization is provided.

## Consumer upgrades

Pin immutable commits. Install a candidate including its Rust binary, run the
consumer's public-import, adapter, state/outbox and redactor integration tests,
then advance the pin. Update this contract when public ownership/invocation
changes; do not preserve unused private helpers as compatibility aliases.
The [local integration example](library-integration.md) and repository checks
supplement, rather than replace, tests of the consuming controller.
