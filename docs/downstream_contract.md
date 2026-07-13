# Downstream Integration Contract

`ml-experiment-control` is consumed by host repositories such as ELF. This
document defines the supported integration surface; internal module contents
are not implicitly downstream APIs.

## Supported Python surface

The package root exports the stable primitives in `experiment_control.__all__`.
Hosts may also import those same names from their defining modules. The primary
integration points are:

- `experiment_control.backends.build_registry` and
  `experiment_control.backends.services.BackendServices` for backend composition;
- the registered `local`, `sensecore`, and `slurm` backend kinds, with `local`
  available as the runnable Linux development and smoke-test backend;
- `Backend`, `ProjectAdapter`, `BackendRegistry`, and `ProjectRegistry` for typed
  host dispatch;
- `RunSpec`, `AttemptManifest`, `BackendStatus`, `BackendLogs`,
  `AssetVerification`, submission/record types, and JSON value aliases for the
  serialization-compatible host/backend boundary;
- `ExperimentStateStore`, `LifecycleStatus`, `RunState`, `append_event`,
  `atomic_write`, `sanitize_command`, `utc_now`, `validate_identity`, and
  `require_immutable` for durable state, durable file updates, and validated
  identity construction;
- `CommandResult`, `CommandRunner`, and `SubprocessRunner` for injected command
  execution;
- `PreflightCheck`, `PreflightReport`, `IdentityReport`, asset/source types,
  checkpoint discovery, manifest construction, and cancel-outbox functions
  exported from the package root.

Names beginning with `_` are private. A host must not import them, even when a
commit pin makes the current implementation appear stable. Module constants or
classes omitted from both `experiment_control.__all__` and the README Public API
section are likewise not compatibility promises.

The mapping contracts are `TypedDict` definitions rather than runtime model
objects: callers continue to persist ordinary JSON/YAML dictionaries, while a
type checker can reject missing required identity fields and incompatible
backend result shapes. Campaign and project summary payloads remain open
because their scientific fields are owned by the host project.

## Submission identity

Hosts must call `ExperimentStateStore.begin_submission(...)` with the selected
backend's `submission_request(...)` before invoking a non-dry-run `submit`.
The returned durable `SubmissionIntent` contains the generated
`submission_token` and must be passed unchanged to both `recover_submission`
and `submit(..., intent=intent)`. Recovery is attempted first; a returned job
ID is reconciled without another scheduler mutation. If recovery returns
`None`, the same intent is used for the first submit and the returned job ID is
then persisted with `reconcile_submission(...)`.

Backends reject non-dry-run calls without a valid intent. An unreconciled
legacy submission record without a token cannot be upgraded automatically,
because no local value can prove ownership of an already-created remote job;
hosts must stop and reconcile it manually.

`BackendStatus` includes optional `reason`, `detail`, `observed_at`, and
`observation_source` fields. Backends should populate them only from observed
scheduler evidence. In particular, a WYD `PENDING`/`CONFIGURING` row may expose
Slurm `%R` as `reason` and `detail.pending_reason`; an allocated node list for a
running job is not a pending reason.

## Sanitizer CLI

`experiment-safe-sco` is a packaged Rust executable installed into the Python
environment's `PATH`. Hosts must invoke the executable directly. The historical
`python -m experiment_control.safe_sco` entrypoint is not part of the supported
surface.

The generated [`cli_reference.md`](cli_reference.md) is the command contract.
Sanitizer failures never echo raw input.

## Consumer upgrade procedure

Consumers should pin an immutable package commit. Before updating that pin:

1. install the candidate package, including its platform wheel or Rust build;
2. run the host's registry, project-adapter, state-store, and CLI integration tests;
3. remove imports absent from the supported surface instead of requesting
   compatibility for unused implementation details;
4. update consumer documentation when module ownership or invocation changes;
5. update the commit pin only after both package and host tests pass.

ELF is the reference downstream consumer. Its integration tests should exercise
the public imports and direct `experiment-safe-sco` invocation against the
candidate package before advancing `requirements.txt`.
