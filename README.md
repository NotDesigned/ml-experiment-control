# ML Experiment Control

`ml-experiment-control` is a small Python package for controlling durable ML
experiments without coupling scheduler code to a particular training
repository. It currently provides SenseCore/SCO and WYD Slurm/Apptainer
backends, durable Run/Attempt state and mutation outboxes, sanitized readiness
checks, command-runner injection, normalized failure states, and the
`ProjectAdapter` protocol used by a host repository.

The package owns the platform-neutral Run/Attempt manifest constructor,
`ExperimentStateStore`, atomic lifecycle events, and submission/cancel
outboxes. It deliberately does **not** own scientific configuration, training
commands, metric semantics, campaign authoring, credentials, or model assets.
A host repository supplies those through a project adapter and injected
backend services.

## Install

Add the package to another uv-managed project directly from GitHub:

```bash
uv add \
  "ml-experiment-control @ git+https://github.com/NotDesigned/ml-experiment-control.git"
```

For local development:

```bash
git clone https://github.com/NotDesigned/ml-experiment-control.git
cd ml-experiment-control
uv sync --locked
uv run python tools/coverage_gate.py
```

`uv sync` creates the local `.venv`, installs the package in editable mode, and
includes the default `dev` dependency group. `uv.lock` is committed so local
development and CI resolve the same dependency versions. Use `uv add <package>`
for runtime dependencies and `uv add --dev <package>` for development tools.
Building from source also requires Rust 1.85 or newer; maturin compiles and
installs the `experiment-safe-sco` sanitizer binary into the environment.

The only runtime dependency outside the Python standard library is PyYAML,
used for canonical Run/Attempt manifest files.

## Public API

```python
from experiment_control.backends import build_registry
from experiment_control.backends.services import BackendServices
from experiment_control.preflight import PreflightCheck, PreflightReport
from experiment_control.identity import IdentityReport
from experiment_control.project import (
    AssetProbe,
    AssetRequirement,
    ProjectAdapter,
    ProjectRegistry,
    SourceBundle,
)
from experiment_control.runner import CommandResult, CommandRunner, SubprocessRunner
from experiment_control.states import FailureClass, classify_failure
from experiment_control.manifest import (
    ExperimentStateStore,
    RunState,
    append_event,
    atomic_write,
    require_immutable,
    sanitize_command,
    utc_now,
    validate_identity,
)
from experiment_control.outbox import execute_cancel_outbox
from experiment_control.run_manifest import build_run_manifest
```

`build_registry` requires one host-provided `BackendServices` instance:

```python
services = BackendServices(...)
registry = build_registry(services)
```

`BackendServices` is the narrow host boundary. The host injects command
execution, run-directory lookup, backend-record lookup, project metric parsing
and summarization, atomic JSON writing, and UTC time. This keeps the package
independent from the host's campaign format, scientific config, and metric
semantics; package-owned manifest and outbox primitives remain usable without
backend services.

The supported Python symbols, Rust CLI boundary, private-name policy, and
consumer upgrade checklist are defined in
[`docs/downstream_contract.md`](docs/downstream_contract.md). Imports whose
names begin with `_` are implementation details and are not downstream APIs.

## Backend lifecycle

Each backend implements:

```text
validate -> preflight -> identity -> stage -> render -> submit
                                          -> status/logs/collect/cancel
```

`preflight` returns a credential-free `PreflightReport`. SenseCore checks the
SCO executable and a sanitized exact-name workspace query. WYD scopes checks
to the operation: observation needs only SSH/Slurm control access, staging adds
rsync and storage, and submission adds live partition/GRES, account/QOS,
Apptainer, and mount validation.

`identity(campaign, run, attempt_id)` returns a typed, sanitized, read-only
`IdentityReport` containing `available`, `ambiguous`, `scheduler_job_ids`, and
`remote_manifest_exists` when the backend can inspect persistent storage.
WYD additionally reports `remote_manifest_matches`; a host may claim an
existing run directory only when this digest-backed ownership evidence is
true.
Recovery fails closed when more than one scheduler job matches one attempt; it
never selects one arbitrarily.
Campaign generation and scientific config resolution remain host
responsibilities. The package owns immutable Run/Attempt publication and local
event/outbox reconciliation once the host has resolved those inputs.

WYD log observation checks exact, attempt-qualified canonical `stdout.log` and
`stderr.log` paths first, then exact `slurm-<job-id>.out/.err` paths in the
attempt and run directories. It does not use remote globs. Returned tails are
bounded and redacted, and `collect()` includes the same sanitized excerpts as
`process_evidence` so failures before the training runtime writes metrics remain
diagnosable by a host controller.

SenseCore creates and observes an exact attempt-qualified resource name. The
authored image tag remains provenance, while submission is pinned to the
manifest's `repository@sha256:...` digest. Status, logs, and cancellation use
the recorded exact resource rather than a prefix search.
Collection also sanitizes the exact worker table, discards host and pod IPs,
and normalizes `Pending`, `Running`, and terminal phases separately from model
progress.

WYD submission also acquires an attempt-qualified directory claim on persistent
storage before copying the manifest or invoking `sbatch`. The claim is not
removed automatically: a controller crash therefore fails closed for manual
reconciliation instead of permitting a second scheduler mutation.

Credentials remain in native providers:

- SCO profile/config for SenseCore;
- SSH config/agent for WYD;
- Docker credential store/helper for registries.

Never pass credentials through campaign YAML, manifests, backend reports, or
training commands.

## CLI reference

The package exposes one deliberately narrow Rust command, `experiment-safe-sco`,
for sanitizing SCO responses before a controller parses them. The binary is
installed by the package wheel and is not an experiment lifecycle CLI. Its
complete generated option reference is in
[`docs/cli_reference.md`](docs/cli_reference.md); the runtime parser is the
single source of truth and CI rejects stale generated help.

## Project integration

A training repository implements `ProjectAdapter` to define config resolution,
the launch command/environment, assets, metric and checkpoint-log parsing,
summaries, and source staging policy. The controller composes one
`ProjectAdapter` with one compute backend.

Copy [`examples/minimal_project_adapter.py`](examples/minimal_project_adapter.py)
into the training repository and replace its JSON config, command, metric,
checkpoint, asset, and summary conventions. The adapter deliberately stays in
the training repository: this package must never import model code.

Start with these methods in order:

1. `validate_run`, `resolve_config`, and `command` freeze what will run.
2. `environment` and `operational_overrides` map controller-owned output paths.
3. `source_bundle`, `plan_assets`, and `asset_probes` define immutable inputs.
4. `parse_metric`, `parse_checkpoint`, and `summarize` define durable evidence.

The example is executable and covered by the package test suite. It is a
contract template, not a universal training launcher; backend scheduling and
project semantics remain intentionally separate.

Asset discovery follows the same boundary. A project adapter returns semantic
requirements and maps them to backend-verifiable filesystem probes:

```python
requirements = adapter.plan_assets(config_path, overrides)
probes = adapter.asset_probes(requirements, runtime_environment)
report = backend.verify_assets(run, probes)
```

`AssetRequirement.identity` is the immutable model, dataset, or file identity;
`AssetProbe.path` is its backend-visible representation. Backends never need to
know model or dataset names.

A project's `SourceBundle.required_paths` may declare relative files that must
exist in the staged source tree. WYD verifies those paths after transfer. The
project launcher remains responsible for importing its runtime dependencies as
its first container-side action; some login nodes cannot mount SIF images
without an allocation, so the backend does not pretend to run a container
preflight there.

## Tool overrides

The backends recognize these non-secret environment variables:

```text
EXPERIMENTCTL_SCO_BIN
EXPERIMENTCTL_SCO_CREATE_TIMEOUT_SECONDS
EXPERIMENTCTL_SAFE_SCO_BIN
EXPERIMENTCTL_SSH_BIN
EXPERIMENTCTL_RSYNC_BIN
```

Each controller process uses its own SSH multiplexing socket. Parallel agents
therefore reuse connections within their own workflow without racing over one
global `/tmp` control socket.

Registry publication remains a host workflow. A host may define its own Docker,
Crane, or Skopeo executable overrides without adding them to this package.

## Safety properties

- Backend reports contain fixed, sanitized messages rather than raw shared
  workspace responses.
- SenseCore JSON is piped through the packaged sanitizer before parsing.
- Preflight failure is fail-closed before remote stage or scheduler submit.
- Consumed or ambiguous scheduler identities are rejected before submission.
- Completed checkpoints require a matching payload, step, and byte-count marker.
- Scheduler state, model progress, and scientific conclusions remain separate.
- Source, image, config, storage, and scheduler identities remain the host
  manifest's responsibility.

## Development verification

```bash
uv sync --locked
cargo fmt --manifest-path rust/Cargo.toml -- --check
cargo clippy --locked --manifest-path rust/Cargo.toml -- -D warnings
cargo test --locked --manifest-path rust/Cargo.toml
uv run python tools/coverage_gate.py
uv run python tools/generate_cli_reference.py --check
uv run python -m compileall -q src tests tools examples
uv build
```

The repository enforces 100% line coverage and at least 95% branch coverage as
independent gates. Testing policy and CI details are documented in
[`docs/development.md`](docs/development.md).
