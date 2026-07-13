# ML Experiment Control

This repository contains the `ml-experiment-control` core package and the
independently runnable `ml-experiment-server` daemon (`ml-expd`). The core
builds durable ML experiment controllers without coupling scheduler code to a
particular training repository. It provides a Linux local-process backend for
development, SenseCore/SCO and WYD Slurm/Apptainer backends, durable Run/Attempt state and
mutation outboxes, sanitized readiness
checks, command-runner injection, normalized failure states, and the
`ProjectAdapter` protocol used by a host repository.

The core package owns the platform-neutral Run/Attempt manifest constructor,
`ExperimentStateStore`, atomic lifecycle events, and submission/cancel
outboxes. It deliberately does **not** own scientific configuration, training
commands, metric semantics, campaign authoring, credentials, or model assets.
A host repository supplies those through a project adapter and injected
backend services. `ml-expd` composes project catalogs, evidence indexing,
polling, Project lifecycle, immutable operation intents, and gated Actions into
one HTTP control-plane process. Goals, conversations, model turns, research
analysis, reports, charts, and hypothesis conclusions belong to the client.

## Audience

This README is for maintainers integrating the core into a host controller and
operators running the daemon. It explains the architecture boundary, public
API, and smallest runnable integration. Research Console is a separate HTTP
client/TUI and client-side Agent loop; ELF is the reference downstream host.
Package contributors should use [`docs/development.md`](docs/development.md)
for environment setup, tests, coverage, Rust checks, and release builds.

## Run the daemon

The daemon is a separate workspace distribution so the core package stays
small for library consumers:

```bash
uv sync --locked --all-packages
uv run --package ml-experiment-server ml-expd \
  --config server/examples/ml-expd.yaml
```

`ml-expd` listens on `127.0.0.1:8765` by default, indexes registered Projects
at startup, and owns one live collector per workspace. It polls immediately
and then at `poll_interval_seconds` (20 seconds by default); `--snapshot` is
the explicit no-poll mode.
Use `/api/health` and `/openapi.json` for machine-readable health and API shape.

The daemon host owns controller/backend credentials, project checkouts, and
Action state. Research Console and other clients fetch evidence and submit
HTTP commands. Their automation loop may analyze evidence and author an
`OperationIntent`; `ml-expd` only validates and prepares that intent. A
separate Action authorization is required before the daemon changes project
files or invokes a scheduler.

The HTTP boundary is deliberately asymmetric:

- the client owns research goals, hypotheses, conversations, analysis, and decisions;
- the daemon owns Projects, Campaign/Run/Attempt evidence, polling, operation
  validation, Action authorization, backend execution, and reconciliation;
- the daemon never converts metrics or evaluation records into a scientific verdict.

Project-specific controllers are currently reached through the declared
`experimentctl` command behind one `ProjectControllerGateway`. This is a
compatibility adapter: command construction and subprocess execution do not
leak into poll or action services. Projects can migrate to native core
`ProjectAdapter` composition without changing the HTTP/client boundary.

Experiment submission is a first-class daemon lifecycle. An authored
materializing Campaign membership does not need a pre-existing Run directory:
`POST /api/experiments/{project}/{run_id}/submissions/prepare` runs the dry-run,
freezes the execution identity, and returns the review gates. Scheduler mutation
still requires separate `/authorize` and `/execute` calls on the returned
Submission. Execution is `VERIFIED` only after an exact Attempt status query
confirms the submitted backend job ID. Timeouts, missing IDs, and delayed
scheduler visibility enter `RECONCILE_REQUIRED`; `/reconcile` performs status
observation only and never repeats `submit`.

## Install

Add the package to another uv-managed project directly from GitHub:

```bash
uv add \
  "ml-experiment-control @ git+https://github.com/NotDesigned/ml-experiment-control.git"
```

The core package's only runtime dependency outside the Python standard library
is PyYAML, used for canonical Run/Attempt manifest files. The daemon
distribution adds FastAPI/Uvicorn, Pydantic, and optional OTLP export support.

## Minimal local integration

`LocalBackend` is a real asynchronous backend for Linux development and smoke
tests. It creates an attempt-qualified process group, persists PID plus process
start-time identity, captures redacted logs, records exit status, supports
recovery, and cancels only the exact recorded process group.

After cloning the repository and following the development setup, run:

```bash
uv run python examples/local_smoke.py
```

The complete, copyable integration is
[`examples/local_smoke.py`](examples/local_smoke.py). Its essential composition
boundary is:

```python
services = BackendServices(...)
backend = build_registry(services).get("local")
backend.preflight(run, scope="submit").require_ready()
intent = store.begin_submission(
    project=project,
    run_id=run["run_id"],
    attempt_id=manifest["attempt_id"],
    backend=backend.kind,
    request=backend.submission_request(campaign, run, manifest["attempt_id"]),
)
job_id = backend.recover_submission(run, intent, manifest["attempt_id"])
if job_id is None:
    job_id = backend.submit(
        campaign, run, manifest, dry_run=False, intent=intent,
    )
store.reconcile_submission(
    project=project,
    run_id=run["run_id"],
    attempt_id=manifest["attempt_id"],
    backend_job_id=job_id,
)
```

The host still owns campaign parsing and immutable manifest construction.
`ExperimentStateStore` must persist the submission intent before any non-dry-run
backend call; direct mutation without that intent is rejected. Local execution
is not a substitute for live SenseCore or Slurm integration tests.

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
from experiment_control.contracts import (
    AttemptManifest,
    BackendLogs,
    BackendStatus,
    RunSpec,
    SubmissionIntent,
    SubmissionRequest,
)
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
validate -> preflight -> identity -> stage -> render
                    -> durable intent -> recover or submit -> reconcile
                                      -> status/logs/collect/cancel
```

`preflight` returns a credential-free `PreflightReport`. SenseCore checks the
SCO executable and a sanitized exact-name workspace query. WYD scopes checks
to the operation: observation needs only SSH/Slurm control access, staging adds
rsync and storage, and submission adds live partition/GRES, account/QOS,
Apptainer, and mount validation.

Local execution checks the work directory, writable run storage, and Linux
`/proc` process identity. It launches a detached attempt process group through
a package worker and persists enough identity to reject PID reuse during status
or cancellation.

`identity(campaign, run, attempt_id)` returns a typed, sanitized, read-only
`IdentityReport` containing `available`, `ambiguous`, `scheduler_job_ids`, and
`remote_manifest_exists` when the backend can inspect persistent storage.
WYD additionally reports `remote_manifest_matches`; a host may claim an
existing run directory only when this digest-backed ownership evidence is
true.
Every new durable intent contains a random 128-bit `submission_token` generated
before scheduler contact. A non-dry-run backend submission requires that intent.
WYD writes the token into Slurm `Comment`; SenseCore derives a unique exact
resource name from it; LocalBackend writes it into its atomic process claim.
Recovery requires both the expected attempt identity and the exact token.
Historical jobs with the same human-readable base name are ignored, while
missing, malformed, or multiply matching token evidence fails closed.
Campaign generation and scientific config resolution remain host
responsibilities. The package owns immutable Run/Attempt publication and local
event/outbox reconciliation once the host has resolved those inputs.

WYD log observation checks exact, attempt-qualified canonical `stdout.log` and
`stderr.log` paths first, then exact `slurm-<job-id>.out/.err` paths in the
attempt and run directories. It does not use remote globs. Returned tails are
bounded and redacted, and `collect()` includes the same sanitized excerpts as
`process_evidence` so failures before the training runtime writes metrics remain
diagnosable by a host controller.

WYD queue observation also records Slurm's `%R` pending reason in
`BackendStatus.reason` and `detail.pending_reason`, together with observation
time and source. For running jobs `%R` is a node list, so it is not reported as
a pending reason. A client must not infer Resources, Priority, QOS, or
Dependency when the scheduler did not provide that detail.

SenseCore creates and observes an exact resource name derived from the attempt
and durable submission token. The
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

Each controller process uses its own SSH multiplexing socket. Parallel
controller processes therefore reuse connections within their own workflow
without racing over one global `/tmp` control socket.

Registry publication remains a host workflow. A host may define its own Docker,
Crane, or Skopeo executable overrides without adding them to this package.

## Safety properties

- Backend reports contain fixed, sanitized messages rather than raw shared
  workspace responses.
- SenseCore JSON is piped through the packaged sanitizer before parsing.
- Preflight failure is fail-closed before remote stage or scheduler submit.
- Consumed or ambiguous scheduler identities are rejected before submission.
- Completed checkpoints require a matching payload, step, and byte-count marker.
- Scheduler state, model progress, and evaluation data remain separate; the daemon
  returns those observations without deriving a scientific conclusion.
- Source, image, config, storage, and scheduler identities remain the host
  manifest's responsibility.
