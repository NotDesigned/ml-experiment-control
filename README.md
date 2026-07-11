# ML Experiment Control

`ml-experiment-control` is a small Python package for controlling durable ML
experiments without coupling scheduler code to a particular training
repository. It currently provides SenseCore/SCO and WYD Slurm/Apptainer
backends, sanitized readiness checks, command-runner injection, normalized
failure states, and the `ProjectAdapter` protocol used by a host repository.

The package deliberately does **not** own scientific configuration, training
commands, metric semantics, campaign files, credentials, or model assets. A
host repository supplies those through a project adapter and injected backend
services.

For the host-side autonomous research loop, command order, mutation boundaries,
and evidence report contract, see
[`docs/agent_research_guide.md`](../../docs/agent_research_guide.md). This README
documents the reusable package boundary rather than a project experiment plan.

## Install

From the ELF repository:

```bash
python -m pip install -e packages/experiment-control
```

For an isolated build/install check:

```bash
python -m pip wheel --no-deps packages/experiment-control
```

The package has no runtime dependencies outside the Python standard library.

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
```

`BackendServices` is the narrow host boundary. The host injects command
execution, run-directory lookup, backend-record lookup, project metric parsing
and summarization, atomic JSON writing, and UTC time. This keeps the package
independent from the host's manifest implementation.

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
Recovery fails closed when more than one scheduler job matches one attempt; it
never selects one arbitrarily.
Campaign generation and local event reconciliation remain host responsibilities
because this package does not own YAML or run manifests.

WYD log observation checks exact, attempt-qualified canonical `stdout.log` and
`stderr.log` paths first, then exact `slurm-<job-id>.out/.err` paths in the
attempt and run directories. It does not use remote globs. Returned tails are
bounded and redacted, and `collect()` includes the same sanitized excerpts as
`process_evidence` so failures before the training runtime writes metrics remain
diagnosable by a host controller.

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

## Project integration

A training repository implements `ProjectAdapter` to define config resolution,
the launch command/environment, assets, metric and checkpoint-log parsing,
summaries, and source staging policy. The controller composes one
`ProjectAdapter` with one compute backend.

ELF's implementation lives outside this package at
`scripts/experiment_projects/elf.py`, demonstrating that the installed package
does not import ELF modules.

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
EXPERIMENTCTL_SSH_BIN
EXPERIMENTCTL_RSYNC_BIN
```

Each controller process uses its own SSH multiplexing socket. Parallel agents
therefore reuse connections within their own workflow without racing over one
global `/tmp` control socket.

Registry publication remains a host workflow; ELF's helper additionally uses
`EXPERIMENTCTL_DOCKER_BIN`, `EXPERIMENTCTL_CRANE_BIN`, and
`EXPERIMENTCTL_SKOPEO_BIN`.

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
