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
validate -> preflight -> stage -> render -> submit
                              -> status/logs/collect/cancel
```

`preflight` returns a credential-free `PreflightReport`. SenseCore checks the
SCO executable and a sanitized exact-name workspace query. WYD scopes checks
to the operation: observation needs only SSH/Slurm control access, staging adds
rsync and storage, and submission adds live partition/GRES, account/QOS,
Apptainer, and mount validation.

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

## Tool overrides

The backends recognize these non-secret environment variables:

```text
EXPERIMENTCTL_SCO_BIN
EXPERIMENTCTL_SSH_BIN
EXPERIMENTCTL_RSYNC_BIN
```

Registry publication remains a host workflow; ELF's helper additionally uses
`EXPERIMENTCTL_DOCKER_BIN`, `EXPERIMENTCTL_CRANE_BIN`, and
`EXPERIMENTCTL_SKOPEO_BIN`.

## Safety properties

- Backend reports contain fixed, sanitized messages rather than raw shared
  workspace responses.
- SenseCore JSON is piped through the packaged sanitizer before parsing.
- Preflight failure is fail-closed before remote stage or scheduler submit.
- Completed checkpoints require a matching payload, step, and byte-count marker.
- Scheduler state, model progress, and scientific conclusions remain separate.
- Source, image, config, storage, and scheduler identities remain the host
  manifest's responsibility.
