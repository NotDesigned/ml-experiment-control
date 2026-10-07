# Core library integration

This path is for developers of their own host controller. A client using the
managed HTTP service needs only [the standalone client](api-quickstart.md),
without a `ProjectAdapter` or cloud/SSH credentials.

## Install and try locally

Pin a reviewed immutable commit, replacing the placeholder:

```bash
uv add 'ml-experiment-control @ git+https://github.com/NotDesigned/ml-experiment-control.git@REVIEWED_COMMIT'
```

The core uses PyYAML and packages a Rust redactor. A source installation needs
Rust 1.85+; a suitable built wheel includes the executable. Follow
[development setup](development.md) in a checkout, then run:

```bash
uv run python examples/local_smoke.py
```

[`examples/local_smoke.py`](../examples/local_smoke.py) is a complete Linux
example: build a backend registry, persist immutable manifests and submission
intent, start a real local process, observe it and read redacted logs. Local
execution uses PID plus process start time and an Attempt-qualified process
group. It is not acceptance evidence for Slurm, ACP or GPUs.

## Compose a host

```python
from experiment_control import ExperimentStateStore
from experiment_control.backends import build_registry
from experiment_control.backends.services import BackendServices

# Supply your runner, run-directory lookup, backend-record lookup,
# metric/checkpoint parsers, summary function, atomic writer and clock.
services = BackendServices(...)
backend = build_registry(services).get("slurm")
```

`BackendServices(...)` is a composition sketch, not executable without the
callbacks. Implement your scientific/configuration boundary using
[`examples/minimal_project_adapter.py`](../examples/minimal_project_adapter.py):

1. Resolve/validate config and freeze the command/environment.
2. Describe source and immutable semantic asset identities.
3. Map assets to backend-visible filesystem probes.
4. Parse project metrics/checkpoints and summarize evidence.

The package must never import your training/model code. Asset requirements name
semantic inputs; `AssetProbe` gives a storage path the backend can verify.
`collection_includes` optionally adds host-specific collection patterns.
Metric names, units, success criteria and aggregation remain host responsibilities.

## Submit with durable identity

After manifest construction and successful preflight, persist an intent before
any scheduler mutation:

```python
intent = store.begin_submission(
    project=project, run_id=run["run_id"],
    attempt_id=attempt["attempt_id"], backend=backend.kind,
    request=backend.submission_request(campaign, run, attempt["attempt_id"]),
)
job_id = backend.recover_submission(run, intent, attempt["attempt_id"])
if job_id is None:
    job_id = backend.submit(campaign, run, attempt, dry_run=False, intent=intent)
store.reconcile_submission(
    project=project, run_id=run["run_id"],
    attempt_id=attempt["attempt_id"], backend_job_id=job_id,
)
```

Recovery must establish exact absence before the first create; an ambiguous or
failed lookup must stop this flow. After an uncertain create, use reconciliation,
not a second submit. `ExperimentStateStore` supplies a durable random token;
backends reject non-dry-run submissions without the corresponding intent.
See [the public contract](downstream_contract.md) for identity and upgrade rules.

`availability()` is run-independent and read-only. `preflight(run, scope=...)`
adds operation-specific tools, resource and storage checks. The lifecycle is:

```text
validate → preflight → identity/assets → stage/render
         → durable intent → recover/submit → reconcile
         → status/logs/collect/cancel
```

Use host-only REST configuration for SenseCore, SSH config/agent for WYD and
private registry auth for publication. Never embed credentials in Run manifests.
Supported tool overrides include `EXPERIMENTCTL_REDACTOR_BIN`,
`EXPERIMENTCTL_SSH_BIN`, `EXPERIMENTCTL_RSYNC_BIN`,
`EXPERIMENTCTL_SENSECORE_REST_CONFIG` and
`EXPERIMENTCTL_SENSECORE_CREATE_TIMEOUT_SECONDS`. There is no SCO fallback.
