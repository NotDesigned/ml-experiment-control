# ML-Expd

Submit code, a Dockerfile and data to an existing HTTP API; run experiments on
SSH Slurm/Apptainer, SenseCore ACP or Alibaba PAI DLC; retrieve verified results.
Backend credentials and image builds stay on the server side.

## Choose a starting point

| Your task | Guide |
|---|---|
| Run an experiment through an existing service | [Client quickstart](docs/api-quickstart.md) |
| Install and configure the service | [Operator guide](docs/operator-guide.md) |
| Connect a client or enable a backend | [Configuration](docs/configuration.md) |
| Deploy a basic API and a separate client | [Two-machine example](docs/two-machine-quickstart.md) |
| Understand package and module boundaries | [Architecture](docs/architecture.md) |
| Develop or test changes | [Development](docs/development.md) |

The [documentation index](docs/README.md) links feature and API references.
Read [compatibility and upgrades](docs/architecture.md#compatibility-and-upgrades)
before changing an existing installation.

## Workflow and responsibilities

```text
code + Dockerfile → immutable Runtime image
local data / backend download → verified backend storage
                           ↓
Runtime + inputs + command + resources → Run → Submission → Attempt
                                                        ↓
                          progress, metrics, checkpoints, verified results
```

A Run freezes the experiment definition; each execution is a separate Attempt.
Training and evaluation use the same mechanism. Experiment code defines metric
meaning, checkpoint contents, recovery and model selection. The platform handles
preparation, scheduling requests, evidence, transfer and result verification.

Use `ml-exp check --json` to discover the installed versions, executors, base-image
digests, capabilities and limits. HTTP protocol remains **2**. Advertised GPU
specifications do not guarantee immediate allocation. Image publication,
scheduler success, training completion and downloadable results are separate
facts; see [recovery](docs/recovery.md).

## Packages and storage

- [Client](client/README.md): standalone Python HTTP client, with no runtime dependencies.
- [Server](server/README.md): API, experiment preparation and result collection, using the reusable [core](docs/library-integration.md).
- [Worker](worker/README.md): standalone standard-library programs copied into job images and remote scripts.

W&B is an [optional server integration](docs/wandb.md). Platform metrics work
without it. Temporary [CCI/DSW debug sessions](docs/debug-sessions.md) are separate
from training; PAI training uses DLC.

Source and data are separate uploads. Data targets shared Slurm storage,
SenseCore NAS through CCR, or approved PAI OSS/CPFS storage. The API uses a bounded
transfer buffer rather than keeping a second permanent dataset. Output archives
are hash-verified on download and may be released after client acknowledgement;
metrics and persistent checkpoint state have separate lifecycles. See
[data](docs/data.md), [checkpoints](docs/persistent-checkpoints.md) and
[administrative deletion](docs/resource-deletion.md).
