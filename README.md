# ML-experiment-control

The default implementation on `main` is native Rust. See the
[latest release](https://github.com/NotDesigned/ml-experiment-control/releases/latest) for the current version.
The client, API, worker and state machines are Rust programs; optional image
build/copy operations retain a narrow Python bridge, and W&B uses its official
SDK process.

Submit code, a Dockerfile and data to an existing HTTP API; run experiments on
SSH Slurm/Apptainer, SenseCore ACP or Alibaba PAI DLC; retrieve verified results.
Backend credentials and image builds stay on the server side.

## Install the native client

Download `install-native.sh` from the
[latest native release](https://github.com/NotDesigned/ml-experiment-control/releases/latest).
Clients are published for Linux x86_64/ARM64, macOS Intel/Apple Silicon and
Windows x86_64. The API and worker deployment bundle targets Linux x86_64.

Replace `X.Y.Z` with that release's version; the installer requires an exact
version. Linux or macOS:

```sh
bash install-native.sh --version X.Y.Z
~/.local/bin/ml-exp --url https://api.example/ml-expd config set --token-file /private/api-token
~/.local/bin/ml-exp check
~/.local/bin/ml-exp backend
```

Keep the token in a private file. Subsequent calls reuse the saved URL and token.
`ml-exp update` selects the latest stable native release and refuses an
automatic downgrade. Windows users download `install-native.ps1` from the same
release and run `./install-native.ps1 -Version X.Y.Z` in PowerShell.

Install with the installer once; later updates use `ml-exp update`. A client from an
earlier release cannot update itself to this one, and its saved configuration is not
read, so run `ml-exp config set` again.

## Choose a starting point

| Your task | Guide |
|---|---|
| Run an experiment through an existing service | [Native usage guide](rust/README.md) |
| Install and configure the service | [Native deployment](rust/README.md#deploying-a-native-service) |
| Connect a client or enable a backend | [Configuration](docs/configuration.md) |
| Understand crates and module boundaries | [Architecture](docs/architecture.md) |
| Check what has been accepted on real providers | [Acceptance status](docs/acceptance-status.md) |
| Develop or test changes | [Development](docs/development.md) |

The [documentation index](docs/README.md) lists every guide. Tag `v1.5.1` preserves
the retired Python service and its documentation.

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

Use `ml-exp check` to read the server version, protocol, capabilities and source
limits, and `ml-exp backend` to list configured executors. Advertised GPU
specifications do not guarantee immediate allocation. Image publication,
scheduler success, training completion and downloadable results are separate
facts; see [recovery](docs/recovery.md).

## Packages and storage

- `rust/crates/core`: pure identities, validation and state transitions.
- `rust/crates/io`: confined filesystem, process and transfer effects.
- `rust/crates/client`: standalone native HTTP client, without Python dependencies.
- `rust/crates/server`: API, durable execution, backend adapters and observation.
- `rust/crates/worker`: in-image execution, checkpoints, metrics and result collection.
- `rust/`: shared credential redaction and workspace configuration.

Python remains only in the optional image builder bridge (`server/`, `worker/`
and `src/`), the official W&B SDK process, and release and upgrade tooling. None
of these is a client or API installation path.

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
