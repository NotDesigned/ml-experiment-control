# ML Experiment Control

`ml-expd` is an HTTP control plane for reproducible experiments on WYD
Slurm/Apptainer and SenseCore/SCO. Import source, combine it with a prepared
container environment, freeze a Run, submit it, inspect progress, and download
results from an exact Attempt. The same source, image and command work across
both platforms; operator-defined profiles supply their different infrastructure.

The client (`client/`, command `ml-exp`) and server (`server/`, command
`ml-expd`) are independently installable distributions. The client communicates
only over HTTP and has no runtime dependencies. The server also uses the
reusable `ml-experiment-control` Python core.
Choose the path that matches your task:

| Task | Start here | Required locally |
| --- | --- | --- |
| Use an existing API from another computer | [API quickstart](docs/api-quickstart.md) | Standalone client, Python 3.10+, API URL and token |
| Deploy your own service | [Operator guide](docs/operator-guide.md) | Linux, daemon, image builder, backend credentials and object storage |
| Embed backend/state primitives in a controller | [Library integration](docs/library-integration.md) | Core package and a host-owned `ProjectAdapter` |
| Contribute to this repository | [Development](docs/development.md) | uv, Rust 1.85+, repository tests |

## First experiment through HTTP

Follow the [API quickstart](docs/api-quickstart.md). It includes the complete
[standalone client package](client/README.md), installed with
`python3 -m pip install ./client`. Its `ml-exp` command includes an offline
`init` template for a tiny output/metrics project. The client imports a
source archive, packages a digest-pinned dependency image, freezes a Run for one
executor, prepares its review gates, submits only on explicit confirmation,
and verifies downloaded files against the uploaded archive's SHA256.

Clients upload source containing a Dockerfile. The server builds and pushes an
exact image to CCR, then WYD converts that digest to a verified reusable SIF;
SenseCore runs the digest directly. Dataset assets stay outside the source/image
and are mounted under `/inputs`. Outputs and atomic checkpoints are returned
through scoped object-storage transfers.

Use `ml-exp experiment experiment.json --state trial.json` to validate, upload,
build, freeze a Run and prepare its gates. Add `--execute --download-to results`
to authorize scheduling within the config's GPU-hour budget and retrieve verified
outputs. `--resume` observes the same identities after interruption; it never
replays an uncertain scheduler request. See the [API quickstart](docs/api-quickstart.md).

New builds use Dockerfile only. Old READY images and immutable Runs remain usable.
The base-image catalogue at `GET /api/environments` supplies approved digest pins
for `FROM`; it is not a second build mechanism. Runtime READY, submission VERIFIED,
scheduler SUCCEEDED and available verified artifacts remain separate states.
Progress endpoints expose phase, real last progress time, diagnostic messages,
build logs and available exact-Attempt queue reasons. Unknown start times stay
unknown. Registry-backed build cache and digest-bound WYD SIFs are reused while
the API host removes transient BuildKit state after every build.

## API and deployment status

The source/container workflow requires server 0.2.0 / protocol 2. The health
endpoint advertises optional capabilities; clients must negotiate these rather
than infer asynchronous behavior from the package version. During review,
use the revision containing the source API; do not assume an older `main`
checkout or package release has these endpoints. Operators should provide the
exact installed revision alongside the connection details.

- [Source/container/artifact contract](docs/source-api.md)
- [HTTP protocol, authentication and recovery](docs/http_contract.md)
- [Project lifecycle](docs/project_lifecycle.md)
- [Action storage](docs/action-storage.md) and [SSE reconnect](docs/sse.md)
- [Supported downstream Python/CLI surface](docs/downstream_contract.md)

A daemon owns one workspace, durable Actions, polling and scheduler execution.
Clients own research goals, analysis and interpretation of metrics. Current
W&B support is removed; numeric metrics and downloadable outputs remain.

For a local read-only daemon scaffold after the development setup:

```bash
uv sync --locked --all-packages
uv run --package ml-experiment-server ml-expd \
  --config server/examples/ml-expd.yaml
```

This binds to `127.0.0.1:8765`. The scaffold has no executors and disables
mutations; it is not a complete remote experiment deployment. Follow the
[operator guide](docs/operator-guide.md) before enabling submission.

Project-defined metric names, units and frozen scoring protocols: [metrics contract](docs/metrics.md).
