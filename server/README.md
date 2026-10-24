# ML-Expd server

`ml-experiment-server` provides the API/daemon `ml-expd`. Install matching core,
worker and server wheels from one release, or use its offline server bundle.
The core carries `experiment-redact`; the worker provides dependency-free
execution scripts. The separate client is not a server runtime dependency.
Training runs on the selected backend; the daemon owns immutable definitions,
checked submissions, observations and result delivery.
The base installation saves and serves metrics without W&B. Choose a full
`ml-experiment-server[wandb]` installation only when publishing to W&B; see
[configuration](../docs/wandb.md). Client and compute images need no W&B SDK.

- [Operator guide](../docs/operator-guide.md): installation and verification.
- [Architecture](../docs/architecture.md): API, builder, controller and workers.
- [HTTP contract](../docs/http_contract.md) and [API workflow](../docs/source-api.md).
- [Recovery](../docs/recovery.md) and [development](../docs/development.md).
- [Administrative Run deletion](../docs/resource-deletion.md): separately enabled, reviewed storage cleanup.

From a development checkout, start the read-only loopback scaffold:

```bash
uv sync --locked --all-packages --all-extras
uv run --no-sync --package ml-experiment-server ml-expd --config server/examples/ml-expd.yaml
```

It has no working executors or scheduling authorization. Provision a remote
service using the operator guide. Versions/capabilities come from `/api/health`.
