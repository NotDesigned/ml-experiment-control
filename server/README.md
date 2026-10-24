# ML-Expd server

`ml-experiment-server` provides `ml-expd`, using `ml-experiment-control` and
`ml-experiment-worker`.
The client is not a server runtime dependency. Training runs on the selected
backend; this daemon owns definitions, checked submissions and observations.
The base installation saves and serves metrics without W&B. Install the
`wandb` extra only when publishing to W&B; see [configuration](../docs/wandb.md).

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
