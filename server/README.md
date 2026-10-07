# ML-Expd server

`ml-experiment-server` provides `ml-expd`, using `ml-experiment-control`.
The client is not a server runtime dependency. Training runs on the selected
backend; this daemon owns definitions, checked submissions and observations.

- [Operator guide](../docs/operator-guide.md): installation and verification.
- [Architecture](../docs/architecture.md): API, builder, controller and workers.
- [HTTP contract](../docs/http_contract.md) and [API workflow](../docs/source-api.md).
- [Recovery](../docs/recovery.md) and [development](../docs/development.md).

From a development checkout, start the read-only loopback scaffold:

```bash
uv run --package ml-experiment-server ml-expd --config server/examples/ml-expd.yaml
```

It has no working executors or scheduling authorization. Provision a remote
service using the operator guide. Versions/capabilities come from `/api/health`.
