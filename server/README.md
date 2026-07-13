# ml-expd

`ml-expd` is the independent HTTP daemon shipped by the
`ml-experiment-control` workspace. It owns Project lifecycle, evidence
indexing, backend polling, Agent-turn/proposal stores, and gated experiment
actions. Model calls and code analysis run in a separate client.

Run from the repository root:

```bash
uv run --package ml-experiment-server ml-expd --config server/examples/ml-expd.yaml
```

The full architecture, configuration, and verification contract is maintained
in the repository root `README.md` and `docs/development.md`.
