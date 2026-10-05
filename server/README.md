# ml-expd

`ml-expd` is the independent HTTP daemon shipped by the
`ml-experiment-control` workspace. It owns Project lifecycle, evidence
indexing, backend polling, immutable operation intents, gated Actions, and
backend reconciliation. Research goals, conversations, model calls, hypothesis
analysis, reports, charts, and scientific verdicts run in a separate client.

Run from the repository root:

```bash
uv run --package ml-experiment-server ml-expd --config server/examples/ml-expd.yaml
```

The operator-facing submission lifecycle is:

```text
POST /api/experiments/{project}/{run_id}/submissions/prepare
POST /api/submissions/{submission_id}/authorize
POST /api/submissions/{submission_id}/execute
POST /api/submissions/{submission_id}/reconcile  # only when uncertain
```

Preparation is non-mutating and supports authored Runs that have not yet been
materialized. Execution returns `EXECUTING` after durably claiming the Action.
Poll `GET /api/submissions/{submission_id}` until it leaves that state;
`VERIFIED` confirms the exact backend job through `status`, before training
completion. Use reconciliation only for `RECONCILE_REQUIRED`; it never resubmits.

For other mutations, clients submit a complete `OperationIntent` to
`POST /api/actions/prepare`, then authorize and execute the returned Action.
There is no `/api/agent/*` API or daemon-owned model-turn state.
`GET /api/objects` includes a neutral `code_identity` so external clients can
verify that their read-only checkout matches the Project managed by the daemon.

The full architecture, configuration, and verification contract is maintained
in the repository root `README.md`, `docs/development.md`, and
`docs/project_lifecycle.md`. Protocol negotiation and native bearer/TLS
deployment are specified in `docs/http_contract.md`.

New managed projects use [source imports, container execution and S3 artifact
downloads](../docs/source-api.md). The current server is 0.2.0 / protocol 2;
W&B support is removed.
