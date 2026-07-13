# Flow Coverage

Line and branch coverage prove that every production statement and every
control-flow graph edge executes in the test suite. They do not prove every
possible path combination, which is generally unbounded once retries and
polling are involved. This repository therefore tracks critical semantic flows
as scenario coverage.

## Covered flows

| Area | Scenarios exercised |
| --- | --- |
| Durable lifecycle | Transition table enforces `CREATED -> SUBMITTING -> QUEUED -> RUNNING -> SUCCEEDED`; allows late scheduler observations and Slurm requeue, but rejects terminal-state regression |
| Submission recovery | random durable token publication, idempotent replay, crash repair of derived records, concurrent publication, legacy token absence, conflicting request or scheduler identity |
| Cancellation outbox | first request, verified replay, terminal reconciliation without a second mutation, uncertain nonterminal state, identity drift |
| Scheduler identity | name-and-token exact match, historical same-name rejection, unrelated evidence, ambiguous matches, malformed or unavailable evidence |
| Preflight | missing local tool, authentication/query failure, resource and storage failure, ready path for observe/stage/submit |
| Observation | active and terminal status, bounded/redacted logs, expired logs, worker evidence, metric and checkpoint presence or absence |
| Local execution | real process success and failure, durable recovery identity, PID/start-time matching, log collection, exact process-group cancellation |
| Immutable storage | canonical create, atomic replacement failure, legacy observation, attempt isolation, root-mirror repair and drift rejection |

The backend unit tests use injected command runners and intentionally do not
claim live-service flow coverage. End-to-end transport, scheduler, credentials,
and host-controller polling remain downstream integration responsibilities;
ELF is the reference consumer named in the downstream contract.

## Gate

Run:

```bash
uv run python tools/coverage_gate.py
```

The automated gate requires 100% line and 100% branch coverage. Reviewers must
also update this scenario inventory when a change introduces a new lifecycle,
recovery, cancellation, or fail-closed flow.
