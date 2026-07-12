# Development and Quality Gates

## Scope

`ml-experiment-control` is a backend library. Backend tests must use injected
command runners and may not contact live schedulers. Project-specific commands,
paths, metrics, and assets belong in host-owned `ProjectAdapter` tests; only a
backend's own protocol vocabulary belongs in backend-specific tests.

## CLI documentation

`experiment_control.safe_sco.build_parser()` is the single source of truth for
the package's only CLI. Generate and check its reference with:

```bash
python tools/generate_cli_reference.py
python tools/generate_cli_reference.py --check
```

The CLI remains intentionally narrow: it sanitizes SCO output and normalizes
states. Experiment lifecycle control is a Python API, not a hidden command tree.

## Tests and coverage

```bash
python tools/coverage_gate.py
```

The gate runs the full suite and checks repository-wide dimensions separately:

- line coverage: at least 90%;
- branch coverage: at least 80%.

Coverage is a regression floor, not a correctness proof. Prefer simplifying
unreachable branches and testing meaningful identity, recovery, redaction,
atomicity, and fail-closed behavior. Do not add live scheduler calls or generic
tests coupled to a private cluster merely to increase coverage.

CI also checks generated CLI documentation, Python compilation, and wheel
construction on every push and pull request.
