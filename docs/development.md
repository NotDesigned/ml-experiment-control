# Development and Quality Gates

## Scope

`ml-experiment-control` is a backend library. Backend tests must use injected
command runners and may not contact live schedulers. Project-specific commands,
paths, metrics, and assets belong in host-owned `ProjectAdapter` tests; only a
backend's own protocol vocabulary belongs in backend-specific tests.

## CLI documentation

The Rust binary in `rust/src/main.rs` is the single source of truth for the
package's only CLI. Generate and check its reference with:

```bash
uv run python tools/generate_cli_reference.py
uv run python tools/generate_cli_reference.py --check
```

The CLI remains intentionally narrow: it sanitizes SCO output and normalizes
states. Experiment lifecycle control is a Python API, not a hidden command tree.

Run the Rust-specific checks before the Python suite:

```bash
cargo fmt --manifest-path rust/Cargo.toml -- --check
cargo clippy --locked --manifest-path rust/Cargo.toml -- -D warnings
cargo test --locked --manifest-path rust/Cargo.toml
```

## Tests and coverage

```bash
uv run python tools/coverage_gate.py
```

The gate runs the full suite and checks repository-wide dimensions separately:

- line coverage: 100%;
- branch coverage: 100%.

Coverage is a regression floor, not a correctness proof. Prefer simplifying
unreachable branches and testing meaningful identity, recovery, redaction,
atomicity, and fail-closed behavior. Do not add live scheduler calls or generic
tests coupled to a private cluster merely to increase coverage.

Branch coverage measures control-flow graph edges, not every possible path
combination. The semantic lifecycle and recovery scenarios tracked as flow
coverage are documented in [`flow_coverage.md`](flow_coverage.md).

Changes to exported Python symbols or the Rust CLI must also update
[`downstream_contract.md`](downstream_contract.md) and be validated against the
ELF integration tests before a downstream commit pin advances.

CI first runs `uv sync --locked`, then checks generated CLI documentation,
Python compilation, and distribution construction with `uv build` on every
push and pull request. Update dependencies with `uv add` or `uv remove` and
commit both `pyproject.toml` and `uv.lock`.
