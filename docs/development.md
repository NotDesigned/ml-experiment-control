# Development and Quality Gates

## Scope

`ml-experiment-control` is the backend core; `server/` is the independently
runnable HTTP daemon distribution. Backend tests must use injected
command runners and may not contact live schedulers. Project-specific commands,
paths, metrics, and assets belong in host-owned `ProjectAdapter` tests; only a
backend's own protocol vocabulary belongs in backend-specific tests.

## Setup

Python dependencies are managed by uv and locked in `uv.lock`:

```bash
git clone https://github.com/NotDesigned/ml-experiment-control.git
cd ml-experiment-control
uv sync --locked
```

`uv sync` creates `.venv`, installs the package in editable mode, and includes
the default `dev` dependency group. Use `uv add <package>` for runtime
dependencies and `uv add --dev <package>` for development tools. Building the
packaged sanitizer requires Rust 1.85 or newer.

Install every workspace distribution when changing the daemon:

```bash
uv sync --locked --all-packages
uv run --package ml-experiment-server pytest server/tests -q
uv run --package ml-experiment-server ml-expd --help
```

Daemon modules may depend on the core package, never the reverse. FastAPI
belongs only in `server/api`; `runtime.py` is the composition root;
`controller_gateway.py` is the sole legacy `experimentctl` subprocess
boundary. The daemon must not import a model provider or run the client Agent
loop. It must also not persist client goals, conversations, model turns, or
derive scientific conclusions from metrics and evaluation records.

Constructing the FastAPI object is side-effect free. Its lifespan first
acquires the workspace lease and only then constructs SQLite stores,
bootstraps the Project registry, indexes Projects, or starts
publisher/collector threads. This ordering is part of the single-writer
contract, not an implementation detail. If the lease is already held, startup
fails before runtime construction; the daemon does not provide a standby
runtime with partially writable stores.

Server-owned JSON state transitions use `storage.DurableJsonState`: callers
hold their store's cross-process lock, compare the expected revision, atomically
replace authoritative state, and append the embedded transition to JSONL. The
next locked access repairs a journal append interrupted by a crash. Non-state
events on the same ledger must use `append_event`, which repairs the transition
first and safely truncates only a crash-incomplete JSONL tail. Complete records
must be mappings and transition revisions must be contiguous; corruption fails
closed. New stores must not invent a separate lock/atomic-write/journal protocol.

Submission changes must cover the complete safety path in
`server/tests/test_submissions.py`: authored-but-unmaterialized Run preparation,
separate authorization/execution, exact backend-job confirmation, idempotent
preparation, and status-only reconciliation after uncertain execution.

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
uv run mypy
uv run python tools/coverage_gate.py
```

Mypy checks the public host/backend contracts in strict mode and verifies that
every backend registered by `build_registry()` structurally implements the
shared `Backend` protocol. The checked surface is intentionally expanded in
stages; untyped backend internals are not evidence that a public boundary is
safe.

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

## Full verification

```bash
uv sync --locked
cargo fmt --manifest-path rust/Cargo.toml -- --check
cargo clippy --locked --manifest-path rust/Cargo.toml -- -D warnings
cargo test --locked --manifest-path rust/Cargo.toml
uv run mypy
uv run python tools/coverage_gate.py
uv run python tools/generate_cli_reference.py --check
uv run python -m compileall -q src tests tools examples
uv run --package ml-experiment-server python -m compileall -q server/src server/tests
uv run --package ml-experiment-server pytest server/tests -q
uv run --package ml-experiment-server ml-expd --help
uv run python examples/local_smoke.py
uv build --all-packages
```
