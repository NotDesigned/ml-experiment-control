# Development

## Setup and focused checks

Use Python 3.12 to match CI and Rust 1.85+ to build `experiment-redact`:

```bash
git clone https://github.com/NotDesigned/ml-experiment-control.git
cd ml-experiment-control
uv sync --locked --all-packages
```

This creates a development `.venv`; do not sync development dependencies into
a deployed service environment. Commit `pyproject.toml` and `uv.lock` together
when dependencies change, and `rust/Cargo.lock` when Rust dependencies change.

Run affected tests first. For documentation and the introductory HTTP workflow:

```bash
uv run pytest tests/test_quality_tools.py -q
uv run --package ml-experiment-server pytest server/tests/test_api_quickstart.py -q
uv run python tools/generate_cli_reference.py --check
```

The quickstart suite uses real loopback HTTP with injected registry, scheduler
and object-store boundaries. It does not prove live GPU availability. Backend
tests use injected runners/REST responses; they must not submit to a scheduler.

## Required validation

The repository workflow runs on pull requests and main pushes, using Python
3.12. Python 3.10 remains the package compatibility floor, not another CI job.
Before merging implementation changes, run the independent full gates once:

```bash
cargo fmt --manifest-path rust/Cargo.toml -- --check
cargo clippy --locked --manifest-path rust/Cargo.toml -- -D warnings
cargo test --locked --manifest-path rust/Cargo.toml
uv run mypy
uv run python tools/coverage_gate.py
uv run --package ml-experiment-server python tools/coverage_gate.py --suite daemon
uv run --package ml-experiment-client pytest client/tests -q
uv run python tools/generate_cli_reference.py --check
uv run python -m compileall -q src tests tools examples
uv run --package ml-experiment-server python -m compileall -q server/src server/tests
uv run --package ml-experiment-client python -m compileall -q client/src client/tests
uv build --all-packages
```

CI also installs the client wheel without core/server dependencies and smoke
tests the installed server/core wheels. Both core and server coverage gates
require **100% line and 100% branch coverage independently**. Coverage is a
regression floor; it does not prove all lifecycle combinations or a live backend.

Meaningful scenarios include lost create acknowledgments, exact identity
reconciliation without replay, concurrent execution claims, cancellation of
only an owned job, corrupt archives/checkpoints, interrupted journal writes,
metric-context isolation and stale observations. Prefer simplifying code and
testing these boundaries over tests that merely repeat implementation details.

## Ownership and durable changes

Follow [architecture](architecture.md). The core must not import the server;
the standalone client must not require either. FastAPI belongs in `server/api`.
Host scientific configuration/parsers belong in `ProjectAdapter`; managed API
projects use the generic worker and metric schema.

Daemon startup must acquire the workspace lease before constructing writable
stores or starting collectors. Actions use [SQLite transactions](action-storage.md).
File-backed state uses `DurableJsonState` under its store's cross-process lock:
revision check, atomic replace and repairable embedded journal transition.
Reviewed multi-file edits bind old/new digests before effects and recover only
that transaction. Do not invent another state/lock/journal protocol.

Submission changes must cover preparation, explicit authorization, exact-job
verification and observation-only reconciliation. New worker behavior needs a
new immutable Runtime identity; do not reinterpret old receipts or silently
replay uncertain requests. A collector is not a retry scheduler.

Live acceptance is separate, explicitly scoped work with project identity,
finite duration/GPU budget, exact Attempt and downloaded file-hash evidence.
Documentation checks never justify a new cloud or GPU job.

## Documentation and public contracts

Maintain one primary page per concept and link from [the index](README.md).
Keep introductory client instructions separate from private operator setup.
Derive route/model examples from current OpenAPI and CLI parsers. Use placeholder
credentials, IDs and image digests; never publish deployment secrets or logs.
Remove obsolete instructions rather than keeping parallel historical guides;
Git retains their history. Label unimplemented designs and known defects.

`cli_reference.md` is generated from the Rust parser:

```bash
uv run python tools/generate_cli_reference.py
uv run python tools/generate_cli_reference.py --check
```

Do not hand-edit it. Changes to exported symbols or redactor invocation must
update [the public contract](downstream_contract.md) and pass affected consumer
integration tests before a consumer's immutable commit pin advances.
