# Development

## Setup and focused checks

Use Python 3.12 to match CI and Rust 1.85+ to build `experiment-redact`:

```bash
git clone https://github.com/NotDesigned/ml-experiment-control.git
cd ml-experiment-control
uv sync --locked --all-packages --all-extras
```

This creates a development `.venv`; do not sync development dependencies into
a deployed service environment. Commit `pyproject.toml` and `uv.lock` together
when dependencies change, and `rust/Cargo.lock` when Rust dependencies change.
The workspace contains core, server, client and worker packages. The full
development environment includes the optional W&B SDK. Use `uv run --no-sync`
after this setup so a narrower package command does not remove optional
dependencies from the shared environment.

Run affected tests first. For documentation and the introductory HTTP workflow:

```bash
uv run --no-sync pytest tests/test_quality_tools.py -q
uv run --no-sync --package ml-experiment-server pytest server/tests/test_api_quickstart.py -q
uv run --no-sync python tools/generate_cli_reference.py --check
```

The quickstart suite uses real loopback HTTP with injected registry, scheduler
and object-store boundaries. It does not prove live GPU availability. Backend
tests use injected runners/REST responses; they must not submit to a scheduler.

## Required validation

PRs run the full workflow using Python 3.12. Python 3.10 remains the package
compatibility floor, not another CI job. During editing, run affected tests; a
green full PR run does not require another unchanged local full run. To run the
full gates locally:

```bash
cargo fmt --manifest-path rust/Cargo.toml -- --check
cargo clippy --locked --manifest-path rust/Cargo.toml -- -D warnings
cargo test --locked --manifest-path rust/Cargo.toml
uv run --no-sync mypy
uv run --no-sync python tools/coverage_gate.py
uv run --no-sync --package ml-experiment-server python tools/coverage_gate.py --suite daemon --workers 2
uv run --no-sync --package ml-experiment-client python tools/coverage_gate.py --suite client --min-line 90.9 --min-branch 84.5
uv run --no-sync python tools/generate_cli_reference.py --check
uv run --no-sync python -m compileall -q tools examples
uv build --all-packages
```

CI also installs the client and worker wheels without dependencies, tests the
base server's authenticated HTTP API and metrics without W&B, and installs a
full server bundle with the optional SDK. The daemon suite runs server and
worker flows together once, then gates their measured coverage separately.
Core, server and worker each require **100% line and 100% branch coverage**. Coverage is a
regression floor; it does not prove all lifecycle combinations or a live backend.
The standalone client previously had no coverage gate. Its measured baseline is
90.96% line / 84.57% branch; CI now independently requires 90.9% / 84.5%. This is
a separate baseline floor, not a claim of 100% client coverage.

Meaningful scenarios include lost create acknowledgments, exact identity
reconciliation without replay, concurrent execution claims, cancellation of
only an owned job, corrupt archives/checkpoints, interrupted journal writes,
metric-context isolation and stale observations. Prefer simplifying code and
testing these boundaries over tests that merely repeat implementation details.

## Build deployment artifacts

Ordinary operators install reviewed wheels or an offline bundle; they do not need
Rust. Source builds require Rust 1.85+ and access to both Python build dependencies
and Cargo dependencies. For Ubuntu 24.04, its approved repositories may provide
versioned `rustc-1.85`/`cargo-1.85` packages even when unversioned Rust is too old.
Check the installed versions and executable PATH before building. Never install
unreviewed compilers or disable TLS checks to bypass an unreachable installer.

uv does not automatically inherit pip's configured index. Set `UV_DEFAULT_INDEX`
to your approved, reachable HTTPS index; a pip mirror working over HTTP does not
prove its hostname has a valid HTTPS certificate. Cargo has separate registry
access. For an isolated builder, provision verified Cargo dependencies from the
lockfile through your normal dependency process. Do not require service operators
to reconstruct a Cargo cache just to install the application.

Build into a clean wheel directory so stale package versions cannot be selected:

```bash
mkdir -p .ops/release-wheels .ops/bundles
uv build --package ml-experiment-control --out-dir .ops/release-wheels
uv build --package ml-experiment-worker --out-dir .ops/release-wheels
uv build --package ml-experiment-server --out-dir .ops/release-wheels
uv build --package ml-experiment-client --out-dir .ops/release-wheels
uv run --no-project --python 3.12 --with pip --with setuptools --with wheel \
  python tools/build_deployment_bundle.py \
  --component server --extra wandb --wheels .ops/release-wheels \
  --output .ops/bundles/server-bundle.tar.gz
python3 tools/build_deployment_bundle.py --component client \
  --wheels .ops/release-wheels --output .ops/bundles/client-bundle.tar.gz
(cd .ops/bundles && sha256sum *.tar.gz > SHA256SUMS)
```

The server bundle includes matching core, worker and server wheels. The example
selects the full `wandb` extra; omit `--extra wandb` for a base server without the
SDK. The bundle manifest records the selected extras. Client bundles contain
only the client wheel. Older published bundles already carry their dependency
closure; do not combine them with newly split source-package requirements.

Use the same target Python and Linux architecture for wheel construction and
dependency download. The standard CI bundle is Python 3.12/Linux x86-64; check
libc compatibility before offering it for another distribution. A different
Python minor/architecture needs its own server bundle. Client bundles are stdlib
only and work on Python 3.10+.

The release tool exports hashed runtime requirements directly from frozen `uv.lock`
and verifies download hashes before building. `PIP_INDEX_URL` can select an approved
HTTPS index for this build step independently from uv's index. Packages with only
a source distribution are built on this release host using its installed build
tools; their locked source hash is recorded in the manifest. No build tools are
needed on the deployment host. The tool then records each package
version/hash, adds the minimal config/guide, and publishes a complete archive
without overwriting an existing one. `install.sh` checks the files and installs
with `--isolated --no-index --require-hashes`. It never starts a service or copies
credentials. Publish the archives and external checksum file with the matching
reviewed release; older releases may contain only individual wheels.

CI builds and actually installs both bundles in new environments and uploads
them as deployment artifacts. This validates package delivery independently of
the normal installed-wheel and source tests; no cloud resources are created.

## CI result reuse and test cost

After every full run passes, CI records the actual checked-out commit and Git
tree as a small seven-day Actions artifact. On a main push, a read-only job checks
the merged PR, its successful run of the same workflow, repository, run attempt,
artifact identity and GitHub's Git object. Only an exact match with the pushed
code tree reuses that full run. Checking a branch-head SHA alone is insufficient:
PR CI checks the synthetic merge result. The main job summary links the original
run; a reused main run does not claim to have executed the suite again.

Direct pushes, changed trees, failed runs, missing/expired artifacts, fork evidence
or API errors require a fresh full run. `workflow_dispatch` always runs the full
suite, even when matching evidence exists. Superseded runs of the same PR are
cancelled; running main checks are not cancelled. Coverage is never inferred from
changed-file filters, lowered thresholds or an empty test selection.

The daemon suite uses two pytest-xdist workers with file-level scheduling. Every
test still runs; pytest-cov combines worker coverage before the independent line
and branch gates. Durations and JUnit results identify future slow scenarios.
Invalid-input matrices that have no side effects can share one application within
a test, retaining each case and checking that no state was published. Lifecycles,
concurrency, crash recovery and cross-process effects keep independent fixtures.

Avoid repeating generated-document checks or source compile checks already
performed by test imports. Keep installed-wheel/standalone-client checks: they
validate different packaging and dependency boundaries than source imports.

## Ownership and durable changes

Follow [architecture](architecture.md). The core must not import the server;
the standalone client and worker must not require either. Workers use only the
standard library; server-only receivers and stores remain in the server.
FastAPI belongs in `server/api`.
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
uv run --no-sync python tools/generate_cli_reference.py
uv run --no-sync python tools/generate_cli_reference.py --check
```

Do not hand-edit it. Changes to exported symbols or redactor invocation must
update [the public contract](downstream_contract.md) and pass affected consumer
integration tests before a consumer's immutable commit pin advances.
