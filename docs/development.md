# Development

This guide is for code contributors. Operators install reviewed wheels or
[offline bundles](two-machine-quickstart.md); source builds and Rust are not
required on an ordinary deployment host.

## Setup and focused checks

Use Python 3.12 (CI's version) and Rust 1.85+:

```bash
git clone https://github.com/NotDesigned/ml-experiment-control.git
cd ml-experiment-control
uv sync --locked --all-packages --all-extras
uv run --no-sync pytest tests/test_quality_tools.py -q
uv run --no-sync --package ml-experiment-server pytest server/tests/test_api_quickstart.py -q
uv run --no-sync python tools/generate_cli_reference.py --check
```

The development environment includes client, server, core, worker and the
optional W&B SDK. Use `--no-sync` afterward so narrower commands do not remove
optional packages from the shared environment. Never sync development dependencies
into the active production environment. Commit dependency metadata and `uv.lock`
together; update `rust/Cargo.lock` when Rust dependencies change.

Start with affected tests. Loopback HTTP tests use injected schedulers, registries
and object stores; they must not create cloud jobs. Real backend/GPU acceptance
is separately scoped work, not an implicit consequence of a test or docs change.

## Required validation

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

Python 3.10 remains the package floor; CI uses Python 3.12. Core, server
and worker independently require **100% line and branch coverage**. The daemon
suite runs server/worker scenarios once, then checks each package separately.
The client has its own measured baseline floor, 90.9% lines / 84.5% branches.
Coverage is not proof of every lifecycle combination or live cloud behavior.

CI also installs the client and worker wheels without dependencies, starts the
base server API without W&B, and installs the full server/client offline bundles.
These checks verify actual package boundaries, entrypoints and dependency closure.
Do not replace them with source imports or assertions about source text.

Prioritize lost acknowledgements, exact-job reconciliation, concurrent claims,
corrupt archives/checkpoints, journal failures, metric-context isolation and
permanent deletion fences. Keep independent fixtures for lifecycle, concurrency
and crash effects. Pure invalid-input cases can share an application while still
checking each input and absence of effects.

## Build deployment artifacts

Source builds require reachable Python and Cargo dependencies. Configure approved
HTTPS indexes/caches instead of disabling TLS. uv does not inherit pip's index;
`UV_DEFAULT_INDEX`, `PIP_INDEX_URL` and Cargo registry access have separate scope.
Build on the deployment target's Python minor, architecture and compatible libc.
The normal server bundle target is Python 3.12/Linux x86-64; another target needs
its own bundle. Client/worker wheels are standard-library Python packages.

Use a clean directory with exactly one reviewed wheel per selected distribution.
For release, **reuse original published bytes for any unchanged package version**;
never replace an existing version with a newly built wheel or source archive.
For example, when only server/client advance and worker is newly introduced:

```bash
mkdir -p .ops/release-wheels .ops/bundles
# Place the original, SHA-verified published core wheel in .ops/release-wheels.
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

Omit `--extra wandb` for a base server. The server bundle includes core, worker,
server and the exported locked dependency closure. The manifest records chosen
extras, versions, wheel hashes and lock hash. The client bundle contains only its
own wheel. Existing releases' offline bundles already include their own closure;
do not mix old bundles with newly split package requirements.

The builder checks downloaded hashes from frozen `uv.lock`; packages available
only as sdist are built on the release host and record their locked source hash.
The complete archive is published without replacing an existing output.
`install.sh` verifies hashes, uses isolated/no-index/no-deps installation and
runs `pip check`; it never starts a service or copies private credentials.
Verify optional integrations as well as the base package after installation.

Publish matching reviewed artifacts and an external checksum file. A four-package
release can contain four wheels, four source archives, server/client bundles and
checksums; an unchanged core artifact must retain its original bytes. Do not
publish test environments or locally regenerated copies under an existing version.

## CI result reuse and test cost

A successful full PR run records the actual tested Git tree in a seven-day Actions
artifact. The main push reuses it only after checking the merged PR, same workflow,
repository, run attempt, artifact identity and exact Git tree. A branch-head SHA
alone is insufficient because PR CI uses a synthetic merge result.

Changed trees, direct pushes, missing/expired/fork evidence and API errors require
a full run. Manual dispatch always runs full checks. Reuse links the original run
and does not claim another execution. An unchanged green PR does not need a second
local full run.

Full checks run core, client, Rust, packaging and sixteen daemon partitions in
parallel. Partitions discover every server/worker test file independently of the
timing hints used to balance them. The final `test` check requires every job,
validates complete same-tree partition manifests, combines coverage data, and
enforces the original independent server/worker gates before recording evidence.
Only coverage data is consumed from shard artifacts. A missing or failed shard
cannot produce reusable evidence.

Cargo and uv caches reduce setup cost; cache misses still build and test the
current source. Packaging builds all four distributions, checks standalone
client/worker and base server installs, then actually installs both offline
bundles. Durations and JUnit identify bottlenecks. Measure full workflow time,
including setup, packaging and the final gate; main's exact-tree reuse is a
separate optimization. Coverage thresholds and test selection must not be
weakened to accelerate a run.

## Ownership and durable changes

Follow [architecture](architecture.md): client and worker depend on neither
server nor core; core never imports server. FastAPI belongs in `server/api`.
Platform records and their schemas belong in `records/`; W&B is an optional
consumer. Scientific parsing for an embedded host controller belongs to that host.

Acquire the workspace lease before initializing writable stores or collectors.
Actions use [SQLite transactions](action-storage.md); file-backed resources use
existing cross-process locks, revision checks and atomic journal transitions.
Do not introduce another state/lock protocol for the same operation.

Submission changes must preserve preparation, authorization, exact ownership and
observation-only reconciliation. New worker/recipe behavior affects new identities,
never frozen images or Run definitions. A collector is not a retry scheduler.
After upgrade/test acceptance, inventory process references and delete inactive
candidate/previous environments and disposable install copies. Keep small reports,
version/hash/CI evidence and separately managed database recovery backups.

## Documentation and public contracts

Keep one primary page per concept and link from [the index](README.md). New users
start with the client guide, operators with deployment, integrators with contracts.
Use placeholders for identities, images and credentials. Private deployment
inventories, failure evidence and account details do not belong in public docs.

Document implemented behavior and current limitations. Git retains historical
plans; remove superseded parallel guides. Keep compatibility rules explicit without
pretending historical paths or scientific state can be rewritten during upgrade.

`cli_reference.md` is generated from the Rust parser:

```bash
uv run --no-sync python tools/generate_cli_reference.py
uv run --no-sync python tools/generate_cli_reference.py --check
```

Update the [core public contract](downstream_contract.md) when changing supported
exports or invocation; verify consuming controllers before advancing immutable
pins. Documentation-only changes never authorize cloud resources, data deletion
or a production restart on their own.
