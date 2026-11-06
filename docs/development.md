# Development

The default client, server, worker and state machines are the Rust workspace in
`rust/`. Use Rust 1.85+; ordinary deployment hosts install verified native release
binaries and need no compiler. The optional image bridge uses Python 3.12 in CI.

## Native checks

```sh
cargo fmt --manifest-path rust/Cargo.toml --all -- --check
cargo clippy --locked --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings
cargo test --locked --manifest-path rust/Cargo.toml --workspace
cargo build --locked --manifest-path rust/Cargo.toml --workspace --bins
bash rust/tests/smoke.sh
```

The executable smoke test uses private temporary state and local fake providers.
Tests cover identities, state transitions, lost acknowledgements, concurrency,
process control, archives, checkpoint recovery and metric publication. They do
not prove that a cloud GPU or multi-node network is usable. Keep provider
acceptance separate and inspect the [recorded limits](rust-rewrite.md).

The core receives values and returns decisions. Put filesystem, network, clock,
randomness and process effects in the outer packages. Never replace uncertain
submission evidence with a fresh submission identity or clear publication cursors.

## Optional Python image bridge

Install the small isolated dependency set, then exercise the bridge without
calling a provider:

```sh
python3 -m venv .ops/image-bridge-venv
.ops/image-bridge-venv/bin/pip install -r rust/tests/image-bridge-requirements.txt
PYTHONPATH=server/src:src:worker/src .ops/image-bridge-venv/bin/python rust/tests/test_native_build_bridge.py
```

This source test does not validate the builder's installed module. Before a
builder rollout, verify a real daemon build and receipt. See the native guide.

## CI and publication

`native-quality` checks PRs. On `main`, it reuses a successful PR run only after
verifying its exact Git tree and matching run-attempt evidence; missing evidence
or an API failure causes full checks. Direct changes to `main` receive full
checks. Rust fmt, clippy, all workspace tests, executable smoke, the optional
image bridge and client tests on Linux ARM64, macOS Intel/Apple Silicon and
Windows x86_64 are required. The final `test` job fails if a required job is skipped
or unsuccessful without verified reuse evidence.

`native-release` builds and tests the exact release executables, including a
static musl worker. A manual run packages candidates without publication. After
acceptance, an immutable `native-vX.Y.Z` tag publishes the binaries, manifest,
installer, deployment bundle and SHA256SUMS. Never replace an existing tag or
asset. Version changes update the workspace and its six owned lockfile entries;
third-party dependencies remain locked.

A tag's cache is readable only by that tag, so tag runs restore caches saved on
`main` and save none. Refresh them after a release with
`gh workflow run native-release.yml --ref main -f version=X.Y.Z`, which packages
without publishing. A stale cache costs only rebuild time.

Deploy with [the upgrade tool](operator-guide.md#upgrade-a-release), only after
checking active builds, submissions, transfers and publisher send boundaries. Preserve frozen definitions, private configuration, unknown
operations and existing worker hashes. A release does not require rebuilding old
experiment images or restarting training. Retain small verification records,
not duplicate build environments or unused installation archives.

The canonical client binary is `ml-exp` (`ml-exp.exe` on Windows). Release jobs
run client tests on each target OS, validate every platform binary, and require
all five client targets before assembling the manifest. Tag publication also
installs and self-updates the public client on all five runner platforms. Linux
ARM64 uses static musl; Windows uses the static MSVC runtime. Server and worker
packaging remains Linux x86_64. The former client name is not a second entrypoint.
