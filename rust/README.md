# Native Rust rewrite preview

This workspace contains five new application crates for the `rewrite/rust`
branch, plus the already-existing `experiment-redact` Rust package. **The new
applications are incomplete development previews, not deployable replacements.**
The Python 1.5.1 implementation remains the compatibility reference.

Read the [architecture, inventory and acceptance plan](../docs/rust-rewrite.md)
and [per-module inventory](compatibility/inventory.json) before porting a domain.
Do not translate Python files one by one or add a Rust launcher for Python
platform programs. User training code may still be Python.

| Crate | Preview executable | Current scope |
| --- | --- | --- |
| `ml-exp-core` | Library | Pure frozen JSON/Run/launch identities, metrics, Action/index reducers, observation views and Slurm plans/parsers |
| `ml-exp-io` | Library | Shared explicit file/hash/atomic-write effects and Unix directory capabilities |
| `ml-exp-client` | `ml-exp-rs` | Private saved settings and bounded read-only JSON API calls |
| `ml-exp-server` | `ml-expd-rs` | Loopback-only HTTP preview; isolated Action/RunIndex SQLite stores and permanent Run fences |
| `ml-exp-worker` | `ml-exp-worker-rs` | Unix local subprocess supervision in a fresh output directory |

There is no executing provider adapter, real experiment API, managed-worker
protocol, artifact/data transfer or W&B implementation in the preview. Metrics
have native schema, observation and aggregation logic, but record ingress and
durable publication are still missing. Slurm planners return data, never execute
SSH or submit a job. The Action store, RunIndex and fences are not yet exposed through
the preview HTTP server. A successful `check` does not mean training is configured.

## Build and check

From the repository root, with a toolchain satisfying the locked dependencies
(the manifests and the Rust rewrite CI use Rust 1.85):

```bash
cargo fmt --manifest-path rust/Cargo.toml --all -- --check
cargo test --manifest-path rust/Cargo.toml --workspace --all-targets --locked
cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets --locked -- -D warnings
cargo build --manifest-path rust/Cargo.toml --workspace --bins --locked
```

The workspace's historical default member is the redactor package. Always use
`--workspace` to check the five new application crates too. These commands do not
install a release, replace system binaries or restart services. Keep build caches
outside live experiment state and observe the host's disk capacity.

## Try an isolated API and client

Use a fresh temporary directory and a synthetic token, never an existing
server workspace or provider credentials. In a disposable shell:

```bash
preview_dir="$(mktemp -d)"
chmod 700 "$preview_dir"
head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$preview_dir/token"
chmod 600 "$preview_dir/token"
cargo run --manifest-path rust/Cargo.toml -p ml-exp-server --bin ml-expd-rs --locked -- \
  --bind 127.0.0.1:18765 --token-file "$preview_dir/token"
```

Keep the server running. In another shell set `preview_dir` to that same temporary
path and use an explicit private preview configuration:

```bash
cargo run --manifest-path rust/Cargo.toml -p ml-exp-client --bin ml-exp-rs --locked -- \
  --config "$preview_dir/client.json" --url http://127.0.0.1:18765 \
  config set --token-file "$preview_dir/token"
cargo run --manifest-path rust/Cargo.toml -p ml-exp-client --bin ml-exp-rs --locked -- \
  --config "$preview_dir/client.json" check
cargo run --manifest-path rust/Cargo.toml -p ml-exp-client --bin ml-exp-rs --locked -- \
  --config "$preview_dir/client.json" get /api/readiness
```

The preview default configuration is under `~/.config/ml-expd-rust/`; it must not
replace the production client's settings. The explicit example path also keeps
synthetic credentials out of normal configuration. The server reports training
`INCOMPLETE`; unsupported API paths return an explicit not-implemented response.
It refuses a non-loopback bind. No job is created.

The preview transport explicitly disables redirects, automatic retries and
environment-derived proxies. Proxy support must later be passed as explicit
transport configuration; it is not inherited silently from the shell.

Stop the test daemon with Ctrl-C. Remove only the temporary directory you created,
after checking that the process is stopped. Never point cleanup at an existing
workspace or a directory returned by a production API.

## Try a local process

Choose a fresh output path that does not exist:

```bash
cargo run --manifest-path rust/Cargo.toml -p ml-exp-worker --bin ml-exp-worker-rs --locked -- \
  --workdir /tmp --output /tmp/ml-exp-rust-UNIQUE-output --timeout-seconds 10 -- \
  sh -c 'printf "native preview\n"'
```

The command captures stdout/stderr and writes an execution receipt with their
hashes. It does not register an Attempt, transfer results, interpret training
metrics, publish a checkpoint or contact a scheduler. A failed command still has
its own local execution evidence; reusing its output directory is refused.
The current supervisor supports Unix process groups. It handles TERM/INT,
timeouts and background descendants in that group; it is not a container or
cgroup sandbox, and cannot recover after the worker is killed with SIGKILL or
the host fails.

## Contribution boundary

Keep `core` pure, shared I/O primitives in `io`, provider execution in `server`,
and runtime helpers in `worker`. `client`, `server` and `worker` may depend on
`core` and `io`; none needs a Python runtime for an accepted native path.
Validators, reducers and request planners accept explicit data (including time,
random identity and evidence) and return values, plans or `Result`. Small effect
executors perform I/O and return evidence. Do not hide environment/clock reads in
core helpers or add a trait for every function. Add compatibility fixtures before
replacing a frozen format. Preserve source failures and unknown effects rather than making
preview output resemble success. Update the inventory's status only when the
stated scope is implemented and its acceptance evidence exists.
