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
| `ml-exp-core` | Library | Pure frozen identities, metrics/journal/tracking decisions, reducers, observations and Slurm/helper protocols |
| `ml-exp-io` | Library | File/hash effects, directory capabilities, bounded process I/O and single-frame pipe transport |
| `ml-exp-client` | `ml-exp-rs` | Private settings, bounded JSON GET and exact-identity status/logs/metrics, bounded watch and pure human summaries |
| `ml-exp-server` | `ml-expd-rs` | Loopback HTTP preview; isolated SQLite/Run fences and authoritative metadata library; explicit Slurm SSH call |
| `ml-exp-worker` | `ml-exp-worker-rs` | Local subprocess supervision and scoped Slurm metadata helper |

There is no real experiment API, managed-worker protocol, artifact/data transfer
or W&B transport in the preview. Metrics and the
isolated canonical journal have native validation, aggregation, persistence and
publication-cursor logic; there is no accepted cloud publisher. Slurm planners
return data. The separate native helper can invoke `sbatch` only after explicit
isolated configuration/bootstrap; it is not wired to Actions or the HTTP server.
The stores and fences are not yet exposed through the preview API. The client
can make exact read-only observations against a compatible protocol-2 API; this
does not make the preview server a complete experiment API.
A successful `check` does not mean training is configured.

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

The client now groups observation commands under `run`:

```bash
ml-exp-rs --config /private/preview/client.json run status --state /private/experiment-state.json
ml-exp-rs --config /private/preview/client.json run status --project demo --run trial --attempt attempt-001 --json
ml-exp-rs --config /private/preview/client.json run status --state /private/experiment-state.json --watch --seconds 60
ml-exp-rs --config /private/preview/client.json run logs --state /private/experiment-state.json --stream stderr --lines 80
ml-exp-rs --config /private/preview/client.json run metrics --state /private/experiment-state.json --name validation_loss
```

These are GET-only. Saved state is bounded to 8 MiB and must be a private regular
file owned by the current user, since it can contain signed capabilities. A
conflicting saved server/identity fails before authentication. Supply a trusted
URL independently through configuration, `--url` or `ML_EXPD_API_URL`; saved state
cannot choose a destination for an environment token. Without saved state, an
omitted Attempt is resolved once. With saved state, logs/metrics require the
original bound Attempt. There is no automatic retry, cancellation or download ACK.
The default status is a human summary; `--json` preserves the full evidence.
`--watch` pins one Attempt, writes intermediate observations to stderr and leaves
the final observation on stdout. Its budget ends waiting; it cannot stop/retry a
job or start a new HTTP read after the budget. An in-flight read keeps its own
timeout. Both scheduler completion and result publication must settle before
early completion.
The isolated preview server still returns 501 for these experiment endpoints.

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
the host fails. Each captured stream retains at most 8 MiB. Exceeding that bound
produces an explicit `OUTPUT_LIMIT_EXCEEDED` receipt error, while the supervisor
continues draining output without extending its captured prefix. Before publishing
a log hash, the worker checks its length and contents against the bytes retained
from the pipe. This is not a filesystem quota or a sandbox: the user process can
write other files, and same-account hostile mutation is outside its protection.
The common process boundary
also bounds stdin and pipe draining; blocking kernel/filesystem I/O can still
delay cleanup, so the deadline is not a hard real-time guarantee. Stopping a local
SSH process would not prove that a remote command stopped.

The common SQLite boundary uses bundled SQLite 3.51.3 or newer for its WAL-reset
fix. Metadata JSON/locks use real directory-relative descriptors; SQLite itself
still opens a pathname. Private ownership, no-follow checks and identity checks
detect ordinary replacement, but do not provide protection from hostile
same-UID/root namespace changes. See the architecture document for that explicit
storage boundary and uncertain-effect handling.

The Slurm helper entry is `slurm-helper --config /absolute/private.json --stdio`;
initialization uses `--initialize` instead, in a fresh dedicated root. Its trusted
configuration fixes target, root, executable, environment and finite budgets.
stdin/stdout must be pipes or sockets: one 4-byte-length-prefixed control message,
at most 8 MiB, followed by EOF, with bounded read/write waiting. This cap is for
metadata, not datasets or checkpoints. There is no general command RPC or implicit
bootstrap. All `safe_to_retry` results are false; a submitted receipt alone does
not verify scheduler ownership. Do not configure this preview against an existing
cluster workspace. Use the temporary-state tests until the remaining provider
and Action integration gates are complete.

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
