# Native Rust rewrite preview

This workspace contains five new application crates for the `rewrite/rust`
branch, plus the already-existing `experiment-redact` Rust package. **The new
applications are incomplete development previews, not deployable replacements.**
The Python 1.5.1 implementation remains the compatibility reference.

Read the [architecture, inventory and acceptance plan](../docs/rust-rewrite.md)
and [per-module inventory](compatibility/inventory.json) before porting a domain.
Do not translate Python files one by one or add a Rust launcher for Python
platform programs. User training code may still be Python.

## Migration order

This branch uses a complexity-first boundary. Rust owns correctness-critical
identities, durable CAS and journals, bounded archives, checkpoint provenance,
receipts, filesystem confinement and uncertain-effect reconciliation. Simple
orchestration may remain Python initially: Dockerfile and image-build wrappers,
provider SDK adapters, thin HTTP routes, compatibility aliases and presentation
code. A Python image builder is acceptable when its immutable image digest,
receipt and retry decision are validated by a native contract before use.

| Crate | Preview executable | Current scope |
| --- | --- | --- |
| `ml-exp-core` | Library | Pure frozen identities, Project/schema/lifecycle and durable transitions, metric/journal decisions, source/archive and checkpoint receipt/restore decisions, data-delivery/result-collection reducers, observations, failure assessment and Slurm/helper protocols |
| `ml-exp-io` | Library | File/hash effects, directory capabilities, bounded process I/O, single-frame pipe transport and descriptor-safe checkpoint verification/sealing |
| `ml-exp-client` | `ml-exp-rs` | Private settings, bounded exact-identity reads/watch, Project operations and deterministic source archive upload/lookup |
| `ml-exp-server` | `ml-expd-rs` | Approved discovery/reads, Project lifecycle, immutable source archive storage/receipts, native checkpoint registry, isolated SQLite/Run fences and Slurm transport |
| `ml-exp-worker` | `ml-exp-worker-rs` | Local subprocess supervision, sealed local managed launch and scoped Slurm metadata helper |

The experiment API observes explicitly approved isolated files. Project metadata
registration/lifecycle is a separate opt-in mode; there is no submission effect
API, cloud managed-worker transport, general artifact/data transfer or W&B
transport. Source archive publication is now a bounded native subchain: the
client packs regular files into deterministic tar/gzip, the server verifies the
transport SHA, rejects links/protected paths, seals the tree, computes the
length-framed tree identity and records an immutable archive receipt. Lost
upload responses are reconciled by archive SHA; a missing receipt does not
authorize replay. Git import and automatic Project creation remain unported.
Metrics and the
isolated canonical journal have native validation, aggregation, persistence and
publication-cursor logic; there is no accepted cloud publisher or submission
effect API. Slurm planners
return data. The separate native helper can invoke `sbatch` only after explicit
isolated configuration/bootstrap; it is not wired to Actions or the HTTP server.
The client can query configured local Run/Attempt status, logs and metrics using
protocol 2. This does not make the preview server a complete experiment API.
A successful `check` does not mean training is configured.

Source archives are stored under a Project-scoped source namespace even when two
Projects have the same tree digest. The transport handler streams the bounded
request body into a temporary file instead of retaining the whole archive in
memory; the durable store removes its archive scratch copy before publishing the
sealed tree and read-only metadata. Artifact contracts are also native at the
pure-core boundary: receipt paths are internal paths, while download proofs use
the compatibility key `outputs/<path>`, and retention timestamps must be valid
RFC3339 values. Object storage, direct downloads, ACKs and provider collection
remain Python shell work until their I/O protocol is ported.

Persistent checkpoints now have a native contract and local registry preview.
The core validates the complete-generation manifest, derives the exact
`checkpoint.<sha256>` identity, rejects mixed generations, binds the restore
plan to the original Run/Attempt and storage scope, and models registration
without moving state bytes. The I/O crate verifies every declared file through
directory descriptors, rejects extra or unstable files, and can seal the
generation. The server registry persists immutable receipts and detects an
existing generation bound to a different manifest. HTTPS registration, worker
polling, provider storage and automatic restore orchestration remain outer
adapters.

The submission boundary is also native: `ml-exp-core::submission` owns frozen
identity checks, bounded redacted request digests, idempotent Create/Reuse/
Conflict/LegacyUnsafe decisions, exact provider reconciliation and the cancel
outbox reducer. Random tokens, durable file publication and Slurm/PAI/SenseCore
calls remain in the imperative shells. An unavailable provider query stays
UNKNOWN and cannot be silently retried.

W&B publication planning is also native in
`ml-exp-core::tracking::publication`. It consumes an explicit local event
suffix, publisher reservations and remote history/display evidence, then
returns deterministic `EmitHistory`, `PublishDisplay` or `Finish` actions.
Local SDK writes are kept distinct from remote history ACKs; sequence/event-ID
conflicts and incomplete provider reads fail closed. The W&B SDK session,
credentials, HTTP operations and durable outbox remain in the Python shell.

`ml-exp-core::transfers` applies the same boundary to the one-GiB unacknowledged
upload buffer. It validates part identities, metadata names, block-rounded
capacity charges and exact durable receipts, and reduces `RECEIVING`, `SEALED`
and `CONFIRMED` states. The Python shell still performs locks, file I/O,
streaming hashes, deletion fences and remote calls.

`ml-exp-core::multipart` covers the adjacent resumable archive contract: it
keeps the historical upload ID encoding, validates part policy and arithmetic,
handles immutable session reuse/restart and exact duplicate parts, and refuses
publication until every part and the whole SHA256 match. Staging files,
streaming reads, locks, capacity effects and object-store publication remain
imperative.

`ml-exp-core::operation_catalog` owns the renderer-neutral operation catalog:
stable operation definitions, parameter metadata, scope filtering and
intent-scope validation. It does not authorize or execute an Action.

`ml-exp-core::capabilities` owns backend-neutral capability matching. It
validates requirements against configured declarations and selects candidates
without claiming live capacity; provider profile loading and availability
probes remain in the Python server shell.

`ml-exp-core::first_submission` owns first-Attempt eligibility and intent
planning: budget checks, exact Attempt/source identity and deterministic
idempotency. Campaign/source reads, YAML compatibility, Action persistence and
scheduler effects remain in the server shell.

`ml-exp-core::input_progress` validates worker input-delivery marker records
and the legacy failure marker. Log reads, freshness calculation and outward
projection remain in the server shell.

`ml-exp-core::data_delivery` covers the SenseCore zero-GPU data-copy contract:
frozen delivery identity, NAS scope, sealed READY reuse, callback transitions,
and exact ACP scheduler-job binding. `ml-exp-core::result_collection` covers
the separate result-recovery contract: terminal Attempt gating, bounded CPU
collector retries, interruption/reconciliation, exact provider observations,
unfinished-archive conflicts, capacity arithmetic and publication planning.
SenseCore/Slurm calls, image builds, locks, clocks, file traversal and object
storage remain Python effects until an outer compatibility facade is accepted.

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
Without `--read-config`, experiment endpoints return 501. With it, only configured
identities are available; no provider is queried or refreshed.

The preview transport explicitly disables redirects, automatic retries and
environment-derived proxies. Proxy support must later be passed as explicit
transport configuration; it is not inherited silently from the shell.

## Enable synthetic experiment reads

Stop the daemon above first and keep its temporary directory. Create only
synthetic metadata beneath it; do not copy production state or credentials.
The registry path below must not already exist:

```bash
umask 077
mkdir -p "$preview_dir/demo/runs/trial/attempts/attempt-001"
printf '{"project":"demo","title":"Synthetic preview","run_roots":["runs"]}\n' > "$preview_dir/demo/research_project.yaml"
printf '{"project":"demo","run_id":"trial"}\n' > "$preview_dir/demo/runs/trial/manifest.json"
printf '{"project":"demo","run_id":"trial","attempt_id":"attempt-001"}\n' > "$preview_dir/demo/runs/trial/attempts/attempt-001/attempt.json"
printf '{"attempt_id":"attempt-001","state":"RUNNING"}\n' > "$preview_dir/demo/runs/trial/status.json"
printf '{"state":"RUNNING"}\n' > "$preview_dir/demo/runs/trial/attempts/attempt-001/status.json"
printf 'synthetic output\n' > "$preview_dir/demo/runs/trial/attempts/attempt-001/stdout.log"
printf '{"step":0,"name":"training_loss","value":1.25,"unit":"nats/token"}\n' > "$preview_dir/demo/runs/trial/attempts/attempt-001/train_metrics.jsonl"
cat > "$preview_dir/reads.json" <<EOF
{
  "schema_version": 1,
  "registry_root": "$preview_dir/registry",
  "projects": [{
    "project": "demo",
    "project_file": "$preview_dir/demo/research_project.yaml",
    "run_roots": ["$preview_dir/demo/runs"]
  }]
}
EOF
cargo run --manifest-path rust/Cargo.toml -p ml-exp-server --bin ml-expd-rs --locked -- \
  --initialize-registry "$preview_dir/registry"
cargo run --manifest-path rust/Cargo.toml -p ml-exp-server --bin ml-expd-rs --locked -- \
  --bind 127.0.0.1:18765 --token-file "$preview_dir/token" --read-config "$preview_dir/reads.json"
```

In the client shell, reuse the explicit preview configuration:

```bash
ml-exp-rs --config "$preview_dir/client.json" get /api/projects
ml-exp-rs --config "$preview_dir/client.json" run status --project demo --run trial --attempt attempt-001
ml-exp-rs --config "$preview_dir/client.json" run logs --project demo --run trial --attempt attempt-001
ml-exp-rs --config "$preview_dir/client.json" run metrics --project demo --run trial --attempt attempt-001 --name training_loss
ml-exp-rs --config "$preview_dir/client.json" get /api/runs/demo/trial/attempts/attempt-001/collection
```

Use the same `cargo run ... --` prefix as above if preview binaries are not on
PATH. Initialization refuses every existing target, including an empty directory;
normal startup requires the preview marker and existing deletion lock. The private
read config selects approved Project files and Run roots. The default scanner
finds canonical manifests within those roots; requests cannot supply Run paths.
Reads never bootstrap missing locks or repair metadata.

These files describe observations, not an actual running training job. A missing
timestamp stays unknown. Current must be explicitly bound by a Run status,
collection or decision mirror; directory sorting never chooses it. Historical
Attempt reads do not borrow the current Attempt's logs or metrics. The configured
roots remain an operator trust boundary. Discovery rejects duplicate Run IDs
even with identical manifest hashes, unsafe aliases and overlapping roots. It
stops at a Run manifest rather than walking into payload directories. Legacy
explicit `runs` locators remain available; combining them with
`discover_runs:true` is refused. JSON is strict; YAML supports a bounded subset described
in the architecture document.

Logs and raw metrics retain bounded tails with truncation/error diagnostics;
partial raw history cannot claim complete validation. Complete canonical journal
reads fail closed beyond 8 MiB or 100,000 observations, before filtering. Collection
returns process observations but cannot verify ArtifactStore publication or
downloads; download, collection execution and ACK are not implemented. There are
four read slots and a 30-second HTTP wait deadline. A timed-out blocking filesystem
read may continue holding its slot; the timeout is not kernel-I/O cancellation.
Project listings use metadata-only snapshots, without opening metric/log tails
or weights, and return the established array of Project summaries.

## Optional Project lifecycle

Stop the preview daemon, then initialize its lifecycle ledger once:

```bash
ml-expd-rs --read-config "$preview_dir/reads.json" --initialize-project-lifecycle
ml-expd-rs --token-file "$preview_dir/token" --read-config "$preview_dir/reads.json" --project-lifecycle
```

In the client shell:

```bash
ml-exp-rs --config "$preview_dir/client.json" project list
ml-exp-rs --config "$preview_dir/client.json" project pause demo --reason "inspection"
ml-exp-rs --config "$preview_dir/client.json" project resume demo
ml-exp-rs --config "$preview_dir/client.json" project archive demo --reason "completed"
ml-exp-rs --config "$preview_dir/client.json" project restore demo
ml-exp-rs --config "$preview_dir/client.json" project unregister demo
ml-exp-rs --config "$preview_dir/client.json" project register "$preview_dir/demo/research_project.yaml"
```

Registration accepts only an exact daemon path already approved in `reads.json`.
It validates the Project schema, not arbitrary client-local source files. Lifecycle
registration currently requires an empty `campaigns` list; Campaign loading is not
yet implemented. `restore` leaves an
archived Project paused; `resume` requires a valid approved Project file. Every
mutation optionally accepts `--expected-revision N` from `project list` for CAS.
`unregister-all` removes all tracking records. None of these commands stops jobs,
deletes artifacts or edits research files. Restart does not reimport removed records.

The head and audit journal keep recoverable legacy-format durability envelopes.
Ordinary reads never repair a partial journal. An explicit valid metadata operation
can finish a missing audit tail; invalid plans and stale revisions do not repair it.
Corrupt complete records are refused. After a lost response or possible write,
inspect `project list` before deciding what to do; the client never replays POST.

Stop the test daemon with Ctrl-C. Remove only the temporary directory you created,
after checking that the process is stopped. Never point cleanup at an existing
workspace or a directory returned by a production API.

## Sealed local worker

`managed-launch` is an operator-only entry for a manifest already staged by
trusted code using `ml_exp_core::launch::encode`. Its private JSON config contains
`manifest_path`, exact HTTPS `endpoint` (identity and SHA256), fresh `attempt_root`,
existing `workdir`, `maximum_timeout_seconds` and optional `base_environment`.
The sealed OUTPUT_DIR must equal `<attempt_root>/outputs`; no command override
is accepted on the command line.

```bash
ml-exp-worker-rs managed-launch --config /private/managed.json --initialize
ml-exp-worker-rs managed-launch --config /private/managed.json
ml-exp-worker-rs managed-launch --config /private/managed.json --inspect
```

Initialization refuses an existing generation. The durable start marker is synced
before process execution. A finished invocation returns its checked original
receipt without running again. STARTED without a valid receipt stays uncertain;
do not delete the ledger to retry. Ambient environment is cleared, and unsupported
managed transfer/checkpoint/publication capabilities fail before execution.
No endpoint is fetched in this preview; HTTPS supplies the frozen binding only.
The native worker tests provide complete synthetic sealed configurations and run
real child processes, including interrupted launch and output-directory replacement.
Training success does not imply published artifacts: publication is
`NOT_IMPLEMENTED`, and downloads remain unavailable.

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
