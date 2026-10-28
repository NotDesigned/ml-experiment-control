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
identities, durable CAS and journals, bounded archives, immutable dataset receipts, checkpoint provenance,
receipts, filesystem confinement and uncertain-effect reconciliation. Simple
orchestration may remain Python initially: Dockerfile and image-build wrappers,
provider SDK adapters, thin HTTP routes, compatibility aliases and presentation
code. A Python image builder is acceptable when its immutable image digest,
receipt and retry decision are validated by a native contract before use.

| Crate | Preview executable | Current scope |
| --- | --- | --- |
| `ml-exp-core` | Library | Pure frozen identities, Project/schema/lifecycle and durable transitions, metric/journal decisions, source/archive and checkpoint receipt/restore decisions, data-delivery/result-collection reducers, observations, failure assessment and Slurm/helper protocols |
| `ml-exp-io` | Library | File/hash effects, directory capabilities, bounded process I/O, single-frame pipe transport descriptor-safe checkpoint verification/sealing, deterministic dataset/result tar, confined data extraction and streaming verification |
| `ml-exp-client` | `ml-exp-rs` | Private settings, bounded exact-identity reads/watch, Project operations, deterministic source upload/lookup, standalone Run creation, resumable gateway dataset delivery and opt-in Slurm submissions |
| `ml-exp-server` | `ml-expd-rs` | Approved discovery/reads, Project lifecycle, immutable source archive storage/receipts, native checkpoint registry, isolated SQLite/Run fences, standalone Run creation, resumable gateway dataset delivery and opt-in Slurm submissions |
| `ml-exp-worker` | `ml-exp-worker-rs` | Local subprocess supervision, sealed managed launch, opt-in automatic Slurm image worker and scoped metadata and dataset helpers |

The experiment API observes explicitly approved isolated files. Project metadata
registration/lifecycle is a separate opt-in mode. A separate opt-in native Slurm
submission API handles first Runs with trusted source and materialized-image
receipts. Explicitly enabled Run creation can publish a standalone Run and first
Attempt; it does not submit a job. Cloud managed-worker transport and managed cloud dataset adapters remain unported. An opt-in native background observer and durable W&B publisher are implemented; real W&B cloud acceptance remains pending. The opt-in native Slurm path now supports deterministic dataset packing, bounded gateway upload, immutable receipts and named Run inputs. Native Slurm now supports explicit backend-resident result collection, streaming download and verified ACK/release. Source archive publication is now a bounded native subchain: the
client packs regular files into deterministic tar/gzip, the server verifies the
transport SHA, rejects links/protected paths, seals the tree, computes the
length-framed tree identity and records an immutable archive receipt. Lost
upload responses are reconciled by archive SHA; a missing receipt does not
authorize replay. Git import and automatic Project creation remain unported.
Metrics and the
isolated canonical journal have native validation, aggregation, persistence and
publication-cursor logic; there is no accepted cloud publisher. Slurm planners
return data; the native submission service connects their immutable Actions to
the native SSH helper after explicit isolated configuration/bootstrap,
authorization and confirmation. Default startup still disables submission.
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
RFC3339 values. The native Slurm result path streams a backend-generated tar through pinned SSH and HTTP, verifies the archive and exact files at the client, then explicitly acknowledges it. Object-store result backends and cloud-provider collectors remain unported.

Persistent checkpoints now have a native contract and local registry preview.
The core validates the complete-generation manifest, derives the exact
`checkpoint.<sha256>` identity, rejects mixed generations, binds the restore
plan to the original Run/Attempt and storage scope, and models registration
without moving state bytes. The I/O crate verifies every declared file through
directory descriptors, rejects extra or unstable files, and can seal the
generation. The server registry persists immutable receipts and detects an
existing generation bound to a different manifest. The native Slurm path now
connects explicit SSH verification, metadata-only registration, exact list/read
APIs and a selected checkpoint in a new Run. Native background worker/history polling is implemented; cloud-provider
storage/registration adapters remain incomplete; no automatic restore is chosen.

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
conflicts and incomplete provider reads fail closed. The Rust server owns the durable outbox, credentials, routing and reconciliation. A narrow optional official-SDK Python subprocess owns only W&B network/session effects; it does not import the Python platform.

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

`ml-exp-core::intent` validates the client-authored `OperationIntent` schema,
field bounds, unknown-field rejection and idempotency-key format. It does not
authorize or execute the resulting Action.

`ml-exp-rs contract intent-validate` is the optional process bridge for this
contract. It reads one JSON intent from stdin and emits the normalized receipt;
it does not load client credentials or contact the server.

`ml-exp-rs contract slurm-allocation-validate` is the matching bounded parser
for the `scontrol`/`squeue` allocation evidence protocol. The Python Slurm
observer can use it through `ML_EXPD_ALLOCATION_BIN`; absent the binary, the
existing Python parser remains the compatibility path.

`ml-exp-core::data_delivery` covers the SenseCore zero-GPU data-copy contract:
frozen delivery identity, NAS scope, sealed READY reuse, callback transitions,
and exact ACP scheduler-job binding. `ml-exp-core::result_collection` covers
the separate result-recovery contract: terminal Attempt gating, bounded CPU
collector retries, interruption/reconciliation, exact provider observations,
unfinished-archive conflicts, capacity arithmetic and publication planning.
SenseCore/Slurm calls, image builds, locks, clocks, file traversal and object
storage remain Python effects until an outer compatibility facade is accepted.


## Exact-job cancellation

The opt-in native Slurm submission path exposes an explicit cancellation command:

```sh
ml-exp-rs run cancel <action_id> --confirmation "CANCEL <action_id>"
```

The server accepts only its already verified original job identity. It saves a
cancellation intent before the pinned helper RPC. The helper's durable cancel
outbox prevents duplicate scheduler cancellation after a lost response. Repeating
the command explicitly reconciles the same intent; GET never sends a cancel.
Submission metadata includes `cancellation.termination_verified`, which remains
false until exact terminal scheduler evidence is observed. A stopped process is
not inferred from a successful local SSH exit. An already terminal job can be
confirmed without sending a scheduler cancellation. Same-Run/new-Attempt retry is an explicit checkpoint-bound command. Automatic restart remains unimplemented.

## Native managed datasets

This preview supports configured SSH Slurm backends. PAI and SenseCore dataset
adapters still need integration. It does not enable a cloud dataset capability
just because the native contracts exist.

The operator creates a private `data-helper` configuration containing only
`root` (an approved existing empty shared directory) and the exact profile
`target`. Initialize that root once on the gateway:

```sh
ml-exp-worker-rs data-helper --config /private/data-helper.json --initialize
```

Add `data_helper_config` to the trusted server backend definition. It is the
absolute path of that gateway configuration; the worker executable, SSH alias,
SSH configuration and pinned host key come from the existing trusted backend.
The data helper is separate from the scheduler helper and cannot submit a job.
Existing roots are never implicitly initialized or repaired.

The client path is:

```sh
ml-exp-rs data upload --project demo --executor slurm --directory ./training-data
ml-exp-rs data show --project demo --executor slurm --asset asset.<archive-sha256>
```

Upload packs regular files deterministically on the client, sends zero-based
16 MiB parts through bounded API memory to the gateway and publishes only after
whole-archive and exact file size/SHA validation. Repeating upload with the same
unchanged directory reconciles durable parts after a lost response. Complete
also reconciles an already published tree and finishes its owned scratch cleanup.
The API stores metadata; the shared backend retains the sealed dataset tree.
Completed parts and generated archives are removed after the READY receipt is
durable. GET returns that stored receipt, not a fresh scan of backend files.

A Run request supplies named references, for example:

```json
{"data":[{"name":"training","asset_id":"asset.<archive-sha256>"}]}
```

The server selects its own READY receipts and freezes their identities and file
manifests in the Run. Data references become part of the evaluation protocol
identity. The gateway verifies bytes before scheduling and the managed worker
verifies them again before starting user code. Datasets are bound read-only at
both their original source path and named input destinations. Training code
reads `$INPUT_DIR/training`; `DATASETS_JSON` maps names to these input paths.
Compute nodes need no outgoing data-download connection. With no data field,
legacy request, configuration and script encodings are retained.

Payload arithmetic allows up to 65,536 parts (1 TiB), subject to backend disk
space and time limits; it has no 4 GiB total ceiling. Metadata remains bounded:
control frames are at most 8 MiB, definitions at most 4 MiB, and all frozen named
input receipts in one Run at most 256 KiB. This is not a 1 TiB transfer benchmark.
API ingress admits at most four data requests before reading their bounded
bodies; one process holds at most a bounded part and framed copies, not the
complete dataset. Gateway completion temporarily needs parts, a full archive
and the expanded tree. No new automatic dataset deletion or orphan sweep is
implied by result-download acknowledgement.

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

When existing native Action storage is present under the isolated registry's
`actions/` directory at startup, saved Submission identities can also be read
through `GET /api/submissions/{identity}`. This reads committed state without
creating missing executions or running interruption recovery. Submission
prepare/authorize/execute/cancel and provider dispatch are disabled in this default
configuration; the separately configured Slurm path is described below.

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
bootstrap. Optional `scheduler` configuration fixes absolute `sacct_program`,
`squeue_program`, `scancel_program` paths and a single `partition`. Owned observe
and cancel operations require a matching durable submission receipt and an exact
scheduler row (ID, name, Comment token and partition). Cancellation persists its
outbox before one effect; repeated calls only observe until terminal evidence.
All `safe_to_retry` results are false; a submitted receipt alone does
not verify scheduler ownership. Do not configure this preview against an existing
cluster workspace. Use the temporary-state tests until the remaining provider
and Action integration gates are complete.

The native `slurm::action` runner connects an approved native Action to the
helper through the existing SQLite claim/CAS store. `native-slurm-action.v1`
binds the reviewed plan to the exact helper request. A private immutable frame
is stored before claiming execution, so reconciliation after restart loads the
original request by Action ID. Submission is verified only after an owned
scheduler read. Lost replies retain uncertainty; reconciliation sends only
INSPECT/OBSERVE. Caller-owned Run/deletion/source guards and frozen-file checks
remain mandatory. An explicitly configured first-submission HTTP service now
supplies these guards; general Run workflows, remaining providers and result
publication remain incomplete. Submission success does not prove training.

New native first submissions also freeze a `slurm-shared-attempt.v1` requirement.
The helper's trusted JSON must set `shared_run_root` to the same existing
directory as the backend Profile's `run_root`. Explicit helper initialization
binds that root's path and filesystem identity; ordinary startup never adds
this permission or repairs a missing root. Use a fresh isolated helper namespace
when enabling it, rather than resetting an existing ledger.

Before calling `sbatch`, the helper publishes the exact Attempt manifest and
creates its `inputs`/`outputs` directories under
`<shared_run_root>/<project>/<run>/attempts/<attempt>`. The receipt binds the
original intent and directory identities. An existing Attempt directory,
partial publication, mismatched manifest, replaced root or changed directory
is refused before scheduler dispatch. Identical requests reuse the original
publication after restart. Once dispatch has started, output-directory changes
do not erase the recorded job ID or authorize another submission.

This ordering matters because Slurm opens the configured log files before
running the job script. The generated native script checks the staged Attempt
SHA256 and the materialized SIF SHA256 before starting the payload. It exports
Project/Run/Attempt identity and `DATA_DIR`; managed I/O additionally provides
`INPUT_DIR=/inputs` and `OUTPUT_DIR=/outputs`. Legacy frozen helper requests
without a staging requirement keep their original encoding. Datasets,
checkpoints, metrics ingestion and artifact publication remain separate
unfinished protocols.

A trusted materialized Runtime can additionally declare a `worker` already
installed inside its immutable image:

```json
{
  "program": "/opt/ml-expd/bin/ml-exp-worker-rs",
  "base_environment": {"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"},
  "runtime_environment": ["CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"]
}
```

The helper then stages a locally sealed launch manifest and configuration before
dispatch. The generated job invokes `managed-launch --initialize-and-run` with
the original configuration SHA256. Inside the image, the worker opens the
container workdir, creates fresh `worker/` control state and `outputs/payload/`,
and exports the latter as `OUTPUT_DIR`. It never adopts the helper's existing
output directory as a new process run. The worker commits STARTED before spawn,
captures bounded logs and seals the process exit receipt on shared storage.
Existing control/output paths cannot trigger a second training execution.

Payload environment values come only from the sealed manifest, explicit image
configuration and the declared runtime-variable snapshot. Only GPU visibility,
NVIDIA capabilities and library paths can be selected from the container runtime;
unrelated ambient variables are never forwarded. The snapshot is immutable and
hash-bound to all execution phases and the final receipt. A local binding does
not invent or contact an HTTPS manifest endpoint. Older frozen Runtimes without
`worker` retain their original serialization and execution path.

Run, exact Attempt and collection reads now use the same committed native
scheduler receipt without borrowing another Attempt or promoting scheduler
success to process success. After submission ownership is VERIFIED, explicit worker observation is available:

```sh
ml-exp-rs run refresh action-0123456789abcdef
ml-exp-rs run status --project demo --run train --json
ml-exp-rs run logs --project demo --run train
ml-exp-rs run metrics --project demo --run train --json
```

Refresh uses the original job identity and shared-storage binding. The helper
cannot submit, cancel or bootstrap through this operation. Process state,
stdout/stderr and metrics are cached independently of the immutable Submission;
ordinary GET and watch read accepted samples without provider calls. Refresh
also works when new scheduler mutations are disabled. A lost refresh response
can be observed or refreshed again; it does not require submission reconciliation.

Each file contributes at most a 128 KiB window. Missing, changing, invalid or
truncated windows have explicit diagnostics; failed file reads preserve the last
accepted bounded text. Complete records retain arbitrary metric names, units,
variants and protocol context. Partial JSONL lines and malformed records never
become zeros or complete history. STARTED remains possible execution, rather
than proof of a live process. A finished receipt proves its process outcome,
independently of artifact publication and download availability. Periodic
observation, complete-history journal ingress and
cloud artifact publication/download remain necessary for the full research workflow.

## Opt-in native Slurm submissions

The server accepts a private `--submission-config` only together with
`--read-config` and `--source-archives`. Initialize its isolated submission
namespace once with `--read-config /absolute/reads.json --initialize-submissions`.
Existing namespaces are never reset or repaired by startup. The config contains
`schema_version: 1`, an explicit `enabled` switch (default false), and `backends`.
Each backend holds a native `Profile`, trusted `ssh` settings and a runtime map
keyed by immutable runtime ID. Runtime receipts bind source ID, OCI digest and
materialized SIF path/SHA256. The operator supplies SSH binary/config/host-key
files, helper executable/config, working directory, optional agent socket and
finite timeout/connect budgets; HTTP callers cannot replace them.

An already-authored Run/`attempt-001` with no inputs can then use:

```bash
ml-exp-rs run prepare --project demo --run train --max-gpu-hours 1
ml-exp-rs run preparation --project demo --run train
ml-exp-rs run authorize action-0123456789abcdef --reason reviewed
ml-exp-rs run execute action-0123456789abcdef --confirmation 'EXECUTE action-0123456789abcdef'
ml-exp-rs run submission action-0123456789abcdef
ml-exp-rs run reconcile action-0123456789abcdef
```

Use the actual identity returned by preparation. `preparation` finds the original
slot after response loss; it does not prepare again. `reconcile` checks the
original helper ledger/scheduler identity and never submits. The execution lock
excludes live dispatch before an interrupted claim is marked uncertain. Unknown
or corrupt evidence remains blocked from replay. No automatic retry or checkpoint
choice is implied. The global training readiness stays `INCOMPLETE` until the
remaining data, worker and results paths are accepted.

To create that first Run through the client, the operator separately enables
`allow_run_creation: true` in the private submission configuration. It defaults
to false. The Project must use discovery with one approved canonical Run root;
the Run root and registry staging directory must support an atomic directory
rename on the same filesystem. Reads alone never grant write permission.

Create a local `run.json`, replacing the runtime and executor with approved
identities. No Campaign or research-question setup is required:

```json
{
  "run_id": "train",
  "runtime_id": "runtime.<64 hex characters>",
  "executor": "slurm-h100",
  "command": ["python", "-u", "/workspace/train.py"],
  "resources": {"gpus": 1, "cpus": 8, "memory_gb": 16},
  "walltime_seconds": 300,
  "seed": 42,
  "parameters": {"lr": 0.001},
  "metrics_schema": {"definitions": {"val_loss": {"unit": "nats/token"}}},
  "evaluation": {"variant_id": "v1", "D_FT": 0}
}
```

```bash
ml-exp-rs run create --project demo --file run.json
ml-exp-rs run status --project demo --run train
ml-exp-rs run prepare --project demo --run train --max-gpu-hours 1
```

Creation atomically publishes the original request, Run manifest, first Attempt
and initial states. The immutable manifest freezes source, runtime, resources,
command, metric schema and evaluation protocol identity. An identical repeated
request returns the original creation time and hashes; a different request for
the same Run is refused. Missing or damaged existing files are not recreated.
A lost response requires querying that Project/Run before deciding to repeat
the same request. The client never repeats a POST automatically. Inputs and
arbitrary environment fields are not accepted by this initial creation contract;
their native worker/data protocols are still being migrated.

## Explicit backend checkpoint restore

Training saves one complete generation under
`$OUTPUT_DIR/checkpoints/<generation>/` and atomically publishes
`$OUTPUT_DIR/checkpoint.ready.json` after the files are closed:

```json
{"step":24,"files":[{"path":"checkpoints/step-24/state.json","bytes":7,"sha256":"9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"}]}
```

The byte count and SHA256 must describe the actual files. This is a format
example, not a model checkpoint. The training program defines the complete
restore state, including any optimizer, scheduler, RNG and data cursor it needs.

```bash
ml-exp-rs run checkpoint action-0123456789abcdef
ml-exp-rs run checkpoints --project demo --run train --attempt attempt-001
```

The explicit checkpoint operation verifies and seals the selected generation
through the existing SSH helper, then saves only immutable metadata on the API
server. It never starts a training job. GET reads the registry without remote
file access. A missing or changed ready generation fails verification and leaves
previous registrations intact.

To start a new Run, add an explicit reference to its normal creation request:

```json
{"resume_from":{"run_id":"train","attempt_id":"attempt-001","checkpoint_id":"checkpoint.1111111111111111111111111111111111111111111111111111111111111111"}}
```

This fragment accompanies the required Run fields and a new `run_id`. The server
resolves the registered choice in the same Project and approved storage scope,
freezes its provenance, and derives the backend path. The new worker rechecks
the sealed file set, sizes and SHA256 before starting user code. Training reads
`RESUME_DIR`, `RESUME_CHECKPOINT_ID` and `RESUME_STEP`; the original state stays
on shared storage. A missing/corrupt generation prevents the process from
starting. No "latest" selection, automatic retry, general file download or
artifact publication is provided by this checkpoint path.

## Backend result collection and verified download

With the opt-in native Slurm service configured, a finished managed worker can
publish a deterministic result tar on its shared backend storage:

```sh
ml-exp-rs run collect action-0123456789abcdef
ml-exp-rs run artifacts action-0123456789abcdef
ml-exp-rs run download action-0123456789abcdef --output results.tar
```

Run creation can freeze `"artifacts":{"required_files":["weights.bin","metrics.jsonl"]}`.
Collection checks those files before publishing metadata. Paths are relative to
`OUTPUT_DIR`; the `checkpoints/` subtree and `checkpoint.ready.json` stay separate.
Collection needs no GPU and never submits a scheduler job.

The API stores metadata and relays a bounded SSH stream. It keeps no result tar
on the API host. The client hashes the whole tar and every exact regular member
without extracting it, publishes the completed local file durably, and sends the
verified proof. By default that ACK immediately releases only the generated
backend tar. `--keep` retains it. Original outputs and persistent checkpoint state
remain; this is not full Run deletion or a retention policy for original outputs.
Repeat the same download command after a lost ACK response: a matching complete
local file is reverified and reused. Corrupt/incomplete transfers send no ACK.
A released archive returns 410 and collection never regenerates it; its metadata
tombstone survives restart. Transfer payload sizes use `u64`; control metadata
and proofs retain their 8 MiB ceiling, and active streams have a one-hour deadline.
The old object-store download URLs and cloud result collectors are not provided.

## Explicit same-Run checkpoint retry

A failed, preempted or completed native job can continue in a new Attempt. Select
one registered checkpoint from the immediately previous Attempt; the server does
not choose a checkpoint. Use the original Action identity:

```bash
ml-exp-rs run retry action-0123456789abcdef \
  --confirmation 'RETRY action-0123456789abcdef' \
  --run train --attempt attempt-001 --checkpoint checkpoint.<sha256> \
  --max-gpu-hours 2
ml-exp-rs run prepare --project demo --run train --max-gpu-hours 2
```

This appends `attempt-002`; it does not submit a job. Review and authorize the new
Action, then execute it explicitly. Every Attempt reserves its full requested GPU
limit, including prior Attempts. For two one-GPU, one-hour Attempts, the cumulative
approval must cover two GPU hours. Up to 256 sequential Attempts are supported.
The command requires fresh exact terminal scheduler evidence and the same source,
image and backend storage. Unknown submissions remain blocked. A repeated retry
with the same request reuses the appended Attempt, including after response loss
or restart. It cannot reset a later current Attempt. Run and historical Attempt
manifests, checkpoint registrations and Action identities remain unchanged.

## Full metrics history

```bash
ml-exp-rs run history action-0123456789abcdef
```

Each explicit call imports the next 128 KiB from the exact managed Attempt's fixed
`metrics.jsonl` file. Repeat until `status=COMPLETE`. During training the last page
remains `PARTIAL`, even when its bytes currently reach EOF. Completion requires a
finished process observation. The maximum accepted history file is 128 MiB.

The backend verifies the original file identity and the accepted SHA256 prefix;
replacement, truncation or changes to accepted bytes stop ingestion. The service
saves canonical records before advancing the private cursor. Lost replies or cursor
updates repeat deterministic event identities, so they cannot duplicate metrics.
`records` and `invalid_lines` distinguish parsed metrics from damaged input. The
existing metrics API queries the imported journal; generic units, protocol,
checkpoint and variant identities still apply. No raw metrics file is retained on
the API host. This command does not enable W&B or install a background collector.

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

## Background observation and optional W&B

Add `"polling":{"enabled":true,"interval_seconds":10,"max_candidates":4096,"rpc_timeout_seconds":5}`
to the private submission configuration. One leader performs at most one bounded
RPC per tick, rotating scheduler, worker and full-history reads fairly. Unmanaged
jobs receive scheduler reads only. Terminal jobs retire after process completion,
full metric history and local terminal journal persistence. GETs never trigger
provider I/O. This observer never submits, retries, cancels or releases a result.

`GET /api/observer` shows observer state. Each Submission includes phase-specific
last attempt, last success and safe error codes. This reports observation progress,
not a claimed worker heartbeat. A finished process that never wrote `metrics.jsonl`
is explicitly `NOT_PRODUCED`; missing metrics are not fabricated.

The optional `--tracking-config /private/tracking.json` accepts:

```json
{
  "schema_version": 1,
  "python": "/absolute/python-with-wandb",
  "api_key_file": "/absolute/private/wandb-key",
  "entity": null,
  "project": null,
  "enabled": true
}
```

The official SDK resolves an omitted entity; an omitted project defaults to the
ML-Expd Project. Each new Run may override `wandb.entity`, `wandb.project` or
`wandb.enabled`; its resolved route is frozen. Existing Runs retain their original
route. No credentials in a manifest, argv, journal or client configuration.
Missing optional SDK/default-entity lookup disables unresolved routes with a
diagnostic and leaves local collection available. Preparation export remains off.

```sh
ml-exp-rs run tracking --project demo --run train
```

This local-only status reports destination, confirmed/pending event counts and
safe synchronization diagnostics. A persistent SDK session keeps active cloud
Runs open. Rust reserves sends durably before SDK I/O, requires sequence/event-ID
remote history proof before ACK, and requires remote display/finish confirmation.
An SDK write is not a cloud ACK. Unknown outcomes never authorize blind replay.
The subprocess contract is bounded and cancelled on shutdown. Fake-SDK subprocess
coverage exists; real cloud connectivity, visibility and live green-dot acceptance
are pending alongside the final provider acceptance phase.

## Native cloud adapters and CPU dataset import

`ml-exp-server::backends::cloud` provides callable native signed PAI DLC and
SenseCore ACP adapters. They share the normal Action authorization/CAS and a
private durable one-effect outbox. Job ownership, full pagination, queue rules,
cluster quota selection and offline logs are validated before accepting evidence.
These adapters are not yet configured by the public preview HTTP Run facade.
Cloud creation/input/result orchestration and real-provider acceptance remain
incomplete. Main startup still supports the opt-in Slurm submission path only.

A backend CPU job can now run the native dataset import worker. Its private
configuration contains `root`, `target`, `project`, the exact dataset `definition`,
`timeout_seconds` and a source: either `{"kind":"file","path":"/image/data.tar"}`
or `{"kind":"https","url":"https://approved-host/data.tar?...","allowed_hosts":["approved-host"]}`.
The source is the native deterministic uncompressed dataset tar; this is not an
implicit importer for legacy tar.gz uploads. The operator initializes an approved
empty cache root once; retries must not repeat initialization:

```sh
ml-exp-worker-rs data-import --config /private/import.json --initialize
ml-exp-worker-rs data-import --config /private/import.json
```

Imports reuse the existing dataset store, 16MiB durable parts, exact file/archive
SHA checks and READY receipt. HTTPS requires exact byte ranges and no redirects
or environment proxy. The final cache is verified again on reuse, without source
network access. Dataset payloads stay on backend storage; no API-host spool is
introduced. Progress uses `ML_EXPD_INPUT_PROGRESS`, with received bytes, rate,
phase and safe failure code. The command deadline also sets the worker cancellation
flag; provider/outer worker supervision remains responsible for hard process limits.
Private signed URLs never appear in receipts or errors. Direct import does not
create a scheduler job, attach the dataset to a Run or publish a server receipt.

## Native cloud Run composition

Cloud submissions are an independent opt-in configuration. Initialize their
namespace once using `--initialize-cloud-submissions --read-config /private/reads.json`.
Start the preview with `--source-archives --cloud-config /private/cloud.json`;
Slurm configuration is optional and independent. Cloud configuration contains
`schema_version: 1`, `enabled`, `allow_run_creation` and `backends`. Each backend
fixes its `executor`, native `profile`, approved `run_root`, `credentials_file`,
optional `registry_auth_file`, timeout, allocation and published `runtimes`.
An entry in `runtimes` fixes `runtime_id`, `source_id`, digest-pinned `image`,
`source_dir`, `worker_path` and `worker_sha256`. These are operator/builder
receipts, not client assertions.

The ordinary `run create`, `prepare`, `authorize`, `execute`, `submission`,
`reconcile`, `refresh` and `cancel` commands are reused. GET reads only cached
evidence. Cancellation remains unconfirmed until an exact terminal provider
observation. The image starts `ml-exp-worker-rs cloud-launch` with one sealed
package; the worker preserves backend state and refuses uncertain replay.

This cloud path currently rejects dataset and checkpoint-restore inputs and
does not yet compose metrics/history, result collection or retries. These
limitations are explicit errors; this preview remains unsuitable as a complete
production replacement. No live cloud/GPU acceptance is claimed.
