# Daemon Observability and W&B Publication

This document defines the data and failure boundary for daemon-owned log
archival and W&B publication. W&B is a query and visualization mirror. Backend
run files remain canonical for experiment identity, status, logs, metrics,
events, checkpoints, and artifacts.

## Data flow

```text
backend Run/Attempt files
        |
        v
incremental sanitized daemon archive
        |
        v
durable publication outbox
        |--------------------|
        v                    v
Local W&B target       W&B Cloud target
        |                    |
        +---------+----------+
                  v
          bounded API read model
                  |
                  v
            client/TUI display
```

The collector may discover new source bytes and records, but it never waits for
a publisher. Archival and publication failures cannot change scheduler state,
Run/Attempt completion, scientific evidence, or Action execution results.
The project-owned `observe` operation is responsible for synchronizing current
backend evidence into canonical Run/Attempt files before the daemon scans
them. The daemon archive never bypasses the controller to read a scheduler or
remote backend directly. Full-log mirroring therefore requires the Project
adapter to materialize complete appendable stdout/stderr files; bounded log
tails alone remain bounded tails and are not presented as a complete archive.

## Canonical sources

The archive reads only exact Run/Attempt sources selected by the existing
manifest and attempt resolution rules:

- `stdout.log` and `stderr.log` as byte streams;
- `train_metrics.jsonl`, evaluation `metrics.jsonl`, and `events.jsonl` as
  complete newline-delimited records;
- structured collection evidence already persisted by the project controller.

Each source cursor binds to Project, Run, Attempt, canonical source path, file
identity, and byte offset. Truncation or replacement starts a new source
generation; it never silently reuses an old offset. Partial trailing lines are
not archived or published until complete.

Secrets are removed before daemon persistence. Raw external output must not be
stored first and redacted later. Log redaction uses the server's credential and
URL rules; metric/event payloads additionally reject secret-like keys and
non-finite values.

## Stable identities

One W&B run represents one immutable Attempt, not a mutable scheduler job:

```text
wandb_run_id = sha256(workspace_id, project, run_id, attempt_id)[:32]
```

The display name may include the authored Run and Attempt names. Backend job
IDs, retries, and publication target are metadata, never W&B run identity.
Local and Cloud targets use the same logical run identity but maintain
independent publication state and cursors.

Every archived record has an idempotency key derived from its exact source
generation, byte range, and sanitized payload digest. Replaying an outbox item
must not create a duplicate history record.

## Target state

Configuration records desired state; observed state is durable and separate.

```text
DISABLED     target not requested
UNAVAILABLE requested, but dependency/service/credential is unavailable
PENDING      durable records exist and have not been attempted
SYNCING      one publisher owns the target lease
READY        backlog is empty and the target passed its last operation
DEGRADED     retryable publication failure; canonical collection continues
FAILED       retry budget exhausted; automatic cooldown recovery is pending
```

Each target exposes only bounded status: target, state, pending/delivered/failed
counts, `updated_at`, sanitized error class, and a credential-free dashboard
URL. Local and Cloud status must never be collapsed into one `wandb_ready`
boolean. Archive status additionally exposes rejected-record totals grouped by
reason class; it never exposes rejected payloads or source paths.

## Local target

Local W&B is the preferred visualization target when it is explicitly enabled.
The daemon owns a loopback service or explicitly references an external service
and publishes through a dedicated worker environment containing only target-specific W&B
settings. The service process and publisher are separate health domains:
Dashboard `READY` does not imply that a Run backlog is synchronized.

The W&B SDK is a direct dependency of the daemon distribution. Docker remains
a host prerequisite for a daemon-managed local service. A configured external
local service does not require daemon process ownership; non-loopback external
URLs must use HTTPS so its API key is never sent over plaintext HTTP. With the
supported
W&B SDK, publishing to a self-managed server requires a local account API key;
configure `publisher_credential_ref` through the same daemon-host CLI. Without
it the dashboard may be `READY`, but the Local publisher is `UNAVAILABLE` and
no Local outbox is created. That key is scoped only to the Local publisher
child and is never reused for Cloud.

For `managed: true`, an empty `command` selects the packaged foreground Docker
wrapper. It binds only the configured loopback address, bind-mounts `data_dir`
to `/vol`, and stops its exact container on daemon shutdown. Configure the
absolute `docker_executable` when Docker is outside `/usr/bin`; production
operators should replace the mutable `wandb/local` tag with an approved image
digest. Dashboard readiness and publisher authentication remain separate.

A complete Local publisher configuration requires all three fields:

```yaml
observability:
  local_wandb:
    enabled: true
    publisher_entity: my-local-entity
    publisher_credential_ref: wandb-local-default
```

Provision the Local account key without placing it in YAML or HTTP:

```bash
ml-expd --config server.yaml credential wandb set wandb-local-default
```

Use a persistent daemon configuration path such as
`~/.config/ml-expd/<workspace>.yaml`; `/tmp` configuration is suitable only for
short-lived smoke tests and must not be the source of a long-running service.

## Cloud target

Cloud publication is opt-in per submission/retry and is independent of Local
publication. The operation stores only desired target state in daemon-owned
submission metadata; the credential reference remains daemon configuration. It
does not add API keys or
Cloud policy to the scientific manifest, controller command, backend
environment, or Run identity.

Cloud credentials are provisioned on the daemon host through a write-only CLI
reading stdin. The HTTP API and TUI expose only aggregate
`publisher_available` state, never the credential reference. A
publisher receives one credential in a minimal child environment for one
target invocation. Cloud failure never blocks Local publication, but a new
Cloud-enabled submission fails preparation when no publisher or credential is
available. On the first VERIFIED activation, the daemon atomically creates the
Cloud target and rewinds that Attempt's archive cursors; the next collection
pass therefore backfills sanitized early records rather than starting only at
the verification boundary. Existing READY/FAILED target state is preserved on
restart reconciliation.

```bash
printf '%s\n' "$WANDB_API_KEY" \
  | ml-expd --config server.yaml credential wandb set wandb-cloud-default --stdin
ml-expd --config server.yaml credential wandb status wandb-cloud-default
```

The CLI output contains only the reference and a configured boolean. Operators
should unset the shell variable after provisioning; it is not a supported
daemon runtime environment variable.

## Audited enable and historical backfill

`observability.backfill` is a first-class direct operation on Project,
Campaign, Run, and Attempt scopes. Preparation freezes the selected Local or
Cloud target and the exact Attempt set in an immutable Action. At most 500
Attempts may be included in one Action; use narrower Campaign/Run scopes for a
larger workspace. Execution requires the normal separate authorization and
exact `EXECUTE <action-id>` confirmation.

The operation creates or re-enables the selected target and rewinds only the
frozen Attempts' archive cursors. Sanitized record identities and target
outbox uniqueness make the replay idempotent. Existing submissions remain
unchanged until this operation is explicitly authorized.

## Durability and concurrency

SQLite owns source cursors, target state, outbox records, retry counts, leases,
and timestamps. Source files remain canonical; SQLite can be rebuilt by
re-scanning them. Outbox insertion and cursor advancement occur in one
transaction so a crash cannot acknowledge data that was not durably queued.

Only one worker may hold a target lease. Expired leases are recoverable after a
daemon restart. Publication acknowledgment is committed only after the SDK
operation finishes successfully. Retries use bounded exponential backoff and
retain the sanitized error class without raw exception text. After the retry
limit, a record enters a one-hour circuit-breaker cooldown; the publisher then
reopens it automatically, preserving strict per-Attempt ordering without
creating a permanent poison-record deadlock.

## Client contract

Clients read observability state but never infer it from `use_wandb`, process
liveness, or a predicted URL. The TUI may open only a server-provided HTTP(S)
URL. Submission UI offers Cloud sync only when the daemon reports publisher
and credential availability. API responses never contain archive paths,
source paths, environment values, API keys, URL userinfo, or raw publisher
exceptions.

## Acceptance criteria

- restarting the daemon resumes cursors and outbox leases without duplicates;
- appending, truncating, and replacing each source is handled explicitly;
- Local and Cloud targets can fail and recover independently;
- no secret fixture reaches archive files, SQLite payloads, API responses, or
  W&B worker environments;
- scheduler submission and collection succeed while either publisher is down;
- TUI states and links are derived only from daemon read models;
- Local/Cloud historical publication is prepared and audited as an Action;
- a fresh clone can install the pinned vendor commit and run all publisher
  tests without relying on globally installed `wandb`.
