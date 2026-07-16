# Project Lifecycle

`ml-expd` maintains a workspace-local registry of research Projects. The
registry controls which Projects this daemon indexes and polls. It does not
replace a Project's authored `research_project.yaml`, and it never owns the
Project repository, scheduler jobs, Runs, Attempts, logs, checkpoints, or
artifacts.

## Identity and ownership

Each registry record binds a stable Project identity to an absolute
`research_project.yaml` path and records its lifecycle state, registration
source, timestamps, and operator reason. The authored file remains the source
of Project title, run roots, controller configuration, Campaign catalog, and
research-question locations.

One daemon workspace has one registry. Two daemon workspaces may register the
same Project independently and assign different lifecycle states. Operators
must avoid enabling live collectors in two workspaces against the same
Project unless the controller explicitly supports that arrangement.

## State machine

```text
                  pause                   archive
        ACTIVE ------------> PAUSED ----------------> ARCHIVED
           ^                   |  ^                      |
           |      resume       |  | restore              |
           +-------------------+  +----------------------+
                    archive from ACTIVE is also allowed
```

| State | Indexed as active | Live collector polls | Meaning |
|---|---:|---:|---|
| `ACTIVE` | yes | yes | Normal observation and operation eligibility |
| `PAUSED` | no | no | Temporarily removed from this live workspace |
| `ARCHIVED` | no | no | Retained as workspace history; restore first returns it to `PAUSED` |

Valid transitions are:

- `ACTIVE -> PAUSED` or `ACTIVE -> ARCHIVED`;
- `PAUSED -> ACTIVE` or `PAUSED -> ARCHIVED`;
- `ARCHIVED -> PAUSED`.

Archive requires an operator reason. Restore is deliberately two-step:
`ARCHIVED -> PAUSED -> ACTIVE`, so restoring a record cannot silently resume
backend polling. Lifecycle verbs are source-state-sensitive API operations, not
aliases for target states: `restore` is valid only from `ARCHIVED`, `resume`
only from `PAUSED`, and `pause` only from `ACTIVE`. A verb whose source state
does not match must fail closed even when another transition shares its target
state.

## Operations

The HTTP API is the canonical transport boundary:

```text
GET  /api/project-lifecycle
POST /api/project-lifecycle/register
POST /api/project-lifecycle/{project}/pause
POST /api/project-lifecycle/{project}/resume
POST /api/project-lifecycle/{project}/archive
POST /api/project-lifecycle/{project}/restore
POST /api/project-lifecycle/{project}/unregister
POST /api/project-lifecycle/unregister-all
```

Registration validates and loads the authored Project, persists the registry
record, adds it to the live runtime, and indexes its existing Run roots.
Pause, archive, and unregister remove the Project from the runtime's shared
active list, so the collector observes the change between calls without a
daemon restart. Resume reloads the authored catalog before returning the
Project to that list; identity drift fails closed.

Unregister removes only the workspace registry record. Archive keeps the
record but makes it inactive. Neither operation deletes or modifies:

- the Project repository or `research_project.yaml`;
- Campaign, Run, Attempt, or evidence files;
- scheduler jobs or backend storage;
- logs, metrics, checkpoints, or artifacts.

Project-file writes and scheduler mutations remain governed by the separate
Action runtime gates. A lifecycle transition does not authorize either.

Zero-config discovery is separately constrained by `project_import_roots`.
The daemon rejects repositories outside those canonical roots and does not
follow discovered controller, run-root, or Campaign symlinks. Generated
manifest imports are persisted as a recoverable four-phase transaction
(`PREPARED`, `MANIFEST_APPLIED`, `REGISTERED`, `COMPLETED`), so retry after a
process crash rolls forward and a completed import is idempotent.
Non-Git repositories receive a no-follow content digest so files changed after
preview also make the plan stale. Manifest creation is anchored to the opened
repository directory with `openat`/`O_NOFOLLOW` and an fsynced rename; swapping
the reviewed path after validation cannot redirect the write outside the
allowlist.
The temporary manifest name is bound to the import identity; a retry removes
only its own anchored pre-rename residue before rechecking repository identity,
so a process kill between fsync and rename still rolls forward safely.

The daemon's operating-system sandbox must agree with this policy. When
`allow_project_writes` is enabled, every configured import root that may
receive a generated manifest must also be writable by the daemon process. For
example, a systemd unit using `ProtectSystem=strict` needs a matching
`ReadWritePaths=/srv/ml-expd/projects` (or the exact configured roots) in
addition to its state-directory allowance. A filesystem denial is reported as
a blocked, retryable import while the persisted plan remains `PREPARED`; after
the service policy is corrected, the same confirmation can be retried safely.

Client-authored patches use another two-phase API. Preview validates the exact
Git base, patch digest, changed-file declaration, protected paths, and Git
diff without executing Project code. Execution requires
`action_runtime.allow_source_imports` plus the exact confirmation and creates a
daemon-owned, read-only, content-addressed source tree. A Campaign may bind a
new Run by setting its concrete `source_id` to that imported identity; submit
preflight then requires the controller preview manifest to contain the same
source identity.
The `source_id` is the SHA-256 identity of the fully materialized tree rather
than merely the patch request. Metadata retains that tree digest, and every
resolve (including submit preflight) re-hashes the read-only tree and rejects
content drift, writable paths, special files, and absolute or escaping
symlinks. Relative symlinks are accepted only when their resolved target stays
inside the same source tree. Identical previews preserve the original durable
plan instead of resetting its execution phase. A digest-valid tree published
before a crash can finish its plan record without reopening the original Git
repository or import policy, and completed imports remain idempotent. The tree
is resolved again immediately before controller dispatch, after authorization,
so a changed tree cannot reach the scheduler through an older prepare result.
For an imported `source.*` identity the Project controller must declare the
`daemon_source_revision` capability and accept `--source-root PATH --source-id
ID` on dry-run, preflight/assets checks, and submit. The daemon resolves PATH
from its own content-addressed metadata rather than accepting it from a client
intent. Controllers without that capability fail closed before scheduler
authorization.

## Bootstrap and durability

`ServerConfig.projects` seeds an empty registry for compatibility and initial
deployment. After bootstrap, the workspace-owned registry is durable under
`project_registry_root`; normal registration and transitions use the API.
The active in-memory Project list is reconstructed from that registry at
daemon startup.

Registry writes are atomic and lifecycle events are retained for audit. A
registered file whose declared Project identity changes is rejected as
identity drift instead of being silently rebound.

## Client contract

Research Console and other clients render lifecycle state and send explicit
commands, but they do not maintain a second registry or collector. Clients
must not interpret an absent Project from the active list as deleted: consult
`GET /api/project-lifecycle` to distinguish `PAUSED`, `ARCHIVED`, unregistered,
and unknown Projects.
