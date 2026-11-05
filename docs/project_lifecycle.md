# Project registration and lifecycle

This is the operator registry for indexing and polling, not a prerequisite
research workflow. The daemon's workspace-local registry selects projects to index
and poll. Each record binds a stable identity to an absolute Project file on the
daemon host (see the [example](../rust/deploy/project.example.json)), lifecycle,
source, timestamps and operator reason. A research question is not required or
loaded. Use `ml-exp project` to register and change state; `project register` takes
a daemon-host path and uploads nothing. Source import uploads to an approved Project.

The authored file owns title, Run roots and optional Campaign catalogue. Registry removal does not delete that file, source, Runs, Attempts,
datasets or artifacts, and never stops scheduler jobs.

## States

| State | Live indexing/collection | Allowed next states |
|---|---|---|
| ACTIVE | Enabled, subject to policy | PAUSED, ARCHIVED |
| PAUSED | Disabled | ACTIVE, ARCHIVED |
| ARCHIVED | Disabled; retained as history | PAUSED |

Archive requires a reason. Restore first returns to PAUSED, then explicit resume
enables ACTIVE collection. Pausing/archiving does not cancel a running job; issue
an exact-Attempt cancel separately when intended. Each workspace has its own
registry; do not run two live collectors against the same project without a
controller designed for it.

## Routes

Send normal auth/protocol headers and use the installed OpenAPI request bodies:

```text
GET  /api/project-lifecycle
POST /api/project-lifecycle/register
POST /api/project-lifecycle/PROJECT/pause
POST /api/project-lifecycle/PROJECT/resume
POST /api/project-lifecycle/PROJECT/archive
POST /api/project-lifecycle/PROJECT/restore
POST /api/project-lifecycle/PROJECT/unregister
POST /api/project-lifecycle/unregister-all
```

These operations are separately policy-gated. Unregister only changes registry
membership; it is not asset cleanup. Registration checks configured allowed
roots, regular files and protected paths. Project imports and reviewed writes
have their own durable transaction/recovery rules; do not edit managed immutable
Run files to bypass them. Registration executes no project code.

See [source API](source-api.md) for managed imports and
[deployment](operator-guide.md) for writable roots, policy and backups.
