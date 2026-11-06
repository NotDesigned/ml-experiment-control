# Action storage and backup

`ACTION_ROOT/actions.sqlite3` is authoritative for Action execution state and
audit events, committed in one transaction. In a native deployment `ACTION_ROOT` is
`actions/` under the registry root. Each Action directory holds its immutable plan
and frozen provider request. Every Action is a Submission. SQLite uses WAL, `synchronous=FULL`,
a 30-second busy timeout and owner-private files; keep it on local storage.

Revision checks and execution claims run inside `BEGIN IMMEDIATE`. Two callers
cannot both claim a prepared Action, and an event-insert failure rolls back
the state change. Cross-process locks still protect composite file operations.
The `execution.claim` file is a best-effort audit artifact, not execution truth.

## Back up current deployments

Use SQLite's backup API, or stop every writer and copy the complete database
including WAL state. Copying only an active `.sqlite3` is insufficient. Retain
immutable plans/config with the snapshot, plus the exact program version,
digest and recovery instructions. Database recovery backups and live state have
their separate retention policy. Validate restoration on an isolated copy, never
over production.

Program rollback must preserve newer state and check compatibility/active work;
never restore an old DB merely to match an old program. A missing DB with
`.sqlite-authoritative` present fails closed. Do not delete the DB/marker to force
initialization.
