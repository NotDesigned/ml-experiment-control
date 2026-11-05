# Action storage and backup

`ACTION_ROOT/actions.sqlite3` is authoritative for Action execution state and
audit events, committed in one transaction. In a native deployment `ACTION_ROOT` is
`actions/` under the registry root. Plans, reviewed artifacts and private command
inputs keep their immutable directory layout. SQLite uses WAL, `synchronous=FULL`,
a 30-second busy timeout and owner-private files; keep it on local storage.

Revision checks and execution claims run inside `BEGIN IMMEDIATE`. Two callers
cannot both claim an authorized Action, and an event-insert failure rolls back
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

## Pre-SQLite records

The native service has no conversion command. It refuses an Action directory that
holds legacy `execution.json` or `journal.jsonl` without a database row, rather than
reading it or resetting execution. The retired Python conversion and rollback-export
commands are in this page at tag `v1.5.1`.
