# Action storage and backup

`ACTION_ROOT/actions.sqlite3` is authoritative for Action execution state and
audit events, committed in one transaction. Plans, reviewed artifacts and
private command inputs keep their immutable directory layout. SQLite uses WAL,
`synchronous=FULL`, a 30-second busy timeout and owner-private files; keep it on
local storage.

Revision checks and execution claims run inside `BEGIN IMMEDIATE`. Two callers
cannot both claim an authorized Action, and an event-insert failure rolls back
the state change. Cross-process locks still protect composite file operations.
The `execution.claim` file is a best-effort audit artifact, not execution truth.
Legacy `execution.json`/`journal.jsonl` are not updated by current transitions.

## Back up current deployments

Use SQLite's backup API, or stop every writer and copy the complete database
including WAL state. Copying only an active `.sqlite3` is insufficient. Retain
immutable plans, config and a verified preceding program with the snapshot.
Validate restoration on an isolated copy, never over production.

Current SQLite installations need no conversion. Program rollback must preserve
newer state and check compatibility/active work; never restore an old DB merely
to match an old program. A missing DB with `.sqlite-authoritative` present fails
closed. Do not delete the DB/marker to force initialization.

## Offline conversion of pre-SQLite records

This is an explicit operator tool, not a runtime fallback:

1. Stop the daemon and other Action writers; back up the whole root and runtime.
2. Validate conversion on an isolated copy:

   ```bash
   python -m ml_exp_server.actions.migrate --root /path/to/actions
   ```

3. Compare plan counts, statuses, revisions, results and events before switching.
4. Start the reviewed runtime and verify authenticated read endpoints.

Import adds missing rows in one transaction and never overwrites existing ones.
Malformed state, incomplete history, revision drift or unreadable plans abort
the import. Legacy repair uses temporary copies and retains originals unchanged.
Interrupted import can be rerun. Runtime access refuses unmigrated legacy state
rather than reading it or resetting execution.

Only if a deliberately chosen old version requires legacy JSON, use the new
runtime's offline export into a new directory outside the Action root:

```bash
python -m ml_exp_server.actions.migrate --root /path/to/actions \
  --export-legacy /path/to/new-rollback-actions
```

Require the final `rollback-export.json` to report `complete: true`; verify
counts, ownership and all events before switching a stopped service. Current
SQLite versions do not need this export. Neither migration nor export submits,
cancels or retries remote work. EXECUTING/RECONCILE_REQUIRED remain subject to
normal exact-identity reconciliation.
