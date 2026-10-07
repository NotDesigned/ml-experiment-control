# Action transactional storage

Action execution state and its audit event now commit in one SQLite transaction.
`ACTION_ROOT/actions.sqlite3` is authoritative for these two records. Immutable
`plan.json`, reviewed artifacts, private command inputs, and activity errors
retain their existing directory layout and HTTP representation.

SQLite uses WAL, `synchronous=FULL`, a 30-second busy timeout, and owner-private
files. Keep the Action root on local storage. State revisions are checked inside
`BEGIN IMMEDIATE`; two writers cannot both move an authorized Action into
execution. A failed event insert rolls back the state update too. Existing
cross-process Action locks still cover file artifacts and composite operations.
This change does not remove the long lock around remote prepare checks.

## Offline conversion of pre-SQLite data

Current SQLite deployments need no migration command. The explicit offline
converter remains for operator recovery of pre-SQLite records; it is never a
runtime fallback or an instruction to roll back a live database.

1. Stop the daemon and other explicit Action operator processes. Back up the
   whole Action root, runtime environment, configuration, and systemd definition.
2. Run the new environment's command:

   ```sh
   python -m ml_exp_server.actions.migrate --root /path/to/actions
   ```

3. Check the reported count against the original plans; compare statuses,
   revisions, execution results, and audit history on an isolated copy first.
4. Start the new daemon and verify its authenticated read endpoints and logs.

Legacy compatibility belongs only to the explicit migration command. The
runtime reads and writes SQLite and refuses an unmigrated Action instead of
reading its JSON or resetting its state. Unknown read-only queries do not create
empty database records. A new plan interrupted before initial state creation
can still be initialized when it has no legacy execution/journal files.

The migration command imports all missing Action rows in one transaction and
never overwrites an existing SQLite row. Legacy journal repair runs on private
temporary copies; original JSON/JSONL files are retained unchanged. Invalid
state, revision drift, incomplete journal history, or an unreadable plan aborts
the import. An interrupted command can be rerun. The `.sqlite-authoritative`
marker prevents silent fallback if the database disappears. Never delete the
database to fix a failed migration.

Status and caller revision checks live in the single database commit method.
ActionStore delegates ordinary transitions and execution claims to this method;
there is no per-Action SQLite wrapper or second file-based execution-claim API.
The `execution.claim` file remains a best-effort audit artifact. SQLite state
and the same-transaction audit event remain authoritative.

New state changes do **not** update legacy `execution.json` or `journal.jsonl`.
Use the API or ActionStore for current state. SQLite backups must use the backup
API or a stopped-service copy that includes WAL state; copying only an active
`.sqlite3` file is insufficient.

## Roll back after new writes

The simplified runtime retains schema version 1 and the same payload/event
format. Returning to the preceding SQLite runtime does not need a data export.
The following export is needed only for a version that reads legacy JSON.

Stop the daemon and operator processes, keep a current database backup, then use
the **new** environment to export all current state into a new directory:

```sh
python -m ml_exp_server.actions.migrate --root /path/to/actions \
  --export-legacy /path/to/new-rollback-actions
```

The destination must not exist and must be outside the Action root. The tool
copies plans/artifacts/commands and exports every audit event, including those
older than the API's last-100-event view. `rollback-export.json` is written last;
require `complete: true` and verify the count before switching the configured
Action root or replacing it while the service is stopped. Preserve ownership.
Then restore the old runtime and start it. Do not restart old code against stale
legacy files in a migrated root. Do not restore an old backup over newer Actions.

Migration/export only reads or converts durable records. It never authorizes,
submits, cancels, or retries a remote experiment. `EXECUTING` and
`RECONCILE_REQUIRED` remain unchanged for ordinary reconciliation after startup.
