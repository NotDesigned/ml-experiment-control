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

## Upgrade

1. Stop the daemon and other explicit Action operator processes. Back up the
   whole Action root, runtime environment, configuration, and systemd definition.
2. Run the new environment's command:

   ```sh
   python -m ml_exp_server.actions.migrate --root /path/to/actions
   ```

3. Check the reported count against the original plans; compare statuses,
   revisions, execution results, and audit history on an isolated copy first.
4. Start the new daemon and verify its authenticated read endpoints and logs.

Legacy records are imported once per Action. Legacy journal repair runs on
private temporary copies; original JSON/JSONL files are retained unchanged.
Invalid state, revision drift, an incomplete journal history, or an unreadable
plan stops explicit migration. An interrupted migration can be rerun: committed
rows remain authoritative, and uncommitted rows are imported on the next try.
The `.sqlite-authoritative` marker prevents silent fallback to stale files if
the database disappears. Never delete the database to fix a failed migration.

New state changes do **not** update legacy `execution.json` or `journal.jsonl`.
Use the API or ActionStore for current state. SQLite backups must use the backup
API or a stopped-service copy that includes WAL state; copying only an active
`.sqlite3` file is insufficient.

## Roll back after new writes

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
