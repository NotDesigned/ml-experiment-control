# Delete a managed Run

`ml-exp delete` prepares permanent administrative deletion of one exact managed
Run, including all its Attempts. It requires the server's
`resource-deletion.v1` capability, the private API bearer credential and enabled
`action_runtime.allow_resource_deletions` policy. That policy defaults to false;
a disabled policy returns a blocked plan with reviewable gates.
Only daemon-managed Runs are deletion targets. Older imported or indexed Runs
remain queryable under their historical identity rules; this endpoint does not
infer permission to delete their arbitrary storage locations.

```bash
ml-exp delete --project my-study --run trial-001 \
  --reason 'Remove this completed experiment and its retained storage' \
  --state trial-001.deletion.json

# Inspect the returned targets, digest, gates and retained resources first.
ml-exp delete --state trial-001.deletion.json --execute

# Observe the same Action after disconnecting or while it is still running.
ml-exp delete --state trial-001.deletion.json
ml-exp delete --state trial-001.deletion.json --reconcile
```

Preparation does not delete anything. The second explicit `--execute` invocation
authorizes and dispatches the saved, reviewed Action. The CLI does not cancel
running jobs automatically. Active or uncertain Attempts and checkpoints
referenced by another Run block deletion; resolve those gates and prepare a new
plan with a new state file. The reviewed targets must still match when execution
begins.

WYD/Slurm deletion uses the frozen SSH storage origin. SenseCore cleanup uses
the configured debug-cluster reserved 2CPU/4GiB/0GPU profile and requires enabled
scheduler mutations for that bounded cleanup job. The accepted receipt and exact
job UID are retained; an uncertain CPU create is never repeated automatically.

The target manifest covers the Run's managed backend outputs, persistent state
and backend-resident persistent checkpoints, server experiment records and result
caches, and current and historical published result objects. Administrative
authorization can delete
unacknowledged archives and archives marked `KEEP`; it records a distinct
administrative deletion proof and never fabricates a download acknowledgement.
This is different from `acknowledge`, which verifies local download proofs and
releases the matching archive while retaining experiment records and persistent
checkpoints.

Client downloads remain on the client's computer. Shared source, images and data
assets remain available. Uploaded checkpoint snapshot assets follow the separate
shared-asset lifecycle and are retained in this release; deleting backend-resident
persistent checkpoints does not delete those uploaded backups. W&B cloud records
are retained. The display schema for each W&B destination also remains so future
Runs cannot reuse a curve name for a different unit, protocol or dataset. These
mappings contain no metric values. Provider job history deletion is unsupported
and is explicitly reported under
`execution.result.retained.provider_history`; it is not counted as reclaimed
storage. The server retains the original Action records, download ACK proofs and
complete current and historical object receipts for audit, plus a deletion
tombstone to prevent the collector or a late worker callback from recreating the
deleted Run. This operation is resource
deletion with retained audit evidence, not complete erasure of every Run reference.
Existing operator backups and small operation audit journals follow their
separate retention policy; this endpoint does not rewrite historical backups.
The permanent write fence is installed before external deletion begins. A 410
on a Run query means that its identity is fenced, not that all cleanup has
finished; use the Action's verified completion below. Object-store physical
garbage collection and NAS quota counters may lag successful logical deletion.

Deletion is complete only when `action.execution.status` is `VERIFIED` and its
result contains `status=DELETED`, the exact project and Run, and the reviewed
`target_digest`. The CLI validates these fields before returning `deleted=true`.
`EXECUTING`, `RECONCILE_REQUIRED`, `FAILED` and blocked plans are not deletion
completion. An execution invocation exits with code 2 until deletion is verified;
its JSON still contains the saved Action and the next operation to perform.

## Lost replies

The client saves the request, a new idempotency key and the exact server/Run
before sending preparation. If that reply is lost, using the same state file
reports an unknown preparation instead of resending it. Query the recovery URL
printed by the client, find the Action with the saved request's exact
`intent_id`, and attach it for review:

```bash
ml-exp delete --state trial-001.deletion.json --action-id action-ACTUAL_16_HEX_ID
ml-exp delete --state trial-001.deletion.json --execute
```

Attachment checks the original request and exact Run; it does not execute. The
client also saves authorization and execution request markers before dispatch.
After a lost authorization or execution reply, observe or reconcile that same
Action. Do not create another deletion plan to replay an uncertain request.
The server journals each frozen target separately so reconciliation can continue
the same deletion after interruption without broadening its scope.
For WYD/SSH, an uncertain destructive command is not sent again automatically:
reconciliation checks that exact target for absence. If its absence cannot be
confirmed, the Action remains `RECONCILE_REQUIRED` for investigation.

## HTTP contract

`POST /api/runs/{project}/{run}/deletion` accepts:

```json
{"reason": "Remove this completed experiment", "idempotency_key": "deletion-client-generated-key"}
```

It returns an Action directly, with `operation=DELETE_RUN_RESOURCES`, exact Run
scope, `intent_id` equal to the idempotency key, `reason`, `deletion_request`,
`target_manifest`, `target_digest`, `gates`, `ready`, sealed review digests and
durable `execution`. The same key and request identify the same Action; a changed
request with that key conflicts.

Use the existing `POST /api/actions/authorize`, `POST /api/actions/execute`,
`POST /api/actions/reconcile` and `GET /api/actions/{action_id}` routes for the
remaining stages. Execution confirmation is exactly `EXECUTE {action_id}`.
`GET /api/actions?project={project}&scope_type=run&object_id={run}` locates a lost
preparation by exact `intent_id`. Worker upload capabilities do not authorize
these administrative endpoints.
