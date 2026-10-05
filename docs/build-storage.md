# Builder storage and progress diagnostics

The private builder can enforce a storage preflight by setting
`build_storage_path` in its administrator-only configuration. This must be the
filesystem holding Docker's BuildKit state volumes (normally
`/var/lib/docker`), not a staging or temporary directory on a different mount.
The check runs inside the private build lock, after owned stale-builder recovery
and before creating a new builder or downloading image layers. READY receipts
are reused without requiring rebuild space.

Pinned base manifests are inspected without downloading their layers. Defaults:

- Required bytes: four times the sum of compressed external base layers, twice
  the source build-context bytes, plus 2 GiB reserved for other services.
- Required free inodes: 100,000.
- Administrator overrides: `build_expansion_factor`, `build_reserve_bytes`,
  `build_min_free_inodes`.

This is a conservative budget, not measured uncompressed size or a guarantee
for arbitrary Dockerfile RUN commands. CUDA images can require much more space
than their compressed layers. If metadata cannot be inspected, the configured
preflight fails closed. The operator must provision sufficient build storage;
cleanup alone does not expand a filesystem. Never prune shared images or
rollback images to make a build fit.

`GET /api/projects/{project}/runtimes/{runtime_id}` and its `/progress` endpoint
return `build_error` with a code and safe structured details:

- `BUILD_STORAGE_INSUFFICIENT`: available/required bytes and inodes; no build
  publication or scheduler submission occurred.
- `BUILD_STORAGE_UNCHECKED`: base manifest or filesystem could not be checked.
- `BUILD_DISK_EXHAUSTED`: ENOSPC occurred during execution; publication may be
  uncertain and must be reconciled before another build.

The first two mark `retry_safe: true`: after fixing the underlying issue,
explicitly execute the same frozen Runtime. They retain the historical
`RECONCILE_REQUIRED` lifecycle status for protocol-2 client compatibility. They
never automatically repeat a build or allocate a GPU. The last marks retry
unsafe; use receipt reconciliation. Builder/container/state-volume cleanup is
restricted to the exact recorded ephemeral builder on success and failure.

## Submission versus workload progress

After a submission is VERIFIED, `/api/submissions/{id}/progress` reports the
current scheduler phase and message. Historical preparation clocks and events
are retained under `submission_progress`; they must not be interpreted as
training-stall timers. The current phase start/duration are unknown unless there
is actual phase evidence. `scheduler_observed_at` describes the scheduler
observation, not a process heartbeat.

`workload_evidence` provides identity-bound worker, process and model states
with their own observation times, stale flags and OBSERVED/NOT_OBSERVED
availability. Absence of a process heartbeat does not establish training health
or failure. Queue estimates remain null when the backend cannot provide one.
A scheduler RUNNING/STARTING state does not prove that data preparation ended
or a training child started. This release does not retrofit stage telemetry into
already built immutable worker images.
