# Upgrade to 1.0

The three independently installable packages remain `ml-experiment-client`,
`ml-experiment-server` and `ml-experiment-control`. HTTP protocol remains **2**.
Use the matching 1.0 client and server for new target-backed uploads. Existing
Run/Attempt identities, Action authorization and saved client state are retained;
this upgrade does not rewrite research configuration or replay pending jobs.

## Client changes

- Standalone data upload now requires `--executor`. The one-config `experiment`
  workflow selects its executor before uploading `inputs[].directory`.
- Custom HTTP clients must create multipart sessions with `target.executor` and
  archive SHA/size. SenseCore also requires file inventory and transport-layer
  hash; the standard client computes and sends them for target-backed uploads; see [data](data.md#multipart-http-contract).
- `POST /api/assets/archive` is removed. Creating an asset-upload session without
  a target is rejected. Existing frozen assets and saved sessions keep their
  original identity; do not invent a new upload ID after an uncertain response.
- New datasets have API metadata and backend storage references, not an API
  download archive. `/archive` and `/download` return typed 404
  `DIRECT_DOWNLOAD_UNAVAILABLE` for target-backed assets. Output downloads and
  exact-Attempt verified ACKs are unchanged.

```bash
ml-exp asset-upload --project my-study --executor wyd-l40s \
  --directory ./data --state data.state.json
# After interruption, use the same command and add --resume.
```

A dataset copied to another storage namespace needs a matching upload/delivery
there. The same content identity can record multiple backend locations; it does
not imply that WYD datapool and SenseCore NAS share files.

## Operator changes

1. Retain the reviewed program/config rollback point and consistent SQLite
   backups. Install staged core/server wheels; distribute the client separately.
2. Verify WYD SSH access to the configured shared `storage_root` beneath
   datapool. New data parts go directly there, without desktop staging.
3. Configure SenseCore `data_registry` with a same-repository immutable base,
   daemon-readable Docker auth and exact blob redirect host allowlist. Configure
   the RESERVED 2CPU/4GiB/0GPU debug `data_delivery` profile. The complete
   [configuration example](operator-guide.md#configure) contains no credentials.
4. Keep the desktop builder/tunnel for Runtime construction. Do not remove old
   desktop data volumes merely because new uploads bypass them: historical
   assets and unfinished sessions can still refer to those bytes.
5. Observe `/api/storage-limits`: `data_upload_storage` is `target-executor`, and
   `transfer_buffer` reports the global 1 GiB conservative budget. Part-size,
   count, archive quotas and actual destination capacity still apply.

The buffer deletes a part only after durable matching destination ACK. It may
retain unconfirmed bytes for reconciliation and return 503 backpressure. This
**does not cap all server disk at 1 GiB**: Garage outputs, source snapshots,
result recovery, metadata and journals have separate lifetimes. Verified client
ACK with release enabled makes an output archive immediately eligible for the
maintenance cycle; KEEP/unacknowledged archives and backend state remain.

## Preserved scientific and execution contracts

Metrics are project-defined schemas with explicit units and protocol/checkpoint/
variant context; no ELF metric names or research question are required. Platform
records remain durable independently of optional W&B publication. Training and
evaluation use the same Run mechanism.

New SenseCore download-script Runs freeze a narrow receipt callback wrapper and
its isolated Python launcher identity. It verifies actual NAS bytes and the original CPU job identity without depending on
stdout indexing. Old inside-job/stdout receipt modes remain frozen. This change
does not add automatic preemption recovery or repair unknown historical failures.

Internal code now has one first-submission builder, direct concrete project
services, shared HTTP leaves and a canonical process runner. These changes keep
Action gates/locks, explicit budgets, uncertain-effect reconciliation, Run fences
and administrative deletion guards. See [architecture](architecture.md).
