# Data uploads through the desktop builder

Server 0.3.6 / client 0.1.7 keep protocol 2. Client code, training environment,
training data and results retain separate identities. The client needs only its
API URL and token; it does not need Docker, SSH or CCR credentials.

```text
client data directory → API chunks → desktop staging volume
                                  → data-only image → CCR
                                  → CPU-only ACP copy → verified shared NAS
training environment image + verified NAS data → GPU Run
```

## Client

```bash
ml-exp asset-upload --project my-training --directory ./data \
  --state data.json --executor sensecore-1gpu
# Keep the same directory and state files after a disconnection:
ml-exp asset-upload --project my-training --directory ./data \
  --state data.json --executor sensecore-1gpu --resume
```

Without `--executor`, this command only uploads and validates the asset.
With it, the client builds a separate data image and prepares NAS through an
ACP job using **2 CPUs, 4 GiB memory, zero GPUs**. This does not submit training.
The normal `ml-exp experiment CONFIG.json --state STATE.json` workflow performs
the same preparation for desktop-staged SenseCore inputs before creating a Run.
Keep the adjacent `*.delivery.json` / `*.delivery-N.json` recovery state files.
Inspect a delivery with its saved GET endpoint if publication/submission is
uncertain; explicit reconciliation observes the exact image/job and never
replays an absent submission.

## API and progress

Use the existing resumable `asset-uploads` API and its zero-based part numbers.
The endpoint and authentication do not change. API workers buffer at most four
active parts; each is bounded by the negotiated part size. Parts, complete
archive and extraction validation are on the desktop. Seoul keeps only small
asset receipts, locators, progress records and the existing audit database.
Whole-body `/api/assets/archive` returns 409 when desktop staging is enabled,
so it cannot accidentally spool a large data archive on Seoul.

After upload:

1. `POST /api/projects/P/assets/A/deliveries/prepare` with `{"executor":"sensecore-1gpu"}`.
2. Save `delivery_id` and `confirmation`; call
   `POST /api/projects/P/data-deliveries/D/execute` with the exact confirmation.
3. Query `GET /api/projects/P/data-deliveries/D`: `BUILDING_IMAGE`, `IMAGE_READY`,
   `SUBMITTING`, `QUEUED`, `COPYING`, then `READY`, `FAILED` or `RECONCILE_REQUIRED`.
4. Only a verified READY receipt permits a new SenseCore GPU Run with this input.
   Run identity freezes the delivery/image/file-inventory digest and requires
   `data-cache-required.v1`; rebuild older training images to obtain this worker.

The fixed data image contains the original archive, file manifest and trusted
copy launcher. It never executes uploaded code. BuildKit reads its tar context
on the desktop's private Docker network. The copy job mounts the same NAS as the
selected executor, verifies archive SHA256 and every file, atomically publishes
`<storage_root>/<project>/data-assets/<asset_id>`, and sends an exact, scoped
receipt. A GPU worker verifies this cache and fails if it is absent/corrupt;
it does not fall back to a slow HTTP download. A dataset can be reused for many
Runs. Training environment and data-image identities remain independent.

Historical assets, frozen Runs and READY runtimes remain unchanged. Restaging a
historical dataset records a separate desktop locator. API archive downloads
stream from the desktop without a server cache; desktop assets have no S3
signed link. Historical object-store assets and result links retain their
existing download behavior. WYD can stream desktop assets into its shared
datapool cache using its existing worker; the CPU ACP preparation is SenseCore only.

## Limits and operations

The fixed 4 GiB archive/file/expanded-data defaults are removed. Nullable private
quotas `max_archive_bytes`, `max_asset_archive_bytes`, `max_asset_bytes` preserve
explicit operator limits when set. Read `configured_byte_limits` (null means
no fixed byte quota), `upload_max_parts`, and desktop free/total bytes from
`/api/storage-limits`. Legacy numeric fields use a protocol-compatible integer
ceiling. The default 65,536 parts of 16 MiB permit at most **1 TiB per archive**;
available storage and reserve checks can reject smaller uploads. Source remains
separately bounded at 64 MiB compressed / 256 MiB expanded. Request/manifest
limits, transfer lifetimes and GPU budgets remain unchanged.

This deployment uses **1 MiB desktop parts** because its SSH path is slow and
lossy; the negotiated 65,536-part transport ceiling is therefore **64 GiB per
archive**. Single files and expanded datasets can exceed 4 GiB. Client packing
also needs space on the client's own disk. Increasing part size raises the
transport ceiling but increases the time to retry a part.

Private builder config needs `data_upload_container: ml-expd-data-stage` and
an approved digest-pinned `data_base_image`. Artifact-store config needs
`data_upload_storage: desktop-builder`, `data_upload_socket` pointing to the
existing private builder socket, and a `data_delivery` CPU profile containing
`sco_bin`, `aec2`, `worker_spec`, `gpus: 0`, `cpus: 2`, `memory_gb: 4`, bounded
`copy_timeout_seconds` / `queue_timeout_seconds` (5..3600). Workspace, NAS mount
and data root come from the selected SenseCore executor. The currently verified
CPU spec is `N6lS.Iu.I10.2c4g` on `debug-cluster-01e`; availability is checked
again before every new job. No CCI permissions are needed.

The owned desktop helper uses the existing reverse Docker socket, a private
Docker bridge and separate code/data volumes, with no published host port.
After configuring the private builder, provision its helper while no upload or
build is active: `python tools/provision_desktop_data.py --builder-config
/etc/ml-expd/image-builder.json`. The tool refuses to adopt unrelated containers
or volumes and never deletes the owned data volume. `data_upload_part_bytes`
controls desktop part size (match the artifact-store `upload_part_bytes`).
Keep the desktop Docker service and SSH tunnel up while uploading/building or
downloading a desktop asset. Complete parts survive restarts; expired unfinished
sessions are pruned under locks. Published archives are retained once on the
desktop and are not automatically deleted. BuildKit's exact temporary builder
and volume are removed after push/verification. CCR images, NAS datasets and
published outputs require deliberate retention decisions; no global prune runs.

Worker output/checkpoint uploads still use the existing server/object store.
Removing their fixed byte quota does not move these bytes to the desktop or
create extra capacity on Seoul. Prefer backend-persistent recovery state for
large checkpoints; export selected weights/results through the output API.

Docker's [remote tar context contract](https://docs.docker.com/build/concepts/context/)
places the context download on the builder host. SenseCore
[CCR documentation](https://www.sensecore.cn/help/docs/developer-tools/ccr/ccr_1)
describes registry integration with compute; measured throughput and CPU queue
capacity still determine actual preparation time.
