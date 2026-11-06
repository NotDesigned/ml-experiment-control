# Data inputs and transfers

Datasets, source code, persistent checkpoints and exported results have separate
identities and lifecycles. Downloading results does not remove datasets or
checkpoint state. The API stores transfer metadata and bounded part buffers;
large dataset payloads live on backend storage or the configured object store.

## Upload and reuse

Select an executor from `ml-exp backend`, then upload regular files:

```bash
ml-exp data upload --project study --executor pai-h200 --directory ./data
ml-exp data list --project study --executor pai-h200
ml-exp data show --project study --executor pai-h200 --asset asset.ACTUAL_HASH
```

The native client creates a deterministic plain tar in a temporary local file,
then sends 16 MiB parts. It needs client-side space for that archive. Repeating
an upload of unchanged bytes queries the same asset and skips accepted parts;
there are no `--state` or `--resume` options on `data upload`. Keep the returned
asset ID when a response is lost. A changed file creates a different identity.

Run inputs use the ordinary native `data` field:

```json
{"data":[{"name":"training","asset_id":"asset.ACTUAL_HASH"}]}
```

The worker supplies `DATASETS_JSON`, mapping each input name to its verified
shared-storage directory. Training code reads the named directory from this map.

`ARCHIVED` means the uploaded archive passed verification. `READY` additionally
means the selected backend has a verified shared-storage tree. These states are
not interchangeable.

| Backend | Delivery before training |
| --- | --- |
| Slurm | Client → bounded API buffer → connected SSH gateway → shared storage. Offline compute nodes read the shared cache. |
| SenseCore | Client → bounded API buffer → CCR data image → debug CPU ACP → NAS. A CPU receipt establishes readiness. |
| PAI | Client → bounded API buffer → personal OSS objects → verified personal CPFS cache. Import runs before training, optionally in a separate CPU job. |

A cache is scoped to its executor/storage identity. An asset's presence on one
backend does not prove readiness on another. Partial trees and hash mismatches
are not accepted as ready data. Dataset directories remain outside results.

## PAI preparation using an existing image

For PAI, an archive-verified asset can be used directly in the Run's `data`
field even when the API has no READY receipt yet. The worker imports and
verifies it on CPFS before starting training, or reuses an exact verified cache.
This inline preparation consumes the allocated job's running time and resources.
A Run's dataset manifests must fit 256 KiB together, and the full worker package,
including checkpoint metadata, must fit 8 MiB. A larger set is refused with 422
`DATASET_MANIFESTS_TOO_LARGE` when the Run is created or prepared, before any CPU
copy job. `ready:null` alone does not prove that a physical cache is absent.

A separate CPU publication avoids doing that work on allocated training GPUs.
It is optional for this inline path:

Save one request ID before the first call and reuse a READY Runtime:

```bash
ml-exp data publish --project study --executor pai-h200 --asset asset.ACTUAL_HASH \
  --runtime runtime.ACTUAL_HASH --request-id YOUR_32_HEX_REQUEST \
  --confirmation 'PUBLISH asset.ACTUAL_HASH'
ml-exp data show --project study --executor pai-h200 --asset asset.ACTUAL_HASH
```

This uses 0 GPUs, 2 CPUs and 4 GiB, with a 3600-second running limit. It creates
no training Run or image build. Worker17 images first run `data-import
--bundle-output` into the CPU container's temporary archive, then `data-copy`
from that local file into CPFS. The temporary archive/configuration is removed
on exit. During the second phase, copying parts does not fetch signed URLs.

The complete command, channels, Runtime and request are retained before any
provider effect. Repeating the same call observes the same CPU job, including
after an API restart or launcher/URL change. Failed or uncertain jobs are never
replaced automatically. `data show` reads cached evidence without querying the
provider. Only a complete verified READY receipt allows later training to reuse
the cache without another download.

The launcher and worker keep the config in a private scratch file. On the provider
request it travels in arguments of at most 64 KiB, and a request above 1 MiB is
refused before any provider call. Whether a provider passes the joined command to a
shell as one argument, which Linux limits to 128 KiB, has not been verified.

## Replace a failed copy

Repeat the original request until `data show` reports `new_request_allowed: true`.
That means the last observation proved this exact job FAILED, was CANCELLED or was
PREEMPTED. Then publish with a new request ID, on PAI or SenseCore. A fresh provider
read must prove the same thing again before the new job is prepared.

Anything else returns `DATA_COPY_IDENTITY_CONFLICT` and creates nothing: a job that
was never prepared or never started, a submission with no provider record (absence
cannot prove an uncertain create did not land), a failed read, a running job, or
SUCCEEDED without READY. The worker exits non-zero unless READY was acknowledged, so
that last case is contradictory evidence.

The new job has its own deterministic identity from the new request ID. On PAI,
`--runtime` may name another READY Runtime; without it the original Runtime is kept.
The old job stays retained for `data diagnose`. Publishing the old ID again returns
`DATA_COPY_REQUEST_SUPERSEDED` and creates nothing. `experiment continue` shows the
failure and never chooses a new ID. READY still requires the worker's verified receipt.

## Read-only diagnostics

For an existing terminal PAI copy, retain its asset and publication request:

```bash
ml-exp data diagnose --project study --executor pai-h200 \
  --asset asset.ACTUAL_HASH --request-id ORIGINAL_REQUEST
# Read the retained diagnostic without contacting the provider:
ml-exp data diagnose --project study --executor pai-h200 \
  --asset asset.ACTUAL_HASH --request-id ORIGINAL_REQUEST --status
```

The first command creates or observes one deterministic diagnostic under that
original publication. Its complete intent is saved before submission; repeated
calls and response loss do not create another job. The diagnostic uses the
original image, 0 GPUs, 2 CPUs and 4 GiB, and a 300-second running limit. CPFS is
mounted read-only. It does not run data-copy, unpack files, publish READY, remove
parts or retry the original copy.

SenseCore has no read-only diagnostic and no provider read for a copy. There,
`data diagnose` returns 422 `READONLY_DIAGNOSTIC_UNSUPPORTED` and the provider read
says `PROVIDER_DIAGNOSTICS_UNSUPPORTED`, with no provider call.

The report contains current available bytes/inodes, marker/state identity and
metadata hashes, modes/ownership, recorded part counts, part sizes, and an
existing-lock nonblocking check. Part contents are not read. Up to 128 part
entries are included with an explicit truncation flag. A missing lock is never
created. `statvfs` does not prove user-quota availability. Missing offline
logs stay `AWAITING_OFFLINE_LOGS`; a later explicit observation can retrieve them.

To read what the provider recorded for the copy and its diagnostic, add `--provider`
(with optional `--lines`, up to 2000). This returns each job's reason, pods, events
and a redacted log tail, creates nothing, and stores nothing. The worker logs
`ML_EXPD_DATA_IO_ERROR` with the failing check.

## Existing OSS dataset objects

When the executor enables native OSS import, register a complete plain tar in
an approved personal or public dataset prefix:

```bash
ml-exp data import-oss --project study --executor pai-h200 \
  --object-key users/USER/datasets/version-1.tar \
  --sha256 ARCHIVE_SHA256 --archive-bytes ARCHIVE_BYTES
ml-exp data download --project study --executor pai-h200 \
  --asset asset.ACTUAL_HASH --output ./downloaded-data
```

Import streams and verifies the archive and member inventory; it is not a claim
that CPFS is ready. Use separate CPU publication or the PAI inline import above.
Downloads obtain object-part tickets, verify the archive and every file, and
publish into a new local directory. Dataset downloads send no result ACK and
never delete the OSS original. Bucket, endpoint, prefix and credentials are
operator configuration; credentials are not sent to clients. Native OSS
filesystem mounts remain disabled; CPFS supplies the training filesystem.

## Native HTTP endpoints

- `GET/POST /api/projects/P/datasets/E`: list assets or begin a definition.
- `GET /api/projects/P/datasets/E/A`: retained state.
- `PUT /api/projects/P/datasets/E/A/parts/NUMBER/SHA256`: exact raw part bytes.
- `POST /api/projects/P/datasets/E/A/complete`: verify the uploaded archive.
- `POST /api/projects/P/datasets/E/A/publish`: explicit backend preparation.
- `GET/POST /api/projects/P/datasets/E/A/publications/REQUEST/diagnostic`:
  retained diagnostic or explicit create/observe.
- `POST /api/projects/P/datasets/E/import-oss`: object key, SHA256 and archive bytes.
- `POST /api/projects/P/datasets/E/A/download-ticket`: one part's private ticket.

Part numbers start at zero. Identical parts are idempotent; changed bytes cannot
replace an accepted part. The native policy uses 16 MiB parts and at most 65,536
parts, subject to file-inventory bounds and actual destination capacity. No
4 GiB total-size limit or desktop staging is required. Metadata, hashes and
backend-specific readiness remain separate from payload retention.
