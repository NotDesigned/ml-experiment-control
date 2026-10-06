# Download data with a source script

Requires server capability `data-preparation.v1` (server 0.2.9+) and a newly
built Runtime advertising the same capability. Existing images do not acquire
new workers. The client supports this in version 0.1.5 / protocol 2.

Put `download_data.py` beside your training code and Dockerfile. Add this to
the ordinary Run request (`POST /api/projects/P/runs`) or experiment JSON:

```json
{
  "data_preparation": {
    "script": "download_data.py",
    "arguments": ["--version", "dataset-v1"],
    "timeout_seconds": 1200
  }
}
```

`script` is relative to the frozen source root, regardless of Runtime workdir.
It must be a regular file. `interpreter` defaults to `python3`; `/bin/sh` also
supports shell scripts. Arguments are an argv array, never shell interpolation.
The worker uses the Runtime's configured workdir and project environment.
Timeout is 5–86,400 seconds; the job's wall-clock limit also applies.

The download script writes regular files under `os.environ["DATA_DIR"]` and
must finish all downloads before exiting zero. Install its dependencies in
the Dockerfile. Use pinned dataset versions and verify expected upstream file
hashes in the script. For example:

```python
import hashlib
import os
from pathlib import Path
import shutil
import urllib.request

target = Path(os.environ["DATA_DIR"]) / "tokens.bin"
with urllib.request.urlopen(PINNED_DATA_URL, timeout=60) as source, target.open("wb") as output:
    shutil.copyfileobj(source, output, length=1024 * 1024)
with target.open("rb") as source:
    assert hashlib.file_digest(source, "sha256").hexdigest() == EXPECTED_FILE_SHA256
```

Replace the URL and hash with your dataset's values. `hashlib.file_digest`
requires Python 3.11+; older images can hash files in chunks. Training reads
`DATA_DIR` and writes only to `OUTPUT_DIR`:

```python
data = Path(os.environ["DATA_DIR"]) / "tokens.bin"
outputs = Path(os.environ["OUTPUT_DIR"])
```

The download subprocess receives no ML-Expd transfer capability or OUTPUT_DIR,
Run/Attempt identity. It receives DATA_DIR, project/source identity and the frozen
user environment. The source script still runs as project code in its container;
this is not a security sandbox against malicious project code.

## What the platform records and reuses

The Run freezes the script SHA256, interpreter, argv, timeout, source/image,
workdir, user environment and uploaded-input identities. Their canonical JSON
hash becomes `preparation.<SHA256>`; Run IDs and training arguments alone do not
change it. Changing source, image, download parameters or user environment
produces a new preparation identity. Caches are project- and backend-local.

WYD uses `<project_data_root>/data-preparations` on shared `/datapool`;
SenseCore uses the corresponding project directory on its mounted NAS. A lock
serializes preparation of the same identity. The script first writes to a unique
staging directory. Only after successful exit and complete hashing does the
worker atomically publish the dataset tree and READY receipt. Partial downloads
are never reused. Failed attempts remove their owned staging directory; SIGKILL
or host loss can leave unreferenced staging directories for operator inspection.

Every regular file gets a path, byte count and SHA256. Links/devices/FIFOs,
empty datasets and more than 20,000 files are rejected. File contents are streamed,
not loaded into RAM. A `dataset.<SHA256>` identifies the canonical file manifest.
The optional `expected_content_sha256` pins that manifest hash, which is distinct
from the hash of one file or an archive. Its calculation is SHA256 of UTF-8
`json.dumps(files, sort_keys=True, separators=(",", ":"), allow_nan=False)`;
files are ordered by their relative paths. It can pin matching data across
backends or later runs after inspecting a verified first receipt.

Reuse rehashes every file before training. A missing/modified file or corrupt
receipt fails closed; it never silently reruns the script. Data is sealed with
0444 files / 0555 directories. This is filesystem permission protection, not a
kernel-enforced read-only mount; root project code can change permissions.
Change the preparation definition for an intentional new dataset version.

`DATA_DIR` points at the verified persistent tree when training starts.
`data-preparation.json` in the exact Attempt's outputs records READY/FAILED,
the full frozen definition, dataset identity, files, hashes and cache reuse.
This metadata is included even with custom output patterns; dataset bytes
are outside OUTPUT_DIR and are not automatically returned as artifacts.
On preparation failure, training never starts and the worker attempts to return
the failure receipt, then exits nonzero. Result-transfer failure remains separate.

Worker logs report WAITING, RUNNING, VERIFYING, READY or FAILED. The download
script can emit its own byte/transfer progress. The platform does not infer
progress from arbitrary script output or promise retries/partial-file resume.

This first version runs preparation inside the allocated job, so download and
cache verification count against wall-clock/GPU budgets. It does not schedule a
separate CPU preparation task. Backend disk space and connectivity govern direct
downloads; the API host does not stage these bytes. The 4 GiB asset-upload and
artifact archive caps are unchanged and do not cap this backend-local dataset.
Data is not automatically copied to object storage or the other backend. Backend
cache retention and cleanup remain operator-managed, with no automatic pruning.

The existing uploaded `inputs` and checkpoint restoration paths remain available
and can coexist with script preparation. An immutable Run freezes the recipe;
the actual downloaded dataset identity becomes known in its Attempt receipt.
Compare these identities when reporting results from mutable external sources.

## Standalone client

Use the same `ml-exp experiment CONFIG.json --state STATE.json` preparation and
explicit `--execute` workflow. It checks that the script exists locally before
upload/build and that the server advertises this capability. No dataset upload
is needed when the download script supplies all data.

The lower-level command accepts the same JSON:

```bash
ml-exp create --runtime-state runtime.json --run trial --executor wyd-l40s \
  --data-preparation '{"script":"download_data.py","arguments":["--version","dataset-v1"],"timeout_seconds":1200}'
```

Use a new Run ID when changing the definition. Query ordinary Run/Attempt logs
and download artifacts to inspect `data-preparation.json`. Preparing a Run never
runs the script or allocates a GPU; execution retains the existing authorization,
GPU budget and reconciliation rules.
