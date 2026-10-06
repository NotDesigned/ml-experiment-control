# ml-experiment-client

A standalone client for the ml-expd HTTP API. Requires Python 3.10+ and has
**no runtime dependencies**. It installs neither the daemon/core nor Docker,
Rust, SCO or Apptainer. The daemon owns those platform integrations.

Distribution: `ml-experiment-client`; Python package: `ml_exp_client`;
command: `ml-exp`. Client version 0.1.5 speaks protocol 2; client and server
package versions are independent.

## Install only the client

From this repository's reviewed revision:

```bash
python3 -m venv .client-venv
. .client-venv/bin/activate
python -m pip install ./client
ml-exp --version
ml-exp --help
```

Or from GitHub without installing the root/server package:

```bash
python3 -m pip install \
  'ml-experiment-client @ git+https://github.com/NotDesigned/ml-experiment-control.git@<reviewed-commit>#subdirectory=client'
```

Replace the commit placeholder with a revision containing this package. Git is
needed for that installation method, not for normal API use. A prebuilt wheel
can instead be installed with `python3 -m pip install <client-wheel.whl>`.
This package has not been published to PyPI; do not assume a bare name install
will retrieve this implementation.

Repository developers can use `uv run --package ml-experiment-client ml-exp`;
remote clients do not need uv. `python -m ml_exp_client` is equivalent to the
installed command.

## Connect and run

The operator supplies the HTTPS URL, token, approved dependency image digest,
executor ID and budget. Keep the token in a private local file:

```bash
export ML_EXPD_API_URL='https://api.example.org/ml-expd'
export ML_EXPD_API_TOKEN_FILE="$HOME/.config/ml-expd/client.token"
ml-exp check --schema openapi.json
ml-exp init ./my-study
```

`init` writes a tiny dependency-free `train.py` and Dockerfile into a **new** directory. It
works offline and requires no token. The program writes metrics/summary under
the backend-provided `OUTPUT_DIR`; replace it with your training code.
`ML_EXPD_API_TOKEN` is also accepted. The client never prints credentials or
rejects remote plaintext HTTP/redirects. Only resumable archive uploads retry
transient failures; scheduler operations are never automatically replayed.

Use one JSON config with `ml-exp experiment experiment.json --state trial.json`.
It checks code/Dockerfile/data binding/executor configuration, uploads separate
source/data assets, builds or reuses a Runtime, freezes a Run and prepares its
Submission. It allocates no GPU by default. Add `--execute --download-to results`
to authorize that exact submission within the config's GPU-hour budget, watch
real progress and retrieve verified artifacts. Use `--resume` with the unchanged
config/source to continue the same IDs. Uncertain build/scheduler requests require
inspection or explicit receipt-only reconciliation; they are never replayed.

The lower-level `pack` → `create` → `prepare` → `execute` → `watch` → `download`
commands remain available. New `pack` uses `Dockerfile` by default. Install your
dependencies in that file, using approved digest-pinned FROM images from
`ml-exp check`. The ordinary image/environment/requirements selectors have been
removed from new builds. The catalogue still helps choose a base image. Existing
READY Runtime IDs and immutable Runs remain usable with their original capabilities.

`init --base-image <approved-digest> DIRECTORY` writes train.py and Dockerfile.
Without a base argument, replace the Dockerfile placeholder before building.
Data is uploaded separately and mounted under `/inputs`; it never belongs inside
source or the image. Current builds support atomic live checkpoints on both
backends. See the [single-config quickstart](../docs/api-quickstart.md) and
[checkpoint recovery walkthrough](../docs/sensecore-user-workflow.md).

Alternatively, specify `data_preparation` in the experiment config or
`create --data-preparation` to run a source download script into the backend's
persistent `DATA_DIR` before training. See [script data preparation](../docs/script-data.md).

Local state saves use a private, unique temporary file and atomic replacement;
an abandoned `.tmp` file cannot block later saves. After a server restart,
use `ml-exp runtime --state runtime.json --reconcile` to look up the published
receipt. This also works for an interrupted legacy `EXECUTING` state and never
starts a build. If no receipt exists, inspect logs before an explicit build retry.

On servers advertising `multipart-upload.v1`, data uploads use 16 MiB parts
and retain completed parts across disconnections. Resume the same unchanged
directory with `ml-exp asset-upload --project PROJECT --directory ./data
--state data.json --resume`. The saved archive identity must match; modifying
the directory requires a new state file. The default archive and expanded-data
limits are 4 GiB, not a per-part allowance. See [upload API](../docs/multipart-uploads.md).

## Python use

```python
import os
from pathlib import Path
from ml_exp_client import Client, download

token = Path(os.environ["ML_EXPD_API_TOKEN_FILE"]).read_text().strip()
client = Client(os.environ["ML_EXPD_API_URL"], token)
health = client.negotiate()
executors = client.call("/api/executors")
# After inspecting an exact completed Attempt:
# download(client, "my-study", "trial-wyd", "attempt-001", Path("new-results"))
```

The public API exports `Client`, `ClientError`, `source_archive`, and `download`.
`Client.call(path)` performs GET; `data=...` performs JSON POST and `raw=...`
performs raw-body POST. Paths are relative to the API base URL and must begin
with `/api/`. Auth/protocol headers and reverse-proxy prefixes are handled by
the client. Synchronous and asynchronous protocol-2 submissions are supported.
CLI state files retain recovery IDs; a disconnected client does not cancel the
server's job.

`download` uses a short-lived object-storage link on supporting deployments.
It downloads the archive once, verifies SHA256/size and unpacks expected files
locally. The object request has no API token; signed URLs are neither printed nor
saved. HTTP 404 falls back to legacy downloads. See [storage lifecycle](../docs/storage-lifecycle.md).

## Build and test independently

```bash
uv build --package ml-experiment-client
uv run --package ml-experiment-client pytest client/tests -q
```

The client wheel contains its offline source template. Full HTTP integration
tests are in `server/tests/test_api_quickstart.py`; they use real loopback HTTP
and inject only the external registry/scheduler/object-store effects. CI also
installs the client wheel alone into an empty environment and invokes the CLI
from outside the repository.
