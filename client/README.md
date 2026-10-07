# ML-Expd client

`ml-experiment-client` is independently installable, with no runtime dependency
outside stdlib Python. It provides `ml-exp` and `python -m ml_exp_client`.
The server/core, local Docker, SSH and backend credentials are unnecessary.

Follow the [quickstart](../docs/api-quickstart.md) for installation, API/token
configuration and an experiment. The operator supplies connection information.

## Python use

```python
import os
from pathlib import Path
from ml_exp_client import Client, download

token = Path(os.environ["ML_EXPD_API_TOKEN_FILE"]).read_text().strip()
client = Client(os.environ["ML_EXPD_API_URL"], token)
health = client.negotiate()
executors = client.call("/api/executors")
# After checking the exact Attempt:
# download(client, "my-study", "trial-001", "attempt-001", Path("new-results"))
```

Exports: `Client`, `ClientError`, `source_archive`, `download`. `call(path)` is
GET; `data=...` is JSON POST; `raw=...` is raw POST. Paths begin `/api/` and are
relative to the API base. Auth/protocol headers and proxy prefixes are handled.

CLI state preserves IDs. Ordinary requests default to 60 seconds; multipart
PUT/completion use 1200 seconds. Only archive transfer retries transient failures.
Uncertain scheduling is reconciled without replay. Signed object downloads carry
no API Authorization and are hash-checked into a new directory.
See [development](../docs/development.md) for independent build/tests.
