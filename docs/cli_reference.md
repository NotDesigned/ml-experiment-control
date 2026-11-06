# Native client commands

Run `ml-exp --help`, then `ml-exp <group> --help` for the installed version's
exact flags. `--help` lists what most people need; `ml-exp --help-all` hides nothing. The client stores its default URL and token in private config;
commands do not require SSH, Docker or cloud credentials.

| Group | Purpose |
| --- | --- |
| `submit` | Confirm once, then build, send and follow a job; see the [quickstart](api-quickstart.md#run-a-repository) |
| `status`, `logs`, `results` | Read the latest submission from this machine, or one named with `--project` and `--run` |
| `cancel`, `retry` | Ask, then stop the latest submission or run it again |
| `config` | Save or inspect the API connection, and the default Project and executor |
| `check` | Validate the connection and read API capabilities |
| `backend` | List executors, read cached capacity or request a refresh |
| `source` | Upload immutable source or inspect its identity |
| `project` | Register and manage Project tracking |
| `run` | Prepare, submit, observe, cancel or recover an experiment |
| `env` | Prepare and reconcile an immutable Runtime |
| `data` | Upload, inspect, publish or download dataset assets |
| `debug` | Create, inspect and stop a leased SSH debug machine |
| `admin` | Plan and execute administrative removal or Run retirement |
| `update` | Discover and install the latest stable native client |
| `get` | Read a supported exact API resource |

Recovery examples:

```bash
# Discover the retained preparation even before a Run/Submission exists.
ml-exp run preparation --project study --run trial-001

# Local observation only; no provider query or CPU creation.
ml-exp run sync ACTION --request-id REQUEST --status

# Explicit one-child recovery, preserving the original sync request.
ml-exp run sync ACTION --request-id REQUEST --retry-failed-job FAILED_CPU_JOB \
  --confirmation 'RETRY SYNC ACTION REQUEST FAILED_CPU_JOB'

# Prepare PAI CPFS data with an existing READY Runtime and zero GPUs.
ml-exp data publish --project study --executor pai-h200 --asset ASSET \
  --runtime RUNTIME --request-id REQUEST --confirmation 'PUBLISH ASSET'
```

Replace placeholders with saved API identities; request IDs are 32 lowercase
hex characters. Keep the same IDs after an uncertain response. See
[data](data.md), [recovery](recovery.md) and [the HTTP contract](http_contract.md)
for resources, durable evidence and retry behavior.
