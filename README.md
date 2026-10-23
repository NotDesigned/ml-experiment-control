# ML-Expd

ML-Expd runs reproducible experiments on WYD (Slurm/Apptainer) and SenseCore
(REST/ACP). A remote client uploads code containing a Dockerfile, prepares
data, builds an image, submits a Run and retrieves verified results through
HTTP. Backend credentials stay on the server.

## Start here

| You want to… | Read | You need |
|---|---|---|
| Use an existing API | [Client quickstart](docs/api-quickstart.md) | Python, client wheel, API URL and token |
| Deploy or maintain the server | [Operator guide](docs/operator-guide.md) | Linux, builder, storage and backend access |
| Understand the code | [Architecture](docs/architecture.md) | Repository checkout |
| Embed the core library | [Library integration](docs/library-integration.md) | Core package and a project adapter |
| Contribute | [Development](docs/development.md) | Python 3.12, uv and Rust |

For upgrades, read the [1.0 migration notes](docs/migration-1.0.md).
The [documentation index](docs/README.md) links the complete reference.
Install the independent [client package](client/README.md) on client machines;
the [server package](server/README.md) uses the reusable core, not the client.

## Workflow

```text
code + Dockerfile ── build ── Runtime image
data ── target-backed upload or backend download ── datapool / NAS
                                       │
Runtime + data + argv + resources ── Run ── submit ── Attempt
                                                     │
                            progress / metrics / checkpoint / output archive
```

Data uploads select an executor. The API buffers at most 1 GiB across transfers,
then removes each part after durable destination acknowledgement. WYD receives
data directly on shared datapool; SenseCore receives a CCR data image and copies
it to NAS with a zero-GPU debug job. The desktop builds environment images only.
Output archives still use object storage and are released after verified client
ACK; this is separate from the transfer buffer.

New builds use Dockerfile only. The base-image catalogue supplies pinned `FROM`
choices, not a second build mechanism. Source, image, data and Run definitions
have immutable identities. Each execution has its own Attempt. Training and
evaluation use the same Run mechanism; experiment code defines scientific
metrics, scoring and checkpoint selection.

Runtime READY, Submission VERIFIED, scheduler SUCCEEDED, checkpoint REGISTERED
and downloadable artifacts are different facts. An interrupted upload can leave
a complete NAS checkpoint without a downloadable result. See [recovery](docs/recovery.md).

The current HTTP protocol is 2. Obtain installed versions, capabilities,
executors, policy and limits with `ml-exp check`. Existing READY images keep their
original workers. Executor specifications do not guarantee free GPUs. Automatic
preemption recovery and a general CPU artifact-recovery API are not implemented.

SSH 调试机支持参见 [CCI / DSW 调试会话](docs/debug-sessions.md)。训练仍使用 DLC、ACP 或 Slurm。
