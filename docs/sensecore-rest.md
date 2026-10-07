# SenseCore REST backend

Both GPU training and CPU-only data-image delivery use signed HTTPS requests.
SCO is not installed or invoked by this adapter. Configuration follows the
HMAC `X-Date` flow used by SLAI-sensecore-tool, with a daemon-owned client rather
than importing its UI or account configuration.

Create `/etc/ml-expd/sensecore-rest.json`, readable only by the daemon and root:

```json
{
  "access_key_id": "YOUR_AK",
  "access_key_secret": "YOUR_SK",
  "subscription_name": "YOUR_EXISTING_SUBSCRIPTION",
  "resource_group_name": "default"
}
```

`EXPERIMENTCTL_SENSECORE_REST_CONFIG` overrides the private file location.
Do not put this file or its values in an executor, Run manifest, image,
client configuration or Git. Restart the daemon after an intentional key change.
The API/client Bearer token and registry credentials are independent.

| Operation | REST service and path |
| --- | --- |
| Authentication | IAM `GET /iam/idp/v1/me` |
| Workspace/storage/log station discovery | RM `GET /rmh/v1/resources` |
| Pool bindings | Workspace `GET …/workspaceAEC2Bindings` |
| Resource specifications | AEC2 `GET …/aec2s/P/resourceSpecs` |
| Exact job identity/reconciliation | ACP `GET …/trainingJobs?filter=name=…` and `GET …/trainingJobs/J` |
| Submit | ACP `POST …/trainingJobs?training_job_name=J` |
| Stop | ACP `POST …/trainingJobs:batchStop`, containing only the exact job name |
| Workers | ACP `GET …/trainingJobs/J/workers` |
| Logs | Monitor log-livestream token and bounded HTTPS log polling |

The workspace zone and pool zone are resolved separately. Jobs retain their
existing workspace, pool, immutable image digest, AFS volume/subdirectory and
submission-token-qualified name. Every observed exact job must belong to the
configured account. Complete pagination is required before absence can be
reported. Foreign owners, malformed responses and scope drift fail closed.

The CPU data-copy task still uses the configured debug pool, verified
2 CPU / 4 GiB / 0 GPU specification and reserved quota. Its requests are 2 CPU
and 3 GiB; limits are 2 CPU and 4 GiB. Its sealed NAS callback and data-delivery
identity are unchanged. A legacy `sco_bin` field can remain in frozen definitions
and existing delivery profiles, but is never used to execute a command.

Requests verify TLS, bypass ambient proxy settings, reject redirects and reject
endpoints outside SenseCore domains. There is no write retry or CLI fallback.
A timeout, invalid create acknowledgment or server-side write failure requires
exact reconciliation; it cannot cause an automatic second job.
`EXPERIMENTCTL_SENSECORE_CREATE_TIMEOUT_SECONDS` defaults to 120 (10–600).
The previous SCO timeout variable is accepted as a compatibility fallback.

Public errors include only the HTTP status and bounded provider reason, never
raw responses, commands, signed URLs or account keys. Log text still passes
through the existing `experiment-safe-sco` Rust redactor; this is a local text
utility and sends no cloud requests. A provider permission failure is distinct
from an explicitly expired log token. Missing logs do not prove training failed.

The migration does not rewrite historical Runs, Attempts, Actions or upload
sessions, and does not add a cloud-side retry policy (`backoff_limit: 0`). A
transport-only rollout is checked using live read-only queries and isolated
create/stop contract tests; a new GPU training run requires its own acceptance
scope.

Primary contracts: [ACP create](https://www.sensecore.cn/help/docs/API/acp/training-job-service-create-training-job),
[ACP list](https://www.sensecore.cn/help/docs/API/acp/training-job-service-list-training-jobs),
[Workers](https://www.sensecore.cn/help/docs/API/acp/training-job-service-list-workers),
[Log token](https://www.sensecore.cn/help/docs/developer-tools/cms/openapi/sign-log-livestream-token).
