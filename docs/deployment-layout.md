# Single deployment directory

The server checkout, installed environment and operations evidence live together:

```text
/root/ml-expd/
  src/                 backend library
  server/              daemon and builder source
  client/              independent client source
  docs/                current interface and operator documentation
  .venv/               installed core/server runtime used by both services
  .ops/                private deployment and verification evidence
  .recovery/previous/  one verified preceding program recovery point
```

Install the client wheel on client machines. The server runtime does not need
it. Build distributions for release, verify them, and remove local installers
and temporary development environments after installation. Do not accumulate
another checkout or a runtime for each deployment date.

The service templates use `ProtectHome=tmpfs` and a read-only bind of only this
directory. Other home directories stay hidden; `.git`, `.ops` and `.recovery`
must remain 0700. Runtime modules are readable by the service account. Persistent
state and credentials remain separately permissioned:

- `/etc/ml-expd`: private service configuration and credentials.
- `/var/lib/ml-expd`: authoritative Action database, index, immutable Runs,
  source, checkpoint and upload metadata.
- `/var/lib/ml-expd-image-builder`: build receipts, progress and scoped leases.
- `/srv/ml-expd/projects`: existing external project adapters and evidence.
- WYD datapool and SenseCore NAS: data and full recovery state.
- Desktop-owned volumes and CCR: large upload staging, build cache and images.

A directory cleanup does not delete these persistent assets or replace a live
database. Migrate operational `controller.python` bindings before removing old
runtimes. Existing prepared gates bind the old program and must be prepared
again; immutable Runs/Attempts and completed Actions must not be rewritten.
Historical READY images retain their original workers. Historical build reads
use verified receipt/definition identities, never an executable old recipe.

Recover only the preceding program and configuration after confirming that
services and submissions are idle and that the recorded state/configuration
has not changed. Refuse rollback when those guards fail. Never restore an old
production database to make program rollback possible.
