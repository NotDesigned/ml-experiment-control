# Client and server boundaries

The 1.0 refactor keeps one repository with independently installable client,
server and core packages. Its goal is fewer concepts and one implementation per
responsibility, while preserving frozen scientific and execution identities.
See [architecture](architecture.md) for the current module map and call graphs,
and [migration](migration-1.0.md) for changed public upload behavior.

## Responsibilities

- The client separates HTTPS, archives, data upload, verified results and saved
  state. `cli.py` parses/prints; concrete commands and domain workflows advance
  explicit operations. The client requires neither server/core nor Docker/SSH.
- The server separates HTTP/auth, daemon lifetime and domain services. Existing
  workspace ownership, recovery order and five bounded execution lanes remain.
- Action planning uses concrete source, Campaign, preview and resource gates.
  One first-submission builder binds source, Attempt and budget; the Submission
  service retains its reuse lock and idempotency policy.
- Routes call concrete project services without a forwarding mixin. Shared
  transfer/error helpers do not import routers. Process-group guarantees use the
  canonical core runner where needed.
- Legacy project-layout guesses are read-only. Managed Runs use their generic
  frozen configuration and project-defined metric schema.

No plugin loader, generic workflow engine, second scheduler or new experiment
journal was added. The six public client exports, explicit authorization,
uncertain-effect no-replay rules and existing durable states are retained.

## Data boundary

New datasets select backend storage at upload creation. One shared 1 GiB disk
buffer bounds in-flight parts; matching durable destination ACK permits local
part removal. WYD uses shared datapool. SenseCore streams a CCR data image and
uses a debug CPU job to establish NAS READY. The desktop constructs Runtime
images, and historical desktop datasets keep their original read/reuse path.
Output archives still use object storage plus verified client ACK.

New script preparation uses an identity-bound NAS receipt callback, separately
from provider logs and final CPU job state. Historical Runs/images are not
rewritten. Worker and recipe changes affect new identities only.

## Verification

Run the unchanged [development gates](development.md): client/core/server tests,
independent coverage, mypy, Rust, CLI and installed-wheel checks. Meaningful cases
include lost replies/ACKs, exact UID and storage-scope isolation, interrupted
publication, bounded concurrent buffers, symlink-safe expiry, corrupt bytes,
metric contexts and administrative deletion fences.

Local and injected-boundary tests do not establish actual GPU availability,
SenseCore restoration, transfer performance or historical failure causes. Real
backend acceptance needs separately scoped jobs, exact identities and verified
result/checkpoint evidence; documentation does not claim those runs succeeded.
