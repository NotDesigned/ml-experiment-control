# Client and server refactor

This work is on `refactor/client-server-boundaries`, based on main
`d3c72d1a6686ce5fa31cd6d922f14c1eade41d6a` (server 0.3.36, client 0.1.18).
It keeps the existing repository and independently installable packages. This
branch does not publish a release or change a deployed service.

## Purpose

The client was already independent of server/backend dependencies. Its main
maintenance cost was mixed responsibilities: HTTP, archives, uploads, downloads
and atomic state all lived in `api.py`; argument parsing also advanced experiments.
The server already had domain directories, but Action preparation, app startup
and private builder access crossed several unrelated concerns.

The changes give each concern one home. They do not add a plugin system, generic
workflow engine, second database or another scheduler. See the current module
map and call graph in [architecture](architecture.md).

## Changes

1. Separate client transport, archives, assets, results, configuration and saved
   state. Keep argparse/presentation in `cli.py`, concrete operations in
   `commands.py`, and explicit experiment/Action workflows in their domain modules.
2. Separate server app construction, HTTP boundary and daemon lifetime. Workspace
   ownership, recovery order, five bounded execution lanes and shutdown ownership
   are preserved.
3. Break controller Action preparation into source binding, Campaign freezing,
   preview, resource/checkpoint gates and exact-job cancellation. The original
   preparation entry still owns locks, identity digests and durable plan saving.
4. Share private builder request/receipt types without importing the builder
   daemon into API clients. Use one controller-output parser for ordinary and
   pinned subprocess execution.

## Compatibility and corrections

Package names, the six public client exports, 25 CLI commands, HTTP protocol 2,
OpenAPI, saved client states and durable server formats remain stable. No worker
source changes are included, so this work does not change image or worker
fingerprints. Internal helper paths move to their canonical modules.

Confirmed defects are corrected alongside the structural work:

- Accept the declared boolean `data_preparation_before_job` requirement.
- Show exact saved Submission staging or failure before submission even when
  its reserved Attempt has not yet materialized. Other missing/contradictory
  Attempt evidence still fails closed.
- Reject malformed Runtime state with a useful client error before resource
  requests, rather than a raw `KeyError`.
- Retain successful `check` results if optional storage diagnostics fail, while
  returning a nonzero exit code and a sanitized error for that check.
- Preserve a proven safe-before-build failure during receipt reconciliation.
  An uncertain publication or conflicting receipt never becomes safe to retry.

## Verification

Run the unchanged repository [validation gates](development.md). Client tests
cover state recovery, lost responses, exact identities, upload resumption,
archive verification, acknowledgements and explicit cancel/retry/delete plans.
Server tests cover plan generation, startup failures, queue admission, disconnect
handoffs, lease retention, builder receipts and existing result/checkpoint APIs.
The standalone wheel check runs without server/core dependencies.

Additionally compare baseline and refactored OpenAPI and all CLI parser options;
compare every worker source SHA256; and check moved lifecycle functions against
their original AST. Fault injections target the new canonical modules rather
than retaining obsolete import paths solely for monkeypatches.

These are local and injected-boundary checks. They do not establish real GPU
availability, SenseCore training restoration, network performance or historical
failure causes. No cloud job, scientific download, acknowledgement or deletion
is needed to validate this refactor.
