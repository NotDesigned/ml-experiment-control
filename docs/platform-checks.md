# Platform checks and restrictions

This audit covers the native client, server, worker and the image bridge on
2026-11-05. A check being present in code does **not** establish that a user or
cloud provider required it. The rows below distinguish product policy,
implementation limits and data integrity. Provider permissions and network
reachability remain external requirements.

## Removed in this change

| Check | Previous behaviour | Change |
| --- | --- | --- |
| Image origin filter | Every external FROM had to match an operator prefix, including auxiliary build stages such as GHCR uv. Data-image bases were also filtered. | Deleted from all four native/legacy training/data-image paths. Removed the deployment/example setting. Registry credentials and connectivity determine whether a registry is usable. |
| ADD / ONBUILD ban | Rejected ordinary Dockerfile instructions even though RUN already allowed arbitrary build code and network access. | Removed. The user's Dockerfile determines the build. |
| Literal `<<` ban | Rejected comments, quoted strings and JSON commands containing `<<`. | Only an unquoted heredoc operator in RUN/COPY/ADD is rejected while the current parser lacks heredoc support. |

Implementations: [native_image.py](../server/src/ml_exp_server/builds/native_image.py),
[native_data.py](../server/src/ml_exp_server/builds/native_data.py),
[data_image_build.py](../server/src/ml_exp_server/builds/data_image_build.py),
[image_builder.py](../server/src/ml_exp_server/image_builder.py),
[dockerfile_build.py](../server/src/ml_exp_server/dockerfile_build.py).

The new end-to-end fixture also exposed an unprivileged-builder bug: copying a
sealed tree made the temporary context readonly before worker injection. The
temporary context and its generated control files are now writable by their
owner; the original frozen source is unchanged. Root privileges are not an
implicit requirement of this packaging step.

## Additional platform policies to simplify

These remain in the code in this change. They are not described as provider
requirements, and the audit does not silently change their existing records.

| Area | Actual restriction and consequence | Proposed simplification | Code |
| --- | --- | --- | --- |
| Submission approval | Preparation requires a positive `max_gpu_hours`, even for a CPU-only job, alongside resources and walltime. Separate prepare/authorize/execute and exact confirmation strings add more calls. | Make the extra budget optional; use one normal submit operation and retain internal idempotency. Destructive deletion can still request confirmation. | `core/preparation.rs`, `core/first_submission/native.rs`, `client/preparations.rs`, `client/submissions.rs` |
| Retry accounting | Every Attempt reserves the full requested GPU duration again; an early failed startup consumes the same reservation as a full run. | Separate a caller-selected spending cap from actual usage and clearly label reservations. | `core/first_submission/native/retry.rs` |
| Retry source | Slurm/SenseCore retry requires a checkpoint; checkpoint-free restart is PAI-only, accepts only DATA_CACHE_UNAVAILABLE, and requires a new CPU sync plus two sequence advances after termination. Other proven startup failures cannot use the same path. | Permit an explicit restart from the beginning once the exact old job is terminal. Keep it distinct from resuming a checkpoint; do not demand proof that no previous training step occurred merely to restart. | `core/cloud/retry.rs`, `server/cloud_runs/service/startup_retry.rs` |
| Environment variables | Blocks `LD_LIBRARY_PATH`, GPU visibility variables, all `RESUME_*`, and platform-owned variables together. The library-path ban can prevent legitimate dependency selection. | Reserve only the exact variables actually owned by the launcher. Treat overriding GPU assignment separately from ordinary library configuration. | `core/environment.rs` |
| Source/data names | Source components such as `keys`, `secrets` and `.env`, or suffixes `.key/.pem/.p12/.pfx`, are rejected by name. Dataset paths have a similar suffix/directory blacklist. A filename alone does not prove it contains a credential. | Use visible client packaging exclusions or an explicit opt-in for intentional files; do not confuse filename heuristics with path traversal checks. | `core/sources.rs:is_protected`, `core/datasets.rs:path` |
| Download interpreter | Only `python3` and `/bin/sh` are accepted. Bash or another installed interpreter is rejected even though the script already executes user code. | Accept an explicit executable/argv; validate argument encoding and paths without inventing an interpreter list. | `core/data_preparation.rs` |
| HTTPS dataset sources | Requires an explicit nonempty host list (max 16), HTTPS port 443 and a dotted DNS hostname; private-network hosts, custom HTTPS ports and IP-address URLs are rejected. | Support configured backend-local sources and custom ports. Keep authentication-redirect handling separate from the origin policy; the API host and a user-owned training node have different network boundaries. | `core/datasets/import.rs` |
| Metric counts | A schema permits at most 256 definitions; W&B display accepts at most 512 distinct series identities, so one metric over many datasets/variants can hit the display limit. | Make resource limits configurable and return the exact exceeded limit. W&B presentation failure must not invalidate accepted experiment metrics. | `core/metrics/schema.rs`, `core/tracking/display.rs` |
| W&B config size | The parameter encoding is capped at 16 KiB, independent of the larger Run request limit. | Limit or select W&B config independently while keeping the full experiment manifest available. | `core/tracking/contract.rs` |
| W&B remote configuration | Publication compares the entire frozen configuration with the remote copy, including a lossless identity string. W&B normalization or an intentional remote config edit can block metrics even though the destination Run is correct. | Separate destination ownership and event deduplication from configuration display consistency. Preserve existing cursors and uncertain sends when changing this contract. | `core/tracking/publication.rs`, `server/tracking/sdk.py` |
| Debug leases | Native CCI has a hardcoded two-hour lease ceiling; the shared Create type also caps all debug leases at four hours. The two-hour CCI rule is a platform policy, not an established provider limit. | Put backend-specific lease policy in discoverable configuration. Retain the user-confirmed four-hour DSW limit. | `core/cloud/debug.rs` |
| Attempt count | `attempt-257` is rejected by a hardcoded 256-Attempt maximum. | Expose a configurable cap if needed for operations, rather than a hidden retry rule. | `core/first_submission/native/retry.rs:number` |
| File/request counts | Source archive: 512 MiB compressed / 1 GiB expanded / 100,000 files. Dataset/result inventory: 20,000 files. Checkpoint manifest: 256 KiB as well as 20,000 files. Run JSON: 64 KiB; argv: 128 entries. These are implementation bounds, not cloud limits. | Keep bounded resource use but publish effective limits and make workload-dependent limits configurable. The checkpoint byte limit can be reached long before its file-count limit. | `core/sources.rs`, `core/artifacts.rs`, `core/checkpoints.rs`, `core/first_submission/native/creation.rs` |
| Project registration | Native project registration rejects nonempty campaigns because loading them is not implemented. Historical namespaces are read-only. | Advertise unsupported migration features as such; do not frame them as scientific admission requirements. | `server/projects/store.rs`, `server/readiness.rs` |
| Readiness display | `training` is unconditionally INCOMPLETE; GPU acceptance is always NOT_RECORDED even when submission is configured. This is informational, not a submission gate, but misleads clients. | Report executable/configuration readiness separately from recorded end-to-end test evidence. Derive each status from actual checks. | `server/readiness.rs` |
| Build errors | Generic Python exceptions become an uninformative 409, and Rust reconcile can replace the original cause with BUILD_RECEIPT_MISSING. | Keep a bounded, redacted stage-specific cause. A pre-dispatch rejection should have a normal same-identity retry path; missing receipt alone is not evidence of failure. | `server/src/ml_exp_server/image_builder.py:Handler`, `server/cloud_runs/service/builds.rs` |
| Result export selection | The generated result archive excludes the entire `checkpoints/` subtree and checkpoint-ready marker, even if explicitly listed as required output. | Keep exclusion as the ordinary small-export default, but permit an explicit backup/export selection. Persistent state must not be deleted with the download copy. | `core/artifacts/publication.rs:export_path` |

Paths beginning with `core/`, `server/` and `client/` in the final column are
relative to `rust/crates/`, except the explicitly written Python path.

## Engineering limits to document, not disguise as user policy

The current image parser cannot reliably interpret custom frontends/escape
directives, heredocs, ARG-based FROM or external-image COPY --from. The worker
and image publication path currently target linux/amd64. These are real
implementation gaps, not a reason to require particular image repositories.
Support should be implemented in the relevant parser/builder before advertising
those forms. Digest pinning and request/base consistency remain explicit
provenance checks; they do not imply that all network-dependent Dockerfile RUN
steps are reproducible.

The full-file hash and readonly-cache checks in `worker/datasets.rs` are used
when reusing a cache too. Their I/O cost scales with the entire dataset, not just
metadata. Repeated verification deserves measurement and a cache policy; the
previous cross-node device-number equality check has already been removed.

DSW four-hour and DLC twenty-four-hour runtime maxima were explicitly requested
for this deployment. GPU/CPU/memory profile caps are operator configuration;
provider quota denial must still be returned with the actual upstream reason.
The 1 GiB transfer buffer bounds transient server disk usage and was also
explicitly requested. None of these justify an unrelated image-origin filter.

## Checks retained for the requested behaviour

- Authentication and project/Run/Attempt ownership; do not send registry
  credentials to arbitrary authentication realms or redirects.
- Reject path traversal and writes outside the selected storage root; distinguish
  this from filename-based secret guessing.
- Bind a Run to its source, image, dataset, protocol and checkpoint. Validate
  bytes/hash at transfer or restore boundaries. Run/Attempt history remains useful.
- Record named, unit-bearing metrics with project-defined requirements. No ELF
  metric names, accuracy or perplexity are mandatory in the current native path.
  Missing, nonfinite, failed and valid results remain distinct.
- Do not silently overwrite same-context metrics with different units/protocols.
  W&B names stay comparable across Runs; accepted raw metrics remain authoritative.
- Preserve idempotency after a lost provider response. Query the exact existing
  job before creating another; a normal retry of a confirmed terminal job is a
  separate case from repeating an unknown create.
- Verify a download before its ACK releases the generated archive. Backend
  training state and datasets are separate from that disposable archive. Required
  result filenames default to empty and are supplied by the experiment, not a
  fixed `final.pt` or a platform-specific scientific protocol.

The main simplification priorities after the image fix are: normal submit and
explicit restart without extra approval concepts; configurable transparent
limits; ordinary user environment/script support; and useful status/errors.
