# Platform behaviour and remaining limits

This describes the native implementation after the 2026-11-05 simplification.
Provider permissions, quota, storage and network connectivity remain external
requirements. The API must report their errors rather than invent additional
scientific admission rules.

## Removed platform policies

- Dockerfile base-image origin lists, ADD/ONBUILD bans and literal `<<` bans.
  The Dockerfile parser still rejects syntax it cannot interpret safely.
- Mandatory GPU-hour budgets and cumulative retry reservations. Existing budget
  fields remain readable as metadata; resources and walltime define the job.
- The 256-Attempt cap and checkpoint-only retry requirement. An explicit restart
  from the beginning is available once the exact parent job is terminal. It
  does not require a particular failure code, a new CPU sync or a specific
  worker feature. An unknown submission still requires reconciliation.
- Filename guesses about secrets, keys and certificates in source/data assets.
- Extra environment variable count/size limits and blanket bans on library,
  GPU-visibility and `RESUME_*` variables. Launcher-owned variables remain
  reserved to avoid changing its paths and identities accidentally.
- The download-script interpreter list and argument-count limit. The executable
  must exist in the chosen image. Arguments cannot contain NUL.
- Mandatory HTTPS host lists, port 443 and public dotted DNS names. An optional
  host list is still honoured when supplied; HTTPS, credential handling and
  redirect boundaries remain explicit.
- Metric-definition and W&B-series count caps, the separate 16 KiB W&B parameter
  cap, and comparison of the entire remote W&B config against a lossless local
  identity string. Publication retains destination identity and event ACKs.
- The hardcoded CCI two-hour lease ceiling. The explicitly requested PAI DSW
  four-hour and DLC twenty-four-hour limits remain.
- Blanket exclusion of checkpoint files from an explicit result export.
- Desktop Docker tunnels, remote Docker engines, desktop data staging and CCR
  HTTP blob relays. Builds use a CPU PAI DSW publishing to a configured registry
  or a local BuildKit engine. Payload
  transfers target backend storage with a bounded API buffer.

## Identity across API address changes

A submitted command is part of its saved Action. A later change to the API's
public address affects future submissions, not the readability of original
jobs. Query validation uses the saved command while still comparing the Run,
Attempt, source, image, data, resources, backend and Action digest. It does not
rewrite old manifests or resend a job. A live old worker may still contain the
old callback URL; restoring its callbacks is a separate operation.

## Retained correctness and engineering boundaries

- Authentication, project/Run/Attempt ownership and backend authorization.
- Path traversal prevention and writes confined to the configured storage root.
- Source/image/data/checkpoint provenance and transfer/restore hashes.
- Named, unit-bearing metrics and project-defined required metrics. Missing,
  failed, nonfinite and valid results stay distinct. Raw metric identity is
  separate from comparable W&B display names.
- Durable idempotency after an uncertain provider response. Query the original
  job before creating another; never treat a missing receipt as permission to
  repeat an unknown create.
- Verified download acknowledgement before release of its disposable archive.
  Persistent checkpoints and datasets have separate lifecycles.
- Bounded request/archive processing, storage checks, and upstream resource
  limits. These are engineering limits, not mandatory experiment protocols.
  See the discovered API limits for effective values.

The current image parser lacks custom frontends/escape directives, heredocs,
ARG-based FROM and external-image COPY --from. Training image publication targets
linux/amd64. The 1 GiB transient transfer buffer and PAI duration maxima were
explicitly requested. Cache verification still reads and hashes full files;
its cost scales with dataset size.

Readiness gaps are reported separately from configured submission capabilities. No
GPU acceptance is implied by a configured capability.
