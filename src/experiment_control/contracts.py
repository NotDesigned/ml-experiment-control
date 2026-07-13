"""Typed, serialization-compatible records crossing host/backend boundaries.

The controller persists these values as JSON or YAML, so the contracts remain
structural mappings instead of runtime-owned model classes.  Host-specific
campaign fields and project-specific summary fields intentionally stay open.
"""

from __future__ import annotations

from typing import Literal, Mapping, TypeAlias, TypedDict


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
Campaign: TypeAlias = Mapping[str, JsonValue]
ProjectRun: TypeAlias = Mapping[str, JsonValue]


class _LocalBackendOptional(TypedDict, total=False):
    cancel_grace_seconds: float


class LocalBackendConfig(_LocalBackendOptional):
    kind: Literal["local"]
    workdir: str


class _SlurmBackendOptional(TypedDict, total=False):
    apptainer_cache_dir: str
    apptainer_tmp_dir: str


class SlurmBackendConfig(_SlurmBackendOptional):
    kind: Literal["slurm"]
    ssh_alias: str
    partition: str
    account: str
    qos: str
    gres: str
    time: str
    mount_root: str
    source_dir: str
    sif_path: str


class _SenseCoreBackendOptional(TypedDict, total=False):
    priority: str
    worker_nodes: int
    sco_bin: str


class SenseCoreBackendConfig(_SenseCoreBackendOptional):
    kind: Literal["sensecore"]
    workspace: str
    aec2: str
    job_name: str
    display_name: str
    image: str
    worker_spec: str
    quota_type: str
    storage_mount: str


BackendConfig: TypeAlias = LocalBackendConfig | SlurmBackendConfig | SenseCoreBackendConfig


class StorageConfig(TypedDict, total=False):
    run_dir: str
    data_root: str
    project_data_root: str
    hf_home: str
    hf_datasets_cache: str


class ResourceConfig(TypedDict, total=False):
    gpus: int
    cpus: int
    memory_gb: int


class _RunSpecOptional(TypedDict, total=False):
    image_id: str
    source_id: str
    storage: StorageConfig
    resources: ResourceConfig


class RunSpec(_RunSpecOptional):
    """Minimum run identity accepted by every backend."""

    run_id: str
    backend: BackendConfig


class _AttemptManifestOptional(TypedDict, total=False):
    source_id: str
    backend: BackendConfig
    storage: StorageConfig
    resources: ResourceConfig
    execution: JsonObject
    campaign: str
    run_id: str


class AttemptManifest(_AttemptManifestOptional):
    """Minimum immutable attempt payload needed for submission."""

    attempt_id: str
    command: list[str]


class SubmissionRequest(TypedDict, total=False):
    """Sanitized, backend-specific scheduler mutation identity."""

    scheduler_name: str
    resource_name: str
    workdir: str
    partition: str
    manifest_sha256: str
    image_tag: str
    image_digest: str
    image_reference: str


class SubmissionIntent(SubmissionRequest, total=False):
    backend: str
    attempt_id: str
    state: str


class BackendRecord(TypedDict):
    """Exact scheduler identity persisted for one attempt."""

    attempt_id: str
    backend_job_id: str | None


class _BackendStatusOptional(TypedDict, total=False):
    attempt_id: str
    raw_state: str | None
    exit_code: int | str | None
    failure_class: str | None
    partition: str
    elapsed: str | None
    pool: str | None
    spec: str | None


class BackendStatus(_BackendStatusOptional):
    """Normalized observation returned by status and cancel operations."""

    run_id: str
    backend: str
    backend_job_id: str
    state: str


class LogSources(TypedDict, total=False):
    stdout: str | None
    stderr: str | None
    scheduler: str | None


class _BackendLogIdentity(TypedDict):
    """Identity shared by every bounded, redacted backend log result."""

    run_id: str
    backend: str
    backend_job_id: str
    tail: int


class StreamBackendLogs(_BackendLogIdentity):
    attempt_id: str
    sources: LogSources
    stdout: list[str]
    stderr: list[str]


class LiveBackendLogs(_BackendLogIdentity):
    lines: list[str]
    expired: bool
    stream_exit_code: int


BackendLogs: TypeAlias = StreamBackendLogs | LiveBackendLogs


class MissingAsset(TypedDict):
    kind: str
    identity: str
    reason: str
    path: str


class AssetVerification(TypedDict):
    missing: list[MissingAsset] | None
    verification: str
    verified_on: str | None


MetricRecord: TypeAlias = JsonObject
CheckpointRecord: TypeAlias = JsonObject
RunSummary: TypeAlias = JsonObject
CollectionResult: TypeAlias = JsonObject
PreflightScope: TypeAlias = Literal["stage", "submit", "observe"]
