"""Reusable scheduler, preflight, project-contract, and execution primitives."""

from .runner import CommandResult, CommandRunner, SubprocessRunner
from .checkpoints import discover_latest_completed_checkpoint
from .preflight import PreflightCheck, PreflightReport
from .identity import IdentityReport
from .project import AssetProbe, AssetRequirement, ProjectAdapter, ProjectRegistry, SourceBundle
from .states import FailureClass
from .manifest import (
    ExperimentStateStore,
    LifecycleStatus,
    RunState,
    append_event,
    atomic_write,
    require_immutable,
    sanitize_command,
    utc_now,
    validate_identity,
)
from .run_manifest import build_run_manifest, comparable_manifest
from .outbox import cancel_intent_path, execute_cancel_outbox
from .contracts import (
    AssetVerification,
    AttemptManifest,
    BackendConfig,
    BackendLogs,
    BackendRecord,
    BackendStatus,
    Campaign,
    CollectionResult,
    JsonObject,
    JsonValue,
    LocalBackendConfig,
    ProjectRun,
    RunSpec,
    SenseCoreBackendConfig,
    SlurmBackendConfig,
    StorageConfig,
    SubmissionIntent,
    SubmissionRequest,
)

__all__ = [
    "CommandResult",
    "CommandRunner",
    "discover_latest_completed_checkpoint",
    "FailureClass",
    "ExperimentStateStore",
    "LifecycleStatus",
    "IdentityReport",
    "PreflightCheck",
    "PreflightReport",
    "AssetProbe",
    "AssetRequirement",
    "ProjectAdapter",
    "ProjectRegistry",
    "SourceBundle",
    "SubprocessRunner",
    "RunState",
    "append_event",
    "atomic_write",
    "require_immutable",
    "sanitize_command",
    "utc_now",
    "validate_identity",
    "build_run_manifest",
    "comparable_manifest",
    "cancel_intent_path",
    "execute_cancel_outbox",
    "AssetVerification",
    "AttemptManifest",
    "BackendConfig",
    "BackendLogs",
    "BackendRecord",
    "BackendStatus",
    "Campaign",
    "CollectionResult",
    "JsonObject",
    "JsonValue",
    "LocalBackendConfig",
    "ProjectRun",
    "RunSpec",
    "SenseCoreBackendConfig",
    "SlurmBackendConfig",
    "StorageConfig",
    "SubmissionIntent",
    "SubmissionRequest",
]
