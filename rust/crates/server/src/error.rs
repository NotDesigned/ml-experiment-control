//! The one error an HTTP handler returns: a status, a stable code and fixed text.
//! Every domain error converts here, so a handler's `?` picks the public answer.
//! No conversion echoes provider or request text. Mappings that depend on the
//! operation stay at their call site: `creation_failure` in `submission_http`, the
//! refresh remap `slurm_failure` in `runs`, and the exact Submission lookup in `api`.
use crate::{experiments, read_model};

#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("{code}")]
pub struct Error {
    pub status: u16,
    pub code: &'static str,
    pub detail: &'static str,
}
pub(crate) fn error(status: u16, code: &'static str, detail: &'static str) -> Error {
    Error {
        status,
        code,
        detail,
    }
}
impl From<crate::projects::Error> for Error {
    fn from(value: crate::projects::Error) -> Self {
        use crate::projects::Error as E;
        match value {
            E::Busy => error(
                503,
                "PROJECT_REGISTRY_BUSY",
                "project registry is busy; no operation was dispatched",
            ),
            E::EffectUncertain => error(
                500,
                "PROJECT_OPERATION_UNCERTAIN",
                "project state may have changed; query the lifecycle ledger before retry",
            ),
            E::Binding => error(
                409,
                "PROJECT_SOURCE_BINDING_MISMATCH",
                "registered Project identity differs from approved source configuration",
            ),
            E::Transition(ml_exp_core::projects::Error::UnknownProject) => {
                error(404, "UNKNOWN_PROJECT", "registered Project is unavailable")
            }
            _ => error(
                409,
                "PROJECT_LIFECYCLE_BLOCKED",
                "project lifecycle or its durable evidence cannot be validated",
            ),
        }
    }
}
impl From<experiments::DiscoveryError> for Error {
    fn from(value: experiments::DiscoveryError) -> Self {
        use experiments::DiscoveryError as E;
        match value {
            E::Depth => error(
                413,
                "EXPERIMENT_DISCOVERY_LIMIT",
                "canonical Run discovery exceeds its depth",
            ),
            E::UnknownProject => error(
                404,
                "EXPERIMENT_NOT_FOUND",
                "the Project is not configured for discovery",
            ),
            E::DuplicateRun | E::OverlappingRoots => error(
                409,
                "EXPERIMENT_DISCOVERY_AMBIGUOUS",
                "approved roots do not establish unique canonical Runs",
            ),
            _ => error(
                409,
                "EXPERIMENT_DISCOVERY_UNAVAILABLE",
                "canonical Run discovery encountered changed, unsafe or unavailable evidence",
            ),
        }
    }
}
impl From<experiments::Error> for Error {
    fn from(value: experiments::Error) -> Self {
        match value {
            experiments::Error::UnknownProject | experiments::Error::Missing => error(
                404,
                "EXPERIMENT_NOT_FOUND",
                "the configured experiment or manifest is unavailable",
            ),
            experiments::Error::Binding => error(
                422,
                "INVALID_EXPERIMENT_IDENTITY",
                "invalid configured experiment identity",
            ),
            _ => error(
                409,
                "EXPERIMENT_EVIDENCE_UNAVAILABLE",
                "experiment evidence changed, was deleted or cannot be validated",
            ),
        }
    }
}
impl From<read_model::Error> for Error {
    fn from(value: read_model::Error) -> Self {
        match value {
            read_model::Error::Query(_) => {
                error(422, "INVALID_READ_QUERY", "invalid experiment read query")
            }
            read_model::Error::UnknownAttempt => {
                error(404, "ATTEMPT_NOT_FOUND", "the exact Attempt is unavailable")
            }
            read_model::Error::OutputLimit => error(
                413,
                "READ_RESPONSE_LIMIT",
                "experiment read exceeds the response limit",
            ),
            _ => error(
                409,
                "EXPERIMENT_EVIDENCE_UNAVAILABLE",
                "experiment evidence cannot be projected safely",
            ),
        }
    }
}
impl From<crate::backends::slurm::service::Error> for Error {
    fn from(error: crate::backends::slurm::service::Error) -> Self {
        use crate::backends::slurm::service::Error as E;
        let (status, code, detail) = match error {
            E::Missing | E::Action(crate::operations::Error::NotFound) => (
                404,
                "SUBMISSION_NOT_FOUND",
                "the original preparation is unavailable; no new identity was created",
            ),
            E::ObservationConflict => (
                409,
                "WORKER_OBSERVATION_CONFLICT",
                "remote worker state or accepted metric history changed; previous evidence is retained and no provider mutation was issued",
            ),
            E::ArtifactsUnavailable => (
                503,
                "ARTIFACTS_UNAVAILABLE",
                "backend archive cannot be collected, verified or streamed; original outputs and checkpoints are retained",
            ),
            E::ArtifactsReleased => (
                410,
                "ARTIFACTS_RELEASED",
                "the generated archive was released after a verified download; checkpoints and original outputs are retained",
            ),
            E::CheckpointUnavailable => (
                503,
                "CHECKPOINT_VERIFICATION_UNAVAILABLE",
                "backend checkpoint ready marker or generation could not be verified; no scheduler operation was requested",
            ),
            E::CheckpointMissing => (
                404,
                "CHECKPOINT_NOT_FOUND",
                "the exact registered checkpoint is unavailable",
            ),
            E::DatasetMissing => (
                404,
                "DATASET_NOT_FOUND",
                "the exact asset is not registered for this Project and backend",
            ),
            E::DatasetUnavailable => (
                503,
                "DATASET_TRANSFER_UNCONFIRMED",
                "dataset transfer or verification is unconfirmed; resume the same asset identity; no scheduler operation was issued",
            ),
            E::CheckpointConflict => (
                409,
                "CHECKPOINT_EVIDENCE_UNAVAILABLE",
                "checkpoint identity or immutable registry evidence cannot be validated",
            ),
            E::ObservationUnavailable => (
                503,
                "WORKER_OBSERVATION_UNAVAILABLE",
                "worker evidence is unavailable; the last accepted sample is retained and no provider mutation was issued",
            ),
            E::Busy => (
                503,
                "SUBMISSION_BUSY",
                "preparation is busy; query the same Run before retrying",
            ),
            E::Disabled => (
                403,
                "SUBMISSION_DISABLED",
                "scheduler execution is disabled by the operator",
            ),
            E::RunCreationDisabled => (
                403,
                "RUN_CREATION_DISABLED",
                "native Run creation is not enabled for this Project",
            ),
            E::Admission(cause) if cause.0 == ml_exp_core::datasets::ManifestsTooLarge::CAUSE => (
                422,
                "DATASET_MANIFESTS_TOO_LARGE",
                "the Run's dataset manifests exceed their 256 KiB budget; use fewer inputs or package many small files into fewer",
            ),
            E::Admission(cause)
                if cause.0 == ml_exp_core::first_submission::native::creation::CODE_DELIVERY_UNSUPPORTED =>
            {
                (422, "CODE_DELIVERY_UNSUPPORTED", cause.0)
            }
            E::Admission(cause) => (422, "SUBMISSION_ADMISSION_REJECTED", cause.0),
            E::Authority | E::Conflict | E::Configuration => (
                409,
                "SUBMISSION_EVIDENCE_UNAVAILABLE",
                "frozen submission authority changed or cannot be validated; no replay is permitted",
            ),
            E::Action(crate::operations::Error::Decision(_))
            | E::Runner(crate::backends::slurm::action::Error::Action(
                crate::operations::Error::Decision(_),
            )) => (
                409,
                "SUBMISSION_STATE_CONFLICT",
                "the requested transition or confirmation does not match the original Action",
            ),
            _ => (
                500,
                "SUBMISSION_OPERATION_UNCERTAIN",
                "submission state may have changed; query the same Run or Submission and reconcile before deciding any further action",
            ),
        };
        Error {
            status,
            code,
            detail,
        }
    }
}
impl From<crate::backends::cloud::Error> for Error {
    fn from(error: crate::backends::cloud::Error) -> Self {
        let (status, code, detail) = if error.uncertain {
            (
                500,
                "CLOUD_OPERATION_UNCERTAIN",
                "cloud operation may have started; query and reconcile the original identity; do not replay",
            )
        } else {
            match (error.code.as_str(), error.http_status) {
                ("OSS_DATASET_SCOPE_DENIED", _) => (
                    422,
                    "OSS_DATASET_SCOPE_DENIED",
                    "dataset object is outside the configured personal or public dataset prefixes",
                ),
                ("OSS_DATASET_READ_FAILED", _) => (
                    503,
                    "OSS_DATASET_READ_FAILED",
                    "the configured OSS dataset object could not be read; no dataset was published",
                ),
                ("OSS_DATASET_VERIFICATION_FAILED", _) => (
                    422,
                    "OSS_DATASET_VERIFICATION_FAILED",
                    "OSS dataset tar, archive SHA256, byte count or member inventory is invalid",
                ),
                ("DATASET_PART_NOT_STORED", _) => (
                    409,
                    "DATASET_PART_NOT_STORED",
                    "the object store has no such part yet; repeat the same upload",
                ),
                ("DATASET_PART_REJECTED", _) => (
                    409,
                    "DATASET_PART_REJECTED",
                    "a stored part differs from what was claimed, so the archive was not accepted and that part was removed; repeat the same upload to send it again",
                ),
                ("DATASET_ARCHIVE_REFUSED", _) => (
                    409,
                    "DATASET_ARCHIVE_REFUSED",
                    "every part matches its record but the archive does not match this asset, so it was not accepted; the uploaded parts are wrong as a whole, and changing any file makes a new asset",
                ),
                ("INPUT_NOT_READY", _) => (
                    409,
                    "INPUT_NOT_READY",
                    "the original input asset has not reached verified READY; inspect its delivery record",
                ),
                ("LAUNCH_METADATA_TOO_LARGE", _) => (
                    422,
                    "LAUNCH_METADATA_TOO_LARGE",
                    "the worker launch package exceeds its 8 MiB download budget; reduce its input or restore metadata",
                ),
                ("DATASET_MANIFESTS_TOO_LARGE", _) => (
                    422,
                    "DATASET_MANIFESTS_TOO_LARGE",
                    "the Run's dataset manifests exceed their 256 KiB budget; use fewer inputs or package many small files into fewer",
                ),
                ("CODE_SOURCE_TOO_LARGE", _) => (
                    422,
                    "CODE_SOURCE_TOO_LARGE",
                    "the code exceeds 64 MiB, 20000 entries or the length of its listing; move large files to data with `ml-exp data upload`",
                ),
                ("CODE_DELIVERY_WORKER_OUTDATED", _) => (
                    422,
                    "CODE_DELIVERY_WORKER_OUTDATED",
                    "this Runtime's worker is not the current one, and may not know how to fetch delivered code; build the Runtime again",
                ),
                ("WORKER_BINARY_UNAVAILABLE", _) => (
                    409,
                    "WORKER_BINARY_UNAVAILABLE",
                    "this server does not hold the worker this Attempt froze, or its copy differs; check builds.worker_store",
                ),
                ("CODE_DELIVERY_UNSUPPORTED", _) => (
                    422,
                    "CODE_DELIVERY_UNSUPPORTED",
                    ml_exp_core::first_submission::native::creation::CODE_DELIVERY_UNSUPPORTED,
                ),
                ("DEBUG_MACHINE_LIMIT", _) => (
                    409,
                    "DEBUG_MACHINE_LIMIT",
                    "the configured active debug machine limit has been reached",
                ),
                ("ARTIFACTS_ADMINISTRATIVELY_REMOVED", _) => (
                    410,
                    "ARTIFACTS_ADMINISTRATIVELY_REMOVED",
                    "result objects were administratively removed; collection cannot recreate them",
                ),
                ("ARTIFACT_NOT_PUBLISHED", _) => (
                    409,
                    "ARTIFACT_NOT_PUBLISHED",
                    "the submission exists but its result archive has not been published",
                ),
                ("SYNC_RETRY_NOT_ELIGIBLE", _) => (
                    409,
                    "SYNC_RETRY_NOT_ELIGIBLE",
                    "CPU recovery requires the exact failed synchronization job and terminal training evidence",
                ),
                ("DIAGNOSTIC_PARENT_NOT_TERMINAL", _) => (
                    409,
                    "DIAGNOSTIC_PARENT_NOT_TERMINAL",
                    "the original CPU job has not been confirmed terminal; no diagnostic was submitted",
                ),
                ("READONLY_DIAGNOSTIC_UNSUPPORTED", _) => (
                    422,
                    "READONLY_DIAGNOSTIC_UNSUPPORTED",
                    "read-only CPU diagnostics currently require PAI with personal CPFS",
                ),
                ("READONLY_DIAGNOSTIC_STORAGE_REQUIRED", _) => (
                    422,
                    "READONLY_DIAGNOSTIC_STORAGE_REQUIRED",
                    "configure personal CPFS before creating this read-only diagnostic",
                ),
                ("DATA_COPY_IDENTITY_CONFLICT", _) => (
                    409,
                    "DATA_COPY_IDENTITY_CONFLICT",
                    "observe the original dataset copy request and Runtime; another copy was not created",
                ),
                ("DATA_COPY_REQUEST_SUPERSEDED", _) => (
                    409,
                    "DATA_COPY_REQUEST_SUPERSEDED",
                    "a newer request replaced this dataset copy request; another copy was not created",
                ),
                ("DATA_COPY_RUNTIME_REQUIRED", _) => (
                    422,
                    "DATA_COPY_RUNTIME_REQUIRED",
                    "PAI data publication requires an existing READY Runtime",
                ),
                ("PREPARATION_ID_REQUIRED", _) => (
                    409,
                    "PREPARATION_ID_REQUIRED",
                    "multiple preparations refer to this Run; query the exact saved preparation identity",
                ),
                (_, Some(422)) => (
                    422,
                    "INVALID_CLOUD_REQUEST",
                    "cloud request violates the configured resource or protocol limits",
                ),
                (_, Some(410)) => (
                    410,
                    "CLOUD_OBJECT_GONE",
                    "the original cloud object has been released or retired",
                ),
                (_, Some(401)) => (
                    401,
                    "WORKER_AUTH_REQUIRED",
                    "Attempt worker authentication required",
                ),
                (_, Some(403)) => (
                    403,
                    "CLOUD_OPERATION_DISABLED",
                    "the operator has disabled this operation",
                ),
                (_, Some(404)) => (
                    404,
                    "CLOUD_SUBMISSION_NOT_FOUND",
                    "the original cloud submission is unavailable",
                ),
                (_, Some(503)) => (
                    503,
                    "CLOUD_RUN_BUSY",
                    "the original cloud operation is busy",
                ),
                _ => (
                    409,
                    "CLOUD_EVIDENCE_UNAVAILABLE",
                    "cloud configuration, frozen authority or provider evidence cannot be validated",
                ),
            }
        };
        Error {
            status,
            code,
            detail,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::cloud;
    #[test]
    fn public_failures_keep_actionable_types_and_never_echo_untrusted_codes() {
        for (code, status) in [
            ("INPUT_NOT_READY", 409),
            ("LAUNCH_METADATA_TOO_LARGE", 422),
            ("DATASET_MANIFESTS_TOO_LARGE", 422),
            ("CODE_SOURCE_TOO_LARGE", 422),
            ("CODE_DELIVERY_UNSUPPORTED", 422),
            ("CODE_DELIVERY_WORKER_OUTDATED", 422),
            ("WORKER_BINARY_UNAVAILABLE", 409),
            ("DEBUG_MACHINE_LIMIT", 409),
            ("ARTIFACTS_ADMINISTRATIVELY_REMOVED", 410),
            ("ARTIFACT_NOT_PUBLISHED", 409),
            ("OSS_DATASET_SCOPE_DENIED", 422),
            ("OSS_DATASET_READ_FAILED", 503),
            ("OSS_DATASET_VERIFICATION_FAILED", 422),
        ] {
            let public = Error::from(cloud::Error {
                code: code.into(),
                http_status: None,
                uncertain: false,
            });
            assert_eq!(public.code, code);
            assert_eq!(public.status, status);
        }
        let public = Error::from(cloud::Error {
            code: "untrusted-secret-provider-error".into(),
            http_status: Some(422),
            uncertain: false,
        });
        assert_eq!(public.status, 422);
        assert!(!public.detail.contains("secret"));
        // A possible effect always takes precedence over a convenient status.
        let public = Error::from(cloud::Error {
            code: "LAUNCH_METADATA_TOO_LARGE".into(),
            http_status: Some(422),
            uncertain: true,
        });
        assert_eq!(public.status, 500);
        assert_eq!(public.code, "CLOUD_OPERATION_UNCERTAIN");
    }
}
