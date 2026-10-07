"""Implemented adapter features, independent of health and live resources."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendCapabilities:
    runtime_materialization: str
    allocation_mode: str
    logs: tuple[str, ...]
    queue_reason: bool
    exit_code: bool
    preemption_reason: bool
    exact_submission_lookup: bool
    walltime_enforcement: tuple[str, ...]
    persistent_filesystem: bool
    resource_inventory_query: bool = False
    offline_output_export: bool = False


SLURM = BackendCapabilities(
    "oci_to_sif", "flexible", ("live", "historical"), True, True, True,
    True, ("scheduler", "worker"), True,
)
SENSECORE = BackendCapabilities(
    "oci", "fixed_spec", ("live", "historical"), False, False, False,
    True, ("worker",), True,
)
LOCAL = BackendCapabilities(
    "native_process", "local", ("live", "historical"), False, False, False,
    True, (), False,
)

DECLARATIONS = {"slurm": SLURM, "sensecore": SENSECORE, "local": LOCAL}
