"""Replaceable scheduler adapters and the application composition root."""

from .base import Backend, BackendRegistry
from .services import BackendServices


def build_registry(services: BackendServices) -> BackendRegistry:
    """Construct installed backends without leaking them into controller core."""
    from .local import LocalBackend
    from .sensecore import SenseCoreBackend
    from .slurm import SlurmBackend
    from .pai import PaiBackend

    return BackendRegistry(
        LocalBackend(services), SlurmBackend(services), SenseCoreBackend(services), PaiBackend(services)
    )


__all__ = ["Backend", "BackendRegistry", "BackendServices", "build_registry"]
