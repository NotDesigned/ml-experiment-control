"""Replaceable scheduler adapters and the application composition root."""

from .base import Backend, BackendRegistry


def build_registry(services) -> BackendRegistry:
    """Construct installed backends without leaking them into controller core."""
    from .local import LocalBackend
    from .sensecore import SenseCoreBackend
    from .wyd import WydSlurmBackend

    return BackendRegistry(
        LocalBackend(services), WydSlurmBackend(services), SenseCoreBackend(services)
    )


__all__ = ["Backend", "BackendRegistry", "build_registry"]
