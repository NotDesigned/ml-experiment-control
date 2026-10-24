"""Independent ML experiment control-plane daemon.

The package owns workspace state, project lifecycle, polling, evidence indexing,
reviewable actions, and the HTTP boundary.  Scheduler and source execution
primitives remain in :mod:`experiment_control`.
"""

# Keep package import lightweight: clients may import protocol models without
# installing FastAPI or OpenTelemetry.  Daemon entry points import
# their runtime modules explicitly.

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ml-experiment-server")
except PackageNotFoundError:
    __version__ = "0+uninstalled"

__all__: list[str] = []
