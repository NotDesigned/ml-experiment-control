"""Stable application facade composed from domain operations."""
from __future__ import annotations
import getpass
import os
import socket
from pathlib import Path
from typing import Any, Callable
from ..controller_gateway import ProjectControllerGateway
from ..schemas import ActionRuntimeConfig
from .policy import ActionExecutionPolicy
from .project_writes import ProjectWriteTransaction
from .store import ActionStore

from .planning import ActionPlanning
from .controller_planning import ControllerPlanning
from .execution import ActionExecution
from .recovery import ActionRecovery


class ActionService(ActionPlanning, ControllerPlanning, ActionExecution, ActionRecovery):
    """Compose domain operations without introducing additional mutable state."""

    def __init__(self, store: ActionStore, config: ActionRuntimeConfig,
                 runner: Callable[..., dict[str, Any]] | None = None,
                 actor_provider: Callable[[], str] | None = None,
                 source_resolver: Callable[[str, str], Path] | None = None):
        self.store = store
        self.config = config
        self.controller = ProjectControllerGateway(runner)
        self.execution_policy = ActionExecutionPolicy(config)
        self.project_write_transaction = ProjectWriteTransaction(store)
        self.actor_provider = actor_provider or self._local_actor
        self.source_resolver = source_resolver
        self.deletion_service = None

    @staticmethod
    def _local_actor() -> str:
        return f"local-process-owner:uid={os.getuid()}:{getpass.getuser()}@{socket.gethostname()}"
