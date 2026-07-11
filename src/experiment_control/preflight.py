"""Small, sanitized readiness report shared by compute backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


CheckStatus = Literal["PASS", "WARN", "FAIL"]


@dataclass(frozen=True)
class PreflightCheck:
    """One credential-free readiness assertion."""

    name: str
    category: str
    status: CheckStatus
    message: str = ""

    def to_dict(self) -> dict[str, str]:
        return {key: value for key, value in asdict(self).items() if value}


@dataclass(frozen=True)
class PreflightReport:
    """Backend readiness for a concrete operation scope."""

    backend: str
    scope: str
    checks: tuple[PreflightCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.status != "FAIL" for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "scope": self.scope,
            "ready": self.ready,
            "checks": [check.to_dict() for check in self.checks],
        }

    def require_ready(self) -> None:
        failures = [check.name for check in self.checks if check.status == "FAIL"]
        if failures:
            raise RuntimeError(
                f"{self.backend} {self.scope} preflight failed: {', '.join(failures)}"
            )
