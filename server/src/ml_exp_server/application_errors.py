"""Transport-neutral application errors with stable HTTP-facing metadata."""

from __future__ import annotations


class ApplicationError(RuntimeError):
    def __init__(
        self, message: str, *, status_code: int = 409,
        code: str = "APPLICATION_ERROR",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
