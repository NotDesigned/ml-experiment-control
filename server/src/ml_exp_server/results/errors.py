"""Safe result recovery errors."""

class RecoveryError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
