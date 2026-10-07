"""Optional publication settings, independent of scientific metric protocols."""
from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class WandbOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    entity: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    project: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class WandbSettings(WandbOptions):
    api_key: SecretStr | None = Field(default=None, min_length=1, max_length=512)
    clear_credentials: bool = False

    @field_validator("api_key")
    @classmethod
    def printable_key(cls, value):
        if value is not None and any(c.isspace() or ord(c) < 33 for c in value.get_secret_value()):
            raise ValueError("API key must not contain whitespace")
        return value


def finite_parameters(value):
    if value is not None and len(json.dumps(value, allow_nan=False).encode()) > 16384:
        raise ValueError("parameters must be finite JSON of at most 16 KiB")
    return value


def resolve(options, settings, project):
    options = options or WandbOptions()
    reason = ("DISABLED_BY_REQUEST" if not options.enabled else
              "DISABLED_BY_SERVER" if not settings.get("enabled", True) else
              "CREDENTIALS_NOT_CONFIGURED" if not settings.get("api_key") else
              "DEFAULT_ENTITY_UNAVAILABLE" if not (options.entity or settings.get("resolved_entity")) else None)
    return {"requested": options.enabled, "enabled": reason is None, "reason": reason,
            "entity": options.entity or settings.get("resolved_entity"),
            "project": options.project or settings.get("project") or project}
