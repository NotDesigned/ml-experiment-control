"""Small, operator-scoped debug machines, independent of scientific Runs."""
import base64
import hashlib
import struct
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DebugConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profiles_file: str | None = None
    max_active: int = Field(default=2, ge=1, le=16)
    max_duration_seconds: int = Field(default=7200, ge=300, le=28800)


def public_key(value):
    """Accept one Ed25519/RSA public key, without authorized_keys options."""
    if not isinstance(value, str) or len(value) > 16384 or any(c in value for c in "\r\n"):
        raise ValueError("provide a single OpenSSH public key")
    parts = value.split()
    if len(parts) < 2 or parts[0] not in {"ssh-ed25519", "ssh-rsa"}:
        raise ValueError("provide an Ed25519 or RSA public key, never a private key")
    try:
        blob = base64.b64decode(parts[1], validate=True)
        fields = []
        offset = 0
        while offset < len(blob):
            size = struct.unpack_from(">I", blob, offset)[0]
            offset += 4
            if size > len(blob) - offset:
                raise ValueError()
            fields.append(blob[offset:offset + size])
            offset += size
        if fields[0] != parts[0].encode() or (
            parts[0] == "ssh-ed25519" and (len(fields) != 2 or len(fields[1]) != 32)
            or parts[0] == "ssh-rsa" and (len(fields) != 3 or not fields[1]
                                         or int.from_bytes(fields[2], "big").bit_length() < 2048)
        ):
            raise ValueError()
    except (ValueError, IndexError, struct.error):
        raise ValueError("invalid SSH public key encoding or strength") from None
    return parts[0] + " " + parts[1]


class DebugRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,62}$")
    profile: str = Field(pattern=r"^[a-z][a-z0-9-]{0,62}$")
    ssh_public_key: str = Field(max_length=16384)
    duration_seconds: int = Field(default=1800, ge=300, le=28800, strict=True)

    _key = field_validator("ssh_public_key")(public_key)


class DebugProfile(BaseModel):
    """Native storage/network fields are operator-owned, never API request data."""
    model_config = ConfigDict(extra="forbid")
    provider: Literal["pai-dsw", "sensecore-cci"]
    enabled: bool = True
    unavailable_reason: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=200)
    cpus: int = Field(ge=1, le=256, strict=True)
    memory_gib: int = Field(ge=1, le=2048, strict=True)
    gpus: int = Field(default=0, ge=0, le=8, strict=True)
    native: dict


def identity(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
