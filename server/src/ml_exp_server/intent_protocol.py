"""Client-authored operation intents accepted by the experiment daemon.

An intent describes a requested code or experiment-management mutation.  It is
not an analysis result, conversation message, or authorization.  Preparation is
read-only; a separate Action authorization is required before execution.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


IntentKind = Literal[
    "CREATE_RESEARCH_QUESTION_DRAFT",
    "CREATE_CAMPAIGN_DRAFT",
    "UPDATE_CAMPAIGN_DRAFT",
    "DERIVE_RUN_DRAFT",
    "RUN_EVALUATION",
    "RETRY_ATTEMPT",
    "SUBMIT_RUN",
    "CANCEL_RUN",
    "ARCHIVE_CAMPAIGN",
    "ARCHIVE_RUN",
    "ARCHIVE_ATTEMPT",
    "OBSERVABILITY_BACKFILL",
]


class OperationIntent(BaseModel):
    """Immutable client request used to derive one idempotent Action plan."""

    model_config = ConfigDict(extra="forbid")

    kind: IntentKind
    title: str = Field(min_length=1, max_length=1000)
    target: str = Field(default="", max_length=2000)
    change_summary: str = Field(default="", max_length=8000)
    resource_estimate: str = Field(default="unknown", max_length=2000)
    rationale: str = Field(default="", max_length=8000)
    risk: str = Field(default="", max_length=8000)
    draft: str = Field(max_length=200_000)
    evidence_digest: str = Field(default="", max_length=200)
    idempotency_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )
