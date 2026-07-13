"""HTTP-safe contract shared by the server and external Agent clients."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ProposalKind = Literal[
    "ANALYSIS_ONLY", "CREATE_REPORT_DRAFT", "CREATE_CHART_SPEC",
    "CREATE_RESEARCH_QUESTION_DRAFT", "CREATE_CAMPAIGN_DRAFT",
    "UPDATE_CAMPAIGN_DRAFT", "CREATE_PROJECT_ADAPTER_DRAFT",
    "DERIVE_RUN_DRAFT", "RUN_EVALUATION", "RETRY_ATTEMPT", "SUBMIT_RUN",
    "CANCEL_RUN", "COMPLETE_CAMPAIGN", "ARCHIVE_CAMPAIGN", "ARCHIVE_RUN",
    "ARCHIVE_ATTEMPT", "UPDATE_VERDICT",
]


class AgentProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ProposalKind
    title: str = Field(min_length=1, max_length=1000)
    target: str = Field(max_length=2000)
    change_summary: str = Field(max_length=8000)
    resource_estimate: str = Field(max_length=2000)
    rationale: str = Field(max_length=8000)
    risk: str = Field(max_length=8000)
    draft: str = Field(max_length=200_000)


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(max_length=100_000)
    proposals: list[AgentProposal] = Field(default_factory=list, max_length=3)
