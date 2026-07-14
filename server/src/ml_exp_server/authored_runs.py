"""Read-model placeholders for authored Runs before controller materialization.

Campaign files are the authority for planned Run identities.  A first
submission dry-run materializes the durable Run directory, but a TUI-first
workflow must be able to select that Run before the dry-run exists.
"""

from __future__ import annotations

from .schemas import (
    CampaignBinding,
    CampaignMembershipBinding,
    CampaignRelationship,
    ResearchProject,
    RunIndexRow,
)


def authored_run_placeholder(
    project: ResearchProject, run_id: str,
) -> RunIndexRow | None:
    """Return an ephemeral NOT_SUBMITTED row for one authored materializer.

    Project loading already rejects duplicate materializing Campaigns.  Keep
    this helper fail closed anyway: an absent or ambiguous materializer is not
    a selectable Run object.
    """
    materializers = []
    bindings: list[CampaignMembershipBinding] = []
    for campaign in project.campaigns:
        revision = campaign.current_revision
        if revision is None:
            continue
        membership = next(
            (item for item in revision.memberships if item.run_id == run_id), None,
        )
        if membership is None:
            continue
        if membership.kind == "materialize":
            materializers.append((campaign, revision, membership))
        bindings.append(CampaignMembershipBinding(
            campaign=campaign.name,
            revision_id=revision.revision_id,
            membership=membership,
            is_origin=membership.kind == "materialize",
        ))
    if len(materializers) != 1:
        return None

    campaign, revision, membership = materializers[0]
    return RunIndexRow(
        project=project.project,
        campaign=campaign.name,
        campaign_source="campaign_file",
        campaign_binding=CampaignBinding(
            relationship=CampaignRelationship.UNRESOLVED,
            current_revision=revision.revision_id,
            membership=membership,
        ),
        campaign_memberships=bindings,
        run_id=run_id,
        role=membership.role,
        role_source="campaign_file",
        # No durable directory exists until the controller dry-run.  An empty
        # path is deliberate and prevents the placeholder from masquerading as
        # collected filesystem evidence.
        run_dir="",
        scheduler_state="NOT_SUBMITTED",
        research_contract=revision.research_contract,
        research_contract_source=(
            "campaign_file" if revision.research_contract is not None else None
        ),
        provenance={
            "authored_only": True,
            "campaign_revision": revision.revision_id,
            **(
                {"source_id": revision.source_bindings[run_id],
                 "source_binding": "campaign_file"}
                if run_id in revision.source_bindings else {}
            ),
        },
        warnings=[
            "Run is authored but has not yet been materialized by a controller dry-run"
        ],
    )


def authored_run_placeholders(
    project: ResearchProject, *, excluding: set[str] | None = None,
) -> list[RunIndexRow]:
    """Return all unique authored materializers absent from durable evidence."""
    omitted = excluding or set()
    run_ids = {
        membership.run_id
        for campaign in project.campaigns
        if campaign.current_revision is not None
        for membership in campaign.current_revision.memberships
        if membership.kind == "materialize" and membership.run_id not in omitted
    }
    return [
        row
        for run_id in sorted(run_ids)
        if (row := authored_run_placeholder(project, run_id)) is not None
    ]
