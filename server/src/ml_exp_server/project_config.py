"""Load daemon workspace, project catalog, and optional research questions."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import ValidationError

from .schemas import (
    CampaignRef,
    CampaignRevision,
    CampaignRunMembership,
    ServerConfig,
    ResearchProject,
    ResearchQuestion,
)


class ConfigError(ValueError):
    """A configuration file is missing or fails schema validation."""


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping at top level of {path}")
    return data


def load_server_config(path: Path) -> ServerConfig:
    path = path.expanduser().resolve()
    data = _load_yaml(path)
    try:
        config = ServerConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid daemon config {path}: {exc}") from exc
    if config.schema_version != 1:
        raise ConfigError(f"unsupported schema_version in {path}: {config.schema_version}")
    # A checked-in workspace must be relocatable.  Project manifests are the
    # only repository-owned paths in this file; state roots intentionally keep
    # their normal XDG/home-relative semantics.  Resolve relative manifests
    # against the config, never against whichever directory launched the daemon.
    for ref in config.projects:
        project_file = Path(ref.project_file).expanduser()
        if not project_file.is_absolute():
            ref.project_file = str((path.parent / project_file).resolve())
    return config


def load_research_question(path: Path) -> ResearchQuestion:
    data = _load_yaml(path)
    try:
        question = ResearchQuestion.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid research question {path}: {exc}") from exc
    if question.schema_version != 1:
        raise ConfigError(f"unsupported schema_version in {path}: {question.schema_version}")
    return question


def _campaign_path(project: ResearchProject, ref: CampaignRef) -> Path:
    path = Path(ref.file or "")
    if not path.is_absolute():
        path = (project.base_dir or Path(".")) / path
    return path.resolve()


def _declared_run_specs(authored: object) -> list[dict]:
    """Extract concrete Run declarations from direct and legacy matrix entries.

    Matrix expansion syntax is controller-specific, so the daemon does not
    attempt to render templates. It indexes concrete run_id leaves already
    present in the authored matrix and ignores placeholder-only template ids.
    """
    if not isinstance(authored, dict):
        return []
    run_id = authored.get("run_id")
    if isinstance(run_id, str) and run_id and "{" not in run_id:
        return [authored]
    found: list[dict] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            nested_run_id = value.get("run_id")
            if isinstance(nested_run_id, str) and nested_run_id and "{" not in nested_run_id:
                found.append(value)
                return
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(authored.get("matrix"))
    return found


def _static_template_value(template: object, key: str) -> object:
    if not isinstance(template, dict):
        return None
    value = template.get(key)
    if isinstance(value, str) and "{" in value:
        return None
    return value


def _load_campaign_revision(project: ResearchProject, ref: CampaignRef) -> CampaignRevision:
    path = _campaign_path(project, ref)
    raw = path.read_bytes() if path.is_file() else None
    if raw is None:
        raise ConfigError(f"campaign file not found for {ref.name!r}: {path}")
    data = _load_yaml(path)
    campaign = data.get("campaign")
    if campaign != ref.name:
        raise ConfigError(
            f"campaign catalog name mismatch for {path}: {ref.name!r} != {campaign!r}"
        )
    campaign_project = data.get("project")
    if campaign_project != project.project:
        raise ConfigError(
            f"campaign project mismatch for {path}: "
            f"{project.project!r} != {campaign_project!r}"
        )
    authored_runs = data.get("runs", [])
    authored_refs = data.get("run_refs", [])
    if not isinstance(authored_runs, list):
        raise ConfigError(f"campaign runs must be a list in {path}")
    if not isinstance(authored_refs, list):
        raise ConfigError(f"campaign run_refs must be a list in {path}")
    if not authored_runs and not authored_refs:
        raise ConfigError(f"campaign requires runs or run_refs in {path}")
    memberships: list[CampaignRunMembership] = []
    run_ids: set[str] = set()
    for authored in authored_runs:
        template = authored.get("template") if isinstance(authored, dict) else None
        for spec in _declared_run_specs(authored):
            run_id = spec["run_id"]
            if run_id in run_ids:
                raise ConfigError(f"duplicate campaign run_id {run_id!r} in {path}")
            run_ids.add(run_id)

            def value(key: str, default=None):
                resolved = spec.get(key)
                if resolved is None:
                    resolved = _static_template_value(template, key)
                return default if resolved is None else resolved

            try:
                memberships.append(CampaignRunMembership(
                    run_id=run_id,
                    kind="materialize",
                    role=value("research_role"),
                    arm=value("arm"),
                    replicate=value("replicate"),
                    purpose=value("purpose"),
                    included_in_analysis=value("included_in_analysis", True),
                ))
            except ValidationError as exc:
                raise ConfigError(f"invalid campaign run membership in {path}: {exc}") from exc
    for authored in authored_refs:
        if not isinstance(authored, dict):
            raise ConfigError(f"campaign run_refs entries must be mappings in {path}")
        run_id = authored.get("run_id")
        if not isinstance(run_id, str) or not run_id or "{" in run_id:
            raise ConfigError(f"campaign run_ref requires a concrete run_id in {path}")
        if run_id in run_ids:
            raise ConfigError(f"duplicate campaign run_id {run_id!r} in {path}")
        run_ids.add(run_id)
        try:
            memberships.append(CampaignRunMembership(
                run_id=run_id,
                kind="reuse",
                role=authored.get("research_role"),
                arm=authored.get("arm"),
                replicate=authored.get("replicate"),
                purpose=authored.get("purpose"),
                included_in_analysis=authored.get("included_in_analysis", True),
            ))
        except ValidationError as exc:
            raise ConfigError(f"invalid campaign run_ref in {path}: {exc}") from exc
    revision_id = f"campaign.{hashlib.sha256(raw).hexdigest()}"
    return CampaignRevision(
        campaign=ref.name,
        project=project.project,
        revision_id=revision_id,
        file=str(path),
        research_contract=(
            data.get("research_contract")
            if isinstance(data.get("research_contract"), dict) else None
        ),
        memberships=memberships,
    )


def load_research_project(path: Path) -> ResearchProject:
    """Load a project-owned campaign catalog plus optional research questions."""
    data = _load_yaml(path)
    try:
        project = ResearchProject.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid research project {path}: {exc}") from exc
    if project.schema_version != 1:
        raise ConfigError(f"unsupported schema_version in {path}: {project.schema_version}")
    project.base_dir = path.parent.parent if path.parent.name == "experiments" else path.parent
    project.authored_file = path.resolve()
    # Convention: research_project.yaml lives in <repo>/experiments/, and its
    # run_roots / research_questions_dir are repo-relative. Fall back to the file's own
    # directory when the layout differs.
    campaign_names: set[str] = set()
    materialized_run_ids: dict[str, str] = {}
    for ref in project.campaigns:
        if ref.name in campaign_names:
            raise ConfigError(f"duplicate campaign name {ref.name!r} in {path}")
        campaign_names.add(ref.name)
        if ref.file:
            ref.current_revision = _load_campaign_revision(project, ref)
            for membership in ref.current_revision.memberships:
                if membership.kind != "materialize":
                    continue
                previous = materialized_run_ids.get(membership.run_id)
                if previous is not None:
                    raise ConfigError(
                        f"run_id {membership.run_id!r} is materialized by both "
                        f"campaign {previous!r} and {ref.name!r}; use run_refs for reuse"
                    )
                materialized_run_ids[membership.run_id] = ref.name

    if project.research_questions_dir:
        question_dir = Path(project.research_questions_dir)
        if not question_dir.is_absolute():
            question_dir = (project.base_dir / question_dir).resolve()
        if question_dir.is_dir():
            ids: set[str] = set()
            paths = sorted(question_dir.glob("*.yml")) + sorted(question_dir.glob("*.yaml"))
            for question_path in paths:
                question = load_research_question(question_path)
                if question.id in ids:
                    raise ConfigError(
                        f"duplicate research question id {question.id!r} in {question_path}"
                    )
                ids.add(question.id)
                project.research_questions.append(question)
    return project


def load_projects(config: ServerConfig) -> list[ResearchProject]:
    projects = []
    names: set[str] = set()
    for ref in config.projects:
        project = load_research_project(Path(ref.project_file).expanduser())
        if project.project in names:
            raise ConfigError(f"duplicate project name {project.project!r}")
        names.add(project.project)
        projects.append(project)
    return projects
