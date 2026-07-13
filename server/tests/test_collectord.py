"""Collector safety boundary: observation verbs only, terminal runs skipped."""

import textwrap
from pathlib import Path

import pytest

from ml_exp_server.collectord import (
    Collector, CollectorConfig, CollectorLease, ForbiddenVerbError, OBSERVATION_VERBS,
)
from ml_exp_server.ingest.indexer import RunIndex, index_project
from ml_exp_server.schemas import (
    CampaignBinding,
    CampaignRef,
    CampaignRelationship,
    ControllerConfig,
    ResearchProject,
    RunIndexRow,
)
from tests.conftest import FIXTURES


def _project(tmp_path: Path, campaign_name: str) -> ResearchProject:
    campaign_file = tmp_path / "campaign.yml"
    campaign_file.write_text("schema_version: 1\ncampaign: " + campaign_name + "\n")
    return ResearchProject(
        project="elf", title="t", run_roots=[str(FIXTURES / "runs")],
        controller=ControllerConfig(python="/usr/bin/python3",
                                    experimentctl="tools/experimentctl.py",
                                    workdir="."),
        base_dir=tmp_path,
        campaigns=[CampaignRef(name=campaign_name, file=str(campaign_file))],
    )


@pytest.fixture
def collector(tmp_path):
    project = _project(tmp_path, "fusion-len256-gate-h100-20260711")
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)
    return Collector(index=index, projects=[project],
                     config=CollectorConfig(dry_run=True))


@pytest.mark.parametrize("verb", ["submit", "cancel", "stage", "prepare", "assets-verify"])
def test_mutation_verbs_are_rejected(collector, tmp_path, verb):
    project = collector.projects[0]
    with pytest.raises(ForbiddenVerbError):
        collector.build_command(project, tmp_path / "campaign.yml", "run-x", verb)


def test_allowlist_is_exactly_observation_verbs():
    assert OBSERVATION_VERBS == {"observe", "decide", "status", "collect"}


def test_collector_defaults_to_twenty_second_polling():
    assert CollectorConfig().poll_interval_seconds == 20


def test_plan_cycle_targets_only_declared_nonterminal_runs(collector):
    # Execution discovery comes from the Project campaign catalog; this fixture
    # deliberately has no ResearchQuestion objects.
    assert collector.projects[0].research_questions == []
    calls = collector.plan_cycle()
    # A1 (RUNNING, campaign declared) gets observe+decide; the smoke run's
    # campaign is not declared in the project catalog → passive only.
    run_ids = {c.run_id for c in calls}
    assert run_ids == {"elf-a1-frozen-t5-l256-s42-h100-v1"}
    assert [c.verb for c in calls] == ["observe", "decide"]
    for call in calls:
        assert call.argv[3] in OBSERVATION_VERBS
        assert "--run" in call.argv


def test_terminal_runs_are_not_polled(tmp_path):
    project = _project(tmp_path, "camp")
    run_dir = tmp_path / "runs" / "camp" / "done-run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text("run_id: done-run\nproject: elf\ncampaign: camp\n")
    (run_dir / "status.json").write_text('{"state": "SUCCEEDED"}')
    project.run_roots = [str(tmp_path / "runs")]
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)
    collector = Collector(index=index, projects=[project],
                          config=CollectorConfig(dry_run=True))
    assert collector.plan_cycle() == []


def test_revision_drift_is_not_polled_through_current_authored_campaign(tmp_path):
    project = _project(tmp_path, "camp")
    index = RunIndex(tmp_path / "index.sqlite")
    index.upsert_run(RunIndexRow(
        project="elf", campaign="camp", campaign_source="manifest",
        campaign_binding=CampaignBinding(
            relationship=CampaignRelationship.CAMPAIGN_REVISION_DRIFT,
            issues=[CampaignRelationship.CAMPAIGN_REVISION_DRIFT],
            origin_campaign="camp", origin_revision="campaign.old",
            current_revision="campaign.current",
        ),
        run_id="drifted-run", run_dir=str(tmp_path / "runs" / "camp" / "drifted-run"),
        scheduler_state="RUNNING",
    ))
    collector = Collector(index=index, projects=[project],
                          config=CollectorConfig(dry_run=True))
    assert collector.plan_cycle() == []


def test_run_cycle_executes_and_records_errors(tmp_path):
    """A fake experimentctl proves subprocess wiring + error capture."""
    project = _project(tmp_path, "fusion-len256-gate-h100-20260711")
    fake_ctl = tmp_path / "tools" / "experimentctl.py"
    fake_ctl.parent.mkdir(parents=True)
    log = tmp_path / "calls.log"
    fake_ctl.write_text(textwrap.dedent(f"""\
        import sys
        open({str(log)!r}, "a").write(" ".join(sys.argv[1:]) + "\\n")
        if sys.argv[2] == "decide":
            print("boom", file=sys.stderr); sys.exit(1)
    """))
    project.controller = ControllerConfig(python=sys_python(),
                                          experimentctl=str(fake_ctl), workdir=".")
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)
    collector = Collector(index=index, projects=[project],
                          config=CollectorConfig(dry_run=False))
    calls = collector.run_cycle()
    assert len(calls) == 2
    logged = log.read_text().splitlines()
    assert len(logged) == 2 and "observe" in logged[0] and "decide" in logged[1]
    statuses = {s.last_verb: s for s in index.collector_statuses("elf")}
    # decide ran last and failed; its error must be recorded.
    assert statuses["decide"].last_error == "boom"
    assert index.get_meta("collector_last_cycle_at") is not None


def test_collector_lease_allows_exactly_one_owner(tmp_path):
    first = CollectorLease(tmp_path / "index.sqlite")
    second = CollectorLease(tmp_path / "index.sqlite")
    assert first.acquire() is True
    assert second.acquire() is False
    first.release()
    assert second.acquire() is True
    second.release()


def test_observe_failure_is_preserved_and_skips_decide(tmp_path):
    project = _project(tmp_path, "fusion-len256-gate-h100-20260711")
    fake_ctl = tmp_path / "tools" / "experimentctl.py"
    fake_ctl.parent.mkdir(parents=True)
    log = tmp_path / "calls.log"
    fake_ctl.write_text(textwrap.dedent(f"""\
        import sys
        open({str(log)!r}, "a").write(" ".join(sys.argv[1:]) + "\\n")
        if sys.argv[2] == "observe":
            print("ssh timeout", file=sys.stderr); sys.exit(1)
    """))
    project.controller = ControllerConfig(
        python=sys_python(), experimentctl=str(fake_ctl), workdir="."
    )
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)
    collector = Collector(index=index, projects=[project])

    calls = collector.run_cycle()

    assert len(calls) == 2
    assert len(log.read_text().splitlines()) == 1
    status = index.collector_statuses("elf")[0]
    assert "ssh timeout" in status.last_error
    assert status.verb_results["observe"]["outcome"] == "failed"
    assert status.verb_results["decide"]["outcome"] == "skipped"
    assert "observe failed" in status.verb_results["decide"]["error"]


def sys_python() -> str:
    import sys
    return sys.executable
