"""The early coverage job cannot bypass a failed or foreign GitHub check."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
import ci_wait as c

REPO, RUN, ATTEMPT, HEAD = 'owner/repo', 123, 2, 'a' * 40


def snapshot(shards=15):
    names = sorted(c.required_jobs(shards)) + ['plan', 'main-core-cache', 'test']
    jobs = [{'id': index + 1, 'name': name, 'run_id': RUN, 'run_attempt': ATTEMPT,
             'head_sha': HEAD, 'run_url': f'https://api.github.com/repos/{REPO}/actions/runs/{RUN}',
             'status': 'completed', 'conclusion': 'success'} for index, name in enumerate(names)]
    jobs[-3].update(conclusion='skipped')
    jobs[-2].update(conclusion='skipped')
    jobs[-1].update(status='in_progress', conclusion=None)
    return {'total_count': len(jobs), 'jobs': jobs}


def identity():
    return {'id': RUN, 'run_attempt': ATTEMPT, 'repository': {'full_name': REPO}, 'head_sha': HEAD}


def test_only_complete_unique_current_attempt_checks_succeed():
    value = snapshot()
    assert c.pending_jobs(value, REPO, RUN, ATTEMPT, HEAD, 15) == []
    value['jobs'][0].update(status='in_progress', conclusion=None)
    assert c.pending_jobs(value, REPO, RUN, ATTEMPT, HEAD, 15) == [value['jobs'][0]['name']]
    missing = value['jobs'].pop(0)
    value['total_count'] -= 1
    assert c.pending_jobs(value, REPO, RUN, ATTEMPT, HEAD, 15) == [missing['name']]


@pytest.mark.parametrize('change', [
    {'name': 'unknown'}, {'run_id': RUN + 1}, {'run_id': True}, {'run_attempt': 1},
    {'run_attempt': True}, {'head_sha': 'b' * 40}, {'run_url': 'https://api.github.com/repos/other/repo/actions/runs/123'},
    {'id': True}, {'id': 0}, {'status': 'unknown'}, {'status': 'queued', 'conclusion': 'success'},
    {'conclusion': None}, {'conclusion': 'failure'}, {'conclusion': 'skipped'}, {'conclusion': 'cancelled'},
])
def test_failed_skipped_foreign_or_malformed_jobs_fail_closed(change):
    value = snapshot()
    value['jobs'][0].update(change)
    with pytest.raises(ValueError):
        c.pending_jobs(value, REPO, RUN, ATTEMPT, HEAD, 15)


@pytest.mark.parametrize('duplicate_id', [False, True])
def test_duplicate_job_names_or_ids_cannot_complete_the_set(duplicate_id):
    value = snapshot()
    value['jobs'][1]['id' if duplicate_id else 'name'] = value['jobs'][0]['id' if duplicate_id else 'name']
    with pytest.raises(ValueError):
        c.pending_jobs(value, REPO, RUN, ATTEMPT, HEAD, 15)


@pytest.mark.parametrize('value', [None, [], {}, {'total_count': True, 'jobs': []},
                                  {'total_count': 1, 'jobs': []}, {'total_count': 1, 'jobs': [None]}])
def test_malformed_or_truncated_response_never_means_no_pending_jobs(value):
    with pytest.raises(ValueError):
        c.pending_jobs(value, REPO, RUN, ATTEMPT, HEAD, 15)


@pytest.mark.parametrize('change', [{'id': RUN + 1}, {'run_attempt': 1},
                                   {'repository': {'full_name': 'foreign/repo'}}, {'head_sha': 'bad'}])
def test_attempt_endpoint_identity_must_match(change):
    value = identity()
    value.update(change)
    with pytest.raises(ValueError):
        c.run_identity(value, REPO, RUN, ATTEMPT)


def test_wait_uses_attempt_endpoint_and_retries_only_unavailable_reads():
    now, paths = [0], []
    pending = snapshot()
    pending['jobs'][0].update(status='queued', conclusion=None)
    values = [OSError('not ready'), identity(), pending, subprocess.TimeoutExpired('gh', 30), snapshot()]
    def read(path):
        paths.append(path)
        result = values.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
    c.wait(REPO, RUN, ATTEMPT, 15, read=read, clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    assert not values and now == [6]
    root = f'repos/{REPO}/actions/runs/{RUN}/attempts/{ATTEMPT}'
    assert paths == [root, root, *(3 * [root + '/jobs?per_page=100'])]


def test_api_errors_and_missing_jobs_expire_without_success():
    for unavailable in (True, False):
        now = [0]
        def read(path):
            if unavailable:
                raise subprocess.CalledProcessError(1, 'gh')
            return identity() if not path.endswith('per_page=100') else {'total_count': 0, 'jobs': []}
        with pytest.raises(TimeoutError):
            c.wait(REPO, RUN, ATTEMPT, 15, timeout=3, read=read,
                   clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
        assert now == [3]


def test_late_success_and_malformed_json_do_not_pass_or_retry():
    now, calls = [0], []
    def read(path):
        calls.append(path)
        if len(calls) == 1:
            return identity()
        now[0] = 4
        return snapshot()
    with pytest.raises(TimeoutError):
        c.wait(REPO, RUN, ATTEMPT, 15, timeout=3, read=read, clock=lambda: now[0], sleep=lambda seconds: None)
    def malformed(path):
        raise ValueError('invalid JSON')
    with pytest.raises(ValueError, match='invalid JSON'):
        c.wait(REPO, RUN, ATTEMPT, 15, read=malformed)


@pytest.mark.parametrize('args', [('bad/path/extra', RUN, ATTEMPT, 15), (REPO, 0, ATTEMPT, 15),
                                  (REPO, RUN, 0, 15), (REPO, RUN, ATTEMPT, 0)])
def test_invalid_wait_identity_is_rejected_before_api_reads(args):
    with pytest.raises(ValueError):
        c.wait(*args, read=lambda path: pytest.fail('invalid identity reached API'))


def test_api_reads_only_with_bounded_gh_get(monkeypatch):
    calls = []
    def command(arguments, **options):
        calls.append((arguments, options))
        return b'{"total_count":0,"jobs":[]}'
    monkeypatch.setattr(c.subprocess, 'check_output', command)
    assert c.api('repos/owner/repo/actions/runs/123/attempts/2/jobs?per_page=100') == {'total_count': 0, 'jobs': []}
    assert calls[0][0] == ['gh', 'api', 'repos/owner/repo/actions/runs/123/attempts/2/jobs?per_page=100']
    assert calls[0][1]['timeout'] == 30
