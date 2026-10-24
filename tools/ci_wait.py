#!/usr/bin/env python3
"""Wait for this exact GitHub run attempt's checks before combining coverage."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time


def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', path], timeout=30))


def required_jobs(shards):
    return {'core', 'client', 'rust', 'packages', *(f'daemon ({i})' for i in range(shards))}


def run_identity(value, repo, run_id, attempt):
    if (not isinstance(value, dict) or type(value.get('id')) is not int or value['id'] != run_id
            or type(value.get('run_attempt')) is not int or value['run_attempt'] != attempt
            or not isinstance(value.get('repository'), dict)
            or value['repository'].get('full_name') != repo
            or re.fullmatch(r'[0-9a-f]{40}', str(value.get('head_sha'))) is None):
        raise ValueError('run attempt identity is missing or foreign')
    return value['head_sha']


def pending_jobs(value, repo, run_id, attempt, head, shards):
    required = required_jobs(shards)
    allowed = required | {'test', 'plan', 'main-core-cache'}
    if (not isinstance(value, dict) or not isinstance(value.get('jobs'), list)
            or type(value.get('total_count')) is not int
            or value['total_count'] != len(value['jobs']) or value['total_count'] >= 100):
        raise ValueError('jobs response is malformed or truncated')
    seen, ids, succeeded = set(), set(), set()
    for job in value['jobs']:
        if not isinstance(job, dict):
            raise ValueError('job is malformed')
        name, job_id = job.get('name'), job.get('id')
        if (not isinstance(name, str) or name not in allowed or name in seen
                or type(job_id) is not int or job_id <= 0 or job_id in ids
                or type(job.get('run_id')) is not int or job['run_id'] != run_id
                or type(job.get('run_attempt')) is not int or job['run_attempt'] != attempt
                or job.get('head_sha') != head
                or job.get('run_url') != f'https://api.github.com/repos/{repo}/actions/runs/{run_id}'
                or job.get('status') not in {'queued', 'in_progress', 'completed'}
                or job['status'] != 'completed' and job.get('conclusion') is not None
                or job['status'] == 'completed' and job.get('conclusion') not in {
                    'success', 'failure', 'cancelled', 'skipped', 'timed_out', 'action_required',
                    'neutral', 'startup_failure', 'stale'}):
            raise ValueError('job identity, name or state is invalid or duplicated')
        seen.add(name)
        ids.add(job_id)
        if name in required and job['status'] == 'completed':
            if job['conclusion'] != 'success':
                raise ValueError(f'required job {name} did not succeed: {job["conclusion"]}')
            succeeded.add(name)
    return sorted(required - succeeded)


def wait(repo, run_id, attempt, shards, *, phase='all', read=api, clock=time.monotonic, sleep=time.sleep, timeout=540):
    if (re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*', repo) is None
            or type(run_id) is not int or run_id <= 0 or type(attempt) is not int or attempt <= 0
            or type(shards) is not int or not 1 <= shards <= 64 or not 0 < timeout <= 540
            or phase not in {'daemon', 'all'}):
        raise ValueError('invalid repository, run, attempt, shard count or timeout')
    path = f'repos/{repo}/actions/runs/{run_id}/attempts/{attempt}'
    deadline, head, previous = clock() + timeout, None, None
    while clock() < deadline:
        try:
            if head is None:
                head = run_identity(read(path), repo, run_id, attempt)
            pending = pending_jobs(read(path + '/jobs?per_page=100'), repo, run_id, attempt, head, shards)
            if phase == 'daemon':
                pending = [name for name in pending if name.startswith('daemon (')]
        except (OSError, subprocess.SubprocessError) as error:
            print('GitHub checks unavailable; retrying: ' + type(error).__name__, flush=True)
        else:
            if clock() >= deadline:
                break
            if not pending:
                count, label = (shards, 'daemon jobs') if phase == 'daemon' else (len(required_jobs(shards)), 'required jobs')
                print(f'All {count} {label} succeeded for run {run_id}, attempt {attempt}', flush=True)
                return
            if pending != previous:
                print('Waiting for checks: ' + ', '.join(pending), flush=True)
                previous = pending
        sleep(min(1, max(0, deadline - clock())))
    raise TimeoutError('required checks did not finish within the bounded wait')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard-count', type=int, required=True)
    parser.add_argument('--phase', choices=('daemon', 'all'), default='all')
    args = parser.parse_args()
    wait(os.environ['GITHUB_REPOSITORY'], int(os.environ['GITHUB_RUN_ID']),
         int(os.environ['GITHUB_RUN_ATTEMPT']), args.shard_count, phase=args.phase)
