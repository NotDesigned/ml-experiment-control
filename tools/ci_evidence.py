#!/usr/bin/env python3
"""Reuse a successful PR suite only for its exact, actually checked-out Git tree."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

EVIDENCE = Path('.ops/tested-tree.json')


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def record():
    # Build tools may generate ignored files, but cannot alter tested source.
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--'], check=True)
    value = {key: os.environ['GITHUB_' + key.upper()] for key in
             ('repository', 'run_id', 'run_attempt')}
    value.update(sha=git('rev-parse', 'HEAD'), tree=git('rev-parse', 'HEAD^{tree}'))
    EVIDENCE.parent.mkdir(exist_ok=True)
    EVIDENCE.write_text(json.dumps(value))


def api(path):
    # gh strips authentication on cross-host artifact redirects. Never forward
    # the API token to the signed artifact URL ourselves or execute its content.
    return subprocess.run(['gh', 'api', path], check=True, capture_output=True,
                          timeout=30).stdout


def reusable(repo, sha, run_id, tree, read=api):
    def get(path):
        return json.loads(read(f'repos/{repo}/' + path))
    workflow = get('actions/runs/' + run_id)['workflow_id']
    for pr in get(f'commits/{sha}/pulls?per_page=30'):
        if not pr.get('merged_at') or pr['merge_commit_sha'] != sha or pr['base']['ref'] != 'main':
            continue
        runs = get(f'actions/workflows/{workflow}/runs?event=pull_request&head_sha={pr["head"]["sha"]}&per_page=10')
        for run in runs['workflow_runs']:
            if (run['status'] != 'completed' or run['conclusion'] != 'success' or run['event'] != 'pull_request'
                    or run['head_sha'] != pr['head']['sha'] or run['workflow_id'] != workflow
                    or run['head_repository']['full_name'] != repo):
                continue
            attempt = str(run['run_attempt'])
            artifacts = get(f'actions/runs/{run["id"]}/artifacts?per_page=100')['artifacts']
            for artifact in artifacts:
                if (artifact['name'] != 'quality-evidence-' + attempt or artifact['expired']
                        or artifact['size_in_bytes'] > 65536):
                    continue
                raw = read(f'repos/{repo}/actions/artifacts/{artifact["id"]}/zip')
                with zipfile.ZipFile(io.BytesIO(raw)) as package:
                    entry = package.getinfo('tested-tree.json')
                    if entry.file_size > 65536:
                        continue
                    proof = json.loads(package.read(entry))
                if (proof.get('repository') != repo or proof.get('run_id') != str(run['id'])
                        or proof.get('run_attempt') != attempt or proof.get('tree') != tree
                        or re.fullmatch(r'[0-9a-f]{40}', str(proof.get('sha'))) is None):
                    continue
                # Also verify the recorded checkout commit against GitHub's Git
                # object, rather than trusting a PR head SHA or artifact alone.
                commit = get('git/commits/' + proof['sha'])
                if commit['tree']['sha'] == tree:
                    return run['html_url']
    return None


def plan():
    evidence = None
    reason = 'full checks requested'
    if os.environ['GITHUB_EVENT_NAME'] == 'push':
        try:
            evidence = reusable(os.environ['GITHUB_REPOSITORY'], os.environ['GITHUB_SHA'],
                                os.environ['GITHUB_RUN_ID'], git('rev-parse', 'HEAD^{tree}'))
            reason = 'identical tested PR tree' if evidence else 'no matching successful PR evidence'
        except Exception as error:
            # A missing/expired artifact or API failure can only add checks.
            reason = 'cannot verify PR evidence: ' + type(error).__name__
    full = 'false' if evidence else 'true'
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
        output.write(f'full={full}\n')
    with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a') as summary:
        summary.write(f'Full suite required: **{full}**. {reason}.\n')
        if evidence:
            summary.write(f'Exact-tree evidence: {evidence}\n')
    print(json.dumps({'full': full, 'reason': reason, 'evidence': evidence}))


if __name__ == '__main__':
    {'record': record, 'plan': plan}[sys.argv[1]]()
