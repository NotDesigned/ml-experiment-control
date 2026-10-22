"""Read-only Slurm allocation snapshots, bounded and cached per SSH/partition scope."""
from copy import deepcopy
import os
import re
import shlex
import subprocess
import threading
import time
from experiment_control.runner import run_process

from ..storage import utc_now

TTL = 25
MARKER = '\nML_EXPD_QUEUE\n'


def count(value):
    if not re.fullmatch(r'\d+', value):
        raise ValueError('invalid scheduler count')
    return int(value)


def parse(output, partition):
    nodes_text, queue = output.split(MARKER)
    nodes = {}
    for line in nodes_text.splitlines():
        fields = dict(item.split('=', 1) for item in line.split() if '=' in item)
        if partition not in fields.get('Partitions', '').split(','):
            continue
        name = fields['NodeName']
        if name in nodes:
            raise ValueError('duplicate scheduler node')
        configured = dict(item.split('=', 1) for item in fields['CfgTRES'].split(','))
        allocated = dict(item.split('=', 1) for item in fields['AllocTRES'].split(',') if item)
        if not allocated and (fields['CPUAlloc'] != '0' or fields['AllocMem'] != '0'):
            raise ValueError('allocation evidence missing')
        resources = {}
        for key, total, used in (
            ('gpus', configured['gres/gpu'], allocated.get('gres/gpu', '0')),
            ('cpus', fields['CPUTot'], fields['CPUAlloc']),
            ('memory_bytes', str(count(fields['RealMemory']) * 1024**2),
             str(count(fields['AllocMem']) * 1024**2)),
        ):
            total, used = count(total), count(used)
            if used > total:
                raise ValueError('allocation exceeds total')
            resources[key] = {'total': total, 'allocated': used, 'unallocated': total - used}
        nodes[name] = {'node': name, 'state': fields['State'], 'resources': resources}
    if not nodes:
        raise ValueError('partition not visible')
    totals = {key: {field: sum(node['resources'][key][field] for node in nodes.values())
                    for field in ('total', 'allocated', 'unallocated')}
              for key in ('gpus', 'cpus', 'memory_bytes')}
    return {'resources': totals, 'nodes': list(nodes.values()),
            'pending_queue_entries': len([line for line in queue.splitlines() if line.strip()])}


def observe(alias, partition):
    for value in (alias, partition):
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
            raise ValueError('invalid configured scheduler scope')
    command = ("scontrol show nodes --oneliner && printf '\\nML_EXPD_QUEUE\\n' && "
               "squeue -h -p " + shlex.quote(partition) + " -t PENDING -o '%i'")
    command = 'bash -o pipefail -c ' + shlex.quote('(' + command + ') 2>/dev/null | head -c 1048577')
    result = run_process([os.environ.get('EXPERIMENTCTL_SSH_BIN', 'ssh'), '-o', 'BatchMode=yes',
                             '-o', 'ConnectTimeout=5', alias, command],
                            capture_output=True, text=True, timeout=12)
    if result.returncode or len(result.stdout) > 1024**2:
        raise ValueError('scheduler observation failed')
    return parse(result.stdout, partition)


class AllocationStatus:
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()

    def get(self, executor, profile):
        backend = profile['backend']
        base = {'executor_id': executor, 'kind': backend['kind'], 'source': 'slurm',
                'scope': 'partition', 'partition': backend.get('partition'),
                'admission_guaranteed': False, 'cache_ttl_seconds': TTL}
        if backend['kind'] != 'slurm':
            return {**base, 'source': None, 'status': 'UNSUPPORTED', 'resources': None,
                    'nodes': [], 'observed_at': None, 'pending_queue_entries': None}
        key = (backend.get('ssh_alias'), backend.get('partition'))
        if not self.lock.acquire(blocking=False):
            return {**base, 'status': 'UNKNOWN', 'resources': None, 'nodes': [],
                    'observed_at': None, 'pending_queue_entries': None, 'error': 'QUERY_IN_PROGRESS'}
        try:
            now = time.monotonic()
            previous = self.cache.get(key)
            if previous is None or now - previous['checked'] >= TTL:
                try:
                    data = {**observe(*key), 'observed_at': utc_now(), 'status': 'OBSERVED', 'error': None}
                except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
                    data = ({**previous['data'], 'status': 'STALE'}
                            if previous and previous['data']['observed_at'] else
                            {'status': 'UNKNOWN', 'resources': None, 'nodes': [], 'observed_at': None,
                             'pending_queue_entries': None})
                    data['error'] = 'SCHEDULER_QUERY_FAILED'
                if len(self.cache) >= 64 and key not in self.cache:
                    self.cache.pop(next(iter(self.cache)))
                data['checked_at'] = utc_now()
                self.cache[key] = {'checked': time.monotonic(), 'data': data}
            return deepcopy({**base, **self.cache[key]['data']})
        finally:
            self.lock.release()
