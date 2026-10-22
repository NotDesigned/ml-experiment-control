"""Read-only Slurm allocation snapshots, bounded and cached per SSH/partition scope."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import os
import re
import shlex
import subprocess
import threading
import time
from experiment_control.runner import run_process
from experiment_control.backends.sensecore_rest import SenseCoreREST, RESTError

from ..storage import utc_now

TTL = 25
MARKER = '\nML_EXPD_QUEUE\n'


def cards(value):
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number < 0 or number > 2147483647:
            return None
        return int(number) if number == int(number) else float(number)
    except InvalidOperation:
        return None


def sensecore_pools(backend):
    rows = SenseCoreREST.from_environment().pools(backend, fresh=True)
    clusters = []
    for row in rows:
        shares = row.get('spot_status')
        counts = [cards(item.get('spot_quota', {}).get('device'))
                  if isinstance(item, dict) and isinstance(item.get('spot_quota'), dict) else None
                  for item in shares] if isinstance(shares, list) else []
        spot = cards(sum(Decimal(str(n)) for n in counts)) if counts and all(n is not None for n in counts) else None
        clusters.append({'name': row['name'], 'zone': row['zone'], 'state': row['state'],
                         'is_debug': 'debug' in row['name'].casefold(),
                         'reported_remaining_cards': cards(row.get('reserved_number')),
                         'spot_allowance_cards': spot})
    return {'clusters': sorted(clusters, key=lambda item: item['name'])}


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
        return self.cached(('slurm', *key), lambda: observe(*key), base,
                           {'resources': None, 'nodes': [], 'pending_queue_entries': None})

    def clusters(self, executor, profile):
        backend = profile['backend']
        return self.cached(('sensecore', backend['workspace']), lambda: sensecore_pools(backend),
                           {'executor_id': executor, 'source': 'sensecore_workspace_bindings',
                            'admission_guaranteed': False, 'cache_ttl_seconds': TTL}, {'clusters': None})

    def cached(self, key, observe, base, empty):
        if not self.lock.acquire(blocking=False):
            return {**base, **empty, 'status': 'UNKNOWN', 'observed_at': None, 'error': 'QUERY_IN_PROGRESS'}
        try:
            now = time.monotonic()
            previous = self.cache.get(key)
            if previous is None or now - previous['checked'] >= TTL:
                try:
                    data = {**observe(), 'observed_at': utc_now(), 'status': 'OBSERVED', 'error': None}
                except (OSError, ValueError, KeyError, subprocess.TimeoutExpired, RESTError):
                    data = ({**previous['data'], 'status': 'STALE'}
                            if previous and previous['data']['observed_at'] else
                            {**empty, 'status': 'UNKNOWN', 'observed_at': None})
                    data['error'] = 'SCHEDULER_QUERY_FAILED'
                if len(self.cache) >= 64 and key not in self.cache:
                    self.cache.pop(next(iter(self.cache)))
                data['checked_at'] = utc_now()
                self.cache[key] = {'checked': time.monotonic(), 'data': data}
            return deepcopy({**base, **self.cache[key]['data']})
        finally:
            self.lock.release()
