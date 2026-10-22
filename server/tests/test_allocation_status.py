"""Allocation evidence, freshness, failures and the authenticated HTTP boundary."""
from types import SimpleNamespace
import subprocess
import pytest
from ml_exp_server.runs import allocation_status as a
from tests.test_container_api import client

NODE = 'NodeName=n1 Partitions=gpu,gpu-debug CfgTRES=cpu=16,gres/gpu=8 AllocTRES=cpu=4,gres/gpu=3 CPUTot=16 CPUAlloc=4 RealMemory=1024 AllocMem=256 State=MIXED'
OUTPUT = NODE + a.MARKER + '123\n124_[1-3]\n'
PROFILE = {'backend': {'kind': 'slurm', 'ssh_alias': 'cluster', 'partition': 'gpu'}}

def test_totals_filter_and_units():
    v=a.parse('ignored line\n'+NODE+'\n'+NODE.replace('n1','n2').replace('Partitions=gpu,','Partitions=other,')+a.MARKER+'123\n','gpu')
    assert v['resources']['gpus']=={'total':8,'allocated':3,'unallocated':5}
    assert v['resources']['cpus']=={'total':16,'allocated':4,'unallocated':12}
    assert v['resources']['memory_bytes']['allocated']==256*1024**2
    assert len(v['nodes'])==1 and v['pending_queue_entries']==1
    idle=NODE.replace('cpu=4,gres/gpu=3','').replace('CPUAlloc=4','CPUAlloc=0').replace('AllocMem=256','AllocMem=0').replace('MIXED','DOWN')
    assert a.parse(idle+a.MARKER,'gpu')['resources']['gpus']['allocated']==0
    assert a.parse(OUTPUT.replace('cpu=4,gres/gpu=3','cpu=4'),'gpu')['resources']['gpus']['allocated']==0

@pytest.mark.parametrize('text',[NODE, NODE+a.MARKER+a.MARKER, 'nothing'+a.MARKER,
    NODE+'\n'+NODE+a.MARKER, OUTPUT.replace('CPUAlloc=4','CPUAlloc=-1'),
    OUTPUT.replace('gres/gpu=3','gres/gpu=9'), OUTPUT.replace('cpu=4,gres/gpu=3',''),
    OUTPUT.replace('CfgTRES=cpu=16,gres/gpu=8','CfgTRES=cpu=16'),
    OUTPUT.replace('RealMemory=1024','RealMemory=oops')])
def test_invalid_evidence_never_becomes_zero(text):
    with pytest.raises((ValueError,KeyError)):a.parse(text,'gpu')

@pytest.mark.parametrize('scope',[('-Fbad','gpu'),('cluster','gpu;touch'),(None,'gpu')])
def test_config_scope_cannot_inject_options_or_shell(scope):
    with pytest.raises(ValueError):a.observe(*scope)

@pytest.mark.parametrize('case',['ok','exit','large','timeout'])
def test_readonly_ssh_query_is_timed_and_errors_are_not_exposed(monkeypatch,case):
    def run(args,**kw):
        assert args[:5]==['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5']
        assert args[5]=='cluster' and 'scontrol show nodes' in args[6] and 'squeue' in args[6]
        assert not any(s in args[6] for s in ['sbatch','scancel'])
        assert kw['timeout']==12
        if case=='timeout':raise subprocess.TimeoutExpired(args,12)
        return SimpleNamespace(returncode=1 if case=='exit' else 0,stdout='x'*(1024**2+1) if case=='large' else OUTPUT)
    monkeypatch.delenv('EXPERIMENTCTL_SSH_BIN',raising=False)
    monkeypatch.setattr(a,'run_process',run)
    if case=='ok':assert a.observe('cluster','gpu')['pending_queue_entries']==2
    else:
        with pytest.raises((ValueError,subprocess.TimeoutExpired)):a.observe('cluster','gpu')

def test_cache_stale_recovery_scope_and_copy_isolation(monkeypatch):
    clock=[0];calls=[]
    monkeypatch.setattr(a.time,'monotonic',lambda:clock[0])
    def observe(*args):calls.append(args);return a.parse(OUTPUT,'gpu')
    monkeypatch.setattr(a,'observe',observe)
    cache=a.AllocationStatus()
    v=cache.get('a',PROFILE);v['resources']['gpus']['allocated']=999
    assert cache.get('b',PROFILE)['resources']['gpus']['allocated']==3 and len(calls)==1
    monkeypatch.setattr(a,'observe',lambda *args:(_ for _ in ()).throw(OSError('private secret')))
    clock[0]=26
    stale=cache.get('a',PROFILE)
    assert stale['status']=='STALE' and stale['observed_at']==v['observed_at']
    assert 'private secret' not in str(stale)
    assert cache.get('a',PROFILE)['status']=='STALE'
    clock[0]=52
    monkeypatch.setattr(a,'observe',observe)
    assert cache.get('a',PROFILE)['status']=='OBSERVED'
    for n in range(65):cache.get(str(n),{'backend':{**PROFILE['backend'],'partition':str(n)}})
    assert len(cache.cache)==64

def test_unknown_unsupported_and_busy(monkeypatch):
    monkeypatch.setattr(a,'observe',lambda *args:(_ for _ in ()).throw(ValueError('secret')))
    cache=a.AllocationStatus()
    v=cache.get('a',PROFILE)
    assert v['status']=='UNKNOWN' and v['resources'] is None and v['observed_at'] is None
    cache.lock.acquire()
    try:assert cache.get('a',PROFILE)['error']=='QUERY_IN_PROGRESS'
    finally:cache.lock.release()
    assert cache.get('c',{'backend':{'kind':'sensecore'}})['status']=='UNSUPPORTED'

def test_http_executor_status_is_readonly_and_unknown_is_404(client,monkeypatch):
    calls=[]
    monkeypatch.setattr(a,'observe',lambda *args:calls.append(args) or a.parse(OUTPUT,'gpu'))
    response=client.get('/api/executors/gpu/status')
    assert response.status_code==200 and response.json()['status']=='OBSERVED'
    assert response.json()['admission_guaranteed'] is False
    assert client.get('/api/executors/gpu/status').json()['resources']['gpus']['allocated']==3
    assert len(calls)==1
    assert client.get('/api/executors/missing/status').status_code==404
    assert client.get('/api/executors/cloud/status').json()['status']=='UNSUPPORTED'
    assert client.get('/api/executors').json()['executors'][1]['capabilities']['resources']['inventory_query'] is True
