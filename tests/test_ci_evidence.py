"""CI reuse never substitutes a different tree, failed run or stale artifact."""
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
import ci_evidence as c

REPO='owner/repo'
SHA='a'*40
TREE='b'*40


def case():
    pr={'merged_at':'now','merge_commit_sha':SHA,'base':{'ref':'main'},'head':{'sha':'c'*40}}
    run={'id':5,'status':'completed','conclusion':'success','event':'pull_request','head_sha':'c'*40,'workflow_id':1,
         'run_attempt':2,'head_repository':{'full_name':REPO},'html_url':'https://github.com/owner/repo/actions/runs/5'}
    artifact={'id':6,'name':'quality-evidence-2','expired':False,'size_in_bytes':500}
    proof={'repository':REPO,'run_id':'5','run_attempt':'2','sha':'d'*40,'tree':TREE}
    def read(path):
        if '/commits/'+SHA+'/pulls?' in path:return json.dumps([pr]).encode()
        if path.endswith('actions/runs/10'):return b'{"workflow_id":1}'
        if 'actions/workflows/1/runs?' in path:return json.dumps({'workflow_runs':[run]}).encode()
        if path.endswith('actions/runs/5/artifacts?per_page=100'):return json.dumps({'artifacts':[artifact]}).encode()
        if path.endswith('/zip'):
            body=io.BytesIO()
            with zipfile.ZipFile(body,'w') as archive:archive.writestr('tested-tree.json',json.dumps(proof))
            return body.getvalue()
        assert path.endswith('/git/commits/'+'d'*40),path
        return json.dumps({'tree':{'sha':TREE}}).encode()
    return pr,run,artifact,proof,read


def test_exact_merge_tree_reuses_successful_suite():
    *_,read=case()
    assert c.reusable(REPO,SHA,'10',TREE,read).endswith('/runs/5')
    assert c.reusable(REPO,SHA,'10','e'*40,read) is None


@pytest.mark.parametrize('index,key,value',[
    (0,'merged_at',None),(0,'merge_commit_sha','other'),(0,'base',{'ref':'other'}),
    (1,'status','in_progress'),(1,'conclusion','failure'),(1,'event','push'),(1,'head_sha','other'),(1,'workflow_id',2),
    (1,'head_repository',{'full_name':'fork/repo'}),
    (2,'name','quality-evidence-1'),(2,'expired',True),(2,'size_in_bytes',65537),
    (3,'sha','not-a-commit'),(3,'repository','other/repo'),(3,'run_id','4'),(3,'run_attempt','1'),(3,'tree','e'*40),
])
def test_mismatched_or_untrusted_evidence_runs_full_suite(index,key,value):
    values=case();values[index][key]=value
    assert c.reusable(REPO,SHA,'10',TREE,values[-1]) is None


def test_git_object_must_match_recorded_tree():
    *_,read=case()
    def changed(path):return b'{"tree":{"sha":"wrong"}}' if '/git/commits/' in path else read(path)
    assert c.reusable(REPO,SHA,'10',TREE,changed) is None


@pytest.mark.parametrize('event,result,error',[
    ('pull_request',None,False),('workflow_dispatch',None,False),('push',None,False),
    ('push','https://github.com/owner/repo/actions/runs/5',False),('push',None,True),
])
def test_plan_defaults_full_and_only_positive_evidence_skips(monkeypatch,tmp_path,event,result,error):
    for k,v in {'EVENT_NAME':event,'REPOSITORY':REPO,'SHA':SHA,'RUN_ID':'10',
                'OUTPUT':str(tmp_path/'output'),'STEP_SUMMARY':str(tmp_path/'summary')}.items():
        monkeypatch.setenv('GITHUB_'+k,v)
    monkeypatch.setattr(c,'git',lambda *a:TREE)
    calls=[]
    def reuse(*args):
        calls.append(args)
        if error:raise OSError('read unavailable')
        return result
    monkeypatch.setattr(c,'reusable',reuse);c.plan()
    assert (tmp_path/'output').read_text()=='full='+('false' if result else 'true')+'\n'
    assert bool(calls)==(event=='push')


def test_record_binds_actual_checkout_and_checks_tracked_source(monkeypatch,tmp_path):
    for key,value in {'REPOSITORY':REPO,'RUN_ID':'5','RUN_ATTEMPT':'2'}.items():monkeypatch.setenv('GITHUB_'+key,value)
    monkeypatch.setattr(c,'EVIDENCE',tmp_path/'proof/tested-tree.json')
    monkeypatch.setattr(c,'git',lambda *a:TREE if a[-1]=='HEAD^{tree}' else SHA)
    calls=[];monkeypatch.setattr(c.subprocess,'run',lambda *a,**k:calls.append((a,k)))
    c.record();value=json.loads(c.EVIDENCE.read_text())
    assert value=={'repository':REPO,'run_id':'5','run_attempt':'2','sha':SHA,'tree':TREE}
    assert calls[0][0][0]==['git','diff','--exit-code','HEAD','--'] and calls[0][1]['check']
