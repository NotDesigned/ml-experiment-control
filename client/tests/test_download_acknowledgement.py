"""A server copy is releasable only after verified, durable local download."""
import hashlib
import io
import json
from pathlib import Path

import pytest

from ml_exp_client.api import ClientError, acknowledge, download
from test_direct_download import archive


class API:
    def __init__(self):
        self.files={'weights.pt':b'weights','sub/metrics.jsonl':b'{}\n'}
        self.body=archive(self.files)
        self.ticket={'transport':'s3-presigned-get','url':'https://objects.example/x?private-signature',
                     'sha256':hashlib.sha256(self.body).hexdigest(),'bytes':len(self.body),
                     'files':[{'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in self.files.items()],
                     'acknowledgement':{'contract':'artifact-retention.v1','grace_seconds':0}}
        self.calls=[];self.downloads=0;self.fail_ack=False;self.out=None
    def call(self,path,**kwargs):
        self.calls.append((path,kwargs))
        if path.endswith('/artifacts/download'):return self.ticket
        assert path == '/api/runs/demo/run-a/attempts/attempt-001/artifacts/ack'
        assert (self.out/'verification.json').exists()
        assert (self.out/'outputs/weights.pt').read_bytes()==b'weights'
        if self.fail_ack:raise ClientError('connection failed',retryable=True)
        proof=kwargs['data']
        assert proof['archive_sha256']==self.ticket['sha256'] and proof['archive_bytes']==len(self.body)
        assert len(proof['files'])==2
        return {'status':'SCHEDULED' if proof['release'] else 'KEEP','release_after':'now' if proof['release'] else None}
    def open_object(self,url):self.downloads+=1;return io.BytesIO(self.body)


@pytest.mark.parametrize('keep', [False,True])
def test_download_acknowledges_only_after_full_validation_and_durable_save(tmp_path,keep,monkeypatch):
    api=API();api.out=tmp_path/'out'
    synced=[]
    from ml_exp_client import api as transport
    original=transport.os.fsync
    def fsync(fd):synced.append(fd);return original(fd)
    monkeypatch.setattr(transport.os,'fsync',fsync)
    report=download(api,'demo','run-a','attempt-001',api.out,keep_server_copy=keep)
    assert report['acknowledgement']['status']==('KEEP' if keep else 'SCHEDULED')
    assert synced and api.downloads==1 and len(api.calls)==2
    assert json.loads((api.out/'verification.json').read_text())==report
    assert 'private' not in (api.out/'verification.json').read_text()


@pytest.mark.parametrize('keep', [False,True])
def test_lost_ack_preserves_results_and_can_retry_without_download(tmp_path,keep):
    api=API();api.out=tmp_path/'out';api.fail_ack=True
    report=download(api,'demo','run-a','attempt-001',api.out,keep_server_copy=keep)
    assert report['acknowledgement']['status']=='PENDING'
    api.fail_ack=False
    result=acknowledge(api,api.out)
    assert result['acknowledgement']['status']==('KEEP' if keep else 'SCHEDULED') and api.downloads==1
    assert sum(path.endswith('/artifacts/ack') for path,_ in api.calls)==2


@pytest.mark.parametrize('failure',['size','sha','file-sha','files'])
def test_corrupt_download_never_sends_a_delete_confirmation(tmp_path,failure):
    api=API();api.out=tmp_path/'out'
    if failure=='size':api.ticket['bytes']+=1
    if failure=='sha':api.ticket['sha256']='a'*64
    if failure=='file-sha':api.ticket['files'][0]['sha256']='b'*64
    if failure=='files':api.ticket['files'].append({'path':'missing','bytes':0,'sha256':'c'*64})
    with pytest.raises(ClientError):download(api,'demo','run-a','attempt-001',api.out)
    assert len(api.calls)==1


@pytest.mark.parametrize('failure',['archive','file','missing-proof','archive-link','file-link','parent-link','escape','wrong-prefix','absolute'])
def test_ack_retry_rechecks_local_bytes_and_rejects_unsafe_paths(tmp_path,failure):
    api=API();api.out=tmp_path/'out';api.fail_ack=True
    download(api,'demo','run-a','attempt-001',api.out)
    calls=len(api.calls);report=json.loads((api.out/'verification.json').read_text())
    if failure=='archive':(api.out/'artifacts.tar').write_bytes(b'corrupt')
    if failure=='file':(api.out/'outputs/weights.pt').write_bytes(b'corrupt')
    if failure=='missing-proof':report['archive_sha256']=None
    if failure=='archive-link':
        p=api.out/'artifacts.tar';data=p.read_bytes();p.unlink();outside=tmp_path/'original.tar';outside.write_bytes(data);p.symlink_to(outside)
    if failure=='file-link':
        p=api.out/'outputs/weights.pt';p.unlink();outside=tmp_path/'weights';outside.write_bytes(b'weights');p.symlink_to(outside)
    if failure=='parent-link':
        p=api.out/'outputs/sub';(p/'metrics.jsonl').unlink();p.rmdir();outside=tmp_path/'external';outside.mkdir();(outside/'metrics.jsonl').write_bytes(b'{}\n');p.symlink_to(outside,target_is_directory=True)
    if failure in {'escape','wrong-prefix','absolute'}:
        name={'escape':'outputs/../external','wrong-prefix':'other/weights.pt','absolute':'/outputs/weights.pt'}[failure]
        report['files'][name]=report['files'].pop('outputs/weights.pt')
    (api.out/'verification.json').write_text(json.dumps(report))
    with pytest.raises(ClientError):acknowledge(api,api.out)
    assert len(api.calls)==calls


def test_ack_cli_keeps_saved_policy_unless_explicitly_overridden():
    from ml_exp_client.cli import parser
    assert parser().parse_args(['acknowledge','--directory','results']).keep_server_copy is None
    assert parser().parse_args(['acknowledge','--directory','results','--keep-server-copy']).keep_server_copy is True
    assert parser().parse_args(['acknowledge','--directory','results','--release-server-copy']).keep_server_copy is False
