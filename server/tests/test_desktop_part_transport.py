"""Ordinary Docker archive HTTP carries bytes; exec carries metadata only."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
from types import SimpleNamespace

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.desktop_upload import DesktopUploads
from ml_exp_server.image_builder import ImageBuilder
from ml_exp_server import image_builder as module


@pytest.fixture
def part_rpc(tmp_path, monkeypatch):
    stage=DesktopUploads(tmp_path/'desktop',{'upload_part_bytes':1024})
    binding={'kind':'asset','project':'demo'}; body=os.urandom(1024)
    upload=stage.call('create',{'binding':binding,'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)})
    metadata={'binding':binding,'upload_id':upload['upload_id'],'number':0,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}
    value=ImageBuilder({'state_root':str(tmp_path/'builder')})
    value.config.update(docker_host='unix:///run/private.sock',data_upload_container='ml-expd-data-stage')
    seen=[]
    def run(command,**kwargs):
        seen.append(command)
        assert '-i' not in command
        if command[3]=='cp':
            assert command[4:]==['-','ml-expd-data-stage:/stage/rpc-parts']
            with tarfile.open(fileobj=io.BytesIO(kwargs['input'])) as archive:
                files=archive.getmembers();assert len(files)==1 and files[0].name.startswith('part.')
                assert abs(files[0].mtime-time.time())<10
                path=stage.incoming/files[0].name
                path.write_bytes(archive.extractfile(files[0]).read())
                os.utime(path,(files[0].mtime,files[0].mtime))
            result=b''
        else:
            DesktopUploads(stage.root,stage.config)
            data=json.loads(command[-1]); result=json.dumps({'ok':True,'result':stage.call(command[-2],data)}).encode()
        return SimpleNamespace(returncode=0,stdout=result,stderr=b'')
    monkeypatch.setattr(module.subprocess,'run',run)
    return value,stage,metadata,body,seen,run


def test_part_carries_tar_and_keeps_confirmed_digest_on_retry(part_rpc):
    value,stage,metadata,body,seen,_=part_rpc
    expected={'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}
    assert value.desktop_stage('part',metadata,body)=={'ok':True,'result':expected}
    assert value.desktop_stage('part',metadata,body)=={'ok':True,'result':expected}
    assert seen[0][3]=='cp' and seen[1][-2]=='part-file' and seen[2][-2]=='part-discard'
    assert not list(stage.incoming.iterdir())
    stored=stage.uploads.read(metadata['upload_id'],metadata['binding'])
    assert stored['parts']=={'0':expected}
    assert (stage.uploads.root/metadata['upload_id']/'parts/0').read_bytes()==body


@pytest.mark.parametrize('case',['cp-failed','exec-failed','timeout','socket','json','shape','cleanup'])
def test_part_failures_are_retryable_and_cleanup_is_scoped(part_rpc,monkeypatch,case):
    value,stage,metadata,body,_,original=part_rpc
    def run(command,**kwargs):
        if command[-2]=='part-discard' and case=='cleanup':raise OSError('private-token')
        if command[3]=='cp':
            if case=='cp-failed':return SimpleNamespace(returncode=1,stdout=b'',stderr=b'private-token')
            if case=='timeout':raise subprocess.TimeoutExpired('private-token',1)
            if case=='socket':raise OSError('private-token')
        if command[-2]=='part-file':
            if case=='exec-failed':return SimpleNamespace(returncode=1,stdout=b'',stderr=b'private-token')
            if case in {'json','shape'}:return SimpleNamespace(returncode=0,stdout=b'private-token' if case=='json' else b'[]',stderr=b'')
        return original(command,**kwargs)
    monkeypatch.setattr(module.subprocess,'run',run)
    if case=='cleanup':assert value.desktop_stage('part',metadata,body)['ok']
    else:
        with pytest.raises(ApplicationError) as error:value.desktop_stage('part',metadata,body)
        assert error.value.status_code==503 and 'private-token' not in str(error.value)
    assert not list(stage.incoming.iterdir())


@pytest.mark.parametrize('case',['empty','size','sha'])
def test_invalid_part_identity_never_reaches_docker(part_rpc,case):
    value,_,metadata,body,seen,_=part_rpc
    if case=='empty':body=b''
    if case=='size':metadata['bytes']+=1
    if case=='sha':metadata['sha256']='f'*64
    with pytest.raises(ValueError):value.desktop_stage('part',metadata,body)
    assert not seen


@pytest.mark.parametrize('case',['identity','symlink','size','hardlink','missing','discard'])
def test_helper_rejects_untrusted_rpc_body_files(part_rpc,case):
    _,stage,metadata,body,_,_=part_rpc
    identity='part.'+'a'*32;path=stage.incoming/identity
    if case=='identity':identity='../other'
    if case=='symlink':path.symlink_to(stage.uploads.root)
    elif case!='missing':path.write_bytes(body)
    if case=='size':metadata['bytes']+=1
    if case=='hardlink':os.link(path,stage.incoming/'other')
    if case=='discard':
        assert stage.call('part-discard',{'body_id':identity})=={'removed':True}
        assert not path.exists()
    else:
        with pytest.raises((ValueError,OSError)):stage.call('part-file',{**metadata,'body_id':identity})


def test_expired_rpc_bodies_leave_active_bodies_and_confirmed_parts(part_rpc,monkeypatch):
    _,stage,metadata,body,_,_=part_rpc
    old=stage.incoming/('part.'+'a'*32);old.write_bytes(body);os.utime(old,(0,0))
    active=stage.incoming/('part.'+'b'*32);active.write_bytes(body)
    retained=stage.incoming/'unrelated';retained.write_bytes(body);os.utime(retained,(0,0))
    directory=stage.incoming/('part.'+'c'*32);directory.mkdir()
    link=stage.incoming/('part.'+'d'*32);link.symlink_to(retained)
    real=Path.lstat
    def stat(path,*args,**kwargs):
        if path==old:raise FileNotFoundError()
        return real(path,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'lstat',stat)
        DesktopUploads(stage.root,stage.config)
    DesktopUploads(stage.root,stage.config)
    assert not old.exists() and active.exists() and retained.exists() and link.is_symlink() and directory.is_dir()
