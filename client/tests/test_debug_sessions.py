"""Standalone client public-key handling and native-cloud credential separation."""
import pytest
from ml_exp_client.api import ClientError
from ml_exp_client.cli import parser
from ml_exp_client.commands import dispatch
from ml_exp_client.debug_sessions import debug_session

H={'capabilities':['debug-sessions.v1']}


class Client:
    def __init__(self):self.calls=[]
    def call(self,path,**kw):self.calls.append((path,kw));return {'status':'ok'}


@pytest.mark.parametrize('action',['profiles','list','status','refresh','stop','delete'])
def test_queries_and_commands(action):
    c=Client();args=parser().parse_args(['debug',action]+([] if action in ['profiles','list'] else ['--id','my-session']))
    assert dispatch(args,c,H)=={'status':'ok'}
    p,kw=c.calls[0]
    if action in ['profiles','list']:assert p in ['/api/debug-session-profiles','/api/debug-sessions'] and kw=={}
    elif action=='delete':assert p.endswith('/my-session') and kw=={'method':'DELETE'}
    elif action=='status':assert p.endswith('/my-session') and kw=={}
    else:assert p.endswith('/'+action) and kw=={'data':{}}


def test_create_uses_public_key_and_stable_identity(tmp_path):
    key=tmp_path/'id.pub';key.write_text('ssh-ed25519 public comment\n')
    args=parser().parse_args(['debug','create','--id','test','--profile','cpu','--ssh-key',str(key),'--duration-seconds','600'])
    c=Client();debug_session(args,c,H)
    assert c.calls==[('/api/debug-sessions',{'data':{'session_id':'test','profile':'cpu','ssh_public_key':'ssh-ed25519 public comment','duration_seconds':600}})]
    with pytest.raises(ClientError,match='lacks'):debug_session(args,c,{})


@pytest.mark.parametrize('contents',['-----BEGIN OPENSSH PRIVATE KEY-----','command="bad" ssh-ed25519 AAAA',
                                      'ssh-ed25519 AAAA\nssh-ed25519 BBBB','x'*16385,'\u4e2d\u6587'])
def test_rejects_private_or_invalid_key_without_http(tmp_path,contents):
    key=tmp_path/'id.pub';key.write_text(contents)
    args=parser().parse_args(['debug','create','--id','test','--profile','cpu','--ssh-key',str(key)])
    c=Client()
    with pytest.raises(ClientError,match='public key'):debug_session(args,c,H)
    assert not c.calls
    key.unlink()
    with pytest.raises(ClientError):debug_session(args,c,H)
