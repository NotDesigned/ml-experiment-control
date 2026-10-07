"""REST wire contracts and failure boundaries; never allocate cloud resources."""
import base64
import hashlib
import hmac
import io
import json
from pathlib import Path
from unittest.mock import Mock
import urllib.error
import urllib.parse

import pytest

from experiment_control.backends import sensecore_rest as m

UID = "11111111-1111-4111-8111-111111111111"
WS = dict(name="ws", region="cn-sh-01", zone="cn-sh-01z",
          subscription_name="sub", resource_group_name="group", id="workspace-id")
POOL = dict(name="pool", state="ACTIVE", uid="pool-id",
            id="/subscriptions/sub/resourceGroups/group/zones/cn-sh-01e/aec2s/pool")
B = dict(workspace="ws", aec2="pool", worker_spec="gpu", storage_mount="volume/user:/data", quota_type="spot")
SPEC = dict(name="gpu", cpu={"vcpu_allocatable": 2}, memory={"allocatable": 4},
            device={"number": 0}, zones=["cn-sh-01e"])
JOB = dict(name="job", uid="job-id", state="RUNNING", ownership={"user_id": UID})


def client():
    return m.SenseCoreREST(dict(access_key_id="fake-ak", access_key_secret="fake-sk",
                               subscription_name="sub", resource_group_name="group"))


def ready():
    c = client()
    c._identity = UID
    c._workspaces["ws"] = WS
    return c


def response(data):
    r = Mock()
    r.__enter__ = Mock(return_value=r)
    r.__exit__ = Mock(return_value=False)
    r.read.return_value = data
    return r


def test_signed_wire_request_and_no_proxy_redirect_or_automatic_retry(monkeypatch):
    opener = Mock()
    opener.open.return_value = response(b'{"ok":true}')
    build = Mock(return_value=opener)
    monkeypatch.setattr(m.urllib.request, "build_opener", build)
    monkeypatch.setattr(m.email.utils, "formatdate", lambda **kw: "fixed-date")
    c = client()
    assert c.request("https://aec2.cn-sh-01.sensecoreapi.cn/jobs", method="POST", body={"name":"job"}, timeout=12) == {"ok": True}
    req = opener.open.call_args.args[0]
    signature = base64.b64encode(hmac.new(b"fake-sk", b"x-date: fixed-date", hashlib.sha256).digest()).decode()
    assert signature in req.get_header("Authorization")
    assert req.get_header("X-date") == "fixed-date"
    assert json.loads(req.data) == {"name":"job"}
    assert opener.open.call_args.kwargs == {"timeout": 12}
    assert build.call_args.args[0].proxies == {}
    assert isinstance(build.call_args.args[1], m.NoRedirect)
    assert m.NoRedirect().redirect_request(None,None,302,None,None,"https://evil.test") is None
    assert opener.open.call_count == 1
    opener.open.return_value = response(b"")
    assert c.request("https://iam.sensecoreapi.cn/me", signed=False) == {}
    assert not opener.open.call_args.args[0].has_header("Authorization")


@pytest.mark.parametrize("url", ["http://iam.sensecoreapi.cn/x", "https://evil.test/x", "https:///x",
    "https://user:pass@iam.sensecoreapi.cn/x", "https://iam.sensecoreapi.cn:444/x", "https://iam.sensecoreapi.cn/x#secret",
    "https://sensecoreapi.cn.evil.test/x"])
def test_endpoint_validation_precedes_credentials(url):
    with pytest.raises(ValueError, match="HTTPS"):
        client().request(url)


@pytest.mark.parametrize("body,reason", [(b'{"details":[{"reason":"Denied","secret":"raw"}]}',"Denied"),
    (b'{"details":[null,{"reason":"raw secret"}]}',None), (b'[]',None), (b'{"details":null}',None), (b'not-json',None)])
def test_http_errors_expose_only_codes_not_provider_body(monkeypatch, body, reason):
    opener = Mock()
    opener.open.side_effect = urllib.error.HTTPError("https://secret.test",403,"secret",{},io.BytesIO(body))
    monkeypatch.setattr(m.urllib.request, "build_opener", lambda *a: opener)
    with pytest.raises(m.RESTError) as e:
        client().request("https://iam.sensecoreapi.cn/me")
    assert e.value.status == 403 and e.value.details["provider_reason"] == reason
    assert "secret" not in str(e.value)


@pytest.mark.parametrize("error", [TimeoutError("secret"), urllib.error.URLError("secret"), OSError("secret")])
@pytest.mark.parametrize("method", ["GET","POST"])
def test_transport_failure_does_not_replay_unknown_writes(monkeypatch, error, method):
    opener = Mock(); opener.open.side_effect = error
    monkeypatch.setattr(m.urllib.request, "build_opener", lambda *a: opener)
    with pytest.raises(m.RESTError) as e:
        client().request("https://iam.sensecoreapi.cn/me", method=method)
    assert e.value.details["uncertain"] == (method == "POST")
    assert "secret" not in str(e.value)
    assert opener.open.call_count == 1


@pytest.mark.parametrize("raw", [b'not-json', b'x'*(32*1024*1024+1)])
def test_successful_but_malformed_or_oversized_write_requires_reconciliation(monkeypatch, raw):
    opener=Mock();opener.open.return_value=response(raw)
    monkeypatch.setattr(m.urllib.request,"build_opener",lambda *a:opener)
    with pytest.raises(m.RESTError) as e:client().request("https://iam.sensecoreapi.cn/me",method="POST")
    assert e.value.details["uncertain"]


def test_server_error_is_uncertain_even_with_provider_response(monkeypatch):
    opener=Mock();opener.open.side_effect=urllib.error.HTTPError("u",503,"m",{},io.BytesIO(b'{}'))
    monkeypatch.setattr(m.urllib.request,"build_opener",lambda *a:opener)
    with pytest.raises(m.RESTError) as e:client().request("https://iam.sensecoreapi.cn/me",method="POST")
    assert e.value.details["uncertain"]


@pytest.mark.parametrize("value", ["../ws", "", ".", "..", "ws/x", None])
def test_resource_components_cannot_escape_scope(value):
    with pytest.raises(ValueError):m.component(value)


def test_mount_and_document_preserve_existing_gpu_and_cpu_scopes():
    d=m.create_document(B,"job","registry/image@sha256:"+"a"*64,"python train.py")
    assert d["resource_pool"] == {"name":"pool"}
    assert d["mount"] == [{"type":"PV_AFS","id":"volume","mount_path":"/data","subdir":"/user"}]
    assert d["roles"][0]["total_replicas"] == 1
    assert "replicas" not in d["roles"][0]["resource_spec"][0]
    assert d["scheduling"]["quota_type"]=="SPOT" and d["fault_tolerance"]["backoff_limit"]==0
    assert m.create_document({**B,"storage_mount":"volume:/data","quota_type":"reserved"},"job","i","cmd")["mount"][0]["subdir"]=="/"
    with pytest.raises(ValueError):m.origin("aec2",{**WS,"region":"bad"})


@pytest.mark.parametrize("change", [{"storage_mount":"volume:relative"},{"storage_mount":":/data"},
    {"storage_mount":"volume/../secret:/data"},{"quota_type":"bad"},{"worker_nodes":True},
    {"worker_nodes":0},{"worker_nodes":"1"}])
def test_invalid_documents_fail_before_submission(change):
    with pytest.raises(ValueError):m.create_document({**B,**change},"job","i","cmd")


def test_private_configuration_errors_do_not_print_file_or_credentials(monkeypatch,tmp_path):
    p=tmp_path/"private.json";monkeypatch.setenv("EXPERIMENTCTL_SENSECORE_REST_CONFIG",str(p))
    for payload in [None,"[]",'{"access_key_id":"secret"}',"not-json"]:
        if payload is not None:p.write_text(payload)
        with pytest.raises(m.RESTError) as e:m.SenseCoreREST.from_environment()
        assert str(p) not in str(e.value) and "secret" not in str(e.value)
    p.write_text(json.dumps(client().config));assert m.SenseCoreREST.from_environment().config==client().config
    monkeypatch.delenv("EXPERIMENTCTL_SENSECORE_REST_CONFIG")
    monkeypatch.setattr(m.Path,"read_text",lambda *a:json.dumps(client().config))
    assert m.SenseCoreREST.from_environment().config==client().config


def test_pagination_follows_real_tokens_and_numbered_apis_without_losing_rows():
    c=client(); c.request=Mock(side_effect=[{"items":[{"name":"a"}],"total_size":2,"next_page_token":"next"},
                                          {"items":[{"name":"b"}],"total_size":2}])
    assert [r['name'] for r in c.pages("https://x","items")]==['a','b']
    assert 'page_token=next' in c.request.call_args.args[0]
    c.request=Mock(side_effect=[{"items":[{"name":"a"}],"total_size":2}, {"items":[{"name":"b"}],"total_size":2}])
    assert len(c.pages("https://x","items",numbered=True))==2
    assert 'page_token=2' in c.request.call_args.args[0]


@pytest.mark.parametrize("data", [[],{}, {"items":{}},{"items":[None]},{"items":[{}]},
    {"items":[{"name":""}]},{"items":[{"name":"a"},{"name":"a"}]},
    {"items":[],"total_size":True},{"items":[],"total_size":"2"},
    {"items":[{"name":"a"}],"total_size":0},{"items":[],"next_page_token":True}])
def test_malformed_pagination_fails_closed(data):
    c=client();c.request=Mock(return_value=data)
    with pytest.raises(ValueError):c.pages("x","items")


def test_incomplete_repeated_or_unbounded_pagination_never_reports_absence():
    c=client()
    for data,numbered in [({"items":[],"total_size":1},True),({"items":[{"name":"a"}],"total_size":2},False),
        ({"items":[],"next_page_token":"1"},False),({"items":[],"next_page_token":"next","total_size":1},True)]:
        c.request=Mock(return_value=data)
        with pytest.raises(ValueError):c.pages("x","items",numbered=numbered)
    count=iter(range(1000));c.request=lambda u:{"items":[],"next_page_token":str(next(count)+2)}
    with pytest.raises(ValueError,match="limit"):c.pages("x","items")
    c.request=Mock(return_value={"items":[],"next_page_token":"0"})
    assert c.pages("x","items")==[]


@pytest.mark.parametrize("data", [None,{}, {"id":"invalid"},{"id":None},{"id":"00000000-0000-0000-0000-000000000000"},{"id":5}])
def test_invalid_account_identity_cannot_be_cached(data):
    c=client();c.request=Mock(return_value=data)
    with pytest.raises(ValueError):c.identity()
    assert c._identity is None


def test_identity_resources_workspace_cache_scope_and_duplicate_names():
    c=client();c.request=Mock(return_value={"id":UID});assert c.identity()==c.identity()==UID;assert c.request.call_count==1
    c.pages=Mock(return_value=[{**WS,"type":"compute.workspace.v1.instance"},
        {**WS,"name":"deleted","type":"compute.workspace.v1.instance","deleted":True},
        {**WS,"name":"foreign","type":"compute.workspace.v1.instance","subscription_name":"other"},
        {**WS,"name":"wrong","type":"other"}])
    assert c.workspace("ws")==WS|{"type":"compute.workspace.v1.instance"}
    assert c.workspace("ws")["name"]=="ws" and c.pages.call_count==1
    with pytest.raises(ValueError):c.workspace("missing")
    c.pages.return_value=[{**WS,"type":"compute.workspace.v1.instance"}]*2;c._workspaces={}
    with pytest.raises(ValueError):c.workspace("ws")


def test_bound_pool_and_spec_discovery_use_pool_zone_not_workspace_zone():
    c=ready();c.request=Mock(side_effect=[{"aec2s":[POOL,{"name":"inactive","state":"DISABLED"}]},
                                         {"resource_specs":[SPEC]}])
    assert c.specs(B)==[SPEC]
    assert '/zones/cn-sh-01e/aec2s/pool/resourceSpecs' in c.request.call_args.args[0]
    assert c.pool(B)["zone"]=='cn-sh-01e' and c.request.call_count==2
    with pytest.raises(ValueError):c.pool({**B,"aec2":"other"})


@pytest.mark.parametrize("change", [{"id":"bad"},{"id":POOL['id'].replace('/sub/','/other/')},
    {"id":POOL['id'].replace('/group/','/other/')},{"name":"other"},
    {"id":POOL['id'].replace('cn-sh-01e','cn-gz-01a')}])
def test_pool_scope_drift_blocks_submission(change):
    c=ready();c.request=Mock(return_value={"aec2s":[{**POOL,**change}]})
    with pytest.raises(ValueError):c.pools(B)


@pytest.mark.parametrize("change", [{"cpu":{}},{"cpu":None},{"cpu":{"vcpu_allocatable":True}},
    {"memory":{"allocatable":0}},{"device":{"number":-1}},{"zones":[]}])
def test_invalid_cpu_gpu_resources_fail_closed(change):
    c=ready();c.pool=lambda b:{**WS,"zone":"cn-sh-01e"};c.pages=Mock(return_value=[{**SPEC,**change}])
    with pytest.raises(ValueError):c.specs(B)


@pytest.mark.parametrize("row", [None,[],{**JOB,"name":"other"},{**JOB,"uid":""},{**JOB,"ownership":None},
                               {**JOB,"ownership":{"user_id":"foreign"}}])
def test_exact_job_owner_and_identity_are_mandatory(row):
    with pytest.raises(ValueError):ready().owned(row,"job")


def test_find_describe_create_and_stop_contracts_and_terminal_guard():
    c=ready();c.pages=Mock(return_value=[JOB,{"name":"job-other"}]);assert c.find(B,"job")==[JOB]
    query=c.pages.call_args.kwargs['query'];assert query['filter']=="name='job'" and query['name']=='job'
    c.request=Mock(return_value=JOB);assert c.describe(B,"job")==JOB
    c.pool=lambda b:{**WS,"zone":"cn-sh-01e"}
    doc=m.create_document(B,"job","image","cmd");assert c.create(B,doc,timeout=42)==JOB
    call=c.request.call_args;assert call.kwargs['method']=='POST' and call.kwargs['timeout']==42
    assert call.kwargs['body']['mount'][0]['zone']=='cn-sh-01e' and 'zone' not in doc['mount'][0]
    assert 'training_job_name=job' in call.args[0]
    c.stop(B,"job");call=c.request.call_args;assert call.args[0].endswith(':batchStop')
    assert call.kwargs['body']==dict(subscription_name='sub',resource_group_name='group',zone='cn-sh-01z',workspace_name='ws',training_job_names=['job'])
    c.request=Mock(return_value={**JOB,'state':'FAILED'});c.stop(B,"job");assert c.request.call_count==1
    c.request=Mock(return_value={"workers":[],"total_size":0});c.describe=Mock(return_value=JOB)
    c.pages=m.SenseCoreREST.pages.__get__(c)
    assert c.workers(B,'job')==[] and '/job/workers?' in c.request.call_args.args[0]


@pytest.mark.parametrize("result", [None,[],{}, {"name":"other","uid":"id"},{"name":"job"}])
def test_unconfirmed_create_is_never_automatically_replayed(result):
    c=ready();c.pool=lambda b:WS;c.request=Mock(return_value=result)
    with pytest.raises(m.RESTError) as e:c.create(B,m.create_document(B,'job','i','cmd'))
    assert e.value.details['uncertain'] and c.request.call_count==1


def log_client():
    c=ready();c.describe=Mock(return_value=JOB)
    c.workers=Mock(return_value=[{'name':'worker','containers':[{'name':'worker-container'}]}])
    c.resources=Mock(return_value=[{**WS,'name':'ts-user-'+UID}])
    return c


def jwt(endpoint):
    body=base64.urlsafe_b64encode(json.dumps({'Endpoint':endpoint}).encode()).decode().rstrip('=')
    return 'header.'+body+'.signature'


def test_log_token_is_scoped_and_not_signed_with_account_key_on_polling():
    c=log_client();encoded=base64.b64encode(b'Step 403 loss 1.5').decode()
    c.request=Mock(side_effect=[{'token':jwt('logs.sensecoreapi.cn')},
        {'entries':[{'type':'BROKEN'},{'type':'EOF','message':'done'},
                    {'type':'HEARTBEAT','content':'alive'},{'data':encoded},{'content':encoded},
                    {'message':encoded,'data':'must not decode fallback'}]}])
    result=c.logs(B,'job',10)
    assert not result['expired'] and 'Step 403' in result['text']
    token_call,poll_call=c.request.call_args_list
    assert token_call.kwargs['body']['resource_id']==WS['id']
    assert {'key':'lepton.sensetime.com/workload-uid','val':'job-id'} in token_call.kwargs['body']['filters']
    assert poll_call.kwargs['signed'] is False and poll_call.kwargs['body']['tail']==10
    assert 'token' not in poll_call.args[0]


def test_unallocated_workers_return_empty_logs_without_token_request():
    c=log_client();c.workers.return_value=[];c.request=Mock()
    assert c.logs(B,'job',10)=={'text':'','expired':False,'exit_code':0};c.request.assert_not_called()


def test_missing_personal_station_cannot_use_foreign_station():
    c=log_client();c.resources.return_value=[{**WS,'name':'foreign'}]
    with pytest.raises(ValueError):c.logs(B,'job',10)


@pytest.mark.parametrize('response', [{}, {'token':'not-jwt'},{'token':jwt('evil.test')},
    {'token':jwt('logs.sensecoreapi.cn/path')},{'token':jwt(1)}])
def test_invalid_log_tokens_cannot_send_keys_elsewhere(response):
    c=log_client();c.request=Mock(return_value=response)
    # Mock transport still enforces URL policy on the external endpoint.
    original=c.request
    def request(url,**kw):
        if '/polling/' in url:
            return client().request(url,**kw)
        return original(url,**kw)
    c.request=request
    with pytest.raises(ValueError):c.logs(B,'job',10)


@pytest.mark.parametrize('status,reason,expired', [(403,'ExpiredPodToken',True),(403,'Denied',False),(503,None,False)])
def test_expiry_and_permission_or_connectivity_failures_are_distinct(status,reason,expired):
    c=log_client();c.request=Mock(side_effect=m.RESTError('token',status=status,reason=reason))
    result=c.logs(B,'job',10)
    assert result['expired']==expired and not result['available']
    assert result['text']=='' and result['error']['provider_reason']==reason


@pytest.mark.parametrize('value', [[],{}, {'entries':{}},{'entries':[None]}, {'entries':[{'data':'not base64!'}]}, {'entries':[{'data':1}]},
    {'entries':[{'data':base64.b64encode(b'\xff').decode()}]}])
def test_log_responses_are_bounded_and_validated(value):
    c=log_client();c.request=Mock(side_effect=[{'token':jwt('logs.sensecoreapi.cn')},value])
    with pytest.raises(ValueError):c.logs(B,'job',10)


def test_read_only_post_failure_never_requires_job_replay(monkeypatch):
    opener = Mock()
    monkeypatch.setattr(m.urllib.request, 'build_opener', lambda *a: opener)
    for error in (TimeoutError('private'), urllib.error.HTTPError('u', 503, 'm', {}, io.BytesIO(b'{}'))):
        opener.open.side_effect = error
        with pytest.raises(m.RESTError) as caught:
            client().request('https://monitor.sensecoreapi.cn/logs', method='POST', read_only=True)
        assert caught.value.details['uncertain'] is False


def test_fresh_binding_discovery_does_not_reuse_stale_quota():
    c = ready()
    c.request = Mock(side_effect=[{'aec2s': [POOL]}, {'aec2s': [{**POOL, 'reserved_number': 3}]}])
    assert 'reserved_number' not in c.pools(B)[0]
    assert c.pools(B, fresh=True)[0]['reserved_number'] == 3
    assert c.request.call_count == 2


def selector(pools):
    c = ready()
    c.pools = Mock(return_value=pools)
    c.pool = lambda b: next(p for p in pools if p['name'] == b['aec2'])
    c.specs = Mock(return_value=[{**SPEC, 'device': {'number': 1}}])
    return c


def binding(name='pool', spot='0', **fields):
    return {**POOL, 'name': name, 'zone': 'cn-sh-01e', 'vpc_id': 'same-network',
            'spot_status': [{'spot_quota': {'device': spot}}], **fields}


def test_select_highest_spot_not_most_remaining_and_freeze_evidence():
    pools = [binding(reserved_number=900), binding('other', '11', reserved_number=2),
             binding('debug-pool', '1000'), binding('foreign', '100', vpc_id='other-network'),
             binding('other-zone', '99', zone='cn-sh-01a'), binding('unknown-network', '99', vpc_id=None)]
    c = selector(pools)
    result = c.select_pool({**B, 'pool_selection': 'highest_spot'}, gpus=1)
    assert result['aec2'] == 'other'
    assert result['pool_selection_evidence']['configured_aec2'] == 'pool'
    assert result['pool_selection_evidence']['candidates'] == [
        {'name': 'other', 'spot_devices': '11'}, {'name': 'pool', 'spot_devices': '0'}]
    c.pools.assert_called_once_with({**B, 'pool_selection': 'highest_spot'}, fresh=True)


def test_allowlist_spec_match_sum_and_deterministic_tie():
    c = selector([binding(), binding('second', '10'), binding('first', '10'), binding('excluded', '999')])
    b = {**B, 'pool_selection': 'highest_spot', 'allowed_clusters': ['first', 'second']}
    assert c.select_pool(b, gpus=1)['aec2'] == 'first'
    c.specs.return_value = [SPEC]
    with pytest.raises(ValueError, match='no compatible'):
        c.select_pool(b, gpus=1)
    c.specs.side_effect = m.RESTError('specs', status=403)
    with pytest.raises(m.RESTError):
        c.select_pool(b, gpus=1)
    c = selector([binding(spot_status=[{'spot_quota': {'device': '1.5'}}, {'spot_quota': {'device': '2.5'}}])])
    assert c.select_pool({**B, 'pool_selection': 'highest_spot'}, gpus=1)['pool_selection_evidence']['candidates'][0]['spot_devices'] == '4.0'


@pytest.mark.parametrize('change,gpus', [({}, 1), ({'pool_selection': 'highest_spot', 'quota_type': 'reserved'}, 1),
    ({'pool_selection': 'highest_spot'}, True), ({'pool_selection': 'highest_spot'}, 0),
    ({'pool_selection': 'highest_spot'}, '1'),
    ({'pool_selection': 'highest_spot', 'allowed_clusters': []}, 1),
    ({'pool_selection': 'highest_spot', 'allowed_clusters': 'pool'}, 1),
    ({'pool_selection': 'highest_spot', 'allowed_clusters': [1]}, 1)])
def test_invalid_selection_policies_do_not_create_jobs(change, gpus):
    with pytest.raises(ValueError):
        selector([binding()]).select_pool({**B, **change}, gpus=gpus)


@pytest.mark.parametrize('shares', [None, [], {}, [{}], [None], [{'spot_quota': {'device': 'bad'}}],
    [{'spot_quota': {'device': 'NaN'}}], [{'spot_quota': {'device': '-1'}}],
    [{'spot_quota': {'device': 'Infinity'}}], [{'spot_quota': {'device': '1e100'}}]])
def test_unknown_or_invalid_spot_quota_is_never_assumed_available(shares):
    with pytest.raises(ValueError, match='no compatible'):
        selector([binding(spot_status=shares)]).select_pool({**B, 'pool_selection': 'highest_spot'}, gpus=1)


@pytest.mark.parametrize('name', ['debug', 'DEBUG-cluster', 'compute-debug'])
def test_gpu_acp_documents_reject_debug_and_explicit_cpu_copy_is_allowed(name):
    with pytest.raises(ValueError, match='debug'):
        m.create_document({**B, 'aec2': name, 'quota_type': 'reserved'}, 'job', 'i', 'cmd')
    cpu = {**B, 'aec2': name, 'quota_type': 'reserved', 'gpus': 0, 'cpus': 2, 'memory_gb': 4, 'worker_spec': 'N6lS.Iu.I10.2c4g'}
    assert m.create_document(cpu, 'job', 'i', 'cmd', cpu_copy=True)['resource_pool']['name'] == name


@pytest.mark.parametrize('change', [{'gpus': 1}, {'gpus': False}, {'cpus': 4}, {'memory_gb': 8}, {'worker_spec': 'gpu'}])
def test_debug_cpu_exception_cannot_be_used_for_gpu_or_unverified_spec(change):
    cpu = {**B, 'aec2': 'debug', 'gpus': 0, 'cpus': 2, 'memory_gb': 4, 'worker_spec': 'N6lS.Iu.I10.2c4g'}
    with pytest.raises(ValueError, match='CPU data copy'):
        m.create_document({**cpu, **change}, 'job', 'i', 'cmd', cpu_copy=True)


def offline_client():
    c = log_client()
    c.describe.return_value = {**JOB, 'state': 'FAILED', 'create_time': '2026-01-01T00:00:00Z'}
    return c


def hit(text='Step 1 loss 1.5', **fields):
    return {'body': text, 'log_time': '2026-01-02T00:00:00Z', 'resource': {'resource_id': WS['id']},
            'attributes': {'k8s.pod.name': 'worker', 'k8s.container.name': 'worker-container'}, **fields}


def test_terminal_job_offline_query_is_read_only_exact_and_historical():
    c = offline_client()
    c.request = Mock(return_value={'hits': [hit('later', log_time='2026-01-03T00:00:00Z'), hit('earlier')], 'total': '3'})
    result = c.logs(B, 'job', 10)
    assert result['historical'] and result['source'] == 'offline' and result['truncated']
    assert result['text'] == 'earlier\nlater' and result['last_log_at'] == '2026-01-03T00:00:00Z'
    call = c.request.call_args
    assert '/product.lepton-acp-new/logs' in call.args[0]
    assert call.kwargs['read_only'] is True
    assert call.kwargs['body']['resource_id'] == [WS['id']]
    assert call.kwargs['body']['custom_filter'][0] == {'key': 'Attributes.k8s.pod.name', 'value': 'worker'}
    c.request.return_value = {'hits': [], 'total': 0}
    assert c.logs(B, 'job', 10)['unavailable_reason'] == 'OFFLINE_LOGS_EMPTY'
    c.request.side_effect = m.RESTError('logs', status=403, reason='Denied')
    result = c.logs(B, 'job', 10)
    assert not result['available'] and result['error']['http_status'] == 403
    assert result['unavailable_reason'] == 'OFFLINE_LOGS_UNAVAILABLE'


def test_active_stream_failure_falls_back_but_does_not_hide_identity_errors():
    c = offline_client()
    c.describe.return_value['state'] = 'RUNNING'
    live = {'available': False, 'error': {'http_status': 403}, 'text': '', 'expired': True}
    c.live_logs = Mock(return_value=live)
    c.request = Mock(return_value={'hits': [hit()], 'total': 1})
    result = c.logs(B, 'job', 10)
    assert result['historical'] and result['live_error'] == live['error']
    c.request.return_value = {'hits': [], 'total': 0}
    assert c.logs(B, 'job', 10) == live
    c.request.side_effect = m.RESTError('offline')
    assert c.logs(B, 'job', 10) == live
    c.request.side_effect = ValueError('foreign worker')
    assert c.logs(B, 'job', 10) == live


@pytest.mark.parametrize('tail', [True, 0, 10001, '1'])
def test_offline_log_bounds(tail):
    with pytest.raises(ValueError):
        offline_client().offline_logs(B, 'job', tail)


@pytest.mark.parametrize('create_time', [None, 'bad', '2026-01-01', '1960-01-01T00:00:00Z', '2100-01-01T00:00:00Z'])
def test_offline_log_interval_must_match_owned_job(create_time):
    c = offline_client()
    c.describe.return_value['create_time'] = create_time
    with pytest.raises(ValueError):
        c.offline_logs(B, 'job', 10)


def test_offline_worker_and_station_scope_empty_containers_and_no_worker():
    c = offline_client()
    c.resources.return_value = []
    with pytest.raises(ValueError, match='station'):
        c.logs(B, 'job', 10)
    c = offline_client()
    c.workers.return_value = []
    c.request = Mock()
    assert c.offline_logs(B, 'job', 10)['worker'] is None
    c.request.assert_not_called()
    c.workers.return_value = [{'name': 'worker', 'containers': []}]
    c.request.return_value = {'hits': [], 'total': '0'}
    assert c.offline_logs(B, 'job', 10)['worker'] == 'worker'
    for containers in ('wrong', [None], [{}]):
        c.workers.return_value[0]['containers'] = containers
        with pytest.raises(ValueError, match='container identity'):
            c.offline_logs(B, 'job', 10)


@pytest.mark.parametrize('data', [None, {}, {'hits': {}}, {'hits': [], 'total': True},
    {'hits': [], 'total': '-1'}, {'hits': [hit()], 'total': '0'}, {'hits': [hit()] * 11, 'total': 11}])
def test_invalid_offline_log_response(data):
    c = offline_client()
    c.request = Mock(return_value=data)
    with pytest.raises(ValueError):
        c.logs(B, 'job', 10)


@pytest.mark.parametrize('record', [None, {}, hit(body=1), hit(attributes={}), hit(resource={}),
    hit(resource=1), hit(log_time=1), hit(log_time='bad'), hit(log_time='2026-01-02'),
    hit(log_time='1960-01-01T00:00:00Z'), hit(log_time='2100-01-01T00:00:00Z')])
def test_offline_foreign_malformed_and_out_of_interval_hits_fail_closed(record):
    c = offline_client()
    c.request = Mock(return_value={'hits': [record], 'total': 1})
    with pytest.raises(ValueError):
        c.logs(B, 'job', 10)


@pytest.mark.parametrize('tail', [True, 0, 10001, '1'])
def test_live_log_bounds_precede_all_provider_requests(tail):
    c = log_client()
    with pytest.raises(ValueError):
        c.logs(B, 'job', tail)
    c.describe.assert_not_called()
