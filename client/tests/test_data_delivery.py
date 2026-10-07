import json

import pytest

from ml_exp_client.api import ClientError
from ml_exp_client.data_delivery import deliver_data
from ml_exp_client.workflow import experiment
from test_experiment_workflow import API, HEALTH, configuration


class DeliveryAPI:
    def __init__(self,status='PREPARED',lost=False):
        self.status,self.lost,self.executes=status,lost,0
    def call(self,path,**kw):
        if path.endswith('/execute'):
            self.executes+=1
            if self.lost:raise ClientError('lost response')
            self.status='READY'
        return {'delivery_id':'delivery.'+'b'*64,'confirmation':'copy exact','status':self.status}


def test_client_prepares_data_once_and_resumes_without_acp_replay(tmp_path):
    state=tmp_path/'data.json';api=DeliveryAPI()
    assert deliver_data(api,'demo','asset.'+'a'*64,'cloud',state,20)['status']=='READY'
    assert deliver_data(api,'demo','asset.'+'a'*64,'cloud',state,20)['status']=='READY'
    assert api.executes==1
    with pytest.raises(ClientError):deliver_data(api,'other','asset.'+'a'*64,'cloud',state,20)


def test_uncertain_data_submission_is_saved_and_never_replayed(tmp_path):
    api=DeliveryAPI(lost=True);state=tmp_path/'data.json'
    with pytest.raises(ClientError,match='lost'):deliver_data(api,'demo','asset.'+'a'*64,'cloud',state,20)
    with pytest.raises(ClientError,match='uncertain'):deliver_data(api,'demo','asset.'+'a'*64,'cloud',state,20)
    assert api.executes==1


@pytest.mark.parametrize('status',['FAILED','RECONCILE_REQUIRED','QUEUED'])
def test_client_wait_failure_keeps_exact_data_identity(tmp_path,status):
    api=DeliveryAPI(status=status);state=tmp_path/'data.json'
    with pytest.raises(ClientError):deliver_data(api,'demo','asset.'+'a'*64,'cloud',state,0)
    assert json.loads(state.read_text())['delivery']['status']==status and api.executes==0


def test_experiment_sends_data_bindings_to_server_preparation(configuration):
    value=json.loads(configuration.read_text());value['inputs']=[{'asset_id':'asset.'+'a'*64,'mount_path':'/inputs/data'}]
    configuration.write_text(json.dumps(value))
    api=API()
    experiment(api,HEALTH,configuration,configuration.with_name('state.json'))
    sent=next(kw['data'] for path,kw in api.calls if path.endswith('/experiment-preparations'))
    assert sent['run']['inputs']==value['inputs']
    assert sent['requirements']['data_assets'] is True
    assert not any('/deliveries/' in path or path.endswith('/runs') for path,_ in api.calls)
