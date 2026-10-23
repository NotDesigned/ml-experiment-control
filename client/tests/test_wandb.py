"""Standalone client routing and private-file configuration; no SDK dependency."""
import json

import pytest

from ml_exp_client.api import ClientError
from ml_exp_client.cli import main
from ml_exp_client.workflow import experiment
from test_experiment_workflow import configuration, API, HEALTH


@pytest.mark.parametrize("options", [[], {"enabled": "yes"}, {"project": "bad/name"}, {"api_key": "private"}, {"entity": 3}, {"preparation_enabled": "yes"}, {"preparation_enabled": None}])
def test_invalid_wandb_options_do_not_upload_source(configuration, options):
    value = json.loads(configuration.read_text()); value["wandb"] = options
    configuration.write_text(json.dumps(value))
    api = API()
    with pytest.raises(ClientError): experiment(api, HEALTH, configuration, configuration.with_name("state.json"))
    assert api.calls == []


@pytest.mark.parametrize("parameters", [[], {"x": float("nan")}, {"x": "a" * 17000}])
def test_invalid_parameters_are_rejected_locally(configuration, parameters):
    value = json.loads(configuration.read_text()); value["parameters"] = parameters
    configuration.write_text(json.dumps(value))
    with pytest.raises(ClientError): experiment(API(), HEALTH, configuration, configuration.with_name("state.json"))


def test_one_configuration_carries_target_and_research_parameters(configuration):
    value = json.loads(configuration.read_text()); value.update(wandb={"project": "fineweb"}, parameters={"wd": 0.1})
    configuration.write_text(json.dumps(value))
    api = API()
    with pytest.raises(ClientError, match="wandb-sync"): experiment(api, HEALTH, configuration, configuration.with_name("state.json"))
    experiment(api, {**HEALTH, "capabilities": [*HEALTH["capabilities"], "wandb-sync.v1"]}, configuration, configuration.with_name("state.json"))
    run = next(kwargs["data"]["run"] for endpoint, kwargs in api.calls if endpoint.endswith("experiment-preparations"))
    assert run["wandb"] == value["wandb"] and run["parameters"] == value["parameters"]


def test_wandb_commands_read_private_file_without_printing_key(tmp_path, monkeypatch, capsys):
    calls = []
    class FakeAPI:
        def __init__(self, *args): pass
        def negotiate(self): return {"capabilities": ["wandb-sync.v1"]}
        def call(self, path, **kwargs): calls.append((path, kwargs)); return {"credentials_configured": True}
    monkeypatch.setattr("ml_exp_client.cli.Client", FakeAPI)
    key = tmp_path / "private.key"; key.write_text("private-test-key\n")
    assert main(["wandb", "--key-file", str(key), "--project", "fineweb"]) == 0
    assert calls[-1][1]["data"]["api_key"] == "private-test-key" and calls[-1][1]["method"] == "PUT"
    assert "private-test-key" not in capsys.readouterr().out
    assert main(["wandb"]) == 0 and calls[-1] == ("/api/tracking/wandb", {})
    assert main(["wandb", "--disable"]) == 0
    assert calls[-1][1]["data"]["enabled"] is False
    assert main(["tracking", "--project", "demo", "--run", "trial"]) == 0
    assert main(["tracking", "--project", "demo", "--run", "trial", "--retry"]) == 0
    assert calls[-1][0].endswith("/tracking/retry")
    monkeypatch.setattr(FakeAPI, "negotiate", lambda self: {"capabilities": []})
    assert main(["wandb"]) == 2


def test_preparation_override_checks_capability_before_upload_and_is_forwarded(configuration):
    value=json.loads(configuration.read_text());value['wandb']={'enabled':True,'preparation_enabled':False}
    configuration.write_text(json.dumps(value));api=API()
    health={**HEALTH,'capabilities':[*HEALTH['capabilities'],'wandb-sync.v1']}
    with pytest.raises(ClientError,match='wandb-preparation-sync'):experiment(api,health,configuration,configuration.with_name('state.json'))
    assert api.calls==[]
    health['capabilities'].append('wandb-preparation-sync.v1')
    experiment(api,health,configuration,configuration.with_name('state.json'))
    body=next(kw['data']['run'] for path,kw in api.calls if path.endswith('experiment-preparations'))
    assert body['wandb']==value['wandb']


def test_preparation_default_cli_can_be_enabled_and_disabled(monkeypatch):
    calls=[]
    class API:
        def __init__(self,*a):pass
        def negotiate(self):return {'capabilities':['wandb-sync.v1']}
        def call(self,path,**kw):calls.append(kw);return {}
    monkeypatch.setattr('ml_exp_client.cli.Client',API)
    assert main(['wandb','--preparation'])==2 and calls==[]
    monkeypatch.setattr(API,'negotiate',lambda self:{'capabilities':['wandb-sync.v1','wandb-preparation-sync.v1']})
    assert main(['wandb','--preparation'])==0 and calls[-1]['data']=={'preparation_enabled':True}
    assert main(['wandb','--no-preparation'])==0 and calls[-1]['data']=={'preparation_enabled':False}

@pytest.mark.parametrize('resources',[{'allocation_policy':'spot'},{'priority':True},{'priority':0},{'priority':10},{'priority':'5'}])
def test_invalid_dlc_selection_fails_before_upload(configuration,resources):
    value=json.loads(configuration.read_text());value.setdefault('resources',{}).update(resources);configuration.write_text(json.dumps(value));api=API()
    with pytest.raises(ClientError):experiment(api,HEALTH,configuration,configuration.with_name('state.json'))
    assert api.calls==[]


def test_dlc_selection_checks_server_capability_and_is_frozen_in_request(configuration):
    value=json.loads(configuration.read_text());value.setdefault('resources',{}).update(allocation_policy='idle_only',priority=5);configuration.write_text(json.dumps(value));api=API()
    with pytest.raises(ClientError,match='pai-idle-resources'):experiment(api,HEALTH,configuration,configuration.with_name('state.json'))
    assert api.calls==[]
    health={**HEALTH,'capabilities':[*HEALTH['capabilities'],'pai-idle-resources.v1']}
    experiment(api,health,configuration,configuration.with_name('state.json'))
    body=next(kw['data']['run'] for path,kw in api.calls if path.endswith('experiment-preparations'))
    assert body['resources']['allocation_policy']=='idle_only' and body['resources']['priority']==5
