"""Client checkpoint references are exact and checked before side effects."""
import json
import pytest

from ml_exp_client.api import ClientError
from ml_exp_client.cli import main
from ml_exp_client.workflow import read_config, experiment
from test_experiment_workflow import configuration, API, HEALTH

REFERENCE={"run_id":"first","attempt_id":"attempt-001","checkpoint_id":"checkpoint."+"a"*64}


def test_single_config_persistence_and_restore_are_forwarded(configuration):
    value=json.loads(configuration.read_text());value.update(checkpoint_persistence={},resume_from=REFERENCE)
    configuration.write_text(json.dumps(value));api=API()
    with pytest.raises(ClientError,match="persistent-checkpoints.v1"):experiment(api,HEALTH,configuration,configuration.with_name("state.json"))
    assert not api.calls
    experiment(api,{"capabilities":[*HEALTH["capabilities"],"persistent-checkpoints.v1"]},configuration,configuration.with_name("state.json"))
    run=next(args["data"] for path,args in api.calls if path.endswith("/runs"))
    assert run["resume_from"]==REFERENCE and run["checkpoint_persistence"]=={}


@pytest.mark.parametrize("key,value",[("checkpoint_persistence",[]),("checkpoint_persistence",{"interval_seconds":1}),
    ("checkpoint_persistence",{"other":1}),("resume_from",{}),("resume_from",{**REFERENCE,"attempt_id":"latest"}),
    ("resume_from",{**REFERENCE,"checkpoint_id":123}),("env",{"STATE_DIR":"/tmp"}),("env",{"RESUME_DIR":"/tmp"})])
def test_invalid_persistent_state_config_is_rejected_locally(configuration,key,value):
    config=json.loads(configuration.read_text());config[key]=value;configuration.write_text(json.dumps(config))
    with pytest.raises(ClientError):read_config(configuration)


@pytest.mark.parametrize("enabled",[True,False])
def test_cli_create_and_checkpoint_query(tmp_path,monkeypatch,enabled):
    calls=[]
    class Client:
        def __init__(self,*a,**k):pass
        def negotiate(self):return {"capabilities":["persistent-checkpoints.v1"] if enabled else []}
        def call(self,path,**kwargs):calls.append((path,kwargs));return {"checkpoints":[]}
    monkeypatch.setattr("ml_exp_client.cli.Client",Client)
    runtime=tmp_path/"runtime.json";runtime.write_text(json.dumps({"project":"demo","runtime_id":"runtime."+"b"*64}))
    assert main(["--url","https://api.example","create","--runtime-state",str(runtime),"--run","resumed","--executor","gpu",
        "--checkpoint-state-interval","5","--resume-from",json.dumps(REFERENCE)])==(0 if enabled else 2)
    assert main(["--url","https://api.example","checkpoints","--project","demo","--run","first","--attempt","attempt-001"])==(0 if enabled else 2)
    if enabled:
        assert calls[0][1]["data"]["checkpoint_persistence"]=={"interval_seconds":5}
        assert calls[0][1]["data"]["resume_from"]==REFERENCE
        assert calls[1][0].endswith("/attempts/attempt-001/checkpoints")
    else:assert not calls
