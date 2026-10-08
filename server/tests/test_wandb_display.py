"""Readable comparisons without conflating scientific context or provenance."""
import json
from types import SimpleNamespace

import pytest

from ml_exp_server.tracking.wandb_display import Projection, identity, curve_record, configure, summaries
from ml_exp_server.tracking.tracking_store import TrackingStore


def metric(**changes):
    return {"name": "validation_loss", "unit": "nats/token", "protocol_id": "p1",
            "dataset_id": "val", "variant_id": "model-a", "checkpoint_id": "cp1",
            "step": 1, "value": 3.8, "status": "VALID", **changes}


def setup_store(tmp_path, run="a", target="study"):
    store = TrackingStore(tmp_path)
    store.bind("local", run, {"enabled": True, "entity": "team", "project": target}, {"ml_expd": {"run_id": run}})
    return store, store.scope("local", run, "attempt-001")


def append(store, scope, *items):
    return store.append(scope, [(str(i), {"kind": "metrics", "observations": [item]}) for i, item in enumerate(items)])


def handle():
    result = SimpleNamespace(summary={},definitions=[])
    result.define_metric = lambda name, **kw: result.definitions.append((name,kw))
    return result


def test_steps_and_checkpoint_changes_are_one_curve_but_separate_provenance(tmp_path):
    s,a = setup_store(tmp_path)
    append(s,a,metric(),metric(step=2,checkpoint_id="cp2",value=3.6))
    display=s.display(a,2); row=next(iter(display["series"].values()))
    assert row["curve"]=="curves/validation_loss" and row["latest"]["checkpoint_id"]=="cp2"
    out=curve_record(s.events(a)[1],display)
    assert out["curves/validation_loss"]==3.6 and out["axes/validation_loss/step"]==2
    h=handle();configure(h,display)
    visible=dict(h.definitions)["curves/validation_loss"]
    assert visible=={"step_metric":"axes/validation_loss/step","step_sync":False,"hidden":False,"summary":"none","overwrite":True}
    summaries(h,{},display,terminal=True,coverage="full")
    assert h.summary["results/validation_loss"]==3.6
    source=json.loads(h.summary["ml_expd/display_definitions_json"])["results/validation_loss"]
    assert source["checkpoint_id"]=="cp2" and source["variant_id"]=="model-a" and source["step"]==2
    t,b=setup_store(tmp_path,run="b");append(t,b,metric(variant_id="model-b",checkpoint_id="other"))
    assert next(iter(t.display(b,1)["series"].values()))["result"]=="results/validation_loss"


@pytest.mark.parametrize("change",[{"unit":"bits/token"},{"protocol_id":"p2"},{"dataset_id":"test"},{"name":"validation/loss"}])
def test_different_scientific_meanings_never_overwrite(tmp_path,change):
    s,a=setup_store(tmp_path);append(s,a,metric());first=next(iter(s.display(a,1)["series"].values()))["result"]
    t,b=setup_store(tmp_path,run="b");append(t,b,metric(**change));other=next(iter(t.display(b,1)["series"].values()))["result"]
    assert other!=first
    # Registry and event history survive restart; destination namespaces are independent.
    assert t.display(b,1)==TrackingStore(tmp_path).display(b,1)
    u,c=setup_store(tmp_path,run="c",target="another");append(u,c,metric(**change))
    assert next(iter(u.display(c,1)["series"].values()))["name"].startswith(change.get("name","validation_loss").replace("/","_"))


def test_sanitized_names_cannot_collide(tmp_path):
    s,a=setup_store(tmp_path);append(s,a,metric(name="a/b"),metric(name="a_b"))
    assert len({x["result"] for x in s.display(a,2)["series"].values()})==2


def test_multiple_variants_promote_separate_curves_and_remove_shared_scalar(tmp_path):
    s,a=setup_store(tmp_path);append(s,a,metric(),metric(step=2),metric(variant_id="mask-b",step=1),metric(variant_id="mask-b",step=2))
    d=s.display(a,4);h=handle();h.summary["results/validation_loss"]=99
    configure(h,d);summaries(h,dict(h.summary),d,terminal=True,coverage="full")
    assert "results/validation_loss" not in h.summary
    assert len([k for k in h.summary if k.startswith("results/")])==2
    assert dict(h.definitions)["curves/validation_loss"]["hidden"]
    for event in s.events(a):
        assert "curves/validation_loss" not in curve_record(event,d)
    assert len([key for key,opts in h.definitions if key.startswith("curves/") and not opts["hidden"]])==2


@pytest.mark.parametrize("items,error",[
    ([metric(),metric(value=7)],"CONFLICTING_LATEST_OBSERVATIONS"),
    ([metric(step=None),metric(step=None,checkpoint_id="other")],"UNORDERED_CHECKPOINTS"),
    ([metric(),metric(checkpoint_id="other")],"CONFLICTING_LATEST_OBSERVATIONS"),
])
def test_conflicting_latest_results_are_not_chosen(tmp_path,items,error):
    s,a=setup_store(tmp_path);append(s,a,*items);d=s.display(a,2);h=handle();summaries(h,{},d,terminal=True,coverage="full")
    assert not any(k.startswith("results/") for k in h.summary)
    assert next(iter(json.loads(h.summary["ml_expd/display_definitions_json"]).values()))["error"]==error


def test_out_of_order_duplicate_missing_failed_and_partial_observations(tmp_path):
    s,a=setup_store(tmp_path)
    append(s,a,metric(step=2,value=3),metric(step=1,value=4),metric(step=2,value=3),metric(name=None),metric(unit=None))
    d=s.display(a,5);assert next(iter(d["series"].values()))["latest"]["value"]==3
    assert not curve_record({"payload":{"kind":"lifecycle"}},d)
    for state in ["FAILED","PARTIAL","MISSING"]:
        projection=Projection();projection.add(metric(value=None,status=state),1);projection.cursor=1
        doc=projection.document({identity(metric()):"validation_loss"});h=handle()
        summaries(h,{"metrics/old/step":6112},doc,terminal=True,coverage="legacy")
        assert not any(k.startswith("results/") for k in h.summary)
        assert not curve_record({"payload":{"kind":"metrics","observations":[metric(value=None,status=state)]}},doc)
    h=handle();h.summary={"metrics/old/step":6112,"results/validation_loss":3}
    summaries(h,dict(h.summary),d,terminal=False,coverage="live")
    assert h.summary.get("results/validation_loss") is None and "metrics/old/step" not in h.summary


def test_epoch_axis_single_points_and_no_axis_are_not_fabricated(tmp_path):
    s,a=setup_store(tmp_path);append(s,a,metric(step=None,epoch=1),metric(step=None,epoch=2))
    d=s.display(a,2);out=curve_record(s.events(a)[0],d)
    assert out["axes/validation_loss/by_epoch/epoch"]==1 and out["curves/validation_loss/by_epoch"]==3.8
    s.display(a,1) # Earlier cutoff reconstructs rather than carrying future observations.
    h=handle();configure(h,s.display(a,1));assert dict(h.definitions)["curves/validation_loss/by_epoch"]["hidden"]
    projection=Projection();projection.add(metric(step=None),1)
    doc=projection.document({identity(metric()):"validation_loss"})
    assert not curve_record({"payload":{"kind":"metrics","observations":[metric(step=None)]}},doc)
    assert not curve_record({"payload":{"kind":"metrics","observations":[metric(name="not-known")]}},doc)


def test_projection_is_bounded_and_disposable(tmp_path):
    p=Projection()
    for i in range(512):p.add(metric(variant_id=str(i)),i)
    with pytest.raises(ValueError,match="DISPLAY_CONTEXT_LIMIT"):p.add(metric(variant_id="extra"),513)
    s,_=setup_store(tmp_path)
    for i in range(17):
        _,scope=setup_store(tmp_path,run=str(i));s.display(scope,0)
    assert len(s._display_cache)==16
    assert s.event_id("absent",0) is None


def test_schema_upgrade_keeps_original_records_and_creates_no_new_scope(tmp_path):
    s,a=setup_store(tmp_path);append(s,a,metric());original=s.events(a)
    with s.connection() as db:db.execute("ALTER TABLE scopes DROP COLUMN display_version")
    new=TrackingStore(tmp_path);assert new.events(a)==original and len(new.pending())==1
    new.outcome(a,confirmed=1);assert new.pending() # Existing opt-in Run requires metadata acknowledgement.
    new.outcome(a,display_version=2);assert new.pending()==[]


def test_changed_axis_preserves_explicit_latest_observation():
    p=Projection();p.add(metric(),1);p.add(metric(step=None,epoch=1),2)
    assert next(iter(p.series.values()))['latest']['axis']=='epoch'


def test_no_axis_duplicate_with_same_checkpoint_keeps_value():
    p=Projection();p.add(metric(step=None),1);p.add(metric(step=None),2)
    assert next(iter(p.series.values()))['latest']['value']==3.8


def test_no_axis_conflicting_values_are_not_silently_overwritten():
    p=Projection();p.add(metric(step=None),1);p.add(metric(step=None,value=7),2)
    assert next(iter(p.series.values()))['latest']['error']=='CONFLICTING_LATEST_OBSERVATIONS'


def test_new_batch_stays_pending_until_its_own_display_ack(tmp_path):
    s,a=setup_store(tmp_path);append(s,a,metric())
    s.outcome(a,confirmed=1,display_version=2);assert s.pending()==[]
    s.append(a,[('second',{'kind':'metrics','observations':[metric(step=2)]})])
    s.outcome(a,confirmed=2,error='REMOTE_DISPLAY_ACK_PENDING')
    assert s.pending()[0]['display_confirmed']==1
    s.outcome(a,error='PUBLISHER_UNAVAILABLE');assert s.pending()
    s.outcome(a,confirmed=2,display_version=2);assert s.pending()==[]


def test_latest_without_axis_is_not_given_a_fabricated_axis():
    p=Projection();p.add(metric(),1);p.add(metric(step=None),2)
    assert next(iter(p.series.values()))['latest']['axis'] is None


def test_duplicate_projection_point_does_not_become_a_conflict():
    p=Projection();p.add(metric(),1);p.add(metric(),2)
    assert next(iter(p.series.values()))['latest']['status']=='VALID'
