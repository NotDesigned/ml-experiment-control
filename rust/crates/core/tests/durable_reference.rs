use ml_exp_core::durable::{self, Snapshot};
use serde_json::Value;
#[test]
fn actual_python_durable_heads_and_events_are_unchanged() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../compatibility/durable.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let snapshot = Snapshot::from_value(&case["before"]).unwrap();
        let events = case["events"].as_array().unwrap();
        let plan = durable::commit(
            &snapshot,
            events,
            &case["payload"],
            &case["event"],
            case["expected_revision"].as_u64(),
            case["identity"].as_str().unwrap(),
        )
        .unwrap();
        assert_eq!(plan.head, case["head"]);
        assert_eq!(plan.transition, case["transition"]);
        assert_eq!(plan.revision, case["revision"].as_u64().unwrap());
        let next = Snapshot::from_value(&plan.head).unwrap();
        assert_eq!(
            durable::missing_transition(&next, events).unwrap(),
            Some(plan.transition.clone())
        );
        let mut complete = events.clone();
        complete.push(plan.transition);
        assert_eq!(durable::missing_transition(&next, &complete).unwrap(), None);
    }
}
