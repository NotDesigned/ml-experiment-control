//! `ml-exp run purge-state`: remove one ended Attempt's `state/` on backend storage
//! now, under the conditions of the automatic purge. One POST and no automatic retry;
//! repeating it observes the same purge job, so a lost response is safe to repeat.
//! A blocked purge changes nothing and the answer says what blocks it.
use crate::{Client, Error, observation};
use serde_json::{Value, json};

pub fn purge(
    client: &Client,
    project: &str,
    run: &str,
    attempt: Option<&str>,
    confirmation: &str,
) -> Result<Value, Error> {
    let project_segment = observation::segment(project)?;
    let run_segment = observation::segment(run)?;
    let attempt = match attempt {
        Some(attempt) => attempt.to_owned(),
        None => {
            client.get(&format!("/api/runs/{project_segment}/{run_segment}"))?["current_attempt_id"]
                .as_str()
                .ok_or(Error::Observation("this Run has no current Attempt"))?
                .to_owned()
        }
    };
    let expected = format!("PURGE {project}/{run}/{attempt}");
    let unfinished = format!("PURGE UNFINISHED {project}/{run}/{attempt}");
    if confirmation != expected && confirmation != unfinished {
        return Err(Error::Submit(format!(
            "confirm with --confirmation '{expected}'; nothing was sent. For an Attempt whose \
             worker never reported its end, such as after a preemption, '{unfinished}' also \
             accepts that a generation it never registered is removed"
        )));
    }
    client.post(
        &format!(
            "/api/runs/{project_segment}/{run_segment}/attempts/{}/purge-state",
            observation::segment(&attempt)?
        ),
        &json!({"confirmation": confirmation}),
    )
}
