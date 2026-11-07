//! A launch that fails before its first worker report says why, once, to the server.
//! Some providers keep no log of such a job. The report is best effort and bounded:
//! it never changes the exit status, and nothing waits on it beyond its short budget.
use ml_exp_core::{cloud::exchange::Channel, launch::Identity};
use std::{io::Read, time::Duration};

/// The server refuses a larger body, so the message is clipped well below it.
const MAX_MESSAGE_CHARS: usize = 1000;
const TIMEOUT: Duration = Duration::from_secs(10);
const RETRY_AFTER: Duration = Duration::from_secs(if cfg!(test) { 0 } else { 1 });

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delivery {
    /// The server keeps this report.
    Recorded,
    /// The server received it but keeps none, because the Attempt already reported.
    Ignored,
    /// Not received; the HTTP status when there was one.
    Failed(Option<u16>),
}
impl std::fmt::Display for Delivery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Recorded => write!(f, "DELIVERED"),
            Self::Ignored => write!(f, "IGNORED (the Attempt already reported)"),
            Self::Failed(None) => write!(f, "NOT_DELIVERED"),
            Self::Failed(Some(status)) => write!(f, "NOT_DELIVERED http_status={status}"),
        }
    }
}

/// The report URL of the exact Attempt a package URL names, under the same checks as
/// its channel. `None` when the command line names no valid channel.
pub fn report_url(package_url: &str, token: &str) -> Option<String> {
    let base = package_url.strip_suffix("/package")?;
    let parts: Vec<_> = reqwest::Url::parse(base)
        .ok()?
        .path()
        .rsplit('/')
        .take(3)
        .map(str::to_owned)
        .collect();
    let [attempt_id, run_id, project] = <[String; 3]>::try_from(parts).ok()?;
    Channel {
        url: base.into(),
        token: token.into(),
    }
    .validate(&Identity {
        project,
        run_id,
        attempt_id,
    })
    .ok()?;
    Some(format!("{base}/startup-failure"))
}

/// One redacted line, clipped, without the token it is sent with.
fn message(text: &str, token: &str) -> String {
    let line: String = text
        .replace(token, "<redacted>")
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    experiment_redact::redact_line(line.trim())
        .chars()
        .take(MAX_MESSAGE_CHARS)
        .collect()
}

/// One POST. A redirect is never followed and no proxy is used, as for every packet.
pub fn send(client: &reqwest::blocking::Client, url: &str, token: &str, text: &str) -> Delivery {
    let body = serde_json::json!({ "message": message(text, token) });
    let Ok(response) = client
        .post(url)
        .bearer_auth(token)
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
    else {
        return Delivery::Failed(None);
    };
    let status = response.status().as_u16();
    let mut bytes = Vec::new();
    if status != 200 || response.take(1025).read_to_end(&mut bytes).is_err() || bytes.len() > 1024 {
        return Delivery::Failed(Some(status));
    }
    match serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .and_then(|v| v["recorded"].as_bool())
    {
        Some(true) => Delivery::Recorded,
        Some(false) => Delivery::Ignored,
        None => Delivery::Failed(Some(status)),
    }
}

/// At most one retry, and only when nothing was received or the server was busy.
pub fn deliver(mut send: impl FnMut() -> Delivery) -> Delivery {
    let first = send();
    if matches!(first, Delivery::Failed(None | Some(429 | 502 | 503 | 504))) {
        std::thread::sleep(RETRY_AFTER);
        return send();
    }
    first
}

/// Tell the server why this launch failed. `None` when there is no channel to tell.
pub fn report(package_url: &str, token: &str, text: &str) -> Option<Delivery> {
    let url = report_url(package_url, token)?;
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .connect_timeout(Duration::from_secs(5))
        .timeout(TIMEOUT)
        .build()
        .ok()?;
    Some(deliver(|| send(&client, &url, token, text)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Write, net::TcpListener};

    const TOKEN: &str = "b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0";

    fn listen(status: u16, body: &'static str) -> (String, std::thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!(
            "http://{}/api/worker-exchange/demo/train/attempt-001/startup-failure",
            listener.local_addr().unwrap()
        );
        let thread = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = Vec::new();
            loop {
                let mut chunk = [0u8; 4096];
                let read = socket.read(&mut chunk).unwrap();
                request.extend_from_slice(&chunk[..read]);
                let text = String::from_utf8_lossy(&request).into_owned();
                if let Some((head, rest)) = text.split_once("\r\n\r\n") {
                    let length: usize = head
                        .lines()
                        .filter_map(|l| l.split_once(':'))
                        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                        .unwrap()
                        .1
                        .trim()
                        .parse()
                        .unwrap();
                    if rest.len() >= length {
                        break;
                    }
                }
                assert!(read > 0);
            }
            write!(
                socket,
                "HTTP/1.1 {status} fixture\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            String::from_utf8(request).unwrap()
        });
        (url, thread)
    }

    #[test]
    fn only_the_exact_attempt_channel_of_the_command_line_is_used() {
        let package =
            "https://api.example/ml-expd/api/worker-exchange/demo/train/attempt-001/package";
        assert_eq!(
            report_url(package, TOKEN).as_deref(),
            Some(
                "https://api.example/ml-expd/api/worker-exchange/demo/train/attempt-001/startup-failure"
            )
        );
        for (url, token) in [
            (package, "short"),
            (
                "http://api.example/api/worker-exchange/demo/train/attempt-001/package",
                TOKEN,
            ),
            (
                "https://user@api.example/api/worker-exchange/demo/train/attempt-001/package",
                TOKEN,
            ),
            (
                "https://api.example/api/worker-exchange/demo/train/attempt-001/code",
                TOKEN,
            ),
            (
                "https://api.example/api/other/demo/train/attempt-001/package",
                TOKEN,
            ),
            (
                "https://api.example/api/worker-exchange/demo/train/package",
                TOKEN,
            ),
            ("not a url/package", TOKEN),
        ] {
            assert_eq!(report_url(url, token), None, "{url}");
        }
    }

    #[test]
    fn a_report_is_one_redacted_line_with_the_attempt_token_only_in_its_header() {
        let (url, server) = listen(200, r#"{"recorded":true}"#);
        let text = format!(
            "managed launch storage failed at crates/worker/src/cloud.rs:123: denied\npassword=hunter2 {TOKEN}"
        );
        let client = reqwest::blocking::Client::new();
        assert_eq!(send(&client, &url, TOKEN, &text), Delivery::Recorded);
        let request = server.join().unwrap();
        assert!(request.starts_with(
            "POST /api/worker-exchange/demo/train/attempt-001/startup-failure HTTP/1.1\r\n"
        ));
        assert!(
            request
                .to_ascii_lowercase()
                .contains(&format!("authorization: bearer {TOKEN}\r\n"))
        );
        let body: serde_json::Value =
            serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        let sent = body["message"].as_str().unwrap();
        assert!(sent.starts_with(
            "managed launch storage failed at crates/worker/src/cloud.rs:123: denied password="
        ));
        assert!(!sent.contains("hunter2") && !sent.contains(TOKEN), "{sent}");
        assert_eq!(body.as_object().unwrap().len(), 1);
        assert_eq!(
            message(&"x".repeat(5000), TOKEN).chars().count(),
            MAX_MESSAGE_CHARS
        );
    }

    #[test]
    fn what_the_server_said_is_reported_and_only_a_lost_or_busy_report_is_tried_again() {
        let (url, server) = listen(200, r#"{"recorded":false}"#);
        let client = reqwest::blocking::Client::new();
        assert_eq!(send(&client, &url, TOKEN, "x"), Delivery::Ignored);
        server.join().unwrap();
        let (url, server) = listen(401, "{}");
        assert_eq!(send(&client, &url, TOKEN, "x"), Delivery::Failed(Some(401)));
        server.join().unwrap();
        for (first, tries) in [
            (Delivery::Failed(None), 2),
            (Delivery::Failed(Some(503)), 2),
            (Delivery::Failed(Some(401)), 1),
            (Delivery::Failed(Some(413)), 1),
            (Delivery::Recorded, 1),
            (Delivery::Ignored, 1),
        ] {
            let mut calls = 0;
            let result = deliver(|| {
                calls += 1;
                if calls == 1 {
                    first
                } else {
                    Delivery::Recorded
                }
            });
            assert_eq!(calls, tries, "{first:?}");
            assert_eq!(
                result,
                if tries == 2 {
                    Delivery::Recorded
                } else {
                    first
                }
            );
        }
    }
}
