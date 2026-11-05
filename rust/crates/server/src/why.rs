//! Logs the cause behind an error value that a call site replaces with a fixed
//! response. `.map_err(why!(value))` returns exactly what `.map_err(|_| value)`
//! returned and adds one redacted stderr line, so control flow never changes.
//!
//! The line carries the thread, site and type name. Text passes through
//! `diagnostic::text`. Its patterns cannot recognise a value echoed by a JSON
//! parser, so serde_json text is withheld apart from schema messages that name a
//! field. `why!(value, "text")` logs fixed text only, for credential documents and
//! for source types without `Display`. `why_chain!(value)` also walks `source()`,
//! because `reqwest::Error` keeps the transport failure there and puts the full
//! request URL, signature included, in its own `Display`.
use ml_exp_core::diagnostic;
use std::{
    any::type_name,
    collections::BTreeMap,
    error::Error,
    fmt::{self, Display},
    io::Write as _,
    sync::{Mutex, PoisonError},
    time::{Duration, Instant},
};

const QUIET: Duration = Duration::from_secs(60);
const WITHHELD: &str = "[content withheld]";
const SCHEMA_MESSAGES: [&str; 2] = ["missing field", "unknown field"];
const MAX_SOURCES: usize = 8;

type Sites = BTreeMap<(&'static str, u32), (String, Instant)>;
/// One entry per site, so the map is bounded by the number of call sites.
static LAST: Mutex<Sites> = Mutex::new(BTreeMap::new());

macro_rules! why {
    ($value:expr, $text:literal $(,)?) => {
        |cause| {
            $crate::why::record_static(file!(), line!(), $text, &cause);
            $value
        }
    };
    ($value:expr $(,)?) => {
        |cause| {
            $crate::why::record(file!(), line!(), &cause);
            $value
        }
    };
}

macro_rules! why_chain {
    ($value:expr $(,)?) => {
        |cause| {
            $crate::why::record_chain(file!(), line!(), &cause);
            $value
        }
    };
}

pub(crate) fn record<E: Display>(file: &'static str, line: u32, cause: &E) {
    write_line(file, line, type_name::<E>(), &cause_text(cause));
}

pub(crate) fn record_static<E>(file: &'static str, line: u32, text: &'static str, _cause: &E) {
    write_line(file, line, type_name::<E>(), text);
}

pub(crate) fn record_chain<E: Error>(file: &'static str, line: u32, cause: &E) {
    write_line(file, line, type_name::<E>(), &chain_text(cause));
}

/// A failure the HTTP boundary reports without a discarded cause to attach.
pub(crate) fn record_status(file: &'static str, line: u32, status: u16, code: &str) {
    let text = format!("HTTP {status} {code}");
    write_line(file, line, "read_api::Error", &text);
}

fn cause_text<E: Display>(cause: &E) -> String {
    redacted(cause, type_name::<E>().contains("serde_json"))
}

fn chain_text<E: Error>(cause: &E) -> String {
    let mut text = cause_text(cause);
    let mut next = cause.source();
    for _ in 0..MAX_SOURCES {
        let Some(level) = next else { break };
        text.push_str(": ");
        text.push_str(&redacted(&level, level.is::<serde_json::Error>()));
        next = level.source();
    }
    text
}

fn redacted(cause: &impl Display, parser_echo: bool) -> String {
    let raw = displayed(cause);
    if parser_echo && !SCHEMA_MESSAGES.iter().any(|name| raw.starts_with(*name)) {
        return WITHHELD.to_owned();
    }
    diagnostic::text(&raw, "")
}

/// `to_string` panics when a `Display` implementation reports an error.
fn displayed(cause: &impl Display) -> String {
    let mut raw = String::new();
    let _ = fmt::write(&mut raw, format_args!("{cause}"));
    raw
}

fn write_line(file: &'static str, line: u32, type_name: &str, text: &str) {
    if repeated(file, line, text, Instant::now()) {
        return;
    }
    let thread = std::thread::current().id();
    let _ = writeln!(
        std::io::stderr().lock(),
        "why: thread={thread:?} site={file}:{line} type={type_name} cause={text}"
    );
}

fn repeated(file: &'static str, line: u32, text: &str, now: Instant) -> bool {
    let mut last = LAST.lock().unwrap_or_else(PoisonError::into_inner);
    let quiet = last.get(&(file, line)).is_some_and(|(previous, at)| {
        previous.as_str() == text && now.saturating_duration_since(*at) < QUIET
    });
    if !quiet {
        last.insert((file, line), (text.to_owned(), now));
    }
    quiet
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug)]
    struct Level {
        text: &'static str,
        source: Option<Box<dyn Error + 'static>>,
    }
    impl Display for Level {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.text)
        }
    }
    impl Error for Level {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            self.source.as_deref()
        }
    }

    #[test]
    fn serde_values_are_withheld_and_schema_messages_remain() {
        let echoed = serde_json::from_str::<u64>("\"hunter2\"").unwrap_err();
        assert!(echoed.to_string().contains("hunter2"));
        assert_eq!(cause_text(&echoed), WITHHELD);

        let imitation = serde_json::from_str::<u64>("\"missing field hunter2\"").unwrap_err();
        assert_eq!(cause_text(&imitation), WITHHELD);

        #[allow(dead_code)]
        #[derive(Debug, Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Document {
            required: u64,
        }
        let missing = serde_json::from_str::<Document>("{}").unwrap_err();
        assert!(cause_text(&missing).starts_with("missing field `required`"));
        let unknown = serde_json::from_str::<Document>(r#"{"other":1}"#).unwrap_err();
        assert!(cause_text(&unknown).starts_with("unknown field `other`"));
    }

    #[test]
    fn signed_urls_and_assignments_are_redacted() {
        let url = "failed for url (https://bucket.example/key?OSSAccessKeyId=id&Signature=abc123)";
        let text = cause_text(&url);
        assert!(text.starts_with("failed for url"));
        assert!(!text.contains("abc123"));
        assert!(!text.contains("bucket.example"));
        assert!(!cause_text(&"api_key=abc123").contains("abc123"));
    }

    #[test]
    fn every_level_of_a_chain_is_redacted() {
        let cause = Level {
            text: "sending https://bucket.example/key?Signature=abc123",
            source: Some(Box::new(Level {
                text: "connect via https://bucket.example/key?Signature=abc123",
                source: Some(Box::new(
                    serde_json::from_str::<u64>("\"hunter2\"").unwrap_err(),
                )),
            })),
        };
        let text = chain_text(&cause);
        assert!(text.starts_with("sending "));
        assert!(text.contains(": connect via "));
        assert!(text.ends_with(WITHHELD));
        assert!(!text.contains("abc123"));
        assert!(!text.contains("hunter2"));
    }

    /// Pins the behaviour of the locked reqwest: its Display includes the whole
    /// URL, and the transport reason is only reachable through `source()`.
    #[test]
    fn reqwest_failures_lose_the_signature_and_keep_the_transport_reason() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let cause = reqwest::blocking::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
            .get(format!("http://127.0.0.1:{port}/key?Signature=abc123"))
            .send()
            .unwrap_err();
        assert!(cause.to_string().contains("abc123"));
        assert!(cause.source().is_some());
        let text = chain_text(&cause);
        assert!(!text.contains("abc123"));
        assert!(text.len() > cause_text(&cause).len());
    }

    #[test]
    fn a_repeated_cause_is_quiet_until_the_window_ends_or_the_text_changes() {
        const SITE: &str = "why-repeated-test";
        let start = Instant::now();
        assert!(!repeated(SITE, 1, "first", start));
        assert!(repeated(SITE, 1, "first", start + QUIET / 2));
        assert!(!repeated(SITE, 2, "first", start + QUIET / 2));
        assert!(!repeated(SITE, 1, "second", start + QUIET / 2));
        assert!(!repeated(SITE, 1, "first", start + QUIET / 2));
        assert!(!repeated(SITE, 1, "first", start + QUIET * 2));
    }

    #[test]
    fn a_poisoned_table_still_answers() {
        let _ = std::thread::spawn(|| {
            let _held = LAST.lock().unwrap_or_else(PoisonError::into_inner);
            panic!("poison the table");
        })
        .join();
        assert!(!repeated("why-poison-test", 1, "first", Instant::now()));
        LAST.clear_poison();
    }

    #[test]
    fn the_macros_return_the_original_value() {
        let plain: Result<(), u8> = Err::<(), _>("cause").map_err(why!(1));
        assert_eq!(plain, Err(1));
        let fixed: Result<(), u8> = Err::<(), _>(()).map_err(why!(2, "no Display"));
        assert_eq!(fixed, Err(2));
        let chained: Result<(), u8> = Err::<(), _>(Level {
            text: "cause",
            source: None,
        })
        .map_err(why_chain!(3));
        assert_eq!(chained, Err(3));
    }
}
