//! The HTTP boundary: authenticated, bounded, no redirects or automatic replay.
//! Mutations are available only through the typed project lifecycle planner.

use std::io::Read;
use std::time::Duration;

use crate::Error;
use crate::config::{api_url, normalize_url, validate_token};

const RESPONSE_LIMIT: u64 = 8 * 1024 * 1024;

// No Debug implementation: the bearer token must never be formatted.
pub struct Client {
    base: String,
    token: String,
    http: reqwest::blocking::Client,
}

impl Client {
    pub(crate) fn send_dataset(
        &self,
        plan: &crate::datasets::Plan,
    ) -> Result<serde_json::Value, Error> {
        let endpoint = api_url(&self.base, &plan.path)?;
        let response = self
            .http
            .request(plan.method.clone(), endpoint)
            .bearer_auth(&self.token)
            .header(
                ml_exp_core::CLIENT_PROTOCOL_HEADER,
                ml_exp_core::API_PROTOCOL_VERSION.to_string(),
            )
            .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
            .timeout(plan.timeout)
            .body(plan.body.clone())
            .send()
            .map_err(|_| Error::Transport)?;
        self.decode(response)
    }
    pub fn negotiate(&self) -> Result<serde_json::Value, Error> {
        let health = self.get("/api/health")?;
        crate::session::validate_health(&health)?;
        Ok(health)
    }

    pub fn base_url(&self) -> &str {
        &self.base
    }
    pub fn new(base: &str, token: String) -> Result<Self, Error> {
        let base = normalize_url(base)?;
        validate_token(&token)?;
        let http = reqwest::blocking::Client::builder()
            .no_proxy()
            .retry(reqwest::retry::never())
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|_| Error::Transport)?;
        Ok(Self { base, token, http })
    }

    pub fn get(&self, path: &str) -> Result<serde_json::Value, Error> {
        let endpoint = api_url(&self.base, path)?;
        let response = self
            .http
            .get(endpoint)
            .bearer_auth(&self.token)
            .header(
                ml_exp_core::CLIENT_PROTOCOL_HEADER,
                ml_exp_core::API_PROTOCOL_VERSION.to_string(),
            )
            .send()
            .map_err(|_| Error::Transport)?;
        self.decode(response)
    }

    pub(crate) fn send_project(
        &self,
        plan: &crate::projects::Plan,
    ) -> Result<serde_json::Value, Error> {
        let endpoint = api_url(&self.base, plan.path())?;
        let request = if let Some(body) = plan.body() {
            if body.len() > crate::projects::MAX_REQUEST_BYTES {
                return Err(Error::Response);
            }
            self.http
                .post(endpoint)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body.to_vec())
        } else {
            self.http.get(endpoint)
        };
        let response = request
            .bearer_auth(&self.token)
            .header(
                ml_exp_core::CLIENT_PROTOCOL_HEADER,
                ml_exp_core::API_PROTOCOL_VERSION.to_string(),
            )
            .send()
            .map_err(|_| Error::Transport)?;
        self.decode(response)
    }

    pub(crate) fn send_submission(
        &self,
        plan: &crate::submissions::Plan,
    ) -> Result<serde_json::Value, Error> {
        let endpoint = api_url(&self.base, plan.path())?;
        let request = match plan.body() {
            Some(body) => self
                .http
                .post(endpoint)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body.to_vec()),
            None => self.http.get(endpoint),
        };
        let response = request
            .bearer_auth(&self.token)
            .header(
                ml_exp_core::CLIENT_PROTOCOL_HEADER,
                ml_exp_core::API_PROTOCOL_VERSION.to_string(),
            )
            .timeout(Duration::from_secs(150))
            .send()
            .map_err(|_| Error::Transport)?;
        self.decode(response)
    }

    /// Consume one finite source archive; only a validated typed plan can pick
    /// the endpoint. The caller retains the local SHA for explicit lookup.
    pub(crate) fn upload_source(
        &self,
        plan: &crate::sources::ArchivePlan,
        file: std::fs::File,
    ) -> Result<serde_json::Value, Error> {
        let endpoint = api_url(&self.base, plan.path())?;
        let response = self
            .http
            .post(endpoint)
            .bearer_auth(&self.token)
            .header(
                ml_exp_core::CLIENT_PROTOCOL_HEADER,
                ml_exp_core::API_PROTOCOL_VERSION.to_string(),
            )
            .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
            .timeout(Duration::from_secs(600))
            .body(reqwest::blocking::Body::sized(file, plan.bytes()))
            .send()
            .map_err(|_| Error::Transport)?;
        self.decode(response)
    }

    pub(crate) fn send_artifact(
        &self,
        plan: &crate::artifacts::Plan,
    ) -> Result<serde_json::Value, Error> {
        let endpoint = api_url(&self.base, plan.path())?;
        let request = if let Some(body) = plan.body() {
            self.http
                .post(endpoint)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body.to_vec())
        } else {
            self.http.get(endpoint)
        };
        let response = request
            .bearer_auth(&self.token)
            .header(
                ml_exp_core::CLIENT_PROTOCOL_HEADER,
                ml_exp_core::API_PROTOCOL_VERSION.to_string(),
            )
            .timeout(Duration::from_secs(330))
            .send()
            .map_err(|_| Error::Transport)?;
        self.decode(response)
    }
    pub(crate) fn download_artifact(
        &self,
        plan: &crate::artifacts::DownloadPlan,
        file: &mut std::fs::File,
    ) -> Result<(), Error> {
        use std::io::Write;
        let endpoint = api_url(&self.base, plan.path())?;
        let mut response = self
            .http
            .get(endpoint)
            .bearer_auth(&self.token)
            .header(
                ml_exp_core::CLIENT_PROTOCOL_HEADER,
                ml_exp_core::API_PROTOCOL_VERSION.to_string(),
            )
            .timeout(Duration::from_secs(3600))
            .send()
            .map_err(|_| Error::Transport)?;
        if !response.status().is_success() {
            return match self.decode(response) {
                Err(e) => Err(e),
                Ok(_) => Err(Error::Response),
            };
        }
        if response.content_length() != Some(plan.bytes())
            || response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                != Some("application/x-tar")
        {
            return Err(Error::Response);
        }
        let mut bytes = 0u64;
        let mut buffer = [0u8; 65536];
        loop {
            let n = response.read(&mut buffer).map_err(|_| Error::Transport)?;
            if n == 0 {
                break;
            }
            bytes = bytes.checked_add(n as u64).ok_or(Error::Response)?;
            if bytes > plan.bytes() {
                return Err(Error::Response);
            }
            file.write_all(&buffer[..n]).map_err(|_| Error::Transport)?;
        }
        if bytes != plan.bytes() {
            return Err(Error::Response);
        }
        Ok(())
    }

    fn decode(&self, response: reqwest::blocking::Response) -> Result<serde_json::Value, Error> {
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let code = response
                .headers()
                .get("X-ML-Expd-Error-Code")
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned);
            let retry = response
                .headers()
                .get("Retry-After")
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned);
            let mut body = Vec::new();
            if response
                .take(ml_exp_core::diagnostic::ERROR_BODY_BYTES as u64 + 1)
                .read_to_end(&mut body)
                .is_err()
            {
                body.clear();
            }
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok()
                .and_then(|value| value.as_secs().try_into().ok())
                .unwrap_or(0);
            return Err(Error::Http(Box::new(
                ml_exp_core::diagnostic::HttpFailure::from_response(
                    status,
                    code.as_deref(),
                    retry.as_deref(),
                    &body,
                    &self.token,
                    now,
                ),
            )));
        }
        let mut bytes = Vec::new();
        response
            .take(RESPONSE_LIMIT + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| Error::Response)?;
        if bytes.len() as u64 > RESPONSE_LIMIT {
            return Err(Error::Response);
        }
        serde_json::from_slice(&bytes).map_err(|_| Error::Response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;

    fn read_request(socket: &mut std::net::TcpStream) -> String {
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        while !request.ends_with(b"\r\n\r\n") {
            let mut byte = [0];
            socket.read_exact(&mut byte).unwrap();
            request.extend_from_slice(&byte);
            assert!(request.len() < 8192);
        }
        String::from_utf8(request).unwrap().to_lowercase()
    }

    #[test]
    fn sends_exact_prefix_auth_and_protocol_once_and_refuses_redirect() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}/ml-expd", listener.local_addr().unwrap());
        let thread = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let request = read_request(&mut socket);
            assert!(request.starts_with("get /ml-expd/api/health http/1.1"));
            assert_eq!(
                request
                    .matches("authorization: bearer native-test\r\n")
                    .count(),
                1
            );
            assert_eq!(
                request.matches("x-ml-expd-client-protocol: 2\r\n").count(),
                1
            );
            socket.write_all(b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/leak\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
        });
        let client = Client::new(&base, "native-test".into()).unwrap();
        assert!(
            matches!(client.get("/api/health"), Err(Error::Http(value)) if value.status == 302)
        );
        assert!(matches!(client.get("/api/../../outside"), Err(Error::Path)));
        thread.join().unwrap();
    }

    #[test]
    fn successful_json_is_returned_and_invalid_json_is_suppressed() {
        for (body, valid) in [
            (br#"{"value":42}"#.as_slice(), true),
            (b"secret-error-body", false),
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let thread = std::thread::spawn(move || {
                let (mut socket, _) = listener.accept().unwrap();
                let request = read_request(&mut socket);
                assert!(request.starts_with("get /api/metrics?limit=1 http/1.1"));
                write!(
                    socket,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                socket.write_all(body).unwrap();
            });
            let result = Client::new(&base, "native-test".into())
                .unwrap()
                .get("/api/metrics?limit=1");
            if valid {
                assert_eq!(result.unwrap(), serde_json::json!({"value":42}));
            } else {
                let error = result.unwrap_err();
                assert!(matches!(error, Error::Response));
                assert!(!error.to_string().contains("secret-error-body"));
            }
            thread.join().unwrap();
        }
    }

    #[test]
    fn response_body_is_bounded_even_when_the_server_sends_valid_json() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let thread = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            read_request(&mut socket);
            let mut body = vec![b' '; RESPONSE_LIMIT as usize];
            body.extend_from_slice(b"{}");
            write!(
                socket,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            // The client may close after reading the bound plus one byte.
            let _ = socket.write_all(&body);
        });
        let client = Client::new(&base, "native-test".into()).unwrap();
        assert!(matches!(client.get("/api/health"), Err(Error::Response)));
        thread.join().unwrap();
    }
}
