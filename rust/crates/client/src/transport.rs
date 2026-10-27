//! The HTTP boundary: authenticated, bounded, read-only and without redirects.

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
        if !response.status().is_success() {
            return Err(Error::Http(response.status().as_u16()));
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
        assert!(matches!(client.get("/api/health"), Err(Error::Http(302))));
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
