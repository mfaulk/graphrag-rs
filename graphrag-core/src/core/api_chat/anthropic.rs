//! Anthropic Messages API chat backend.
//!
//! Uses the `/v1/messages` endpoint with `x-api-key` + `anthropic-version`
//! headers. Response `content` is an array of typed blocks (`text`,
//! `tool_use`, etc.); we concatenate `text` blocks and ignore others —
//! tool-use is out of scope for issue #90.

use async_trait::async_trait;

use crate::core::backend::{ChatBackend, ChatParams};
use crate::core::error::{GraphRAGError, Result};

const DEFAULT_ANTHROPIC_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Default model — cheap Claude 4.5 Haiku, suitable for GraphRAG synthesis.
pub const DEFAULT_ANTHROPIC_MODEL: &str = "claude-haiku-4-5-20251001";
/// Default `max_tokens` when the caller doesn't pass one in `ChatParams`.
/// Anthropic's API requires this field.
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

/// HTTP `ChatBackend` for Anthropic's Messages API.
pub struct AnthropicChat {
    api_key: String,
    model: String,
    endpoint: String,
    /// Fallback `max_tokens` used when `ChatParams::max_tokens` is `None`.
    /// Anthropic requires the field on every request.
    default_max_tokens: u32,
    client: ureq::Agent,
}

impl AnthropicChat {
    /// Construct from explicit API key + model. Sends
    /// `x-api-key` and `anthropic-version: 2023-06-01` headers per the
    /// public Messages API spec.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            endpoint: DEFAULT_ANTHROPIC_ENDPOINT.to_string(),
            default_max_tokens: DEFAULT_MAX_TOKENS,
            client: ureq::Agent::new(),
        }
    }

    /// Construct from the `ANTHROPIC_API_KEY` env var.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| GraphRAGError::Auth {
            message: "ANTHROPIC_API_KEY environment variable not set".to_string(),
        })?;
        if key.is_empty() {
            return Err(GraphRAGError::Auth {
                message: "ANTHROPIC_API_KEY is empty".to_string(),
            });
        }
        Ok(Self::new(key, model))
    }

    /// Construct from env using the default model (`claude-haiku-4-5-20251001`).
    pub fn from_env_default() -> Result<Self> {
        Self::from_env(DEFAULT_ANTHROPIC_MODEL)
    }

    /// Override the fallback `max_tokens` used when `ChatParams` doesn't
    /// specify one.
    pub fn with_default_max_tokens(mut self, n: u32) -> Self {
        self.default_max_tokens = n;
        self
    }

    /// Override the HTTP endpoint. Intended for redirecting requests to a
    /// local mock server in integration tests.
    #[doc(hidden)]
    pub fn with_endpoint_for_tests(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    fn request_ctx(&self) -> RequestCtx {
        RequestCtx {
            api_key: self.api_key.clone(),
            model: self.model.clone(),
            endpoint: self.endpoint.clone(),
            default_max_tokens: self.default_max_tokens,
            client: self.client.clone(),
        }
    }
}

#[derive(Clone)]
struct RequestCtx {
    api_key: String,
    model: String,
    endpoint: String,
    default_max_tokens: u32,
    client: ureq::Agent,
}

impl RequestCtx {
    fn complete(self, prompt: &str, params: &ChatParams) -> Result<String> {
        let max_tokens = params.max_tokens.unwrap_or(self.default_max_tokens);
        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        });
        if let Some(t) = params.temperature {
            body["temperature"] = serde_json::json!(t);
        }

        let response = self
            .client
            .post(&self.endpoint)
            .set("x-api-key", &self.api_key)
            .set("anthropic-version", ANTHROPIC_VERSION)
            .set("content-type", "application/json")
            .send_json(body);

        let response = match response {
            Ok(r) => r,
            Err(ureq::Error::Status(code, r)) => {
                let body = r
                    .into_string()
                    .unwrap_or_else(|_| "<unreadable error body>".to_string());
                let detail = extract_anthropic_error_message(&body).unwrap_or_else(|| body.clone());
                return Err(GraphRAGError::Generation {
                    message: format!("Anthropic HTTP {}: {}", code, detail),
                });
            },
            Err(e) => {
                return Err(GraphRAGError::Generation {
                    message: format!("Anthropic request failed: {}", e),
                })
            },
        };

        let json: serde_json::Value =
            response
                .into_json()
                .map_err(|e| GraphRAGError::Generation {
                    message: format!("Anthropic response parse failed: {}", e),
                })?;

        let blocks = json["content"]
            .as_array()
            .ok_or_else(|| GraphRAGError::Generation {
                message: "Anthropic response missing content array".to_string(),
            })?;

        // Concatenate text blocks; ignore tool_use and others (out of scope).
        let mut out = String::new();
        for block in blocks {
            if block["type"] == "text" {
                if let Some(t) = block["text"].as_str() {
                    out.push_str(t);
                }
            }
        }

        if out.is_empty() {
            return Err(GraphRAGError::Generation {
                message: "Anthropic response had no text blocks".to_string(),
            });
        }

        Ok(out)
    }
}

fn extract_anthropic_error_message(body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    v["error"]["message"].as_str().map(|s| s.to_string())
}

#[async_trait]
impl ChatBackend for AnthropicChat {
    async fn complete(&self, prompt: &str, params: &ChatParams) -> Result<String> {
        let ctx = self.request_ctx();
        let prompt = prompt.to_string();
        let params = params.clone();
        tokio::task::spawn_blocking(move || ctx.complete(&prompt, &params))
            .await
            .map_err(|e| GraphRAGError::Generation {
                message: format!("Anthropic worker task panicked or was cancelled: {}", e),
            })?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    #[derive(Default, Debug, Clone)]
    struct CapturedRequest {
        method: String,
        path: String,
        headers: Vec<(String, String)>,
        body: String,
    }

    struct MockServer {
        port: u16,
        shutdown: Arc<AtomicBool>,
        captured: Arc<Mutex<Option<CapturedRequest>>>,
    }

    impl MockServer {
        fn start(status: u16, response_body: &'static str) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            let port = listener.local_addr().unwrap().port();
            let shutdown = Arc::new(AtomicBool::new(false));
            let shutdown_clone = shutdown.clone();
            let captured: Arc<Mutex<Option<CapturedRequest>>> = Arc::new(Mutex::new(None));
            let captured_clone = captured.clone();

            listener.set_nonblocking(true).expect("nonblocking");

            thread::spawn(move || {
                while !shutdown_clone.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let cap = captured_clone.clone();
                            thread::spawn(move || handle_conn(stream, status, response_body, cap));
                        },
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(10));
                        },
                        Err(_) => break,
                    }
                }
            });

            MockServer {
                port,
                shutdown,
                captured,
            }
        }

        fn endpoint(&self, path: &str) -> String {
            format!("http://127.0.0.1:{}{}", self.port, path)
        }

        fn captured(&self) -> Option<CapturedRequest> {
            self.captured.lock().unwrap().clone()
        }
    }

    impl Drop for MockServer {
        fn drop(&mut self) {
            self.shutdown.store(true, Ordering::Relaxed);
        }
    }

    fn handle_conn(
        stream: std::net::TcpStream,
        status: u16,
        response_body: &str,
        captured: Arc<Mutex<Option<CapturedRequest>>>,
    ) {
        stream.set_nonblocking(false).ok();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut stream = stream;

        let mut request_line = String::new();
        if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
            return;
        }
        let parts: Vec<&str> = request_line.split_whitespace().collect();
        let method = parts.first().copied().unwrap_or("").to_string();
        let path = parts.get(1).copied().unwrap_or("").to_string();

        let mut headers = Vec::new();
        let mut content_length = 0usize;
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                return;
            }
            if line == "\r\n" {
                break;
            }
            if let Some((k, v)) = line.split_once(':') {
                let k = k.trim().to_string();
                let v = v.trim().to_string();
                if k.eq_ignore_ascii_case("content-length") {
                    content_length = v.parse().unwrap_or(0);
                }
                headers.push((k, v));
            }
        }

        let mut body = String::new();
        if content_length > 0 {
            let mut buf = vec![0u8; content_length];
            if reader.read_exact(&mut buf).is_ok() {
                body = String::from_utf8_lossy(&buf).into_owned();
            }
        }

        *captured.lock().unwrap() = Some(CapturedRequest {
            method,
            path,
            headers,
            body,
        });

        let status_text = match status {
            200 => "OK",
            400 => "Bad Request",
            401 => "Unauthorized",
            500 => "Internal Server Error",
            _ => "OK",
        };
        let header = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            status,
            status_text,
            response_body.len()
        );
        let _ = stream.write_all(header.as_bytes());
        let _ = stream.write_all(response_body.as_bytes());
        let _ = stream.flush();
    }

    /// `AnthropicChat::complete` posts to `/v1/messages` with `x-api-key`
    /// and `anthropic-version` headers and includes `max_tokens` in the
    /// body.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn anthropic_chat_posts_to_messages_endpoint() {
        let server = MockServer::start(200, r#"{"content":[{"type":"text","text":"ok"}]}"#);
        tokio::time::sleep(Duration::from_millis(30)).await;

        let backend = AnthropicChat::new("test-key", "claude-haiku-4-5-20251001")
            .with_endpoint_for_tests(server.endpoint("/v1/messages"));

        let _ = backend
            .complete("hello", &ChatParams::default())
            .await
            .expect("ok");

        let req = server.captured().expect("request captured");
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/v1/messages");

        let api_key_h = req
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("x-api-key"))
            .expect("x-api-key header present");
        assert_eq!(api_key_h.1, "test-key");

        let version_h = req
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("anthropic-version"))
            .expect("anthropic-version header present");
        assert_eq!(version_h.1, ANTHROPIC_VERSION);

        let body: serde_json::Value = serde_json::from_str(&req.body).expect("body json");
        assert_eq!(body["model"], "claude-haiku-4-5-20251001");
        assert!(body["max_tokens"].is_number(), "max_tokens must be present");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hello");
    }

    /// `AnthropicChat::complete` extracts the single `text` block from the
    /// `content` array.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn anthropic_chat_extracts_text_from_content_array() {
        let server = MockServer::start(
            200,
            r#"{"content":[{"type":"text","text":"the sky is blue"}],"role":"assistant","stop_reason":"end_turn"}"#,
        );
        tokio::time::sleep(Duration::from_millis(30)).await;

        let backend = AnthropicChat::new("k", DEFAULT_ANTHROPIC_MODEL)
            .with_endpoint_for_tests(server.endpoint("/v1/messages"));
        let out = backend
            .complete("why?", &ChatParams::default())
            .await
            .expect("ok");
        assert_eq!(out, "the sky is blue");
    }

    /// `AnthropicChat::complete` concatenates multiple `text` blocks and
    /// ignores non-text blocks (e.g. `tool_use`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn anthropic_chat_handles_multi_block_response() {
        let server = MockServer::start(
            200,
            r#"{"content":[
                {"type":"text","text":"part one. "},
                {"type":"tool_use","id":"abc","name":"calc","input":{}},
                {"type":"text","text":"part two."}
            ]}"#,
        );
        tokio::time::sleep(Duration::from_millis(30)).await;

        let backend = AnthropicChat::new("k", DEFAULT_ANTHROPIC_MODEL)
            .with_endpoint_for_tests(server.endpoint("/v1/messages"));
        let out = backend
            .complete("hi", &ChatParams::default())
            .await
            .expect("ok");
        assert_eq!(out, "part one. part two.");
    }

    /// `AnthropicChat::from_env` reads `ANTHROPIC_API_KEY`.
    #[test]
    fn anthropic_chat_from_env_reads_api_key() {
        let prev = std::env::var("ANTHROPIC_API_KEY").ok();
        std::env::set_var("ANTHROPIC_API_KEY", "sk-ant-from-env");
        let backend = AnthropicChat::from_env_default().expect("env-based ctor ok");
        assert_eq!(backend.api_key, "sk-ant-from-env");
        assert_eq!(backend.model, DEFAULT_ANTHROPIC_MODEL);
        match prev {
            Some(v) => std::env::set_var("ANTHROPIC_API_KEY", v),
            None => std::env::remove_var("ANTHROPIC_API_KEY"),
        }
    }
}
