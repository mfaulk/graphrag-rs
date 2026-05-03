//! OpenAI chat-completions backend.
//!
//! Mirrors the sync-`ureq`-on-blocking-pool pattern used by
//! `embeddings::api_providers::HttpEmbeddingProvider` so concurrent calls
//! don't park tokio worker threads (issue #4).

use async_trait::async_trait;

use crate::core::backend::{ChatBackend, ChatParams};
use crate::core::error::{GraphRAGError, Result};

const DEFAULT_OPENAI_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";
/// Default model — cheap, capable, suitable for GraphRAG answer synthesis.
pub const DEFAULT_OPENAI_MODEL: &str = "gpt-4o-mini";

/// HTTP `ChatBackend` for OpenAI's `/v1/chat/completions` endpoint.
pub struct OpenAiChat {
    api_key: String,
    model: String,
    endpoint: String,
    client: ureq::Agent,
}

impl OpenAiChat {
    /// Create a backend with an explicit API key and model. Use
    /// [`OpenAiChat::from_env`] to read `OPENAI_API_KEY` from the
    /// environment instead.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            endpoint: DEFAULT_OPENAI_ENDPOINT.to_string(),
            client: ureq::Agent::new(),
        }
    }

    /// Construct from `OPENAI_API_KEY` env var. Returns `Auth` error if
    /// unset or empty.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let key = std::env::var("OPENAI_API_KEY").map_err(|_| GraphRAGError::Auth {
            message: "OPENAI_API_KEY environment variable not set".to_string(),
        })?;
        if key.is_empty() {
            return Err(GraphRAGError::Auth {
                message: "OPENAI_API_KEY is empty".to_string(),
            });
        }
        Ok(Self::new(key, model))
    }

    /// Construct using the default model (`gpt-4o-mini`).
    pub fn from_env_default() -> Result<Self> {
        Self::from_env(DEFAULT_OPENAI_MODEL)
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
            client: self.client.clone(),
        }
    }
}

#[derive(Clone)]
struct RequestCtx {
    api_key: String,
    model: String,
    endpoint: String,
    client: ureq::Agent,
}

impl RequestCtx {
    fn complete(self, prompt: &str, params: &ChatParams) -> Result<String> {
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        });
        if let Some(max) = params.max_tokens {
            body["max_tokens"] = serde_json::json!(max);
        }
        if let Some(temp) = params.temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        let response = self
            .client
            .post(&self.endpoint)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(body);

        let response = match response {
            Ok(r) => r,
            Err(ureq::Error::Status(code, r)) => {
                let body = r
                    .into_string()
                    .unwrap_or_else(|_| "<unreadable error body>".to_string());
                let detail = extract_openai_error_message(&body).unwrap_or_else(|| body.clone());
                return Err(GraphRAGError::Generation {
                    message: format!("OpenAI HTTP {}: {}", code, detail),
                });
            },
            Err(e) => {
                return Err(GraphRAGError::Generation {
                    message: format!("OpenAI request failed: {}", e),
                })
            },
        };

        let json: serde_json::Value =
            response
                .into_json()
                .map_err(|e| GraphRAGError::Generation {
                    message: format!("OpenAI response parse failed: {}", e),
                })?;

        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| GraphRAGError::Generation {
                message: "OpenAI response missing choices[0].message.content".to_string(),
            })?
            .to_string();

        Ok(content)
    }
}

fn extract_openai_error_message(body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    v["error"]["message"].as_str().map(|s| s.to_string())
}

#[async_trait]
impl ChatBackend for OpenAiChat {
    async fn complete(&self, prompt: &str, params: &ChatParams) -> Result<String> {
        let ctx = self.request_ctx();
        let prompt = prompt.to_string();
        let params = params.clone();
        tokio::task::spawn_blocking(move || ctx.complete(&prompt, &params))
            .await
            .map_err(|e| GraphRAGError::Generation {
                message: format!("OpenAI worker task panicked or was cancelled: {}", e),
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

    /// Records what the mock server received so tests can assert on it.
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

        // Read request line.
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
            return;
        }
        let parts: Vec<&str> = request_line.split_whitespace().collect();
        let method = parts.first().copied().unwrap_or("").to_string();
        let path = parts.get(1).copied().unwrap_or("").to_string();

        // Read headers.
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

    /// `OpenAiChat::complete` posts JSON to `/v1/chat/completions` with the
    /// `Authorization: Bearer …` header and a single user message.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openai_chat_posts_to_chat_completions_endpoint() {
        let server = MockServer::start(200, r#"{"choices":[{"message":{"content":"hi"}}]}"#);
        tokio::time::sleep(Duration::from_millis(30)).await;

        let backend = OpenAiChat::new("sk-test-key", "gpt-4o-mini")
            .with_endpoint_for_tests(server.endpoint("/v1/chat/completions"));

        let _ = backend
            .complete("hello world", &ChatParams::default())
            .await
            .expect("ok");

        let req = server.captured().expect("request was captured");
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/v1/chat/completions");

        let auth_header = req
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
            .expect("authorization header present");
        assert_eq!(auth_header.1, "Bearer sk-test-key");

        let body: serde_json::Value = serde_json::from_str(&req.body).expect("body is JSON");
        assert_eq!(body["model"], "gpt-4o-mini");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hello world");
    }

    /// `OpenAiChat::complete` returns `choices[0].message.content` from the
    /// upstream response.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openai_chat_extracts_response_text_from_choices_array() {
        let server = MockServer::start(
            200,
            r#"{"choices":[{"message":{"role":"assistant","content":"the answer is 42"}}]}"#,
        );
        tokio::time::sleep(Duration::from_millis(30)).await;

        let backend = OpenAiChat::new("sk-x", "gpt-4o-mini")
            .with_endpoint_for_tests(server.endpoint("/v1/chat/completions"));

        let out = backend
            .complete("what is the answer?", &ChatParams::default())
            .await
            .expect("ok");
        assert_eq!(out, "the answer is 42");
    }

    /// `OpenAiChat::complete` surfaces 4xx errors as `Generation` with the
    /// upstream `error.message` extracted.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openai_chat_returns_error_on_4xx_with_message() {
        let server = MockServer::start(
            400,
            r#"{"error":{"message":"invalid model","type":"invalid_request_error"}}"#,
        );
        tokio::time::sleep(Duration::from_millis(30)).await;

        let backend = OpenAiChat::new("sk-x", "bogus-model")
            .with_endpoint_for_tests(server.endpoint("/v1/chat/completions"));

        let err = backend
            .complete("hi", &ChatParams::default())
            .await
            .expect_err("4xx should error");
        let s = format!("{}", err);
        assert!(
            s.contains("400") && s.contains("invalid model"),
            "unexpected error: {s}"
        );
    }

    /// `OpenAiChat::from_env` reads `OPENAI_API_KEY` and errors when unset.
    #[test]
    fn openai_chat_from_env_reads_api_key() {
        // Use a unique env var setup to avoid clobbering other tests.
        // Save and restore.
        let prev = std::env::var("OPENAI_API_KEY").ok();
        std::env::set_var("OPENAI_API_KEY", "sk-from-env");
        let backend = OpenAiChat::from_env_default().expect("env-based ctor ok");
        assert_eq!(backend.api_key, "sk-from-env");
        assert_eq!(backend.model, DEFAULT_OPENAI_MODEL);
        // Restore.
        match prev {
            Some(v) => std::env::set_var("OPENAI_API_KEY", v),
            None => std::env::remove_var("OPENAI_API_KEY"),
        }
    }
}
