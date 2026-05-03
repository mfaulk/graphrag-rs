//! Concurrency regression test for issue #4 — embeddings path.
//!
//! `HttpEmbeddingProvider` (OpenAI / Voyage / Cohere / etc.) used the same
//! sync-`ureq`-inside-`async-fn` pattern that `OllamaClient` did, so concurrent
//! `embed` calls serialised on the tokio worker pool and contributed to the
//! `build_graph` deadlock under load. This test asserts that 8 concurrent
//! `embed` calls against a 500ms mock server complete in roughly one
//! round-trip when run on a 2-worker tokio runtime.

#![cfg(all(feature = "ureq", feature = "async"))]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use graphrag_core::embeddings::api_providers::HttpEmbeddingProvider;
use graphrag_core::embeddings::EmbeddingProvider;

struct MockServer {
    port: u16,
    shutdown: Arc<AtomicBool>,
}

impl MockServer {
    fn start(delay: Duration) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        listener
            .set_nonblocking(true)
            .expect("set_nonblocking on listener");

        thread::spawn(move || {
            while !shutdown_clone.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _addr)) => {
                        thread::spawn(move || handle_connection(stream, delay));
                    },
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    },
                    Err(_) => break,
                }
            }
        });

        MockServer { port, shutdown }
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

fn handle_connection(mut stream: std::net::TcpStream, delay: Duration) {
    stream
        .set_nonblocking(false)
        .expect("set blocking on accepted stream");

    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    let mut content_length: usize = 0;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        if line == "\r\n" {
            break;
        }
        if let Some(rest) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = rest.trim().parse().unwrap_or(0);
        }
    }
    if content_length > 0 {
        let mut body = vec![0u8; content_length];
        let _ = reader.read_exact(&mut body);
    }

    thread::sleep(delay);

    // OpenAI-compatible embeddings response with a 4-dim vector.
    let body = br#"{"data":[{"embedding":[0.1,0.2,0.3,0.4],"index":0}],"model":"test","usage":{}}"#;
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

/// 8 concurrent `embed` calls against a 500ms mock should complete in roughly
/// one round-trip (~600ms) on a 2-worker runtime once the sync ureq call is
/// dispatched to the blocking pool. Without the fix the calls serialise on
/// the worker pool and total time exceeds 2s.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_embed_does_not_block_runtime_workers() {
    let server = MockServer::start(Duration::from_millis(500));
    tokio::time::sleep(Duration::from_millis(50)).await;

    let endpoint = format!("http://127.0.0.1:{}/v1/embeddings", server.port);

    // Use the openai() constructor and swap the endpoint via the
    // `with_endpoint_for_tests` helper exposed on `HttpEmbeddingProvider`.
    let provider = HttpEmbeddingProvider::openai("sk-test".to_string(), "model".to_string())
        .with_endpoint_for_tests(endpoint);

    let n = 8usize;
    let provider = std::sync::Arc::new(provider);
    let start = Instant::now();
    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let p = provider.clone();
        handles.push(tokio::spawn(async move { p.embed("hello").await }));
    }
    for h in handles {
        h.await.expect("join").expect("embed ok");
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_millis(1500),
        "concurrent embed calls took {elapsed:?}; expected well under 1.5s, \
         which means sync ureq is parking tokio worker threads"
    );
}
