//! Concurrency regression test for issue #4.
//!
//! `OllamaClient` uses synchronous `ureq` calls inside `async fn` bodies. Without
//! `tokio::task::spawn_blocking`, every in-flight LLM request parks an entire
//! tokio worker thread for the duration of the HTTP round-trip, which serialises
//! concurrent calls and — combined with sequential per-chunk extraction — wedges
//! `build_graph` once the document count exceeds the worker pool size.
//!
//! These tests assert that N concurrent generate calls against a slow mock
//! server complete in roughly one round-trip, not N round-trips.

#![cfg(all(feature = "ollama", feature = "async", feature = "ureq"))]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use graphrag_core::ollama::{OllamaClient, OllamaConfig, OllamaGenerationParams};

/// A tiny single-threaded HTTP mock that delays every response by `delay_ms`.
///
/// Returns the bound port and a guard whose drop signals the listener thread
/// to exit on the next accept. The server uses one OS thread per connection so
/// it can serve many simultaneous requests in parallel — the contention we are
/// measuring is on the *client* side, not the server side.
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

        // Use non-blocking accept with a short poll so the thread can observe
        // the shutdown flag and exit promptly when the test finishes.
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

    // Parse just enough of the request to know the content-length, then drain it.
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

    // Simulate a slow LLM response.
    thread::sleep(delay);

    let body =
        br#"{"response":"ok","context":[],"prompt_eval_count":1,"eval_count":1,"done":true}"#;
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn make_client(port: u16) -> OllamaClient {
    let config = OllamaConfig {
        enabled: true,
        host: "http://127.0.0.1".to_string(),
        port,
        embedding_model: "nomic-embed-text".to_string(),
        chat_model: "test-model".to_string(),
        timeout_seconds: 10,
        max_retries: 1,
        fallback_to_hash: false,
        max_tokens: Some(8),
        temperature: Some(0.0),
        enable_caching: false,
        keep_alive: None,
        num_ctx: None,
    };
    OllamaClient::new(config)
}

/// Asserts that 8 concurrent generate_with_params calls against a 500ms mock
/// complete well under the time it would take to serialise them on a 2-worker
/// runtime. Without spawn_blocking around the sync ureq call this test
/// effectively serialises (>= ~2s); with spawn_blocking it parallelises on the
/// blocking pool (~600ms in practice).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_generate_does_not_block_runtime_workers() {
    let server = MockServer::start(Duration::from_millis(500));
    // Give the server a beat to start its accept loop.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = make_client(server.port);

    let n = 8usize;
    let start = Instant::now();
    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let c = client.clone();
        handles.push(tokio::spawn(async move {
            c.generate_with_params(
                "hi",
                OllamaGenerationParams {
                    num_predict: Some(1),
                    ..Default::default()
                },
            )
            .await
        }));
    }
    for h in handles {
        h.await.expect("join").expect("generate ok");
    }
    let elapsed = start.elapsed();

    // Hard upper bound: a fully serialised run on 2 workers would be ~2s
    // (8 * 500ms / 2). We pick 1500ms as the failure threshold so the test
    // fails before the fix and passes comfortably after.
    assert!(
        elapsed < Duration::from_millis(1500),
        "concurrent generate calls took {elapsed:?}; expected well under 1.5s, \
         which means sync ureq is parking tokio worker threads"
    );
}
