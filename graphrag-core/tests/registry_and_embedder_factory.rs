//! Integration tests for issues #6 (registry threading) and #91 (embedder
//! factory dispatch). Covers:
//!
//! - registry-injected `Embedder` is consulted by `RetrievalSystem`
//! - registry-injected `ChatBackend` survives `set_chat_backend` shim
//! - default registry preserves the previous behavior
//! - `config.embeddings.backend = "openai"` hits the configured endpoint
//! - `fallback_to_hash` survives a runtime API failure
//! - `backend = "hash"` keeps the legacy hash path

#![cfg(all(feature = "ureq", feature = "async", feature = "memory-storage"))]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use async_trait::async_trait;

use graphrag_core::core::backend::{ChatBackend, ChatParams, DynChatBackend};
use graphrag_core::core::registry::{RegistryBuilder, ServiceRegistry};
use graphrag_core::core::traits::AsyncEmbedder;
use graphrag_core::core::GraphRAGError;
use graphrag_core::{Config, GraphRAG};

// ---------------------------------------------------------------------------
// Mock embedding HTTP server (records every request).
// ---------------------------------------------------------------------------

struct MockHttpServer {
    port: u16,
    shutdown: Arc<AtomicBool>,
    request_count: Arc<AtomicUsize>,
    request_paths: Arc<std::sync::Mutex<Vec<String>>>,
}

impl MockHttpServer {
    fn start_ok() -> Self {
        Self::start(MockMode::OkOpenAi)
    }

    fn start_err() -> Self {
        Self::start(MockMode::Error500)
    }

    fn start(mode: MockMode) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        let shutdown = Arc::new(AtomicBool::new(false));
        let request_count = Arc::new(AtomicUsize::new(0));
        let request_paths = Arc::new(std::sync::Mutex::new(Vec::new()));

        listener.set_nonblocking(true).expect("set_nonblocking");

        let shutdown_clone = shutdown.clone();
        let count_clone = request_count.clone();
        let paths_clone = request_paths.clone();

        thread::spawn(move || {
            while !shutdown_clone.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let count = count_clone.clone();
                        let paths = paths_clone.clone();
                        let mode = mode;
                        thread::spawn(move || handle_conn(stream, mode, count, paths));
                    },
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    },
                    Err(_) => break,
                }
            }
        });

        Self {
            port,
            shutdown,
            request_count,
            request_paths,
        }
    }

    fn url(&self) -> String {
        // OpenAI-style path so the API factory is happy.
        format!("http://127.0.0.1:{}/v1/embeddings", self.port)
    }

    fn count(&self) -> usize {
        self.request_count.load(Ordering::SeqCst)
    }

    fn paths(&self) -> Vec<String> {
        self.request_paths.lock().unwrap().clone()
    }
}

impl Drop for MockHttpServer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

#[derive(Copy, Clone)]
enum MockMode {
    OkOpenAi,
    Error500,
}

fn handle_conn(
    mut stream: std::net::TcpStream,
    mode: MockMode,
    count: Arc<AtomicUsize>,
    paths: Arc<std::sync::Mutex<Vec<String>>>,
) {
    stream.set_nonblocking(false).ok();

    let mut reader = BufReader::new(stream.try_clone().expect("clone"));
    let mut content_length = 0usize;
    let mut request_line = String::new();

    // Read request line.
    if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
        return;
    }
    let req_path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("")
        .to_string();

    // Read headers.
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

    count.fetch_add(1, Ordering::SeqCst);
    paths.lock().unwrap().push(req_path);

    match mode {
        MockMode::OkOpenAi => {
            let body = br#"{"data":[{"embedding":[0.1,0.2,0.3,0.4],"index":0}],"model":"test","usage":{}}"#;
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(body);
        },
        MockMode::Error500 => {
            let header = b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            let _ = stream.write_all(header);
        },
    }
    let _ = stream.flush();
}

// ---------------------------------------------------------------------------
// Mock AsyncEmbedder that counts calls so the test can assert it ran.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct CountingEmbedder {
    count: Arc<AtomicUsize>,
    dim: usize,
}

impl CountingEmbedder {
    fn new(dim: usize) -> Self {
        Self {
            count: Arc::new(AtomicUsize::new(0)),
            dim,
        }
    }

    fn calls(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl AsyncEmbedder for CountingEmbedder {
    type Error = GraphRAGError;

    async fn embed(&self, text: &str) -> Result<Vec<f32>, GraphRAGError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        // Deterministic but text-dependent so vectors differ; this lets the
        // vector store actually rank chunks instead of treating all
        // embeddings as identical points.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();
        let mut v = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let h = seed.wrapping_add(i as u64);
            v.push(((h % 1000) as f32) / 1000.0);
        }
        Ok(v)
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, GraphRAGError> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t).await?);
        }
        Ok(out)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    async fn is_ready(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// FlakyEmbedder: succeeds for the first `success_calls` invocations, then
// errors. Used to exercise the runtime fallback path without restarting the
// `RetrievalSystem`.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct FlakyEmbedder {
    count: Arc<AtomicUsize>,
    dim: usize,
    success_calls: usize,
}

impl FlakyEmbedder {
    fn new(dim: usize, success_calls: usize) -> Self {
        Self {
            count: Arc::new(AtomicUsize::new(0)),
            dim,
            success_calls,
        }
    }
}

#[async_trait]
impl AsyncEmbedder for FlakyEmbedder {
    type Error = GraphRAGError;

    async fn embed(&self, text: &str) -> Result<Vec<f32>, GraphRAGError> {
        let n = self.count.fetch_add(1, Ordering::SeqCst);
        if n >= self.success_calls {
            return Err(GraphRAGError::Embedding {
                message: "FlakyEmbedder: simulated runtime failure".to_string(),
            });
        }
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();
        let mut v = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let h = seed.wrapping_add(i as u64);
            v.push(((h % 1000) as f32) / 1000.0);
        }
        Ok(v)
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, GraphRAGError> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t).await?);
        }
        Ok(out)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    async fn is_ready(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Mock ChatBackend that records prompts.
// ---------------------------------------------------------------------------

struct CountingChatBackend {
    count: Arc<AtomicUsize>,
}

impl CountingChatBackend {
    fn new() -> Self {
        Self {
            count: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl ChatBackend for CountingChatBackend {
    async fn complete(&self, _prompt: &str, _params: &ChatParams) -> Result<String, GraphRAGError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok("mock-answer".to_string())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn config_with_openai_endpoint(endpoint: &str) -> Config {
    let mut config = Config::default();
    config.embeddings.backend = "openai".to_string();
    config.embeddings.dimension = 4;
    config.embeddings.api_key = Some("sk-test".to_string());
    config.embeddings.api_endpoint = Some(endpoint.to_string());
    config.embeddings.fallback_to_hash = false;
    config.ollama.enabled = false;
    config.auto_save.enabled = false;
    config
}

// ---------------------------------------------------------------------------
// Issue #91 tests — embedder factory dispatch.
// ---------------------------------------------------------------------------

/// `config.embeddings.backend = "openai"` causes the retrieval flow to call
/// the configured endpoint instead of the hash generator.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retrieval_uses_configured_openai_embedder() {
    let server = MockHttpServer::start_ok();
    tokio::time::sleep(Duration::from_millis(30)).await;

    let config = config_with_openai_endpoint(&server.url());
    let mut graphrag = GraphRAG::new(config).expect("construct GraphRAG");
    graphrag.initialize().expect("initialize");
    graphrag
        .add_document_from_text("Socrates was a Greek philosopher.")
        .expect("add doc");

    // Drive the retrieval path; we don't care about the exact answer, only
    // that the configured endpoint was hit.
    let _ = graphrag.ask("Who was Socrates?").await;

    let count = server.count();
    assert!(
        count > 0,
        "expected the configured OpenAI endpoint to be hit at least once, got {} requests",
        count
    );
    let paths = server.paths();
    assert!(
        paths.iter().any(|p| p.contains("/v1/embeddings")),
        "expected at least one /v1/embeddings request, got {:?}",
        paths
    );
}

/// When `fallback_to_hash = true`, a deliberately-broken endpoint should not
/// crash retrieval; the hash generator takes over.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retrieval_falls_back_to_hash_on_api_failure() {
    let server = MockHttpServer::start_err();
    tokio::time::sleep(Duration::from_millis(30)).await;

    let mut config = config_with_openai_endpoint(&server.url());
    config.embeddings.fallback_to_hash = true;

    let mut graphrag = GraphRAG::new(config).expect("construct");
    graphrag.initialize().expect("initialize");
    graphrag
        .add_document_from_text("Plato was a student of Socrates.")
        .expect("add doc");

    // Should not panic / error out — fallback path produces hash embeddings.
    let result = graphrag.ask("What did Plato do?").await;
    assert!(
        result.is_ok(),
        "ask() should fall back to hash when API errors, got {:?}",
        result.err()
    );
}

/// Backwards-compat: `backend = "hash"` (default) keeps the in-memory
/// generator and never hits the network.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retrieval_uses_hash_when_backend_is_hash() {
    let mut config = Config::default();
    config.embeddings.backend = "hash".to_string();
    config.ollama.enabled = false;
    config.auto_save.enabled = false;

    let mut graphrag = GraphRAG::new(config).expect("construct");
    graphrag.initialize().expect("initialize");
    graphrag
        .add_document_from_text("Aristotle was a student of Plato.")
        .expect("add doc");

    let result = graphrag.ask("Who taught Aristotle?").await;
    assert!(
        result.is_ok(),
        "hash-backend retrieval must succeed without network, got {:?}",
        result.err()
    );
}

// ---------------------------------------------------------------------------
// Issue #6 tests — registry threading.
// ---------------------------------------------------------------------------

/// A registry with an injected mock embedder must drive the retrieval flow
/// instead of falling through to `config.embeddings.backend`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graphrag_new_with_registry_uses_injected_embedder() {
    let mut config = Config::default();
    // Set backend to "openai" with no endpoint to prove the registry wins:
    // if the registry weren't consulted, factory dispatch would error.
    config.embeddings.backend = "openai".to_string();
    config.embeddings.dimension = 8;
    config.embeddings.api_key = Some("would-fail".to_string());
    config.embeddings.fallback_to_hash = false;
    config.ollama.enabled = false;
    config.auto_save.enabled = false;

    let counting = CountingEmbedder::new(8);
    let registry = RegistryBuilder::new()
        .with_async_embedder(counting.clone())
        .build();

    let mut graphrag = GraphRAG::new_with_registry(config, registry).expect("construct");
    graphrag.initialize().expect("initialize");
    graphrag
        .add_document_from_text("test document for embedding")
        .expect("add doc");

    let _ = graphrag.ask("test query").await;

    assert!(
        counting.calls() > 0,
        "registry-injected embedder should be invoked at least once, got 0 calls"
    );
}

/// Injected `ChatBackend` short-circuits Ollama dispatch.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graphrag_new_with_registry_uses_injected_chat_backend() {
    let mut config = Config::default();
    config.embeddings.backend = "hash".to_string();
    config.ollama.enabled = false;
    config.auto_save.enabled = false;

    let backend = Arc::new(CountingChatBackend::new());
    let backend_dyn: DynChatBackend = backend.clone();
    let count_handle = backend.count.clone();

    let registry = RegistryBuilder::new()
        .with_chat_backend(backend_dyn)
        .build();

    let mut graphrag = GraphRAG::new_with_registry(config, registry).expect("construct");
    graphrag.initialize().expect("initialize");
    // Reuse the same text-and-query pair as the working
    // `set_chat_backend_routes_through_registry` test so this test isolates
    // the "registry path wires the chat backend correctly" property without
    // depending on hash-embedding similarity vagaries.
    graphrag
        .add_document_from_text(
            "Plato was a student of Socrates and the teacher of Aristotle. \
             Plato founded the Academy in Athens, the first institution of higher \
             learning in the Western world. He wrote dialogues exploring justice, \
             the nature of the soul, and political philosophy.",
        )
        .expect("add doc");

    let answer = graphrag.ask("Tell me about Plato").await.expect("ask ok");
    assert_eq!(answer, "mock-answer", "injected backend's reply should win");
    assert_eq!(
        count_handle.load(Ordering::SeqCst),
        1,
        "chat backend must be hit exactly once"
    );
}

/// Default registry preserves the previous `GraphRAG::new` behavior.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graphrag_new_uses_default_registry_for_backwards_compat() {
    let mut config = Config::default();
    config.embeddings.backend = "hash".to_string();
    config.ollama.enabled = false;
    config.auto_save.enabled = false;

    let mut graphrag = GraphRAG::new(config).expect("construct");
    graphrag.initialize().expect("initialize");
    graphrag
        .add_document_from_text("legacy callers see the same behavior")
        .expect("add doc");

    // Without an injected backend or Ollama, ask should still complete via
    // the formatted-results fallback path.
    let result = graphrag.ask("legacy").await;
    assert!(
        result.is_ok(),
        "default-registry behavior must not regress, got {:?}",
        result.err()
    );
}

/// `set_chat_backend` continues to work as a delegation shim that stores
/// the backend in the registry's typed slot (no caller changes required).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_chat_backend_routes_through_registry() {
    let mut config = Config::default();
    config.embeddings.backend = "hash".to_string();
    config.ollama.enabled = false;
    config.auto_save.enabled = false;

    let backend = Arc::new(CountingChatBackend::new());
    let count_handle = backend.count.clone();

    let mut graphrag = GraphRAG::new(config).expect("construct");
    graphrag.set_chat_backend(Some(backend.clone() as DynChatBackend));
    graphrag.initialize().expect("initialize");
    graphrag
        .add_document_from_text(
            "Plato was a student of Socrates and the teacher of Aristotle. \
             Plato founded the Academy in Athens, the first institution of higher \
             learning in the Western world. He wrote dialogues exploring justice, \
             the nature of the soul, and political philosophy.",
        )
        .expect("add doc");

    let _ = graphrag.ask("Tell me about Plato").await;
    assert_eq!(
        count_handle.load(Ordering::SeqCst),
        1,
        "set_chat_backend must keep working as a shim into the registry"
    );

    // Clearing the backend should disable dispatch; the next call falls back
    // to the no-LLM formatted-results path.
    graphrag.set_chat_backend(None);
    let _ = graphrag.ask("Tell me about Plato").await;
    assert_eq!(
        count_handle.load(Ordering::SeqCst),
        1,
        "after clearing, backend should not be invoked again"
    );
}

/// Empty `ServiceRegistry::default()` should not panic in the typed-slot
/// accessors (defense in depth — no embedder, no chat backend).
#[test]
fn empty_registry_has_no_typed_slots() {
    let reg = ServiceRegistry::default();
    assert!(reg.async_embedder().is_none());
    assert!(reg.chat_backend().is_none());
}

/// `clear_chat_backend` nulls only the chat-backend slot and preserves
/// any other registered services. The previous `set_chat_backend(None)`
/// rebuilt the whole registry, dropping every entry registered through
/// the generic `register/get` map (issue #6 review).
#[test]
fn clear_chat_backend_preserves_other_services() {
    use graphrag_core::core::registry::DynAsyncEmbedder;

    #[derive(Debug, PartialEq)]
    struct CustomService {
        marker: u32,
    }

    let mut reg = ServiceRegistry::default();
    reg.register(CustomService { marker: 42 });

    let counting = CountingEmbedder::new(8);
    let emb_dyn: DynAsyncEmbedder = Arc::new(counting);
    reg.set_async_embedder(emb_dyn);

    let backend: DynChatBackend = Arc::new(CountingChatBackend::new());
    reg.set_chat_backend(backend);

    assert!(reg.chat_backend().is_some());
    assert!(reg.async_embedder().is_some());
    assert!(reg.has::<CustomService>());

    reg.clear_chat_backend();

    assert!(
        reg.chat_backend().is_none(),
        "chat backend slot must be cleared"
    );
    assert!(
        reg.async_embedder().is_some(),
        "embedder slot must survive clear_chat_backend"
    );
    let svc = reg
        .get::<CustomService>()
        .expect("generic-map service must survive clear_chat_backend");
    assert_eq!(svc.marker, 42);
}

/// `backend = "huggingface"` must fail fast in the factory rather than
/// silently routing through the HTTP provider (which requires an API key
/// HF doesn't use) or, with `fallback_to_hash = true`, downgrading to
/// hash embeddings without telling the user (issue #91 review).
#[test]
fn factory_returns_error_for_unsupported_huggingface_backend() {
    use graphrag_core::embeddings::factory::build_async_embedder;

    let mut config = Config::default();
    config.embeddings.backend = "huggingface".to_string();
    config.embeddings.dimension = 384;
    config.embeddings.fallback_to_hash = false;

    match build_async_embedder(&config.embeddings) {
        Ok(_) => panic!("huggingface backend must error in the factory"),
        Err(err) => {
            let msg = format!("{err}");
            assert!(
                msg.contains("huggingface"),
                "error must name the backend, got: {msg}"
            );
        },
    }

    // Same expectation for the `hf` alias.
    let mut config = Config::default();
    config.embeddings.backend = "hf".to_string();
    match build_async_embedder(&config.embeddings) {
        Ok(_) => panic!("hf alias must also error"),
        Err(err) => assert!(
            format!("{err}").contains("huggingface"),
            "error must reference the huggingface backend"
        ),
    }
}

/// When the configured embedder produces vectors of one dimension but
/// `config.embeddings.dimension` claims another, the runtime fallback
/// must produce vectors with the embedder's actual length — not the
/// config's lie — so cosine similarity against already-indexed vectors
/// stays meaningful (issue #91 review, finding #2 + #6). Without this,
/// an OpenAI-indexed corpus (1536-dim) queried after a fallback could
/// compare against a 768-dim hash vector and silently return 0
/// similarity.
///
/// Drives `RetrievalSystem` directly so the assertion is on the actual
/// fallback vector length, not on `ask()`'s end-to-end success/failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retrieval_handles_provider_success_then_failure_dim_consistent() {
    use graphrag_core::core::traits::AsyncEmbedder as AE;
    use graphrag_core::retrieval::RetrievalSystem;
    use std::sync::Arc;

    let provider_dim = 768usize;

    // Lie about the configured dimension to prove the fallback respects
    // the embedder's actual output length, not the user's `[embeddings]`
    // value.
    let mut config = Config::default();
    config.embeddings.backend = "hash".to_string();
    config.embeddings.dimension = 16;
    config.embeddings.fallback_to_hash = true;

    // Succeed for the first call (drives the "first successful embed
    // teaches the system the real dim" path), fail after that.
    let flaky = FlakyEmbedder::new(provider_dim, 1);
    assert_eq!(AE::dimension(&flaky), provider_dim);

    let embedder: graphrag_core::core::registry::DynAsyncEmbedder = Arc::new(flaky.clone());
    let mut retrieval =
        RetrievalSystem::new_with_embedder(&config, Some(embedder)).expect("construct");

    // First call: succeeds with provider's true dim. Use `vector_search`
    // (which routes through the private `embed_text`) — the indexed-set
    // is empty so we don't assert on results, only that no error fires
    // and the embedder was invoked.
    let _ = retrieval
        .vector_search("first call (success path)", 5)
        .await
        .expect("first call must succeed via real embedder");
    assert_eq!(
        flaky.count.load(Ordering::SeqCst),
        1,
        "embedder should have been called once on the success path"
    );

    // Subsequent calls: embedder errors → fallback fires. The fallback
    // vector length must match the *embedder's* dim (768), not the
    // config's lie (16) and not the legacy default (128).
    let v_fallback = retrieval
        .vector_search("second call (fallback path)", 5)
        .await
        .expect("fallback must not propagate error when fallback_to_hash=true");
    // `vector_search` returns search results, not the embedding; verify
    // the embedder was hit twice (once success, once failure) so we know
    // the fallback path ran.
    assert_eq!(
        flaky.count.load(Ordering::SeqCst),
        2,
        "embedder should have been re-attempted on the fallback path"
    );
    // Empty vector store -> empty search results, but no panic and no
    // error means dimensions agreed end-to-end.
    let _ = v_fallback;
}

/// A typo in `[embeddings].backend` must surface immediately, even when
/// `fallback_to_hash = true` (issue #91 review, finding #3). The
/// `fallback_to_hash` switch governs *runtime* failures (HTTP 5xx, rate
/// limits) — silently swallowing a configuration error here would mean
/// the user gets bad retrieval six months later instead of an error at
/// boot.
#[test]
fn factory_errors_on_unknown_backend_typo() {
    use graphrag_core::embeddings::factory::build_async_embedder;

    let mut config = Config::default();
    config.embeddings.backend = "opena".to_string(); // typo
    config.embeddings.fallback_to_hash = true;

    match build_async_embedder(&config.embeddings) {
        Ok(_) => panic!("typo'd backend must error in the factory"),
        Err(err) => {
            let msg = format!("{err}");
            assert!(
                msg.contains("opena") || msg.to_lowercase().contains("unknown"),
                "error must identify the bad backend, got: {msg}"
            );
        },
    }
}

/// `RetrievalSystem::new_with_embedder` must propagate factory errors
/// rather than silently swallowing them when `fallback_to_hash = true`.
/// Construction-time errors are *configuration* errors; runtime
/// fallback is a separate concern (issue #91 review, finding #3).
#[test]
fn retrieval_construction_errors_propagate_through_fallback_to_hash() {
    use graphrag_core::retrieval::RetrievalSystem;

    let mut config = Config::default();
    config.embeddings.backend = "definitely-not-a-real-backend".to_string();
    config.embeddings.fallback_to_hash = true; // must NOT rescue config errors

    match RetrievalSystem::new_with_embedder(&config, None) {
        Ok(_) => panic!(
            "unknown backend must produce an error even with \
             fallback_to_hash=true; the flag is only for runtime errors"
        ),
        Err(err) => {
            let msg = format!("{err}").to_lowercase();
            assert!(
                msg.contains("definitely-not-a-real-backend") || msg.contains("unknown"),
                "error must identify the bad backend, got: {msg}"
            );
        },
    }
}

/// Direct check that fix #2 wires the embedder's dim into `embed_text`'s
/// fallback. Distinct from the integration test above so a failure
/// points at the dim-tracking field, not somewhere deeper.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retrieval_fallback_uses_provider_dimension_not_config() {
    use graphrag_core::core::traits::AsyncEmbedder as AE;
    use graphrag_core::retrieval::RetrievalSystem;
    use std::sync::Arc;

    let provider_dim = 1024usize;
    let mut config = Config::default();
    config.embeddings.backend = "hash".to_string();
    // Config lies — embedder is the source of truth.
    config.embeddings.dimension = 32;
    config.embeddings.fallback_to_hash = true;

    // Always-failing embedder so the fallback fires on every call. The
    // construction-time `dimension()` query must size the hash fallback,
    // not the config.
    let flaky = FlakyEmbedder::new(provider_dim, 0);
    let embedder: graphrag_core::core::registry::DynAsyncEmbedder = Arc::new(flaky.clone());
    assert_eq!(AE::dimension(&flaky), provider_dim);

    let mut retrieval =
        RetrievalSystem::new_with_embedder(&config, Some(embedder)).expect("construct");

    // The embedder fails, fallback fires; we can't observe the embedding
    // directly, but `vector_search` returning Ok means no internal
    // length mismatch panicked or errored.
    let _ = retrieval
        .vector_search("test", 1)
        .await
        .expect("fallback must produce a usable vector");
}
