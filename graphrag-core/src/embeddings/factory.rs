//! Build an [`AsyncEmbedder`] from `config.embeddings`.
//!
//! Two-tier dispatch: the registry's pre-injected embedder (if any) wins;
//! otherwise the `backend` string in [`crate::config::EmbeddingConfig`]
//! selects between hash, Ollama, and HTTP API providers. Errors that
//! surface during construction propagate; runtime errors during `embed`
//! are handled separately by `RetrievalSystem` so it can honour
//! `fallback_to_hash`.

use std::sync::Arc;

use async_trait::async_trait;

use crate::config::EmbeddingConfig as RuntimeEmbeddingConfig;
use crate::core::error::{GraphRAGError, Result};
use crate::core::registry::DynAsyncEmbedder;
use crate::core::traits::AsyncEmbedder;

#[cfg(feature = "ureq")]
use crate::embeddings::api_providers::HttpEmbeddingProvider;
#[cfg(feature = "ureq")]
use crate::embeddings::config::EmbeddingProviderConfig;
#[cfg(feature = "ollama")]
use crate::embeddings::ollama::OllamaEmbeddings;
use crate::embeddings::EmbeddingProvider;

/// Adapter exposing an [`EmbeddingProvider`] as an [`AsyncEmbedder`].
///
/// `EmbeddingProvider` (provider-specific trait, lives in `embeddings/`)
/// returns provider-typed errors via `GraphRAGError`. `AsyncEmbedder`
/// (registry-facing trait, lives in `core/traits.rs`) is the one
/// `ServiceRegistry` and `RetrievalSystem` understand. This adapter bridges
/// them so the factory can produce a single canonical type.
pub struct EmbeddingProviderAdapter<P: EmbeddingProvider + ?Sized> {
    inner: Arc<P>,
}

impl<P: EmbeddingProvider + ?Sized> EmbeddingProviderAdapter<P> {
    /// Wrap a shared embedding provider as an `AsyncEmbedder`.
    pub fn new(inner: Arc<P>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl<P> AsyncEmbedder for EmbeddingProviderAdapter<P>
where
    P: EmbeddingProvider + ?Sized + Send + Sync + 'static,
{
    type Error = GraphRAGError;

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.inner.embed(text).await
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.inner.embed_batch(texts).await
    }

    fn dimension(&self) -> usize {
        self.inner.dimensions()
    }

    async fn is_ready(&self) -> bool {
        self.inner.is_available()
    }
}

/// Build an [`AsyncEmbedder`] from `config.embeddings.backend`.
///
/// Returns `Ok(Some(_))` for backends that produce a real embedder,
/// `Ok(None)` for `"hash"` (caller should use the in-memory hash
/// generator), and `Err(_)` if the configured backend cannot be
/// constructed (missing API key, unsupported feature flags, etc.).
///
/// Each branch exists for a different reason:
/// - `"hash"` returns `None` so `RetrievalSystem` keeps using
///   `EmbeddingGenerator` (no allocation, no I/O, deterministic for
///   tests).
/// - `"ollama"` builds a local-network embedder pointed at Ollama;
///   needs the `ollama` feature.
/// - HTTP providers (`"openai"`, `"voyage"`, `"cohere"`, `"jina"`,
///   `"mistral"`, `"together"`) go through `HttpEmbeddingProvider`,
///   which already handles auth headers + the sync-`ureq`-on-blocking
///   pool dance described in `embeddings/api_providers.rs`.
pub fn build_async_embedder(config: &RuntimeEmbeddingConfig) -> Result<Option<DynAsyncEmbedder>> {
    let backend = config.backend.to_lowercase();

    match backend.as_str() {
        "hash" => Ok(None),

        "ollama" => {
            #[cfg(feature = "ollama")]
            {
                let model = config.model.clone().unwrap_or_else(|| {
                    "nomic-embed-text".to_string()
                });
                let dim = config.dimension.max(1);
                let provider = OllamaEmbeddings::new(model).with_dimensions(dim);
                let arc: Arc<dyn EmbeddingProvider> = Arc::new(provider);
                Ok(Some(Arc::new(EmbeddingProviderAdapter::new(arc))))
            }
            #[cfg(not(feature = "ollama"))]
            {
                Err(GraphRAGError::Config {
                    message: "embeddings.backend=\"ollama\" requires the `ollama` feature"
                        .to_string(),
                })
            }
        }

        "openai" | "voyage" | "voyageai" | "voyage-ai" | "cohere" | "jina" | "jinaai"
        | "jina-ai" | "mistral" | "mistralai" | "mistral-ai" | "together" | "togetherai"
        | "together-ai" | "huggingface" | "hf" => {
            #[cfg(feature = "ureq")]
            {
                let provider_cfg = EmbeddingProviderConfig {
                    provider: backend.clone(),
                    model: config
                        .model
                        .clone()
                        .unwrap_or_else(default_model_for_backend),
                    api_key: config.api_key.clone(),
                    cache_dir: config.cache_dir.clone(),
                    batch_size: config.batch_size,
                    dimensions: Some(config.dimension),
                };
                let embedding_cfg = provider_cfg.to_embedding_config()?;

                let provider = HttpEmbeddingProvider::from_config(&embedding_cfg)?;
                let provider = if let Some(endpoint) = config.api_endpoint.as_deref() {
                    provider.with_endpoint_for_tests(endpoint)
                } else {
                    provider
                };
                let arc: Arc<dyn EmbeddingProvider> = Arc::new(provider);
                Ok(Some(Arc::new(EmbeddingProviderAdapter::new(arc))))
            }
            #[cfg(not(feature = "ureq"))]
            {
                let _ = backend;
                Err(GraphRAGError::Config {
                    message: format!(
                        "embeddings.backend=\"{}\" requires the `ureq` feature",
                        config.backend
                    ),
                })
            }
        }

        other => Err(GraphRAGError::Config {
            message: format!(
                "Unknown embeddings.backend \"{}\". Expected one of: hash, ollama, openai, voyage, cohere, jina, mistral, together, huggingface",
                other
            ),
        }),
    }
}

fn default_model_for_backend() -> String {
    // Fallback model when a backend is configured without an explicit model.
    // Most users will set `[embeddings.model]` themselves; this keeps the
    // factory total rather than panicking.
    "text-embedding-3-small".to_string()
}
