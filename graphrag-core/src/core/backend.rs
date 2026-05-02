//! Pluggable backend hooks for downstream consumers.
//!
//! `GraphRAG`'s built-in answer generation calls Ollama directly. Crates that
//! want to use a different LLM provider (OpenAI, Azure, etc.) without
//! forking the whole query pipeline can implement [`ChatBackend`] and inject
//! it via [`crate::GraphRAG::set_chat_backend`]. When set, the backend takes
//! over for the final answer-generation step in `ask` / `ask_explained`.

use std::sync::Arc;

use async_trait::async_trait;

use crate::core::Result;

/// Generation parameters passed to a [`ChatBackend`]. Mirrors a small subset
/// of `OllamaGenerationParams` so backends do not need to depend on
/// Ollama-specific types.
#[derive(Debug, Clone, Default)]
pub struct ChatParams {
    /// Soft cap on generated tokens. Backends should map this to whatever
    /// concept they have (`max_tokens`, `num_predict`, etc.).
    pub max_tokens: Option<u32>,
    /// Sampling temperature in [0.0, 1.0]. `None` means use backend default.
    pub temperature: Option<f32>,
    /// Hint at the prompt's required context window. Backends may ignore this
    /// — relevant mostly to local models with configurable context size.
    pub num_ctx: Option<u32>,
}

/// A pluggable chat-completion backend.
#[async_trait]
pub trait ChatBackend: Send + Sync {
    /// Generate a single completion for `prompt` honouring `params` where
    /// supported. Errors are surfaced through `crate::core::Result`.
    async fn complete(&self, prompt: &str, params: &ChatParams) -> Result<String>;
}

/// Convenience alias for trait objects.
pub type DynChatBackend = Arc<dyn ChatBackend>;
