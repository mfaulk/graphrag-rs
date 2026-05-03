//! Native HTTP `ChatBackend` implementations for OpenAI and Anthropic.
//!
//! Both providers speak `ureq`-based JSON over HTTPS and are wired through
//! the standard `ChatBackend` trait so they can replace the default Ollama
//! backend via `GraphRAG::set_chat_backend`. Streaming, tool calls, and
//! cloud-router variants (Azure, Bedrock, Vertex) are intentionally out of
//! scope for this module — see issue #90.

#[cfg(feature = "ureq")]
pub mod anthropic;
#[cfg(feature = "ureq")]
pub mod config;
#[cfg(feature = "ureq")]
pub mod openai;

#[cfg(feature = "ureq")]
pub use anthropic::AnthropicChat;
#[cfg(feature = "ureq")]
pub use config::{ChatProvider, ChatProviderConfig};
#[cfg(feature = "ureq")]
pub use openai::OpenAiChat;

#[cfg(all(test, feature = "ureq"))]
pub(crate) mod test_env_lock {
    //! Process-wide mutex serializing tests that mutate `OPENAI_API_KEY`
    //! / `ANTHROPIC_API_KEY`. `cargo test` runs unit tests in parallel
    //! within a single process, so concurrent set/remove of shared env
    //! vars across modules races and produces flaky failures.
    use std::sync::Mutex;

    pub(crate) static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Acquire the lock, recovering from a poisoned mutex left over by a
    /// previous test panic — poisoning is irrelevant here because we
    /// only protect env-var ordering, not invariants over the `()` guard.
    pub(crate) fn lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }
}
