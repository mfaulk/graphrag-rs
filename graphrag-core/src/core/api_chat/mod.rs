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
