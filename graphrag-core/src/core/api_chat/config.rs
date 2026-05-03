//! TOML-friendly chat-provider configuration.
//!
//! Companion to `embeddings::config::EmbeddingProviderConfig`, but for the
//! chat side. Exposes a `provider = "openai" | "anthropic" | "ollama"`
//! switch and falls back to `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` when
//! the config doesn't carry an explicit `api_key`.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::core::backend::{ChatBackend, DynChatBackend};
use crate::core::error::{GraphRAGError, Result};

use super::anthropic::{AnthropicChat, DEFAULT_ANTHROPIC_MODEL};
use super::openai::{OpenAiChat, DEFAULT_OPENAI_MODEL};

/// Selectable chat backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatProvider {
    /// OpenAI `/v1/chat/completions` (`gpt-4o-mini` default).
    OpenAi,
    /// Anthropic `/v1/messages` (`claude-haiku-4-5-20251001` default).
    Anthropic,
    /// Local Ollama. Construction is left to the caller — this enum
    /// variant exists so config files can express the choice; building
    /// the actual `OllamaAdapter` requires the `ollama` feature.
    Ollama,
}

impl ChatProvider {
    fn parse(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(Self::OpenAi),
            "anthropic" | "claude" => Ok(Self::Anthropic),
            "ollama" => Ok(Self::Ollama),
            other => Err(GraphRAGError::Config {
                message: format!("Unknown chat provider: {}", other),
            }),
        }
    }
}

/// TOML-friendly chat provider configuration.
///
/// ```toml
/// [chat]
/// provider = "anthropic"          # or "openai" / "ollama"
/// model = "claude-haiku-4-5-20251001"
/// # api_key = "sk-ant-..."        # optional; falls back to ANTHROPIC_API_KEY
/// max_tokens = 4096               # only honoured by Anthropic
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatProviderConfig {
    /// `"openai" | "anthropic" | "ollama"`.
    #[serde(default = "default_provider")]
    pub provider: String,

    /// Model identifier. Defaults differ per provider — see
    /// [`ChatProviderConfig::resolved_model`].
    #[serde(default)]
    pub model: Option<String>,

    /// API key (OpenAI/Anthropic). When absent we read
    /// `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` from the environment.
    #[serde(default)]
    pub api_key: Option<String>,

    /// Anthropic-only fallback `max_tokens` used when the call site
    /// doesn't pass one in `ChatParams`.
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

impl Default for ChatProviderConfig {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model: None,
            api_key: None,
            max_tokens: None,
        }
    }
}

fn default_provider() -> String {
    "ollama".to_string()
}

impl ChatProviderConfig {
    /// Parse the `provider` string into a typed enum.
    pub fn provider_kind(&self) -> Result<ChatProvider> {
        ChatProvider::parse(&self.provider)
    }

    /// Resolve the model name, applying the provider-specific default
    /// when `model` is unset.
    pub fn resolved_model(&self) -> Result<String> {
        let provider = self.provider_kind()?;
        Ok(match (&self.model, provider) {
            (Some(m), _) => m.clone(),
            (None, ChatProvider::OpenAi) => DEFAULT_OPENAI_MODEL.to_string(),
            (None, ChatProvider::Anthropic) => DEFAULT_ANTHROPIC_MODEL.to_string(),
            (None, ChatProvider::Ollama) => "llama3.2:3b".to_string(),
        })
    }

    /// Resolve the API key, falling back to the provider-specific env
    /// var when the config doesn't carry one. Returns `Auth` if the env
    /// var is missing for OpenAI/Anthropic. Ollama returns `None`.
    pub fn resolved_api_key(&self) -> Result<Option<String>> {
        if let Some(k) = &self.api_key {
            return Ok(Some(k.clone()));
        }
        match self.provider_kind()? {
            ChatProvider::OpenAi => std::env::var("OPENAI_API_KEY").ok().map_or_else(
                || {
                    Err(GraphRAGError::Auth {
                        message: "OPENAI_API_KEY env var not set and no api_key in config"
                            .to_string(),
                    })
                },
                |v| Ok(Some(v)),
            ),
            ChatProvider::Anthropic => std::env::var("ANTHROPIC_API_KEY").ok().map_or_else(
                || {
                    Err(GraphRAGError::Auth {
                        message: "ANTHROPIC_API_KEY env var not set and no api_key in config"
                            .to_string(),
                    })
                },
                |v| Ok(Some(v)),
            ),
            ChatProvider::Ollama => Ok(None),
        }
    }

    /// Build a `DynChatBackend` from this config. Returns `Unsupported`
    /// for the Ollama variant, which the caller must construct via the
    /// existing `core::ollama_adapters` path.
    pub fn build(&self) -> Result<DynChatBackend> {
        let provider = self.provider_kind()?;
        let model = self.resolved_model()?;
        match provider {
            ChatProvider::OpenAi => {
                let key = self
                    .resolved_api_key()?
                    .ok_or_else(|| GraphRAGError::Auth {
                        message: "OpenAI requires an api_key".to_string(),
                    })?;
                let backend = OpenAiChat::new(key, model);
                Ok(Arc::new(backend) as Arc<dyn ChatBackend>)
            },
            ChatProvider::Anthropic => {
                let key = self
                    .resolved_api_key()?
                    .ok_or_else(|| GraphRAGError::Auth {
                        message: "Anthropic requires an api_key".to_string(),
                    })?;
                let mut backend = AnthropicChat::new(key, model);
                if let Some(mt) = self.max_tokens {
                    backend = backend.with_default_max_tokens(mt);
                }
                Ok(Arc::new(backend) as Arc<dyn ChatBackend>)
            },
            ChatProvider::Ollama => Err(GraphRAGError::Unsupported {
                operation: "ChatProviderConfig::build for Ollama".to_string(),
                reason: "Ollama backend must be constructed via core::ollama_adapters \
                         (requires the `ollama` feature)"
                    .to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ChatProvider::parse` accepts the documented aliases and rejects
    /// unknown names.
    #[test]
    fn provider_parse_accepts_known_aliases() {
        assert_eq!(ChatProvider::parse("openai").unwrap(), ChatProvider::OpenAi);
        assert_eq!(
            ChatProvider::parse("Anthropic").unwrap(),
            ChatProvider::Anthropic
        );
        assert_eq!(
            ChatProvider::parse("CLAUDE").unwrap(),
            ChatProvider::Anthropic
        );
        assert_eq!(ChatProvider::parse("ollama").unwrap(), ChatProvider::Ollama);
        assert!(ChatProvider::parse("bogus").is_err());
    }

    /// `resolved_model` falls back to provider-specific defaults when no
    /// model is set in the config.
    #[test]
    fn resolved_model_uses_provider_default_when_absent() {
        let cfg = ChatProviderConfig {
            provider: "openai".into(),
            model: None,
            api_key: Some("k".into()),
            max_tokens: None,
        };
        assert_eq!(cfg.resolved_model().unwrap(), DEFAULT_OPENAI_MODEL);

        let cfg = ChatProviderConfig {
            provider: "anthropic".into(),
            model: None,
            api_key: Some("k".into()),
            max_tokens: None,
        };
        assert_eq!(cfg.resolved_model().unwrap(), DEFAULT_ANTHROPIC_MODEL);
    }

    /// When `api_key` is unset and the env var is absent, `build` errors
    /// instead of attempting an unauthenticated request.
    #[test]
    fn build_errors_when_no_api_key_and_no_env() {
        let prev = std::env::var("OPENAI_API_KEY").ok();
        std::env::remove_var("OPENAI_API_KEY");
        let cfg = ChatProviderConfig {
            provider: "openai".into(),
            model: None,
            api_key: None,
            max_tokens: None,
        };
        let res = cfg.build();
        assert!(res.is_err(), "expected error without api key");
        if let Some(v) = prev {
            std::env::set_var("OPENAI_API_KEY", v);
        }
    }

    /// `build` constructs an OpenAI backend when given an explicit key.
    #[test]
    fn build_constructs_openai_backend_with_explicit_key() {
        let cfg = ChatProviderConfig {
            provider: "openai".into(),
            model: Some("gpt-4o-mini".into()),
            api_key: Some("sk-test".into()),
            max_tokens: None,
        };
        let backend = cfg.build().expect("ok");
        // We only assert that we got a trait object; behaviour is covered
        // in `openai.rs` tests.
        let _ = &*backend;
    }

    /// `build` returns `Unsupported` for the Ollama variant — callers
    /// must use the dedicated adapter.
    #[test]
    fn build_rejects_ollama_variant() {
        let cfg = ChatProviderConfig {
            provider: "ollama".into(),
            model: None,
            api_key: None,
            max_tokens: None,
        };
        match cfg.build() {
            Ok(_) => panic!("ollama should not be buildable via ChatProviderConfig"),
            Err(GraphRAGError::Unsupported { .. }) => {},
            Err(other) => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
