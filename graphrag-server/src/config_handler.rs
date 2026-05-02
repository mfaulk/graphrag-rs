//! Dynamic configuration handler for GraphRAG Server
//!
//! This module provides JSON-based configuration for the full GraphRAG pipeline
//! via REST API endpoints, allowing dynamic initialization without requiring
//! TOML files or environment variables.

use graphrag_core::Config;
use serde::{Deserialize, Serialize};
use serde_json;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Errors produced by [`ConfigManager`].
///
/// Distinguishing parse from validation lets the API handler boundary map each
/// case to the right HTTP status (both BadRequest, but the body shape differs)
/// and lets future callers programmatically branch on the failure mode.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to parse JSON: {0}")]
    Parse(#[from] serde_json::Error),

    #[error("configuration validation failed: {0:?}")]
    Validation(Vec<String>),

    #[error("no configuration set")]
    NotSet,
}

/// Server configuration state
#[derive(Clone)]
pub struct ConfigManager {
    /// Current active configuration (if initialized)
    config: Arc<RwLock<Option<Config>>>,
    /// Configuration validation errors
    validation_errors: Arc<RwLock<Vec<String>>>,
}

impl ConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        Self {
            config: Arc::new(RwLock::new(None)),
            validation_errors: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Set configuration from JSON
    pub async fn set_from_json(&self, json_str: &str) -> Result<(), ConfigError> {
        // Parse JSON into Config — `?` lifts serde_json::Error via #[from]
        let config: Config = serde_json::from_str(json_str)?;

        // Validate configuration
        let errors = self.validate_config(&config).await;
        if !errors.is_empty() {
            *self.validation_errors.write().await = errors.clone();
            return Err(ConfigError::Validation(errors));
        }

        // Store configuration
        *self.config.write().await = Some(config);
        *self.validation_errors.write().await = Vec::new();

        Ok(())
    }

    /// Get current configuration (clone)
    pub async fn get_config(&self) -> Option<Config> {
        self.config.read().await.clone()
    }

    /// Check if configuration is set
    pub async fn is_configured(&self) -> bool {
        self.config.read().await.is_some()
    }

    /// Get validation errors
    #[allow(dead_code)]
    pub async fn get_validation_errors(&self) -> Vec<String> {
        self.validation_errors.read().await.clone()
    }

    /// Validate configuration
    async fn validate_config(&self, config: &Config) -> Vec<String> {
        let mut errors = Vec::new();

        // Validate chunk size
        if config.chunk_size == 0 {
            errors.push("chunk_size must be greater than 0".to_string());
        }
        if config.chunk_size < config.chunk_overlap {
            errors.push("chunk_size must be greater than chunk_overlap".to_string());
        }

        // Validate embeddings
        if config.embeddings.dimension == 0 {
            errors.push("embeddings.dimension must be greater than 0".to_string());
        }

        // Validate graph config
        if config.graph.max_connections == 0 {
            errors.push("graph.max_connections must be greater than 0".to_string());
        }

        // Validate retrieval config
        if config.retrieval.top_k == 0 {
            errors.push("retrieval.top_k must be greater than 0".to_string());
        }

        errors
    }

    /// Convert Config to JSON string
    pub async fn to_json(&self) -> Result<String, ConfigError> {
        let config = self.config.read().await;
        match config.as_ref() {
            Some(cfg) => Ok(serde_json::to_string_pretty(cfg)?),
            None => Err(ConfigError::NotSet),
        }
    }

    /// Get default configuration as JSON
    pub fn default_config_json() -> String {
        let config = Config::default();
        serde_json::to_string_pretty(&config).unwrap_or_else(|_| "{}".to_string())
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration request for API endpoint
#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
pub struct ConfigRequest {
    /// Full configuration in JSON format (same as Config struct)
    pub config: serde_json::Value,
}

/// Configuration response
#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub struct ConfigResponse {
    pub success: bool,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub errors: Option<Vec<String>>,
}

/// Get configuration template endpoint response
#[derive(Debug, Serialize)]
pub struct ConfigTemplateResponse {
    pub template: serde_json::Value,
    pub description: String,
    pub examples: Vec<ConfigExample>,
}

/// Configuration example
#[derive(Debug, Serialize)]
pub struct ConfigExample {
    pub name: String,
    pub description: String,
    pub config: serde_json::Value,
}

/// Generate configuration templates
pub fn get_config_templates() -> ConfigTemplateResponse {
    let default_config = Config::default();
    let template = serde_json::to_value(&default_config).unwrap_or(serde_json::json!({}));

    let examples = vec![
        ConfigExample {
            name: "minimal".to_string(),
            description: "Minimal configuration with hash-based embeddings".to_string(),
            config: serde_json::json!({
                "output_dir": "./output",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "embeddings": {
                    "backend": "hash",
                    "dimension": 384,
                    "fallback_to_hash": true,
                    "batch_size": 32
                },
                "graph": {
                    "max_connections": 10,
                    "similarity_threshold": 0.8
                },
                "text": {
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "languages": ["en"]
                },
                "entities": {
                    "min_confidence": 0.7,
                    "entity_types": ["PERSON", "ORG", "LOCATION"]
                },
                "retrieval": {
                    "top_k": 10,
                    "search_algorithm": "cosine"
                },
                "parallel": {
                    "num_threads": 0,
                    "enabled": true,
                    "min_batch_size": 10,
                    "chunk_batch_size": 100,
                    "parallel_embeddings": true,
                    "parallel_graph_ops": true,
                    "parallel_vector_ops": true
                },
                "ollama": {
                    "enabled": false,
                    "host": "http://localhost",
                    "port": 11434,
                    "embedding_model": "nomic-embed-text",
                    "chat_model": "llama3.2:3b",
                    "timeout_seconds": 30,
                    "max_retries": 3,
                    "fallback_to_hash": true
                },
                "enhancements": {
                    "enabled": true
                }
            }),
        },
        ConfigExample {
            name: "ollama_production".to_string(),
            description: "Production setup with Ollama LLM and real embeddings".to_string(),
            config: serde_json::json!({
                "output_dir": "./output",
                "chunk_size": 800,
                "chunk_overlap": 200,
                "embeddings": {
                    "backend": "ollama",
                    "dimension": 768,
                    "model": "nomic-embed-text",
                    "fallback_to_hash": true,
                    "batch_size": 32
                },
                "graph": {
                    "max_connections": 25,
                    "similarity_threshold": 0.75
                },
                "text": {
                    "chunk_size": 800,
                    "chunk_overlap": 200,
                    "languages": ["en"]
                },
                "entities": {
                    "min_confidence": 0.65,
                    "entity_types": ["PERSON", "CONCEPT", "LOCATION", "EVENT", "ORGANIZATION"]
                },
                "retrieval": {
                    "top_k": 15,
                    "search_algorithm": "cosine"
                },
                "parallel": {
                    "num_threads": 8,
                    "enabled": true,
                    "min_batch_size": 10,
                    "chunk_batch_size": 100,
                    "parallel_embeddings": true,
                    "parallel_graph_ops": true,
                    "parallel_vector_ops": true
                },
                "ollama": {
                    "enabled": true,
                    "host": "http://localhost",
                    "port": 11434,
                    "embedding_model": "nomic-embed-text",
                    "chat_model": "llama3.1:8b",
                    "timeout_seconds": 300,
                    "max_retries": 3,
                    "fallback_to_hash": true
                },
                "enhancements": {
                    "enabled": true,
                    "query_analysis": {
                        "enabled": true,
                        "min_confidence": 0.6
                    },
                    "adaptive_retrieval": {
                        "enabled": true,
                        "use_query_analysis": true
                    }
                }
            }),
        },
        ConfigExample {
            name: "high_performance".to_string(),
            description: "Optimized for speed with parallel processing".to_string(),
            config: serde_json::json!({
                "output_dir": "./output",
                "chunk_size": 512,
                "chunk_overlap": 100,
                "embeddings": {
                    "backend": "hash",
                    "dimension": 256,
                    "fallback_to_hash": true,
                    "batch_size": 64
                },
                "graph": {
                    "max_connections": 15,
                    "similarity_threshold": 0.7
                },
                "text": {
                    "chunk_size": 512,
                    "chunk_overlap": 100,
                    "languages": ["en"]
                },
                "entities": {
                    "min_confidence": 0.6,
                    "entity_types": ["PERSON", "ORG"]
                },
                "retrieval": {
                    "top_k": 20,
                    "search_algorithm": "cosine"
                },
                "parallel": {
                    "num_threads": 16,
                    "enabled": true,
                    "min_batch_size": 5,
                    "chunk_batch_size": 200,
                    "parallel_embeddings": true,
                    "parallel_graph_ops": true,
                    "parallel_vector_ops": true
                },
                "ollama": {
                    "enabled": false
                },
                "enhancements": {
                    "enabled": false
                }
            }),
        },
    ];

    ConfigTemplateResponse {
        template,
        description: "Full GraphRAG configuration template with all options".to_string(),
        examples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_config_manager_creation() {
        let manager = ConfigManager::new();
        assert!(!manager.is_configured().await);
    }

    #[tokio::test]
    async fn test_set_config_from_json() {
        let manager = ConfigManager::new();
        let json = ConfigManager::default_config_json();

        let result = manager.set_from_json(&json).await;
        assert!(result.is_ok());
        assert!(manager.is_configured().await);
    }

    #[tokio::test]
    async fn test_invalid_json() {
        let manager = ConfigManager::new();
        let result = manager.set_from_json("{invalid json}").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validation() {
        let manager = ConfigManager::new();
        let invalid_config = serde_json::json!({
            "output_dir": "./output",
            "chunk_size": 0,  // Invalid!
            "chunk_overlap": 200,
        });

        let json = serde_json::to_string(&invalid_config).unwrap();
        let result = manager.set_from_json(&json).await;
        assert!(result.is_err());
    }

    // Malformed JSON should produce a typed Parse error, not a freeform String
    // (regression for #47: callers can now branch on the failure mode).
    #[tokio::test]
    async fn set_from_json_returns_parse_variant_on_malformed_input() {
        let manager = ConfigManager::new();
        let err = manager
            .set_from_json("{not valid json")
            .await
            .expect_err("malformed JSON should error");
        match err {
            ConfigError::Parse(_) => {},
            other => panic!("expected ConfigError::Parse, got: {other:?}"),
        }
    }

    // A well-formed Config that fails semantic validation should produce a
    // Validation variant carrying the structured error list. Build the JSON
    // from the defaults so we exercise the validation path rather than
    // tripping a missing-field parse error.
    #[tokio::test]
    async fn set_from_json_returns_validation_variant_with_error_list() {
        let manager = ConfigManager::new();
        let mut value: serde_json::Value =
            serde_json::from_str(&ConfigManager::default_config_json())
                .expect("default config JSON should parse");
        value["chunk_size"] = serde_json::json!(0);
        let json = serde_json::to_string(&value).unwrap();

        let err = manager
            .set_from_json(&json)
            .await
            .expect_err("zero chunk_size should fail validation");
        match err {
            ConfigError::Validation(errors) => {
                assert!(
                    !errors.is_empty(),
                    "validation error list should be populated"
                );
                assert!(
                    errors.iter().any(|e| e.contains("chunk_size")),
                    "should mention chunk_size, got: {errors:?}"
                );
            },
            other => panic!("expected ConfigError::Validation, got: {other:?}"),
        }
    }

    // Reading config when none has been set returns a typed NotSet variant.
    #[tokio::test]
    async fn to_json_returns_not_set_variant_when_unconfigured() {
        let manager = ConfigManager::new();
        let err = manager
            .to_json()
            .await
            .expect_err("unconfigured to_json should error");
        assert!(matches!(err, ConfigError::NotSet));
    }
}
