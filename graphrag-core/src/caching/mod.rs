//! Transparent LLM response cache built on `moka` (TTL, LRU/LFU, statistics).
//!
//! Wrap any [`LanguageModel`](crate::core::traits::LanguageModel) implementation
//! with [`client::CachedLLMClient`] to deduplicate repeated prompts.

pub mod cache_config;
pub mod cache_key;
pub mod client;
pub mod distributed;
pub mod stats;
pub mod warming;

pub use cache_config::{CacheConfig, CacheConfigBuilder, EvictionPolicy};
pub use cache_key::{CacheKey, CacheKeyGenerator};
pub use client::CachedLLMClient;
pub use distributed::{DistributedCache, DistributedCacheStats, L1Cache};
pub use stats::{CacheHealth, CacheMetrics, CacheStatistics};
pub use warming::{CacheWarmer, WarmingConfig, WarmingStrategy};

use crate::core::GraphRAGError;

/// Re-export the LanguageModel trait for convenience
pub use crate::core::traits::{GenerationParams, LanguageModel, ModelInfo};

/// Cache-specific error types
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    /// Failed to initialize the cache system or underlying storage
    #[error("Cache initialization failed: {0}")]
    InitializationFailed(String),

    /// Failed to generate a valid cache key from the input parameters
    #[error("Cache key generation failed: {0}")]
    KeyGenerationFailed(String),

    /// A cache operation (get, set, invalidate, etc.) encountered an error
    #[error("Cache operation failed: {0}")]
    OperationFailed(String),

    /// Failed to preload cache entries during the warming phase
    #[error("Cache warming failed: {0}")]
    WarmingFailed(String),

    /// JSON serialization or deserialization of cache entries failed
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Invalid cache configuration parameters provided
    #[error("Configuration error: {0}")]
    Configuration(String),
}

impl From<CacheError> for GraphRAGError {
    fn from(err: CacheError) -> Self {
        GraphRAGError::Generation {
            message: format!("Cache error: {err:?}"),
        }
    }
}

/// Result type for cache operations
pub type CacheResult<T> = std::result::Result<T, CacheError>;

/// Cache entry metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheEntry {
    /// The cached response
    pub response: String,
    /// When this entry was created
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// How many times this entry has been accessed
    pub access_count: u64,
    /// Last access time
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    /// Optional metadata tags
    pub metadata: std::collections::HashMap<String, String>,
}

impl CacheEntry {
    /// Creates a new cache entry with the given response.
    ///
    /// Initializes timestamps to the current time and sets access_count to 1.
    pub fn new(response: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            response,
            created_at: now,
            access_count: 1,
            last_accessed: now,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Records an access to this cache entry.
    ///
    /// Increments the access counter and updates the last_accessed timestamp.
    pub fn access(&mut self) {
        self.access_count += 1;
        self.last_accessed = chrono::Utc::now();
    }

    /// Returns how long this entry has been in the cache.
    ///
    /// Calculates the duration between now and when the entry was created.
    pub fn age(&self) -> chrono::Duration {
        chrono::Utc::now() - self.created_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_entry() {
        let mut entry = CacheEntry::new("test response".to_string());
        assert_eq!(entry.response, "test response");
        assert_eq!(entry.access_count, 1);

        entry.access();
        assert_eq!(entry.access_count, 2);
        assert!(entry.age().num_seconds() >= 0);
    }
}
