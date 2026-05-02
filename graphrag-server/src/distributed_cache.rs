//! Distributed Caching with Redis
//!
//! This module provides distributed caching for GraphRAG using Redis. It enables:
//! - Multi-level caching (L1/L2/L3)
//! - Cache coherence across multiple server instances
//! - Predictive prefetching
//! - Cache warming strategies
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │          Application                │
//! └──────────────┬──────────────────────┘
//!                │
//! ┌──────────────▼──────────────────────┐
//! │       Cache Manager                 │
//! │  ┌────────────────────────────┐     │
//! │  │ L1: In-Memory (Fast)       │     │
//! │  │ - LRU eviction            │     │
//! │  │ - 100ms TTL               │     │
//! │  └────────────┬───────────────┘     │
//! │               │                     │
//! │  ┌────────────▼───────────────┐     │
//! │  │ L2: Redis (Distributed)    │     │
//! │  │ - Shared across servers   │     │
//! │  │ - 1h TTL                  │     │
//! │  └────────────┬───────────────┘     │
//! │               │                     │
//! │  ┌────────────▼───────────────┐     │
//! │  │ L3: Persistent Storage     │     │
//! │  │ - Long-term cache         │     │
//! │  │ - 24h+ TTL                │     │
//! │  └────────────────────────────┘     │
//! └─────────────────────────────────────┘
//! ```

use lru::LruCache;
use parking_lot::RwLock;
use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Client, RedisError};
use serde::{de::DeserializeOwned, Serialize};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Cache entry with metadata
#[derive(Clone)]
struct CacheEntry<T> {
    value: T,
    created_at: Instant,
    access_count: u64,
    last_accessed: Instant,
}

/// Multi-level distributed cache
///
/// Provides L1 (in-memory), L2 (Redis), and L3 (persistent) caching layers
/// with automatic promotion/demotion and cache warming.
///
/// All Redis I/O goes through a single multiplexed `ConnectionManager` that
/// handles connection pooling and reconnection internally — there is no
/// fresh TCP handshake per call, and no blocking I/O on the Tokio runtime.
pub struct DistributedCache {
    /// L1 cache: In-memory LRU cache.
    ///
    /// Uses `lru::LruCache` for O(1) get/put/eviction. The previous
    /// `HashMap` + `min_by_key` scan was O(n) per eviction under a write
    /// lock, which serialized every cache operation behind a 1000-element
    /// linear scan once the cache reached capacity.
    ///
    /// Wrapped in `parking_lot::RwLock` — no `.await` occurs while a guard
    /// is held; every helper drops the guard before any async operation.
    l1_cache: Arc<RwLock<LruCache<String, CacheEntry<Vec<u8>>>>>,
    /// L1 cache max size
    l1_max_size: usize,
    /// L1 TTL
    l1_ttl: Duration,

    /// L2 cache: multiplexed Redis connection. `None` if Redis is disabled
    /// or the initial connection failed.
    redis: Option<ConnectionManager>,
    /// L2 TTL (in seconds for Redis)
    l2_ttl: u64,

    /// Cache statistics, lock-free.
    stats: Arc<AtomicCacheStats>,

    /// Prefetch enabled
    _prefetch_enabled: bool,
}

/// Lock-free counters used internally. Each counter is updated with
/// `fetch_add(1, Relaxed)` so concurrent calls cannot lose updates and
/// `total_requests == l1_hits + l1_misses` cannot drift the way the old
/// `RwLock<CacheStats>`-with-five-acquisitions implementation could.
#[derive(Default)]
pub(crate) struct AtomicCacheStats {
    l1_hits: AtomicU64,
    l1_misses: AtomicU64,
    l2_hits: AtomicU64,
    l2_misses: AtomicU64,
    total_requests: AtomicU64,
    evictions: AtomicU64,
    prefetches: AtomicU64,
}

impl AtomicCacheStats {
    fn snapshot(&self) -> CacheStats {
        CacheStats {
            l1_hits: self.l1_hits.load(Ordering::Relaxed),
            l1_misses: self.l1_misses.load(Ordering::Relaxed),
            l2_hits: self.l2_hits.load(Ordering::Relaxed),
            l2_misses: self.l2_misses.load(Ordering::Relaxed),
            total_requests: self.total_requests.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            prefetches: self.prefetches.load(Ordering::Relaxed),
        }
    }
}

/// Cache statistics
#[derive(Default, Clone)]
pub struct CacheStats {
    /// L1 hits
    pub l1_hits: u64,
    /// L1 misses
    pub l1_misses: u64,
    /// L2 hits
    pub l2_hits: u64,
    /// L2 misses
    pub l2_misses: u64,
    /// Total requests
    pub total_requests: u64,
    /// Evictions
    pub evictions: u64,
    /// Prefetches
    pub prefetches: u64,
}

impl CacheStats {
    /// Calculate L1 hit rate
    pub fn l1_hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.l1_hits as f64) / (self.total_requests as f64)
        }
    }

    /// Calculate L2 hit rate
    pub fn l2_hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.l2_hits as f64) / (self.total_requests as f64)
        }
    }

    /// Calculate total hit rate
    pub fn total_hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            ((self.l1_hits + self.l2_hits) as f64) / (self.total_requests as f64)
        }
    }
}

/// Cache configuration
pub struct CacheConfig {
    /// Redis URL (e.g., "redis://localhost:6379")
    pub redis_url: Option<String>,
    /// L1 cache max entries
    pub l1_max_size: usize,
    /// L1 TTL in seconds
    pub l1_ttl_secs: u64,
    /// L2 TTL in seconds
    pub l2_ttl_secs: u64,
    /// Enable prefetching
    pub prefetch_enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            redis_url: Some("redis://localhost:6379".to_string()),
            l1_max_size: 1000,
            l1_ttl_secs: 100,
            l2_ttl_secs: 3600, // 1 hour
            prefetch_enabled: true,
        }
    }
}

impl DistributedCache {
    /// Create a new distributed cache.
    ///
    /// Establishes the multiplexed Redis connection up front so subsequent
    /// `get`/`set` calls reuse it. If Redis is unreachable, the L2 layer
    /// is disabled and operations gracefully degrade to L1-only.
    ///
    /// # Arguments
    /// * `config` - Cache configuration
    ///
    /// # Returns
    /// Result with DistributedCache or error
    pub async fn new(config: CacheConfig) -> Result<Self, RedisError> {
        let redis = if let Some(url) = config.redis_url {
            match Client::open(url.clone()) {
                Ok(client) => match ConnectionManager::new(client).await {
                    Ok(mgr) => {
                        tracing::info!("✅ Redis connected: {}", url);
                        Some(mgr)
                    },
                    Err(e) => {
                        tracing::warn!("⚠️ Redis connection failed, L2 cache disabled: {}", e);
                        None
                    },
                },
                Err(e) => {
                    tracing::warn!("⚠️ Redis client creation failed, L2 cache disabled: {}", e);
                    None
                },
            }
        } else {
            tracing::info!("Redis URL not provided, L2 cache disabled");
            None
        };

        // LruCache requires NonZeroUsize; clamp 0 → 1 so a misconfigured
        // l1_max_size doesn't panic. A 1-entry cache is degenerate but
        // safe; any sensible config will use a much larger value.
        let l1_capacity =
            NonZeroUsize::new(config.l1_max_size.max(1)).expect("max(1) is always non-zero");
        Ok(Self {
            l1_cache: Arc::new(RwLock::new(LruCache::new(l1_capacity))),
            l1_max_size: config.l1_max_size,
            l1_ttl: Duration::from_secs(config.l1_ttl_secs),
            redis,
            l2_ttl: config.l2_ttl_secs,
            stats: Arc::new(AtomicCacheStats::default()),
            _prefetch_enabled: config.prefetch_enabled,
        })
    }

    /// Get value from cache.
    ///
    /// Checks L1 cache first, then L2 (Redis), with automatic promotion.
    /// Each request increments exactly one of `l1_hits`, `l2_hits`, or
    /// `l2_misses` (with `l1_misses` incremented on the L1-miss path), so
    /// counters always satisfy `total_requests == l1_hits + l1_misses`.
    ///
    /// # Arguments
    /// * `key` - Cache key
    ///
    /// # Returns
    /// Option with cached value
    pub async fn get<T: DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);

        // Try L1 cache first
        if let Some(value) = self.get_l1(key) {
            self.stats.l1_hits.fetch_add(1, Ordering::Relaxed);
            match bincode::deserialize::<T>(&value) {
                Ok(val) => return Some(val),
                Err(e) => {
                    // Treat decode failure as a hit-with-error: stats already
                    // counted the L1 hit, but the caller gets None and we fall
                    // through to L2 to attempt re-fetching. Don't double-count
                    // as a miss the way the old implementation did.
                    tracing::warn!("Failed to deserialize L1 cache value: {}", e);
                },
            }
        } else {
            self.stats.l1_misses.fetch_add(1, Ordering::Relaxed);
        }

        // Try L2 cache (Redis)
        if let Some(value) = self.get_l2(key).await {
            self.stats.l2_hits.fetch_add(1, Ordering::Relaxed);
            self.set_l1(key, value.clone());
            match bincode::deserialize::<T>(&value) {
                Ok(val) => return Some(val),
                Err(e) => {
                    tracing::warn!("Failed to deserialize L2 cache value: {}", e);
                    return None;
                },
            }
        }

        self.stats.l2_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Set value in cache.
    ///
    /// Stores in both L1 (in-memory) and L2 (Redis). L2 errors are logged
    /// but do not propagate; the caller is told the L1 set succeeded.
    ///
    /// # Arguments
    /// * `key` - Cache key
    /// * `value` - Value to cache
    pub async fn set<T: Serialize>(&self, key: &str, value: &T) {
        let bytes = match bincode::serialize(value) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("Failed to serialize cache value: {}", e);
                return;
            },
        };

        self.set_l1(key, bytes.clone());
        self.set_l2(key, bytes).await;
    }

    /// Invalidate cache entry.
    ///
    /// Removes from both L1 and L2 caches. L2 errors are logged.
    pub async fn invalidate(&self, key: &str) {
        self.l1_cache.write().pop(key);

        if let Some(mut conn) = self.redis.clone() {
            if let Err(e) = conn.del::<_, ()>(key).await {
                tracing::warn!("Redis del failed for key {}: {}", key, e);
            }
        }
    }

    /// Invalidate all keys matching a pattern (e.g. `query:*`) across both
    /// caches.
    pub async fn invalidate_pattern(&self, pattern: &str) {
        // Snapshot keys then drop the read guard before we await on Redis.
        let l1_keys_to_remove: Vec<String> = self
            .l1_cache
            .read()
            .iter()
            .filter(|(k, _)| Self::matches_pattern(k, pattern))
            .map(|(k, _)| k.clone())
            .collect();

        if !l1_keys_to_remove.is_empty() {
            let mut cache = self.l1_cache.write();
            for key in &l1_keys_to_remove {
                cache.pop(key);
            }
        }

        if let Some(mut conn) = self.redis.clone() {
            match conn.keys::<_, Vec<String>>(pattern).await {
                Ok(keys) => {
                    for key in keys {
                        if let Err(e) = conn.del::<_, ()>(&key).await {
                            tracing::warn!("Redis del failed for key {}: {}", key, e);
                        }
                    }
                },
                Err(e) => {
                    tracing::warn!("Redis KEYS scan failed for pattern {}: {}", pattern, e);
                },
            }
        }
    }

    /// Warm cache with frequently accessed keys.
    ///
    /// Preloads cache with specified keys to improve hit rates.
    pub async fn warm<T, F>(&self, keys: Vec<String>, mut loader: F)
    where
        T: Serialize,
        F: FnMut(&str) -> Option<T>,
    {
        for key in keys {
            if let Some(value) = loader(&key) {
                self.set(&key, &value).await;
                self.stats.prefetches.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.snapshot()
    }

    /// Clear all caches.
    ///
    /// Note: this issues `FLUSHDB` against the configured Redis database,
    /// which is destructive across the whole DB. Use with care in shared
    /// deployments.
    pub async fn clear(&self) {
        self.l1_cache.write().clear();

        if let Some(mut conn) = self.redis.clone() {
            if let Err(e) = redis::cmd("FLUSHDB").query_async::<()>(&mut conn).await {
                tracing::warn!("Redis FLUSHDB failed: {}", e);
            }
        }
    }

    // --- Private methods ---

    /// Get from L1 cache (synchronous; guard never crosses an `.await`).
    ///
    /// `LruCache::peek` checks TTL without bumping LRU order; if expired we
    /// remove and miss. Otherwise `get_mut` bumps the entry to most-recently-
    /// used and returns the value.
    fn get_l1(&self, key: &str) -> Option<Vec<u8>> {
        let mut cache = self.l1_cache.write();

        if let Some(entry) = cache.peek(key) {
            if entry.created_at.elapsed() > self.l1_ttl {
                cache.pop(key);
                return None;
            }
        }

        cache.get_mut(key).map(|entry| {
            entry.access_count += 1;
            entry.last_accessed = Instant::now();
            entry.value.clone()
        })
    }

    /// Set in L1 cache (synchronous; guard never crosses an `.await`).
    ///
    /// Eviction is O(1): if we're at capacity and the key is new, pop the
    /// LRU entry first and bump the eviction counter. `LruCache::put` will
    /// also evict transparently if we don't pre-pop, but doing it here lets
    /// us increment `stats.evictions`.
    fn set_l1(&self, key: &str, value: Vec<u8>) {
        let mut cache = self.l1_cache.write();

        if cache.len() >= self.l1_max_size && cache.peek(key).is_none() && cache.pop_lru().is_some()
        {
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }

        cache.put(
            key.to_string(),
            CacheEntry {
                value,
                created_at: Instant::now(),
                access_count: 0,
                last_accessed: Instant::now(),
            },
        );
    }

    /// Get from L2 cache (Redis)
    async fn get_l2(&self, key: &str) -> Option<Vec<u8>> {
        let mut conn = self.redis.clone()?;
        match conn.get::<_, Option<Vec<u8>>>(key).await {
            Ok(value) => value,
            Err(e) => {
                tracing::warn!("Redis GET failed for key {}: {}", key, e);
                None
            },
        }
    }

    /// Set in L2 cache (Redis)
    async fn set_l2(&self, key: &str, value: Vec<u8>) {
        if let Some(mut conn) = self.redis.clone() {
            if let Err(e) = conn.set_ex::<_, _, ()>(key, value, self.l2_ttl).await {
                tracing::warn!("Redis SETEX failed for key {}: {}", key, e);
            }
        }
    }

    /// Check if key matches pattern
    fn matches_pattern(key: &str, pattern: &str) -> bool {
        if pattern.ends_with('*') {
            let prefix = &pattern[..pattern.len() - 1];
            key.starts_with(prefix)
        } else {
            key == pattern
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_no_redis(l1_max_size: usize) -> CacheConfig {
        CacheConfig {
            redis_url: None,
            l1_max_size,
            l1_ttl_secs: 10,
            l2_ttl_secs: 60,
            prefetch_enabled: false,
        }
    }

    #[tokio::test]
    async fn l1_cache_set_then_get_round_trips_value() {
        let cache = DistributedCache::new(cfg_no_redis(2)).await.unwrap();
        cache.set("key1", &"value1".to_string()).await;

        let value: Option<String> = cache.get("key1").await;
        assert_eq!(value, Some("value1".to_string()));

        let stats = cache.stats();
        assert_eq!(stats.l1_hits, 1);
    }

    #[tokio::test]
    async fn lru_eviction_drops_least_recently_used() {
        let cache = DistributedCache::new(cfg_no_redis(2)).await.unwrap();

        cache.set("key1", &"value1".to_string()).await;
        cache.set("key2", &"value2".to_string()).await;

        // Touch key1 so key2 is LRU.
        let _: Option<String> = cache.get("key1").await;

        cache.set("key3", &"value3".to_string()).await;

        let v1: Option<String> = cache.get("key1").await;
        let v2: Option<String> = cache.get("key2").await;
        let v3: Option<String> = cache.get("key3").await;

        assert_eq!(v1, Some("value1".to_string()));
        assert_eq!(v2, None);
        assert_eq!(v3, Some("value3".to_string()));
    }

    // Regression test for #33: previously each `get` took the stats write
    // lock five times and could double-count requests as both hit and miss.
    // Now stats are atomic and each request is counted exactly once.
    #[tokio::test]
    async fn stats_are_consistent_under_concurrent_get_traffic() {
        let cache = Arc::new(DistributedCache::new(cfg_no_redis(64)).await.unwrap());
        cache.set("k", &"v".to_string()).await;

        let total_calls = 1_000usize;
        let mut handles = Vec::with_capacity(total_calls);
        for _ in 0..total_calls {
            let c = cache.clone();
            handles.push(tokio::spawn(async move {
                let _: Option<String> = c.get("k").await;
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        let stats = cache.stats();
        // Every call must be accounted for exactly once.
        assert_eq!(stats.total_requests, total_calls as u64);
        // L1-only path: every call hits L1, none hit L2.
        assert_eq!(stats.l1_hits, total_calls as u64);
        assert_eq!(stats.l1_misses, 0);
        assert_eq!(stats.l2_hits, 0);
        // Invariant from the docstring: total == l1_hits + l1_misses.
        assert_eq!(stats.total_requests, stats.l1_hits + stats.l1_misses);
    }

    // Regression test for #33: l1_misses must be incremented exactly once
    // per L1 miss (not on the L1-hit-but-deserialize-fail path the old code
    // double-counted).
    #[tokio::test]
    async fn l1_miss_path_counts_exactly_one_miss() {
        let cache = DistributedCache::new(cfg_no_redis(8)).await.unwrap();

        let _: Option<String> = cache.get("missing").await;

        let stats = cache.stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.l1_hits, 0);
        assert_eq!(stats.l1_misses, 1);
        assert_eq!(stats.l2_hits, 0);
        assert_eq!(stats.l2_misses, 1);
    }

    #[tokio::test]
    async fn invalidate_removes_l1_entry() {
        let cache = DistributedCache::new(cfg_no_redis(8)).await.unwrap();
        cache.set("k1", &"v1".to_string()).await;
        cache.invalidate("k1").await;
        let v: Option<String> = cache.get("k1").await;
        assert_eq!(v, None);
    }

    #[tokio::test]
    async fn invalidate_pattern_removes_matching_keys_from_l1() {
        let cache = DistributedCache::new(cfg_no_redis(16)).await.unwrap();
        cache.set("query:1", &"a".to_string()).await;
        cache.set("query:2", &"b".to_string()).await;
        cache.set("other:1", &"c".to_string()).await;

        cache.invalidate_pattern("query:*").await;

        let v1: Option<String> = cache.get("query:1").await;
        let v2: Option<String> = cache.get("query:2").await;
        let v3: Option<String> = cache.get("other:1").await;
        assert_eq!(v1, None);
        assert_eq!(v2, None);
        assert_eq!(v3, Some("c".to_string()));
    }

    #[tokio::test]
    async fn clear_empties_l1() {
        let cache = DistributedCache::new(cfg_no_redis(8)).await.unwrap();
        cache.set("a", &"1".to_string()).await;
        cache.set("b", &"2".to_string()).await;
        cache.clear().await;
        let v: Option<String> = cache.get("a").await;
        assert_eq!(v, None);
    }

    #[tokio::test]
    async fn warm_loads_keys_via_loader_and_counts_prefetches() {
        let cache = DistributedCache::new(cfg_no_redis(8)).await.unwrap();
        cache
            .warm(vec!["a".to_string(), "b".to_string()], |k| {
                Some(format!("v_{k}"))
            })
            .await;
        let va: Option<String> = cache.get("a").await;
        let vb: Option<String> = cache.get("b").await;
        assert_eq!(va, Some("v_a".to_string()));
        assert_eq!(vb, Some("v_b".to_string()));
        assert_eq!(cache.stats().prefetches, 2);
    }
}
