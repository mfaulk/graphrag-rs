//! Distributed caching with Redis
//!
//! Implements multi-level caching with Redis for horizontal scaling:
//! - L1: In-memory LRU cache (fastest)
//! - L2: Redis cache (distributed)
//! - L3: Persistent storage (fallback)

use lru::LruCache;
use parking_lot::RwLock;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "redis_storage")]
use redis::{Commands, Connection};

use crate::core::GraphRAGError;

type Result<T> = std::result::Result<T, GraphRAGError>;

/// Cache entry with TTL and access tracking
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    /// The cached value
    pub value: T,
    /// Timestamp when this entry was created
    pub created_at: Instant,
    /// Timestamp when this entry was last accessed
    pub last_accessed: Instant,
    /// Number of times this entry has been accessed
    pub access_count: u64,
    /// Optional time-to-live for automatic expiration
    pub ttl: Option<Duration>,
}

impl<T: Clone> CacheEntry<T> {
    /// Create a new cache entry with the given value and optional TTL
    pub fn new(value: T, ttl: Option<Duration>) -> Self {
        let now = Instant::now();
        Self {
            value,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            ttl,
        }
    }

    /// Check if this entry has expired based on its TTL
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }

    /// Access the entry, updating access time and count, and return a clone of the value
    pub fn access(&mut self) -> T {
        self.last_accessed = Instant::now();
        self.access_count += 1;
        self.value.clone()
    }
}

/// L1 Cache: In-memory LRU cache.
///
/// Backed by `lru::LruCache` for O(1) get/put/eviction. The previous
/// `HashMap` + `min_by_key` scan was O(n) per eviction under the write
/// lock. The `RwLock` is preserved (rather than a `Mutex`) for parity with
/// the rest of the file, but readers cannot share access — every operation
/// (including `get`, which bumps recency) needs a write guard.
pub struct L1Cache<K, V> {
    cache: Arc<RwLock<LruCache<K, CacheEntry<V>>>>,
    max_size: usize,
    default_ttl: Option<Duration>,
}

impl<K, V> L1Cache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    /// Create a new L1 (in-memory) cache with the given maximum size and default TTL
    pub fn new(max_size: usize, default_ttl: Option<Duration>) -> Self {
        // LruCache requires NonZeroUsize; clamp 0 → 1 so a misconfigured
        // max_size doesn't panic.
        let cap = NonZeroUsize::new(max_size.max(1)).expect("max(1) is non-zero");
        Self {
            cache: Arc::new(RwLock::new(LruCache::new(cap))),
            max_size,
            default_ttl,
        }
    }

    /// Get a value from the cache, returning None if not found or expired.
    ///
    /// Resolves the entry once under a single write-lock borrow: a single
    /// `get_mut` both promotes recency and lets us read TTL/value, so there
    /// is no window between the expiry check and the access in which the
    /// entry could become stale. If expired, we pop and return None. The
    /// previous peek-then-get_mut shape worked but did two hash lookups
    /// per call and is more fragile if the lock scope is ever loosened.
    pub fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.write();
        // Single lookup: `get_mut` both finds the entry and bumps it to MRU.
        let entry = cache.get_mut(key)?;
        if entry.is_expired() {
            // End the mutable borrow before mutating the cache via `pop`.
            cache.pop(key);
            return None;
        }
        Some(entry.access())
    }

    /// Put a value into the cache, evicting the LRU entry if at capacity
    pub fn put(&self, key: K, value: V) {
        let mut cache = self.cache.write();
        // `LruCache::put` handles capacity-driven eviction internally.
        cache.put(key, CacheEntry::new(value, self.default_ttl));
    }

    /// Invalidate (remove) a specific entry from the cache
    pub fn invalidate(&self, key: &K) {
        self.cache.write().pop(key);
    }

    /// Clear all entries from the cache
    pub fn clear(&self) {
        self.cache.write().clear();
    }

    /// Get the current number of entries in the cache
    pub fn size(&self) -> usize {
        self.cache.read().len()
    }

    /// Get cache statistics including size, capacity, and access count
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.read();
        let total_accesses: u64 = cache.iter().map(|(_, e)| e.access_count).sum();
        CacheStats {
            size: cache.len(),
            capacity: self.max_size,
            total_accesses,
        }
    }
}

/// L2 Cache: Redis distributed cache
#[cfg(feature = "redis_storage")]
pub struct L2Cache {
    client: redis::Client,
    key_prefix: String,
    default_ttl: Option<Duration>,
}

#[cfg(feature = "redis_storage")]
impl L2Cache {
    /// Create a new L2 (Redis) cache with the given connection URL, key prefix, and default TTL
    pub fn new(url: &str, key_prefix: String, default_ttl: Option<Duration>) -> Result<Self> {
        let client = redis::Client::open(url).map_err(|e| GraphRAGError::Storage {
            message: format!("Failed to connect to Redis: {}", e),
        })?;

        Ok(Self {
            client,
            key_prefix,
            default_ttl,
        })
    }

    /// Generate a prefixed key for Redis storage
    fn prefixed_key(&self, key: &str) -> String {
        format!("{}:{}", self.key_prefix, key)
    }

    /// Get a value from Redis cache by key
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let mut conn = self.get_connection()?;
        let prefixed = self.prefixed_key(key);

        conn.get(&prefixed).map_err(|e| GraphRAGError::Storage {
            message: format!("Redis GET failed: {}", e),
        })
    }

    /// Put a value into Redis cache with optional TTL
    pub fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        let mut conn = self.get_connection()?;
        let prefixed = self.prefixed_key(key);

        if let Some(ttl) = self.default_ttl {
            conn.set_ex::<_, _, ()>(&prefixed, value, ttl.as_secs())
                .map_err(|e| GraphRAGError::Storage {
                    message: format!("Redis SETEX failed: {}", e),
                })?;
        } else {
            conn.set::<_, _, ()>(&prefixed, value)
                .map_err(|e| GraphRAGError::Storage {
                    message: format!("Redis SET failed: {}", e),
                })?;
        }

        Ok(())
    }

    /// Invalidate (remove) a specific entry from Redis cache
    pub fn invalidate(&self, key: &str) -> Result<()> {
        let mut conn = self.get_connection()?;
        let prefixed = self.prefixed_key(key);

        conn.del::<_, ()>(&prefixed)
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Redis DEL failed: {}", e),
            })?;

        Ok(())
    }

    /// Clear all entries with the configured key prefix from Redis cache
    pub fn clear(&self) -> Result<()> {
        let mut conn = self.get_connection()?;
        let pattern = format!("{}:*", self.key_prefix);

        // Get all keys matching pattern
        let keys: Vec<String> = conn.keys(&pattern).map_err(|e| GraphRAGError::Storage {
            message: format!("Redis KEYS failed: {}", e),
        })?;

        // Delete all keys
        if !keys.is_empty() {
            conn.del::<_, ()>(&keys)
                .map_err(|e| GraphRAGError::Storage {
                    message: format!("Redis DEL failed: {}", e),
                })?;
        }

        Ok(())
    }

    /// Get a connection to the Redis server
    fn get_connection(&self) -> Result<Connection> {
        self.client
            .get_connection()
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to get Redis connection: {}", e),
            })
    }
}

/// Multi-level cache combining L1 (memory) and L2 (Redis)
pub struct DistributedCache<K, V>
where
    K: Eq + std::hash::Hash + Clone + ToString,
    V: Clone + serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    l1: L1Cache<K, V>,
    #[cfg(feature = "redis_storage")]
    #[allow(dead_code)]
    l2: Option<L2Cache>,
    #[cfg(not(feature = "redis_storage"))]
    #[allow(dead_code)]
    l2: Option<()>,
    stats: Arc<AtomicCacheStats>,
}

/// Lock-free counters for `DistributedCache` get/put hot paths.
///
/// Replaces the previous `RwLock<DistributedCacheStats>` so concurrent
/// gets and puts no longer serialize on a write-lock just to bump
/// counters — counters are now plain atomics.
///
/// Scope note (post-merge review): this only removes the *stats* lock.
/// L1 hits still serialize on `L1Cache::cache.write()` because the
/// underlying `lru::LruCache` requires `&mut` to promote an entry to
/// most-recently-used (see `L1Cache::get`). Eliminating that
/// serialization is a separate follow-up — e.g., switching the L1
/// backend to a concurrent LRU like `moka` — and is out of scope for
/// this change.
#[derive(Debug, Default)]
struct AtomicCacheStats {
    l1_hits: AtomicU64,
    l1_misses: AtomicU64,
    l2_hits: AtomicU64,
    l2_misses: AtomicU64,
    l2_deserialize_failures: AtomicU64,
}

impl<K, V> DistributedCache<K, V>
where
    K: Eq + std::hash::Hash + Clone + ToString,
    V: Clone + serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    /// Create a new distributed cache with L1 (memory) and optional L2 (Redis) tiers
    pub fn new(
        l1_size: usize,
        l1_ttl: Option<Duration>,
        #[cfg(feature = "redis_storage")] redis_url: Option<&str>,
        #[cfg(not(feature = "redis_storage"))] _redis_url: Option<&str>,
        _l2_ttl: Option<Duration>,
    ) -> Result<Self> {
        let l1 = L1Cache::new(l1_size, l1_ttl);

        #[cfg(feature = "redis_storage")]
        let l2 = if let Some(url) = redis_url {
            Some(L2Cache::new(url, "graphrag".to_string(), _l2_ttl)?)
        } else {
            None
        };

        #[cfg(not(feature = "redis_storage"))]
        let l2 = None;

        Ok(Self {
            l1,
            l2,
            stats: Arc::new(AtomicCacheStats::default()),
        })
    }

    /// Get value from cache (checks L1 then L2)
    pub fn get(&self, key: &K) -> Option<V> {
        // Try L1 first
        if let Some(value) = self.l1.get(key) {
            self.stats.l1_hits.fetch_add(1, Ordering::Relaxed);
            return Some(value);
        }

        self.stats.l1_misses.fetch_add(1, Ordering::Relaxed);

        // Try L2 (Redis) if available
        #[cfg(feature = "redis_storage")]
        if let Some(l2) = &self.l2 {
            match l2.get(&key.to_string()) {
                Ok(Some(bytes)) => {
                    // Bytes present — distinguish a successful deserialize
                    // (true L2 hit) from a deserialize failure (corrupt
                    // payload / serialization-format drift). The previous
                    // `if let Ok(value) = ...` collapsed these into a plain
                    // miss, masking format drift in metrics. (#23)
                    match Self::try_deserialize(&bytes) {
                        Ok(value) => {
                            self.stats.l2_hits.fetch_add(1, Ordering::Relaxed);
                            self.l1.put(key.clone(), value.clone());
                            return Some(value);
                        },
                        Err(_e) => {
                            self.stats
                                .l2_deserialize_failures
                                .fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "tracing")]
                            tracing::warn!(
                                key = %key.to_string(),
                                error = %_e,
                                "L2 cache: bytes present but deserialize failed — \
                                 treating as miss. Likely format drift or corruption."
                            );
                        },
                    }
                },
                Ok(None) => {
                    self.stats.l2_misses.fetch_add(1, Ordering::Relaxed);
                },
                Err(_e) => {
                    // Connection / transport error against Redis. Count as
                    // a miss but log so the operator sees it.
                    self.stats.l2_misses.fetch_add(1, Ordering::Relaxed);
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        key = %key.to_string(),
                        error = %_e,
                        "L2 cache GET failed (Redis transport error)"
                    );
                },
            }
        }

        None
    }

    /// Attempt to deserialize a bincode-encoded cache payload.
    ///
    /// Extracted so tests can exercise the deserialize-failure path
    /// without spinning up Redis.
    #[cfg(feature = "redis_storage")]
    fn try_deserialize(bytes: &[u8]) -> std::result::Result<V, bincode::Error> {
        bincode::deserialize::<V>(bytes)
    }

    /// Put value into cache (writes to both L1 and L2)
    pub fn put(&self, key: K, value: V) -> Result<()> {
        // Write to L1
        self.l1.put(key.clone(), value.clone());

        // Write to L2 (Redis) if available
        #[cfg(feature = "redis_storage")]
        if let Some(l2) = &self.l2 {
            let bytes = bincode::serialize(&value).map_err(|e| GraphRAGError::Storage {
                message: format!("Serialization failed: {}", e),
            })?;
            l2.put(&key.to_string(), &bytes)?;
        }

        Ok(())
    }

    /// Invalidate key from all cache levels
    pub fn invalidate(&self, key: &K) -> Result<()> {
        self.l1.invalidate(key);

        #[cfg(feature = "redis_storage")]
        if let Some(l2) = &self.l2 {
            l2.invalidate(&key.to_string())?;
        }

        Ok(())
    }

    /// Clear all cache levels
    pub fn clear(&self) -> Result<()> {
        self.l1.clear();

        #[cfg(feature = "redis_storage")]
        if let Some(l2) = &self.l2 {
            l2.clear()?;
        }

        Ok(())
    }

    /// Get comprehensive cache statistics
    pub fn stats(&self) -> DistributedCacheStats {
        let l1_stats = self.l1.stats();
        DistributedCacheStats {
            l1_hits: self.stats.l1_hits.load(Ordering::Relaxed),
            l1_misses: self.stats.l1_misses.load(Ordering::Relaxed),
            l1_size: l1_stats.size,
            l1_capacity: l1_stats.capacity,
            l2_hits: self.stats.l2_hits.load(Ordering::Relaxed),
            l2_misses: self.stats.l2_misses.load(Ordering::Relaxed),
            l2_deserialize_failures: self.stats.l2_deserialize_failures.load(Ordering::Relaxed),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Current number of entries in the cache
    pub size: usize,
    /// Maximum capacity of the cache
    pub capacity: usize,
    /// Total number of accesses across all entries
    pub total_accesses: u64,
}

/// Distributed cache statistics
#[derive(Debug, Clone, Default)]
pub struct DistributedCacheStats {
    /// Number of cache hits in L1 (in-memory) cache
    pub l1_hits: u64,
    /// Number of cache misses in L1 (in-memory) cache
    pub l1_misses: u64,
    /// Current size of L1 cache
    pub l1_size: usize,
    /// Maximum capacity of L1 cache
    pub l1_capacity: usize,
    /// Number of cache hits in L2 (Redis) cache
    pub l2_hits: u64,
    /// Number of cache misses in L2 (Redis) cache (key absent)
    pub l2_misses: u64,
    /// L2 lookups that returned bytes but failed to deserialize. Counted
    /// separately from `l2_misses` so format drift / corruption shows up
    /// in metrics instead of disguised as a normal miss. (#23)
    pub l2_deserialize_failures: u64,
}

impl DistributedCacheStats {
    /// Calculate the overall cache hit rate across both L1 and L2.
    ///
    /// `l2_deserialize_failures` count toward the denominator: a deserialize
    /// failure is a request that did not return cached data, so excluding it
    /// would inflate the reported hit rate and mask cache-health regressions
    /// caused by format drift or corruption (#23).
    pub fn hit_rate(&self) -> f64 {
        let total_hits = self.l1_hits + self.l2_hits;
        let total_requests =
            total_hits + self.l1_misses + self.l2_misses + self.l2_deserialize_failures;
        if total_requests == 0 {
            0.0
        } else {
            total_hits as f64 / total_requests as f64
        }
    }

    /// Calculate the L1 cache hit rate
    pub fn l1_hit_rate(&self) -> f64 {
        let total = self.l1_hits + self.l1_misses;
        if total == 0 {
            0.0
        } else {
            self.l1_hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_cache() {
        let cache = L1Cache::new(3, Some(Duration::from_secs(60)));

        cache.put("key1", "value1");
        cache.put("key2", "value2");
        cache.put("key3", "value3");

        assert_eq!(cache.get(&"key1"), Some("value1"));
        assert_eq!(cache.get(&"key2"), Some("value2"));
        assert_eq!(cache.size(), 3);

        // Test eviction
        cache.put("key4", "value4");
        assert_eq!(cache.size(), 3);
    }

    #[test]
    fn test_cache_entry_expiration() {
        let entry = CacheEntry::new("value", Some(Duration::from_millis(10)));
        assert!(!entry.is_expired());

        std::thread::sleep(Duration::from_millis(15));
        assert!(entry.is_expired());
    }

    // Regression for #106 review: L1Cache::get must check expiry under the
    // same write-lock borrow as the recency promotion. A peek-then-get_mut
    // split (or any code path that re-resolves the entry between TTL check
    // and access) leaves a window where an entry that was live during peek
    // can expire before access. Inserting with a 1ms TTL and reading after
    // 5ms must always return None — never the stale value.
    #[test]
    fn l1_cache_get_does_not_return_expired_entry() {
        let cache: L1Cache<&'static str, &'static str> =
            L1Cache::new(4, Some(Duration::from_millis(1)));

        cache.put("k", "v");
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(
            cache.get(&"k"),
            None,
            "an entry past its TTL must never be returned"
        );
    }

    // Inserting beyond max_size evicts the least-recently-used entry, not
    // an arbitrary one — guards against regression to a non-LRU eviction.
    #[test]
    fn l1_cache_evicts_least_recently_used_entry() {
        let cache: L1Cache<&'static str, &'static str> =
            L1Cache::new(2, Some(Duration::from_secs(60)));

        cache.put("a", "1");
        cache.put("b", "2");
        // Insert a third; "a" is LRU and should be evicted.
        cache.put("c", "3");

        assert_eq!(cache.size(), 2);
        assert_eq!(cache.get(&"a"), None, "a was LRU and should be evicted");
        assert_eq!(cache.get(&"b"), Some("2"));
        assert_eq!(cache.get(&"c"), Some("3"));
    }

    // Calling get() on an existing entry must mark it as most-recently-used
    // so a subsequent capacity-triggered eviction targets the next-oldest,
    // not the just-touched key.
    #[test]
    fn l1_cache_get_marks_entry_as_most_recently_used() {
        let cache: L1Cache<&'static str, &'static str> =
            L1Cache::new(2, Some(Duration::from_secs(60)));

        cache.put("a", "1");
        cache.put("b", "2");
        // Touch "a"; now "b" is the LRU.
        assert_eq!(cache.get(&"a"), Some("1"));
        cache.put("c", "3");

        assert_eq!(cache.get(&"a"), Some("1"), "a was touched, must survive");
        assert_eq!(cache.get(&"b"), None, "b became LRU and should be evicted");
        assert_eq!(cache.get(&"c"), Some("3"));
    }

    // Regression for #23: bincode garbage in the L2 byte stream must surface
    // as a deserialize failure (caller can count it separately from miss).
    #[cfg(feature = "redis_storage")]
    #[test]
    fn try_deserialize_returns_err_on_garbage_bytes() {
        // We never deserialize an `i32` from a single 0xff byte successfully.
        let result: std::result::Result<i32, _> =
            DistributedCache::<String, i32>::try_deserialize(&[0xff]);
        assert!(
            result.is_err(),
            "deserialize must fail loudly on corrupt bytes so callers can \
             distinguish miss from format drift"
        );
    }

    // The new `l2_deserialize_failures` counter starts at zero and is
    // distinct from `l2_misses` so dashboards can split the two signals.
    #[test]
    fn distributed_cache_stats_separates_misses_from_deserialize_failures() {
        let stats = DistributedCacheStats::default();
        assert_eq!(stats.l2_misses, 0);
        assert_eq!(stats.l2_deserialize_failures, 0);
    }

    // hit_rate() must count L2 deserialize failures as non-hits in its
    // denominator; otherwise a wave of corruption-induced failures would
    // leave the reported hit rate unchanged and hide the regression (#23).
    #[test]
    fn hit_rate_counts_deserialize_failures_as_non_hits() {
        let baseline = DistributedCacheStats {
            l1_hits: 1,
            l1_misses: 0,
            l1_size: 0,
            l1_capacity: 0,
            l2_hits: 0,
            l2_misses: 1,
            l2_deserialize_failures: 0,
        };
        let with_failure = DistributedCacheStats {
            l2_deserialize_failures: 1,
            ..baseline.clone()
        };

        assert!(
            with_failure.hit_rate() < baseline.hit_rate(),
            "deserialize failures must lower the hit rate (baseline {}, with failure {})",
            baseline.hit_rate(),
            with_failure.hit_rate()
        );
        // 1 hit / (1 hit + 1 miss + 1 deserialize failure) = 1/3.
        assert!((with_failure.hit_rate() - (1.0 / 3.0)).abs() < f64::EPSILON);
    }

    // Concurrent gets across many threads must produce an exact l1_hits
    // count in the snapshot — atomic counters, not lock-protected, must not
    // lose increments.
    #[test]
    fn stats_l1_hits_accumulate_under_concurrent_gets() {
        let cache: Arc<DistributedCache<String, String>> = Arc::new(
            DistributedCache::new(
                32,
                Some(Duration::from_secs(60)),
                #[cfg(feature = "redis_storage")]
                None,
                #[cfg(not(feature = "redis_storage"))]
                None,
                None,
            )
            .expect("cache construction"),
        );

        cache
            .put("k".to_string(), "v".to_string())
            .expect("put succeeds");

        const THREADS: usize = 8;
        const GETS_PER_THREAD: u64 = 250;

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let c = Arc::clone(&cache);
                std::thread::spawn(move || {
                    for _ in 0..GETS_PER_THREAD {
                        assert_eq!(c.get(&"k".to_string()), Some("v".to_string()));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("worker panicked");
        }

        let snap = cache.stats();
        assert_eq!(snap.l1_hits, THREADS as u64 * GETS_PER_THREAD);
        assert_eq!(snap.l1_misses, 0);
    }

    // Snapshot returned by stats() reflects the current atomic counters
    // and is a plain (non-atomic) struct safe to clone and inspect.
    #[test]
    fn stats_snapshot_reflects_accumulated_counts() {
        let cache: DistributedCache<String, String> = DistributedCache::new(
            8,
            Some(Duration::from_secs(60)),
            #[cfg(feature = "redis_storage")]
            None,
            #[cfg(not(feature = "redis_storage"))]
            None,
            None,
        )
        .expect("cache construction");

        cache
            .put("present".to_string(), "v".to_string())
            .expect("put succeeds");
        let _ = cache.get(&"present".to_string()); // hit
        let _ = cache.get(&"absent".to_string()); // miss
        let _ = cache.get(&"present".to_string()); // hit

        let snap = cache.stats();
        assert_eq!(snap.l1_hits, 2);
        assert_eq!(snap.l1_misses, 1);
    }
}
