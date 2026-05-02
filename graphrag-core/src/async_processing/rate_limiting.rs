//! Rate limiting for API call throttling and concurrency control.
//!
//! This module provides the [`RateLimiter`] for controlling both the concurrency and
//! frequency of API calls to prevent throttling and respect service limits.
//!
//! # Main Types
//!
//! - [`RateLimiter`]: Dual-level rate limiter with semaphore-based concurrency control
//!   and time-based request throttling
//!
//! # Features
//!
//! - Separate rate limiting for LLM and embedding API calls
//! - Semaphore-based concurrency control (max N simultaneous calls)
//! - Time-based rate limiting (max N calls per second)
//! - Automatic waiting when limits are reached
//! - RAII-style permit handling with automatic release
//! - Health checking for congestion detection
//! - Per-second rate window with automatic reset
//!
//! # Rate Limiting Strategy
//!
//! The rate limiter implements a two-tier approach:
//!
//! 1. **Concurrency Control**: Uses semaphores to limit how many API calls can run
//!    simultaneously. This prevents overwhelming the system with too many parallel requests.
//!
//! 2. **Time-Based Rate Limiting**: Tracks requests per second and automatically waits
//!    when the limit is reached. The counter resets every second.
//!
//! # Basic Usage
//!
//! ```rust,ignore
//! use graphrag_core::async_processing::{RateLimiter, AsyncConfig};
//!
//! let config = AsyncConfig {
//!     max_concurrent_llm_calls: 3,
//!     llm_rate_limit_per_second: 2.0,
//!     max_concurrent_embeddings: 5,
//!     embedding_rate_limit_per_second: 10.0,
//!     ..Default::default()
//! };
//!
//! let rate_limiter = RateLimiter::new(&config);
//!
//! // Acquire permit for LLM call (blocks if needed)
//! let permit = rate_limiter.acquire_llm_permit().await?;
//! // ... make LLM API call ...
//! // Permit is automatically released when dropped
//!
//! // Check available capacity
//! let available = rate_limiter.get_available_llm_permits();
//! println!("Available LLM permits: {}", available);
//!
//! // Health check
//! let status = rate_limiter.health_check();
//! ```

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Semaphore, SemaphorePermit};
use tokio::time::{self, Instant};

use super::{AsyncConfig, ComponentStatus};
use crate::core::GraphRAGError;

/// Rate limiter for controlling API call frequency and concurrency
///
/// Provides dual-level throttling: semaphore-based concurrency control and
/// time-based rate limiting for both LLM and embedding API calls.
#[derive(Debug)]
pub struct RateLimiter {
    /// Semaphore limiting concurrent LLM API calls
    llm_semaphore: Arc<Semaphore>,
    /// Semaphore limiting concurrent embedding API calls
    embedding_semaphore: Arc<Semaphore>,
    /// Tracker for LLM API call rate limiting
    llm_rate_tracker: Arc<tokio::sync::Mutex<RateTracker>>,
    /// Tracker for embedding API call rate limiting
    embedding_rate_tracker: Arc<tokio::sync::Mutex<RateTracker>>,
    /// Configuration settings
    config: AsyncConfig,
}

/// Internal tracker for time-based rate limiting
#[derive(Debug)]
struct RateTracker {
    /// Timestamp of the last request
    last_request: Option<Instant>,
    /// Number of requests made in the current second
    requests_this_second: u32,
    /// Maximum requests allowed per second
    rate_limit: f64,
}

impl RateTracker {
    /// Creates a new rate tracker with specified rate limit
    ///
    /// # Parameters
    /// - `rate_limit`: Maximum requests allowed per second
    fn new(rate_limit: f64) -> Self {
        Self {
            last_request: None,
            requests_this_second: 0,
            rate_limit,
        }
    }

    /// Reserve a slot in the rate window and return how long the caller must
    /// sleep before proceeding. Synchronous so callers can drop the owning
    /// mutex guard before awaiting the returned duration (see issue #21).
    fn next_wait(&mut self) -> Duration {
        let now = Instant::now();
        let mut wait = Duration::ZERO;

        if let Some(last_request) = self.last_request {
            // A previously scheduled caller may sit in the future; this caller
            // cannot start before that scheduled time.
            let effective_now = now.max(last_request);
            wait += effective_now.saturating_duration_since(now);

            let time_since_last = effective_now.saturating_duration_since(last_request);
            if time_since_last >= Duration::from_secs(1) {
                self.requests_this_second = 0;
            }

            if (self.requests_this_second as f64) >= self.rate_limit {
                wait += Duration::from_secs(1).saturating_sub(time_since_last);
                self.requests_this_second = 0;
            }
        }

        self.last_request = Some(now + wait);
        self.requests_this_second += 1;
        wait
    }
}

impl RateLimiter {
    /// Creates a new rate limiter from configuration
    ///
    /// Initializes semaphores and rate trackers for both LLM and embedding API calls.
    ///
    /// # Parameters
    /// - `config`: Configuration specifying concurrency and rate limits
    pub fn new(config: &AsyncConfig) -> Self {
        Self {
            llm_semaphore: Arc::new(Semaphore::new(config.max_concurrent_llm_calls)),
            embedding_semaphore: Arc::new(Semaphore::new(config.max_concurrent_embeddings)),
            llm_rate_tracker: Arc::new(tokio::sync::Mutex::new(RateTracker::new(
                config.llm_rate_limit_per_second,
            ))),
            embedding_rate_tracker: Arc::new(tokio::sync::Mutex::new(RateTracker::new(
                config.embedding_rate_limit_per_second,
            ))),
            config: config.clone(),
        }
    }

    /// Acquires a permit for making an LLM API call
    ///
    /// Blocks until both concurrency and rate limits allow the call to proceed.
    /// The permit must be held for the duration of the API call and will be
    /// released when dropped.
    ///
    /// # Returns
    /// Semaphore permit on success, or an error if acquisition fails
    pub async fn acquire_llm_permit(&self) -> Result<SemaphorePermit<'_>, GraphRAGError> {
        // First acquire the semaphore permit for concurrency control
        let permit = self
            .llm_semaphore
            .acquire()
            .await
            .map_err(|e| GraphRAGError::RateLimit {
                message: format!("Failed to acquire LLM permit: {e}"),
            })?;

        // Reserve a rate-limit slot, then drop the guard before sleeping so
        // concurrent callers aren't serialized behind the wait window.
        let wait = {
            let mut rate_tracker = self.llm_rate_tracker.lock().await;
            rate_tracker.next_wait()
        };
        if wait > Duration::ZERO {
            time::sleep(wait).await;
        }

        Ok(permit)
    }

    /// Acquires a permit for making an embedding API call
    ///
    /// Blocks until both concurrency and rate limits allow the call to proceed.
    /// The permit must be held for the duration of the API call and will be
    /// released when dropped.
    ///
    /// # Returns
    /// Semaphore permit on success, or an error if acquisition fails
    pub async fn acquire_embedding_permit(&self) -> Result<SemaphorePermit<'_>, GraphRAGError> {
        // First acquire the semaphore permit for concurrency control
        let permit =
            self.embedding_semaphore
                .acquire()
                .await
                .map_err(|e| GraphRAGError::RateLimit {
                    message: format!("Failed to acquire embedding permit: {e}"),
                })?;

        let wait = {
            let mut rate_tracker = self.embedding_rate_tracker.lock().await;
            rate_tracker.next_wait()
        };
        if wait > Duration::ZERO {
            time::sleep(wait).await;
        }

        Ok(permit)
    }

    /// Returns the number of available LLM permits
    ///
    /// # Returns
    /// Number of LLM API calls that can be made immediately without waiting
    pub fn get_available_llm_permits(&self) -> usize {
        self.llm_semaphore.available_permits()
    }

    /// Returns the number of available embedding permits
    ///
    /// # Returns
    /// Number of embedding API calls that can be made immediately without waiting
    pub fn get_available_embedding_permits(&self) -> usize {
        self.embedding_semaphore.available_permits()
    }

    /// Performs a health check on the rate limiter
    ///
    /// Checks permit availability to determine if the system is healthy or
    /// experiencing congestion.
    ///
    /// # Returns
    /// Component status indicating health (Healthy, Warning, or Error)
    pub fn health_check(&self) -> ComponentStatus {
        let llm_available = self.get_available_llm_permits();
        let embedding_available = self.get_available_embedding_permits();

        if llm_available == 0 && embedding_available == 0 {
            ComponentStatus::Warning("No permits available".to_string())
        } else if llm_available == 0 {
            ComponentStatus::Warning("No LLM permits available".to_string())
        } else if embedding_available == 0 {
            ComponentStatus::Warning("No embedding permits available".to_string())
        } else {
            ComponentStatus::Healthy
        }
    }

    /// Returns the current configuration
    ///
    /// # Returns
    /// Reference to the async processing configuration
    pub fn get_config(&self) -> &AsyncConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression for issue #21: the rate-tracker mutex must not be held while a caller
    // sleeps inside the rate-limit window, or every concurrent caller serializes behind it.
    #[tokio::test(start_paused = true)]
    async fn rate_limit_mutex_is_not_held_across_sleep() {
        let config = AsyncConfig {
            max_concurrent_llm_calls: 10,
            llm_rate_limit_per_second: 1.0,
            ..Default::default()
        };
        let limiter = Arc::new(RateLimiter::new(&config));

        // Burn the only slot in the current 1-second window.
        let _first = limiter.acquire_llm_permit().await.unwrap();

        // Spawn a second caller; it must enter the rate-limit sleep.
        let limiter_for_waiter = Arc::clone(&limiter);
        let waiter = tokio::spawn(async move {
            let _permit = limiter_for_waiter.acquire_llm_permit().await.unwrap();
        });

        // Let the spawned task progress to its inner sleep.
        for _ in 0..4 {
            tokio::task::yield_now().await;
        }

        assert!(
            limiter.llm_rate_tracker.try_lock().is_ok(),
            "rate tracker mutex must be released before sleeping (see issue #21)"
        );

        tokio::time::advance(Duration::from_secs(2)).await;
        waiter.await.unwrap();
    }

    // Calls within the configured rate are admitted with no required wait.
    #[tokio::test(start_paused = true)]
    async fn next_wait_is_zero_within_limit() {
        let mut tracker = RateTracker::new(2.0);
        assert_eq!(tracker.next_wait(), Duration::ZERO);
        assert_eq!(tracker.next_wait(), Duration::ZERO);
    }

    // Exceeding the rate within a window forces the caller to wait until the next window.
    #[tokio::test(start_paused = true)]
    async fn next_wait_pushes_excess_to_next_window() {
        let mut tracker = RateTracker::new(2.0);
        tracker.next_wait();
        tracker.next_wait();
        assert_eq!(tracker.next_wait(), Duration::from_secs(1));
    }

    // After advancing past the window, the counter resets and admits the next call immediately.
    #[tokio::test(start_paused = true)]
    async fn next_wait_resets_after_window_expires() {
        let mut tracker = RateTracker::new(2.0);
        tracker.next_wait();
        tracker.next_wait();
        tokio::time::advance(Duration::from_secs(1)).await;
        assert_eq!(tracker.next_wait(), Duration::ZERO);
    }

    // When earlier callers have reserved into the future, subsequent callers stack into the
    // following windows rather than racing ahead — this is what prevents the bug fix from
    // letting too many requests through during one window.
    #[tokio::test(start_paused = true)]
    async fn next_wait_stacks_concurrent_overflow() {
        let mut tracker = RateTracker::new(2.0);
        tracker.next_wait();
        tracker.next_wait();
        let third = tracker.next_wait();
        let fourth = tracker.next_wait();
        let fifth = tracker.next_wait();
        assert_eq!(third, Duration::from_secs(1));
        assert_eq!(fourth, Duration::from_secs(1));
        assert_eq!(fifth, Duration::from_secs(2));
    }
}
