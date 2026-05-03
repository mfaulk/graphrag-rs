//! Simplified APIs for different user experience levels
//!
//! This module provides progressive disclosure of GraphRAG functionality,
//! allowing users to start simple and add complexity as needed.

/// One-call wrapper around the full pipeline (load → build → query).
pub mod easy;
/// Axum HTTP handlers and request/response types for the REST API.
pub mod handlers;
/// REST server and blocking HTTP client for the API.
pub mod rest;
/// Builder-style API for callers that need more control than [`easy`].
pub mod simple;

#[cfg(test)]
mod tests;

// Re-export for convenience
pub use easy::SimpleGraphRAG;
pub use handlers::AppState;
pub use simple::*;
