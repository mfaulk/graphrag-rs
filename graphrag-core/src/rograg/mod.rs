//! ROGRAG (Robustly Optimized GraphRAG): logic-form, fuzzy, and decomposition-based query strategies.
//!
//! Strategies fall back in order — logic-form parsing, then fuzzy matching, then query decomposition.
//! See [`processor`] for the orchestration entry point.

#[cfg(feature = "rograg")]
pub mod decomposer;
#[cfg(feature = "rograg")]
pub mod fuzzy_matcher;
#[cfg(feature = "rograg")]
pub mod intent_classifier;
#[cfg(feature = "rograg")]
pub mod logic_form;
#[cfg(feature = "rograg")]
pub mod processor;
#[cfg(feature = "rograg")]
pub mod quality_metrics;
#[cfg(feature = "rograg")]
pub mod streaming;
#[cfg(feature = "rograg")]
pub mod validator;

// Re-export main types with specific naming to avoid conflicts
#[cfg(feature = "rograg")]
pub use decomposer::*;
#[cfg(feature = "rograg")]
pub use fuzzy_matcher::*;
#[cfg(feature = "rograg")]
pub use intent_classifier::*;
#[cfg(feature = "rograg")]
pub use logic_form::*;
#[cfg(feature = "rograg")]
pub use processor::*;
#[cfg(feature = "rograg")]
pub use quality_metrics::{
    ComparativeAnalysis, PerformanceStatistics, QualityMetrics as QualityMetricsConfig,
    QualityMetricsConfig as QualityMetricsOptions, QueryMetrics, ResponseQuality,
};
#[cfg(feature = "rograg")]
pub use streaming::*;
#[cfg(feature = "rograg")]
pub use validator::{
    IssueSeverity, IssueType, QualityMetrics as ValidatorQualityMetrics, QueryValidator,
    ValidationIssue, ValidationResult,
};

#[cfg(feature = "rograg")]
use crate::Result;

/// Initialize the ROGRAG subsystem.
///
/// This function initializes all ROGRAG components and performs any necessary
/// startup configuration. Currently this is a no-op but serves as a future
/// extension point for:
///
/// - Loading pre-compiled pattern databases
/// - Initializing statistical models
/// - Warming up caches
/// - Validating system configuration
///
/// # Returns
///
/// Returns `Ok(())` on successful initialization, or an error if any subsystem
/// fails to initialize properly.
///
/// # Example
///
/// ```rust,ignore
/// use graphrag_core::rograg::initialize_rograg;
///
/// // Initialize before using ROGRAG features
/// initialize_rograg()?;
/// ```
#[cfg(feature = "rograg")]
pub fn initialize_rograg() -> Result<()> {
    // Initialize ROGRAG subsystems
    // Future: Load pattern databases, warm caches, etc.
    Ok(())
}
