// UpdateMonitor: tracks update operations, metrics, and performance stats.

use super::types::UpdateId;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[cfg(feature = "incremental")]
use {
    dashmap::DashMap,
    parking_lot::{Mutex, RwLock},
};

/// Monitor for tracking update operations and performance
#[cfg(feature = "incremental")]
pub struct UpdateMonitor {
    metrics: DashMap<String, UpdateMetric>,
    operations_log: Mutex<Vec<OperationLog>>,
    performance_stats: RwLock<PerformanceStats>,
}

#[cfg(feature = "incremental")]
impl Default for UpdateMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Metric for tracking update operations
#[derive(Debug, Clone)]
pub struct UpdateMetric {
    /// Name of the metric
    pub name: String,
    /// Metric value
    pub value: f64,
    /// When the metric was recorded
    pub timestamp: DateTime<Utc>,
    /// Tags for categorizing the metric
    pub tags: HashMap<String, String>,
}

/// Log entry for an operation
#[derive(Debug, Clone)]
pub struct OperationLog {
    /// Unique operation identifier
    pub operation_id: UpdateId,
    /// Type of operation performed
    pub operation_type: String,
    /// When the operation started
    pub start_time: Instant,
    /// When the operation ended
    pub end_time: Option<Instant>,
    /// Whether the operation succeeded
    pub success: Option<bool>,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Number of entities affected
    pub affected_entities: usize,
    /// Number of relationships affected
    pub affected_relationships: usize,
}

/// Performance statistics for monitoring
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Total number of operations performed
    pub total_operations: u64,
    /// Number of successful operations
    pub successful_operations: u64,
    /// Number of failed operations
    pub failed_operations: u64,
    /// Average time per operation
    pub average_operation_time: Duration,
    /// Peak throughput in operations per second
    pub peak_operations_per_second: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Conflict resolution rate (0.0 to 1.0)
    pub conflict_resolution_rate: f64,
}

#[cfg(feature = "incremental")]
impl UpdateMonitor {
    /// Creates a new update monitor
    pub fn new() -> Self {
        Self {
            metrics: DashMap::new(),
            operations_log: Mutex::new(Vec::new()),
            performance_stats: RwLock::new(PerformanceStats {
                total_operations: 0,
                successful_operations: 0,
                failed_operations: 0,
                average_operation_time: Duration::from_millis(0),
                peak_operations_per_second: 0.0,
                cache_hit_rate: 0.0,
                conflict_resolution_rate: 0.0,
            }),
        }
    }

    /// Starts tracking a new operation and returns its ID
    pub fn start_operation(&self, operation_type: &str) -> UpdateId {
        let operation_id = UpdateId::new();
        let log_entry = OperationLog {
            operation_id: operation_id.clone(),
            operation_type: operation_type.to_string(),
            start_time: Instant::now(),
            end_time: None,
            success: None,
            error_message: None,
            affected_entities: 0,
            affected_relationships: 0,
        };

        self.operations_log.lock().push(log_entry);
        operation_id
    }

    /// Marks an operation as complete with results
    pub fn complete_operation(
        &self,
        operation_id: &UpdateId,
        success: bool,
        error: Option<String>,
        affected_entities: usize,
        affected_relationships: usize,
    ) {
        // parking_lot::Mutex is not reentrant; release the operations_log
        // guard before update_performance_stats reacquires it.
        {
            let mut log = self.operations_log.lock();
            if let Some(entry) = log.iter_mut().find(|e| &e.operation_id == operation_id) {
                entry.end_time = Some(Instant::now());
                entry.success = Some(success);
                entry.error_message = error;
                entry.affected_entities = affected_entities;
                entry.affected_relationships = affected_relationships;
            }
        }

        self.update_performance_stats();
    }

    fn update_performance_stats(&self) {
        let log = self.operations_log.lock();
        let completed_ops: Vec<_> = log
            .iter()
            .filter(|op| op.end_time.is_some() && op.success.is_some())
            .collect();

        if completed_ops.is_empty() {
            return;
        }

        let mut stats = self.performance_stats.write();
        stats.total_operations = completed_ops.len() as u64;
        stats.successful_operations = completed_ops
            .iter()
            .filter(|op| op.success == Some(true))
            .count() as u64;
        stats.failed_operations = stats.total_operations - stats.successful_operations;

        // Calculate average operation time
        let total_time: Duration = completed_ops
            .iter()
            .filter_map(|op| op.end_time.map(|end| end.duration_since(op.start_time)))
            .sum();

        if !completed_ops.is_empty() {
            stats.average_operation_time = total_time / completed_ops.len() as u32;
        }
    }

    /// Records a metric with tags
    pub fn record_metric(&self, name: &str, value: f64, tags: HashMap<String, String>) {
        let metric = UpdateMetric {
            name: name.to_string(),
            value,
            timestamp: Utc::now(),
            tags,
        };
        self.metrics.insert(name.to_string(), metric);
    }

    /// Gets the current performance statistics
    pub fn get_performance_stats(&self) -> PerformanceStats {
        self.performance_stats.read().clone()
    }

    /// Gets the most recent operations up to the specified limit
    pub fn get_recent_operations(&self, limit: usize) -> Vec<OperationLog> {
        let log = self.operations_log.lock();
        log.iter().rev().take(limit).cloned().collect()
    }
}
