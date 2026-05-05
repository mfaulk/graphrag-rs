// Core types and identifiers for incremental graph updates.

use crate::core::{DocumentId, Entity, EntityId, Relationship, TextChunk};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[cfg(feature = "incremental")]
use uuid::Uuid;

/// Unique identifier for update operations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UpdateId(String);

impl UpdateId {
    /// Creates a new unique update identifier
    pub fn new() -> Self {
        #[cfg(feature = "incremental")]
        {
            Self(Uuid::new_v4().to_string())
        }
        #[cfg(not(feature = "incremental"))]
        {
            Self(format!(
                "update_{}",
                Utc::now().timestamp_nanos_opt().unwrap_or(0)
            ))
        }
    }

    /// Creates an update identifier from an existing string
    pub fn from_string(id: String) -> Self {
        Self(id)
    }

    /// Returns the update ID as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for UpdateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for UpdateId {
    fn default() -> Self {
        Self::new()
    }
}

/// Change record for tracking individual modifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeRecord {
    /// Unique identifier for this change
    pub change_id: UpdateId,
    /// Timestamp when the change occurred
    pub timestamp: DateTime<Utc>,
    /// Type of change performed
    pub change_type: ChangeType,
    /// Optional entity ID affected by this change
    pub entity_id: Option<EntityId>,
    /// Optional document ID affected by this change
    pub document_id: Option<DocumentId>,
    /// Operation type (insert, update, delete, upsert)
    pub operation: Operation,
    /// Data associated with the change
    pub data: ChangeData,
    /// Additional metadata for the change
    pub metadata: HashMap<String, String>,
}

/// Types of changes that can occur
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChangeType {
    /// An entity was added to the graph
    EntityAdded,
    /// An existing entity was updated
    EntityUpdated,
    /// An entity was removed from the graph
    EntityRemoved,
    /// A relationship was added to the graph
    RelationshipAdded,
    /// An existing relationship was updated
    RelationshipUpdated,
    /// A relationship was removed from the graph
    RelationshipRemoved,
    /// A document was added
    DocumentAdded,
    /// An existing document was updated
    DocumentUpdated,
    /// A document was removed
    DocumentRemoved,
    /// A text chunk was added
    ChunkAdded,
    /// An existing text chunk was updated
    ChunkUpdated,
    /// A text chunk was removed
    ChunkRemoved,
    /// An embedding was added
    EmbeddingAdded,
    /// An existing embedding was updated
    EmbeddingUpdated,
    /// An embedding was removed
    EmbeddingRemoved,
}

/// Operations that can be performed
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Operation {
    /// Insert a new item
    Insert,
    /// Update an existing item
    Update,
    /// Delete an item
    Delete,
    /// Insert or update (upsert) an item
    Upsert,
}

/// Data associated with a change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeData {
    /// Entity data
    Entity(Entity),
    /// Relationship data
    Relationship(Relationship),
    /// Document data
    Document(Document),
    /// Text chunk data
    Chunk(Box<TextChunk>),
    /// Embedding data with entity ID and vector
    Embedding {
        /// Entity ID for the embedding
        entity_id: EntityId,
        /// Embedding vector
        embedding: Vec<f32>,
    },
    /// Empty change data placeholder
    Empty,
}

/// Document type for incremental updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier for the document
    pub id: DocumentId,
    /// Document title
    pub title: String,
    /// Document content
    pub content: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Atomic change set representing a transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDelta {
    /// Unique identifier for this delta
    pub delta_id: UpdateId,
    /// Timestamp when the delta was created
    pub timestamp: DateTime<Utc>,
    /// List of changes in this delta
    pub changes: Vec<ChangeRecord>,
    /// Delta IDs that this delta depends on
    pub dependencies: Vec<UpdateId>,
    /// Current status of the delta
    pub status: DeltaStatus,
    /// Data needed to rollback this delta
    pub rollback_data: Option<RollbackData>,
}

/// Status of a delta operation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeltaStatus {
    /// Delta is pending application
    Pending,
    /// Delta has been applied but not committed
    Applied,
    /// Delta has been committed
    Committed,
    /// Delta has been rolled back
    RolledBack,
    /// Delta failed with error message
    Failed {
        /// Error message describing the failure
        error: String,
    },
}

/// Data needed for rollback operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackData {
    /// Previous state of entities before the change
    pub previous_entities: Vec<Entity>,
    /// Previous state of relationships before the change
    pub previous_relationships: Vec<Relationship>,
    /// Cache keys affected by the change
    pub affected_caches: Vec<String>,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictStrategy {
    /// Keep the existing data, discard new changes
    KeepExisting,
    /// Keep the new data, discard existing
    KeepNew,
    /// Merge existing and new data intelligently
    Merge,
    /// Use LLM to decide how to resolve conflict
    LLMDecision,
    /// Prompt user to resolve conflict
    UserPrompt,
    /// Use a custom resolver by name
    Custom(String),
}

/// Conflict detected during update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    /// Unique identifier for this conflict
    pub conflict_id: UpdateId,
    /// Type of conflict detected
    pub conflict_type: ConflictType,
    /// Existing data in the graph
    pub existing_data: ChangeData,
    /// New data attempting to be applied
    pub new_data: ChangeData,
    /// Resolution if already resolved
    pub resolution: Option<ConflictResolution>,
}

/// Types of conflicts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    /// Entity already exists with different data
    EntityExists,
    /// Relationship already exists with different data
    RelationshipExists,
    /// Version mismatch between expected and actual
    VersionMismatch,
    /// Data is inconsistent with graph state
    DataInconsistency,
    /// Change violates a constraint
    ConstraintViolation,
}

/// Resolution for a conflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResolution {
    /// Strategy used to resolve the conflict
    pub strategy: ConflictStrategy,
    /// Resolved data after applying strategy
    pub resolved_data: ChangeData,
    /// Metadata about the resolution
    pub metadata: HashMap<String, String>,
}

/// Transaction identifier for atomic operations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransactionId(String);

impl TransactionId {
    /// Creates a new unique transaction identifier
    pub fn new() -> Self {
        #[cfg(feature = "incremental")]
        {
            Self(Uuid::new_v4().to_string())
        }
        #[cfg(not(feature = "incremental"))]
        {
            Self(format!(
                "tx_{}",
                Utc::now().timestamp_nanos_opt().unwrap_or(0)
            ))
        }
    }

    /// Returns the transaction ID as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TransactionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for TransactionId {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    /// Total number of nodes (entities)
    pub node_count: usize,
    /// Total number of edges (relationships)
    pub edge_count: usize,
    /// Average degree of nodes
    pub average_degree: f64,
    /// Maximum degree of any node
    pub max_degree: usize,
    /// Number of connected components
    pub connected_components: usize,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// When statistics were last updated
    pub last_updated: DateTime<Utc>,
}

/// Consistency validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyReport {
    /// Whether the graph is consistent
    pub is_consistent: bool,
    /// Entities with no relationships
    pub orphaned_entities: Vec<EntityId>,
    /// Relationships referencing non-existent entities
    pub broken_relationships: Vec<(EntityId, EntityId, String)>,
    /// Entities missing embeddings
    pub missing_embeddings: Vec<EntityId>,
    /// When validation was performed
    pub validation_time: DateTime<Utc>,
    /// Total number of issues found
    pub issues_found: usize,
}

/// Change event for monitoring and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeEvent {
    /// Unique identifier for the event
    pub event_id: UpdateId,
    /// Type of change event
    pub event_type: ChangeEventType,
    /// Optional entity ID associated with the event
    pub entity_id: Option<EntityId>,
    /// When the event occurred
    pub timestamp: DateTime<Utc>,
    /// Additional metadata about the event
    pub metadata: HashMap<String, String>,
}

/// Types of change events that can be published
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeEventType {
    /// An entity was upserted
    EntityUpserted,
    /// An entity was deleted
    EntityDeleted,
    /// A relationship was upserted
    RelationshipUpserted,
    /// A relationship was deleted
    RelationshipDeleted,
    /// An embedding was updated
    EmbeddingUpdated,
    /// A transaction was started
    TransactionStarted,
    /// A transaction was committed
    TransactionCommitted,
    /// A transaction was rolled back
    TransactionRolledBack,
    /// A conflict was resolved
    ConflictResolved,
    /// Cache was invalidated
    CacheInvalidated,
    /// A batch was processed
    BatchProcessed,
}

/// Cache invalidation strategies
#[derive(Debug, Clone)]
pub enum InvalidationStrategy {
    /// Invalidate specific cache keys
    Selective(Vec<String>),
    /// Invalidate all caches in a region
    Regional(String),
    /// Invalidate all caches
    Global,
    /// Invalidate based on entity relationships
    Relational(EntityId, u32), // entity_id, depth
}

/// Cache region affected by changes
#[derive(Debug, Clone)]
pub struct CacheRegion {
    /// Unique identifier for the cache region
    pub region_id: String,
    /// Entity IDs in this region
    pub entity_ids: HashSet<EntityId>,
    /// Relationship types in this region
    pub relationship_types: HashSet<String>,
    /// Document IDs in this region
    pub document_ids: HashSet<DocumentId>,
    /// When the region was last modified
    pub last_modified: DateTime<Utc>,
}

/// Statistics about cache invalidations
#[derive(Debug, Clone)]
pub struct InvalidationStats {
    /// Total number of invalidations performed
    pub total_invalidations: usize,
    /// Number of cache regions registered
    pub cache_regions: usize,
    /// Number of entity-to-region mappings
    pub entity_mappings: usize,
    /// Timestamp of last invalidation
    pub last_invalidation: Option<DateTime<Utc>>,
}

/// Configuration for incremental operations
#[derive(Debug, Clone)]
pub struct IncrementalConfig {
    /// Maximum number of changes to keep in the log
    pub max_change_log_size: usize,
    /// Maximum number of changes in a single delta
    pub max_delta_size: usize,
    /// Default conflict resolution strategy
    pub conflict_strategy: ConflictStrategy,
    /// Whether to enable performance monitoring
    pub enable_monitoring: bool,
    /// Cache invalidation strategy name
    pub cache_invalidation_strategy: String,
    /// Default batch size for batch operations
    pub batch_size: usize,
    /// Maximum number of concurrent operations
    pub max_concurrent_operations: usize,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            max_change_log_size: 10000,
            max_delta_size: 1000,
            conflict_strategy: ConflictStrategy::Merge,
            enable_monitoring: true,
            cache_invalidation_strategy: "selective".to_string(),
            batch_size: 100,
            max_concurrent_operations: 10,
        }
    }
}

/// Comprehensive statistics for incremental operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalStatistics {
    /// Total number of update operations
    pub total_updates: usize,
    /// Number of successful updates
    pub successful_updates: usize,
    /// Number of failed updates
    pub failed_updates: usize,
    /// Number of entities added
    pub entities_added: usize,
    /// Number of entities updated
    pub entities_updated: usize,
    /// Number of entities removed
    pub entities_removed: usize,
    /// Number of relationships added
    pub relationships_added: usize,
    /// Number of relationships updated
    pub relationships_updated: usize,
    /// Number of relationships removed
    pub relationships_removed: usize,
    /// Number of conflicts resolved
    pub conflicts_resolved: usize,
    /// Number of cache invalidations performed
    pub cache_invalidations: usize,
    /// Average update time in milliseconds
    pub average_update_time_ms: f64,
    /// Peak updates per second achieved
    pub peak_updates_per_second: f64,
    /// Current size of the change log
    pub current_change_log_size: usize,
    /// Current number of active deltas
    pub current_delta_count: usize,
}

impl IncrementalStatistics {
    /// Creates an empty statistics instance
    pub fn empty() -> Self {
        Self {
            total_updates: 0,
            successful_updates: 0,
            failed_updates: 0,
            entities_added: 0,
            entities_updated: 0,
            entities_removed: 0,
            relationships_added: 0,
            relationships_updated: 0,
            relationships_removed: 0,
            conflicts_resolved: 0,
            cache_invalidations: 0,
            average_update_time_ms: 0.0,
            peak_updates_per_second: 0.0,
            current_change_log_size: 0,
            current_delta_count: 0,
        }
    }

    /// Prints statistics to stdout in a formatted way
    pub fn print(&self) {
        println!("🔄 Incremental Updates Statistics");
        println!("  Total updates: {}", self.total_updates);
        println!(
            "  Successful: {} ({:.1}%)",
            self.successful_updates,
            if self.total_updates > 0 {
                (self.successful_updates as f64 / self.total_updates as f64) * 100.0
            } else {
                0.0
            }
        );
        println!("  Failed: {}", self.failed_updates);
        println!(
            "  Entities: +{} ~{} -{}",
            self.entities_added, self.entities_updated, self.entities_removed
        );
        println!(
            "  Relationships: +{} ~{} -{}",
            self.relationships_added, self.relationships_updated, self.relationships_removed
        );
        println!("  Conflicts resolved: {}", self.conflicts_resolved);
        println!("  Cache invalidations: {}", self.cache_invalidations);
        println!("  Avg update time: {:.2}ms", self.average_update_time_ms);
        println!("  Peak updates/sec: {:.1}", self.peak_updates_per_second);
        println!("  Change log size: {}", self.current_change_log_size);
        println!("  Active deltas: {}", self.current_delta_count);
    }
}

/// Helper trait for extracting entity ID from ChangeData
#[allow(dead_code)] // Used only when feature = "incremental" is enabled.
pub(crate) trait ChangeDataExt {
    fn get_entity_id(&self) -> Option<EntityId>;
}

impl ChangeDataExt for ChangeData {
    fn get_entity_id(&self) -> Option<EntityId> {
        match self {
            ChangeData::Entity(entity) => Some(entity.id.clone()),
            ChangeData::Embedding { entity_id, .. } => Some(entity_id.clone()),
            _ => None,
        }
    }
}
