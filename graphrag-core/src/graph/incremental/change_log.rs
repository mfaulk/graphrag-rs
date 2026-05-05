// Selective cache invalidation tracking changes against cache regions.

#[cfg(feature = "incremental")]
use super::types::{
    CacheRegion, ChangeData, ChangeRecord, ChangeType, InvalidationStats, InvalidationStrategy,
};
#[cfg(feature = "incremental")]
use crate::core::EntityId;
#[cfg(feature = "incremental")]
use chrono::{DateTime, Utc};
#[cfg(feature = "incremental")]
use std::collections::HashSet;

#[cfg(feature = "incremental")]
use {dashmap::DashMap, parking_lot::Mutex};

/// Selective cache invalidation manager
#[cfg(feature = "incremental")]
pub struct SelectiveInvalidation {
    cache_regions: DashMap<String, CacheRegion>,
    entity_to_regions: DashMap<EntityId, HashSet<String>>,
    invalidation_log: Mutex<Vec<(DateTime<Utc>, InvalidationStrategy)>>,
}

#[cfg(feature = "incremental")]
impl Default for SelectiveInvalidation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "incremental")]
impl SelectiveInvalidation {
    /// Creates a new selective invalidation manager
    pub fn new() -> Self {
        Self {
            cache_regions: DashMap::new(),
            entity_to_regions: DashMap::new(),
            invalidation_log: Mutex::new(Vec::new()),
        }
    }

    /// Registers a cache region for invalidation tracking
    pub fn register_cache_region(&self, region: CacheRegion) {
        let region_id = region.region_id.clone();

        // Update entity mappings
        for entity_id in &region.entity_ids {
            self.entity_to_regions
                .entry(entity_id.clone())
                .or_default()
                .insert(region_id.clone());
        }

        self.cache_regions.insert(region_id, region);
    }

    /// Determines invalidation strategies for a set of changes
    pub fn invalidate_for_changes(&self, changes: &[ChangeRecord]) -> Vec<InvalidationStrategy> {
        let mut strategies = Vec::new();
        let mut affected_regions = HashSet::new();

        for change in changes {
            match &change.change_type {
                ChangeType::EntityAdded | ChangeType::EntityUpdated | ChangeType::EntityRemoved => {
                    if let Some(entity_id) = &change.entity_id {
                        if let Some(regions) = self.entity_to_regions.get(entity_id) {
                            affected_regions.extend(regions.clone());
                        }
                        strategies.push(InvalidationStrategy::Relational(entity_id.clone(), 2));
                    }
                },
                ChangeType::RelationshipAdded
                | ChangeType::RelationshipUpdated
                | ChangeType::RelationshipRemoved => {
                    // Invalidate based on relationship endpoints
                    if let ChangeData::Relationship(rel) = &change.data {
                        strategies.push(InvalidationStrategy::Relational(rel.source.clone(), 1));
                        strategies.push(InvalidationStrategy::Relational(rel.target.clone(), 1));
                    }
                },
                _ => {
                    // For other changes, use selective invalidation
                    let cache_keys = self.generate_cache_keys_for_change(change);
                    if !cache_keys.is_empty() {
                        strategies.push(InvalidationStrategy::Selective(cache_keys));
                    }
                },
            }
        }

        // Add regional invalidation for affected regions
        for region_id in affected_regions {
            strategies.push(InvalidationStrategy::Regional(region_id));
        }

        // Log invalidation
        let mut log = self.invalidation_log.lock();
        for strategy in &strategies {
            log.push((Utc::now(), strategy.clone()));
        }

        strategies
    }

    fn generate_cache_keys_for_change(&self, change: &ChangeRecord) -> Vec<String> {
        let mut keys = Vec::new();

        // Generate cache keys based on change type and data
        match &change.change_type {
            ChangeType::EntityAdded | ChangeType::EntityUpdated => {
                if let Some(entity_id) = &change.entity_id {
                    keys.push(format!("entity:{entity_id}"));
                    keys.push(format!("entity_neighbors:{entity_id}"));
                }
            },
            ChangeType::DocumentAdded | ChangeType::DocumentUpdated => {
                if let Some(doc_id) = &change.document_id {
                    keys.push(format!("document:{doc_id}"));
                    keys.push(format!("document_chunks:{doc_id}"));
                }
            },
            ChangeType::EmbeddingAdded | ChangeType::EmbeddingUpdated => {
                if let Some(entity_id) = &change.entity_id {
                    keys.push(format!("embedding:{entity_id}"));
                    keys.push(format!("similarity:{entity_id}"));
                }
            },
            _ => {},
        }

        keys
    }

    /// Gets statistics about cache invalidations
    pub fn get_invalidation_stats(&self) -> InvalidationStats {
        let log = self.invalidation_log.lock();

        InvalidationStats {
            total_invalidations: log.len(),
            cache_regions: self.cache_regions.len(),
            entity_mappings: self.entity_to_regions.len(),
            last_invalidation: log.last().map(|(time, _)| *time),
        }
    }
}
