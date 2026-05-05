// Conflict resolution policies and merge strategies.

use super::types::{ChangeData, Conflict, ConflictResolution, ConflictStrategy};
use crate::core::{Entity, GraphRAGError, Relationship, Result};
use std::collections::HashMap;

/// Conflict resolver with multiple strategies
pub struct ConflictResolver {
    pub(super) strategy: ConflictStrategy,
    custom_resolvers: HashMap<String, ConflictResolverFn>,
}

// Reduce type complexity for custom resolver function type
type ConflictResolverFn = Box<dyn Fn(&Conflict) -> Result<ConflictResolution> + Send + Sync>;

impl ConflictResolver {
    /// Creates a new conflict resolver with the given strategy
    pub fn new(strategy: ConflictStrategy) -> Self {
        Self {
            strategy,
            custom_resolvers: HashMap::new(),
        }
    }

    /// Adds a custom resolver function by name
    pub fn with_custom_resolver<F>(mut self, name: String, resolver: F) -> Self
    where
        F: Fn(&Conflict) -> Result<ConflictResolution> + Send + Sync + 'static,
    {
        self.custom_resolvers.insert(name, Box::new(resolver));
        self
    }

    /// Resolves a conflict using the configured strategy
    pub async fn resolve_conflict(&self, conflict: &Conflict) -> Result<ConflictResolution> {
        match &self.strategy {
            ConflictStrategy::KeepExisting => Ok(ConflictResolution {
                strategy: ConflictStrategy::KeepExisting,
                resolved_data: conflict.existing_data.clone(),
                metadata: HashMap::new(),
            }),
            ConflictStrategy::KeepNew => Ok(ConflictResolution {
                strategy: ConflictStrategy::KeepNew,
                resolved_data: conflict.new_data.clone(),
                metadata: HashMap::new(),
            }),
            ConflictStrategy::Merge => self.merge_conflict_data(conflict).await,
            ConflictStrategy::Custom(resolver_name) => {
                if let Some(resolver) = self.custom_resolvers.get(resolver_name) {
                    resolver(conflict)
                } else {
                    Err(GraphRAGError::ConflictResolution {
                        message: format!("Custom resolver '{resolver_name}' not found"),
                    })
                }
            },
            _ => Err(GraphRAGError::ConflictResolution {
                message: "Conflict resolution strategy not implemented".to_string(),
            }),
        }
    }

    async fn merge_conflict_data(&self, conflict: &Conflict) -> Result<ConflictResolution> {
        match (&conflict.existing_data, &conflict.new_data) {
            (ChangeData::Entity(existing), ChangeData::Entity(new)) => {
                let merged = self.merge_entities(existing, new)?;
                Ok(ConflictResolution {
                    strategy: ConflictStrategy::Merge,
                    resolved_data: ChangeData::Entity(merged),
                    metadata: [("merge_strategy".to_string(), "entity_merge".to_string())]
                        .into_iter()
                        .collect(),
                })
            },
            (ChangeData::Relationship(existing), ChangeData::Relationship(new)) => {
                let merged = self.merge_relationships(existing, new)?;
                Ok(ConflictResolution {
                    strategy: ConflictStrategy::Merge,
                    resolved_data: ChangeData::Relationship(merged),
                    metadata: [(
                        "merge_strategy".to_string(),
                        "relationship_merge".to_string(),
                    )]
                    .into_iter()
                    .collect(),
                })
            },
            _ => Err(GraphRAGError::ConflictResolution {
                message: "Cannot merge incompatible data types".to_string(),
            }),
        }
    }

    pub(super) fn merge_entities(&self, existing: &Entity, new: &Entity) -> Result<Entity> {
        let mut merged = existing.clone();

        // Use higher confidence
        if new.confidence > existing.confidence {
            merged.confidence = new.confidence;
            merged.name = new.name.clone();
            merged.entity_type = new.entity_type.clone();
        }

        // Merge mentions
        let mut all_mentions = existing.mentions.clone();
        for new_mention in &new.mentions {
            if !all_mentions.iter().any(|m| {
                m.chunk_id == new_mention.chunk_id && m.start_offset == new_mention.start_offset
            }) {
                all_mentions.push(new_mention.clone());
            }
        }
        merged.mentions = all_mentions;

        // Prefer new embedding if available
        if new.embedding.is_some() {
            merged.embedding = new.embedding.clone();
        }

        Ok(merged)
    }

    fn merge_relationships(
        &self,
        existing: &Relationship,
        new: &Relationship,
    ) -> Result<Relationship> {
        let mut merged = existing.clone();

        // Use higher confidence
        if new.confidence > existing.confidence {
            merged.confidence = new.confidence;
            merged.relation_type = new.relation_type.clone();
        }

        // Merge contexts
        let mut all_contexts = existing.context.clone();
        for new_context in &new.context {
            if !all_contexts.contains(new_context) {
                all_contexts.push(new_context.clone());
            }
        }
        merged.context = all_contexts;

        Ok(merged)
    }
}
