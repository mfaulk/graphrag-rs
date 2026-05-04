//! Apache Parquet persistence backend for GraphRAG
//!
//! This module implements efficient columnar storage for knowledge graph components
//! using Apache Arrow and Parquet formats.
//!
//! ## File Structure
//!
//! ```text
//! workspace/
//! ├── entities.parquet          # Entity nodes
//! ├── entity_mentions.parquet   # EntityMention rows joined by entity_id
//! ├── relationships.parquet     # Relationship edges
//! ├── chunks.parquet            # Text chunks
//! └── documents.parquet         # Document metadata
//! ```
//!
//! ## Features
//!
//! - Columnar storage with Snappy compression
//! - Fast selective column reads
//! - Schema evolution support
//! - Integration with Arrow ecosystem (Polars, DuckDB)
//!
//! ## Example
//!
//! ```no_run
//! use graphrag_core::{KnowledgeGraph, persistence::ParquetPersistence};
//! use std::path::PathBuf;
//!
//! # fn example() -> graphrag_core::Result<()> {
//! let graph = KnowledgeGraph::new();
//! let persistence = ParquetPersistence::new(PathBuf::from("./workspace"))?;
//!
//! // Save graph to Parquet files
//! persistence.save_graph(&graph)?;
//!
//! // Load graph from Parquet files
//! let loaded_graph = persistence.load_graph()?;
//! # Ok(())
//! # }
//! ```

use crate::core::{
    ChunkId, Document, DocumentId, Entity, EntityId, GraphRAGError, KnowledgeGraph, Relationship,
    Result, TextChunk,
};
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "persistent-storage")]
use arrow::array::{
    Array, Float32Array, Int64Array, ListArray, ListBuilder, RecordBatch, StringArray,
    StringBuilder, UInt64Array,
};
#[cfg(feature = "persistent-storage")]
use arrow::datatypes::{DataType, Field, Schema};
#[cfg(feature = "persistent-storage")]
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
#[cfg(feature = "persistent-storage")]
use parquet::arrow::arrow_writer::ArrowWriter;
#[cfg(feature = "persistent-storage")]
use parquet::file::properties::WriterProperties;

/// Configuration for Parquet persistence
#[derive(Debug, Clone)]
pub struct ParquetConfig {
    /// Compression codec (default: Snappy)
    pub compression: ParquetCompression,
    /// Row group size (default: 10000)
    pub row_group_size: usize,
    /// Enable dictionary encoding (default: true)
    pub dictionary_encoding: bool,
}

/// Parquet compression codecs
#[derive(Debug, Clone, Copy)]
pub enum ParquetCompression {
    /// No compression
    Uncompressed,
    /// Snappy compression (default, fast)
    Snappy,
    /// Gzip compression (better ratio, slower)
    Gzip,
    /// LZ4 compression (very fast)
    Lz4,
    /// Zstd compression (best ratio, moderate speed)
    Zstd,
}

impl Default for ParquetConfig {
    fn default() -> Self {
        Self {
            compression: ParquetCompression::Snappy,
            row_group_size: 10000,
            dictionary_encoding: true,
        }
    }
}

/// Parquet persistence backend
#[derive(Debug, Clone)]
pub struct ParquetPersistence {
    /// Base directory for Parquet files
    base_dir: PathBuf,
    /// Configuration
    config: ParquetConfig,
}

impl ParquetPersistence {
    /// Create a new Parquet persistence backend
    ///
    /// # Arguments
    /// * `base_dir` - Directory to store Parquet files
    ///
    /// # Example
    /// ```no_run
    /// use graphrag_core::persistence::ParquetPersistence;
    /// use std::path::PathBuf;
    ///
    /// let persistence = ParquetPersistence::new(PathBuf::from("./workspace")).unwrap();
    /// ```
    pub fn new(base_dir: PathBuf) -> Result<Self> {
        // Create directory if it doesn't exist
        if !base_dir.exists() {
            std::fs::create_dir_all(&base_dir)?;
        }

        Ok(Self {
            base_dir,
            config: ParquetConfig::default(),
        })
    }

    /// Create with custom configuration
    pub fn with_config(base_dir: PathBuf, config: ParquetConfig) -> Result<Self> {
        if !base_dir.exists() {
            std::fs::create_dir_all(&base_dir)?;
        }

        Ok(Self { base_dir, config })
    }

    /// Save knowledge graph to Parquet files
    pub fn save_graph(&self, graph: &KnowledgeGraph) -> Result<()> {
        #[cfg(feature = "tracing")]
        tracing::info!("Saving knowledge graph to Parquet files");

        // Save entities
        self.save_entities(graph)?;

        // Save relationships
        self.save_relationships(graph)?;

        // Save chunks
        self.save_chunks(graph)?;

        // Save documents
        self.save_documents(graph)?;

        #[cfg(feature = "tracing")]
        tracing::info!("Successfully saved knowledge graph to Parquet");

        Ok(())
    }

    /// Load knowledge graph from Parquet files
    pub fn load_graph(&self) -> Result<KnowledgeGraph> {
        #[cfg(feature = "tracing")]
        tracing::info!("Loading knowledge graph from Parquet files");

        let mut graph = KnowledgeGraph::new();

        // Load documents
        let documents = self.load_documents()?;
        for document in documents {
            graph.add_document(document)?;
        }

        // Load chunks (if not already loaded from documents)
        let chunks = self.load_chunks()?;
        for chunk in chunks {
            graph.add_chunk(chunk)?;
        }

        // Load entities
        let entities = self.load_entities()?;
        for entity in entities {
            graph.add_entity(entity)?;
        }

        // Load relationships
        let relationships = self.load_relationships()?;
        let mut dropped_relationships: usize = 0;
        for relationship in relationships {
            // A relationship can fail to attach if its endpoints reference
            // entities that aren't in the graph (schema drift between save/
            // load, partial corruption, etc.). Don't crash the load — but
            // also don't silently ignore: count and log so users see a
            // signal instead of "queries return slightly wrong answers."
            if graph.add_relationship(relationship).is_err() {
                dropped_relationships += 1;
            }
        }
        if dropped_relationships > 0 {
            #[cfg(feature = "tracing")]
            tracing::warn!(
                "Parquet load: dropped {} relationship(s) whose endpoints \
                 are not present in the graph (possible schema drift or partial corruption)",
                dropped_relationships
            );
        }

        #[cfg(feature = "tracing")]
        tracing::info!(
            "Successfully loaded knowledge graph: {} entities, {} relationships",
            graph.entity_count(),
            graph.relationship_count()
        );

        Ok(graph)
    }

    /// Save entities to Parquet.
    ///
    /// Entity rows go to `entities.parquet`. The `EntityMention` payload
    /// (chunk id + offsets + per-mention confidence) is written to a
    /// sidecar `entity_mentions.parquet` keyed by entity id, so that
    /// entity-to-chunk provenance (`source_chunks`) survives a save/load
    /// round trip. See `load_entities`/`load_entity_mentions` for the read
    /// side. Fixes #24.
    #[cfg(feature = "persistent-storage")]
    fn save_entities(&self, graph: &KnowledgeGraph) -> Result<()> {
        let entities: Vec<_> = graph.entities().collect();

        if entities.is_empty() {
            #[cfg(feature = "tracing")]
            tracing::warn!("No entities to save");
            return Ok(());
        }

        // Build Arrow schema for entities. `description` is nullable so old
        // parquet files written before #97 still load (the reader looks the
        // column up by name and tolerates its absence).
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("entity_type", DataType::Utf8, false),
            Field::new("confidence", DataType::Float32, false),
            Field::new("mention_count", DataType::Int64, false),
            Field::new(
                "embedding",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            ),
            Field::new("description", DataType::Utf8, true),
        ]));

        // Convert entities to Arrow arrays
        let ids: StringArray = entities.iter().map(|e| Some(e.id.0.as_str())).collect();
        let names: StringArray = entities.iter().map(|e| Some(e.name.as_str())).collect();
        let types: StringArray = entities
            .iter()
            .map(|e| Some(e.entity_type.as_str()))
            .collect();
        let confidences: Float32Array = entities.iter().map(|e| Some(e.confidence)).collect();
        let mention_counts: Int64Array = entities
            .iter()
            .map(|e| Some(e.mentions.len() as i64))
            .collect();

        // Build embeddings ListArray
        let mut embedding_builder = ListBuilder::new(arrow::array::Float32Builder::new());
        for entity in entities.iter() {
            if let Some(ref emb) = entity.embedding {
                for &val in emb {
                    embedding_builder.values().append_value(val);
                }
                embedding_builder.append(true);
            } else {
                embedding_builder.append(false); // null
            }
        }
        let embeddings = embedding_builder.finish();

        // Build descriptions array (nullable Utf8).
        let descriptions: StringArray = entities.iter().map(|e| e.description.as_deref()).collect();

        // Create RecordBatch
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids),
                Arc::new(names),
                Arc::new(types),
                Arc::new(confidences),
                Arc::new(mention_counts),
                Arc::new(embeddings),
                Arc::new(descriptions),
            ],
        )
        .map_err(|e| GraphRAGError::Config {
            message: format!("Failed to create RecordBatch: {}", e),
        })?;

        // Write to Parquet file
        let file_path = self.base_dir.join("entities.parquet");
        let file = std::fs::File::create(&file_path)?;

        let props = WriterProperties::builder()
            .set_compression(self.get_compression())
            .build();

        let mut writer =
            ArrowWriter::try_new(file, schema, Some(props)).map_err(|e| GraphRAGError::Config {
                message: format!("Failed to create ArrowWriter: {}", e),
            })?;

        writer.write(&batch).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to write batch: {}", e),
        })?;

        writer.close().map_err(|e| GraphRAGError::Config {
            message: format!("Failed to close writer: {}", e),
        })?;

        #[cfg(feature = "tracing")]
        tracing::info!("Saved {} entities to {:?}", entities.len(), file_path);

        // Persist the mention payload separately so retrieval has provenance
        // back to source chunks after reload (#24).
        self.save_entity_mentions(&entities)?;

        Ok(())
    }

    /// Save entity mentions to a sidecar `entity_mentions.parquet`.
    ///
    /// One row per mention; entity_id is the join key back to entities.parquet.
    /// Always (re)written — including as an empty file — so a stale sidecar
    /// from a previous save with mentions doesn't get re-attached to a fresh
    /// graph that has none.
    #[cfg(feature = "persistent-storage")]
    fn save_entity_mentions(&self, entities: &[&Entity]) -> Result<()> {
        let file_path = self.base_dir.join("entity_mentions.parquet");

        let schema = Arc::new(Schema::new(vec![
            Field::new("entity_id", DataType::Utf8, false),
            Field::new("chunk_id", DataType::Utf8, false),
            Field::new("start_offset", DataType::UInt64, false),
            Field::new("end_offset", DataType::UInt64, false),
            Field::new("confidence", DataType::Float32, false),
        ]));

        let total_mentions: usize = entities.iter().map(|e| e.mentions.len()).sum();

        let mut entity_ids: Vec<&str> = Vec::with_capacity(total_mentions);
        let mut chunk_ids: Vec<&str> = Vec::with_capacity(total_mentions);
        let mut starts: Vec<u64> = Vec::with_capacity(total_mentions);
        let mut ends: Vec<u64> = Vec::with_capacity(total_mentions);
        let mut confidences: Vec<f32> = Vec::with_capacity(total_mentions);

        for entity in entities {
            for mention in &entity.mentions {
                entity_ids.push(entity.id.0.as_str());
                chunk_ids.push(mention.chunk_id.0.as_str());
                starts.push(mention.start_offset as u64);
                ends.push(mention.end_offset as u64);
                confidences.push(mention.confidence);
            }
        }

        let entity_id_arr: StringArray = entity_ids.into_iter().map(Some).collect();
        let chunk_id_arr: StringArray = chunk_ids.into_iter().map(Some).collect();
        let start_arr: UInt64Array = starts.into_iter().map(Some).collect();
        let end_arr: UInt64Array = ends.into_iter().map(Some).collect();
        let confidence_arr: Float32Array = confidences.into_iter().map(Some).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(entity_id_arr),
                Arc::new(chunk_id_arr),
                Arc::new(start_arr),
                Arc::new(end_arr),
                Arc::new(confidence_arr),
            ],
        )
        .map_err(|e| GraphRAGError::Config {
            message: format!("Failed to create mentions RecordBatch: {}", e),
        })?;

        let file = std::fs::File::create(&file_path)?;
        let props = WriterProperties::builder()
            .set_compression(self.get_compression())
            .build();
        let mut writer =
            ArrowWriter::try_new(file, schema, Some(props)).map_err(|e| GraphRAGError::Config {
                message: format!("Failed to create ArrowWriter: {}", e),
            })?;

        writer.write(&batch).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to write mentions batch: {}", e),
        })?;

        writer.close().map_err(|e| GraphRAGError::Config {
            message: format!("Failed to close mentions writer: {}", e),
        })?;

        #[cfg(feature = "tracing")]
        tracing::info!(
            "Saved {} entity mentions to {:?}",
            total_mentions,
            file_path
        );

        Ok(())
    }

    /// Load entities from Parquet (rejoins mentions from the sidecar file).
    #[cfg(feature = "persistent-storage")]
    fn load_entities(&self) -> Result<Vec<Entity>> {
        let file_path = self.base_dir.join("entities.parquet");

        if !file_path.exists() {
            #[cfg(feature = "tracing")]
            tracing::warn!("No entities.parquet found");
            return Ok(Vec::new());
        }

        let file = std::fs::File::open(&file_path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to create Parquet reader: {}", e),
            })?
            .build()
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to build reader: {}", e),
            })?;

        let mut entities = Vec::new();

        for batch in reader {
            let batch = batch.map_err(|e| GraphRAGError::Config {
                message: format!("Failed to read batch: {}", e),
            })?;

            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid id column type".to_string(),
                })?;

            let names = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid name column type".to_string(),
                })?;

            let types = batch
                .column(2)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid entity_type column type".to_string(),
                })?;

            let confidences = batch
                .column(3)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid confidence column type".to_string(),
                })?;

            let embeddings = batch
                .column(5)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid embedding column type".to_string(),
                })?;

            // `description` was added in #97. Resolve it by name so that
            // legacy parquet files written before this column existed still
            // load — `column_by_name` returns `None` and we leave description
            // unset for those rows.
            let descriptions = batch
                .schema()
                .index_of("description")
                .ok()
                .and_then(|idx| batch.column(idx).as_any().downcast_ref::<StringArray>());

            for i in 0..batch.num_rows() {
                // Extract embedding
                let embedding = if !embeddings.is_null(i) {
                    let emb_list = embeddings.value(i);
                    let emb_floats = emb_list
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| GraphRAGError::Config {
                            message: "Invalid embedding list type".to_string(),
                        })?;

                    let mut emb_vec = Vec::with_capacity(emb_floats.len());
                    for j in 0..emb_floats.len() {
                        if !emb_floats.is_null(j) {
                            emb_vec.push(emb_floats.value(j));
                        }
                    }
                    Some(emb_vec)
                } else {
                    None
                };

                let mut entity = Entity::new(
                    EntityId::new(ids.value(i).to_string()),
                    names.value(i).to_string(),
                    types.value(i).to_string(),
                    confidences.value(i),
                );
                entity.embedding = embedding;
                entity.description = descriptions.and_then(|arr| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i).to_string())
                    }
                });
                entities.push(entity);
            }
        }

        // Reattach mentions from the sidecar so retrieval has provenance back
        // to source chunks (#24). Tolerate a missing sidecar file (legacy
        // workspaces written before this fix). Use `remove` to take
        // ownership of each `Vec<EntityMention>` rather than cloning.
        let mut mentions_by_entity = self.load_entity_mentions()?;
        if !mentions_by_entity.is_empty() {
            for entity in entities.iter_mut() {
                if let Some(mentions) = mentions_by_entity.remove(&entity.id) {
                    entity.mentions = mentions;
                }
            }
        }

        #[cfg(feature = "tracing")]
        tracing::info!("Loaded {} entities from {:?}", entities.len(), file_path);

        Ok(entities)
    }

    /// Load entity mentions from `entity_mentions.parquet`, grouped by
    /// entity id. Returns an empty map if the file is missing — legacy
    /// workspaces written before issue #24 don't have it.
    #[cfg(feature = "persistent-storage")]
    fn load_entity_mentions(
        &self,
    ) -> Result<std::collections::HashMap<EntityId, Vec<crate::core::EntityMention>>> {
        use crate::core::EntityMention;

        let file_path = self.base_dir.join("entity_mentions.parquet");
        let mut by_entity: std::collections::HashMap<EntityId, Vec<EntityMention>> =
            std::collections::HashMap::new();

        if !file_path.exists() {
            #[cfg(feature = "tracing")]
            tracing::debug!(
                "No entity_mentions.parquet found — mentions will be empty (legacy workspace)"
            );
            return Ok(by_entity);
        }

        let file = std::fs::File::open(&file_path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to create mentions Parquet reader: {}", e),
            })?
            .build()
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to build mentions reader: {}", e),
            })?;

        for batch in reader {
            let batch = batch.map_err(|e| GraphRAGError::Config {
                message: format!("Failed to read mentions batch: {}", e),
            })?;

            let entity_ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid entity_id column type in entity_mentions.parquet".to_string(),
                })?;
            let chunk_ids = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid chunk_id column type in entity_mentions.parquet".to_string(),
                })?;
            let starts = batch
                .column(2)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid start_offset column type in entity_mentions.parquet"
                        .to_string(),
                })?;
            let ends = batch
                .column(3)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid end_offset column type in entity_mentions.parquet"
                        .to_string(),
                })?;
            let confidences = batch
                .column(4)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid confidence column type in entity_mentions.parquet"
                        .to_string(),
                })?;

            for i in 0..batch.num_rows() {
                let entity_id = EntityId::new(entity_ids.value(i).to_string());
                let mention = EntityMention {
                    chunk_id: ChunkId::new(chunk_ids.value(i).to_string()),
                    start_offset: starts.value(i) as usize,
                    end_offset: ends.value(i) as usize,
                    confidence: confidences.value(i),
                };
                by_entity.entry(entity_id).or_default().push(mention);
            }
        }

        #[cfg(feature = "tracing")]
        tracing::info!(
            "Loaded mentions for {} entities from {:?}",
            by_entity.len(),
            file_path
        );

        Ok(by_entity)
    }

    /// Placeholder implementations for relationships, chunks, and documents
    /// These will be similar to entities but with different schemas

    #[cfg(feature = "persistent-storage")]
    fn save_relationships(&self, graph: &KnowledgeGraph) -> Result<()> {
        let relationships: Vec<&Relationship> = graph.relationships().collect();

        if relationships.is_empty() {
            #[cfg(feature = "tracing")]
            tracing::info!("No relationships to save");
            return Ok(());
        }

        // Build Arrow schema for relationships
        let schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("relation_type", DataType::Utf8, false),
            Field::new("confidence", DataType::Float32, false),
            Field::new(
                "context",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
        ]));

        // Convert relationships to Arrow arrays
        let sources: StringArray = relationships
            .iter()
            .map(|r| Some(r.source.0.as_str()))
            .collect();
        let targets: StringArray = relationships
            .iter()
            .map(|r| Some(r.target.0.as_str()))
            .collect();
        let types: StringArray = relationships
            .iter()
            .map(|r| Some(r.relation_type.as_str()))
            .collect();
        let confidences: Float32Array = relationships.iter().map(|r| Some(r.confidence)).collect();

        // Build context ListArray
        let mut context_builder = ListBuilder::new(StringBuilder::new());
        for rel in relationships.iter() {
            for chunk_id in &rel.context {
                context_builder.values().append_value(&chunk_id.0);
            }
            context_builder.append(true);
        }
        let contexts = context_builder.finish();

        // Create RecordBatch
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(sources),
                Arc::new(targets),
                Arc::new(types),
                Arc::new(confidences),
                Arc::new(contexts),
            ],
        )
        .map_err(|e| GraphRAGError::Config {
            message: format!("Failed to create RecordBatch: {}", e),
        })?;

        // Write to Parquet file
        let file_path = self.base_dir.join("relationships.parquet");
        let file = std::fs::File::create(&file_path)?;

        let props = WriterProperties::builder()
            .set_compression(self.get_compression())
            .build();

        let mut writer =
            ArrowWriter::try_new(file, schema, Some(props)).map_err(|e| GraphRAGError::Config {
                message: format!("Failed to create ArrowWriter: {}", e),
            })?;

        writer.write(&batch).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to write batch: {}", e),
        })?;

        writer.close().map_err(|e| GraphRAGError::Config {
            message: format!("Failed to close writer: {}", e),
        })?;

        #[cfg(feature = "tracing")]
        tracing::info!(
            "Saved {} relationships to {:?}",
            relationships.len(),
            file_path
        );

        Ok(())
    }

    #[cfg(feature = "persistent-storage")]
    fn load_relationships(&self) -> Result<Vec<Relationship>> {
        let file_path = self.base_dir.join("relationships.parquet");

        if !file_path.exists() {
            #[cfg(feature = "tracing")]
            tracing::warn!("No relationships.parquet found");
            return Ok(Vec::new());
        }

        let file = std::fs::File::open(&file_path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to create Parquet reader: {}", e),
            })?
            .build()
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to build reader: {}", e),
            })?;

        let mut relationships = Vec::new();

        for batch in reader {
            let batch = batch.map_err(|e| GraphRAGError::Config {
                message: format!("Failed to read batch: {}", e),
            })?;

            let sources = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid source column type".to_string(),
                })?;

            let targets = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid target column type".to_string(),
                })?;

            let types = batch
                .column(2)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid relation_type column type".to_string(),
                })?;

            let confidences = batch
                .column(3)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid confidence column type".to_string(),
                })?;

            let contexts = batch
                .column(4)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid context column type".to_string(),
                })?;

            for i in 0..batch.num_rows() {
                // Extract context chunk IDs
                let mut context = Vec::new();
                if !contexts.is_null(i) {
                    let context_list = contexts.value(i);
                    let context_strings = context_list
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| GraphRAGError::Config {
                            message: "Invalid context list type".to_string(),
                        })?;

                    for j in 0..context_strings.len() {
                        if !context_strings.is_null(j) {
                            context.push(ChunkId::new(context_strings.value(j).to_string()));
                        }
                    }
                }

                let relationship = Relationship::new(
                    EntityId::new(sources.value(i).to_string()),
                    EntityId::new(targets.value(i).to_string()),
                    types.value(i).to_string(),
                    confidences.value(i),
                )
                .with_context(context);

                relationships.push(relationship);
            }
        }

        #[cfg(feature = "tracing")]
        tracing::info!(
            "Loaded {} relationships from {:?}",
            relationships.len(),
            file_path
        );

        Ok(relationships)
    }

    #[cfg(feature = "persistent-storage")]
    fn save_chunks(&self, graph: &KnowledgeGraph) -> Result<()> {
        let chunks: Vec<&TextChunk> = graph.chunks().collect();

        if chunks.is_empty() {
            #[cfg(feature = "tracing")]
            tracing::info!("No chunks to save");
            return Ok(());
        }

        // Build Arrow schema for chunks
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("document_id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("start_offset", DataType::UInt64, false),
            Field::new("end_offset", DataType::UInt64, false),
            Field::new(
                "embedding",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            ),
            Field::new(
                "entities",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            // Metadata fields
            Field::new("chapter", DataType::Utf8, true),
            Field::new(
                "keywords",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new("summary", DataType::Utf8, true),
        ]));

        // Convert chunks to Arrow arrays
        let ids: StringArray = chunks.iter().map(|c| Some(c.id.0.as_str())).collect();
        let doc_ids: StringArray = chunks
            .iter()
            .map(|c| Some(c.document_id.0.as_str()))
            .collect();
        let contents: StringArray = chunks.iter().map(|c| Some(c.content.as_str())).collect();
        let start_offsets: UInt64Array =
            chunks.iter().map(|c| Some(c.start_offset as u64)).collect();
        let end_offsets: UInt64Array = chunks.iter().map(|c| Some(c.end_offset as u64)).collect();

        // Build embeddings ListArray
        let mut embedding_builder = ListBuilder::new(arrow::array::Float32Builder::new());
        for chunk in chunks.iter() {
            if let Some(ref emb) = chunk.embedding {
                for &val in emb {
                    embedding_builder.values().append_value(val);
                }
                embedding_builder.append(true);
            } else {
                embedding_builder.append(false); // null
            }
        }
        let embeddings = embedding_builder.finish();

        // Build entities ListArray
        let mut entities_builder = ListBuilder::new(StringBuilder::new());
        for chunk in chunks.iter() {
            for entity_id in &chunk.entities {
                entities_builder.values().append_value(&entity_id.0);
            }
            entities_builder.append(true);
        }
        let entities = entities_builder.finish();

        // Metadata fields
        let chapters: StringArray = chunks
            .iter()
            .map(|c| c.metadata.chapter.as_deref())
            .collect();

        let mut keywords_builder = ListBuilder::new(StringBuilder::new());
        for chunk in chunks.iter() {
            for keyword in &chunk.metadata.keywords {
                keywords_builder.values().append_value(keyword);
            }
            keywords_builder.append(true);
        }
        let keywords = keywords_builder.finish();

        let summaries: StringArray = chunks
            .iter()
            .map(|c| c.metadata.summary.as_deref())
            .collect();

        // Create RecordBatch
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids),
                Arc::new(doc_ids),
                Arc::new(contents),
                Arc::new(start_offsets),
                Arc::new(end_offsets),
                Arc::new(embeddings),
                Arc::new(entities),
                Arc::new(chapters),
                Arc::new(keywords),
                Arc::new(summaries),
            ],
        )
        .map_err(|e| GraphRAGError::Config {
            message: format!("Failed to create RecordBatch: {}", e),
        })?;

        // Write to Parquet file
        let file_path = self.base_dir.join("chunks.parquet");
        let file = std::fs::File::create(&file_path)?;

        let props = WriterProperties::builder()
            .set_compression(self.get_compression())
            .build();

        let mut writer =
            ArrowWriter::try_new(file, schema, Some(props)).map_err(|e| GraphRAGError::Config {
                message: format!("Failed to create ArrowWriter: {}", e),
            })?;

        writer.write(&batch).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to write batch: {}", e),
        })?;

        writer.close().map_err(|e| GraphRAGError::Config {
            message: format!("Failed to close writer: {}", e),
        })?;

        #[cfg(feature = "tracing")]
        tracing::info!("Saved {} chunks to {:?}", chunks.len(), file_path);

        Ok(())
    }

    #[cfg(feature = "persistent-storage")]
    fn load_chunks(&self) -> Result<Vec<TextChunk>> {
        let file_path = self.base_dir.join("chunks.parquet");

        if !file_path.exists() {
            #[cfg(feature = "tracing")]
            tracing::warn!("No chunks.parquet found");
            return Ok(Vec::new());
        }

        let file = std::fs::File::open(&file_path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to create Parquet reader: {}", e),
            })?
            .build()
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to build reader: {}", e),
            })?;

        let mut chunks = Vec::new();

        for batch in reader {
            let batch = batch.map_err(|e| GraphRAGError::Config {
                message: format!("Failed to read batch: {}", e),
            })?;

            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid id column type".to_string(),
                })?;

            let doc_ids = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid document_id column type".to_string(),
                })?;

            let contents = batch
                .column(2)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid content column type".to_string(),
                })?;

            let start_offsets = batch
                .column(3)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid start_offset column type".to_string(),
                })?;

            let end_offsets = batch
                .column(4)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid end_offset column type".to_string(),
                })?;

            let embeddings = batch
                .column(5)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid embedding column type".to_string(),
                })?;

            let entities_col = batch
                .column(6)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid entities column type".to_string(),
                })?;

            let chapters = batch
                .column(7)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid chapter column type".to_string(),
                })?;

            let keywords_col = batch
                .column(8)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid keywords column type".to_string(),
                })?;

            let summaries = batch
                .column(9)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid summary column type".to_string(),
                })?;

            for i in 0..batch.num_rows() {
                // Extract embedding
                let embedding = if !embeddings.is_null(i) {
                    let emb_list = embeddings.value(i);
                    let emb_floats = emb_list
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| GraphRAGError::Config {
                            message: "Invalid embedding list type".to_string(),
                        })?;

                    let mut emb_vec = Vec::with_capacity(emb_floats.len());
                    for j in 0..emb_floats.len() {
                        if !emb_floats.is_null(j) {
                            emb_vec.push(emb_floats.value(j));
                        }
                    }
                    Some(emb_vec)
                } else {
                    None
                };

                // Extract entities
                let mut entities = Vec::new();
                if !entities_col.is_null(i) {
                    let ent_list = entities_col.value(i);
                    let ent_strings =
                        ent_list
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| GraphRAGError::Config {
                                message: "Invalid entities list type".to_string(),
                            })?;

                    for j in 0..ent_strings.len() {
                        if !ent_strings.is_null(j) {
                            entities.push(EntityId::new(ent_strings.value(j).to_string()));
                        }
                    }
                }

                // Extract keywords
                let mut keywords = Vec::new();
                if !keywords_col.is_null(i) {
                    let kw_list = keywords_col.value(i);
                    let kw_strings =
                        kw_list
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| GraphRAGError::Config {
                                message: "Invalid keywords list type".to_string(),
                            })?;

                    for j in 0..kw_strings.len() {
                        if !kw_strings.is_null(j) {
                            keywords.push(kw_strings.value(j).to_string());
                        }
                    }
                }

                // Build metadata
                let metadata = crate::core::ChunkMetadata {
                    chapter: if !chapters.is_null(i) {
                        Some(chapters.value(i).to_string())
                    } else {
                        None
                    },
                    keywords,
                    summary: if !summaries.is_null(i) {
                        Some(summaries.value(i).to_string())
                    } else {
                        None
                    },
                    ..Default::default()
                };

                let chunk = TextChunk {
                    id: ChunkId::new(ids.value(i).to_string()),
                    document_id: DocumentId::new(doc_ids.value(i).to_string()),
                    content: contents.value(i).to_string(),
                    start_offset: start_offsets.value(i) as usize,
                    end_offset: end_offsets.value(i) as usize,
                    embedding,
                    entities,
                    metadata,
                };

                chunks.push(chunk);
            }
        }

        #[cfg(feature = "tracing")]
        tracing::info!("Loaded {} chunks from {:?}", chunks.len(), file_path);

        Ok(chunks)
    }

    #[cfg(feature = "persistent-storage")]
    fn save_documents(&self, graph: &KnowledgeGraph) -> Result<()> {
        let documents: Vec<&Document> = graph.documents().collect();

        if documents.is_empty() {
            #[cfg(feature = "tracing")]
            tracing::info!("No documents to save");
            return Ok(());
        }

        // Build Arrow schema for documents
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("title", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new(
                "metadata_keys",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new(
                "metadata_values",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new("chunk_count", DataType::Int64, false),
        ]));

        // Convert documents to Arrow arrays
        let ids: StringArray = documents.iter().map(|d| Some(d.id.0.as_str())).collect();
        let titles: StringArray = documents.iter().map(|d| Some(d.title.as_str())).collect();
        let contents: StringArray = documents.iter().map(|d| Some(d.content.as_str())).collect();
        let chunk_counts: Int64Array = documents
            .iter()
            .map(|d| Some(d.chunks.len() as i64))
            .collect();

        // Build metadata keys and values ListArrays
        let mut keys_builder = ListBuilder::new(StringBuilder::new());
        let mut values_builder = ListBuilder::new(StringBuilder::new());

        for doc in documents.iter() {
            for (key, value) in &doc.metadata {
                keys_builder.values().append_value(key);
                values_builder.values().append_value(value);
            }
            keys_builder.append(true);
            values_builder.append(true);
        }

        let metadata_keys = keys_builder.finish();
        let metadata_values = values_builder.finish();

        // Create RecordBatch
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids),
                Arc::new(titles),
                Arc::new(contents),
                Arc::new(metadata_keys),
                Arc::new(metadata_values),
                Arc::new(chunk_counts),
            ],
        )
        .map_err(|e| GraphRAGError::Config {
            message: format!("Failed to create RecordBatch: {}", e),
        })?;

        // Write to Parquet file
        let file_path = self.base_dir.join("documents.parquet");
        let file = std::fs::File::create(&file_path)?;

        let props = WriterProperties::builder()
            .set_compression(self.get_compression())
            .build();

        let mut writer =
            ArrowWriter::try_new(file, schema, Some(props)).map_err(|e| GraphRAGError::Config {
                message: format!("Failed to create ArrowWriter: {}", e),
            })?;

        writer.write(&batch).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to write batch: {}", e),
        })?;

        writer.close().map_err(|e| GraphRAGError::Config {
            message: format!("Failed to close writer: {}", e),
        })?;

        #[cfg(feature = "tracing")]
        tracing::info!("Saved {} documents to {:?}", documents.len(), file_path);

        Ok(())
    }

    #[cfg(feature = "persistent-storage")]
    fn load_documents(&self) -> Result<Vec<Document>> {
        let file_path = self.base_dir.join("documents.parquet");

        if !file_path.exists() {
            #[cfg(feature = "tracing")]
            tracing::warn!("No documents.parquet found");
            return Ok(Vec::new());
        }

        let file = std::fs::File::open(&file_path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to create Parquet reader: {}", e),
            })?
            .build()
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to build reader: {}", e),
            })?;

        let mut documents = Vec::new();

        for batch in reader {
            let batch = batch.map_err(|e| GraphRAGError::Config {
                message: format!("Failed to read batch: {}", e),
            })?;

            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid id column type".to_string(),
                })?;

            let titles = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid title column type".to_string(),
                })?;

            let contents = batch
                .column(2)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid content column type".to_string(),
                })?;

            let metadata_keys = batch
                .column(3)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid metadata_keys column type".to_string(),
                })?;

            let metadata_values = batch
                .column(4)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid metadata_values column type".to_string(),
                })?;

            for i in 0..batch.num_rows() {
                // Extract metadata
                let mut metadata = indexmap::IndexMap::new();

                if !metadata_keys.is_null(i) && !metadata_values.is_null(i) {
                    let keys_list = metadata_keys.value(i);
                    let values_list = metadata_values.value(i);

                    let keys_strings = keys_list
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| GraphRAGError::Config {
                            message: "Invalid metadata keys list type".to_string(),
                        })?;

                    let values_strings = values_list
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| GraphRAGError::Config {
                            message: "Invalid metadata values list type".to_string(),
                        })?;

                    for j in 0..keys_strings.len().min(values_strings.len()) {
                        if !keys_strings.is_null(j) && !values_strings.is_null(j) {
                            metadata.insert(
                                keys_strings.value(j).to_string(),
                                values_strings.value(j).to_string(),
                            );
                        }
                    }
                }

                let document = Document {
                    id: DocumentId::new(ids.value(i).to_string()),
                    title: titles.value(i).to_string(),
                    content: contents.value(i).to_string(),
                    metadata,
                    chunks: Vec::new(), // Chunks will be loaded separately and associated
                };

                documents.push(document);
            }
        }

        #[cfg(feature = "tracing")]
        tracing::info!("Loaded {} documents from {:?}", documents.len(), file_path);

        Ok(documents)
    }

    /// Get compression codec for Parquet writer
    #[cfg(feature = "persistent-storage")]
    fn get_compression(&self) -> parquet::basic::Compression {
        use parquet::basic::Compression;

        match self.config.compression {
            ParquetCompression::Uncompressed => Compression::UNCOMPRESSED,
            ParquetCompression::Snappy => Compression::SNAPPY,
            ParquetCompression::Gzip => Compression::GZIP(parquet::basic::GzipLevel::default()),
            ParquetCompression::Lz4 => Compression::LZ4,
            ParquetCompression::Zstd => Compression::ZSTD(parquet::basic::ZstdLevel::default()),
        }
    }

    /// Stub implementations for when persistent-storage feature is disabled
    #[cfg(not(feature = "persistent-storage"))]
    fn save_entities(&self, _graph: &KnowledgeGraph) -> Result<()> {
        Err(GraphRAGError::Config {
            message: "persistent-storage feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "persistent-storage"))]
    fn load_entities(&self) -> Result<Vec<Entity>> {
        Err(GraphRAGError::Config {
            message: "persistent-storage feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "persistent-storage"))]
    fn save_relationships(&self, _graph: &KnowledgeGraph) -> Result<()> {
        Err(GraphRAGError::Config {
            message: "persistent-storage feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "persistent-storage"))]
    fn load_relationships(&self) -> Result<Vec<Relationship>> {
        Err(GraphRAGError::Config {
            message: "persistent-storage feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "persistent-storage"))]
    fn save_chunks(&self, _graph: &KnowledgeGraph) -> Result<()> {
        Err(GraphRAGError::Config {
            message: "persistent-storage feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "persistent-storage"))]
    fn load_chunks(&self) -> Result<Vec<TextChunk>> {
        Err(GraphRAGError::Config {
            message: "persistent-storage feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "persistent-storage"))]
    fn save_documents(&self, _graph: &KnowledgeGraph) -> Result<()> {
        Err(GraphRAGError::Config {
            message: "persistent-storage feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "persistent-storage"))]
    fn load_documents(&self) -> Result<Vec<Document>> {
        Err(GraphRAGError::Config {
            message: "persistent-storage feature not enabled".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_parquet_persistence_creation() {
        let temp_dir = TempDir::new().unwrap();
        let persistence = ParquetPersistence::new(temp_dir.path().to_path_buf()).unwrap();
        assert!(persistence.base_dir.exists());
    }

    #[test]
    #[cfg(feature = "persistent-storage")]
    fn test_save_and_load_entities() {
        let temp_dir = TempDir::new().unwrap();
        let persistence = ParquetPersistence::new(temp_dir.path().to_path_buf()).unwrap();

        let mut graph = KnowledgeGraph::new();
        let entity = Entity::new(
            EntityId::new("test_entity".to_string()),
            "Test Entity".to_string(),
            "PERSON".to_string(),
            0.9,
        );
        graph.add_entity(entity).unwrap();

        // Save
        persistence.save_entities(&graph).unwrap();

        // Load
        let entities = persistence.load_entities().unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "Test Entity");
    }

    /// Entity descriptions survive a parquet round-trip (#97).
    #[test]
    #[cfg(feature = "persistent-storage")]
    fn entity_description_survives_save_and_load_round_trip() {
        let temp_dir = TempDir::new().unwrap();
        let persistence = ParquetPersistence::new(temp_dir.path().to_path_buf()).unwrap();

        let mut graph = KnowledgeGraph::new();
        graph
            .add_entity(
                Entity::new(
                    EntityId::new("alice".to_string()),
                    "Alice".to_string(),
                    "PERSON".to_string(),
                    0.9,
                )
                .with_description("Alice the engineer | Alice the founder".to_string()),
            )
            .unwrap();
        graph
            .add_entity(Entity::new(
                EntityId::new("bob".to_string()),
                "Bob".to_string(),
                "PERSON".to_string(),
                0.8,
            ))
            .unwrap();

        persistence.save_entities(&graph).unwrap();
        let entities = persistence.load_entities().unwrap();

        let alice = entities
            .iter()
            .find(|e| e.name == "Alice")
            .expect("alice present");
        assert_eq!(
            alice.description.as_deref(),
            Some("Alice the engineer | Alice the founder")
        );
        let bob = entities
            .iter()
            .find(|e| e.name == "Bob")
            .expect("bob present");
        assert!(
            bob.description.is_none(),
            "entities without a description must remain None after round-trip"
        );
    }

    // Regression for #24: entity mentions must survive a save/load round
    // trip so entity-to-chunk provenance is preserved for retrieval.
    #[test]
    #[cfg(feature = "persistent-storage")]
    fn entity_mentions_survive_save_and_load_round_trip() {
        use crate::core::EntityMention;

        let temp_dir = TempDir::new().unwrap();
        let persistence = ParquetPersistence::new(temp_dir.path().to_path_buf()).unwrap();

        let mut graph = KnowledgeGraph::new();
        let mentions = vec![
            EntityMention {
                chunk_id: ChunkId::new("chunk-a".to_string()),
                start_offset: 10,
                end_offset: 17,
                confidence: 0.91,
            },
            EntityMention {
                chunk_id: ChunkId::new("chunk-b".to_string()),
                start_offset: 22,
                end_offset: 29,
                confidence: 0.42,
            },
        ];
        let entity = Entity::new(
            EntityId::new("ent-1".to_string()),
            "Alice".to_string(),
            "PERSON".to_string(),
            0.9,
        )
        .with_mentions(mentions.clone());
        graph.add_entity(entity).unwrap();

        persistence.save_entities(&graph).unwrap();

        let entities = persistence.load_entities().unwrap();
        assert_eq!(entities.len(), 1);
        let loaded = &entities[0];
        assert_eq!(loaded.mentions.len(), mentions.len());
        for (got, want) in loaded.mentions.iter().zip(mentions.iter()) {
            assert_eq!(got.chunk_id, want.chunk_id);
            assert_eq!(got.start_offset, want.start_offset);
            assert_eq!(got.end_offset, want.end_offset);
            assert!((got.confidence - want.confidence).abs() < 1e-6);
        }
    }

    // Regression for #24: an entity with no mentions still round-trips
    // successfully (we shouldn't write a malformed sidecar file).
    #[test]
    #[cfg(feature = "persistent-storage")]
    fn entities_without_mentions_round_trip_cleanly() {
        let temp_dir = TempDir::new().unwrap();
        let persistence = ParquetPersistence::new(temp_dir.path().to_path_buf()).unwrap();

        let mut graph = KnowledgeGraph::new();
        graph
            .add_entity(Entity::new(
                EntityId::new("loner".to_string()),
                "Loner".to_string(),
                "THING".to_string(),
                0.5,
            ))
            .unwrap();

        persistence.save_entities(&graph).unwrap();
        let entities = persistence.load_entities().unwrap();
        assert_eq!(entities.len(), 1);
        assert!(entities[0].mentions.is_empty());
    }
}
