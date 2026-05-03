//! Qdrant Vector Store Integration
//!
//! Provides integration with Qdrant vector database for production deployments.
//!
//! ## Features
//!
//! - Store document embeddings with JSON payload metadata
//! - Store entities and relationships as payload
//! - Advanced filtering and search
//! - Collection management
//! - Batch operations
//!
//! ## Usage
//!
//! ```rust
//! let store = QdrantStore::new("http://localhost:6334", "graphrag").await?;
//! store.create_collection(384).await?;
//! store.add_document("doc1", embedding, metadata).await?;
//! let results = store.search(query_embedding, 10, None).await?;
//! ```

use qdrant_client::{
    qdrant::{
        CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointStruct, PointsIdsList,
        ScoredPoint, SearchPointsBuilder, UpsertPointsBuilder, Value as QdrantValue,
        VectorParamsBuilder,
    },
    Qdrant,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::warn;

/// Qdrant store errors
#[derive(Debug, thiserror::Error)]
pub enum QdrantError {
    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Collection error: {0}")]
    CollectionError(String),

    #[error("Operation error: {0}")]
    OperationError(String),

    #[error("Not found: {0}")]
    #[allow(dead_code)]
    NotFound(String),
}

/// Entity stored in Qdrant payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub properties: HashMap<String, serde_json::Value>,
}

/// Relationship stored in Qdrant payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub source: String,
    pub relation: String,
    pub target: String,
    pub properties: HashMap<String, serde_json::Value>,
}

/// Document metadata stored in Qdrant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub id: String,
    pub title: String,
    pub text: String,
    pub chunk_index: usize,
    pub entities: Vec<Entity>,
    pub relationships: Vec<Relationship>,
    pub timestamp: String,
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

/// Search result from Qdrant
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: DocumentMetadata,
}

/// Qdrant vector store
pub struct QdrantStore {
    client: Qdrant,
    collection_name: String,
}

impl QdrantStore {
    /// Create a new Qdrant store
    ///
    /// # Arguments
    /// * `url` - Qdrant server URL (e.g., "http://localhost:6334")
    /// * `collection_name` - Collection name for this graph
    pub async fn new(url: &str, collection_name: &str) -> Result<Self, QdrantError> {
        let client = Qdrant::from_url(url)
            .build()
            .map_err(|e| QdrantError::ConnectionError(e.to_string()))?;

        Ok(Self {
            client,
            collection_name: collection_name.to_string(),
        })
    }

    /// Create a collection with the specified dimension
    ///
    /// # Arguments
    /// * `dimension` - Embedding dimension (e.g., 384 for MiniLM, 768 for BERT)
    pub async fn create_collection(&self, dimension: u64) -> Result<(), QdrantError> {
        self.client
            .create_collection(
                CreateCollectionBuilder::new(&self.collection_name)
                    .vectors_config(VectorParamsBuilder::new(dimension, Distance::Cosine)),
            )
            .await
            .map_err(|e| QdrantError::CollectionError(e.to_string()))?;

        Ok(())
    }

    /// Check if the collection exists.
    ///
    /// Delegates to `qdrant_client::Qdrant::collection_exists`, which uses
    /// the dedicated `CollectionExists` RPC and reports presence/absence
    /// without conflating it with transport, permission, or other RPC
    /// failures. Errors are propagated as `CollectionError` so callers
    /// can fail fast instead of silently treating a transient/permission
    /// failure as "collection does not exist" — that misclassification
    /// would route startup through the create-collection branch and
    /// bypass `verify_collection_dimension` (#37 follow-up from the
    /// Codex review).
    pub async fn collection_exists(&self) -> Result<bool, QdrantError> {
        self.client
            .collection_exists(self.collection_name.clone())
            .await
            .map_err(|e| QdrantError::CollectionError(e.to_string()))
    }

    /// Fetch the configured vector dimension of the existing collection.
    ///
    /// Returns `CollectionError` if the collection cannot be inspected or
    /// its `vectors_config` is missing the expected single-vector
    /// `params.size` field. Used by `verify_collection_dimension` so we
    /// can refuse to start a server whose `EMBEDDING_DIM` doesn't match
    /// the persisted collection (#37).
    pub async fn collection_vector_dim(&self) -> Result<u64, QdrantError> {
        use qdrant_client::qdrant::{vectors_config::Config, VectorsConfig};

        let info = self
            .client
            .collection_info(&self.collection_name)
            .await
            .map_err(|e| QdrantError::CollectionError(e.to_string()))?;

        let params = info
            .result
            .and_then(|r| r.config)
            .and_then(|c| c.params)
            .ok_or_else(|| {
                QdrantError::CollectionError(format!(
                    "collection '{}' has no params",
                    self.collection_name
                ))
            })?;

        let cfg: VectorsConfig = params.vectors_config.ok_or_else(|| {
            QdrantError::CollectionError(format!(
                "collection '{}' has no vectors_config",
                self.collection_name
            ))
        })?;

        match cfg.config {
            Some(Config::Params(p)) => Ok(p.size),
            Some(Config::ParamsMap(_)) => Err(QdrantError::CollectionError(format!(
                "collection '{}' uses named-vector ParamsMap; \
                 dimension check expects a single unnamed vector",
                self.collection_name
            ))),
            None => Err(QdrantError::CollectionError(format!(
                "collection '{}' vectors_config is empty",
                self.collection_name
            ))),
        }
    }

    /// Refuse to use the collection if its persisted vector dimension
    /// disagrees with `expected` (#37).
    ///
    /// Mirrors `LanceDBStore::add_document`'s per-row dimension check
    /// (`graphrag-server/src/lancedb_store.rs:153`), but is invoked once
    /// at startup so a misconfigured `EMBEDDING_DIM` produces a clear
    /// fatal log instead of silently corrupting the collection one
    /// upsert at a time.
    pub async fn verify_collection_dimension(&self, expected: u64) -> Result<(), QdrantError> {
        let actual = self.collection_vector_dim().await?;
        check_collection_dimension(expected, actual, &self.collection_name)
    }

    /// Delete the collection
    #[allow(dead_code)]
    pub async fn delete_collection(&self) -> Result<(), QdrantError> {
        self.client
            .delete_collection(&self.collection_name)
            .await
            .map_err(|e| QdrantError::CollectionError(e.to_string()))?;

        Ok(())
    }

    /// Add a document chunk with metadata
    ///
    /// # Arguments
    /// * `id` - Unique document ID
    /// * `embedding` - Embedding vector
    /// * `metadata` - Document metadata including entities and relationships
    pub async fn add_document(
        &self,
        id: &str,
        embedding: Vec<f32>,
        metadata: DocumentMetadata,
    ) -> Result<(), QdrantError> {
        let payload = metadata_to_payload(&metadata)?;
        let point = PointStruct::new(id.to_string(), embedding, payload);

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, vec![point]))
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        Ok(())
    }

    /// Add multiple document chunks in batch
    #[allow(dead_code)]
    pub async fn add_documents_batch(
        &self,
        documents: Vec<(String, Vec<f32>, DocumentMetadata)>,
    ) -> Result<(), QdrantError> {
        let points: Vec<PointStruct> = documents
            .into_iter()
            .map(|(id, embedding, metadata)| {
                metadata_to_payload(&metadata)
                    .map(|payload| PointStruct::new(id, embedding, payload))
            })
            .collect::<Result<_, _>>()?;

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, points))
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        Ok(())
    }

    /// Search for similar documents
    ///
    /// # Arguments
    /// * `query_embedding` - Query embedding vector
    /// * `limit` - Maximum number of results
    /// * `filter` - Optional filter on metadata fields
    ///
    /// # Returns
    /// Vector of search results with scores and metadata
    pub async fn search(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchResult>, QdrantError> {
        let mut search_builder =
            SearchPointsBuilder::new(&self.collection_name, query_embedding, limit as u64)
                .with_payload(true);

        if let Some(f) = filter {
            search_builder = search_builder.filter(f);
        }

        let results = self
            .client
            .search_points(search_builder)
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        let search_results: Vec<SearchResult> = results
            .result
            .into_iter()
            .filter_map(|point| match try_decode_search_result(point) {
                Ok(result) => Some(result),
                Err(e) => {
                    warn!(error = %e, "skipping qdrant point with undecodable payload");
                    None
                },
            })
            .collect();

        Ok(search_results)
    }

    /// Delete a document by ID
    pub async fn delete_document(&self, id: &str) -> Result<(), QdrantError> {
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection_name).points(PointsIdsList {
                    ids: vec![id.to_string().into()],
                }),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        Ok(())
    }

    /// Delete every point in the collection while preserving the collection
    /// itself (schema, dimension, distance metric).
    ///
    /// Implemented as a single `delete_points` request with a match-all filter
    /// rather than `delete_collection` + `create_collection`, so a transient
    /// failure cannot leave the collection missing or with the wrong dimension.
    ///
    /// This is destructive: every point's vector and payload is removed.
    #[allow(dead_code)]
    pub async fn clear(&self) -> Result<(), QdrantError> {
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection_name)
                    .points(Filter::default())
                    .wait(true),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        Ok(())
    }

    /// Get collection statistics
    pub async fn stats(&self) -> Result<(usize, usize), QdrantError> {
        let info = self
            .client
            .collection_info(&self.collection_name)
            .await
            .map_err(|e| QdrantError::CollectionError(e.to_string()))?;

        let count = info
            .result
            .as_ref()
            .and_then(|c| c.points_count)
            .unwrap_or(0) as usize;

        let vectors = info
            .result
            .as_ref()
            .and_then(|c| c.vectors_count)
            .unwrap_or(0) as usize;

        Ok((count, vectors))
    }

    /// Get the collection name
    #[allow(dead_code)]
    pub fn collection_name(&self) -> &str {
        &self.collection_name
    }
}

/// What startup should do after probing whether the target Qdrant
/// collection already exists.
///
/// Returned by `classify_collection_existence` so the decision can be
/// unit-tested without a live Qdrant server. Aborting on probe failure
/// is the whole point — a swallowed error would skip
/// `verify_collection_dimension` (#37 follow-up).
#[derive(Debug)]
pub(crate) enum CollectionStartupAction {
    /// Probe returned `Ok(false)`: create the collection.
    Create,
    /// Probe returned `Ok(true)`: run the dimension check against the
    /// existing collection before serving traffic.
    VerifyExisting,
    /// Probe failed: refuse to start. Treating this as "does not exist"
    /// would route to `Create` and bypass the dimension verification.
    Abort(QdrantError),
}

/// Map a `collection_exists` probe result to the startup action.
///
/// Pure helper so the regression test for the Codex-reported bug
/// (existence-probe errors silently bypassing #37) can be a unit test.
pub(crate) fn classify_collection_existence(
    probe: Result<bool, QdrantError>,
) -> CollectionStartupAction {
    match probe {
        Ok(true) => CollectionStartupAction::VerifyExisting,
        Ok(false) => CollectionStartupAction::Create,
        Err(e) => CollectionStartupAction::Abort(e),
    }
}

/// Pure dimension-mismatch check, factored out of
/// `verify_collection_dimension` so it can be unit-tested without a
/// live Qdrant server (#37).
fn check_collection_dimension(
    expected: u64,
    actual: u64,
    collection: &str,
) -> Result<(), QdrantError> {
    if expected == actual {
        Ok(())
    } else {
        Err(QdrantError::CollectionError(format!(
            "vector dimension mismatch on collection '{collection}': \
             EMBEDDING_DIM={expected} but stored vectors are {actual}-dim. \
             Refuse to start to avoid corrupting the collection."
        )))
    }
}

/// Convert `DocumentMetadata` into the payload map Qdrant expects.
///
/// Returns `OperationError` if the metadata cannot be JSON-encoded; this
/// should be impossible for the current `DocumentMetadata` shape but we
/// surface the error rather than panic so future schema changes can't
/// crash the worker.
fn metadata_to_payload(
    metadata: &DocumentMetadata,
) -> Result<HashMap<String, QdrantValue>, QdrantError> {
    let payload = serde_json::to_value(metadata)
        .map_err(|e| QdrantError::OperationError(format!("payload encode: {e}")))?;

    let object = payload.as_object().ok_or_else(|| {
        QdrantError::OperationError("payload encode: metadata did not serialize to object".into())
    })?;

    Ok(object
        .clone()
        .into_iter()
        .map(|(k, v)| (k, QdrantValue::from(v)))
        .collect())
}

/// Decode a single Qdrant search hit into our `SearchResult`.
///
/// Returns `OperationError` instead of panicking when the remote payload
/// doesn't match the current `DocumentMetadata` schema or the point id is
/// missing — both can happen when a collection contains data written by
/// another version of the server, by `qdrant-client` CLI, or by a partial
/// upsert. Callers in `search()` log and skip such points.
fn try_decode_search_result(point: ScoredPoint) -> Result<SearchResult, QdrantError> {
    let payload_value = serde_json::to_value(&point.payload)
        .map_err(|e| QdrantError::OperationError(format!("payload re-encode: {e}")))?;
    let metadata: DocumentMetadata = serde_json::from_value(payload_value)
        .map_err(|e| QdrantError::OperationError(format!("payload decode: {e}")))?;

    let id_str = match point.id {
        Some(qdrant_client::qdrant::PointId {
            point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(s)),
        }) => s,
        Some(qdrant_client::qdrant::PointId {
            point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)),
        }) => n.to_string(),
        Some(_) | None => {
            return Err(QdrantError::OperationError(
                "search hit missing point id".into(),
            ));
        },
    };

    Ok(SearchResult {
        id: id_str,
        score: point.score,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use qdrant_client::qdrant::{point_id::PointIdOptions, PointId, ScoredPoint};

    fn point_with_payload(id: Option<PointId>, json: serde_json::Value) -> ScoredPoint {
        let payload: HashMap<String, QdrantValue> = json
            .as_object()
            .expect("test payload must be a JSON object")
            .clone()
            .into_iter()
            .map(|(k, v)| (k, QdrantValue::from(v)))
            .collect();

        ScoredPoint {
            id,
            payload,
            score: 0.5,
            ..Default::default()
        }
    }

    fn good_metadata_json() -> serde_json::Value {
        serde_json::json!({
            "id": "doc1",
            "title": "Test",
            "text": "hello",
            "chunk_index": 0,
            "entities": [],
            "relationships": [],
            "timestamp": "2026-01-01T00:00:00Z",
        })
    }

    // Decoding a well-formed payload + UUID id returns Ok with the right id.
    #[test]
    fn decode_succeeds_on_well_formed_uuid_point() {
        let point = point_with_payload(
            Some(PointId {
                point_id_options: Some(PointIdOptions::Uuid("doc1".into())),
            }),
            good_metadata_json(),
        );

        let result = try_decode_search_result(point).expect("should decode");
        assert_eq!(result.id, "doc1");
        assert_eq!(result.metadata.title, "Test");
    }

    // Decoding a numeric id returns Ok with the id formatted as a string.
    #[test]
    fn decode_succeeds_on_numeric_id() {
        let point = point_with_payload(
            Some(PointId {
                point_id_options: Some(PointIdOptions::Num(42)),
            }),
            good_metadata_json(),
        );

        let result = try_decode_search_result(point).expect("should decode");
        assert_eq!(result.id, "42");
    }

    // A payload missing required fields returns Err instead of panicking
    // (regression for #36 — schema drift used to crash the worker).
    #[test]
    fn decode_returns_err_on_payload_missing_required_field() {
        let bad = serde_json::json!({
            "id": "doc1",
            // intentionally missing title/text/chunk_index/etc
        });
        let point = point_with_payload(
            Some(PointId {
                point_id_options: Some(PointIdOptions::Uuid("doc1".into())),
            }),
            bad,
        );

        let err = try_decode_search_result(point).expect_err("should fail, not panic");
        assert!(
            matches!(err, QdrantError::OperationError(ref msg) if msg.contains("payload decode")),
            "unexpected error: {err:?}"
        );
    }

    // A payload with a wrong-typed required field returns Err, not panic.
    #[test]
    fn decode_returns_err_on_type_mismatch() {
        let mut payload = good_metadata_json();
        payload["chunk_index"] = serde_json::json!("not-a-number");
        let point = point_with_payload(
            Some(PointId {
                point_id_options: Some(PointIdOptions::Uuid("doc1".into())),
            }),
            payload,
        );

        let err = try_decode_search_result(point).expect_err("should fail, not panic");
        assert!(
            matches!(err, QdrantError::OperationError(_)),
            "unexpected error: {err:?}"
        );
    }

    // A point with no id returns Err instead of panicking on .unwrap().
    #[test]
    fn decode_returns_err_on_missing_point_id() {
        let point = point_with_payload(None, good_metadata_json());
        let err = try_decode_search_result(point).expect_err("should fail, not panic");
        assert!(
            matches!(err, QdrantError::OperationError(ref msg) if msg.contains("missing point id")),
            "unexpected error: {err:?}"
        );
    }

    // A point with a PointId whose oneof is None also returns Err.
    #[test]
    fn decode_returns_err_on_empty_point_id_oneof() {
        let point = point_with_payload(
            Some(PointId {
                point_id_options: None,
            }),
            good_metadata_json(),
        );
        let err = try_decode_search_result(point).expect_err("should fail, not panic");
        assert!(matches!(err, QdrantError::OperationError(_)));
    }

    // check_collection_dimension passes when expected == actual (#37).
    #[test]
    fn check_collection_dimension_accepts_match() {
        check_collection_dimension(384, 384, "graphrag").expect("matching dims must pass");
    }

    // check_collection_dimension surfaces a CollectionError when the
    // configured EMBEDDING_DIM differs from the live collection's stored
    // vector size, with the actual/expected values in the message so the
    // operator can act on it. (#37 — previously a 768-dim embedding shipped
    // against a 384-dim collection silently corrupted the index.)
    #[test]
    fn check_collection_dimension_errors_on_mismatch() {
        let err = check_collection_dimension(768, 384, "graphrag")
            .expect_err("mismatched dims must error");
        let msg = match err {
            QdrantError::CollectionError(s) => s,
            other => panic!("expected CollectionError, got {other:?}"),
        };
        assert!(
            msg.contains("graphrag"),
            "msg should name collection: {msg}"
        );
        assert!(msg.contains("384"), "msg should include actual dim: {msg}");
        assert!(
            msg.contains("768"),
            "msg should include expected dim: {msg}"
        );
    }

    // classify_collection_existence reports Create when the existence
    // probe returns Ok(false), so startup takes the create-collection
    // branch.
    #[test]
    fn classify_returns_create_when_collection_absent() {
        match classify_collection_existence(Ok(false)) {
            CollectionStartupAction::Create => {},
            other => panic!("expected Create, got {other:?}"),
        }
    }

    // classify_collection_existence reports VerifyExisting when the
    // existence probe returns Ok(true), so startup runs the dimension
    // check against the live collection (#37).
    #[test]
    fn classify_returns_verify_when_collection_present() {
        match classify_collection_existence(Ok(true)) {
            CollectionStartupAction::VerifyExisting => {},
            other => panic!("expected VerifyExisting, got {other:?}"),
        }
    }

    // Regression: a transient/permission failure on the existence probe
    // must abort startup instead of being coerced to "collection does
    // not exist". Previously `collection_exists` returned `Ok(false)` on
    // any error, so `main.rs` took the create-collection branch and
    // skipped `verify_collection_dimension` — silently bypassing the
    // #37 fail-fast dimension check (Codex review follow-up).
    #[test]
    fn classify_aborts_on_existence_probe_error() {
        let err = QdrantError::CollectionError("rpc unavailable".into());
        match classify_collection_existence(Err(err)) {
            CollectionStartupAction::Abort(QdrantError::CollectionError(msg)) => {
                assert!(
                    msg.contains("rpc unavailable"),
                    "abort should preserve underlying error: {msg}"
                );
            },
            other => panic!("expected Abort(CollectionError), got {other:?}"),
        }
    }

    // metadata_to_payload round-trips a normal DocumentMetadata.
    #[test]
    fn metadata_to_payload_round_trips() {
        let metadata = DocumentMetadata {
            id: "doc1".into(),
            title: "Test".into(),
            text: "hello".into(),
            chunk_index: 3,
            entities: vec![],
            relationships: vec![],
            timestamp: "2026-01-01T00:00:00Z".into(),
            custom: HashMap::new(),
        };

        let payload = metadata_to_payload(&metadata).expect("should encode");
        assert!(payload.contains_key("id"));
        assert!(payload.contains_key("chunk_index"));
    }

    #[tokio::test]
    #[ignore] // Requires Qdrant server running
    async fn test_qdrant_store() {
        let store = QdrantStore::new("http://localhost:6334", "test-collection")
            .await
            .unwrap();
        store.create_collection(384).await.unwrap();

        let metadata = DocumentMetadata {
            id: "doc1".to_string(),
            title: "Test Document".to_string(),
            text: "This is a test document".to_string(),
            chunk_index: 0,
            entities: vec![],
            relationships: vec![],
            timestamp: chrono::Utc::now().to_rfc3339(),
            custom: HashMap::new(),
        };

        store
            .add_document("doc1", vec![0.1; 384], metadata)
            .await
            .unwrap();

        let results = store.search(vec![0.1; 384], 10, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");

        store.delete_collection().await.unwrap();
    }

    // Integration test for #38: clear() must preserve the collection (and its
    // dimension) even though it removes all points. Requires Qdrant server.
    #[tokio::test]
    #[ignore]
    async fn clear_preserves_collection_schema() {
        let store = QdrantStore::new("http://localhost:6334", "test-clear-collection")
            .await
            .unwrap();
        // start clean
        let _ = store.delete_collection().await;
        store.create_collection(384).await.unwrap();

        let metadata = DocumentMetadata {
            id: "doc1".to_string(),
            title: "Test".to_string(),
            text: "hello".to_string(),
            chunk_index: 0,
            entities: vec![],
            relationships: vec![],
            timestamp: chrono::Utc::now().to_rfc3339(),
            custom: HashMap::new(),
        };
        store
            .add_document("doc1", vec![0.1; 384], metadata)
            .await
            .unwrap();

        store.clear().await.unwrap();

        // Collection must still exist with the original dimension — adding
        // a 384-dim vector after clear() must succeed.
        assert!(store.collection_exists().await.unwrap());
        let metadata = DocumentMetadata {
            id: "doc2".to_string(),
            title: "Test2".to_string(),
            text: "hello again".to_string(),
            chunk_index: 0,
            entities: vec![],
            relationships: vec![],
            timestamp: chrono::Utc::now().to_rfc3339(),
            custom: HashMap::new(),
        };
        store
            .add_document("doc2", vec![0.2; 384], metadata)
            .await
            .expect("collection schema should be preserved");

        store.delete_collection().await.unwrap();
    }
}
