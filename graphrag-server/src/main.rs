//! GraphRAG REST API server (Actix-web + Apistos OpenAPI).
//!
//! See the README for setup, feature flags, and Swagger UI usage.

use actix_cors::Cors;
use actix_web::{
    web::{self, Data, Json, Path as WebPath},
    App, HttpServer, Responder,
};
use apistos::{
    api_operation,
    app::OpenApiWrapper,
    info::Info,
    spec::Spec,
    web::{delete, get, post, resource, scope},
};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing_subscriber;

mod models;
use models::*;

#[cfg(feature = "qdrant")]
mod qdrant_store;
#[cfg(feature = "qdrant")]
use qdrant_store::{
    classify_collection_existence, CollectionStartupAction, DocumentMetadata, QdrantStore,
};

#[cfg(feature = "auth")]
mod auth;
#[cfg(feature = "auth")]
use auth::AuthState;

mod embeddings;
use embeddings::{EmbeddingConfig, EmbeddingService};

mod validation;
use validation::{
    sanitize_string, validate_content, validate_query, validate_title, validate_top_k,
};

mod config_handler;
use config_handler::ConfigManager;

mod config_endpoints;

// Import full GraphRAG pipeline
use graphrag_core::GraphRAG;

/// Application state with optional Qdrant backend and full GraphRAG pipeline
#[derive(Clone)]
struct AppState {
    #[cfg(feature = "qdrant")]
    qdrant: Option<Arc<QdrantStore>>,

    // Embedding service (real or fallback)
    embeddings: Arc<EmbeddingService>,

    // Full GraphRAG pipeline (when configured via JSON)
    graphrag: Arc<RwLock<Option<GraphRAG>>>,

    // Configuration manager for JSON config
    config_manager: Arc<ConfigManager>,

    // Authentication state (optional)
    #[cfg(feature = "auth")]
    auth: Arc<AuthState>,

    // Fallback in-memory storage (used when Qdrant unavailable or simple mode)
    documents: Arc<RwLock<Vec<Document>>>,
    graph_built: Arc<RwLock<bool>>,
    query_count: Arc<RwLock<usize>>,
}

/// Build an `AuthState` from `JWT_SECRET`, exiting the process if the
/// env var is missing or the value is shorter than the HS256 minimum
/// (#31). Centralised so all three `AppState::new` arms — Qdrant
/// connected, Qdrant unavailable, Qdrant feature off — share one rule
/// and can't drift out of sync.
#[cfg(feature = "auth")]
fn load_auth_state() -> AuthState {
    let secret = match std::env::var("JWT_SECRET") {
        Ok(s) => s,
        Err(_) => {
            tracing::error!(
                "❌ JWT_SECRET is not set. Refusing to start the auth-enabled \
                 server with a default secret. Set JWT_SECRET to a value of \
                 at least {} bytes.",
                auth::JWT_SECRET_MIN_BYTES
            );
            std::process::exit(1);
        },
    };
    match AuthState::try_new(secret) {
        Ok(state) => state,
        Err(e) => {
            // `try_new`'s error message names the threshold and the
            // observed length — never the secret value itself (#31).
            tracing::error!("❌ JWT_SECRET rejected: {}. Refusing to start.", e);
            std::process::exit(1);
        },
    }
}

impl AppState {
    async fn new() -> Self {
        // Initialize embedding service
        let embedding_backend =
            std::env::var("EMBEDDING_BACKEND").unwrap_or_else(|_| "hash".to_string()); // Default to hash fallback
        let embedding_dim: usize = std::env::var("EMBEDDING_DIM")
            .unwrap_or_else(|_| "384".to_string())
            .parse()
            .unwrap_or(384);

        let embedding_config = EmbeddingConfig {
            backend: embedding_backend,
            dimension: embedding_dim,
            ollama_url: std::env::var("OLLAMA_URL")
                .unwrap_or_else(|_| "http://localhost".to_string()),
            ollama_model: std::env::var("OLLAMA_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "nomic-embed-text".to_string()),
            enable_cache: true,
        };

        let embeddings = match EmbeddingService::new(embedding_config).await {
            Ok(service) => {
                tracing::info!(
                    "✅ Embedding service initialized: {}",
                    service.backend_name()
                );
                Arc::new(service)
            },
            Err(e) => {
                tracing::error!(
                    "❌ Failed to initialize embedding service: {}. Server may not work correctly.",
                    e
                );
                std::process::exit(1);
            },
        };

        #[cfg(feature = "qdrant")]
        {
            // Try to connect to Qdrant
            let qdrant_url =
                std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());
            let collection_name =
                std::env::var("COLLECTION_NAME").unwrap_or_else(|_| "graphrag".to_string());

            match QdrantStore::new(&qdrant_url, &collection_name).await {
                Ok(store) => {
                    // Probe whether the collection exists. A probe failure
                    // (transport, permission, etc.) MUST abort startup —
                    // treating it as "does not exist" would route to the
                    // create-collection branch and bypass the #37
                    // dimension check on the existing collection.
                    match classify_collection_existence(store.collection_exists().await) {
                        CollectionStartupAction::Create => {
                            match store.create_collection(embedding_dim as u64).await {
                                Ok(_) => {
                                    tracing::info!(
                                        "✅ Created Qdrant collection: {}",
                                        collection_name
                                    );
                                },
                                Err(e) => {
                                    tracing::warn!("⚠️  Could not create collection: {}", e);
                                },
                            }
                        },
                        CollectionStartupAction::VerifyExisting => {
                            // Refuse to start if EMBEDDING_DIM disagrees with the
                            // collection's stored vector size (#37). A silent
                            // mismatch otherwise corrupts the index one upsert
                            // at a time. Mirrors the LanceDB per-row dim check.
                            if let Err(e) = store
                                .verify_collection_dimension(embedding_dim as u64)
                                .await
                            {
                                tracing::error!(
                                    "❌ Qdrant collection dimension check failed: {}. \
                                     Set EMBEDDING_DIM to match the existing collection, \
                                     use a different COLLECTION_NAME, or recreate the \
                                     collection.",
                                    e
                                );
                                std::process::exit(1);
                            }
                            tracing::info!(
                                "✅ Connected to existing Qdrant collection: {} ({} dims)",
                                collection_name,
                                embedding_dim
                            );
                        },
                        CollectionStartupAction::Abort(e) => {
                            tracing::error!(
                                "❌ Could not determine if Qdrant collection '{}' exists: {}. \
                                 Refusing to start: a probe failure could otherwise route to \
                                 the create-collection path and skip the EMBEDDING_DIM check \
                                 against an existing collection (#37).",
                                collection_name,
                                e
                            );
                            std::process::exit(1);
                        },
                    }

                    tracing::info!("🗄️  Using Qdrant at: {}", qdrant_url);

                    Self {
                        qdrant: Some(Arc::new(store)),
                        embeddings,
                        graphrag: Arc::new(RwLock::new(None)),
                        config_manager: Arc::new(ConfigManager::new()),
                        #[cfg(feature = "auth")]
                        auth: Arc::new(load_auth_state()),
                        documents: Arc::new(RwLock::new(Vec::new())),
                        graph_built: Arc::new(RwLock::new(false)),
                        query_count: Arc::new(RwLock::new(0)),
                    }
                },
                Err(e) => {
                    tracing::warn!(
                        "⚠️  Could not connect to Qdrant: {}. Using in-memory storage.",
                        e
                    );
                    Self {
                        qdrant: None,
                        embeddings,
                        graphrag: Arc::new(RwLock::new(None)),
                        config_manager: Arc::new(ConfigManager::new()),
                        #[cfg(feature = "auth")]
                        auth: Arc::new(load_auth_state()),
                        documents: Arc::new(RwLock::new(Vec::new())),
                        graph_built: Arc::new(RwLock::new(false)),
                        query_count: Arc::new(RwLock::new(0)),
                    }
                },
            }
        }

        #[cfg(not(feature = "qdrant"))]
        {
            tracing::info!("📦 Using in-memory storage (Qdrant feature disabled)");
            Self {
                embeddings,
                graphrag: Arc::new(RwLock::new(None)),
                config_manager: Arc::new(ConfigManager::new()),
                #[cfg(feature = "auth")]
                auth: Arc::new(load_auth_state()),
                documents: Arc::new(RwLock::new(Vec::new())),
                graph_built: Arc::new(RwLock::new(false)),
                query_count: Arc::new(RwLock::new(0)),
            }
        }
    }

    /// Check if Qdrant is available
    fn has_qdrant(&self) -> bool {
        #[cfg(feature = "qdrant")]
        {
            self.qdrant.is_some()
        }
        #[cfg(not(feature = "qdrant"))]
        {
            false
        }
    }
}

// ============================================================================
// API Handlers
// ============================================================================

/// Root endpoint - API information
#[api_operation(
    tag = "info",
    summary = "Get API information",
    description = "Returns basic information about the GraphRAG API, including version, status, and available endpoints"
)]
async fn root(state: Data<AppState>) -> impl Responder {
    Json(json!({
        "name": "GraphRAG REST API",
        "version": env!("CARGO_PKG_VERSION"),
        "status": "running",
        "backend": if state.has_qdrant() { "qdrant" } else { "memory" },
        "graphrag_configured": state.graphrag.read().await.is_some(),
        "documentation": "/swagger",
        "openapi_spec": "/openapi.json",
        "endpoints": {
            "health": "GET /health",
            "config": {
                "get": "GET /api/config - Get current configuration",
                "set": "POST /api/config - Set configuration and initialize GraphRAG",
                "template": "GET /api/config/template - Get configuration templates and examples",
                "default": "GET /api/config/default - Get default configuration",
                "validate": "POST /api/config/validate - Validate configuration without applying"
            },
            "query": "POST /api/query",
            "documents": {
                "list": "GET /api/documents",
                "add": "POST /api/documents",
                "delete": "DELETE /api/documents/{id}"
            },
            "graph": {
                "build": "POST /api/graph/build",
                "stats": "GET /api/graph/stats"
            }
        }
    }))
}

/// Health check endpoint
#[api_operation(
    tag = "health",
    summary = "Health check",
    description = "Returns the current health status of the service, including document count, graph status, and total queries processed"
)]
async fn health(state: Data<AppState>) -> Result<Json<HealthResponse>, ApiError> {
    let doc_count;
    let graph_built;
    let query_count = *state.query_count.read().await;

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.stats().await {
            Ok((count, _)) => {
                doc_count = count;
                graph_built = count > 0;
            },
            Err(_) => {
                doc_count = 0;
                graph_built = false;
            },
        }
    } else {
        doc_count = state.documents.read().await.len();
        graph_built = *state.graph_built.read().await;
    }

    #[cfg(not(feature = "qdrant"))]
    {
        doc_count = state.documents.read().await.len();
        graph_built = *state.graph_built.read().await;
    }

    Ok(Json(HealthResponse {
        status: "healthy".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        document_count: doc_count,
        graph_built,
        total_queries: query_count,
        backend: if state.has_qdrant() {
            "qdrant".to_string()
        } else {
            "memory".to_string()
        },
    }))
}

/// Query the knowledge graph
#[api_operation(
    tag = "query",
    summary = "Query the knowledge graph",
    description = "Search documents using semantic similarity. Returns ranked results with similarity scores.",
    error_code = 400,
    error_code = 500
)]
async fn query(
    state: Data<AppState>,
    body: Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    // Validate input
    if let Err(e) = validate_query(&body.query) {
        tracing::warn!(query = %body.query, error = %e.error, "Invalid query");
        return Err(ApiError::BadRequest(e.error));
    }

    if let Err(e) = validate_top_k(body.top_k) {
        tracing::warn!(top_k = body.top_k, error = %e.error, "Invalid top_k");
        return Err(ApiError::BadRequest(e.error));
    }

    let start = std::time::Instant::now();

    // Note: query_count is incremented on success only (see #50). The
    // previous unconditional pre-increment drifted the /health metric
    // upward on every embedding/search failure.

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // Real vector search with Qdrant using real embeddings
        let query_embedding = match state.embeddings.generate_single(&body.query).await {
            Ok(embedding) => embedding,
            Err(e) => {
                // Diagnostic detail goes to logs only; the response carries
                // a generic component name (issue #41).
                tracing::error!("Failed to generate query embedding: {}", e);
                return Err(ApiError::InternalError(
                    "embedding generation failed".to_string(),
                ));
            },
        };

        match qdrant.search(query_embedding, body.top_k, None).await {
            Ok(search_results) => {
                let results: Vec<QueryResult> = search_results
                    .into_iter()
                    .map(|r| QueryResult {
                        document_id: r.id,
                        title: r.metadata.title,
                        similarity: r.score,
                        excerpt: if r.metadata.text.len() > 200 {
                            // Clamp to UTF-8 char boundary so multi-byte input
                            // (emoji, accented characters, CJK) doesn't panic.
                            format!(
                                "{}...",
                                graphrag_core::util::text_safe::truncate_chars(
                                    &r.metadata.text,
                                    200
                                )
                            )
                        } else {
                            r.metadata.text
                        },
                    })
                    .collect();

                let processing_time = start.elapsed().as_millis() as u64;

                // Count only successful queries (#50)
                *state.query_count.write().await += 1;

                return Ok(Json(QueryResponse {
                    query: body.query.clone(),
                    results,
                    processing_time_ms: processing_time,
                    backend: "qdrant".to_string(),
                }));
            },
            Err(e) => {
                tracing::error!("Qdrant search failed: {}", e);
                return Err(ApiError::InternalError("vector search failed".to_string()));
            },
        }
    }

    // Fallback: in-memory search
    let documents = state.documents.read().await;

    if documents.is_empty() {
        return Err(ApiError::BadRequest(
            "No documents available. Add documents first.".to_string(),
        ));
    }

    // Simple keyword matching for demonstration
    let mut results: Vec<QueryResult> = documents
        .iter()
        .map(|doc| {
            let query_lower = body.query.to_lowercase();
            let content_lower = doc.content.to_lowercase();
            let title_lower = doc.title.to_lowercase();

            let similarity =
                if content_lower.contains(&query_lower) || title_lower.contains(&query_lower) {
                    0.85
                } else {
                    0.1
                };

            let excerpt = if doc.content.len() > 200 {
                format!(
                    "{}...",
                    graphrag_core::util::text_safe::truncate_chars(&doc.content, 200)
                )
            } else {
                doc.content.clone()
            };

            QueryResult {
                document_id: doc.id.clone(),
                title: doc.title.clone(),
                similarity,
                excerpt,
            }
        })
        .filter(|r| r.similarity > 0.5)
        .collect();

    results.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
    results.truncate(body.top_k);

    let processing_time = start.elapsed().as_millis() as u64;

    // Count only successful queries (#50)
    *state.query_count.write().await += 1;

    Ok(Json(QueryResponse {
        query: body.query.clone(),
        results,
        processing_time_ms: processing_time,
        backend: "memory".to_string(),
    }))
}

/// Add a document to the knowledge graph
#[api_operation(
    tag = "documents",
    summary = "Add a new document",
    description = "Add a new document to the knowledge graph. The document will be embedded and indexed for search.",
    error_code = 400,
    error_code = 500
)]
async fn add_document(
    state: Data<AppState>,
    body: Json<AddDocumentRequest>,
) -> Result<Json<DocumentOperationResponse>, ApiError> {
    // Validate input
    if let Err(e) = validate_title(&body.title) {
        tracing::warn!(title = %body.title, error = %e.error, "Invalid title");
        return Err(ApiError::BadRequest(e.error));
    }

    if let Err(e) = validate_content(&body.content) {
        tracing::warn!(content_len = body.content.len(), error = %e.error, "Invalid content");
        return Err(ApiError::BadRequest(e.error));
    }

    // Sanitize inputs
    let title = sanitize_string(&body.title);
    let content = sanitize_string(&body.content);

    let id = uuid::Uuid::new_v4().to_string();
    let timestamp = chrono::Utc::now().to_rfc3339();

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // Chunk before embedding (#35).
        //
        // The previous code passed the entire `content` (up to 5 MB) to
        // `embeddings.generate_single` and to Qdrant as a single dense
        // vector. Most embedding models have an 8k–32k token window; on
        // overflow the backend either errors or silently truncates, and the
        // hash fallback would have produced one vector that "represents" 5
        // MB of text — which dominates similarity search and effectively
        // poisons the index.
        let chunks = chunk_for_embedding(&content);
        if chunks.len() > MAX_CHUNKS_PER_DOCUMENT {
            tracing::warn!(
                document_id = %id,
                chunks = chunks.len(),
                "document chunked into more than MAX_CHUNKS_PER_DOCUMENT"
            );
            return Err(ApiError::BadRequest(format!(
                "Document chunks ({}) exceed maximum of {}. Split the document or increase chunk_size.",
                chunks.len(),
                MAX_CHUNKS_PER_DOCUMENT
            )));
        }
        let chunk_count = chunks.len();
        for (chunk_index, chunk_text) in chunks.iter().enumerate() {
            let embedding = match state.embeddings.generate_single(chunk_text).await {
                Ok(emb) => emb,
                Err(e) => {
                    tracing::error!(
                        document_id = %id,
                        chunk_index,
                        "Failed to generate chunk embedding: {}",
                        e
                    );
                    return Err(ApiError::InternalError(
                        "embedding generation failed".to_string(),
                    ));
                },
            };

            // Per-chunk Qdrant id derived from doc id + chunk index. Each
            // chunk also carries the original `id` and `title` in its
            // payload so cross-chunk retrieval can rejoin them.
            let chunk_id = format!("{}#{:04}", id, chunk_index);
            let metadata = DocumentMetadata {
                id: chunk_id.clone(),
                title: title.clone(),
                text: (*chunk_text).to_string(),
                chunk_index,
                entities: Vec::new(),
                relationships: Vec::new(),
                timestamp: timestamp.clone(),
                custom: {
                    let mut m = HashMap::new();
                    m.insert(
                        "parent_document_id".to_string(),
                        serde_json::Value::String(id.clone()),
                    );
                    m
                },
            };

            if let Err(e) = qdrant.add_document(&chunk_id, embedding, metadata).await {
                tracing::error!(
                    document_id = %id,
                    chunk_index,
                    "Failed to add chunk to Qdrant: {}",
                    e
                );
                return Err(ApiError::InternalError(
                    "document storage failed".to_string(),
                ));
            }
        }
        tracing::info!(
            "Added document to Qdrant: {} ({}) — {} chunks",
            title,
            id,
            chunk_count
        );
        return Ok(Json(DocumentOperationResponse {
            success: true,
            document_id: Some(id),
            message: format!(
                "Document added to Qdrant successfully ({} chunks)",
                chunk_count
            ),
            backend: "qdrant".to_string(),
        }));
    }

    // Fallback: in-memory storage
    let document = Document {
        id: id.clone(),
        title,
        content,
        added_at: timestamp,
    };

    state.documents.write().await.push(document.clone());
    *state.graph_built.write().await = false;

    tracing::info!("Added document to memory: {} ({})", document.title, id);

    Ok(Json(DocumentOperationResponse {
        success: true,
        document_id: Some(id),
        message: "Document added to memory successfully".to_string(),
        backend: "memory".to_string(),
    }))
}

/// List all documents
#[api_operation(
    tag = "documents",
    summary = "List all documents",
    description = "Retrieve a list of all documents in the knowledge graph"
)]
async fn list_documents(state: Data<AppState>) -> Json<ListDocumentsResponse> {
    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.stats().await {
            Ok((count, _vectors)) => {
                return Json(ListDocumentsResponse {
                    documents: Vec::new(),
                    total: count,
                    backend: "qdrant".to_string(),
                    note: Some("Full document listing from Qdrant not implemented yet".to_string()),
                });
            },
            Err(e) => {
                tracing::error!("Failed to get Qdrant stats: {}", e);
            },
        }
    }

    // Fallback: in-memory storage
    let documents = state.documents.read().await;

    let doc_list: Vec<DocumentSummary> = documents
        .iter()
        .map(|doc| DocumentSummary {
            id: doc.id.clone(),
            title: doc.title.clone(),
            content_length: doc.content.len(),
            added_at: doc.added_at.clone(),
        })
        .collect();

    Json(ListDocumentsResponse {
        documents: doc_list.clone(),
        total: doc_list.len(),
        backend: "memory".to_string(),
        note: None,
    })
}

/// Delete a document
#[api_operation(
    tag = "documents",
    summary = "Delete a document",
    description = "Remove a document from the knowledge graph by ID",
    error_code = 404,
    error_code = 500
)]
async fn delete_document(
    state: Data<AppState>,
    id: WebPath<String>,
) -> Result<Json<DocumentOperationResponse>, ApiError> {
    let doc_id = id.into_inner();

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.delete_document(&doc_id).await {
            Ok(_) => {
                tracing::info!("Deleted document from Qdrant: {}", doc_id);
                return Ok(Json(DocumentOperationResponse {
                    success: true,
                    document_id: Some(doc_id.clone()),
                    message: format!("Document {} deleted from Qdrant", doc_id),
                    backend: "qdrant".to_string(),
                }));
            },
            Err(e) => {
                tracing::error!("Failed to delete from Qdrant: {}", e);
                return Err(ApiError::InternalError(
                    "document delete failed".to_string(),
                ));
            },
        }
    }

    // Fallback: in-memory storage
    let mut documents = state.documents.write().await;
    let original_len = documents.len();
    documents.retain(|doc| doc.id != doc_id);

    if documents.len() == original_len {
        return Err(ApiError::NotFound(format!(
            "Document with id '{}' not found",
            doc_id
        )));
    }

    *state.graph_built.write().await = false;
    tracing::info!("Deleted document from memory: {}", doc_id);

    Ok(Json(DocumentOperationResponse {
        success: true,
        document_id: Some(doc_id.clone()),
        message: format!("Document {} deleted from memory", doc_id),
        backend: "memory".to_string(),
    }))
}

/// Build the knowledge graph
#[api_operation(
    tag = "graph",
    summary = "Build the knowledge graph",
    description = "Process all documents and build the knowledge graph structure",
    error_code = 400,
    error_code = 500
)]
async fn build_graph(state: Data<AppState>) -> Result<Json<BuildGraphResponse>, ApiError> {
    let start = std::time::Instant::now();

    // Try the real GraphRAG pipeline first
    {
        let mut graphrag_guard = state.graphrag.write().await;
        if let Some(ref mut graphrag) = *graphrag_guard {
            // Use actual pipeline to build graph
            match graphrag.build_graph().await {
                Ok(_) => {
                    let processing_time = start.elapsed().as_millis() as u64;
                    let (entities, relationships) = graphrag
                        .knowledge_graph()
                        .map(|kg| (kg.entities().count(), kg.relationships().count()))
                        .unwrap_or((0, 0));

                    *state.graph_built.write().await = true;

                    tracing::info!(
                        "Built knowledge graph via pipeline in {}ms ({} entities, {} relationships)",
                        processing_time, entities, relationships
                    );

                    return Ok(Json(BuildGraphResponse {
                        success: true,
                        document_count: state.documents.read().await.len(),
                        processing_time_ms: processing_time,
                        message: format!(
                            "Knowledge graph built: {} entities, {} relationships",
                            entities, relationships
                        ),
                        backend: "graphrag-pipeline".to_string(),
                    }));
                },
                Err(e) => {
                    tracing::warn!("GraphRAG pipeline build failed, trying fallback: {}", e);
                    // Fall through to lower-priority backends
                },
            }
        }
    }

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.stats().await {
            Ok((count, _)) => {
                if count == 0 {
                    return Err(ApiError::BadRequest(
                        "No documents in Qdrant. Add documents first.".to_string(),
                    ));
                }

                let processing_time = start.elapsed().as_millis() as u64;

                tracing::info!(
                    "Built knowledge graph from {} Qdrant documents in {}ms",
                    count,
                    processing_time
                );

                *state.graph_built.write().await = true;

                return Ok(Json(BuildGraphResponse {
                    success: true,
                    document_count: count,
                    processing_time_ms: processing_time,
                    message: "Knowledge graph built from Qdrant successfully".to_string(),
                    backend: "qdrant".to_string(),
                }));
            },
            Err(e) => {
                tracing::error!("Failed to access Qdrant: {}", e);
                return Err(ApiError::InternalError(
                    "vector store access failed".to_string(),
                ));
            },
        }
    }

    // Fallback: in-memory storage
    let doc_count = state.documents.read().await.len();

    if doc_count == 0 {
        return Err(ApiError::BadRequest(
            "No documents to build graph from. Add documents first.".to_string(),
        ));
    }

    *state.graph_built.write().await = true;
    let processing_time = start.elapsed().as_millis() as u64;

    tracing::info!(
        "Built knowledge graph from {} memory documents in {}ms",
        doc_count,
        processing_time
    );

    Ok(Json(BuildGraphResponse {
        success: true,
        document_count: doc_count,
        processing_time_ms: processing_time,
        message: "Knowledge graph built from memory successfully".to_string(),
        backend: "memory".to_string(),
    }))
}

/// Get graph statistics
#[api_operation(
    tag = "graph",
    summary = "Get graph statistics",
    description = "Retrieve statistics about the knowledge graph, including document count, entity count, and relationship count"
)]
async fn graph_stats(state: Data<AppState>) -> Json<GraphStatsResponse> {
    // Try real GraphRAG pipeline stats first
    {
        let graphrag_guard = state.graphrag.read().await;
        if let Some(ref graphrag) = *graphrag_guard {
            if let Some(kg) = graphrag.knowledge_graph() {
                let entity_count = kg.entities().count();
                let relationship_count = kg.relationships().count();
                let doc_count = kg.documents().count();
                let chunk_count = kg.chunks().count();

                return Json(GraphStatsResponse {
                    document_count: doc_count,
                    entity_count,
                    relationship_count,
                    vector_count: chunk_count,
                    graph_built: true,
                    backend: "graphrag-pipeline".to_string(),
                });
            }
        }
    }

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.stats().await {
            Ok((count, vectors)) => {
                return Json(GraphStatsResponse {
                    document_count: count,
                    entity_count: 0,
                    relationship_count: 0,
                    vector_count: vectors,
                    graph_built: count > 0,
                    backend: "qdrant".to_string(),
                });
            },
            Err(e) => {
                tracing::error!("Failed to get Qdrant stats: {}", e);
            },
        }
    }

    // Fallback: in-memory storage
    let doc_count = state.documents.read().await.len();
    let graph_built = *state.graph_built.read().await;

    Json(GraphStatsResponse {
        document_count: doc_count,
        entity_count: 0,
        relationship_count: 0,
        vector_count: 0,
        graph_built,
        backend: "memory".to_string(),
    })
}

// ============================================================================
// Authentication Endpoints (feature-gated)
// ============================================================================

// Note: `#[api_operation]` deliberately omitted on the auth handlers. The
// `error_code` arm of the macro doesn't currently resolve against
// `ApiError`'s `ApiErrorComponent` schema, and these routes are not
// mounted (see the commented-out `/auth/*` block in `main()`). Re-add
// the macro alongside the actix port (issue #40).
#[cfg(feature = "auth")]
#[allow(dead_code)]
async fn login(
    _state: Data<AppState>,
    body: Json<LoginRequest>,
) -> Result<Json<LoginResponse>, ApiError> {
    // SECURITY: the previous body issued an Admin-role JWT to any caller
    // submitting `{"username":"admin","password":"anything"}` — no
    // password verification, no user store, no bcrypt. Even though the
    // route is currently NOT mounted (see the commented-out `/auth/*`
    // section in `main()`), the handler is shipped in the binary and
    // would be live the moment someone wires the routes.
    //
    // Refuse to issue tokens until a real credential store is in place
    // (#30). The bcrypt dep declared under the `auth` feature is the
    // intended verification path; backing it requires a persisted
    // `(username, bcrypt_hash, role)` store that doesn't exist yet.
    tracing::warn!(
        username = %body.username,
        "Login attempt rejected: no credential store configured (see #30)"
    );
    Err(ApiError::InternalError(
        "Login is not configured on this deployment. Set up a credential store first.".to_string(),
    ))
}

#[cfg(feature = "auth")]
#[allow(dead_code)]
async fn create_api_key(
    state: Data<AppState>,
    body: Json<ApiKeyRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let role = body
        .role
        .as_deref()
        .and_then(|r| match r {
            "Admin" => Some(auth::UserRole::Admin),
            _ => Some(auth::UserRole::User),
        })
        .unwrap_or(auth::UserRole::User);

    match state
        .auth
        .create_api_key(&body.user_id, role.clone(), None)
        .await
    {
        Ok(api_key) => {
            tracing::info!(
                "✅ Created API key for user: {} (role: {:?})",
                body.user_id,
                role
            );
            Ok(Json(json!({
                "success": true,
                "api_key": api_key,
                "user_id": body.user_id,
                "role": format!("{:?}", role),
                "usage": "Add header: Authorization: ApiKey <key>",
                "rate_limit": {
                    "max_requests": 1000,
                    "window_seconds": 3600
                }
            })))
        },
        Err(e) => {
            tracing::error!("❌ Failed to create API key: {}", e);
            Err(ApiError::InternalError(
                "API key creation failed".to_string(),
            ))
        },
    }
}

// ============================================================================
// Main Server Configuration
// ============================================================================

/// Target chunk size in bytes for `add_document`'s in-handler chunker.
/// ~4000 bytes is well under the 8k–32k token windows of every common
/// embedding model (a "token" is on the order of 3–4 bytes for English).
const EMBEDDING_CHUNK_SIZE: usize = 4000;

/// Overlap between adjacent chunks (helps retrieval find spans that
/// straddle a chunk boundary).
const EMBEDDING_CHUNK_OVERLAP: usize = 400;

/// Maximum chunks per document. With `EMBEDDING_CHUNK_SIZE = 4000` this
/// caps a single \`add_document\` request at ~2 MB of effective text — past
/// that we ask the caller to split, rather than silently emitting hundreds
/// of embedding requests.
const MAX_CHUNKS_PER_DOCUMENT: usize = 500;

/// Char-boundary-aware chunker for `add_document`.
///
/// Walks `content` in `EMBEDDING_CHUNK_SIZE`-byte windows, advancing by
/// `(EMBEDDING_CHUNK_SIZE - EMBEDDING_CHUNK_OVERLAP)` bytes per step.
/// Each window is clamped to a UTF-8 char boundary so multi-byte input
/// (CJK, emoji) doesn't panic. Returns `&str` slices into the original
/// content (no allocation per chunk beyond the Vec).
fn chunk_for_embedding(content: &str) -> Vec<&str> {
    if content.is_empty() {
        return Vec::new();
    }
    if content.len() <= EMBEDDING_CHUNK_SIZE {
        return vec![content];
    }

    let stride = EMBEDDING_CHUNK_SIZE
        .checked_sub(EMBEDDING_CHUNK_OVERLAP)
        .and_then(|n| if n == 0 { None } else { Some(n) })
        .unwrap_or(EMBEDDING_CHUNK_SIZE);

    let mut chunks = Vec::new();
    let bytes = content.len();
    let mut start = 0;
    while start < bytes {
        let end = (start + EMBEDDING_CHUNK_SIZE).min(bytes);
        // Clamp `end` down to a char boundary (str::is_char_boundary is
        // O(1)). Same for `start` (already at a boundary on first iter;
        // re-check after stride moves it).
        let safe_end = floor_char_boundary(content, end);
        let safe_start = floor_char_boundary(content, start);
        if safe_end <= safe_start {
            break;
        }
        chunks.push(&content[safe_start..safe_end]);
        if safe_end == bytes {
            break;
        }
        start = safe_start + stride;
    }
    chunks
}

/// Largest char-boundary index `<= idx`. Hand-rolled because
/// `str::floor_char_boundary` is unstable.
fn floor_char_boundary(s: &str, idx: usize) -> usize {
    let idx = idx.min(s.len());
    let mut i = idx;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Build the allowed-origin list for CORS.
///
/// Reads the comma-separated `ALLOWED_ORIGINS` env var. If unset (or empty
/// after trimming), defaults to a localhost-only set so dev startup works
/// without configuration but a production deploy can't accidentally
/// inherit "any origin." The previous `Cors::default().allow_any_origin()`
/// was a credentials-exposing landmine — see #34.
fn build_allowed_origins() -> Vec<String> {
    match std::env::var("ALLOWED_ORIGINS") {
        Ok(raw) if !raw.trim().is_empty() => raw
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        _ => vec![
            "http://localhost:3000".to_string(),
            "http://localhost:5173".to_string(),
            "http://localhost:8080".to_string(),
            "http://127.0.0.1:3000".to_string(),
            "http://127.0.0.1:5173".to_string(),
            "http://127.0.0.1:8080".to_string(),
        ],
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    // Create application state (connects to Qdrant if available)
    let state = AppState::new().await;
    let state_data = Data::new(state.clone());

    // Configure OpenAPI specification
    let spec = Spec {
        info: Info {
            title: "GraphRAG REST API".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: Some(concat!(
                "Production-ready REST API for GraphRAG operations with Qdrant vector database.\n\n",
                "## Features\n",
                "- Semantic search over documents\n",
                "- Knowledge graph construction\n",
                "- Real-time vector embeddings\n",
                "- Qdrant integration (optional)\n",
                "- JWT authentication (optional)\n\n",
                "## Getting Started\n",
                "1. Add documents via `POST /api/documents`\n",
                "2. Build graph via `POST /api/graph/build`\n",
                "3. Query via `POST /api/query`\n"
            ).to_string()),
            ..Default::default()
        },
        ..Default::default()
    };

    // JWT_SECRET is now validated up front in `load_auth_state`, which
    // exits the process if the var is missing or under
    // `JWT_SECRET_MIN_BYTES` bytes (#31). The previous "using insecure
    // default" warning is gone because there is no longer a default.

    tracing::info!("🚀 GraphRAG Server starting...");
    tracing::info!("📡 Listening on http://0.0.0.0:8080");
    tracing::info!("📚 Swagger UI: http://0.0.0.0:8080/swagger");
    tracing::info!("📄 OpenAPI spec: http://0.0.0.0:8080/openapi.json");
    tracing::info!(
        "🗄️  Backend: {}",
        if state.has_qdrant() {
            "Qdrant"
        } else {
            "In-memory"
        }
    );

    HttpServer::new(move || {
        // Configure CORS for each app instance.
        //
        // The previous \`allow_any_origin().allow_any_method().allow_any_header()\`
        // let any web page on any origin issue authenticated requests against
        // the API on behalf of a user with a token in localStorage. Combined
        // with the JWT issuance issues (#31), this widened the blast radius
        // significantly. (#34)
        //
        // Read \`ALLOWED_ORIGINS\` (comma-separated). Default to localhost-only
        // for safe dev startup. Methods and headers are explicit, not "any".
        let allowed_origins = build_allowed_origins();
        let mut cors = Cors::default()
            .allowed_methods(vec!["GET", "POST", "DELETE"])
            .allowed_headers(vec!["Content-Type", "Authorization", "X-API-Key"])
            .max_age(3600);
        for origin in &allowed_origins {
            cors = cors.allowed_origin(origin);
        }

        App::new()
            // OpenAPI documentation
            .document(spec.clone())

            // Global middleware
            .wrap(cors)
            .wrap(actix_web::middleware::Logger::default())

            // Application state
            .app_data(state_data.clone())

            // Request body size limits (10MB for general payload, 10MB for JSON)
            .app_data(web::PayloadConfig::new(validation::MAX_BODY_SIZE))
            .app_data(web::JsonConfig::default().limit(validation::MAX_BODY_SIZE))

            // Public routes
            .service(resource("/").route(get().to(root)))
            .service(resource("/health").route(get().to(health)))

            // API routes
            .service(
                scope("/api")
                    // Documents endpoints
                    .service(
                        scope("/documents")
                            .service(resource("").route(get().to(list_documents)))
                            .service(resource("").route(post().to(add_document)))
                            .service(resource("/{id}").route(delete().to(delete_document)))
                    )
                    // Query endpoints
                    .service(
                        scope("/query")
                            .service(resource("").route(post().to(query)))
                    )
                    // Graph endpoints
                    .service(
                        scope("/graph")
                            .service(resource("/build").route(post().to(build_graph)))
                            .service(resource("/stats").route(get().to(graph_stats)))
                    )
            )

            // Auth routes (temporarily disabled - feature "auth" is disabled)
            // #[cfg(feature = "auth")]
            // .service(
            //     scope("/auth")
            //         .service(resource("/login").route(post().to(login)))
            //         .service(resource("/api-key").route(post().to(create_api_key)))
            // )

            // Config endpoints (item 3.3): registered via plain Actix-web routes.
            // NOTE: To include them in OpenAPI spec, add #[api_operation] macros to
            //       each handler in config_endpoints.rs, then register via Apistos scope/resource.

            // Build OpenAPI spec endpoint
            .build("/openapi.json")

            // Config endpoints (plain Actix-web routing — no #[api_operation] yet)
            .service(
                web::scope("/api/config")
                    .route("", web::get().to(config_endpoints::get_config))
                    .route("", web::post().to(config_endpoints::set_config))
                    .route("/template", web::get().to(config_endpoints::get_config_template))
                    .route("/default", web::get().to(config_endpoints::get_default_config))
                    .route("/validate", web::post().to(config_endpoints::validate_config))
            )
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    // A small document fits in one chunk and yields the original text.
    #[test]
    fn chunk_for_embedding_returns_single_chunk_for_small_input() {
        let chunks = chunk_for_embedding("hello world");
        assert_eq!(chunks, vec!["hello world"]);
    }

    // An exactly-empty document yields zero chunks (regression for #35;
    // an empty doc would otherwise still trigger one embedding call).
    #[test]
    fn chunk_for_embedding_returns_empty_for_empty_input() {
        let chunks = chunk_for_embedding("");
        assert!(chunks.is_empty());
    }

    // A document larger than the chunk size produces overlapping chunks.
    #[test]
    fn chunk_for_embedding_splits_large_input_with_overlap() {
        let big = "x".repeat(EMBEDDING_CHUNK_SIZE * 3);
        let chunks = chunk_for_embedding(&big);
        assert!(
            chunks.len() >= 3,
            "expected ≥3 chunks, got {}",
            chunks.len()
        );
        // Each chunk respects the size cap.
        for c in &chunks {
            assert!(c.len() <= EMBEDDING_CHUNK_SIZE);
        }
    }

    // Multi-byte UTF-8 input must not panic at chunk boundaries
    // (regression for #35; naive byte slicing would split a 3-byte CJK
    // codepoint mid-character).
    #[test]
    fn chunk_for_embedding_does_not_panic_on_utf8_boundaries() {
        let cjk_chunk = "中文测试".repeat(EMBEDDING_CHUNK_SIZE);
        let chunks = chunk_for_embedding(&cjk_chunk);
        for c in &chunks {
            // If we accidentally split mid-codepoint, this would panic
            // earlier inside the slice. Reaching here means each chunk
            // is valid UTF-8.
            assert!(c.is_char_boundary(0) && c.is_char_boundary(c.len()));
        }
    }

    // Default behavior when ALLOWED_ORIGINS is unset: localhost only, no
    // wildcard. Regression for #34's allow_any_origin landmine.
    #[test]
    fn build_allowed_origins_defaults_to_localhost_only() {
        // Save & restore env to avoid cross-test pollution.
        let prev = std::env::var("ALLOWED_ORIGINS").ok();
        std::env::remove_var("ALLOWED_ORIGINS");

        let origins = build_allowed_origins();
        assert!(
            origins.iter().any(|o| o.contains("localhost")),
            "expected localhost in default origins, got {origins:?}"
        );
        assert!(
            !origins.iter().any(|o| o == "*"),
            "default must not include wildcard, got {origins:?}"
        );

        if let Some(p) = prev {
            std::env::set_var("ALLOWED_ORIGINS", p);
        } else {
            std::env::remove_var("ALLOWED_ORIGINS");
        }
    }

    // ALLOWED_ORIGINS env var is parsed comma-separated, with trimming.
    #[test]
    fn build_allowed_origins_parses_env_var() {
        let prev = std::env::var("ALLOWED_ORIGINS").ok();
        std::env::set_var(
            "ALLOWED_ORIGINS",
            "https://app.example.com, https://admin.example.com",
        );

        let origins = build_allowed_origins();
        assert_eq!(origins.len(), 2);
        assert!(origins.contains(&"https://app.example.com".to_string()));
        assert!(origins.contains(&"https://admin.example.com".to_string()));

        if let Some(p) = prev {
            std::env::set_var("ALLOWED_ORIGINS", p);
        } else {
            std::env::remove_var("ALLOWED_ORIGINS");
        }
    }
}
