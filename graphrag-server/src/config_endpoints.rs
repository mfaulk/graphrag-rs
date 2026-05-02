//! Configuration endpoints for GraphRAG Server
//!
//! These endpoints allow dynamic configuration of the GraphRAG pipeline via JSON REST API

use super::{config_handler, AppState};
use crate::config_handler::ConfigError;
use crate::models::ApiError;
use actix_web::web::{Data, Json};
use serde_json::json;

/// Map [`ConfigError`] to the right [`ApiError`] variant for the HTTP boundary.
///
/// Parse and validation errors are both client errors (400) but with distinct
/// shapes. NotSet is a 404 for read endpoints.
fn config_error_to_api(err: ConfigError) -> ApiError {
    match err {
        ConfigError::Parse(e) => ApiError::BadRequest(format!("Invalid JSON: {}", e)),
        ConfigError::Validation(errors) => {
            ApiError::BadRequest(format!("Configuration validation failed: {:?}", errors))
        },
        ConfigError::NotSet => ApiError::NotFound(
            "No configuration set. Use POST /api/config to initialize.".to_string(),
        ),
    }
}

/// GET /api/config - Get current configuration
pub async fn get_config(state: Data<AppState>) -> Result<Json<serde_json::Value>, ApiError> {
    if !state.config_manager.is_configured().await {
        return Err(ApiError::NotFound(
            "No configuration set. Use POST /api/config to initialize.".to_string(),
        ));
    }

    let config_json = state
        .config_manager
        .to_json()
        .await
        .map_err(config_error_to_api)?;

    let config: serde_json::Value =
        serde_json::from_str(&config_json).map_err(|e| ApiError::InternalError(e.to_string()))?;

    Ok(Json(json!({
        "success": true,
        "config": config,
        "graphrag_initialized": state.graphrag.read().await.is_some()
    })))
}

/// POST /api/config - Set configuration and initialize GraphRAG
pub async fn set_config(
    state: Data<AppState>,
    payload: Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    tracing::info!("Received configuration request");

    // Parse the configuration from JSON
    let config_json = serde_json::to_string(&payload)
        .map_err(|e| ApiError::BadRequest(format!("Invalid JSON: {}", e)))?;

    // Set configuration via ConfigManager
    state
        .config_manager
        .set_from_json(&config_json)
        .await
        .map_err(config_error_to_api)?;

    // Get the validated config
    let config = state
        .config_manager
        .get_config()
        .await
        .ok_or(ApiError::InternalError("Failed to get config".to_string()))?;

    // Initialize GraphRAG with the config.
    //
    // `GraphRAG::new` and `initialize` are CPU/IO-bound sync calls that can
    // take several seconds (loading embedding models, scanning workspace).
    // Run them on the blocking pool so we don't park an async worker for
    // the duration — every other handler that does
    // `state.graphrag.read().await` would queue behind a parked worker.
    tracing::info!("Initializing GraphRAG with custom configuration...");
    let graphrag = tokio::task::spawn_blocking(
        move || -> Result<graphrag_core::GraphRAG, graphrag_core::GraphRAGError> {
            let mut g = graphrag_core::GraphRAG::new(config)?;
            g.initialize()?;
            Ok(g)
        },
    )
    .await
    .map_err(|join_err| {
        tracing::error!("GraphRAG init task panicked: {}", join_err);
        ApiError::InternalError("GraphRAG init task panicked".to_string())
    })?
    .map_err(|e| {
        tracing::error!("GraphRAG initialization failed: {}", e);
        ApiError::InternalError("GraphRAG initialization failed".to_string())
    })?;

    // Store the initialized GraphRAG (write lock is taken only for the swap)
    *state.graphrag.write().await = Some(graphrag);

    tracing::info!("✅ GraphRAG initialized successfully with custom configuration");

    Ok(Json(json!({
        "success": true,
        "message": "GraphRAG initialized with custom configuration",
        "configured": true,
        "mode": "full_pipeline"
    })))
}

/// GET /api/config/template - Get configuration template
pub async fn get_config_template() -> Json<config_handler::ConfigTemplateResponse> {
    Json(config_handler::get_config_templates())
}

/// GET /api/config/default - Get default configuration
pub async fn get_default_config() -> Json<serde_json::Value> {
    let default_json = config_handler::ConfigManager::default_config_json();
    let config: serde_json::Value = serde_json::from_str(&default_json).unwrap_or(json!({}));

    Json(json!({
        "config": config,
        "description": "Default GraphRAG configuration with sensible defaults"
    }))
}

/// POST /api/config/validate - Validate configuration without applying
pub async fn validate_config(
    _state: Data<AppState>,
    payload: Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let config_json = serde_json::to_string(&payload)
        .map_err(|e| ApiError::BadRequest(format!("Invalid JSON: {}", e)))?;

    // Try to parse as Config
    match serde_json::from_str::<graphrag_core::Config>(&config_json) {
        Ok(_) => Ok(Json(json!({
            "valid": true,
            "message": "Configuration is valid"
        }))),
        Err(e) => Ok(Json(json!({
            "valid": false,
            "errors": [format!("Parse error: {}", e)]
        }))),
    }
}
