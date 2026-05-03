//! Workspace management for GraphRAG persistence
//!
//! Provides multi-workspace support with checkpointing and metadata tracking.

use crate::core::{GraphRAGError, KnowledgeGraph, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Workspace metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkspaceMetadata {
    /// Workspace name
    pub name: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modified timestamp
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// Number of entities
    pub entity_count: usize,
    /// Number of relationships
    pub relationship_count: usize,
    /// Number of documents
    pub document_count: usize,
    /// Number of chunks
    pub chunk_count: usize,
    /// Storage format version
    pub format_version: String,
    /// Description (optional)
    pub description: Option<String>,
}

/// Current on-disk workspace metadata format version.
///
/// Bumped when the metadata schema changes in a backward-incompatible way.
/// Loading a workspace whose `format_version` differs from this constant
/// emits a warning (no migration story exists yet — see #23).
pub const CURRENT_FORMAT_VERSION: &str = "1.0";

impl WorkspaceMetadata {
    /// Create new workspace metadata
    pub fn new(name: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            name,
            created_at: now,
            modified_at: now,
            entity_count: 0,
            relationship_count: 0,
            document_count: 0,
            chunk_count: 0,
            format_version: CURRENT_FORMAT_VERSION.to_string(),
            description: None,
        }
    }

    /// Update counts from knowledge graph
    pub fn update_from_graph(&mut self, graph: &KnowledgeGraph) {
        self.entity_count = graph.entity_count();
        self.relationship_count = graph.relationship_count();
        self.document_count = graph.document_count();
        self.chunk_count = graph.chunks().count();
        self.modified_at = chrono::Utc::now();
    }
}

/// Workspace information (lightweight version for listing)
#[derive(Debug, Clone)]
pub struct WorkspaceInfo {
    /// Workspace name
    pub name: String,
    /// Path to workspace directory
    pub path: PathBuf,
    /// Workspace metadata
    pub metadata: WorkspaceMetadata,
    /// Total size in bytes
    pub size_bytes: u64,
}

/// Workspace manager for multi-workspace support
#[derive(Debug, Clone)]
pub struct WorkspaceManager {
    /// Base directory for all workspaces
    base_dir: PathBuf,
}

impl WorkspaceManager {
    /// Create a new workspace manager
    ///
    /// # Arguments
    /// * `base_dir` - Base directory path (e.g., "./workspace")
    ///
    /// # Example
    /// ```no_run
    /// use graphrag_core::persistence::WorkspaceManager;
    ///
    /// let workspace = WorkspaceManager::new("./workspace").unwrap();
    /// ```
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();

        // Create base directory if it doesn't exist
        if !base_dir.exists() {
            fs::create_dir_all(&base_dir)?;
            #[cfg(feature = "tracing")]
            tracing::info!("Created workspace base directory: {:?}", base_dir);
        }

        Ok(Self { base_dir })
    }

    /// Get workspace directory path
    pub fn workspace_path(&self, workspace_name: &str) -> PathBuf {
        self.base_dir.join(workspace_name)
    }

    /// Check if workspace exists
    pub fn workspace_exists(&self, workspace_name: &str) -> bool {
        self.workspace_path(workspace_name).exists()
    }

    /// Create a new workspace
    pub fn create_workspace(&self, workspace_name: &str) -> Result<()> {
        let workspace_path = self.workspace_path(workspace_name);

        if workspace_path.exists() {
            return Err(GraphRAGError::Config {
                message: format!("Workspace '{}' already exists", workspace_name),
            });
        }

        // Create workspace directory
        fs::create_dir_all(&workspace_path)?;

        // Create metadata
        let metadata = WorkspaceMetadata::new(workspace_name.to_string());
        self.save_metadata(&metadata, workspace_name)?;

        #[cfg(feature = "tracing")]
        tracing::info!("Created workspace: {}", workspace_name);

        Ok(())
    }

    /// Delete a workspace
    pub fn delete_workspace(&self, workspace_name: &str) -> Result<()> {
        let workspace_path = self.workspace_path(workspace_name);

        if !workspace_path.exists() {
            return Err(GraphRAGError::Config {
                message: format!("Workspace '{}' does not exist", workspace_name),
            });
        }

        fs::remove_dir_all(&workspace_path)?;

        #[cfg(feature = "tracing")]
        tracing::info!("Deleted workspace: {}", workspace_name);

        Ok(())
    }

    /// List all workspaces
    pub fn list_workspaces(&self) -> Result<Vec<WorkspaceInfo>> {
        let mut workspaces = Vec::new();

        if !self.base_dir.exists() {
            return Ok(workspaces);
        }

        for entry in fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let workspace_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                // Load metadata. If it's missing or unreadable, skip the
                // workspace from the listing rather than synthesize fresh
                // defaults — the previous behaviour silently surfaced
                // corrupt or partially-initialized directories as healthy
                // workspaces with `modified_at = now`, which then sorted to
                // the top of the list. Surface the failure in the log so
                // operators can investigate (#23).
                let metadata = match self.load_metadata(&workspace_name) {
                    Ok(m) => m,
                    Err(_e) => {
                        #[cfg(feature = "tracing")]
                        tracing::warn!(
                            workspace = %workspace_name,
                            error = %_e,
                            "Skipping workspace from listing: metadata unreadable. \
                             Inspect metadata.toml or remove the directory."
                        );
                        continue;
                    },
                };

                // Calculate size
                let size_bytes = Self::calculate_dir_size(&path).unwrap_or(0);

                workspaces.push(WorkspaceInfo {
                    name: workspace_name,
                    path,
                    metadata,
                    size_bytes,
                });
            }
        }

        // Sort by modification time (newest first)
        workspaces.sort_by(|a, b| b.metadata.modified_at.cmp(&a.metadata.modified_at));

        Ok(workspaces)
    }

    /// Save knowledge graph to workspace
    pub fn save_graph(&self, graph: &KnowledgeGraph, workspace_name: &str) -> Result<()> {
        // Create workspace if it doesn't exist
        if !self.workspace_exists(workspace_name) {
            self.create_workspace(workspace_name)?;
        }

        let workspace_path = self.workspace_path(workspace_name);

        // Save to JSON (always available as fallback)
        let json_path = workspace_path.join("graph.json");
        graph.save_to_json(json_path.to_str().unwrap())?;

        // Save to Parquet (if feature enabled)
        #[cfg(feature = "persistent-storage")]
        {
            use super::parquet::ParquetPersistence;
            let parquet = ParquetPersistence::new(workspace_path.clone())?;
            parquet.save_graph(graph)?;
        }

        // Update metadata
        let mut metadata = self
            .load_metadata(workspace_name)
            .unwrap_or_else(|_| WorkspaceMetadata::new(workspace_name.to_string()));
        metadata.update_from_graph(graph);
        self.save_metadata(&metadata, workspace_name)?;

        #[cfg(feature = "tracing")]
        tracing::info!("Saved graph to workspace: {}", workspace_name);

        Ok(())
    }

    /// Load knowledge graph from workspace.
    ///
    /// Refuses to load if the workspace's `metadata.toml` declares a
    /// `format_version` that doesn't match [`CURRENT_FORMAT_VERSION`] —
    /// continuing would silently couple a stale on-disk schema to in-memory
    /// types. Recreate the workspace or write a migration before retrying.
    /// (Workspaces without metadata.toml fall through to the existing
    /// best-effort load path.) See issue #23.
    pub fn load_graph(&self, workspace_name: &str) -> Result<KnowledgeGraph> {
        if !self.workspace_exists(workspace_name) {
            return Err(GraphRAGError::Config {
                message: format!("Workspace '{}' does not exist", workspace_name),
            });
        }

        // Refuse on format_version mismatch. Missing metadata.toml is
        // treated as legacy and tolerated below.
        if let Ok(meta) = self.load_metadata(workspace_name) {
            if meta.format_version != CURRENT_FORMAT_VERSION {
                return Err(GraphRAGError::Config {
                    message: format!(
                        "Workspace '{}' has metadata format_version '{}' but this build \
                         expects '{}'. Refusing to load: no migration is implemented. \
                         Recreate the workspace or downgrade graphrag-core.",
                        workspace_name, meta.format_version, CURRENT_FORMAT_VERSION
                    ),
                });
            }
        }

        let workspace_path = self.workspace_path(workspace_name);

        // Try loading from Parquet first (if feature enabled)
        #[cfg(feature = "persistent-storage")]
        {
            use super::parquet::ParquetPersistence;
            let parquet = ParquetPersistence::new(workspace_path.clone())?;
            match parquet.load_graph() {
                Ok(graph) => {
                    #[cfg(feature = "tracing")]
                    tracing::info!("Loaded graph from Parquet: {}", workspace_name);
                    return Ok(graph);
                },
                Err(_e) => {
                    // Parquet load failed — fall through to JSON. Surface
                    // the failure instead of silently degrading: the parquet
                    // file might be corrupt, version-mismatched, or missing.
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        workspace = %workspace_name,
                        error = %_e,
                        "Parquet load failed; falling back to JSON. \
                         Investigate the parquet artifact before relying on this load."
                    );
                },
            }
        }

        // Fallback to JSON
        let json_path = workspace_path.join("graph.json");
        if json_path.exists() {
            #[cfg(feature = "tracing")]
            tracing::info!("Loading graph from JSON fallback: {}", workspace_name);
            return KnowledgeGraph::load_from_json(json_path.to_str().unwrap());
        }

        Err(GraphRAGError::Config {
            message: format!("No graph data found in workspace '{}'", workspace_name),
        })
    }

    /// Save workspace metadata
    fn save_metadata(&self, metadata: &WorkspaceMetadata, workspace_name: &str) -> Result<()> {
        let workspace_path = self.workspace_path(workspace_name);
        let metadata_path = workspace_path.join("metadata.toml");

        let toml_string = toml::to_string_pretty(metadata).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to serialize metadata: {}", e),
        })?;

        fs::write(metadata_path, toml_string)?;

        Ok(())
    }

    /// Load workspace metadata.
    ///
    /// Warns (without failing) if the on-disk `format_version` doesn't match
    /// [`CURRENT_FORMAT_VERSION`]. There is no migration story yet — see #23.
    fn load_metadata(&self, workspace_name: &str) -> Result<WorkspaceMetadata> {
        let workspace_path = self.workspace_path(workspace_name);
        let metadata_path = workspace_path.join("metadata.toml");

        if !metadata_path.exists() {
            return Err(GraphRAGError::Config {
                message: format!("Metadata not found for workspace '{}'", workspace_name),
            });
        }

        let toml_string = fs::read_to_string(metadata_path)?;
        let metadata: WorkspaceMetadata =
            toml::from_str(&toml_string).map_err(|e| GraphRAGError::Config {
                message: format!("Failed to parse metadata: {}", e),
            })?;

        if metadata.format_version != CURRENT_FORMAT_VERSION {
            #[cfg(feature = "tracing")]
            tracing::warn!(
                workspace = %workspace_name,
                on_disk = %metadata.format_version,
                expected = %CURRENT_FORMAT_VERSION,
                "Workspace metadata format_version mismatch — proceeding without migration; \
                 some fields may be missing or interpreted incorrectly (see issue #23)"
            );
        }

        Ok(metadata)
    }

    /// Calculate directory size recursively
    fn calculate_dir_size(path: &Path) -> Result<u64> {
        let mut total_size = 0u64;

        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    total_size += Self::calculate_dir_size(&path)?;
                } else {
                    total_size += entry.metadata()?.len();
                }
            }
        } else {
            total_size = fs::metadata(path)?.len();
        }

        Ok(total_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_workspace_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let workspace = WorkspaceManager::new(temp_dir.path()).unwrap();
        assert!(workspace.base_dir.exists());
    }

    #[test]
    fn test_create_and_list_workspaces() {
        let temp_dir = TempDir::new().unwrap();
        let workspace = WorkspaceManager::new(temp_dir.path()).unwrap();

        workspace.create_workspace("test1").unwrap();
        workspace.create_workspace("test2").unwrap();

        let workspaces = workspace.list_workspaces().unwrap();
        assert_eq!(workspaces.len(), 2);
    }

    #[test]
    fn test_delete_workspace() {
        let temp_dir = TempDir::new().unwrap();
        let workspace = WorkspaceManager::new(temp_dir.path()).unwrap();

        workspace.create_workspace("test").unwrap();
        assert!(workspace.workspace_exists("test"));

        workspace.delete_workspace("test").unwrap();
        assert!(!workspace.workspace_exists("test"));
    }

    #[test]
    fn test_save_and_load_graph() {
        let temp_dir = TempDir::new().unwrap();
        let workspace = WorkspaceManager::new(temp_dir.path()).unwrap();

        let graph = KnowledgeGraph::new();
        workspace.save_graph(&graph, "test").unwrap();

        let loaded_graph = workspace.load_graph("test").unwrap();
        assert_eq!(loaded_graph.entity_count(), 0);
    }

    // Loading a workspace whose metadata declares a different format_version
    // must succeed (we don't gate on it), but the version mismatch should be
    // observable in the returned struct so callers / log inspection can see
    // it (regression for #23).
    #[test]
    fn load_metadata_returns_on_disk_version_even_when_mismatched() {
        let temp_dir = TempDir::new().unwrap();
        let workspace = WorkspaceManager::new(temp_dir.path()).unwrap();
        workspace.create_workspace("legacy").unwrap();

        // Hand-write a metadata.toml with a deliberately stale format_version.
        let stale_meta = WorkspaceMetadata {
            name: "legacy".to_string(),
            created_at: chrono::Utc::now(),
            modified_at: chrono::Utc::now(),
            entity_count: 0,
            relationship_count: 0,
            document_count: 0,
            chunk_count: 0,
            format_version: "0.0-prehistoric".to_string(),
            description: None,
        };
        let toml_string = toml::to_string_pretty(&stale_meta).unwrap();
        let meta_path = temp_dir.path().join("legacy").join("metadata.toml");
        fs::write(meta_path, toml_string).unwrap();

        let loaded = workspace.load_metadata("legacy").unwrap();
        assert_eq!(loaded.format_version, "0.0-prehistoric");
        assert_ne!(loaded.format_version, CURRENT_FORMAT_VERSION);
    }

    // Newly-written metadata always carries the current format version.
    #[test]
    fn new_metadata_carries_current_format_version() {
        let m = WorkspaceMetadata::new("fresh".to_string());
        assert_eq!(m.format_version, CURRENT_FORMAT_VERSION);
    }

    // Regression for #23: load_graph must refuse to load a workspace whose
    // metadata.toml declares an unrecognized `format_version`. Continuing
    // would silently couple stale on-disk schemas to in-memory types and is
    // a correctness footgun; surface it as an error so callers either
    // migrate or recreate the workspace.
    #[test]
    fn load_graph_refuses_on_format_version_mismatch() {
        let temp_dir = TempDir::new().unwrap();
        let workspace = WorkspaceManager::new(temp_dir.path()).unwrap();

        let graph = KnowledgeGraph::new();
        workspace.save_graph(&graph, "future").unwrap();

        // Replace metadata with a version we don't understand.
        let meta_path = temp_dir.path().join("future").join("metadata.toml");
        let mut meta: WorkspaceMetadata =
            toml::from_str(&fs::read_to_string(&meta_path).unwrap()).unwrap();
        meta.format_version = "9.9-from-the-future".to_string();
        fs::write(&meta_path, toml::to_string_pretty(&meta).unwrap()).unwrap();

        let result = workspace.load_graph("future");
        assert!(
            result.is_err(),
            "load_graph must refuse to load a workspace with an incompatible format_version"
        );
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("format_version") || msg.contains("9.9-from-the-future"),
            "error message should mention the version mismatch, got: {}",
            msg
        );
    }

    // Regression for #23: a workspace dir whose metadata.toml is unreadable
    // (corrupt TOML) must not be silently surfaced as a fresh, healthy
    // workspace by list_workspaces. It should be skipped from the listing.
    #[test]
    fn list_workspaces_skips_workspace_with_corrupt_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let workspace = WorkspaceManager::new(temp_dir.path()).unwrap();

        workspace.create_workspace("good").unwrap();
        workspace.create_workspace("broken").unwrap();

        // Stomp on the broken workspace's metadata with garbage TOML.
        let bad_meta_path = temp_dir.path().join("broken").join("metadata.toml");
        fs::write(&bad_meta_path, "this is :: not :: valid :: toml @@@").unwrap();

        let listed = workspace.list_workspaces().unwrap();
        let names: Vec<&str> = listed.iter().map(|w| w.name.as_str()).collect();
        assert!(names.contains(&"good"), "good workspace should be listed");
        assert!(
            !names.contains(&"broken"),
            "broken workspace must not be silently listed with synthesized defaults"
        );
    }
}
