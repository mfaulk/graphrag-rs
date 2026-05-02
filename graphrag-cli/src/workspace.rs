//! CLI workspace registry.
//!
//! Manages the on-disk workspace directory tree under a single canonical
//! base path (see [`cli_workspaces_dir`]). The type is named
//! [`CliWorkspaceManager`] to avoid colliding with
//! [`graphrag_core::persistence::WorkspaceManager`] — the two types lived
//! under the same name and confused the eye when both were imported into
//! `graphrag.rs` (#64 item 1).

use color_eyre::eyre::{eyre, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Workspace metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceMetadata {
    /// Workspace ID
    pub id: String,
    /// Workspace name
    pub name: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last accessed timestamp
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    /// Configuration file path (if any)
    pub config_path: Option<PathBuf>,
}

impl WorkspaceMetadata {
    /// Create a new workspace
    pub fn new(name: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            created_at: now,
            last_accessed: now,
            config_path: None,
        }
    }

    /// Update last accessed time
    pub fn touch(&mut self) {
        self.last_accessed = chrono::Utc::now();
    }
}

/// Resolve the canonical base directory for CLI workspaces.
///
/// Priority:
/// 1. `GRAPHRAG_WORKSPACES_DIR` env var (used by tests + custom deployments)
/// 2. `dirs::data_dir().join("graphrag/workspaces")` — XDG-friendly default
///    matching what the TUI's `/workspace save` already writes to. (#52
///    consolidates the previously-divergent `~/.graphrag/workspaces` path
///    into this one.)
///
/// Returns an error if neither path is resolvable. Callers that want a
/// different base should construct via [`CliWorkspaceManager::with_base_dir`].
pub fn cli_workspaces_dir() -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("GRAPHRAG_WORKSPACES_DIR") {
        if !custom.trim().is_empty() {
            return Ok(PathBuf::from(custom));
        }
    }
    let base = dirs::data_dir().ok_or_else(|| {
        eyre!(
            "Cannot determine data directory; set $XDG_DATA_HOME or \
             GRAPHRAG_WORKSPACES_DIR"
        )
    })?;
    Ok(base.join("graphrag").join("workspaces"))
}

/// CLI workspace registry.
pub struct CliWorkspaceManager {
    /// Base directory for workspaces
    base_dir: PathBuf,
}

impl CliWorkspaceManager {
    /// Create a new workspace manager rooted at the canonical
    /// [`cli_workspaces_dir`].
    pub fn new() -> Result<Self> {
        Ok(Self {
            base_dir: cli_workspaces_dir()?,
        })
    }

    /// Create a workspace manager rooted at an explicit base directory
    /// (primarily for tests).
    pub fn with_base_dir(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    /// Get the base directory.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Get workspace directory.
    ///
    /// Validates `id` matches a UUID v4 shape so a malicious or careless
    /// `id` like `../../../etc/passwd` cannot escape `base_dir`. Closes
    /// the path-traversal half of #54.
    pub fn workspace_dir(&self, id: &str) -> Result<PathBuf> {
        validate_workspace_id(id)?;
        Ok(self.base_dir.join(id))
    }

    /// Get workspace metadata file path.
    pub fn metadata_path(&self, id: &str) -> Result<PathBuf> {
        Ok(self.workspace_dir(id)?.join("metadata.json"))
    }

    /// Get query history file path.
    pub fn query_history_path(&self, id: &str) -> Result<PathBuf> {
        Ok(self.workspace_dir(id)?.join("query_history.json"))
    }

    /// Create a new workspace.
    pub async fn create_workspace(&self, name: String) -> Result<WorkspaceMetadata> {
        let metadata = WorkspaceMetadata::new(name);
        let workspace_dir = self.workspace_dir(&metadata.id)?;
        tokio::fs::create_dir_all(&workspace_dir).await?;
        self.save_metadata(&metadata).await?;
        Ok(metadata)
    }

    /// Load workspace metadata.
    ///
    /// Pure read — does NOT update `last_accessed` or write back. The
    /// previous behavior caused every `list_workspaces` call to rewrite
    /// every metadata file (each write a torn-write risk under crash or
    /// concurrent CLI). Callers that want to record access call
    /// [`Self::touch_workspace`] explicitly. Closes the read-amplification
    /// half of #53.
    pub async fn load_metadata(&self, id: &str) -> Result<WorkspaceMetadata> {
        let path = self.metadata_path(id)?;
        let content = tokio::fs::read_to_string(&path).await?;
        let metadata: WorkspaceMetadata = serde_json::from_str(&content)?;
        Ok(metadata)
    }

    /// Update `last_accessed` and persist. Call only when the workspace is
    /// actually opened, not on listing.
    pub async fn touch_workspace(&self, id: &str) -> Result<WorkspaceMetadata> {
        let mut metadata = self.load_metadata(id).await?;
        metadata.touch();
        self.save_metadata(&metadata).await?;
        Ok(metadata)
    }

    /// Save workspace metadata atomically (`.tmp` + rename).
    ///
    /// Closes the non-atomic-write half of #53. A Ctrl-C / OOM mid-write
    /// no longer truncates the registry; the rename is atomic on POSIX
    /// and NTFS once the data is fully written.
    pub async fn save_metadata(&self, metadata: &WorkspaceMetadata) -> Result<()> {
        let path = self.metadata_path(&metadata.id)?;
        let tmp = path.with_extension("json.tmp");
        let json = serde_json::to_string_pretty(metadata)?;
        tokio::fs::write(&tmp, json).await?;
        tokio::fs::rename(&tmp, &path).await?;
        Ok(())
    }

    /// List all workspaces (does NOT mutate `last_accessed` — see #53).
    pub async fn list_workspaces(&self) -> Result<Vec<WorkspaceMetadata>> {
        let mut workspaces = Vec::new();
        if !self.base_dir.exists() {
            return Ok(workspaces);
        }
        let mut entries = tokio::fs::read_dir(&self.base_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_dir() {
                if let Some(id) = entry.file_name().to_str() {
                    if validate_workspace_id(id).is_err() {
                        // Skip dirs that don't look like workspaces — they
                        // can't have been created by us.
                        continue;
                    }
                    if let Ok(metadata) = self.load_metadata(id).await {
                        workspaces.push(metadata);
                    }
                }
            }
        }
        // Sort by last accessed (most recent first)
        workspaces.sort_by(|a, b| b.last_accessed.cmp(&a.last_accessed));
        Ok(workspaces)
    }

    /// Delete a workspace.
    ///
    /// `id` is validated as a UUID v4 by [`Self::workspace_dir`], so
    /// `../../../etc/passwd` and similar cannot reach this code path.
    /// `tokio::fs::remove_dir_all` already handles "not found" by
    /// returning `NotFound`; the previous explicit `if .exists()` check
    /// was a TOCTOU window. (#54.)
    pub async fn delete_workspace(&self, id: &str) -> Result<()> {
        let workspace_dir = self.workspace_dir(id)?;
        match tokio::fs::remove_dir_all(&workspace_dir).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}

/// Validate that `id` matches the UUID v4 shape used by
/// [`WorkspaceMetadata::new`]. Closes the path-traversal half of #54: any
/// id that would join to a path escaping `base_dir` is rejected before
/// the join happens.
fn validate_workspace_id(id: &str) -> Result<()> {
    // Length check first (cheap reject for obvious traversal attempts).
    if id.len() != 36 {
        return Err(eyre!(
            "Invalid workspace id: expected UUID v4, got {:?}",
            id
        ));
    }
    // Strict UUID parse via the same crate that produced it.
    Uuid::parse_str(id).map_err(|e| eyre!("Invalid workspace id {:?}: {}", id, e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn manager_in_temp() -> (CliWorkspaceManager, TempDir) {
        let tmp = TempDir::new().unwrap();
        let mgr = CliWorkspaceManager::with_base_dir(tmp.path());
        (mgr, tmp)
    }

    // create + load round-trips metadata.
    #[tokio::test]
    async fn create_then_load_round_trips() {
        let (mgr, _tmp) = manager_in_temp();
        let ws = mgr.create_workspace("test".to_string()).await.unwrap();
        let loaded = mgr.load_metadata(&ws.id).await.unwrap();
        assert_eq!(loaded.id, ws.id);
        assert_eq!(loaded.name, "test");
    }

    // load_metadata must NOT write back (regression for #53 read amplification).
    #[tokio::test]
    async fn load_metadata_does_not_rewrite_file() {
        let (mgr, _tmp) = manager_in_temp();
        let ws = mgr.create_workspace("test".to_string()).await.unwrap();
        let path = mgr.metadata_path(&ws.id).unwrap();
        let mtime_before = std::fs::metadata(&path).unwrap().modified().unwrap();

        // Sleep so any rewrite would bump mtime.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let _ = mgr.load_metadata(&ws.id).await.unwrap();

        let mtime_after = std::fs::metadata(&path).unwrap().modified().unwrap();
        assert_eq!(
            mtime_before, mtime_after,
            "load_metadata must not rewrite the file"
        );
    }

    // touch_workspace explicitly bumps last_accessed and persists.
    #[tokio::test]
    async fn touch_workspace_updates_last_accessed() {
        let (mgr, _tmp) = manager_in_temp();
        let ws = mgr.create_workspace("test".to_string()).await.unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let touched = mgr.touch_workspace(&ws.id).await.unwrap();
        assert!(touched.last_accessed > ws.last_accessed);
    }

    // delete is idempotent on a non-existent workspace.
    #[tokio::test]
    async fn delete_nonexistent_workspace_is_idempotent() {
        let (mgr, _tmp) = manager_in_temp();
        let fresh_id = Uuid::new_v4().to_string();
        // No create — just delete.
        assert!(mgr.delete_workspace(&fresh_id).await.is_ok());
    }

    // Path traversal attempts are rejected at the id-validation stage
    // (regression for #54).
    #[tokio::test]
    async fn delete_rejects_path_traversal_in_id() {
        let (mgr, _tmp) = manager_in_temp();
        let bad = mgr.delete_workspace("../../../etc/passwd").await;
        assert!(bad.is_err(), "path traversal id must be rejected");
    }

    #[tokio::test]
    async fn workspace_dir_rejects_non_uuid_id() {
        let (mgr, _tmp) = manager_in_temp();
        assert!(mgr.workspace_dir("not-a-uuid").is_err());
        assert!(mgr.workspace_dir("../escape").is_err());
        assert!(mgr.workspace_dir("").is_err());
    }

    // cli_workspaces_dir honors the env var override (used by tests + ops).
    #[test]
    fn cli_workspaces_dir_honors_env_var() {
        let prev = std::env::var("GRAPHRAG_WORKSPACES_DIR").ok();
        std::env::set_var("GRAPHRAG_WORKSPACES_DIR", "/tmp/graphrag-test-ws");
        let dir = cli_workspaces_dir().unwrap();
        assert_eq!(dir, PathBuf::from("/tmp/graphrag-test-ws"));
        if let Some(p) = prev {
            std::env::set_var("GRAPHRAG_WORKSPACES_DIR", p);
        } else {
            std::env::remove_var("GRAPHRAG_WORKSPACES_DIR");
        }
    }
}
