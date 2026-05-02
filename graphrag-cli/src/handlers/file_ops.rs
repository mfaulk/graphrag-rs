//! File operations utilities
//!
//! Provides helpers for loading and validating files.

use color_eyre::eyre::{eyre, Result};
use std::path::{Path, PathBuf};
use tokio::fs;

/// File operations utility
pub struct FileOperations;

impl FileOperations {
    /// Check if a file exists
    pub async fn exists(path: &Path) -> bool {
        fs::metadata(path).await.is_ok()
    }

    /// Validate that a file exists and is readable
    pub async fn validate_file(path: &Path) -> Result<()> {
        // Debug log the exact path being checked
        tracing::debug!("Validating file path: {:?}", path);
        tracing::debug!("Path as string: {}", path.display());
        tracing::debug!("Path extension: {:?}", path.extension());

        if !Self::exists(path).await {
            return Err(eyre!("File not found: {}", path.display()));
        }

        if !path.is_file() {
            return Err(eyre!("Path is not a file: {}", path.display()));
        }

        // Try to read metadata to check permissions
        fs::metadata(path)
            .await
            .map_err(|e| eyre!("Cannot read file: {}", e))?;

        Ok(())
    }

    /// Read a file as string
    pub async fn read_to_string(path: &Path) -> Result<String> {
        Self::validate_file(path).await?;

        fs::read_to_string(path)
            .await
            .map_err(|e| eyre!("Failed to read file {}: {}", path.display(), e))
    }

    /// Write string to file
    #[allow(dead_code)]
    pub async fn write_string(path: &Path, content: &str) -> Result<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| eyre!("Failed to create directory {}: {}", parent.display(), e))?;
        }

        fs::write(path, content)
            .await
            .map_err(|e| eyre!("Failed to write file {}: {}", path.display(), e))
    }

    /// Expand tilde (~) in path.
    ///
    /// Only `~/...` (with a path separator) is expanded. `~user/...`
    /// (POSIX user-name expansion) is intentionally NOT supported: the old
    /// implementation passed any `~`-prefixed string through unchanged on a
    /// home-dir-less environment, which let `/load ~user/foo` resolve to
    /// the literal directory `~user/foo` (a real, attacker-controllable
    /// directory in the cwd).
    ///
    /// Returns an error when the input starts with `~/` and the home
    /// directory can't be determined — closes the silent-fallback half of
    /// #54. Non-`~` paths are returned unchanged.
    pub fn expand_tilde(path: &Path) -> Result<PathBuf> {
        let s = match path.to_str() {
            Some(s) => s,
            None => return Ok(path.to_path_buf()),
        };
        // Only the `~/` form (or bare `~`) expands. `~user` stays literal —
        // it's never been supported and silently passing it through opened
        // a path-traversal surface.
        if s == "~" || s.starts_with("~/") {
            let home = dirs::home_dir().ok_or_else(|| {
                eyre!(
                    "Cannot expand `{}`: $HOME is not set and dirs::home_dir() returned None",
                    path.display()
                )
            })?;
            if s == "~" {
                return Ok(home);
            }
            // strip "~/"
            return Ok(home.join(&s[2..]));
        }
        Ok(path.to_path_buf())
    }

    /// Resolve relative path to absolute
    #[allow(dead_code)]
    pub fn canonicalize(path: &Path) -> Result<PathBuf> {
        let expanded = Self::expand_tilde(path)?;

        if expanded.is_absolute() {
            Ok(expanded)
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(expanded))
                .map_err(|e| eyre!("Failed to get current directory: {}", e))
        }
    }

    /// Get file extension
    #[allow(dead_code)]
    pub fn get_extension(path: &Path) -> Option<String> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase())
    }

    /// Check if file is a supported document format
    #[allow(dead_code)]
    pub fn is_supported_document(path: &Path) -> bool {
        if let Some(ext) = Self::get_extension(path) {
            matches!(ext.as_str(), "txt" | "md" | "rst" | "log")
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_tilde() {
        let path = Path::new("~/test.txt");
        let expanded = FileOperations::expand_tilde(path).unwrap();

        if let Some(home) = dirs::home_dir() {
            assert_eq!(expanded, home.join("test.txt"));
        }
    }

    // The bare `~` form expands to home (no trailing slash).
    #[test]
    fn expand_tilde_handles_bare_tilde() {
        let path = Path::new("~");
        let expanded = FileOperations::expand_tilde(path).unwrap();
        if let Some(home) = dirs::home_dir() {
            assert_eq!(expanded, home);
        }
    }

    // Non-tilde paths pass through unchanged.
    #[test]
    fn expand_tilde_passes_non_tilde_through() {
        let path = Path::new("/etc/passwd");
        let expanded = FileOperations::expand_tilde(path).unwrap();
        assert_eq!(expanded, PathBuf::from("/etc/passwd"));
    }

    // The previous code silently passed `~user/foo` through as a literal
    // relative path. Now `~user/...` is left as a literal `PathBuf` (not
    // expanded), so callers don't accidentally resolve to a real `~user`
    // directory in cwd. Regression for #54 expand_tilde half.
    #[test]
    fn expand_tilde_does_not_expand_user_form() {
        let path = Path::new("~bob/foo");
        let expanded = FileOperations::expand_tilde(path).unwrap();
        assert_eq!(expanded, PathBuf::from("~bob/foo"));
    }

    #[test]
    fn test_get_extension() {
        assert_eq!(
            FileOperations::get_extension(Path::new("test.txt")),
            Some("txt".to_string())
        );
        assert_eq!(
            FileOperations::get_extension(Path::new("test.TXT")),
            Some("txt".to_string())
        );
        assert_eq!(FileOperations::get_extension(Path::new("test")), None);
    }

    #[test]
    fn test_is_supported_document() {
        assert!(FileOperations::is_supported_document(Path::new("test.txt")));
        assert!(FileOperations::is_supported_document(Path::new("test.md")));
        assert!(!FileOperations::is_supported_document(Path::new(
            "test.pdf"
        )));
    }
}
