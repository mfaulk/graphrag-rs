//! Slash-command detection for TUI input.

/// Check if input text is a slash command
pub fn is_slash_command(input: &str) -> bool {
    input.trim().starts_with('/')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slash_command_detection() {
        assert!(is_slash_command("/config file.toml"));
        assert!(is_slash_command("  /load doc.txt"));
        assert!(!is_slash_command("What is GraphRAG?"));
        assert!(!is_slash_command(""));
    }
}
