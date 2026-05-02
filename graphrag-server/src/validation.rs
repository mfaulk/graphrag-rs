//! Input validation middleware and utilities for GraphRAG Server
//!
//! Provides request validation, sanitization, and security checks.
//!
//! Length limits below are byte-based (not codepoint/grapheme counts) — they
//! exist to bound DoS surface, not to mirror user-visible character counts.
//! Error messages refer to "bytes" so multi-byte UTF-8 input behaves
//! predictably (a CJK paragraph hits the byte limit much earlier than a
//! "characters"-worded message would suggest).

/// Maximum request body size (10MB)
pub const MAX_BODY_SIZE: usize = 10 * 1024 * 1024;

/// Maximum query length, in bytes
pub const MAX_QUERY_LENGTH: usize = 10_000;

/// Maximum document title length, in bytes
pub const MAX_TITLE_LENGTH: usize = 500;

/// Maximum document content length, in bytes (5MB of text)
pub const MAX_CONTENT_LENGTH: usize = 5 * 1024 * 1024;

/// Maximum top_k value for queries
pub const MAX_TOP_K: usize = 100;

/// Validation error response
#[derive(serde::Serialize)]
pub struct ValidationError {
    pub error: String,
    pub field: Option<String>,
    pub max_length: Option<usize>,
}

/// Validate query string
pub fn validate_query(query: &str) -> Result<(), ValidationError> {
    if query.is_empty() {
        return Err(ValidationError {
            error: "Query cannot be empty".to_string(),
            field: Some("query".to_string()),
            max_length: None,
        });
    }

    if query.len() > MAX_QUERY_LENGTH {
        return Err(ValidationError {
            error: format!("Query exceeds maximum length of {} bytes", MAX_QUERY_LENGTH),
            field: Some("query".to_string()),
            max_length: Some(MAX_QUERY_LENGTH),
        });
    }

    // Note: removed the SQL-injection blocklist that previously lived here.
    // None of the configured backends (Qdrant, in-memory) are SQL — there is
    // no SQL surface to inject into, so the check was producing false
    // positives on legitimate questions ("What does the SQL DROP TABLE
    // clause do?") while masking real concerns (prompt injection into
    // Ollama, payload injection into Qdrant filters). Real injection
    // hardening should live next to the backend that actually parses the
    // input, not as a generic substring match here.

    Ok(())
}

/// Validate document title
pub fn validate_title(title: &str) -> Result<(), ValidationError> {
    if title.is_empty() {
        return Err(ValidationError {
            error: "Title cannot be empty".to_string(),
            field: Some("title".to_string()),
            max_length: None,
        });
    }

    if title.len() > MAX_TITLE_LENGTH {
        return Err(ValidationError {
            error: format!("Title exceeds maximum length of {} bytes", MAX_TITLE_LENGTH),
            field: Some("title".to_string()),
            max_length: Some(MAX_TITLE_LENGTH),
        });
    }

    Ok(())
}

/// Validate document content
pub fn validate_content(content: &str) -> Result<(), ValidationError> {
    if content.is_empty() {
        return Err(ValidationError {
            error: "Content cannot be empty".to_string(),
            field: Some("content".to_string()),
            max_length: None,
        });
    }

    if content.len() > MAX_CONTENT_LENGTH {
        return Err(ValidationError {
            error: format!(
                "Content exceeds maximum length of {} bytes",
                MAX_CONTENT_LENGTH
            ),
            field: Some("content".to_string()),
            max_length: Some(MAX_CONTENT_LENGTH),
        });
    }

    Ok(())
}

/// Validate top_k parameter
pub fn validate_top_k(top_k: usize) -> Result<(), ValidationError> {
    if top_k == 0 {
        return Err(ValidationError {
            error: "top_k must be greater than 0".to_string(),
            field: Some("top_k".to_string()),
            max_length: None,
        });
    }

    if top_k > MAX_TOP_K {
        return Err(ValidationError {
            error: format!("top_k exceeds maximum value of {}", MAX_TOP_K),
            field: Some("top_k".to_string()),
            max_length: Some(MAX_TOP_K),
        });
    }

    Ok(())
}

/// Sanitize string by removing control characters
pub fn sanitize_string(input: &str) -> String {
    input
        .chars()
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
        .collect()
}

// Note: Request body size limits are now configured in main.rs using
// PayloadConfig and JsonConfig with MAX_BODY_SIZE constant.
//
// The previous `contains_sql_injection_patterns` helper was removed: none
// of the configured backends are SQL, so the substring blocklist (`drop
// table`, `delete from`, etc.) only produced false positives on legitimate
// queries. See #42.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_query() {
        // Valid queries
        assert!(validate_query("What is GraphRAG?").is_ok());
        assert!(validate_query("A".repeat(1000).as_str()).is_ok());

        // Invalid queries (length-based only — no SQL substring matching)
        assert!(validate_query("").is_err());
        assert!(validate_query(&"A".repeat(MAX_QUERY_LENGTH + 1)).is_err());
    }

    // Legitimate questions that *talk about* SQL must pass (regression for
    // #42 — the old SQL-injection blocklist 400'd these).
    #[test]
    fn validate_query_accepts_questions_that_mention_sql_keywords() {
        assert!(validate_query("What does the SQL DROP TABLE clause do?").is_ok());
        assert!(validate_query("How do I write an INSERT INTO statement?").is_ok());
        assert!(validate_query("Compare DELETE FROM and TRUNCATE.").is_ok());
        assert!(validate_query("'; DROP TABLE users; --").is_ok());
    }

    // Length errors must mention "bytes" (not "characters") — see #42.
    #[test]
    fn validate_query_length_error_says_bytes_not_characters() {
        let err = validate_query(&"A".repeat(MAX_QUERY_LENGTH + 1)).expect_err("too long");
        assert!(
            err.error.contains("bytes"),
            "expected 'bytes' in length error, got: {}",
            err.error
        );
    }

    #[test]
    fn test_validate_title() {
        assert!(validate_title("My Document").is_ok());
        assert!(validate_title("").is_err());
        assert!(validate_title(&"A".repeat(MAX_TITLE_LENGTH + 1)).is_err());
    }

    #[test]
    fn test_validate_content() {
        assert!(validate_content("Some content").is_ok());
        assert!(validate_content("").is_err());
        assert!(validate_content(&"A".repeat(MAX_CONTENT_LENGTH + 1)).is_err());
    }

    #[test]
    fn test_validate_top_k() {
        assert!(validate_top_k(5).is_ok());
        assert!(validate_top_k(100).is_ok());
        assert!(validate_top_k(0).is_err());
        assert!(validate_top_k(101).is_err());
    }

    #[test]
    fn test_sanitize_string() {
        assert_eq!(sanitize_string("Hello\x00World"), "HelloWorld");
        assert_eq!(sanitize_string("Hello\nWorld"), "Hello\nWorld");
        assert_eq!(sanitize_string("Normal text"), "Normal text");
    }

    // (test_sql_injection_detection removed alongside the
    // contains_sql_injection_patterns helper — see #42.)
}
