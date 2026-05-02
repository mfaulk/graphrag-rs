//! UTF-8-safe string helpers.
//!
//! Slicing a `&str` by byte index panics with `byte index N is not a char
//! boundary` when N falls inside a multi-byte UTF-8 codepoint. These helpers
//! clamp byte offsets to the nearest preceding char boundary so that user
//! and LLM-supplied text containing emoji, accented characters, or CJK
//! cannot crash the caller.

/// Truncate `s` to at most `max_bytes` bytes, clamping back to a UTF-8
/// char boundary so the returned slice is always valid.
///
/// Returns the original string when it is already shorter than `max_bytes`.
pub fn truncate_chars(s: &str, max_bytes: usize) -> &str {
    let mut end = max_bytes.min(s.len());
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Slice `s` from byte index `start` to byte index `end`, clamping each
/// endpoint backward to the nearest UTF-8 char boundary.
///
/// Useful when both endpoints are derived from arithmetic
/// (e.g. `pos.saturating_sub(N)` and `pos + N`) where neither is
/// guaranteed to land on a codepoint boundary even if `pos` itself does.
/// Returns an empty slice when `start >= end`.
pub fn slice_on_char_boundary(s: &str, start: usize, end: usize) -> &str {
    let mut start = start.min(s.len());
    let mut end = end.min(s.len());
    if start >= end {
        return "";
    }
    while start > 0 && !s.is_char_boundary(start) {
        start -= 1;
    }
    while end > start && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_returns_full_string_when_shorter_than_max() {
        assert_eq!(truncate_chars("hello", 100), "hello");
    }

    #[test]
    fn truncate_clamps_to_char_boundary_inside_multi_byte_codepoint() {
        // The crab emoji (🦀) is 4 bytes (U+1F980). max_bytes=8 lands inside
        // the second emoji's codepoint.
        let s = "héllo🦀world"; // 'é' = 2 bytes, "🦀" = 4 bytes
        let truncated = truncate_chars(s, 8);
        // Must not panic and must produce valid UTF-8.
        assert_eq!(truncated, "héllo");
    }

    #[test]
    fn truncate_at_exact_boundary_preserves_codepoint() {
        let s = "🦀abc";
        assert_eq!(truncate_chars(s, 4), "🦀");
    }

    #[test]
    fn truncate_zero_length_returns_empty() {
        assert_eq!(truncate_chars("héllo", 0), "");
    }

    #[test]
    fn truncate_handles_pure_ascii() {
        assert_eq!(truncate_chars("abcdefghij", 5), "abcde");
    }

    #[test]
    fn slice_on_char_boundary_clamps_both_endpoints() {
        let s = "héllo🦀world";
        // Bytes: h=1, é=2, l=1, l=1, o=1, 🦀=4, w=1...
        // Char boundaries at byte offsets: 0,1,3,4,5,6,10,11,...
        // start=2 lands inside é → clamps back to 1 (boundary before é).
        // end=8 lands inside 🦀 (bytes 6..10) → clamps back to 6.
        // Resulting slice [1..6] = "éllo".
        let snip = slice_on_char_boundary(s, 2, 8);
        assert_eq!(snip, "éllo");
    }

    #[test]
    fn slice_on_char_boundary_clamps_when_only_end_is_inside_codepoint() {
        let s = "abc🦀def";
        // start=0 (boundary), end=5 (mid-🦀, bytes 3..7) → clamps back to 3.
        assert_eq!(slice_on_char_boundary(s, 0, 5), "abc");
    }

    #[test]
    fn slice_on_char_boundary_caps_at_string_length() {
        let s = "abc";
        assert_eq!(slice_on_char_boundary(s, 0, 100), "abc");
    }

    #[test]
    fn slice_on_char_boundary_returns_empty_when_start_past_end() {
        let s = "abc";
        assert_eq!(slice_on_char_boundary(s, 5, 1), "");
    }
}
