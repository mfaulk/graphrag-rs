//! Regression test for #20: byte-range string slicing panics on multi-byte UTF-8.
//!
//! Previously the workspace had ~12 sites that did things like
//! `&content[..200]` or `&s[..s.len().min(N)]`. When N landed inside a
//! multi-byte codepoint (any document with emoji, accented characters, or
//! CJK at the wrong offset) the slice panicked with "byte index N is not a
//! char boundary". The two helpers in `graphrag_core::util::text_safe`
//! clamp back to the nearest preceding char boundary.

use graphrag_core::util::text_safe::{slice_on_char_boundary, truncate_chars};

#[test]
fn truncate_does_not_panic_when_max_lands_inside_emoji() {
    // The crab emoji 🦀 is 4 bytes (U+1F980).
    let s = "abc🦀def";
    // Bytes: 'a'(1) 'b'(1) 'c'(1) 🦀(4) 'd'(1) 'e'(1) 'f'(1) — total 10.
    // max_bytes = 5 lands inside the emoji (between bytes 4 and 7).
    let truncated = truncate_chars(s, 5);
    // Must not panic and must produce valid UTF-8.
    assert_eq!(truncated, "abc");
}

#[test]
fn truncate_does_not_panic_on_cjk_text() {
    // 中 is 3 bytes in UTF-8.
    let s = "hello 中文 world";
    for n in 0..s.len() {
        // None of these may panic, regardless of where N falls.
        let _ = truncate_chars(s, n);
    }
}

#[test]
fn slice_on_char_boundary_handles_arithmetic_endpoints() {
    // Mimics the usage in inference.rs / rograg/logic_form.rs:
    //   start = pos.saturating_sub(N); end = (pos + len + N).min(content.len())
    // Either may land mid-codepoint. The helper must clamp.
    let content = "前文 keyword 後文 emoji 🎉 trailing";
    let pattern = "keyword";
    let pos = content.find(pattern).unwrap();
    let start = pos.saturating_sub(5);
    let end = (pos + pattern.len() + 5).min(content.len());

    let context = slice_on_char_boundary(content, start, end);
    assert!(context.contains("keyword"));
}

#[test]
fn helpers_are_safe_under_aggressive_fuzz() {
    // Sweep all possible (start, end) pairs against a multi-byte string.
    // Any panic would fail the test.
    let s = "a中b🦀c";
    for start in 0..=s.len() + 2 {
        for end in 0..=s.len() + 2 {
            let _ = slice_on_char_boundary(s, start, end);
        }
        let _ = truncate_chars(s, start);
    }
}
