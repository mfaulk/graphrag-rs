// Hierarchical chunker with both character- and token-based windowing.
//
// Token mode uses the cl100k_base BPE tokenizer (`tiktoken-rs`), which matches
// OpenAI's `gpt-4o`/`gpt-4o-mini` tokenization. Token-mode default sizes follow
// Edge et al. 2024 (arXiv 2404.16130) §2.1 ablation: 600-token windows with
// 60-token overlap.

use std::sync::{Arc, OnceLock};
use tiktoken_rs::{cl100k_base, CoreBPE};

/// Unit used by `chunk_size`, `overlap`, and `min_chunk_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkingMode {
    /// Counts in raw byte length (legacy behavior, kept for backwards compat).
    Chars,
    /// Counts in cl100k_base BPE tokens (default).
    Tokens,
}

impl Default for ChunkingMode {
    fn default() -> Self {
        ChunkingMode::Tokens
    }
}

/// Default chunk size in tokens (Edge et al. 2024 §2.1 ablation).
pub const DEFAULT_TOKEN_CHUNK_SIZE: usize = 600;
/// Default overlap in tokens (10% of chunk size).
pub const DEFAULT_TOKEN_OVERLAP: usize = 60;
/// Default minimum chunk size in tokens.
pub const DEFAULT_TOKEN_MIN_CHUNK_SIZE: usize = 50;

fn cl100k() -> &'static Arc<CoreBPE> {
    static BPE: OnceLock<Arc<CoreBPE>> = OnceLock::new();
    BPE.get_or_init(|| Arc::new(cl100k_base().expect("cl100k_base BPE bundled with tiktoken-rs")))
}

/// Hierarchical text chunking with semantic boundary preservation.
///
/// Mirrors the LangChain `RecursiveCharacterTextSplitter` approach in `Chars`
/// mode and uses fixed-size cl100k_base token windows in `Tokens` mode.
pub struct HierarchicalChunker {
    /// Hierarchical separators in order of preference (used in `Chars` mode).
    separators: Vec<String>,
    /// Minimum chunk size, in units of `mode`.
    min_chunk_size: usize,
    /// Counting mode.
    mode: ChunkingMode,
}

impl HierarchicalChunker {
    /// Create a new hierarchical chunker with default separators (char mode).
    pub fn new() -> Self {
        Self {
            // Following 2024 research best practices - hierarchical separators
            separators: vec![
                "\n\n".to_string(), // Paragraph breaks (highest priority)
                "\n".to_string(),   // Line breaks
                ". ".to_string(),   // Sentence endings with space
                "! ".to_string(),   // Exclamation sentences
                "? ".to_string(),   // Question sentences
                "; ".to_string(),   // Semicolon clauses
                ": ".to_string(),   // Colon clauses
                " ".to_string(),    // Word boundaries
                "".to_string(),     // Character level (fallback)
            ],
            min_chunk_size: 50,
            mode: ChunkingMode::Chars,
        }
    }

    /// Create chunker with custom separators (char mode).
    pub fn with_separators(separators: Vec<String>) -> Self {
        Self {
            separators,
            min_chunk_size: 50,
            mode: ChunkingMode::Chars,
        }
    }

    /// Set minimum chunk size (in units of the current `mode`).
    pub fn with_min_size(mut self, min_size: usize) -> Self {
        self.min_chunk_size = min_size;
        self
    }

    /// Switch counting mode. Defaults to `Chars` for backwards compat;
    /// callers that want paper-aligned token windows should call
    /// `.with_mode(ChunkingMode::Tokens)`.
    pub fn with_mode(mut self, mode: ChunkingMode) -> Self {
        self.mode = mode;
        self
    }

    /// Current counting mode.
    pub fn mode(&self) -> ChunkingMode {
        self.mode
    }

    /// Split text into semantically coherent chunks.
    ///
    /// `chunk_size` and `overlap` are interpreted in units of `self.mode()`.
    /// In `Tokens` mode, decoded chunks may have slightly different
    /// leading/trailing whitespace than the same byte range from a char-mode
    /// slice — this is expected: BPE round-trips are exact for token-aligned
    /// slices but not for arbitrary byte offsets.
    pub fn chunk_text(&self, text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
        match self.mode {
            ChunkingMode::Chars => self.chunk_text_chars(text, chunk_size, overlap),
            ChunkingMode::Tokens => self.chunk_text_tokens(text, chunk_size, overlap),
        }
    }

    fn chunk_text_chars(&self, text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < text.len() {
            let mut end = (start + chunk_size).min(text.len());

            // Ensure we're on a UTF-8 character boundary first
            while end > start && !text.is_char_boundary(end) {
                end -= 1;
            }

            // If we're at the exact end, no need to adjust
            if end >= text.len() {
                let chunk = &text[start..];
                if chunk.trim().len() >= self.min_chunk_size {
                    chunks.push(chunk.to_string());
                }
                break;
            }

            // Find the best boundary to avoid semantic truncation
            let optimal_end = self.find_optimal_boundary(text, start, end);

            // If we found a good boundary, use it
            if optimal_end > start {
                end = optimal_end;
            }

            let chunk = &text[start..end];

            if chunk.trim().len() >= self.min_chunk_size {
                chunks.push(chunk.to_string());
            }

            if end >= text.len() {
                break;
            }

            // Calculate next start with overlap, preserving semantic boundaries
            let mut next_start = end.saturating_sub(overlap);

            // Ensure next start is on a UTF-8 boundary
            while next_start > 0 && !text.is_char_boundary(next_start) {
                next_start -= 1;
            }

            // Try to align next start with word boundary
            next_start = self.find_word_boundary_backward(text, next_start);

            start = next_start;
        }

        chunks
    }

    /// Token-based windowing using cl100k_base BPE.
    ///
    /// Produces fixed-size token windows (`chunk_size` tokens) with `overlap`
    /// tokens of step-back. Uses `encode_ordinary` (no special-token handling)
    /// so prose containing literal "<|...|>" sequences is treated as text.
    fn chunk_text_tokens(&self, text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
        if chunk_size == 0 {
            return Vec::new();
        }
        let bpe = cl100k();
        let tokens = bpe.encode_ordinary(text);
        if tokens.is_empty() {
            return Vec::new();
        }

        let stride = chunk_size.saturating_sub(overlap).max(1);
        let mut chunks = Vec::new();
        let mut start = 0usize;

        while start < tokens.len() {
            let end = (start + chunk_size).min(tokens.len());
            let slice = &tokens[start..end];

            // cl100k_base BPE encodes UTF-8 byte-level: a single multi-byte
            // codepoint may span multiple tokens, so a token-aligned slice
            // CAN begin or end mid-codepoint. Use `_decode_native` to recover
            // the raw bytes and `from_utf8_lossy` to replace any incomplete
            // sequences at the boundaries with U+FFFD. This preserves all
            // input rather than silently dropping chunks on rare boundaries.
            let bytes = bpe._decode_native(slice);
            let decoded = String::from_utf8_lossy(&bytes).into_owned();
            let token_count = slice.len();
            if token_count >= self.min_chunk_size || end >= tokens.len() {
                if !decoded.trim().is_empty() {
                    chunks.push(decoded);
                }
            }

            if end >= tokens.len() {
                break;
            }
            start += stride;
        }

        chunks
    }

    /// Find optimal boundary using hierarchical separators
    fn find_optimal_boundary(&self, text: &str, start: usize, max_end: usize) -> usize {
        let search_text = &text[start..max_end];

        // Try each separator in order of preference
        for separator in &self.separators {
            if separator.is_empty() {
                continue;
            }

            // Find the last occurrence of this separator within our range
            if let Some(sep_pos) = search_text.rfind(separator) {
                let boundary = start + sep_pos + separator.len();

                // Make sure we're not too close to the start (maintain minimum chunk size)
                if boundary > start + (max_end - start) / 4 {
                    return boundary;
                }
            }
        }

        // If no good separator found, try to at least end at a word boundary
        self.find_word_boundary_backward(text, max_end)
    }

    /// Find the nearest word boundary going backward from the given position
    fn find_word_boundary_backward(&self, text: &str, mut pos: usize) -> usize {
        // Ensure we're on a UTF-8 boundary
        while pos > 0 && !text.is_char_boundary(pos) {
            pos -= 1;
        }

        // Look for whitespace (word boundary) going backward
        while pos > 0 {
            if let Some(ch) = text.chars().nth(pos.saturating_sub(1)) {
                if ch.is_whitespace() {
                    return pos;
                }
            }
            pos = pos.saturating_sub(1);

            // Ensure we stay on UTF-8 boundaries
            while pos > 0 && !text.is_char_boundary(pos) {
                pos -= 1;
            }
        }

        pos
    }

    /// Advanced sentence boundary detection
    pub fn find_sentence_boundary(
        &self,
        text: &str,
        start: usize,
        preferred_end: usize,
    ) -> Option<usize> {
        let safe_start = self.find_char_boundary(text, start);
        let safe_end = self.find_char_boundary(text, preferred_end);

        if safe_start >= safe_end {
            return None;
        }

        let search_window = &text[safe_start..safe_end];

        // Look for sentence boundaries in the last part of the chunk
        let search_start = search_window.len().saturating_sub(300); // Larger window for better context
        let safe_search_start = self.find_char_boundary_in_slice(search_window, search_start);
        let search_text = &search_window[safe_search_start..];

        // Enhanced sentence boundary detection
        let sentence_endings = ['.', '!', '?'];
        let mut last_boundary = None;

        for (i, ch) in search_text.char_indices() {
            if sentence_endings.contains(&ch) {
                // Check if next character is whitespace or end of text
                let next_pos = i + ch.len_utf8();
                if next_pos >= search_text.len() {
                    last_boundary = Some(safe_start + safe_search_start + next_pos);
                } else if let Some(next_char) = search_text.chars().nth(next_pos) {
                    // More sophisticated sentence boundary detection
                    if next_char.is_whitespace() && (next_char == '\n' || next_char == ' ') {
                        // Make sure this isn't an abbreviation or decimal
                        if !self.is_likely_abbreviation(search_text, i) {
                            last_boundary = Some(safe_start + safe_search_start + next_pos);
                        }
                    }
                } else {
                    // Character at next_pos does not exist
                }
            }
        }

        last_boundary
    }

    /// Check if a period is likely part of an abbreviation
    fn is_likely_abbreviation(&self, text: &str, period_pos: usize) -> bool {
        // Simple heuristics for common abbreviations
        if period_pos == 0 {
            return false;
        }

        // Check for common abbreviation patterns
        let before_period = &text[..period_pos];
        if let Some(word_start) = before_period.rfind(' ') {
            let potential_abbrev = &before_period[word_start + 1..];

            // Common abbreviations
            let abbreviations = [
                "Dr", "Mr", "Mrs", "Ms", "Prof", "Jr", "Sr", "Inc", "Corp", "Ltd", "Co", "etc",
                "vs", "e.g", "i.e", "cf", "pp",
            ];

            return abbreviations
                .iter()
                .any(|&abbrev| potential_abbrev.eq_ignore_ascii_case(abbrev));
        }

        // Single letter followed by period (likely initial)
        if period_pos == 1
            && before_period
                .chars()
                .next()
                .unwrap_or(' ')
                .is_ascii_uppercase()
        {
            return true;
        }

        false
    }

    /// Find a safe character boundary at or before the given position
    fn find_char_boundary(&self, text: &str, mut pos: usize) -> usize {
        pos = pos.min(text.len());
        while pos > 0 && !text.is_char_boundary(pos) {
            pos -= 1;
        }
        pos
    }

    /// Find a safe character boundary within a slice at or before the given position
    fn find_char_boundary_in_slice(&self, text: &str, mut pos: usize) -> usize {
        pos = pos.min(text.len());
        while pos > 0 && !text.is_char_boundary(pos) {
            pos -= 1;
        }
        pos
    }
}

impl Default for HierarchicalChunker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_chunking() {
        let chunker = HierarchicalChunker::new();
        let text = "This is a test document.\n\nIt has multiple paragraphs. Each paragraph should be preserved as much as possible. This helps maintain semantic coherence in the chunks.";

        let chunks = chunker.chunk_text(text, 100, 20);

        assert!(!chunks.is_empty(), "Chunks should not be empty");

        // The chunker respects \n\n as highest priority separator
        // With min_chunk_size=50, first paragraph (26 chars: "This is a test document.")
        // is too short and will be filtered out
        // The second paragraph is long enough (128 chars) and will be chunked

        // Verify that we got meaningful chunks from the second paragraph
        assert!(chunks.len() >= 1, "Should have at least one chunk");

        // First chunk should start from second paragraph
        assert!(
            chunks[0].contains("multiple paragraphs")
                || chunks[0].contains("preserved")
                || chunks[0].contains("coherence"),
            "Chunks should contain content from second paragraph. Got: {:?}",
            chunks
        );

        // Verify chunks respect semantic boundaries (don't split in middle of words)
        for (i, chunk) in chunks.iter().enumerate() {
            let trimmed = chunk.trim();
            if !trimmed.is_empty() {
                // Should have substantial content (above min_chunk_size)
                assert!(
                    trimmed.len() >= 50,
                    "Chunk {} should be >= min_chunk_size (50): length={}",
                    i,
                    trimmed.len()
                );

                let last_char = trimmed.chars().last().unwrap();
                assert!(
                    last_char.is_whitespace()
                        || last_char.is_ascii_punctuation()
                        || trimmed == text.trim(),
                    "Chunk {} should end at word/sentence boundary",
                    i
                );
            }
        }
    }

    #[test]
    fn test_sentence_boundary_detection() {
        let chunker = HierarchicalChunker::new();
        let text = "Dr. Smith went to the store. He bought some milk. Then he went home.";

        // Should not break on "Dr." abbreviation
        if let Some(boundary) = chunker.find_sentence_boundary(text, 0, 30) {
            let chunk = &text[0..boundary];
            assert!(!chunk.ends_with("Dr."));
        }
    }

    #[test]
    fn test_word_boundary_preservation() {
        let chunker = HierarchicalChunker::new();
        let text = "This is a very long sentence that should be split at word boundaries rather than in the middle of words.";

        let chunks = chunker.chunk_text(text, 50, 10);

        // No chunk should end with a partial word
        for chunk in &chunks {
            let trimmed = chunk.trim();
            if !trimmed.is_empty() {
                let last_char = trimmed.chars().last().unwrap();
                // Should end with whitespace, punctuation, or be the complete text
                assert!(
                    last_char.is_whitespace()
                        || last_char.is_ascii_punctuation()
                        || chunk.trim() == text.trim()
                );
            }
        }
    }

    /// Pins existing char-mode behavior so the token-mode refactor doesn't change it.
    #[test]
    fn chunk_text_with_chars_mode_unchanged_from_old_default() {
        let chunker = HierarchicalChunker::new();
        assert_eq!(chunker.mode(), ChunkingMode::Chars);
        let text = "This is a test document.\n\nIt has multiple paragraphs. \
                    Each paragraph should be preserved as much as possible. \
                    This helps maintain semantic coherence in the chunks.";
        let chunks = chunker.chunk_text(text, 100, 20);
        // Same behavior as the legacy `test_hierarchical_chunking` assertions.
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(c.trim().len() >= 50);
        }
    }

    /// Token mode produces the expected number of windows for known token counts.
    #[test]
    fn chunk_text_with_token_mode_emits_expected_chunk_count() {
        // 100 tokens of "word " repeats ≈ 100 BPE tokens for cl100k_base.
        // Use a deterministic count: encode and slide manually to compute the
        // expected chunk count, then compare.
        let bpe = cl100k_base().unwrap();
        let raw = "lorem ipsum dolor sit amet ".repeat(60);
        let tokens = bpe.encode_ordinary(&raw);
        let n = tokens.len();
        let chunk_size = 32usize;
        let overlap = 8usize;
        let stride = chunk_size - overlap;
        // Sliding-window count: number of starts s ∈ {0, stride, 2*stride, ...}
        // such that s < n.
        let expected = (n + stride - 1) / stride;
        let chunker = HierarchicalChunker::new()
            .with_mode(ChunkingMode::Tokens)
            .with_min_size(0);
        let chunks = chunker.chunk_text(&raw, chunk_size, overlap);
        assert_eq!(
            chunks.len(),
            expected,
            "chunk count mismatch for {} tokens, size={}, overlap={}",
            n,
            chunk_size,
            overlap
        );
    }

    /// Token mode round-trips simple English prose without garbage characters.
    #[test]
    fn chunk_text_with_token_mode_round_trips_text_for_simple_input() {
        let chunker = HierarchicalChunker::new()
            .with_mode(ChunkingMode::Tokens)
            .with_min_size(0);
        let text = "The quick brown fox jumps over the lazy dog. \
                    Pack my box with five dozen liquor jugs. \
                    How vexingly quick daft zebras jump.";
        let chunks = chunker.chunk_text(text, 32, 0);
        let joined: String = chunks.concat();
        // No-overlap concatenation should reproduce original token stream.
        let bpe = cl100k_base().unwrap();
        let original = bpe.decode(bpe.encode_ordinary(text)).unwrap();
        assert_eq!(joined, original);
        // Each chunk must be valid UTF-8 (it already is, since `String`).
        for c in &chunks {
            assert!(!c.is_empty());
            assert!(c.is_char_boundary(0));
            assert!(c.is_char_boundary(c.len()));
        }
    }

    /// Token mode produces valid UTF-8 strings even when multi-byte codepoints
    /// straddle chunk boundaries (lossy decode replaces incomplete sequences
    /// with U+FFFD rather than dropping or panicking).
    #[test]
    fn chunk_text_with_token_mode_handles_unicode_cleanly() {
        let chunker = HierarchicalChunker::new()
            .with_mode(ChunkingMode::Tokens)
            .with_min_size(0);
        // Mix CJK, emoji, accented Latin.
        let text = "こんにちは世界。Привет мир. Hello world! 🚀🌍🎉 \
                    日本語のテキスト Français naïve café résumé"
            .repeat(8);
        let chunks = chunker.chunk_text(&text, 16, 4);
        assert!(!chunks.is_empty());
        for c in &chunks {
            // Every chunk must be valid UTF-8 (String guarantees it) and
            // start/end at char boundaries.
            assert!(c.is_char_boundary(0));
            assert!(c.is_char_boundary(c.len()));
        }
        // Replacement chars (U+FFFD) may appear at chunk boundaries because
        // cl100k_base BPE is byte-level: a single multi-byte codepoint can
        // span two tokens and a token-aligned slice can begin or end mid-
        // codepoint. The previous implementation silently dropped those
        // chunks; we now lossily decode so no input bytes are lost. Cap the
        // total replacement-char count to a small, finite number to catch
        // regressions where the whole stream becomes garbled.
        let total_replacements: usize = chunks.iter().map(|c| c.matches('\u{FFFD}').count()).sum();
        assert!(
            total_replacements <= chunks.len() * 2,
            "too many U+FFFD replacements ({}) across {} chunks: {:?}",
            total_replacements,
            chunks.len(),
            chunks
        );
    }
}
