// End-to-end demo of ingest → index → query against the gwenshap/sales-transcripts
// HuggingFace dataset. Public surface is `run_pipeline`, which the bins and the
// integration smoke test both go through.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use graphrag_core::retrieval::ExplainedAnswer;
use tokio::fs;

/// Minimal handle returned by [`run_pipeline`]. Owns the live `GraphRAG`
/// instance so callers can keep asking follow-up questions without rebuilding
/// the graph.
pub struct DemoSession {
    inner: graphrag_core::GraphRAG,
    docs_loaded: usize,
}

impl DemoSession {
    /// Number of transcript files ingested into this session.
    pub fn docs_loaded(&self) -> usize {
        self.docs_loaded
    }

    /// Run a follow-up query against the already-built graph.
    pub async fn ask(&mut self, query: &str) -> Result<ExplainedAnswer> {
        let answer = self
            .inner
            .ask_explained(query)
            .await
            .map_err(|e| anyhow!("ask_explained failed: {e}"))?;
        Ok(answer)
    }
}

/// Ingest every `*.txt` file under `transcripts_dir` (one transcript per file),
/// build the knowledge graph using the supplied [`graphrag_core::Config`], and
/// return a session ready to answer queries.
///
/// This is the single integration point exercised by the smoke test. Bins layer
/// fetch + parquet→txt conversion on top of it.
pub async fn run_pipeline(
    transcripts_dir: &Path,
    config: graphrag_core::Config,
) -> Result<DemoSession> {
    let metadata = fs::metadata(transcripts_dir)
        .await
        .with_context(|| format!("metadata({})", transcripts_dir.display()))?;
    if !metadata.is_dir() {
        return Err(anyhow!(
            "transcripts_dir is not a directory: {}",
            transcripts_dir.display()
        ));
    }

    let mut paths = Vec::new();
    let mut entries = fs::read_dir(transcripts_dir)
        .await
        .with_context(|| format!("read_dir({})", transcripts_dir.display()))?;
    while let Some(entry) = entries
        .next_entry()
        .await
        .with_context(|| format!("next_entry({})", transcripts_dir.display()))?
    {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("txt") {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(anyhow!(
            "no .txt transcripts found under {}",
            transcripts_dir.display()
        ));
    }

    let mut graphrag =
        graphrag_core::GraphRAG::new(config).map_err(|e| anyhow!("GraphRAG::new failed: {e}"))?;
    graphrag
        .initialize()
        .map_err(|e| anyhow!("GraphRAG::initialize failed: {e}"))?;

    let mut docs_loaded = 0usize;
    for path in &paths {
        let content = fs::read_to_string(path)
            .await
            .with_context(|| format!("read {}", path.display()))?;
        if content.trim().is_empty() {
            continue;
        }
        graphrag
            .add_document_from_text(&content)
            .map_err(|e| anyhow!("add_document_from_text({}): {e}", path.display()))?;
        docs_loaded += 1;
    }

    graphrag
        .build_graph()
        .await
        .map_err(|e| anyhow!("build_graph failed: {e}"))?;

    Ok(DemoSession {
        inner: graphrag,
        docs_loaded,
    })
}
