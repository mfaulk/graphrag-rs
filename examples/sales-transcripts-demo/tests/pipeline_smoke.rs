use std::path::PathBuf;

use graphrag_core::Config;
use sales_transcripts_demo::run_pipeline;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("sample_transcripts")
}

#[allow(clippy::field_reassign_with_default)]
fn hash_only_config() -> Config {
    let mut config = Config::default();
    config.chunk_size = 256;
    config.chunk_overlap = 40;
    config.top_k_results = Some(5);
    // Force the Ollama LLM call off so the test runs offline; the retrieval
    // surface is what the smoke test validates, not LLM synthesis.
    config.ollama.enabled = false;
    config.embeddings.fallback_to_hash = true;
    config
}

// End-to-end ingest → index → query against six committed sample transcripts:
// run_pipeline must build a graph from per-transcript .txt files and surface
// the recurring "Helio Analytics" entity in retrieval sources.
#[tokio::test]
async fn smoke_ingest_index_query_returns_sources() {
    let dir = fixtures_dir();
    assert!(
        dir.is_dir(),
        "fixtures dir missing at {} — committed sample transcripts required",
        dir.display()
    );
    let txt_count = std::fs::read_dir(&dir)
        .expect("read fixtures dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("txt"))
        .count();
    assert!(
        txt_count >= 5,
        "expected at least 5 fixture transcripts, found {txt_count}"
    );

    let mut session = run_pipeline(&dir, hash_only_config())
        .await
        .expect("run_pipeline should succeed on committed fixtures");

    let answer = session
        .ask("What is Helio Analytics and what objections do customers raise?")
        .await
        .expect("ask should return an explained answer");

    assert!(
        !answer.sources.is_empty(),
        "expected at least one retrieved source for a query about Helio Analytics"
    );

    let any_source_mentions_helio = answer
        .sources
        .iter()
        .any(|s| s.excerpt.to_lowercase().contains("helio"));
    let answer_mentions_helio = answer.answer.to_lowercase().contains("helio");
    assert!(
        any_source_mentions_helio || answer_mentions_helio,
        "neither answer nor any source excerpt mentioned 'helio' — retrieval likely broken.\n\
         answer: {}\n\
         sources: {:#?}",
        answer.answer,
        answer.sources
    );
}
