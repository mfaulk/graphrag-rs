use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use sales_transcripts_demo::run_pipeline;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Mode {
    /// Algorithmic extraction + hash embeddings. Zero deps, deterministic, fast.
    Hash,
    /// Semantic extraction via local Ollama (mistral-nemo + nomic-embed-text).
    /// Requires `ollama serve` running locally and both models pulled.
    Ollama,
}

#[derive(Parser, Debug)]
#[command(
    name = "sales-pipeline",
    about = "End-to-end demo: ingest sales-transcripts → build graph → answer queries"
)]
struct Args {
    /// Directory of per-transcript .txt files (output of `sales-to-text`).
    #[arg(
        short,
        long,
        default_value = "examples/sales-transcripts-demo/data/txt"
    )]
    transcripts: PathBuf,

    /// Pipeline mode.
    #[arg(short, long, value_enum, default_value_t = Mode::Hash)]
    mode: Mode,

    /// Pipe-separated query list. Each query is run with `ask_explained`.
    #[arg(
        short,
        long,
        default_value = "What products are mentioned across these transcripts?\
                        |What objections do customers raise about pricing?\
                        |What integration concerns come up?\
                        |Which customers are blocked by procurement timelines?"
    )]
    queries: String,
}

#[allow(clippy::field_reassign_with_default)]
fn build_config(mode: Mode) -> graphrag_core::Config {
    let mut config = graphrag_core::Config::default();
    config.chunk_size = 256;
    config.chunk_overlap = 40;
    config.text.chunk_size = 256;
    config.text.chunk_overlap = 40;
    config.top_k_results = Some(5);
    config.entities.entity_types = vec![
        "PERSON".to_string(),
        "ORG".to_string(),
        "PRODUCT".to_string(),
        "PAIN_POINT".to_string(),
        "OBJECTION".to_string(),
    ];

    match mode {
        Mode::Hash => {
            config.approach = "algorithmic".to_string();
            config.ollama.enabled = false;
            config.embeddings.fallback_to_hash = true;
        },
        Mode::Ollama => {
            config.approach = "semantic".to_string();
            config.ollama.enabled = true;
            config.ollama.chat_model = "mistral-nemo:latest".to_string();
            config.ollama.embedding_model = "nomic-embed-text".to_string();
            config.entities.use_gleaning = true;
            config.entities.max_gleaning_rounds = 1;
            config.embeddings.fallback_to_hash = true;
        },
    }

    config
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let args = Args::parse();
    let queries: Vec<&str> = args.queries.split('|').map(str::trim).collect();

    if matches!(args.mode, Mode::Ollama) {
        println!(
            "ℹ Ollama mode: chat synthesis + LLM entity extraction use Ollama, but \
             retrieval embeddings remain hash-based until graphrag-rs#91 wires the \
             configured embedder into RetrievalSystem."
        );
    }

    println!(
        "▶ ingesting transcripts from {} (mode: {:?})",
        args.transcripts.display(),
        args.mode
    );
    let started = std::time::Instant::now();
    let mut session = run_pipeline(&args.transcripts, build_config(args.mode)).await?;
    println!(
        "✔ ingested {} transcript(s) and built graph in {:.2}s\n",
        session.docs_loaded(),
        started.elapsed().as_secs_f32()
    );

    for (i, query) in queries.iter().enumerate() {
        println!("─── query {} ───────────────────────────────", i + 1);
        println!("Q: {query}");
        let answer = session.ask(query).await?;
        println!("A: {}", answer.answer);
        println!("confidence: {:.0}%", answer.confidence * 100.0);
        println!("sources ({}):", answer.sources.len());
        for src in answer.sources.iter().take(3) {
            let snippet: String = src.excerpt.chars().take(120).collect();
            println!(
                "  • [{:?} {} · {:.2}] {snippet}",
                src.source_type, src.id, src.relevance_score
            );
        }
        println!();
    }

    Ok(())
}
