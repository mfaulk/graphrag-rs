use std::fs::{self, File};
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, StringArray};
use clap::Parser;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const DEFAULT_INPUT: &str = "examples/sales-transcripts-demo/data/raw/0000.parquet";
const DEFAULT_OUTPUT: &str = "examples/sales-transcripts-demo/data/txt";

#[derive(Parser, Debug)]
#[command(
    name = "sales-to-text",
    about = "Flatten gwenshap/sales-transcripts parquet into per-transcript .txt files"
)]
struct Args {
    /// Path to the parquet shard downloaded from HuggingFace.
    #[arg(short, long, default_value = DEFAULT_INPUT)]
    input: PathBuf,

    /// Directory where transcript_NNNN.txt files are written.
    #[arg(short, long, default_value = DEFAULT_OUTPUT)]
    output: PathBuf,

    /// Optional cap on the number of transcripts emitted (useful for demos).
    #[arg(long)]
    limit: Option<usize>,

    /// Name of the parquet column containing transcript text.
    #[arg(long, default_value = "text")]
    column: String,
}

// Remove every transcript_*.txt under `dir` so a re-run with a smaller --limit
// doesn't leave stale rows in the corpus that `run_pipeline` would still ingest.
fn purge_existing_transcripts(dir: &std::path::Path) -> Result<usize> {
    if !dir.exists() {
        return Ok(0);
    }
    let mut removed = 0usize;
    for entry in fs::read_dir(dir).with_context(|| format!("read_dir({})", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let is_transcript = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("transcript_") && n.ends_with(".txt"));
        if is_transcript {
            fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
            removed += 1;
        }
    }
    Ok(removed)
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.input.exists() {
        return Err(anyhow!(
            "input parquet not found at {} — run `cargo run -p sales-transcripts-demo --bin sales-fetch` first",
            args.input.display()
        ));
    }
    fs::create_dir_all(&args.output)
        .with_context(|| format!("create_dir_all({})", args.output.display()))?;

    let removed = purge_existing_transcripts(&args.output)?;
    if removed > 0 {
        println!(
            "removed {removed} stale transcript_*.txt file(s) from {}",
            args.output.display()
        );
    }

    let file = File::open(&args.input).with_context(|| format!("open {}", args.input.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .build()
        .context("build parquet reader")?;

    let mut written = 0usize;
    'batches: for batch in reader {
        let batch = batch.context("decode parquet batch")?;
        let col = batch.column_by_name(&args.column).ok_or_else(|| {
            anyhow!(
                "column '{}' not found — available: {:?}",
                args.column,
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect::<Vec<_>>()
            )
        })?;
        let strings = col
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("column '{}' is not Utf8/StringArray", args.column))?;

        for i in 0..strings.len() {
            if strings.is_null(i) {
                continue;
            }
            let text = strings.value(i);
            if text.trim().is_empty() {
                continue;
            }
            let out_path = args.output.join(format!("transcript_{written:04}.txt"));
            fs::write(&out_path, text).with_context(|| format!("write {}", out_path.display()))?;
            written += 1;
            if args.limit.is_some_and(|cap| written >= cap) {
                break 'batches;
            }
        }
    }

    println!(
        "wrote {} transcript file(s) to {}",
        written,
        args.output.display()
    );
    Ok(())
}
