use std::fs::{self, File};
use std::io;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::Parser;

const DEFAULT_URL: &str =
    "https://huggingface.co/datasets/gwenshap/sales-transcripts/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet";

const DEFAULT_OUTPUT: &str = "examples/sales-transcripts-demo/data/raw/0000.parquet";

#[derive(Parser, Debug)]
#[command(
    name = "sales-fetch",
    about = "Download the gwenshap/sales-transcripts parquet shard from HuggingFace"
)]
struct Args {
    /// Override the source URL (defaults to the HF auto-converted parquet shard).
    #[arg(long, default_value = DEFAULT_URL)]
    url: String,

    /// Destination path for the downloaded parquet file.
    #[arg(short, long, default_value = DEFAULT_OUTPUT)]
    output: PathBuf,

    /// Re-download even if the destination file already exists.
    #[arg(long)]
    force: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.output.exists() && !args.force {
        println!(
            "{} already exists — pass --force to re-download",
            args.output.display()
        );
        return Ok(());
    }

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create_dir_all({})", parent.display()))?;
    }

    println!("downloading {} → {}", args.url, args.output.display());
    let response = ureq::get(&args.url)
        .call()
        .map_err(|e| anyhow!("HTTP GET {}: {e}", args.url))?;

    let status = response.status();
    if !(200..300).contains(&status) {
        return Err(anyhow!("unexpected HTTP status: {}", status));
    }

    // Atomic write: stream to a sibling .tmp file, then rename. If the HTTP
    // body drops mid-stream, the destination path is left untouched and a
    // subsequent run (without --force) will retry instead of feeding a
    // half-written file to the parquet parser.
    let mut tmp_path = args.output.clone();
    let tmp_name = match args.output.file_name() {
        Some(n) => format!("{}.tmp", n.to_string_lossy()),
        None => {
            return Err(anyhow!(
                "output path has no file name: {}",
                args.output.display()
            ))
        },
    };
    tmp_path.set_file_name(&tmp_name);

    let mut reader = response.into_reader();
    let bytes = {
        let mut tmp_file =
            File::create(&tmp_path).with_context(|| format!("create {}", tmp_path.display()))?;
        io::copy(&mut reader, &mut tmp_file)
            .with_context(|| format!("write {}", tmp_path.display()))?
    };

    fs::rename(&tmp_path, &args.output)
        .with_context(|| format!("rename {} -> {}", tmp_path.display(), args.output.display()))?;

    println!("wrote {} bytes to {}", bytes, args.output.display());
    Ok(())
}
