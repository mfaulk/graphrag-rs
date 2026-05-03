# Sales-Transcripts End-to-End Demo

End-to-end ingest → index → query against the public
[`gwenshap/sales-transcripts`](https://huggingface.co/datasets/gwenshap/sales-transcripts)
dataset on Hugging Face.

The demo pulls Hugging Face's auto-converted parquet shard at
`refs/convert/parquet/default/train/0000.parquet` — that's 995 rows of pure
transcript text (~140 KB compressed). The dataset card reports 1,935 rows and
~34 MB total because it counts a separate config that bundles pre-computed
OpenAI embeddings; we don't need those here, since `graphrag-core` builds its
own.

This crate is a self-contained reference for embedding `graphrag-core` into your
own project: a small library entry point (`run_pipeline`) plus three thin
binaries (`fetch`, `to_text`, `pipeline`) that wire it together.

## Run the demo (zero-LLM, ~2 seconds)

```bash
# 1. Download the parquet shard from Hugging Face (~140 KB, no auth required).
cargo run -p sales-transcripts-demo --bin sales-fetch

# 2. Flatten parquet into per-transcript .txt files (cap with --limit for speed).
cargo run -p sales-transcripts-demo --bin sales-to-text -- --limit 200

# 3. Build the graph and answer a curated query set.
cargo run -p sales-transcripts-demo --bin sales-pipeline
```

The default mode (`--mode hash`) uses pattern-based entity extraction and hash
embeddings — no API keys, no Ollama, no model downloads. End-to-end on 200
transcripts takes around 2 seconds.

## Run with local LLM synthesis (Ollama)

```bash
ollama pull mistral-nemo
ollama pull nomic-embed-text

cargo run -p sales-transcripts-demo --bin sales-pipeline -- \
  --mode ollama \
  --transcripts examples/sales-transcripts-demo/data/txt
```

`--mode ollama` enables LLM-driven entity extraction (gleaning) and answer
synthesis. Embeddings still come from the configured Ollama model. Expect
roughly 30s–2min depending on transcript count and your local hardware.

## Run via the graphrag CLI / TUI

The demo ships two JSON5 configs that work with the existing graphrag-cli flows:

```bash
# Non-interactive bench against the hash-only config.
cargo run -p graphrag-cli -- bench \
  --config examples/sales-transcripts-demo/configs/sales_hash_only.json5 \
  --book examples/sales-transcripts-demo/data/all.txt \
  --questions "What products are mentioned?|What objections come up most?"

# Interactive TUI with local Ollama.
cargo run -p graphrag-cli -- tui \
  --config examples/sales-transcripts-demo/configs/sales_ollama.json5
```

For TUI use, generate the concatenated `data/all.txt` from per-transcript files:

```bash
cat examples/sales-transcripts-demo/data/txt/*.txt \
  > examples/sales-transcripts-demo/data/all.txt
```

## Layout

| Path | What it does |
| --- | --- |
| `src/lib.rs` | `run_pipeline(transcripts_dir, config) -> DemoSession` — one-call ingest+index, returns a session you keep asking. |
| `src/bin/fetch.rs` | Downloads `gwenshap/sales-transcripts` parquet shard from HF (no `huggingface-cli` required, just `ureq`). |
| `src/bin/to_text.rs` | Reads parquet via `arrow`/`parquet`, writes one `transcript_NNNN.txt` per row. |
| `src/bin/pipeline.rs` | The full demo: builds Config programmatically, runs ingest+index+query loop, prints `ask_explained` answers with sources. |
| `configs/sales_hash_only.json5` | Algorithmic mode, no API keys. Drives `graphrag-cli bench` / `tui`. |
| `configs/sales_ollama.json5` | Semantic mode, local Ollama. Drives `graphrag-cli bench` / `tui`. |
| `tests/pipeline_smoke.rs` | TDD smoke test: ingest 6 committed sample transcripts, assert retrieval surfaces the recurring entity. Runs in CI without API keys. |
| `tests/fixtures/sample_transcripts/` | 6 hand-crafted transcripts referencing "Helio Analytics" and "Northwind Logistics" — the smoke test's input. |

## Hybrid mode (cloud embeddings + local Ollama chat)

Not yet wired end-to-end. `graphrag-core` currently constructs a hash embedder
unconditionally inside `RetrievalSystem::new` regardless of
`config.embeddings.backend`, so setting `backend = "openai"` has no runtime
effect (acknowledged in-tree at `graphrag-core/src/retrieval/mod.rs:75-81`).
Tracked in [#91](https://github.com/mfaulk/graphrag-rs/issues/91).

A native Anthropic / OpenAI **chat** backend (separate from embeddings) is
tracked in [#90](https://github.com/mfaulk/graphrag-rs/issues/90).

## What zero-LLM mode actually delivers

A heads-up so the demo output isn't surprising: with `--mode hash`, entity
extraction is regex + capitalization heuristics and answer "synthesis" is just
top-3 retrieved chunks concatenated. You'll see entities like `PERSON_repthanks`
extracted from `**Sales Rep**: Thanks` — that's the algorithmic pipeline working
as designed, not a bug. Switch to `--mode ollama` for a coherent experience.
