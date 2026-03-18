# Task: Create llm-bench — LLM Backend Benchmarking Tool

## Context

This is a standalone Rust CLI tool for benchmarking and comparing LLM backends side by side. It lives at `~/Developer/llm-bench/` — outside the NeMo monorepo. It has no dependency on any NeMo crate.

The tool hits four backends with the same prompt suite and records latency, throughput, and output quality metrics. It is used to evaluate which local LLM backend (Ollama, mistral.rs, MLX) is most suitable for NeMo's SemanticActor, and to compare local model quality against Claude.

---

## Step 1: Create the project

```bash
cd ~/Developer
cargo new llm-bench
cd llm-bench
```

---

## Step 2: Cargo.toml

Replace the generated `Cargo.toml` with:

```toml
[package]
name = "llm-bench"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "llm-bench"
path = "src/main.rs"

[dependencies]
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json", "stream"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
futures-util = "0.3"
chrono = { version = "0.4", features = ["serde"] }
anyhow = "1"
```

---

## Step 3: Prompt suite

Create `prompts/suite.json`:

```json
[
  {
    "id": "structured_json",
    "description": "Emit valid JSON matching a schema",
    "system": "You are a process modeling assistant. Respond ONLY with valid JSON, no prose, no markdown fences.",
    "user": "A user says: 'I need to track my team's tasks'. Respond ONLY with this exact JSON shape: {\"intent\": \"<string>\", \"entities\": [\"<string>\"], \"proclet_hint\": \"<string>\"}"
  },
  {
    "id": "label_generation",
    "description": "Generate a short display label from a formal subnet description",
    "system": "You are a process modeling assistant. Respond ONLY with valid JSON, no prose, no markdown fences.",
    "user": "Given this Petri net subnet description: 'single transition with one input place and one output place, inhibitor arc from a third place preventing firing when the third place is marked'. Respond ONLY with this exact JSON shape: {\"label\": \"<2-4 words>\", \"description\": \"<one sentence>\"}"
  },
  {
    "id": "clarifying_dialogue",
    "description": "Generate a single clarifying question from ambiguous input",
    "system": "You are a process modeling assistant. Ask exactly one clarifying question. Reply with only the question text, no preamble, no explanation.",
    "user": "A user wants to model: 'our approval process'."
  },
  {
    "id": "graph_write",
    "description": "Translate natural language to graph operation",
    "system": "You are a process modeling assistant. Respond ONLY with valid JSON, no prose, no markdown fences.",
    "user": "Convert this request to a graph operation: 'Add a task called Review Contract to the Legal proclet'. Respond ONLY with this exact JSON shape: {\"operation\": \"<string>\", \"node_type\": \"<string>\", \"label\": \"<string>\", \"parent_hint\": \"<string>\"}"
  },
  {
    "id": "deadlock_reasoning",
    "description": "Reason over a multi-step process to identify a concurrency hazard",
    "system": "You are a process modeling assistant. Respond with ONLY a step number (integer) followed by a colon and a single sentence explanation. No other text.",
    "user": "Here is a process: step 1: receive invoice, step 2: validate fields, step 3: check for duplicate, step 4: route to approver, step 5: approver reviews, step 6: approve or reject, step 7: if rejected notify sender, step 8: if approved post to ledger, step 9: schedule payment, step 10: confirm payment. Which single step is most likely to cause a deadlock in concurrent execution?"
  }
]
```

---

## Step 4: src/main.rs

Create `src/main.rs` with the following complete implementation:

```rust
use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Parser, ValueEnum};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "llm-bench", about = "Benchmark LLM backends side by side")]
struct Cli {
    /// Backends to run (default: all configured)
    #[arg(short, long, value_enum, num_args = 1..)]
    backends: Option<Vec<BackendKind>>,

    /// Path to prompt suite JSON (default: prompts/suite.json)
    #[arg(short, long, default_value = "prompts/suite.json")]
    prompts: PathBuf,

    /// Number of runs per (backend, prompt) pair
    #[arg(short, long, default_value_t = 3)]
    runs: u32,

    /// Output JSONL file (default: results/<timestamp>.jsonl)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Print a summary table after all runs
    #[arg(short, long, default_value_t = true)]
    summary: bool,
}

#[derive(ValueEnum, Clone, Debug, PartialEq)]
enum BackendKind {
    Ollama,
    Mistralrs,
    Mlx,
    Claude,
}

// ── Prompt suite ─────────────────────────────────────────────────────────────

#[derive(Deserialize, Clone)]
struct Prompt {
    id: String,
    description: String,
    system: String,
    user: String,
}

// ── Backend config ────────────────────────────────────────────────────────────

#[derive(Clone)]
enum Backend {
    OpenAiCompat {
        kind: String,
        base_url: String,
        model: String,
    },
    Claude {
        api_key: String,
        model: String,
    },
}

impl Backend {
    fn name(&self) -> String {
        match self {
            Backend::OpenAiCompat { kind, model, .. } => format!("{}/{}", kind, model),
            Backend::Claude { model, .. } => format!("claude/{}", model),
        }
    }

    fn kind(&self) -> &str {
        match self {
            Backend::OpenAiCompat { kind, .. } => kind,
            Backend::Claude { .. } => "claude",
        }
    }
}

fn build_backends(requested: &Option<Vec<BackendKind>>) -> Vec<Backend> {
    let all = requested.is_none();
    let wants = |k: &BackendKind| -> bool {
        requested.as_ref().map(|v| v.contains(k)).unwrap_or(true)
    };

    let mut backends = vec![];

    if all || wants(&BackendKind::Ollama) {
        // Default model — override with OLLAMA_MODEL env var
        let model = std::env::var("OLLAMA_MODEL")
            .unwrap_or_else(|_| "qwen2.5:14b".to_string());
        backends.push(Backend::OpenAiCompat {
            kind: "ollama".to_string(),
            base_url: std::env::var("OLLAMA_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string()),
            model,
        });
    }

    if all || wants(&BackendKind::Mistralrs) {
        let model = std::env::var("MISTRALRS_MODEL")
            .unwrap_or_else(|_| "default".to_string());
        backends.push(Backend::OpenAiCompat {
            kind: "mistralrs".to_string(),
            base_url: std::env::var("MISTRALRS_URL")
                .unwrap_or_else(|_| "http://localhost:1234".to_string()),
            model,
        });
    }

    if all || wants(&BackendKind::Mlx) {
        let model = std::env::var("MLX_MODEL")
            .unwrap_or_else(|_| "default".to_string());
        backends.push(Backend::OpenAiCompat {
            kind: "mlx".to_string(),
            base_url: std::env::var("MLX_URL")
                .unwrap_or_else(|_| "http://localhost:8080".to_string()),
            model,
        });
    }

    if all || wants(&BackendKind::Claude) {
        if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
            let model = std::env::var("CLAUDE_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string());
            backends.push(Backend::Claude { api_key, model });
        } else {
            eprintln!("⚠ ANTHROPIC_API_KEY not set — skipping Claude backend");
        }
    }

    backends
}

// ── Result ───────────────────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
struct BenchResult {
    timestamp: String,
    backend: String,
    backend_kind: String,
    prompt_id: String,
    run: u32,
    ttft_ms: u64,
    total_ms: u64,
    output_tokens: u32,
    tokens_per_sec: f64,
    valid_json: Option<bool>,
    error: Option<String>,
    raw_output: String,
}

// ── Completion ────────────────────────────────────────────────────────────────

async fn complete_openai(
    client: &Client,
    base_url: &str,
    model: &str,
    prompt: &Prompt,
) -> Result<(u64, u64, u32, String)> {
    // Returns (ttft_ms, total_ms, output_tokens, raw_output)
    let url = format!("{}/v1/chat/completions", base_url);
    let body = json!({
        "model": model,
        "stream": true,
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user",   "content": prompt.user}
        ],
        "max_tokens": 512,
        "temperature": 0.0
    });

    let start = Instant::now();
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("Failed to connect to backend")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("HTTP {}: {}", status, text);
    }

    let mut stream = resp.bytes_stream();
    let mut ttft_ms: Option<u64> = None;
    let mut full_text = String::new();
    let mut output_tokens: u32 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Stream error")?;
        let text = String::from_utf8_lossy(&chunk);

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line == "data: [DONE]" {
                continue;
            }
            let line = line.strip_prefix("data: ").unwrap_or(line);
            if let Ok(val) = serde_json::from_str::<Value>(line) {
                // Extract content delta
                if let Some(content) = val
                    .pointer("/choices/0/delta/content")
                    .and_then(|v| v.as_str())
                {
                    if ttft_ms.is_none() {
                        ttft_ms = Some(start.elapsed().as_millis() as u64);
                    }
                    full_text.push_str(content);
                }
                // Extract usage if present (some backends include it on final chunk)
                if let Some(tokens) = val
                    .pointer("/usage/completion_tokens")
                    .and_then(|v| v.as_u64())
                {
                    output_tokens = tokens as u32;
                }
            }
        }
    }

    let total_ms = start.elapsed().as_millis() as u64;
    let ttft_ms = ttft_ms.unwrap_or(total_ms);

    // Estimate tokens if backend didn't report them (~4 chars/token heuristic)
    if output_tokens == 0 {
        output_tokens = (full_text.len() as u32).saturating_div(4).max(1);
    }

    Ok((ttft_ms, total_ms, output_tokens, full_text))
}

async fn complete_claude(
    client: &Client,
    api_key: &str,
    model: &str,
    prompt: &Prompt,
) -> Result<(u64, u64, u32, String)> {
    let body = json!({
        "model": model,
        "max_tokens": 512,
        "temperature": 0.0,
        "stream": true,
        "system": prompt.system,
        "messages": [
            {"role": "user", "content": prompt.user}
        ]
    });

    let start = Instant::now();
    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .context("Failed to connect to Anthropic API")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("HTTP {}: {}", status, text);
    }

    let mut stream = resp.bytes_stream();
    let mut ttft_ms: Option<u64> = None;
    let mut full_text = String::new();
    let mut output_tokens: u32 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Stream error")?;
        let text = String::from_utf8_lossy(&chunk);

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("event:") {
                continue;
            }
            let line = line.strip_prefix("data: ").unwrap_or(line);
            if let Ok(val) = serde_json::from_str::<Value>(line) {
                // content_block_delta carries the text
                if val.get("type").and_then(|v| v.as_str()) == Some("content_block_delta") {
                    if let Some(text) = val
                        .pointer("/delta/text")
                        .and_then(|v| v.as_str())
                    {
                        if ttft_ms.is_none() {
                            ttft_ms = Some(start.elapsed().as_millis() as u64);
                        }
                        full_text.push_str(text);
                    }
                }
                // message_delta carries usage
                if val.get("type").and_then(|v| v.as_str()) == Some("message_delta") {
                    if let Some(tokens) = val
                        .pointer("/usage/output_tokens")
                        .and_then(|v| v.as_u64())
                    {
                        output_tokens = tokens as u32;
                    }
                }
            }
        }
    }

    let total_ms = start.elapsed().as_millis() as u64;
    let ttft_ms = ttft_ms.unwrap_or(total_ms);

    if output_tokens == 0 {
        output_tokens = (full_text.len() as u32).saturating_div(4).max(1);
    }

    Ok((ttft_ms, total_ms, output_tokens, full_text))
}

// ── JSON validation ───────────────────────────────────────────────────────────

fn check_json(prompt_id: &str, raw: &str) -> Option<bool> {
    // Only structured prompts require JSON
    let json_prompts = ["structured_json", "label_generation", "graph_write"];
    if !json_prompts.contains(&prompt_id) {
        return None;
    }
    // Strip markdown fences if present
    let cleaned = raw.trim();
    let cleaned = cleaned
        .strip_prefix("```json")
        .or_else(|| cleaned.strip_prefix("```"))
        .unwrap_or(cleaned);
    let cleaned = cleaned.strip_suffix("```").unwrap_or(cleaned).trim();
    Some(serde_json::from_str::<Value>(cleaned).is_ok())
}

// ── Runner ────────────────────────────────────────────────────────────────────

async fn run_once(
    client: &Client,
    backend: &Backend,
    prompt: &Prompt,
    run: u32,
) -> BenchResult {
    let timestamp = Utc::now().to_rfc3339();

    let result = match backend {
        Backend::OpenAiCompat { base_url, model, .. } => {
            complete_openai(client, base_url, model, prompt).await
        }
        Backend::Claude { api_key, model } => {
            complete_claude(client, api_key, model, prompt).await
        }
    };

    match result {
        Ok((ttft_ms, total_ms, output_tokens, raw_output)) => {
            let gen_secs = (total_ms - ttft_ms) as f64 / 1000.0;
            let tokens_per_sec = if gen_secs > 0.0 {
                output_tokens as f64 / gen_secs
            } else {
                0.0
            };
            let valid_json = check_json(&prompt.id, &raw_output);
            BenchResult {
                timestamp,
                backend: backend.name(),
                backend_kind: backend.kind().to_string(),
                prompt_id: prompt.id.clone(),
                run,
                ttft_ms,
                total_ms,
                output_tokens,
                tokens_per_sec,
                valid_json,
                error: None,
                raw_output,
            }
        }
        Err(e) => BenchResult {
            timestamp,
            backend: backend.name(),
            backend_kind: backend.kind().to_string(),
            prompt_id: prompt.id.clone(),
            run,
            ttft_ms: 0,
            total_ms: 0,
            output_tokens: 0,
            tokens_per_sec: 0.0,
            valid_json: None,
            error: Some(e.to_string()),
            raw_output: String::new(),
        },
    }
}

// ── Summary table ─────────────────────────────────────────────────────────────

fn print_summary(results: &[BenchResult]) {
    // Group by (backend, prompt_id), compute medians across runs
    use std::collections::HashMap;
    type Key = (String, String);
    let mut groups: HashMap<Key, Vec<&BenchResult>> = HashMap::new();
    for r in results {
        groups
            .entry((r.backend.clone(), r.prompt_id.clone()))
            .or_default()
            .push(r);
    }

    println!("\n{:-<100}", "");
    println!(
        "{:<35} {:<22} {:>9} {:>9} {:>8} {:>8} {:>8}",
        "Backend", "Prompt", "TTFT(ms)", "Total(ms)", "Tok/s", "Tokens", "JSON✓"
    );
    println!("{:-<100}", "");

    let mut keys: Vec<Key> = groups.keys().cloned().collect();
    keys.sort();

    for key in &keys {
        let runs: Vec<&BenchResult> = groups[key]
            .iter()
            .filter(|r| r.error.is_none())
            .cloned()
            .collect();

        if runs.is_empty() {
            println!(
                "{:<35} {:<22} {:>9} {:>9} {:>8} {:>8} {:>8}",
                key.0, key.1, "ERROR", "-", "-", "-", "-"
            );
            continue;
        }

        let median_ttft = median(&runs.iter().map(|r| r.ttft_ms).collect::<Vec<_>>());
        let median_total = median(&runs.iter().map(|r| r.total_ms).collect::<Vec<_>>());
        let median_tps = median_f64(&runs.iter().map(|r| r.tokens_per_sec).collect::<Vec<_>>());
        let median_tokens = median(&runs.iter().map(|r| r.output_tokens as u64).collect::<Vec<_>>());
        let json_ok = runs.iter().filter_map(|r| r.valid_json).all(|v| v);
        let has_json = runs.iter().any(|r| r.valid_json.is_some());

        let json_str = if has_json {
            if json_ok { "✓" } else { "✗" }
        } else {
            "n/a"
        };

        println!(
            "{:<35} {:<22} {:>9} {:>9} {:>8.1} {:>8} {:>8}",
            key.0, key.1, median_ttft, median_total, median_tps, median_tokens, json_str
        );
    }
    println!("{:-<100}", "");
}

fn median(values: &[u64]) -> u64 {
    let mut v = values.to_vec();
    v.sort();
    v[v.len() / 2]
}

fn median_f64(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load prompt suite
    let prompts_raw = fs::read_to_string(&cli.prompts)
        .with_context(|| format!("Cannot read prompts file: {}", cli.prompts.display()))?;
    let prompts: Vec<Prompt> = serde_json::from_str(&prompts_raw)
        .context("Failed to parse prompts/suite.json")?;

    // Build backends
    let backends = build_backends(&cli.backends);
    if backends.is_empty() {
        eprintln!("No backends available. Check that servers are running and env vars are set.");
        std::process::exit(1);
    }

    // Output file
    let output_path = cli.output.unwrap_or_else(|| {
        let ts = Utc::now().format("%Y%m%dT%H%M%S");
        PathBuf::from(format!("results/{}.jsonl", ts))
    });
    fs::create_dir_all(output_path.parent().unwrap_or(&PathBuf::from(".")))?;
    let mut out = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&output_path)?;

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    println!("llm-bench · {} backends · {} prompts · {} runs each",
        backends.len(), prompts.len(), cli.runs);
    println!("Output: {}\n", output_path.display());

    let mut all_results: Vec<BenchResult> = vec![];

    for backend in &backends {
        for prompt in &prompts {
            print!("  {} · {} ... ", backend.name(), prompt.id);
            std::io::stdout().flush()?;

            let mut run_results = vec![];
            for run in 1..=cli.runs {
                let result = run_once(&client, backend, prompt, run).await;
                run_results.push(result);
            }

            // Print quick status
            let errors = run_results.iter().filter(|r| r.error.is_some()).count();
            if errors == cli.runs as usize {
                println!("FAILED ({})", run_results[0].error.as_deref().unwrap_or("unknown"));
            } else {
                let ok: Vec<&BenchResult> = run_results.iter().filter(|r| r.error.is_none()).collect();
                let med_total = median(&ok.iter().map(|r| r.total_ms).collect::<Vec<_>>());
                let med_tps = median_f64(&ok.iter().map(|r| r.tokens_per_sec).collect::<Vec<_>>());
                let json_status = ok.first()
                    .and_then(|r| r.valid_json)
                    .map(|v| if v { " JSON✓" } else { " JSON✗" })
                    .unwrap_or("");
                println!("{med_total}ms  {med_tps:.1} tok/s{json_status}");
            }

            // Write all runs to JSONL
            for r in &run_results {
                let line = serde_json::to_string(r)?;
                writeln!(out, "{}", line)?;
            }
            all_results.extend(run_results);
        }
    }

    if cli.summary {
        print_summary(&all_results);
    }

    println!("\nDone. Results written to {}", output_path.display());
    Ok(())
}
```

---

## Step 5: Verify it builds

```bash
cd ~/Developer/llm-bench
cargo build
```

Fix any dependency version conflicts if needed. The program will build even if no backends are running.

---

## Step 6: Usage

### Start backends before running

**Ollama** (pull model first if needed):
```bash
ollama pull qwen2.5:14b
ollama serve
```

**mistral.rs:**
```bash
mistralrs serve -m Qwen/Qwen3-14B --port 1234
```

**MLX:**
```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/Qwen3-14B-4bit --port 8080
```

### Run the benchmark

```bash
# All available backends (skips any that aren't running)
./target/debug/llm-bench

# Specific backends only
./target/debug/llm-bench --backends ollama claude

# More runs for better median accuracy
./target/debug/llm-bench --runs 5

# Custom model for Ollama
OLLAMA_MODEL=llama3.1:8b ./target/debug/llm-bench --backends ollama
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required for Claude) | Anthropic API key |
| `OLLAMA_MODEL` | `qwen2.5:14b` | Model name as known to Ollama |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `MISTRALRS_MODEL` | `default` | Model ID for mistral.rs |
| `MISTRALRS_URL` | `http://localhost:1234` | mistral.rs server URL |
| `MLX_MODEL` | `default` | Model ID for mlx-lm server |
| `MLX_URL` | `http://localhost:8080` | mlx-lm server URL |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model string |

---

## Notes for Claude Code

- Do not modify any files in `~/Developer/nemo/`
- The `results/` directory is created automatically on first run
- If `cargo build` fails due to dependency version conflicts, resolve them in `Cargo.toml` — do not change `src/main.rs` logic
- The token-per-second calculation uses `(total_ms - ttft_ms)` as the generation window, which is more accurate than using `total_ms` directly (excludes the time-to-first-token prefill phase)
- The JSON validator strips markdown fences before parsing — local models often wrap JSON in ```json blocks despite being told not to; this is intentional defensive behaviour
