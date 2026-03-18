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
#[allow(dead_code)]
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
        let model = std::env::var("OLLAMA_MODEL")
            .unwrap_or_else(|_| "qwen3:14b-q4_K_M".to_string());
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
    // Quality
    valid_json: Option<bool>,       // kept for backward compat
    quality: QualityScore,
    error: Option<String>,
    raw_output: String,
    cleaned_output: String,
}

// ── Completion ────────────────────────────────────────────────────────────────

async fn complete_openai(
    client: &Client,
    kind: &str,
    base_url: &str,
    model: &str,
    prompt: &Prompt,
) -> Result<(u64, u64, u32, String)> {
    let url = format!("{}/v1/chat/completions", base_url);
    let user_content = if kind == "ollama" {
        format!("{} /no-think", prompt.user)
    } else {
        prompt.user.clone()
    };
    let body = json!({
        "model": model,
        "stream": true,
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user",   "content": user_content}
        ],
        "max_tokens": 512,
        "temperature": 0.0,
        "thinking": false,          // mistral.rs / some OpenAI-compat servers
        "enable_thinking": false    // MLX-LM and others
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
                if let Some(content) = val
                    .pointer("/choices/0/delta/content")
                    .and_then(|v| v.as_str())
                {
                    if ttft_ms.is_none() {
                        ttft_ms = Some(start.elapsed().as_millis() as u64);
                    }
                    full_text.push_str(content);
                }
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

// ── Output cleaning ──────────────────────────────────────────────────────────

/// Strip thinking tags and markdown fences from raw model output.
/// Returns cleaned string suitable for JSON parsing or display.
fn clean_output(raw: &str) -> String {
    let s = raw.trim();

    // Strip <think>...</think> blocks (Qwen3 thinking mode leakage)
    // Handle both single-line and multi-line think blocks
    let s = if let Some(start) = s.find("<think>") {
        if let Some(end) = s.find("</think>") {
            if end > start {
                let before = &s[..start];
                let after = &s[end + "</think>".len()..];
                format!("{}{}", before, after)
            } else {
                s.to_string()
            }
        } else {
            // <think> without closing </think> — strip from <think> to end
            s[..start].to_string()
        }
    } else {
        s.to_string()
    };
    let s = s.trim();

    // Strip markdown fences
    let s = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s);
    let s = s.strip_suffix("```").unwrap_or(s);

    s.trim().to_string()
}

// ── Quality scoring ──────────────────────────────────────────────────────────

#[derive(Serialize, Clone, Default)]
struct QualityScore {
    /// JSON parses successfully (structured prompts only)
    valid_json: Option<bool>,
    /// JSON keys and value types match expected schema (structured prompts only)
    schema_ok: Option<bool>,
    /// Output obeys the "no preamble" instruction (all prompts)
    no_preamble: Option<bool>,
    /// For clarifying_dialogue: output contains exactly one question mark
    single_question: Option<bool>,
    /// For clarifying_dialogue: output is concise (under 120 chars)
    concise: Option<bool>,
    /// For deadlock_reasoning: output starts with a digit (step number)
    starts_with_digit: Option<bool>,
    /// Composite 0.0–1.0 score: fraction of applicable checks that passed
    composite: f64,
}

fn score_quality(prompt_id: &str, raw: &str) -> QualityScore {
    let cleaned = clean_output(raw);
    let mut score = QualityScore::default();
    let mut checks_passed = 0u32;
    let mut checks_total = 0u32;

    // ── Preamble check (all prompts) ────────────────────────────────────────
    let preamble_phrases = [
        "sure", "of course", "great", "certainly", "absolutely",
        "happy to", "i'd be", "i will", "let me", "here is", "here's",
    ];
    let lower = cleaned.to_lowercase();
    let has_preamble = preamble_phrases
        .iter()
        .any(|p| lower.starts_with(p));
    score.no_preamble = Some(!has_preamble);
    checks_total += 1;
    if !has_preamble { checks_passed += 1; }

    match prompt_id {
        "structured_json" => {
            let valid = serde_json::from_str::<Value>(&cleaned).is_ok();
            score.valid_json = Some(valid);
            checks_total += 1;
            if valid { checks_passed += 1; }

            if let Ok(v) = serde_json::from_str::<Value>(&cleaned) {
                let ok = v.get("intent").and_then(|x| x.as_str()).is_some()
                    && v.get("entities").and_then(|x| x.as_array()).is_some()
                    && v.get("proclet_hint").and_then(|x| x.as_str()).is_some();
                score.schema_ok = Some(ok);
                checks_total += 1;
                if ok { checks_passed += 1; }
            } else {
                score.schema_ok = Some(false);
                checks_total += 1;
            }
        }

        "label_generation" => {
            let valid = serde_json::from_str::<Value>(&cleaned).is_ok();
            score.valid_json = Some(valid);
            checks_total += 1;
            if valid { checks_passed += 1; }

            if let Ok(v) = serde_json::from_str::<Value>(&cleaned) {
                let label_ok = v.get("label")
                    .and_then(|x| x.as_str())
                    .map(|s| {
                        let words = s.split_whitespace().count();
                        words >= 2 && words <= 4
                    })
                    .unwrap_or(false);
                let desc_ok = v.get("description")
                    .and_then(|x| x.as_str())
                    .map(|s| !s.is_empty())
                    .unwrap_or(false);
                score.schema_ok = Some(label_ok && desc_ok);
                checks_total += 1;
                if label_ok && desc_ok { checks_passed += 1; }
            } else {
                score.schema_ok = Some(false);
                checks_total += 1;
            }
        }

        "graph_write" => {
            let valid = serde_json::from_str::<Value>(&cleaned).is_ok();
            score.valid_json = Some(valid);
            checks_total += 1;
            if valid { checks_passed += 1; }

            if let Ok(v) = serde_json::from_str::<Value>(&cleaned) {
                let ok = v.get("operation").and_then(|x| x.as_str()).is_some()
                    && v.get("node_type").and_then(|x| x.as_str()).is_some()
                    && v.get("label").and_then(|x| x.as_str()).is_some()
                    && v.get("parent_hint").and_then(|x| x.as_str()).is_some();
                score.schema_ok = Some(ok);
                checks_total += 1;
                if ok { checks_passed += 1; }
            } else {
                score.schema_ok = Some(false);
                checks_total += 1;
            }
        }

        "clarifying_dialogue" => {
            let q_count = cleaned.chars().filter(|&c| c == '?').count();
            let single_q = q_count == 1;
            score.single_question = Some(single_q);
            checks_total += 1;
            if single_q { checks_passed += 1; }

            let concise = cleaned.len() <= 120;
            score.concise = Some(concise);
            checks_total += 1;
            if concise { checks_passed += 1; }
        }

        "deadlock_reasoning" => {
            let starts_digit = cleaned
                .chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false);
            score.starts_with_digit = Some(starts_digit);
            checks_total += 1;
            if starts_digit { checks_passed += 1; }
        }

        _ => {}
    }

    score.composite = if checks_total > 0 {
        checks_passed as f64 / checks_total as f64
    } else {
        0.0
    };

    score
}

// ── Backend isolation ─────────────────────────────────────────────────────

/// Free memory used by a backend after benchmarking it.
/// - Ollama: unload the model via API (keep_alive: 0)
/// - mistral.rs / MLX: kill the server process by port
/// - Claude: no-op (remote API)
async fn cleanup_backend(client: &Client, backend: &Backend) {
    match backend {
        Backend::OpenAiCompat { kind, base_url, model, .. } => {
            match kind.as_str() {
                "ollama" => {
                    // Unload model by requesting zero keep-alive
                    let url = format!("{}/api/generate", base_url);
                    let _ = client
                        .post(&url)
                        .json(&json!({"model": model, "keep_alive": 0}))
                        .send()
                        .await;
                    eprintln!("  → Unloaded {} from Ollama", model);
                }
                "mistralrs" | "mlx" => {
                    // Kill the server process to free GPU/unified memory
                    if let Some(port) = base_url.rsplit(':').next() {
                        let port = port.trim_end_matches('/');
                        let my_pid = std::process::id().to_string();
                        let output = std::process::Command::new("lsof")
                            .args(["-ti", &format!(":{}", port)])
                            .output();
                        if let Ok(output) = output {
                            let pids = String::from_utf8_lossy(&output.stdout);
                            for pid in pids.lines() {
                                let pid = pid.trim();
                                if !pid.is_empty() && pid != my_pid {
                                    let _ = std::process::Command::new("kill")
                                        .arg(pid)
                                        .output();
                                }
                            }
                            eprintln!("  → Stopped {} server on port {}", kind, port);
                        }
                    }
                }
                _ => {}
            }
        }
        Backend::Claude { .. } => {} // remote API, no cleanup needed
    }
    // Wait for memory to be freed
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
}

/// Start a backend server if it's not already running.
/// Returns true if the server is ready for benchmarking.
async fn ensure_backend_running(client: &Client, backend: &Backend) -> bool {
    match backend {
        Backend::OpenAiCompat { kind, base_url, model, .. } => {
            // Check if already running
            let check_url = format!("{}/v1/models", base_url);
            if client.get(&check_url).send().await.map(|r| r.status().is_success()).unwrap_or(false) {
                return true;
            }

            eprintln!("  → Starting {} server...", kind);
            match kind.as_str() {
                "ollama" => {
                    // Ollama daemon is typically already running; just need to
                    // trigger model load by sending a tiny request
                    let url = format!("{}/api/generate", base_url);
                    let _ = client
                        .post(&url)
                        .json(&json!({"model": model, "prompt": "", "keep_alive": "5m"}))
                        .send()
                        .await;
                    // Wait for model to load
                    for _ in 0..60 {
                        if client.get(&check_url).send().await
                            .map(|r| r.status().is_success()).unwrap_or(false) {
                            return true;
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    }
                }
                "mistralrs" => {
                    let port = base_url.rsplit(':').next().unwrap_or("1234").trim_end_matches('/');
                    let _ = std::process::Command::new("mistralrs")
                        .args(["serve", "-m", "Qwen/Qwen3-14B", "--isq", "Q4K", "--port", port])
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .spawn();
                    for _ in 0..120 {
                        if client.get(&check_url).send().await
                            .map(|r| r.status().is_success()).unwrap_or(false) {
                            return true;
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    }
                }
                "mlx" => {
                    let port = base_url.rsplit(':').next().unwrap_or("8080").trim_end_matches('/');
                    let mlx_model = std::env::var("MLX_MODEL")
                        .unwrap_or_else(|_| "default".to_string());
                    let _ = std::process::Command::new("mlx_lm.server")
                        .args(["--model", &mlx_model, "--port", port])
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .spawn();
                    for _ in 0..120 {
                        if client.get(&check_url).send().await
                            .map(|r| r.status().is_success()).unwrap_or(false) {
                            return true;
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    }
                }
                _ => {}
            }
            false
        }
        Backend::Claude { .. } => true, // always available
    }
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
        Backend::OpenAiCompat { kind, base_url, model } => {
            complete_openai(client, kind, base_url, model, prompt).await
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
            let cleaned_output = clean_output(&raw_output);
            let quality = score_quality(&prompt.id, &raw_output);
            let valid_json = quality.valid_json;
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
                quality,
                error: None,
                raw_output,
                cleaned_output,
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
            quality: QualityScore::default(),
            error: Some(e.to_string()),
            raw_output: String::new(),
            cleaned_output: String::new(),
        },
    }
}

// ── Summary table ─────────────────────────────────────────────────────────────

fn print_summary(results: &[BenchResult]) {
    use std::collections::HashMap;
    type Key = (String, String);
    let mut groups: HashMap<Key, Vec<&BenchResult>> = HashMap::new();
    for r in results {
        groups
            .entry((r.backend.clone(), r.prompt_id.clone()))
            .or_default()
            .push(r);
    }

    println!("\n{:-<110}", "");
    println!(
        "{:<35} {:<22} {:>9} {:>9} {:>8} {:>8} {:>8} {:>7}",
        "Backend", "Prompt", "TTFT(ms)", "Total(ms)", "Tok/s", "Tokens", "JSON✓", "Quality"
    );
    println!("{:-<110}", "");

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
                "{:<35} {:<22} {:>9} {:>9} {:>8} {:>8} {:>8} {:>7}",
                key.0, key.1, "ERROR", "-", "-", "-", "-", "-"
            );
            continue;
        }

        let median_ttft = median(&runs.iter().map(|r| r.ttft_ms).collect::<Vec<_>>());
        let median_total = median(&runs.iter().map(|r| r.total_ms).collect::<Vec<_>>());
        let median_tps = median_f64(&runs.iter().map(|r| r.tokens_per_sec).collect::<Vec<_>>());
        let median_tokens = median(&runs.iter().map(|r| r.output_tokens as u64).collect::<Vec<_>>());
        let avg_quality = runs.iter().map(|r| r.quality.composite).sum::<f64>() / runs.len() as f64;

        let json_ok = runs.iter().filter_map(|r| r.valid_json).all(|v| v);
        let has_json = runs.iter().any(|r| r.valid_json.is_some());
        let json_str = if has_json {
            if json_ok { "✓" } else { "✗" }
        } else {
            "n/a"
        };

        println!(
            "{:<35} {:<22} {:>9} {:>9} {:>8.1} {:>8} {:>8} {:>6.0}%",
            key.0, key.1, median_ttft, median_total, median_tps,
            median_tokens, json_str, avg_quality * 100.0
        );
    }
    println!("{:-<110}", "");

    // Per-backend aggregate quality
    println!("\nAggregate quality by backend (avg composite across all prompts and runs):");
    let mut backend_quality: HashMap<String, Vec<f64>> = HashMap::new();
    for r in results.iter().filter(|r| r.error.is_none()) {
        backend_quality
            .entry(r.backend.clone())
            .or_default()
            .push(r.quality.composite);
    }
    let mut bq: Vec<(String, f64)> = backend_quality
        .iter()
        .map(|(k, v)| (k.clone(), v.iter().sum::<f64>() / v.len() as f64))
        .collect();
    bq.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (backend, score) in &bq {
        println!("  {:<35} {:.0}%", backend, score * 100.0);
    }
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

    let prompts_raw = fs::read_to_string(&cli.prompts)
        .with_context(|| format!("Cannot read prompts file: {}", cli.prompts.display()))?;
    let prompts: Vec<Prompt> = serde_json::from_str(&prompts_raw)
        .context("Failed to parse prompts/suite.json")?;

    let backends = build_backends(&cli.backends);
    if backends.is_empty() {
        eprintln!("No backends available. Check that servers are running and env vars are set.");
        std::process::exit(1);
    }

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

    let multiple_local = backends.iter()
        .filter(|b| matches!(b, Backend::OpenAiCompat { .. }))
        .count() > 1;

    for (i, backend) in backends.iter().enumerate() {
        // When running multiple local backends, ensure isolation:
        // start this backend's server and verify it's ready
        if multiple_local {
            if !ensure_backend_running(&client, backend).await {
                eprintln!("  ⚠ Could not start {} — skipping", backend.name());
                continue;
            }
        }

        for prompt in &prompts {
            print!("  {} · {} ... ", backend.name(), prompt.id);
            std::io::stdout().flush()?;

            let mut run_results = vec![];
            for run in 1..=cli.runs {
                let result = run_once(&client, backend, prompt, run).await;
                run_results.push(result);
            }

            let errors = run_results.iter().filter(|r| r.error.is_some()).count();
            if errors == cli.runs as usize {
                println!("FAILED ({})", run_results[0].error.as_deref().unwrap_or("unknown"));
            } else {
                let ok: Vec<&BenchResult> = run_results.iter().filter(|r| r.error.is_none()).collect();
                let med_total = median(&ok.iter().map(|r| r.total_ms).collect::<Vec<_>>());
                let med_tps = median_f64(&ok.iter().map(|r| r.tokens_per_sec).collect::<Vec<_>>());
                let avg_q = ok.iter().map(|r| r.quality.composite).sum::<f64>() / ok.len() as f64;
                let json_status = ok.first()
                    .and_then(|r| r.valid_json)
                    .map(|v| if v { " JSON✓" } else { " JSON✗" })
                    .unwrap_or("");
                println!("{med_total}ms  {med_tps:.1} tok/s  Q:{:.0}%{json_status}",
                    avg_q * 100.0);
            }

            for r in &run_results {
                let line = serde_json::to_string(r)?;
                writeln!(out, "{}", line)?;
            }
            all_results.extend(run_results);
        }

        // After finishing a backend, free its memory before the next one
        if multiple_local && i < backends.len() - 1 {
            cleanup_backend(&client, backend).await;
        }
    }

    if cli.summary {
        print_summary(&all_results);
    }

    println!("\nDone. Results written to {}", output_path.display());
    Ok(())
}
