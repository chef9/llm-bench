# Task: Upgrade llm-bench to v2 — Quality Measures + Qwen3 Fixes

## Context

This upgrades the existing `~/Developer/llm-bench/` tool. Do not create a new project.
Do not touch `~/Developer/nemo/`.

Three problems to fix from the v1 benchmark run:
1. **Qwen3 thinking mode** causes massive TTFT on mistral.rs (5–17s) and leaks `<think>` tags into output on MLX, breaking JSON. Fix: disable thinking mode in the API request for all OpenAI-compatible backends.
2. **`<think>` tag stripping** missing from the JSON cleaner. Even with thinking disabled, defensive stripping is needed.
3. **No quality measurement** beyond binary JSON validity. Add schema conformance checking, instruction-following heuristics, and cross-run consistency scoring.

Additionally: align all backends to **Qwen3 14B** so comparisons are fair. The v1 run mixed qwen2.5 (Ollama) with qwen3 (mistral.rs, MLX).

---

## Changes required

### 1. Cargo.toml — no changes needed

---

### 2. Default model env vars — update in `build_backends()`

Change the Ollama default model from `qwen2.5:14b` to `qwen3:14b`:

```rust
let model = std::env::var("OLLAMA_MODEL")
    .unwrap_or_else(|_| "qwen3:14b".to_string());
```

All four backends now default to Qwen3 14B. Users must `ollama pull qwen3:14b` before running.

---

### 3. Disable thinking mode in OpenAI-compatible requests

In `complete_openai()`, add `"thinking": false` to the request body. Some backends use `enable_thinking` instead — send both for maximum compatibility:

```rust
let body = json!({
    "model": model,
    "stream": true,
    "messages": [
        {"role": "system", "content": prompt.system},
        {"role": "user",   "content": prompt.user}
    ],
    "max_tokens": 512,
    "temperature": 0.0,
    "thinking": false,          // mistral.rs / some OpenAI-compat servers
    "enable_thinking": false    // MLX-LM and others
});
```

Claude does not have thinking mode in the same sense — no change needed to `complete_claude()`.

---

### 4. Fix the output cleaner — add `<think>` tag stripping

Replace the existing `check_json()` function with a two-part approach: a separate `clean_output()` function that sanitises raw model output, and an updated `check_json()` that calls it.

Add this function:

```rust
/// Strip thinking tags and markdown fences from raw model output.
/// Returns cleaned string suitable for JSON parsing or display.
fn clean_output(raw: &str) -> String {
    let s = raw.trim();

    // Strip <think>...</think> blocks (Qwen3 thinking mode leakage)
    // Handle both single-line and multi-line think blocks
    let s = if let (Some(start), Some(end)) = (s.find("<think>"), s.find("</think>")) {
        if end > start {
            let before = &s[..start];
            let after = &s[end + "</think>".len()..];
            format!("{}{}", before, after)
        } else {
            s.to_string()
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
```

Update `check_json()` to use `clean_output()`:

```rust
fn check_json(prompt_id: &str, raw: &str) -> Option<bool> {
    let json_prompts = ["structured_json", "label_generation", "graph_write"];
    if !json_prompts.contains(&prompt_id) {
        return None;
    }
    let cleaned = clean_output(raw);
    Some(serde_json::from_str::<Value>(&cleaned).is_ok())
}
```

Also store the cleaned output in `BenchResult` alongside `raw_output` — add a `cleaned_output: String` field to the struct and populate it in `run_once()`.

---

### 5. Add quality scoring

Add a `QualityScore` struct and a `score_quality()` function. Call it from `run_once()` after the completion returns.

```rust
#[derive(Serialize, Clone, Default)]
struct QualityScore {
    /// JSON parses successfully (structured prompts only)
    valid_json: Option<bool>,
    /// JSON keys and value types match expected schema (structured prompts only)
    schema_ok: Option<bool>,
    /// Output obeys the "no preamble" instruction (all prompts)
    /// Heuristic: output does not start with common preamble phrases
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
```

Implement `score_quality()`:

```rust
fn score_quality(prompt_id: &str, raw: &str) -> QualityScore {
    let cleaned = clean_output(raw);
    let mut score = QualityScore::default();
    let mut checks_passed = 0u32;
    let mut checks_total = 0u32;

    // ── Preamble check (all prompts) ────────────────────────────────────────
    // Models often add "Sure!", "Of course!", "Great question!" etc.
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
            // valid_json
            let valid = serde_json::from_str::<Value>(&cleaned).is_ok();
            score.valid_json = Some(valid);
            checks_total += 1;
            if valid { checks_passed += 1; }

            // schema: expect {"intent": string, "entities": [...], "proclet_hint": string}
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

            // schema: expect {"label": string (2-4 words), "description": string}
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

            // schema: expect {"operation": string, "node_type": string,
            //                 "label": string, "parent_hint": string}
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
            // Should contain exactly one question mark
            let q_count = cleaned.chars().filter(|&c| c == '?').count();
            let single_q = q_count == 1;
            score.single_question = Some(single_q);
            checks_total += 1;
            if single_q { checks_passed += 1; }

            // Should be concise — under 120 chars
            let concise = cleaned.len() <= 120;
            score.concise = Some(concise);
            checks_total += 1;
            if concise { checks_passed += 1; }
        }

        "deadlock_reasoning" => {
            // Should start with a digit (the step number)
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
```

---

### 6. Update `BenchResult` struct

Add `cleaned_output` and `quality` fields:

```rust
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
```

Update `run_once()` to populate both `cleaned_output` and `quality`:

```rust
// After getting raw_output from complete_*():
let cleaned_output = clean_output(&raw_output);
let quality = score_quality(&prompt.id, &raw_output);
let valid_json = quality.valid_json; // keep top-level field in sync
```

---

### 7. Update the summary table to show quality

Replace `print_summary()` with a version that shows the composite quality score alongside the performance metrics:

```rust
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
```

---

### 8. Update the progress line in `main()`

Show quality score inline as runs complete:

```rust
// Replace the existing "ok" branch print with:
let avg_q = ok.iter().map(|r| r.quality.composite).sum::<f64>() / ok.len() as f64;
let json_status = ok.first()
    .and_then(|r| r.valid_json)
    .map(|v| if v { " JSON✓" } else { " JSON✗" })
    .unwrap_or("");
println!("{med_total}ms  {med_tps:.1} tok/s  Q:{:.0}%{json_status}",
    avg_q * 100.0);
```

---

### 9. Update the README / usage notes

Add to the env var table:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:14b` | Changed from qwen2.5:14b |

Add a note before the env var table:

```
Before running, ensure Qwen3 14B is pulled in Ollama:
  ollama pull qwen3:14b

For mistral.rs, thinking mode is disabled automatically via the API request.
For MLX, thinking mode is disabled automatically. If <think> tags still appear
in output they will be stripped before JSON parsing and quality scoring.
```

---

### 10. Verify the build and run a quick smoke test

```bash
cd ~/Developer/llm-bench
cargo build

# Smoke test against Claude only (fastest to verify)
./target/debug/llm-bench --backends claude --runs 1
```

Expected: all five prompts complete, quality scores appear, composite % shown in summary. No `<think>` tags in `cleaned_output`.

---

## Notes for Claude Code

- All changes are to `src/main.rs` only, plus the `OLLAMA_MODEL` default. No new files needed.
- The `clean_output()` function must handle the case where `<think>` appears without a closing `</think>` — in that case strip from `<think>` to end of string.
- The `QualityScore` struct derives `Default` so zero-initialisation works for the error case in `run_once()`.
- Do not remove `valid_json` from the top-level `BenchResult` — it is kept for backward compatibility with any tooling reading the JSONL output. Keep it in sync with `quality.valid_json`.
- The composite score is a simple fraction of passed/total checks — do not weight checks differently. Weighting can be added later once we have more runs to calibrate against.
- `score_quality()` should never panic. All indexing is bounds-checked. Unknown `prompt_id` values return a default score with `composite: 0.0`.
