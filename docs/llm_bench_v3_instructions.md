# Task: llm-bench fixes + decisions log update

Two independent tasks. Do them in order.

---

## Task A: Fix llm-bench

Working directory: `~/Developer/llm-bench/`
All changes to `src/main.rs` only.

### A1. Fix Ollama model default

In `build_backends()`, the Ollama default model must use the explicitly quantized variant:

```rust
let model = std::env::var("OLLAMA_MODEL")
    .unwrap_or_else(|_| "qwen3:14b-instruct-q4_K_M".to_string());
```

Rationale: `qwen3:14b` auto-selects a high-precision variant on 24GB RAM, yielding ~2 tok/s. The Q4_K_M variant is expected to yield ~30 tok/s.

### A2. Tighten system prompts for structured JSON prompts

Claude's 93% quality score (vs 100% for mistral.rs) is caused by the preamble heuristic firing. Claude adds a sentence of context before JSON despite being told not to.

In `prompts/suite.json`, strengthen the system prompt for `structured_json`, `label_generation`, and `graph_write` to be more emphatic:

Replace every `"system"` field on JSON-output prompts with this pattern — adapt the wording to match each prompt's task, but always include both sentences:

```
"You are a process modeling assistant. Output ONLY raw JSON — no explanation, no preamble, no markdown fences, no commentary before or after the JSON object."
```

Specifically:

- `structured_json`: `"You are a process modeling assistant. Output ONLY raw JSON — no explanation, no preamble, no markdown fences, no commentary before or after the JSON object."`
- `label_generation`: `"You are a process modeling assistant. Output ONLY raw JSON — no explanation, no preamble, no markdown fences, no commentary before or after the JSON object."`
- `graph_write`: `"You are a process modeling assistant. Output ONLY raw JSON — no explanation, no preamble, no markdown fences, no commentary before or after the JSON object."`
- `clarifying_dialogue`: `"You are a process modeling assistant. Output ONLY the question text — no preamble, no explanation, no greeting, nothing before or after the question itself."`
- `deadlock_reasoning`: `"You are a process modeling assistant. Output ONLY a step number followed by a colon and one sentence. Nothing before the step number, nothing after the sentence."`

### A3. Handle `<think>` blocks that are never closed

The current `clean_output()` function only strips complete `<think>...</think>` pairs. Some models emit a `<think>` block without a closing tag when thinking mode is partially active. Add a fallback:

```rust
fn clean_output(raw: &str) -> String {
    let s = raw.trim();

    // Strip complete <think>...</think> blocks
    let s = if let (Some(start), Some(end)) = (s.find("<think>"), s.find("</think>")) {
        if end > start {
            let before = &s[..start];
            let after = &s[end + "</think>".len()..];
            format!("{}{}", before, after)
        } else {
            s.to_string()
        }
    } else if let Some(start) = s.find("<think>") {
        // Unclosed <think> — strip from tag to end of string
        s[..start].to_string()
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

### A4. Verify the build

```bash
cd ~/Developer/llm-bench
cargo build
```

Must compile cleanly with no warnings about unused variables.

---

## Task B: Update decisions log in NeMo monorepo

Working directory: `~/Developer/nemo/`
File: `docs/architecture/decisions_log.md`

Place the file from `docs/_incoming/decisions_log.md` verbatim, replacing the existing `docs/architecture/decisions_log.md`.

The incoming file adds three new decision entries at the end of the log (before the closing `---` line):
- mistral.rs with Qwen3-14B-Q4K as LocalUnikernelBackend (with benchmark table)
- No custom LLM for NeMo
- SemanticBackend abstraction with three-tier fallback

Do not modify any other file in `docs/architecture/`.

---

## Task C: Commit and push

### C1. Commit llm-bench changes

```bash
cd ~/Developer/llm-bench
git add -A
git commit -m "fix: Ollama Q4K_M default, tighten system prompts, fix unclosed think tag stripping"
git push
```

If the llm-bench repo does not yet have a remote, initialise it and push:

```bash
cd ~/Developer/llm-bench
git init  # only if not already a git repo
git add -A
git commit -m "fix: Ollama Q4K_M default, tighten system prompts, fix unclosed think tag stripping"
```

Then inform the user that a remote needs to be added manually (`git remote add origin <url> && git push -u origin main`).

### C2. Commit NeMo decisions log

```bash
cd ~/Developer/nemo
git add docs/architecture/decisions_log.md
git commit -m "docs: add LLM backend benchmark results and SemanticBackend decisions"
git push
```

---

## Notes for Claude Code

- Do not modify any file in `~/Developer/nemo/` other than `docs/architecture/decisions_log.md`
- Do not modify `Cargo.toml` in llm-bench
- The `prompts/suite.json` changes are text edits only — do not change prompt IDs, add prompts, or remove prompts
- If `cargo build` produces errors unrelated to the changes above, report them without attempting to fix them
- Run `cargo build` after all `src/main.rs` changes before committing
