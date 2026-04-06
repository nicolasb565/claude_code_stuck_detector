# Context Management for AI Coding Agents

Research into reducing context window waste and detecting circular reasoning in AI coding agents, tested on Claude Code with a real GCC compiler bug.

## The Problem

AI coding agents accumulate all tool output in their context window forever. After 30 minutes of debugging, half the context is stale test output, old tree dumps, and failed approach artifacts. The model has no mechanism to:
1. Discard tool outputs it has already processed
2. Recognize when it's going in circles
3. Backtrack from a failed approach

## HTTP Proxy (`stuck-detector-proxy/`)

A local proxy between Claude Code and the Anthropic API. Intercepts requests to compact tool outputs and detect stuck reasoning. **No patches, no plugins, works with vanilla Claude Code.**

```
Claude Code (unmodified)
    │
    │  ANTHROPIC_BASE_URL=http://localhost:8080
    │
    ▼
Proxy (localhost:8080)
    ├── Compact old Bash tool results in message array
    ├── Detect circular thinking (trained LogReg classifier, pure JS)
    ├── Inject corrective nudge when stuck detected
    ├── Retry with exponential backoff on 429/529
    ├── Concurrency limiter (semaphore, default 5 in-flight)
    └── Forward to api.anthropic.com with auth headers
```

### Usage

```bash
cd stuck-detector-proxy
node proxy.mjs &

# Run vanilla Claude Code through the proxy
ANTHROPIC_BASE_URL=http://localhost:8080 claude "your prompt"

# A/B testing is trivial — without proxy:
claude "your prompt"

# Monitor concurrency under load:
curl http://localhost:8080/stats
```

### Configuration

| Variable | Default | Description |
|---|---|---|
| `PROXY_PORT` | `8080` | Listen port |
| `PROXY_UPSTREAM` | `https://api.anthropic.com` | Upstream API |
| `PROXY_MAX_CONCURRENT` | `5` | Max in-flight upstream requests |
| `PROXY_MAX_RETRIES` | `8` | Max retries on 429/529 |
| `PROXY_BASE_DELAY_MS` | `1000` | Initial backoff delay |
| `PROXY_MAX_DELAY_MS` | `60000` | Max backoff delay |
| `COMPACT_ENABLED` | `1` | Auto-compact Bash outputs |
| `COMPACT_STALE_TURNS` | `2` | Turns before compaction |
| `COMPACT_KEEP_FIRST` | `30` | Lines kept from start |
| `COMPACT_KEEP_LAST` | `10` | Lines kept from end |
| `COMPACT_MIN_LINES` | `50` | Minimum lines to trigger |
| `STUCK_ENABLED` | `1` | Stuck detection |
| `STUCK_THRESHOLD` | `0.85` | Classifier confidence threshold |
| `STUCK_COOLDOWN` | `5` | Turns between nudges |
| `STUCK_CROSS_WINDOW_THRESHOLD` | `0.5` | Cross-window similarity threshold |

### Stuck Classifier

The stuck detector uses a logistic regression classifier trained on 1,878 labeled thinking-block windows (9 features, `class_weight=balanced`). Inference runs in pure JS — no Python dependency at runtime.

Features: `self_sim`, `max_substr_repeat`, `circle_kw`, `false_starts`, `avg_sent_len`, `sent_len_std`, `vocab_diversity`, `code_ratio`, `question_marks`.

A cross-window similarity check (comparing current thinking to the last 3 turns) suppresses false positives: high classifier score + low cross-window similarity means the model is exploring a wrong hypothesis but making progress, so no nudge is injected.

## Benchmark: GCC Compiler Bug (PR 123310)

Tested on [GCC PR 123310](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=123310) — a wrong-code bug in the value numbering pass (`tree-ssa-sccvn.cc`). The fix is a 1-character change: `-1U` → `-1` in an offset comparison.

| Run | Duration | Compactions | Stuck nudges | Correct fix? |
|---|---|---|---|---|
| Proxy | 1636s | 7 | 3 (turns 72, 86, 117) | Yes |

## Key Findings

1. **Only compact Bash outputs.** Truncating Read/Edit/Write outputs causes the model to re-read files, costing more than it saves.

2. **Models don't use novel tools without training.** `ephemeral` parameter, `Rewind` tool, CLAUDE.md instructions — the model ignores all of them. Agent-mode behavior is trained, not prompted.

3. **Proxy > patches > plugins.** Proxy gives full message control, survives updates, works with vanilla Claude Code, enables trivial A/B testing.

4. **Variance dominates.** Same task, same model: 219s to 1731s range across trials. Non-deterministic token sampling determines the reasoning path.

## Related Work

- [MemGPT](https://arxiv.org/abs/2310.08560) — Virtual memory paging for LLMs
- [LATS](https://arxiv.org/abs/2310.04406) — Tree search with backtracking for agents
- [Reflexion](https://arxiv.org/abs/2303.11366) — Self-reflection for LLM agents
- [Meta-Harness](https://arxiv.org/abs/2603.28052) — End-to-end harness optimization (raw traces beat summaries)
- [context-mode](https://github.com/mksglu/context-mode) — MCP-based context savings plugin for Claude Code

## Next Steps

1. Add cross-window similarity as a training feature in the classifier
2. Collect more stuck samples to improve precision
3. LoRA fine-tune an open source model (Qwen 3.5 Coder) on context management behaviors
4. Benchmark on SWE-bench with the proxy

## License

MIT for all code in this repo. Claude Code is under Anthropic's license — the proxy does not modify or redistribute it.
