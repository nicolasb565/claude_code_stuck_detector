# Context Management for AI Coding Agents

Research into detecting when Claude Code goes in circles — a 2,605-parameter CNN trained on 85K windows of real Claude Code sessions, running entirely in JavaScript inside a local proxy. When the agent gets stuck, the proxy injects a corrective nudge. Language-agnostic, no Python runtime, no patches to Claude Code.

## The Problem

When AI coding agents get stuck on a hard task, they can burn a significant portion of their token budget going in circles:
- Re-running the same failing command with minor variations
- Cycling through the same files without making progress
- Generating "summary" text that rationalizes not having solved the problem

On our 13-task benchmark, the worst stuck cases burned 10× more time than a normal solve (e.g. rbtree went from 671s stuck to 45s with a nudge). Stuck episodes are not the common case — most sessions are productive — but when they happen they dominate the cost, and the agent has no built-in mechanism to recognize circular reasoning or backtrack. A lightweight external monitor can.

## HTTP Proxy (`proxy/`)

A local proxy between Claude Code and the Anthropic API. Intercepts requests, scores the recent tool-call history with a CNN, and injects a corrective nudge when stuck is detected. **No patches, no plugins, works with vanilla Claude Code.**

```
Claude Code (unmodified)
    │
    │  ANTHROPIC_BASE_URL=http://localhost:8080
    │
    ▼
Proxy (localhost:8080)
    ├── Parse tool calls from message history
    ├── Abstract into 10-step sliding windows
    ├── Score with CNN (pure JS, 56 KB weights)
    ├── Inject escalating nudge when stuck detected
    ├── Retry with exponential backoff on 429/529
    └── Forward to api.anthropic.com
```

### Usage

```bash
cd proxy
node proxy_cnn.mjs &

# Run vanilla Claude Code through the proxy
ANTHROPIC_BASE_URL=http://localhost:8080 claude "your prompt"

# A/B testing is trivial — just unset the env var
```

### Configuration

| Variable | Default | Description |
|---|---|---|
| `PROXY_PORT` | `8080` | Listen port |
| `PROXY_UPSTREAM` | `https://api.anthropic.com` | Upstream API |
| `STUCK_ENABLED` | `1` | Enable stuck detection |
| `STUCK_COOLDOWN` | `5` | Turns between nudges |
| `STUCK_RESET_THRESHOLD` | `threshold × 0.94` | Score must drop below this to reset nudge escalation |
| `COMPACT_ENABLED` | `0` | Auto-compact Bash outputs (optional) |

### Escalating Nudge

When the CNN fires, it injects a corrective message into the conversation. If the agent stays stuck across multiple cooldown windows, the nudge escalates:

| Level | Trigger | Behavior |
|---|---|---|
| 0 (soft) | First detection | Asks the agent to reflect — "are you going in circles?" |
| 1 (medium) | Still stuck after cooldown | Demands a 3-step explicit diagnosis before the next tool call |
| 2 (hard) | Still stuck after two cooldowns | STOP directive — no tool calls until root cause is stated |

`nudgeLevel` resets to 0 when the CNN score drops below `STUCK_RESET_THRESHOLD` (default: threshold × 0.94), indicating the agent has responded and moved on.

## CNN Stuck Detector

A 2,605-parameter CNN trained on 85,416 labeled windows from real Claude Code sessions. Uses cycle-detection features (CRC32-hashed `base_command:target_file` keys, Jaccard output similarity) that generalize across programming languages, agent scaffolds, and model families.

### Architecture

- **Input:** 10-step sliding windows of tool calls (stride 5)
- **Features per step:** 11 continuous (cycle detection, repetition, errors, output similarity) + 7-way tool embedding (4d)
- **Window-level features:** 5 aggregates (unique tool/file/cmd ratios, error rate, output similarity avg)
- **Model:** 2 parallel Conv1d branches (kernels 3+5, 16 filters each), max pool, MLP head
- **Output:** Sigmoid stuck probability
- **Size:** 56 KB JSON weights, runs in pure JS (no Python, no GPU)

### Results

**Test set** (29,283 windows from held-out trajectories):

| Metric | Value |
|---|---|
| Precision | 93.0% |
| Recall | 93.0% |
| F1 | 0.930 |
| Threshold | 0.96 |
| Weights | 56 KB |

**Benchmark on the LogReg-era task suite** (29 sessions, 6 stuck):

| Metric | Value |
|---|---|
| Benchmark fires | 2 (02_gcc off_2, 03_llvm off_1 — both genuine) |
| False positives | 0 on heldout tasks |

**Held-out tasks** (6 tasks, never seen in training): **all clean**.

### Data Generation Pipeline

```
Claude Code sessions (.jsonl)
    │
    │  python src/label_sessions.py <source> <sessions.jsonl>
    ▼
Abstract to numeric features (CRC32 semantic key, Jaccard output similarity)
    │
    ▼
Heuristic classifier (high-precision rules)
    │
    ├── PRODUCTIVE → data/sources/<source>_labeled.jsonl   (numeric only)
    └── CANDIDATE  → data/review/batches/<source>_batch_*.jsonl
                         (includes raw cmd/output snippets for review)
    │
    │  [Run Sonnet review agents on batch files]
    │  python src/review_sonnet.py <source>
    ▼
Sonnet decisions
    ├── PRODUCTIVE → appended to labeled file
    ├── STUCK      → appended to labeled file
    └── UNCLEAR    → data/review/escalated/<source>_batch_*.jsonl
    │
    │  [Run Opus review agents on escalated files]
    │  python src/review_opus.py <source>
    ▼
Opus decisions
    ├── PRODUCTIVE / STUCK → appended to labeled file
    └── UNCLEAR            → dropped (Opus is final arbiter)
    │
    ▼
data/sources/<source>_labeled.jsonl  →  gzip  →  merge_sources.py
```

**Privacy guarantee:** The final `.gz` contains only numeric features (tool indices, timing ratios, similarity scores). Raw commands, file paths, output text, and LLM review comments are stripped at the `_full_window` boundary in `label_sessions.py`. Sessions on proprietary codebases can be labeled and contributed without leaking sensitive content.

### Training Pipeline

```
85,416 labeled windows (770 STUCK, all Sonnet-confirmed)
    │ DataClaw oversampled 5x (physical duplication)
    ▼
Train CNN (class-balanced BCEWithLogitsLoss, early stopping on test F1)
    │  python src/train_cnn_oversample.py
    ▼
proxy/cnn_weights.json + proxy/cnn_config.json
```

### Key Innovations

1. **`cmd_semantic_key`** — Extracts `base_command:target_file` from bash commands. `cd build && ./gcc/xgcc -O2 test.c | tail` → `xgcc:test.c`. Makes command-repetition features work across projects without per-project retraining.

2. **Three-tier labeling with mandatory STUCK verification** — Heuristic STUCK labels are never used directly. All 2,606 heuristic-labeled STUCK windows were sent through Sonnet review: 770 confirmed STUCK, 1,815 flipped to PRODUCTIVE (false positives), 21 dropped. Final training labels are 100% LLM-verified for the STUCK class.

3. **Feature ablation to v4** — Left-one-out ablation on 24 feature variants with dual evaluation (labeled test F1 + false positive rate on 4 known-productive sessions). Dropped 2 features:
   - `thinking_length`: zero in 97.5% of training data (only DataClaw has thinking blocks) — pure noise
   - `output_diversity` (window-level): redundant with `output_similarity_avg` which already averages per-step Jaccard
   - Going from 12→11 continuous + 6→5 window features improved F1 from 0.910 → 0.930

4. **DataClaw oversample sweep** — Swept 0x/1x/5x/10x with v4 features. 5x is optimal: F1 0.930, 22 false fires on 4 productive sessions vs 29 at 10x. 10x was tuned for `thinking_length` which is now dropped; without it, the extra DataClaw weight is noise.

5. **Training improvement study (null result)** — Tested validation split, pos_weight from natural distribution, label smoothing (ε=0.1), and threshold tuning sequentially. All four made things worse. At 2,605 parameters and 98K training windows, the model is data-hungry: removing 20% for a val split costs more than unbiased model selection gains. Label smoothing at 0.1 is too aggressive for a 1.2% STUCK rate. The baseline config (full training set, pos_weight from oversampled counts, fixed threshold 0.96) wins on both axes.

6. **Confirmation rules tested** — 2-of-3, 2-consecutive, streak-based, EMA smoothing. None improved on direct thresholding (stuck patterns are short and bursty; multi-window rules mostly hurt recall). The proxy uses direct CNN output at threshold 0.96.

### Datasets

| Dataset | License | Sessions | Role |
|---|---|---|---|
| [nlile/misc-merged](https://huggingface.co/datasets/nlile/misc-merged) | Apache-2.0 | 16,841 | Primary source, no thinking blocks |
| [DataClaw](https://huggingface.co/datasets/DataClaw) (woctordho) | Apache-2.0 | 136 | Has thinking blocks, oversampled 5x |
| work_embedded_c | Internal | ~500 | Embedded C sessions, 1x (no oversample) |

### Feature Set (11 continuous + tool embed + 5 window-level)

| Feature | Level | Signal |
|---|---|---|
| `steps_since_same_cmd` | Per-step | Core — command repetition via semantic key |
| `cmd_count_in_window` | Per-step | Core — repetition count within window |
| `output_similarity` | Per-step | Core — Jaccard on output lines; same result = stuck |
| `is_error` | Per-step | Errors in loops = stuck, errors alone = debugging |
| `output_length` | Per-step | Log of output line count |
| `step_index_norm` | Per-step | Position in trajectory |
| `tool_count_in_window` | Per-step | Tool repetition frequency |
| `steps_since_same_tool` | Per-step | Tool type repetition |
| `steps_since_same_file` | Per-step | File access repetition |
| `file_count_in_window` | Per-step | File repetition count |
| `has_prior_output` | Per-step | Whether this command has been run before |
| `unique_tools_ratio` | Window | Tool diversity across the window |
| `unique_files_ratio` | Window | File diversity across the window |
| `unique_cmds_ratio` | Window | Command diversity across the window |
| `error_rate` | Window | Fraction of steps that produced errors |
| `output_similarity_avg` | Window | Mean per-step Jaccard across window |

Dropped features: `false_start`, `strategy_change`, `circular_lang`, `self_similarity` (regex patterns, near-dead in training data); `thinking_length` (zero in 97.5% of windows); `output_diversity` (window-level, redundant with `output_similarity_avg`).

**JS forward pass verified:** Pure JS inference matches Python with max diff 6.2e-9 across 100 test vectors. No Node dependencies beyond `node:zlib` for CRC32.

## Key Findings

1. **Stuck is detectable with a tiny model.** 2,605 parameters, trained on ~770 real stuck examples (Sonnet-verified), reaches 93% precision / 93% recall on held-out trajectories.

2. **The right signals are behavioral, not textual.** Command repetition and output similarity dominate. Thinking-block regex features (`false_start`, `circular_lang`) are either redundant or sparse. The model generalizes because it measures *repetition patterns*, not language or domain.

3. **Every STUCK label must be LLM-verified.** Heuristic STUCK rules have a ~70% false positive rate — 1,815 of 2,606 heuristic-labeled STUCK windows were actually productive exploration. Direct heuristic labeling is only reliable for PRODUCTIVE (high-diversity, no tight loops). All STUCK training labels go through Sonnet review.

4. **DataClaw oversampling: 5x, not 10x.** The 10x rate was chosen when `thinking_length` was a feature, giving DataClaw a structural advantage. Once `thinking_length` is dropped (it's zero in 97.5% of windows), 10x becomes harmful overfit. 5x retains the distribution benefit without the noise amplification.

5. **Training hyperparameters are saturated.** At this scale (2,605 params, 98K windows), validation splits, label smoothing, pos_weight correction, and threshold tuning all make things worse. The bottleneck is data quality and feature expressiveness, not optimization.

6. **Remaining weakness: productive edit→build→test cycles.** The persistent false positives are agents iterating on test/build failures — structurally identical to stuck loops at the feature level. Fixing this requires tracking output *change* between repeated commands, not just similarity.

7. **We need more Claude Code datasets with thinking blocks.** DataClaw (136 sessions) is currently the only source with extended thinking. 5x oversampling is a workaround — a larger labeled corpus with thinking blocks would move the model further than any architectural change.

## Related Work

### Stuck/loop detection in agents

- [SpecRA](https://openreview.net/forum?id=xVO4BqmzVD) (Oct 2025) — FFT autocorrelation on token sequences to detect periodicity. Signal-processing at the token level, no behavioral features.
- [Agentic Metacognition](https://arxiv.org/abs/2509.19783) (Xu, Sep 2025) — External metacognitive layer monitors a primary agent for repetitive actions. Closest architectural match.
- [strongdm/attractor](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md) — Open-source spec tracking tool-call signatures in a sliding window. Same concept as our system but no ML classifier.

### Context management

- [MemGPT](https://arxiv.org/abs/2310.08560) — Virtual memory paging for LLMs
- [LATS](https://arxiv.org/abs/2310.04406) — Tree search with backtracking for agents
- [context-mode](https://github.com/mksglu/context-mode) — MCP-based context savings plugin

### What's different about our approach

This work combines: proxy-based interception, tool-call behavioral features, a trained CNN, and corrective nudge injection — all running in pure JavaScript inside the proxy with no Python runtime. The cleanest comparison point is `strongdm/attractor` which uses similar behavioral signals but no ML.

Longer-term, this kind of monitoring belongs inside the model or API — similar to how speculative decoding uses a small draft model alongside the main model. A lightweight "reasoning monitor" model could run in parallel during inference, detecting stuck patterns at the token level before a full stuck episode forms.

## Next Steps

1. Collect/label more Claude Code sessions **with thinking blocks** — currently the biggest lever, 5x DataClaw oversampling is only a partial substitute
2. Add an "output change between repeated commands" feature to break the edit→build→test false positive pattern
3. Run a 5-run benchmark for statistical significance
4. Add timestamp-based heuristics in the proxy (fast retries boost stuck score, slow gaps dampen)
5. Explore a lightweight speculative-decoding-style parallel monitor inside inference

## License

MIT for all code in this repo. Claude Code is under Anthropic's license — the proxy does not modify or redistribute it.
