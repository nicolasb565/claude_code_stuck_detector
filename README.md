# Claude Code: Context Rewind & Auto-Compact

A patching system for [Claude Code](https://github.com/anthropics/claude-code) that adds two context management mechanisms:

1. **Auto-compact** (invisible) — Bash tool outputs are ephemeral by default: automatically compacted after the model processes them, freeing context window space. The model can preserve specific outputs by setting `ephemeral: false`.
2. **Rewind** (model-initiated tool) — lets the model abandon a failed approach, prune the conversation history, and inject a summary of what was tried.

Everything runs on the existing Max subscription. No API costs.

## How it works

Claude Code ships as a minified JS binary. This project applies **text patches** to that binary, inserting new code at specific anchor points. The patcher finds unique strings in the minified source and splices in the modifications.

### Patches applied (6 total)

1. **Ephemeral parameter** — adds `ephemeral` (default: true) to Bash tool's input schema
2. **Preservation tracking** — tracks Bash calls where the model explicitly sets `ephemeral: false`
3. **Auto-compact** — after each turn, compacts Bash outputs the model has already processed (unless preserved)
4. **Rewind tool** — registers a `Rewind(turns_back, summary)` tool alongside Read/Write/Edit/etc.
5. **Rewind handler** — in the main loop, detects pending Rewind requests and truncates the message array
6. **System prompt** — tells the model about ephemeral defaults and the Rewind tool

## Setup

```bash
git clone https://github.com/nicolasb565/claude_rewind_and_prune.git
cd claude_rewind_and_prune/claude-code-rewind

# Supply the Claude Code CLI (not redistributable)
mkdir -p vendor
npm pack @anthropic-ai/claude-code
tar xzf anthropic-ai-claude-code-*.tgz
mv package/cli.js vendor/cli-original.js
mv package/vendor vendor/bin-vendor
rm -rf package anthropic-ai-claude-code-*.tgz

# Build the patched CLI
node src/patch.mjs

# Run it
./bin/claude "your prompt here"
```

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `CLAUDE_REWIND_MODE` | `full` | `off` / `compact_only` / `full` |
| `REWIND_KEEP_FIRST` | `30` | Lines to keep from start of compacted output |
| `REWIND_KEEP_LAST` | `10` | Lines to keep from end of compacted output |
| `REWIND_MIN_LINES` | `50` | Minimum output length to trigger compaction |

### Three conditions (for benchmarking)

```bash
# Stock Claude Code (no patches active)
CLAUDE_REWIND_MODE=off ./bin/claude

# Auto-compact only (no Rewind tool)
CLAUDE_REWIND_MODE=compact_only ./bin/claude

# Auto-compact + Rewind tool
CLAUDE_REWIND_MODE=full ./bin/claude
```

## How auto-compact works

Bash output is **ephemeral by default**. After the model processes a Bash result (1+ assistant turns later), the orchestrator truncates it to first 30 + last 10 lines.

The model can override this per-call:
```
Bash({ command: "npm test", ephemeral: false })
```

Other tools (Read, Edit, Write, Grep, Glob) are never compacted — the model refers back to these while making iterative edits.

## How Rewind works

The model calls `Rewind(turns_back=N, summary="...")` when stuck:
1. Validates: summary >= 100 chars, turns_back >= 1, max 5 rewinds/session
2. Truncates conversation history at the Nth assistant message from the end
3. Injects a user message with the summary
4. Next turn continues with the pruned history

Rewind does **not** undo file changes — the summary should mention what was modified.

## Benchmark findings

Tested on a rate limiter debugging task (3 bugs, 14 tests) and an async refactoring task (13 tests).

### Evolution of auto-compact

| Version | Approach | Result |
|---|---|---|
| v1: truncate everything | All tool outputs after 2 turns | **Caused 8 re-reads**, 86% slower |
| v2: Bash-only | Only truncate Bash, never Read/Edit | 0 re-reads, modest savings |
| v4: ephemeral default-true | Bash ephemeral by default, model can preserve | 0 re-reads, clean savings |

Key insight: **truncating Read/Edit outputs causes the model to re-read files**, wasting more tokens than it saves. Only Bash outputs (test runs, builds, installs) are truly consume-once.

### GCC compiler bug (PR 123310)

The real stress test: a 1-character bug in GCC's value numbering pass (`-1U` vs `-1` in `tree-ssa-sccvn.cc`). Each run involves reading thousands of lines of compiler source, running tree dumps, adding debug prints, rebuilding GCC, and iterating. 54-197 tool calls per run.

| Trial | Stock | Full (v4) | Winner |
|---|---|---|---|
| T2 (parallel) | 1537s, 156 tools, 17.5M tokens | 1169s, 111 tools, 14.3M tokens | **Full by 24%** |
| T3 (parallel) | 867s, 54 tools, 5.7M tokens | 1731s, 142 tools, 21.5M tokens | Stock |
| v5 (+CLAUDE.md) | — | 1303s, 142 tools, 21.5M tokens | Same as T3 |

**Result: variance dominates signal.** Stock ranged 867-1537s, Full ranged 1169-1731s. The reasoning path (determined by non-deterministic token sampling) matters more than context management at this scale.

Auto-compact fires correctly (8-21 compaction events, 5-10K tokens saved per run). On trial 2, full mode used 18% fewer input tokens and 22% fewer output tokens. But trial 3 went the other way.

### Model doesn't use the tools

Across all trials:
- **Rewind: 0 calls.** The model never recognizes when it's going in circles.
- **ephemeral: false: 0 calls.** The model never explicitly preserves Bash output.
- **CLAUDE.md instructions: ignored.** v5 added detailed instructions in CLAUDE.md with examples of when to use Rewind and ephemeral. The model read them but behavior was identical to runs without CLAUDE.md (same tool counts, same token usage).

The model's agent-mode behavior (tool selection, investigation strategy) is entirely learned from training. System prompts, tool descriptions, and CLAUDE.md instructions do not change these habits. **New tool behaviors require fine-tuning, not prompting.**

### The real opportunity

These tools would have dramatically more impact on **smaller models** (7-35B) that have:
- Smaller context windows where bloat hits harder
- Weaker attention over long contexts where pruning removes distractors
- More frequent wrong turns where Rewind would actually trigger

Fine-tuning via LoRA on an open source model (e.g., Qwen 3.5 Coder 35B) with synthetic context management examples is the natural next step. Training data can be generated from Opus transcripts by retroactively labeling ephemeral outputs and backtracking sequences.

Related prior work: [MemGPT](https://arxiv.org/abs/2310.08560) (virtual memory for LLMs), [LATS](https://arxiv.org/abs/2310.04406) (tree search with backtracking), [Reflexion](https://arxiv.org/abs/2303.11366) (self-reflection). None combine model-driven context management + coding agent + fine-tuning.

## Telemetry

Events logged to `~/.claude-rewind-logs/events-YYYY-MM-DD.jsonl`:
- `session_start/end` — config, duration
- `compact` — tool name, lines saved, tokens saved estimate
- `rewind` / `rewind_applied` — turns pruned, summary length

## Updating

When Claude Code releases a new version:
```bash
npm pack @anthropic-ai/claude-code
# extract new cli.js to vendor/cli-original.js
node src/patch.mjs --check  # verify anchors still exist
node src/patch.mjs           # rebuild
```

## License

MIT for all code in this repo. Claude Code itself is under Anthropic's license — you must supply your own copy.
