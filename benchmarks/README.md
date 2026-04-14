# benchmarks/

Per-task Docker-isolated benchmark for the v5 stuck detector + nudge proxy.
See `PLAN.md` for the full design doc.

## Prerequisites

- Linux host (uses `--network host` so the container can reach the proxy)
- Docker (user in the `docker` group, no sudo required)
- `jq`, `git`, `node`, `npm` on the host
- A Claude subscription login (`claude` CLI logged in) **or** an Anthropic
  API key in `benchmarks/.env`
- ~20 GB free disk for fixtures + build artifacts
- Network access for the initial fixture clones

## One-time setup

```bash
# Build the runner image (debian:trixie + toolchains + claude-code 2.1.105)
docker build -t benchmark-runner:latest benchmarks/

# Clone every task's fixture and compile it inside the runner image.
# Takes ~30 min for a cold cache (gcc + llvm builds dominate).
bash benchmarks/setup.sh
```

Useful `setup.sh` flags:

```bash
bash benchmarks/setup.sh --tasks 04_sqlite_cte,08_express_async   # subset
bash benchmarks/setup.sh --skip-build                         # clone only
bash benchmarks/setup.sh --force                              # re-clone even if cached
```

## Running the benchmark

```bash
# Baseline pass: proxy OFF
bash benchmarks/run.sh --runs 3 --proxy off

# Treatment pass: proxy ON (stuck detection + nudge)
bash benchmarks/run.sh --runs 3 --proxy on

# Compare
python3 benchmarks/compare.py benchmarks/results/run_001 benchmarks/results/run_002
```

`run.sh` flags:

- `--runs N` — repeat each task N times (default 1)
- `--proxy on|off` — default off
- `--auth subscription|env` — subscription mounts `~/.claude/.credentials.json`;
  env loads `benchmarks/.env` (`ANTHROPIC_API_KEY=...`)
- `--tasks a,b,c` — run a subset
- `--concurrency N` — max parallel containers (default 6)
- `--mode run|surrogate` — `surrogate` swaps `claude -p` for a trivial probe
  that exercises the plumbing without making an API call. Use this during
  harness development to avoid burning credits.
- `--run-id NAME` — label the results dir (default: auto `run_NNN`)

## Sanity-checking the harness without real Claude calls

```bash
# 1. Container smoke test
docker run --rm --entrypoint bash benchmark-runner:latest \
  -c 'python3 -c "print(\"hello\")" && claude --version'

# 2. Real-clone fixture + container-compile path (no Claude API)
bash benchmarks/setup.sh --tasks 08_express_async
bash benchmarks/run.sh    --tasks 08_express_async --mode surrogate
```

After step 2, confirm `benchmarks/fixtures/08_express_async/node_modules/`
exists and is owned by your user, and that
`benchmarks/results/run_NNN/08_express_async/summary_1.json` was written.

## Results layout

```
benchmarks/results/run_NNN/
  run.log                      # human-readable log of the whole run
  manifest_snapshot.json       # copy of manifest.json at run time
  proxy.log                    # proxy stdout/stderr (if --proxy on)
  proxy_events.jsonl           # proxy's event stream copied from ~/.stuck-detector/logs/
  <task_id>/
    summary_1.json             # {task_id, duration_seconds, exit_code, ...}
    stdout_1.log               # claude -p stdout
    stderr_1.log               # claude -p stderr
    docker_1.log               # docker wrapper log
    verify_1.json              # if tasks/<id>/verify.sh is present
    verify_1.log
    summary_2.json             # if --runs >= 2
    ...
```

## Task taxonomy

See `PLAN.md §"The 10 tasks"`. 4 stuck-prone (tier 1), 6 productive
(tier 2/3), with `08_express_async` as the known-FP control. `24_rbtree`
and `27_minicoro_cet` were dropped from the original 12 — see PLAN.md
§"The 10 tasks" for the reasoning.

Prompts and SHAs are sourced from
`/home/nicolas/source/classifier-repos/{prompts,worktrees}/` — see
`PLAN.md §"Sources of truth"` for the ID mapping.
