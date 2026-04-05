#!/usr/bin/env bash
# Run a single SWE-bench problem with Claude Code.
# Usage: ./benchmark/run_swebench.sh <instance_id> <condition>
#   instance_id: e.g. django__django-11001
#   condition:   stock | compact | full

set -euo pipefail

INSTANCE_ID="${1:?Usage: run_swebench.sh <instance_id> <condition>}"
CONDITION="${2:?Usage: run_swebench.sh <instance_id> <condition>}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$REPO_ROOT/results"
WORK_DIR="/tmp/rewind-runs/swebench_${INSTANCE_ID}_${CONDITION}"

mkdir -p "$RESULTS_DIR"

# Extract problem data
PROBLEM_DATA=$(python3 -c "
import json
with open('/tmp/rewind-tasks/swebench_selected.json') as f:
    data = json.load(f)
p = next((d for d in data if d['instance_id'] == '$INSTANCE_ID'), None)
if not p:
    print('NOT_FOUND')
    exit(1)
print(json.dumps(p))
")

if [ "$PROBLEM_DATA" = "NOT_FOUND" ]; then
  echo "Instance $INSTANCE_ID not found"
  exit 1
fi

REPO=$(echo "$PROBLEM_DATA" | python3 -c "import json,sys; print(json.load(sys.stdin)['repo'])")
COMMIT=$(echo "$PROBLEM_DATA" | python3 -c "import json,sys; print(json.load(sys.stdin)['base_commit'])")
PROBLEM_STMT=$(echo "$PROBLEM_DATA" | python3 -c "import json,sys; print(json.load(sys.stdin)['problem_statement'])")
FAIL_TESTS=$(echo "$PROBLEM_DATA" | python3 -c "import json,sys; print(json.load(sys.stdin)['FAIL_TO_PASS'])")
GROUND_TRUTH=$(echo "$PROBLEM_DATA" | python3 -c "import json,sys; print(json.load(sys.stdin)['patch'])")

# Pick binary
case $CONDITION in
  stock)
    CLAUDE_BIN="$(which claude)"
    export CLAUDE_REWIND_MODE="off"
    ;;
  compact)
    CLAUDE_BIN="$REPO_ROOT/bin/claude"
    export CLAUDE_REWIND_MODE="compact_only"
    ;;
  full)
    CLAUDE_BIN="$REPO_ROOT/bin/claude"
    export CLAUDE_REWIND_MODE="full"
    ;;
esac

echo "=== SWE-bench: $INSTANCE_ID | $CONDITION ==="
echo "Repo: $REPO @ ${COMMIT:0:12}"
echo ""

# Clone repo at the right commit
if [ -d "$WORK_DIR" ]; then
  echo "Cleaning previous run..."
  rm -rf "$WORK_DIR"
fi

echo "Cloning $REPO..."
git clone --quiet "https://github.com/$REPO.git" "$WORK_DIR" 2>/dev/null
cd "$WORK_DIR"
git checkout --quiet "$COMMIT"
echo "Checked out $COMMIT"
echo ""

# Build the prompt
PROMPT="You are fixing a bug in the $REPO repository (currently at commit ${COMMIT:0:12}).

## Problem Statement (from GitHub issue)

$PROBLEM_STMT

## Your Task

Find and fix the bug described above. The fix should be minimal — change only what's necessary.

## Tests

After your fix, these tests should pass:
$FAIL_TESTS

You can run the relevant tests to verify your fix. For this repo:
$(case $REPO in
    django/django) echo "Run: python -m pytest <test_file>::<test_name> --no-header -q";;
    sympy/sympy) echo "Run: python -m pytest <test_file>::<test_name> -x -q or python -c 'from sympy.utilities.iterables import partitions; ...'";;
    pytest-dev/pytest) echo "Run: python -m pytest <test_file>::<test_name> --no-header -q";;
    scikit-learn/scikit-learn) echo "Run: python -m pytest <test_file>::<test_name> --no-header -q";;
    pallets/flask) echo "Run: python -m pytest <test_file>::<test_name> --no-header -q";;
esac)

Important: Make your edits directly to the source files. Do not create new files unless necessary."

START_TIME=$(date +%s)

"$CLAUDE_BIN" -p "$PROMPT" \
  --allowedTools "Read,Write,Edit,Bash,Grep,Glob" \
  2>"$WORK_DIR/stderr.log" \
  >"$WORK_DIR/claude_output.txt" || true

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Generate the diff
DIFF=$(git diff)
echo "$DIFF" > "$WORK_DIR/model_patch.diff"

echo ""
echo "=== RESULT: $INSTANCE_ID | $CONDITION ==="
echo "Duration: ${DURATION}s"
echo "Diff size: $(echo "$DIFF" | wc -l) lines"
echo ""
echo "--- Model patch ---"
echo "$DIFF" | head -40
echo ""

# Check telemetry
if [ -f ~/.claude-rewind-logs/events-$(date +%Y-%m-%d).jsonl ]; then
  COMPACTS=$(grep -c '"type":"compact"' ~/.claude-rewind-logs/events-$(date +%Y-%m-%d).jsonl 2>/dev/null || echo 0)
  REWINDS=$(grep -c '"type":"rewind"' ~/.claude-rewind-logs/events-$(date +%Y-%m-%d).jsonl 2>/dev/null || echo 0)
  echo "Compactions: $COMPACTS"
  echo "Rewinds: $REWINDS"
fi

# Log result
HAS_DIFF="false"
[ -n "$DIFF" ] && HAS_DIFF="true"
cat >> "$RESULTS_DIR/swebench_log.jsonl" << JSONEOF
{"instance_id":"$INSTANCE_ID","condition":"$CONDITION","duration":$DURATION,"has_diff":$HAS_DIFF,"diff_lines":$(echo "$DIFF" | wc -l),"compactions":${COMPACTS:-0},"rewinds":${REWINDS:-0},"timestamp":"$(date -Iseconds)"}
JSONEOF
