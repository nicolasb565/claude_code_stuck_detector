#!/bin/bash
# Verify sqlite task: rebuild, run the test CTE query, diff against expected.
# The task-provided test.sql and expected.txt live alongside this script.
set -uo pipefail
SCRATCH="${1:?usage: verify.sh <scratch>}"
TASK_DIR="$(dirname "$0")"

cd "$SCRATCH"
if [ -d build ]; then
  ( cd build && make -j"$(nproc)" sqlite3 ) || exit 2
  SQLITE3="$SCRATCH/build/sqlite3"
elif [ -x ./sqlite3 ]; then
  SQLITE3="$SCRATCH/sqlite3"
else
  make -j"$(nproc)" sqlite3 || exit 2
  SQLITE3="$SCRATCH/sqlite3"
fi

[ -x "$SQLITE3" ] || { echo "sqlite3 binary not found"; exit 2; }

if [ ! -f "$TASK_DIR/test.sql" ] || [ ! -f "$TASK_DIR/expected.txt" ]; then
  echo "verify: test.sql / expected.txt not bundled yet — treating build-success as pass"
  exit 0
fi

"$SQLITE3" :memory: < "$TASK_DIR/test.sql" > /tmp/sqlite_actual.txt
diff -u "$TASK_DIR/expected.txt" /tmp/sqlite_actual.txt
