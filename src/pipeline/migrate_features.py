"""Migrate or verify .gz artifact feature files between schema versions.

Usage:
  python src/pipeline/migrate_features.py <path.gz> [--to-version N] [--verify]
"""

import argparse
import gzip
import json
import os
import sys
import tempfile
from typing import Callable

CURRENT_VERSION = 2

# Migration functions: (from_version, to_version) -> callable(step_dict) -> step_dict
MIGRATIONS: dict[tuple[int, int], Callable[[dict], dict]] = {
    # (1, 2): migrate_v1_to_v2,
}


def migrate_step(step: dict, from_version: int, to_version: int) -> dict:
    """Chain migrations from from_version to to_version.

    Args:
        step: step feature dict
        from_version: current schema version
        to_version: target schema version

    Returns:
        migrated step dict
    """
    current = from_version
    result = dict(step)
    while current < to_version:
        key = (current, current + 1)
        if key not in MIGRATIONS:
            raise ValueError(
                f"No migration path from version {current} to {current + 1}"
            )
        result = MIGRATIONS[key](result)
        current += 1
    return result


def migrate_artifact(
    gz_path: str,
    to_version: int | None = None,
    verify: bool = False,
) -> None:
    """Migrate or verify a .gz artifact.

    Reads rows from gz_path. If verify=True, checks schema consistency.
    Otherwise, migrates rows to to_version and rewrites atomically.

    Args:
        gz_path: path to .gz artifact
        to_version: target schema version (default: CURRENT_VERSION)
        verify: if True, just verify consistency without modifying
    """
    target_version = to_version if to_version is not None else CURRENT_VERSION

    rows = []
    with gzip.open(gz_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if verify:
        _verify_artifact(rows, gz_path)
        return

    # Check if migration is needed
    versions_found = {row.get("schema_version") for row in rows}
    if versions_found == {target_version}:
        print(f"Already at version {target_version}, nothing to do.")
        return

    # Migrate rows
    migrated = []
    for row in rows:
        row_version = row.get("schema_version", 1)
        if row_version == target_version:
            migrated.append(row)
            continue
        if row_version > target_version:
            raise ValueError(
                f"Row has schema_version={row_version} > target {target_version}"
            )
        # Migrate steps
        new_steps = [
            migrate_step(step, row_version, target_version)
            for step in row.get("steps", [])
        ]
        new_row = dict(row)
        new_row["schema_version"] = target_version
        new_row["steps"] = new_steps
        migrated.append(new_row)

    # Atomic write
    dir_path = os.path.dirname(gz_path) or "."
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=dir_path, delete=False, suffix=".tmp.gz"
        ) as tmp:
            tmp_path = tmp.name
            with gzip.open(tmp, "wt") as gz_out:
                for row in migrated:
                    gz_out.write(json.dumps(row) + "\n")
        os.replace(tmp_path, gz_path)
        print(f"Migrated {len(migrated)} rows to version {target_version} in {gz_path}")
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _verify_artifact(rows: list[dict], gz_path: str) -> None:
    """Verify schema consistency of artifact rows."""
    errors = []

    # Check schema_version consistency across rows of same session
    session_versions: dict[str, set] = {}
    session_n_steps: dict[str, list] = {}

    for i, row in enumerate(rows):
        sid = row.get("session_id", f"row_{i}")
        ver = row.get("schema_version")
        n_steps = row.get("n_steps")
        steps = row.get("steps", [])

        session_versions.setdefault(sid, set()).add(ver)

        if n_steps is not None and len(steps) != n_steps:
            errors.append(
                f"Row {i} (session {sid}): n_steps={n_steps} but len(steps)={len(steps)}"
            )

        session_n_steps.setdefault(sid, []).append(n_steps)

    # Check for inconsistent n_steps across rows of same session
    for sid, ns_list in session_n_steps.items():
        unique_ns = set(ns_list)
        if len(unique_ns) > 1:
            errors.append(
                f"Session {sid} has inconsistent n_steps across rows: {unique_ns}"
            )

    for sid, vers in session_versions.items():
        if len(vers) > 1:
            errors.append(f"Session {sid} has mixed schema_versions: {vers}")

    if errors:
        print(f"VERIFY FAILED for {gz_path}: {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"OK: {gz_path} verified ({len(rows)} rows, no issues)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate or verify .gz artifact")
    parser.add_argument("gz_path", help="Path to .gz artifact")
    parser.add_argument("--to-version", type=int, default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    migrate_artifact(args.gz_path, to_version=args.to_version, verify=args.verify)


if __name__ == "__main__":
    main()
