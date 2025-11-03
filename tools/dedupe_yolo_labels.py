# tools/dedupe_yolo_labels.py
"""
Deduplicate YOLO labels named like: <id>-<file_base>.txt
Examples:
  0a725e7e-chart_0140.txt
  0a1235e7e-chart_0140.txt   # duplicate for same base

Default strategy: keep the file with the NEWEST modification time (mtime).
Optional strategy: keep by LEX order of <id>.

Usage:
  python tools/dedupe_yolo_labels.py --root datasets/tradingview/labels --recursive --dry-run
  python tools/dedupe_yolo_labels.py --root datasets/tradingview/labels --recursive --rename
  python tools/dedupe_yolo_labels.py --root datasets/tradingview/labels --recursive --strategy id_lex
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Matches "<id>-<base>.txt" where id can be any non-dash run
PATTERN = re.compile(r"^([^-]+)-(.+)\.txt$", re.IGNORECASE)


def collect_label_files(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return [p for p in root.rglob("*.txt") if p.is_file()]
    return [p for p in root.glob("*.txt") if p.is_file()]


def group_by_base(files: List[Path]) -> Dict[Tuple[Path, str], List[Path]]:
    """
    Group by (directory, base_name) where base_name is the filename WITHOUT the '<id>-' prefix.
    Returns mapping -> list of paths for that base.
    """
    groups: Dict[Tuple[Path, str], List[Path]] = {}
    for p in files:
        m = PATTERN.match(p.name)
        if not m:
            # Not id-prefixed, ignore for dedupe. (Canonical already)
            continue
        base_name = m.group(2) + ".txt"
        key = (p.parent, base_name)
        groups.setdefault(key, []).append(p)
    return groups


def pick_keep_file(paths: List[Path], strategy: str) -> Path:
    """
    Pick which path to keep among duplicates.
    strategy:
      - 'mtime'  : keep newest by modification time (default)
      - 'id_lex' : keep lexicographically largest <id> (prefix before first dash)
    """
    if strategy == "id_lex":

        def id_of(p: Path) -> str:
            m = PATTERN.match(p.name)
            return m.group(1) if m else ""

        return max(paths, key=lambda p: id_of(p))
    # default: mtime (nanosecond precision when available)
    return max(paths, key=lambda p: p.stat().st_mtime_ns)


def dedupe_group(
    dir_path: Path,
    base_name: str,
    candidates: List[Path],
    rename: bool,
    dry_run: bool,
    strategy: str,
) -> None:
    keep = pick_keep_file(candidates, strategy=strategy)
    to_delete = [p for p in candidates if p != keep]

    target = dir_path / base_name  # canonical name without id
    if rename:
        # If a canonical file already exists and is different from keep, remove it first
        if target.exists() and target != keep:
            if dry_run:
                print(f"[DRY] rm existing canonical {target}")
            else:
                target.unlink()
        if keep != target:
            if dry_run:
                print(f"[DRY] rename {keep.name} -> {base_name}")
            else:
                keep.rename(target)
            keep = target

    for p in to_delete:
        if dry_run:
            print(f"[DRY] rm {p}")
        else:
            p.unlink()

    kept_name = keep.name if keep.exists() else base_name
    print(f"Kept: {kept_name} in {dir_path} | Deleted: {len(to_delete)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root labels dir (e.g., datasets/tradingview/labels)",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subfolders (train/val/test).",
    )
    ap.add_argument(
        "--rename",
        action="store_true",
        help="Rename kept file to canonical '<base>.txt' (strip the '<id>-').",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print actions without modifying files."
    )
    ap.add_argument(
        "--strategy",
        choices=["mtime", "id_lex"],
        default="mtime",
        help="How to decide the 'latest' label. Default: mtime.",
    )
    args = ap.parse_args()

    files = collect_label_files(args.root, args.recursive)
    groups = group_by_base(files)

    if not groups:
        print("No id-prefixed label files found. Nothing to do.")
        return

    total_groups = 0
    total_dups = 0
    for (dir_path, base_name), paths in groups.items():
        if len(paths) <= 1:
            continue
        total_groups += 1
        total_dups += len(paths) - 1
        dedupe_group(
            dir_path, base_name, paths, args.rename, args.dry_run, args.strategy
        )

    print(f"\nProcessed {total_groups} groups | Removed {total_dups} duplicates.")
    if args.dry_run:
        print("Dry run only. Re-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
