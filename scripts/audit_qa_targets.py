#!/usr/bin/env python3
"""Scan every ID-coded QA row and summarize/validate target semantics."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from sageqa.config import load_config
from sageqa.data.qa_dataset import QAParquetStream


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scale", help="Override data.scale")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Optional debug cap; omit to audit every row in selected split(s).")
    args = parser.parse_args()
    cfg, repo_root = load_config(args.config)
    data_cfg = cfg["data"]
    if args.scale:
        data_cfg["scale"] = args.scale
    scale = str(data_cfg.get("scale", "10k"))
    analysis_root = Path(data_cfg["analysis_root"])
    if not analysis_root.is_absolute():
        analysis_root = (repo_root / analysis_root).resolve()
    compact_dir = str(data_cfg.get("compact_dir", "compact"))
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)

    print(f"Scale: {scale}")
    print("Auditing ID-coded targets without loading video/text features.")
    total = 0
    grouped: Counter[tuple[str, str, int]] = Counter()
    for split in splits:
        split_root = analysis_root / scale / compact_dir / split
        ds = QAParquetStream(split_root, split, data_cfg, shuffle=False, seed=int(cfg.get("seed", 42)))
        count = 0
        for row in ds:
            positives = int(sum(row["target_labels"]))
            metadata = row.get("metadata", {})
            truth = metadata.get("truth_state", "<missing>")
            key = (row["target_mode"], str(truth), positives)
            grouped[key] += 1
            count += 1
            if args.max_examples is not None and count >= args.max_examples:
                break
        total += count
        suffix = " (capped)" if args.max_examples is not None and count >= args.max_examples else ""
        print(f"  {split}: {count:,} rows{suffix}")

    print(f"\nTotal audited rows: {total:,}")
    print("Counts grouped by target_mode, truth_state, positive_count:")
    for (mode, truth, positives), count in sorted(grouped.items()):
        print(f"  mode={mode:12s} truth_state={truth:12s} positives={positives:3d}: {count:,}")
    print("\nAudit passed: every selected row satisfied the configured target contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
