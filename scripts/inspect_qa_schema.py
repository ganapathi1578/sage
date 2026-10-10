#!/usr/bin/env python3
"""Print Parquet schema and sample target labels before training."""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from sageqa.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scale")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()
    cfg, repo_root = load_config(args.config)
    scale = args.scale or cfg["data"].get("scale", "10k")
    root = Path(cfg["data"]["analysis_root"])
    if not root.is_absolute():
        root = repo_root / root
    split_root = root / scale / str(cfg["data"].get("compact_dir", "compact")) / args.split
    files = sorted(split_root.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No Parquet files found: {split_root}")
    columns = cfg["data"].get("columns", {})
    target_cfg = cfg["data"].get("target", {})
    show = [columns.get("video_id", "video_id"), columns.get("query_id", "query_sentence_id"),
            columns.get("option_ids", "option_sentence_ids"), target_cfg.get("column", "labels")]
    print(f"Root: {split_root}\nFiles: {len(files)}\nConfigured relevant columns: {show}\n")
    for path in files[:min(len(files), 3)]:
        pf = pq.ParquetFile(path)
        print(f"FILE: {path.relative_to(split_root)} rows={pf.metadata.num_rows}")
        print(pf.schema_arrow)
        try:
            table = pq.read_table(path, columns=[c for c in show if c in pf.schema_arrow.names])
            for row in table.slice(0, max(1, args.rows)).to_pylist():
                for col in show:
                    if col in row:
                        value = row[col]
                        if isinstance(value, list) and len(value) > 12:
                            value = value[:12] + ["..."]
                        print(f"  {col}: {value!r} (type={type(row[col]).__name__})")
                print()
        except Exception as exc:
            print(f"Could not read configured sample columns: {exc}")
    print("Use the actual schema/value representation to set data.columns and data.target explicitly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
