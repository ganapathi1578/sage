#!/usr/bin/env python3
"""Validate selected Parquet, token store, and video store before training."""
from __future__ import annotations

import argparse
from pathlib import Path

from sageqa.config import load_config
from sageqa.data.build_loaders import build_loader
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scale")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--batches", type=int, default=1, help="Collate this many batches to validate feature ID mapping")
    args = parser.parse_args()
    cfg, root = load_config(args.config)
    if args.scale:
        cfg["data"]["scale"] = args.scale
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        loader, dataset, collator = build_loader(cfg, root, split)
        print(f"{split}: {len(dataset):,} rows in {dataset.root} across {len(dataset.files)} shards")
        n = 0
        for batch in loader:
            print("  batch shapes:", {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, "shape")})
            print("  sample video IDs:", batch["video_ids"][:3])
            print("  option counts:", batch["option_mask"].sum(dim=1).tolist())
            print("  positive-option indices:", [
                torch.nonzero(row > 0.5, as_tuple=False).flatten().tolist()
                for row in batch["target_labels"]
            ])
            print("  positive-label counts:", batch["target_counts"].tolist())
            print("  target modes:", batch["target_modes"])
            print("  empty multi-label targets:", [
                mode == "multi_label" and int(count) == 0
                for mode, count in zip(batch["target_modes"], batch["target_counts"])
            ])
            n += 1
            if n >= max(1, args.batches):
                break
        collator.close()
        if n == 0:
            raise RuntimeError(f"No batches produced for split {split}")
    print("Data/feature validation passed for requested split(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
