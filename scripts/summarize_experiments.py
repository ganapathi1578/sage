#!/usr/bin/env python3
"""Create a compact CSV summary from outputs/experiments/*/metrics.jsonl."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/experiments")
    parser.add_argument("--output", default="outputs/experiment_summary.csv")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    rows = []
    for metrics_path in sorted(root.glob("*/metrics.jsonl")):
        latest = None
        best_val = -1.0
        best_record = None
        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                latest = json.loads(line)
                val = latest.get("val", {}).get("accuracy", -1.0)
                if val > best_val:
                    best_val, best_record = val, latest
        if latest is None:
            continue
        meta_path = metrics_path.parent / "model_summary.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        rows.append({
            "experiment": metrics_path.parent.name,
            "architecture": meta.get("architecture"),
            "parameters": meta.get("parameters"),
            "best_val_accuracy": best_val,
            "best_val_loss": (best_record or {}).get("val", {}).get("loss"),
            "best_epoch": (best_record or {}).get("epoch"),
            "last_train_accuracy": latest.get("train", {}).get("accuracy"),
            "last_val_accuracy": latest.get("val", {}).get("accuracy"),
            "global_step": latest.get("global_step"),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    cols = ["experiment", "architecture", "parameters", "best_val_accuracy", "best_val_loss", "best_epoch", "last_train_accuracy", "last_val_accuracy", "global_step"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} experiment rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
