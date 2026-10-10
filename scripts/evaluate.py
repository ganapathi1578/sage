#!/usr/bin/env python3
"""Evaluate a saved checkpoint on validation or test split."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sageqa.config import load_config
from sageqa.data.build_loaders import build_loader
from sageqa.engine.evaluator import evaluate
from sageqa.models.registry import build_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--scale")
    parser.add_argument("--predictions", help="Optional JSONL file for per-example predictions")
    args = parser.parse_args()
    cfg, root = load_config(args.config)
    if args.scale:
        cfg["data"]["scale"] = args.scale
    device_name = cfg["training"].get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = build_model(cfg["model"]).to(device)
    state = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    loader, _, collator = build_loader(cfg, root, args.split)
    metrics = evaluate(
        model, loader, device, output_path=args.predictions,
        threshold=float(cfg["training"].get("multi_label_threshold", 0.5)),
        loss_cfg=cfg["training"],
    )
    collator.close()
    print(f"Split={args.split} scale={cfg['data'].get('scale')} checkpoint={args.checkpoint}")
    for key, value in metrics.items():
        if key == "by_group":
            print("by_group:")
            for name, group in sorted(value.items()):
                print(f"  {name}: {group}")
        else:
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
