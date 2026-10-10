#!/usr/bin/env python3
"""Train a config-selected multiple-choice video-QA model."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from sageqa.config import load_config
from sageqa.data.build_loaders import build_loader
from sageqa.engine.trainer import train
from sageqa.models.registry import build_model


def parse_value(value: str):
    return yaml.safe_load(value)


def apply_override(cfg: dict, expression: str):
    if "=" not in expression:
        raise ValueError(f"Override must be KEY=VALUE, got {expression!r}")
    key, raw = expression.split("=", 1)
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"Override path crosses a non-mapping value: {key}")
    node[parts[-1]] = parse_value(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scale", help="Override data.scale")
    parser.add_argument("--run-name", help="Override experiment.name")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override config, e.g. --set model.num_layers=4")
    args = parser.parse_args()
    cfg, repo_root = load_config(args.config)
    for override in args.set:
        apply_override(cfg, override)
    if args.scale:
        cfg.setdefault("data", {})["scale"] = args.scale
    if args.run_name:
        cfg.setdefault("experiment", {})["name"] = args.run_name
    if "model" not in cfg or "data" not in cfg or "training" not in cfg:
        raise SystemExit("Resolved config must define model, data, and training")
    if int(cfg["training"].get("batch_size", 1)) <= 0:
        raise SystemExit("training.batch_size must be > 0")
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    train_loader, train_ds, train_collator = build_loader(cfg, repo_root, "train")
    val_loader, _, val_collator = build_loader(cfg, repo_root, "val")
    model = build_model(cfg["model"])
    print(f"Repo root: {repo_root}")
    print(f"Experiment: {cfg.get('experiment', {}).get('name', 'experiment')}")
    print(f"Scale: {cfg['data'].get('scale', '10k')}; train rows={len(train_ds):,}; batch={cfg['training'].get('batch_size', 2)}")
    print(f"Model: {cfg['model'].get('name')} | {model.model_summary()}")
    print(f"Video store: {cfg['data'].get('video_store')}")
    print(f"Token store: {cfg['data'].get('text_token_store_root')}/{cfg['data'].get('scale', '10k')}")
    best = train(cfg, repo_root, train_loader, train_ds, train_collator, val_loader, val_collator, model)
    print(f"Training complete. Best checkpoint: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
