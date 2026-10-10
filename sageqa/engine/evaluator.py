"""Validation/test evaluation for variable-candidate multiple-choice batches."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .metrics import MetricAccumulator


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device: torch.device, *, output_path: str | Path | None = None) -> dict[str, Any]:
    model.eval()
    acc = MetricAccumulator()
    writer = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = output_path.open("w", encoding="utf-8")
    try:
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch).float()
            targets = batch["target_index"]
            loss = F.cross_entropy(logits, targets)
            acc.update(logits, targets, loss, batch.get("metadata"))
            if writer:
                preds = logits.argmax(1).detach().cpu().tolist()
                scores = logits.detach().cpu().tolist()
                targets_list = targets.detach().cpu().tolist()
                for i, pred in enumerate(preds):
                    writer.write(json.dumps({
                        "video_id": batch["video_ids"][i],
                        "query_sentence_id": batch["query_sentence_ids"][i],
                        "option_sentence_ids": batch["option_sentence_ids"][i],
                        "target_index": targets_list[i],
                        "prediction": pred,
                        "correct": pred == targets_list[i],
                        "scores": scores[i],
                        "metadata": batch["metadata"][i],
                    }, ensure_ascii=False, default=str) + "\n")
    finally:
        if writer:
            writer.close()
    return acc.compute()
