"""Validation/test evaluation for variable-candidate multiple-choice batches."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .metrics import MetricAccumulator
from .losses import option_supervision_loss


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(
    model, loader, device: torch.device, *, output_path: str | Path | None = None,
    threshold: float = 0.5, loss_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    acc = MetricAccumulator(threshold=threshold)
    writer = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = output_path.open("w", encoding="utf-8")
    try:
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch).float()
            targets = batch["target_labels"]
            option_mask = batch["option_mask"]
            loss = option_supervision_loss(logits, targets, option_mask, loss_cfg or {})
            acc.update(logits, targets, option_mask, loss, batch.get("metadata"))
            if writer:
                valid = option_mask.bool()
                masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
                preds = masked_logits.argmax(1).detach().cpu().tolist()
                scores = logits.detach().cpu().tolist()
                targets_list = targets.detach().cpu().tolist()
                probs = torch.sigmoid(logits).detach().cpu()
                threshold_preds = ((probs >= threshold) & valid.detach().cpu()).tolist()
                valid_counts = valid.sum(dim=1).detach().cpu().tolist()
                for i, pred in enumerate(preds):
                    target_indices = [j for j, value in enumerate(targets_list[i][:valid_counts[i]]) if value > 0.5]
                    predicted_indices = [j for j, value in enumerate(threshold_preds[i][:valid_counts[i]]) if value]
                    writer.write(json.dumps({
                        "video_id": batch["video_ids"][i],
                        "query_sentence_id": batch["query_sentence_ids"][i],
                        "option_sentence_ids": batch["option_sentence_ids"][i][:valid_counts[i]],
                        "target_type": "single" if len(target_indices) == 1 else "multi",
                        "target_indices": target_indices,
                        "prediction_top1": pred,
                        "top1_is_correct": pred in target_indices,
                        "predicted_positive_indices_at_threshold": predicted_indices,
                        "exact_set_match": (predicted_indices == target_indices) if len(target_indices) > 1 else None,
                        "scores": scores[i][:valid_counts[i]],
                        "metadata": batch["metadata"][i],
                    }, ensure_ascii=False, default=str) + "\n")
    finally:
        if writer:
            writer.close()
    return acc.compute()
