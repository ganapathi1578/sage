"""Loss functions for mixed single-answer and multi-answer QA."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def option_supervision_loss(
    logits: torch.Tensor,
    target_labels: torch.Tensor,
    option_mask: torch.Tensor,
    cfg: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Compute a batch-mean loss supporting single- and multi-correct questions.

    ``target_labels`` is multi-hot [B, K]. ``option_mask`` is true only for
    real options. In ``mixed_ce_bce`` mode, one-positive rows use cross entropy
    while rows with multiple positives use masked, per-example BCE. ``bce`` is
    available as an ablation that uses BCE for every example.
    """
    cfg = cfg or {}
    if logits.ndim != 2 or target_labels.shape != logits.shape or option_mask.shape != logits.shape:
        raise ValueError(
            "logits, target_labels, and option_mask must have the same [batch, options] shape; "
            f"got {tuple(logits.shape)}, {tuple(target_labels.shape)}, {tuple(option_mask.shape)}"
        )

    valid = option_mask.bool()
    targets = target_labels.to(dtype=logits.dtype)
    positive_counts = ((targets > 0.5) & valid).sum(dim=1)
    valid_counts = valid.sum(dim=1)
    if torch.any(valid_counts <= 0):
        raise ValueError("Every QA example must have at least one valid answer option")
    if torch.any(positive_counts <= 0):
        bad = torch.nonzero(positive_counts <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Examples have no positive target among valid options: batch indices {bad}")
    if torch.any((targets > 0.5) & ~valid):
        raise ValueError("A padded/non-existent option is marked as a positive target")
    invalid_values = ((targets != 0.0) & (targets != 1.0)) & valid
    if torch.any(invalid_values):
        raise ValueError("Target labels must be binary 0/1 values on valid options")

    mode = str(cfg.get("target_loss", "mixed_ce_bce")).lower()
    safe_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    bce_logits = logits.masked_fill(~valid, 0.0)
    if mode == "bce":
        elementwise = F.binary_cross_entropy_with_logits(bce_logits, targets, reduction="none")
        per_example = (elementwise * valid.to(elementwise.dtype)).sum(dim=1) / valid_counts.clamp_min(1)
        return per_example.mean()
    if mode != "mixed_ce_bce":
        raise ValueError(f"Unknown training.target_loss={mode!r}; expected mixed_ce_bce or bce")

    single = positive_counts == 1
    multi = positive_counts > 1
    per_example = logits.new_zeros((logits.shape[0],))
    if single.any():
        single_targets = (targets[single] * valid[single].to(targets.dtype)).argmax(dim=1)
        per_example[single] = F.cross_entropy(safe_logits[single], single_targets, reduction="none")
    if multi.any():
        elementwise = F.binary_cross_entropy_with_logits(bce_logits[multi], targets[multi], reduction="none")
        valid_multi = valid[multi].to(elementwise.dtype)
        per_example[multi] = (elementwise * valid_multi).sum(dim=1) / valid_counts[multi].clamp_min(1)
    return per_example.mean()
