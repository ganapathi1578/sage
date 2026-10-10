"""Loss functions for mixed single-choice and multi-label QA."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def option_supervision_loss(
    logits: torch.Tensor,
    target_labels: torch.Tensor,
    option_mask: torch.Tensor,
    cfg: dict[str, Any] | None = None,
    is_multi_label_task: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a batch-mean loss based on each row's *task semantics*.

    ``target_labels`` is multi-hot [B, K]. The explicit task mask is essential:
    a multi-label task with exactly one positive still uses BCE, and a known-false
    multi-label task with zero positives also uses BCE. If no task mask is supplied,
    the legacy convenience behavior infers multi-label rows from counts != 1; the
    training/evaluation pipeline always supplies the explicit task mask.
    """
    cfg = cfg or {}
    if logits.ndim != 2 or target_labels.shape != logits.shape or option_mask.shape != logits.shape:
        raise ValueError(
            "logits, target_labels, and option_mask must have the same [batch, options] shape; "
            f"got {tuple(logits.shape)}, {tuple(target_labels.shape)}, {tuple(option_mask.shape)}"
        )

    valid = option_mask.bool()
    targets = target_labels.to(dtype=logits.dtype)
    valid_counts = valid.sum(dim=1)
    positive = targets > 0.5
    positive_counts = (positive & valid).sum(dim=1)
    if torch.any(valid_counts <= 0):
        raise ValueError("Every QA example must have at least one valid answer option")
    if torch.any(positive & ~valid):
        raise ValueError("A padded/non-existent option is marked as a positive target")
    if torch.any((targets != 0.0) & ~valid):
        raise ValueError("Padded/non-existent options must have target value zero")
    if torch.any(((targets != 0.0) & (targets != 1.0)) & valid):
        raise ValueError("Target labels must be binary 0/1 values on valid options")

    if is_multi_label_task is None:
        # Compatibility for direct callers/tests. Production loaders provide the
        # explicit per-row task type, avoiding ambiguity for one-positive multi-label rows.
        multi_task = positive_counts != 1
    else:
        multi_task = is_multi_label_task.to(device=logits.device, dtype=torch.bool).reshape(-1)
        if multi_task.shape != (logits.shape[0],):
            raise ValueError(f"is_multi_label_task must have shape [{logits.shape[0]}]")

    single_task = ~multi_task
    invalid_single = single_task & (positive_counts != 1)
    if invalid_single.any():
        bad = torch.nonzero(invalid_single, as_tuple=False).flatten().tolist()
        counts = positive_counts[invalid_single].detach().cpu().tolist()
        raise ValueError(f"Single-choice examples must have exactly one positive target; batch indices={bad}, counts={counts}")

    mode = str(cfg.get("target_loss", "mixed_ce_bce")).lower()
    safe_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    bce_logits = logits.masked_fill(~valid, 0.0)
    valid_float = valid.to(dtype=logits.dtype)
    if mode == "bce":
        elementwise = F.binary_cross_entropy_with_logits(bce_logits, targets, reduction="none")
        per_example = (elementwise * valid_float).sum(dim=1) / valid_counts.clamp_min(1)
        return per_example.mean()
    if mode != "mixed_ce_bce":
        raise ValueError(f"Unknown training.target_loss={mode!r}; expected mixed_ce_bce or bce")

    per_example = logits.new_zeros((logits.shape[0],))
    if single_task.any():
        single_targets = (targets[single_task] * valid[single_task].to(targets.dtype)).argmax(dim=1)
        per_example[single_task] = F.cross_entropy(safe_logits[single_task], single_targets, reduction="none")
    if multi_task.any():
        # Includes zero-positive and one-positive multi-label rows.
        elementwise = F.binary_cross_entropy_with_logits(bce_logits[multi_task], targets[multi_task], reduction="none")
        valid_multi = valid[multi_task].to(elementwise.dtype)
        per_example[multi_task] = (
            (elementwise * valid_multi).sum(dim=1) / valid_counts[multi_task].clamp_min(1)
        )
    return per_example.mean()
