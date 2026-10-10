"""Task metrics for supervised multiple-choice decision models."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


class MetricAccumulator:
    def __init__(self):
        self.examples = 0
        self.loss_sum = 0.0
        self.correct = 0
        self.top2_correct = 0
        self.top3_correct = 0
        self.reciprocal_rank_sum = 0.0
        # Integer aggregates keep memory bounded even for multi-million-row evaluation.
        self.groups: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "correct": 0})

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor | float, metadata: list[dict[str, Any]] | None = None):
        pred = logits.argmax(dim=1)
        n = int(targets.numel())
        self.examples += n
        self.loss_sum += float(loss) * n
        hits = pred.eq(targets).detach().cpu().tolist()
        self.correct += sum(bool(x) for x in hits)
        ranking = logits.argsort(dim=1, descending=True)
        ranks = ranking.eq(targets[:, None]).float().argmax(dim=1) + 1
        self.top2_correct += int(ranks.le(2).sum().item())
        self.top3_correct += int(ranks.le(3).sum().item())
        self.reciprocal_rank_sum += float((1.0 / ranks.float()).sum().item())
        if metadata:
            for item, hit in zip(metadata, hits):
                if not isinstance(item, dict):
                    continue
                for key in ("task_type", "reasoning_family", "generator_family", "truth_state"):
                    value = item.get(key)
                    if value is not None:
                        g = self.groups[f"{key}={value}"]
                        g["count"] += 1
                        g["correct"] += int(bool(hit))

    def compute(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "examples": self.examples,
            "loss": self.loss_sum / max(1, self.examples),
            "accuracy": self.correct / max(1, self.examples),
            "correct": self.correct,
            "top2_accuracy": self.top2_correct / max(1, self.examples),
            "top3_accuracy": self.top3_correct / max(1, self.examples),
            "mean_reciprocal_rank": self.reciprocal_rank_sum / max(1, self.examples),
        }
        group_metrics = {}
        for name, values in self.groups.items():
            count = values["count"]
            group_metrics[name] = {"examples": count, "accuracy": values["correct"] / max(1, count)}
        if group_metrics:
            result["by_group"] = group_metrics
        return result
