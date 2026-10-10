"""Metrics for mixed single-answer and multi-answer QA."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


class MetricAccumulator:
    """Bounded-memory metrics for per-option multi-hot targets.

    ``accuracy`` remains a backward-compatible top-1 hit rate: the highest
    scoring real option is counted correct if it is among the valid targets.
    Single-answer accuracy and true multi-label metrics are reported separately.
    ``selection_score`` averages single-answer accuracy and multi-answer micro-F1
    when both categories occur; otherwise it uses the available category metric.
    """
    def __init__(self, threshold: float = 0.5):
        if not 0.0 < threshold < 1.0:
            raise ValueError("multi-label threshold must be between 0 and 1")
        self.threshold = float(threshold)
        self.examples = 0
        self.loss_sum = 0.0
        self.top1_hits = 0
        self.single_examples = 0
        self.single_correct = 0
        self.multi_examples = 0
        self.multi_top1_hits = 0
        self.multi_exact_match = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.groups: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "top1_hits": 0})

    def update(
        self,
        logits: torch.Tensor,
        target_labels: torch.Tensor,
        option_mask: torch.Tensor,
        loss: torch.Tensor | float,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        valid = option_mask.bool()
        targets = target_labels > 0.5
        if logits.shape != targets.shape or valid.shape != logits.shape:
            raise ValueError("logits, targets and option_mask must share [B, K] shape")
        masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        pred_top1 = masked_logits.argmax(dim=1)
        target_counts = (targets & valid).sum(dim=1)
        if torch.any(target_counts < 1):
            raise ValueError("Metric update received an example without a positive valid target")
        rows = torch.arange(logits.shape[0], device=logits.device)
        top1_hit = targets[rows, pred_top1]
        single = target_counts == 1
        multi = target_counts > 1

        n = int(logits.shape[0])
        self.examples += n
        self.loss_sum += float(loss.detach() if torch.is_tensor(loss) else loss) * n
        self.top1_hits += int(top1_hit.sum().item())
        self.single_examples += int(single.sum().item())
        self.single_correct += int((top1_hit & single).sum().item())
        self.multi_examples += int(multi.sum().item())
        self.multi_top1_hits += int((top1_hit & multi).sum().item())

        if multi.any():
            probabilities = torch.sigmoid(logits[multi])
            predicted = (probabilities >= self.threshold) & valid[multi]
            actual = targets[multi] & valid[multi]
            self.tp += int((predicted & actual).sum().item())
            self.fp += int((predicted & ~actual & valid[multi]).sum().item())
            self.fn += int((~predicted & actual).sum().item())
            exact = ((predicted == actual) | ~valid[multi]).all(dim=1)
            self.multi_exact_match += int(exact.sum().item())

        if metadata:
            hits_cpu = top1_hit.detach().cpu().tolist()
            for item, hit in zip(metadata, hits_cpu):
                if not isinstance(item, dict):
                    continue
                for key in ("task_type", "reasoning_family", "generator_family", "truth_state"):
                    value = item.get(key)
                    if value is not None:
                        group = self.groups[f"{key}={value}"]
                        group["count"] += 1
                        group["top1_hits"] += int(bool(hit))

    def compute(self) -> dict[str, Any]:
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        category_scores = []
        single_acc = None
        if self.single_examples:
            single_acc = self.single_correct / self.single_examples
            category_scores.append(single_acc)
        multi_f1 = f1 if self.multi_examples else None
        if multi_f1 is not None:
            category_scores.append(multi_f1)
        selection_score = sum(category_scores) / len(category_scores) if category_scores else 0.0
        result: dict[str, Any] = {
            "examples": self.examples,
            "loss": self.loss_sum / max(1, self.examples),
            # For multi-answer rows, top-1 hit means the top option is any correct option.
            "accuracy": self.top1_hits / max(1, self.examples),
            "top1_any_correct": self.top1_hits / max(1, self.examples),
            "single_choice_examples": self.single_examples,
            "single_choice_accuracy": single_acc,
            "multi_answer_examples": self.multi_examples,
            "multi_answer_top1_hit": self.multi_top1_hits / max(1, self.multi_examples),
            "multi_label_precision": precision if self.multi_examples else None,
            "multi_label_recall": recall if self.multi_examples else None,
            "multi_label_f1": multi_f1,
            "multi_label_exact_match": self.multi_exact_match / max(1, self.multi_examples) if self.multi_examples else None,
            "multi_label_threshold": self.threshold,
            "selection_score": selection_score,
        }
        group_metrics = {}
        for name, values in self.groups.items():
            count = values["count"]
            group_metrics[name] = {"examples": count, "top1_any_correct": values["top1_hits"] / max(1, count)}
        if group_metrics:
            result["by_group"] = group_metrics
        return result
