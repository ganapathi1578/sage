#!/usr/bin/env python3
"""Compute calibration metrics from Sage evaluate.py JSONL predictions.

No fitting is performed. Use this on val/test predictions to measure calibration
without changing logits or learning a temperature on the test set.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / max(float(ex.sum()), 1e-300)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # Stable sigmoid without scipy.
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _ece(conf: np.ndarray, outcome: np.ndarray, bins: int) -> tuple[float, list[dict[str, Any]]]:
    conf = np.asarray(conf, dtype=np.float64).reshape(-1)
    outcome = np.asarray(outcome, dtype=np.float64).reshape(-1)
    if conf.size != outcome.size:
        raise ValueError("confidence and outcome lengths differ")
    if conf.size == 0:
        return float("nan"), []
    result = 0.0
    details = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        chosen = (conf >= lo) & ((conf < hi) if b < bins - 1 else (conf <= hi))
        count = int(chosen.sum())
        if not count:
            continue
        mean_conf = float(conf[chosen].mean())
        empirical = float(outcome[chosen].mean())
        weight = count / conf.size
        result += weight * abs(mean_conf - empirical)
        details.append({
            "bin": b,
            "count": count,
            "mean_confidence": mean_conf,
            "empirical_rate": empirical,
            "absolute_gap": abs(mean_conf - empirical),
        })
    return float(result), details


def _single_metrics(rows: list[dict[str, Any]], bins: int) -> dict[str, Any]:
    correct, confs, briers, nlls = [], [], [], []
    for row in rows:
        scores = np.asarray(row.get("scores", []), dtype=np.float64)
        if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
            continue
        targets = [int(i) for i in row.get("target_indices", [])]
        if len(targets) != 1 or not 0 <= targets[0] < scores.size:
            continue
        probs = _softmax(scores)
        pred = int(np.argmax(probs))
        y = np.zeros_like(probs)
        y[targets[0]] = 1.0
        correct.append(float(pred == targets[0]))
        confs.append(float(probs[pred]))
        briers.append(float(np.square(probs - y).sum()))
        nlls.append(float(-np.log(max(float(probs[targets[0]]), 1e-15))))
    ece, bins_report = _ece(np.asarray(confs), np.asarray(correct), bins)
    return {
        "examples": len(correct),
        "accuracy": float(np.mean(correct)) if correct else float("nan"),
        "mean_top1_confidence": float(np.mean(confs)) if confs else float("nan"),
        "top1_ece": ece,
        "multiclass_brier": float(np.mean(briers)) if briers else float("nan"),
        "nll": float(np.mean(nlls)) if nlls else float("nan"),
        "top1_ece_bins": bins_report,
    }


def _multi_metrics(rows: list[dict[str, Any]], bins: int) -> dict[str, Any]:
    all_p, all_y = [], []
    exact, top1_hits = [], []
    n_examples = 0
    for row in rows:
        scores = np.asarray(row.get("scores", []), dtype=np.float64)
        if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
            continue
        targets = [int(i) for i in row.get("target_indices", [])]
        if any(i < 0 or i >= scores.size for i in targets):
            continue
        p = _sigmoid(scores)
        y = np.zeros_like(p)
        y[targets] = 1.0
        predicted = [int(i) for i in row.get("predicted_positive_indices_at_threshold", [])]
        exact.append(float(set(predicted) == set(targets)))
        top1 = int(np.argmax(scores))
        top1_hits.append(float(top1 in targets))
        all_p.extend(p.tolist())
        all_y.extend(y.tolist())
        n_examples += 1
    p = np.asarray(all_p, dtype=np.float64)
    y = np.asarray(all_y, dtype=np.float64)
    ece, bins_report = _ece(p, y, bins)
    if p.size:
        brier = float(np.square(p - y).mean())
        nll = float(-(y * np.log(np.clip(p, 1e-15, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p, 1e-15, 1.0))).mean())
    else:
        brier = nll = float("nan")
    return {
        "examples": n_examples,
        "valid_option_labels": int(p.size),
        "option_binary_ece": ece,
        "option_binary_brier": brier,
        "option_binary_nll": nll,
        "top1_any_correct": float(np.mean(top1_hits)) if top1_hits else float("nan"),
        "exact_set_match_at_configured_threshold": float(np.mean(exact)) if exact else float("nan"),
        "option_ece_bins": bins_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="JSONL emitted by scripts.evaluate.py")
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    if args.bins < 2:
        parser.error("--bins must be >= 2")
    path = Path(args.predictions)
    if not path.is_file():
        raise FileNotFoundError(path)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            target_type = str(row.get("target_type", "single_choice"))
            groups[target_type].append(row)
            count += 1
    report: dict[str, Any] = {
        "source_predictions": str(path),
        "records_read": count,
        "bins": args.bins,
        "single_choice": _single_metrics(groups.get("single_choice", []), args.bins),
        "multi_label": _multi_metrics(groups.get("multi_label", []), args.bins),
        "notes": [
            "These are raw-logit calibration metrics; no temperature was fitted.",
            "Single-choice ECE bins compare top-option confidence with top-1 correctness.",
            "Multi-label ECE/Brier/NLL are measured over valid option-level Bernoulli probabilities.",
            "Do not fit calibration parameters on the test split.",
        ],
    }
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            return None
        if isinstance(value, np.integer):
            return int(value)
        return value

    output = json.dumps(clean(report), indent=2, ensure_ascii=False, allow_nan=False)
    print(output)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
