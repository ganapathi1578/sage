"""Laya-style RLCD objectives adapted to Sage's mixed video-QA targets.

Reference method: Laya's Reinforcement Learning for Calibrated Decisions (RLCD),
which combines a REINFORCE-style term over Gaussian-perturbed logits, a group-mean
reward baseline, strictly proper scoring rules, and a supervised cross-entropy term.

Sage-specific adaptation:
* single-choice rows use a categorical distribution over valid options;
* multi-label rows use independent Bernoulli distributions over valid options;
* ordinal ranked-probability scoring is not enabled because Sage's current options
  are not guaranteed to be ordered score levels.

This file intentionally does not change the existing supervised trainer.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .losses import option_supervision_loss

_MASKED_LOGIT = -1.0e4
_EPS = 1.0e-8


def _validate_inputs(
    logits: torch.Tensor,
    target_labels: torch.Tensor,
    option_mask: torch.Tensor,
    is_multi_label_task: torch.Tensor,
    samples: int,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B,K], got {tuple(logits.shape)}")
    if target_labels.shape != logits.shape or option_mask.shape != logits.shape:
        raise ValueError("logits, target_labels, and option_mask must have identical [B,K] shapes")
    if samples < 2:
        raise ValueError("RLCD needs at least 2 exploration samples per example for a group baseline")
    if not (sigma > 0.0):
        raise ValueError(f"sigma must be > 0, got {sigma}")

    mask = option_mask.to(device=logits.device, dtype=torch.bool)
    targets = target_labels.to(device=logits.device, dtype=torch.float32)
    multi = is_multi_label_task.to(device=logits.device, dtype=torch.bool).reshape(-1)
    if multi.shape != (logits.shape[0],):
        raise ValueError(f"is_multi_label_task must have shape [{logits.shape[0]}]")
    if (mask.sum(-1) == 0).any():
        raise ValueError("Every example must have at least one valid option")
    if ((targets != 0) & (targets != 1) & mask).any():
        raise ValueError("target_labels must be binary on valid options")
    if ((targets != 0) & ~mask).any():
        raise ValueError("Padded options must have target zero")

    positive_counts = ((targets > 0.5) & mask).sum(-1)
    invalid_single = (~multi) & (positive_counts != 1)
    if invalid_single.any():
        raise ValueError(
            "Every single-choice RLCD example must have exactly one positive target; "
            f"bad rows={torch.nonzero(invalid_single).flatten().tolist()}"
        )
    return mask, targets, multi


def categorical_proper_reward(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    w_sph: float = 0.75,
) -> torch.Tensor:
    """Log + spherical proper reward for categorical answer distributions.

    `probabilities` can be [B,K] or [G,B,K], target is [B,K], and mask is [B,K].
    Returns one scalar reward per example (or per sample/example).
    """
    m = mask.to(probabilities.dtype)
    y = target.to(probabilities.dtype)
    y = y / (y * m).sum(-1, keepdim=True).clamp_min(1.0)
    q = probabilities * m
    log_score = (y * torch.log(q.clamp_min(1e-12)) * m).sum(-1)
    spherical = (y * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    return log_score + float(w_sph) * spherical


def bernoulli_proper_reward(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    w_sph: float = 0.75,
) -> torch.Tensor:
    """Mean independent-Bernoulli log + spherical reward for multi-label rows.

    Each valid option is treated as a binary decision. The Bernoulli outcome
    distribution is [1-p, p]; both the log and spherical components are strictly
    proper for that binary distribution. Padded options contribute nothing.
    """
    p = probabilities.clamp(1e-7, 1.0 - 1e-7)
    y = target.to(p.dtype)
    m = mask.to(p.dtype)
    log_score = y * torch.log(p) + (1.0 - y) * torch.log1p(-p)
    outcome_probability = y * p + (1.0 - y) * (1.0 - p)
    norm = torch.sqrt(p.square() + (1.0 - p).square()).clamp_min(1e-9)
    spherical = outcome_probability / norm
    per_option = log_score + float(w_sph) * spherical
    return (per_option * m).sum(-1) / m.sum(-1).clamp_min(1.0)


def laya_style_rlcd_loss(
    logits: torch.Tensor,
    target_labels: torch.Tensor,
    option_mask: torch.Tensor,
    is_multi_label_task: torch.Tensor,
    *,
    sigma: float = 0.4,
    samples: int = 4,
    w_sph: float = 0.75,
    supervised_weight: float = 1.0,
    supervised_cfg: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Laya-style RLCD loss for Sage's option-scoring model.

    The policy is the model's score distribution. We sample `samples` perturbed
    logits per example, evaluate each sampled distribution with a proper reward,
    subtract the per-example group-mean reward, normalize advantages, and use the
    Gaussian log-density score-function term. A supervised CE/BCE term is retained
    as in Laya's reference objective to stabilize fine-tuning.

    For categorical single-choice rows, the exploration noise is mean-centered over
    valid options (a shared logit shift does not change softmax probabilities). For
    multi-label rows, independent Bernoulli probabilities are used and noise is not
    centered because a shared logit shift changes sigmoid probabilities.
    """
    mask, targets, multi = _validate_inputs(
        logits, target_labels, option_mask, is_multi_label_task, samples, sigma
    )
    mask_f = mask.to(dtype=torch.float32)
    base = logits.float()
    bsz, n_options = base.shape
    counts = mask_f.sum(-1, keepdim=True).clamp_min(1.0)

    # One forward pass; the group consists of stochastic decision samples from its logits.
    noise = torch.randn((samples, bsz, n_options), device=base.device, dtype=base.dtype) * float(sigma)
    noise = noise * mask_f.unsqueeze(0)
    centered = (noise - noise.sum(-1, keepdim=True) / counts.unsqueeze(0)) * mask_f.unsqueeze(0)
    noise = torch.where(multi.view(1, bsz, 1), noise, centered)

    # Reward must not backpropagate through the sampled outcome; policy gradients
    # flow through the Gaussian log-density below.
    sampled_logits = base.detach().unsqueeze(0) + noise
    with torch.no_grad():
        categorical_probs = torch.softmax(
            sampled_logits.masked_fill(~mask.unsqueeze(0), _MASKED_LOGIT), dim=-1
        )
        bernoulli_probs = torch.sigmoid(sampled_logits)
        cat_reward = categorical_proper_reward(
            categorical_probs,
            targets.unsqueeze(0),
            mask.unsqueeze(0),
            w_sph=w_sph,
        )
        multi_reward = bernoulli_proper_reward(
            bernoulli_probs,
            targets.unsqueeze(0),
            mask.unsqueeze(0),
            w_sph=w_sph,
        )
        rewards = torch.where(multi.unsqueeze(0), multi_reward, cat_reward)
        advantages = rewards - rewards.mean(dim=0, keepdim=True)
        advantages = advantages / advantages.std(unbiased=False).clamp_min(1e-6)

    # Negative Gaussian log-density up to its constant. Keep `sampled_logits` detached
    # but logits live in this expression so gradients update the policy mean.
    policy_noise = sampled_logits - logits.float().unsqueeze(0)
    log_density = -(
        policy_noise.square() * mask_f.unsqueeze(0)
    ).sum(-1) / (2.0 * float(sigma) ** 2)
    policy_gradient_loss = -(advantages * log_density).mean()

    supervised_loss = option_supervision_loss(
        logits.float(), target_labels, option_mask,
        supervised_cfg or {}, is_multi_label_task,
    )
    total = policy_gradient_loss + float(supervised_weight) * supervised_loss
    details = {
        "policy_gradient_loss": policy_gradient_loss.detach(),
        "supervised_loss": supervised_loss.detach(),
        "reward_mean": rewards.mean().detach(),
        "reward_std": rewards.std(unbiased=False).detach(),
        "advantage_std": advantages.std(unbiased=False).detach(),
    }
    return total, details
