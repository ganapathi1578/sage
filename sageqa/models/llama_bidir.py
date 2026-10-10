"""LLaMA-inspired, bidirectional token mixer for multiple-choice video QA.

Uses RMSNorm, RoPE, SwiGLU and pre-norm residual blocks. Attention is BIDIRECTIONAL
(non-causal): every valid video/query/option/[MASK] token can attend to every other valid
 token. It is LLaMA-inspired, not a checkpoint-compatible LLaMA implementation.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CandidateScoringModel, assemble_sequence


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)) * self.weight.to(dtype=x.dtype)


def apply_rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, base: float = 10000.0):
    """Apply 1-D RoPE with per-token positions; q/k are [N,H,L,head_dim]."""
    n, h, length, hd = q.shape
    if hd % 2:
        raise ValueError(f"RoPE requires an even head dimension, got {hd}")
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=q.device, dtype=torch.float32) / hd))
    angles = positions.to(device=q.device, dtype=torch.float32)[:, None, :, None] * inv[None, None, None, :]
    cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        paired = x.reshape(n, h, length, hd // 2, 2)
        even, odd = paired[..., 0], paired[..., 1]
        c, s = cos, sin
        out0 = even * c - odd * s
        out1 = even * s + odd * c
        return torch.stack((out0, out1), dim=-1).flatten(-2)
    return rotate(q), rotate(k)


class XYPositionEncoding(nn.Module):
    """Learned Fourier-feature encoding added to each visual patch token."""
    def __init__(self, dim: int, frequencies: int = 8):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(frequencies, dtype=torch.float32)) * torch.pi, persistent=False)
        in_dim = 2 + frequencies * 4
        self.net = nn.Sequential(nn.Linear(in_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        xy = xy.clamp(0.0, 1.0)
        phase = xy.unsqueeze(-1) * self.freqs.to(xy.device)
        features = torch.cat([xy, phase.sin().flatten(-2), phase.cos().flatten(-2)], dim=-1)
        return self.net(features)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.value = nn.Linear(dim, hidden, bias=False)
        self.out = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.out(F.silu(self.gate(x)) * self.value(x)))


class BidirectionalRoPEAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, rope_base: float, use_rope: bool = True):
        super().__init__()
        if dim % heads:
            raise ValueError(f"model.d_model={dim} must be divisible by model.num_heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        if self.head_dim % 2:
            raise ValueError("d_model/num_heads must be even for RoPE")
        self.rope_base = float(rope_base)
        self.use_rope = bool(use_rope)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        n, length, _ = x.shape
        qkv = self.qkv(x).view(n, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.use_rope:
            q, k = apply_rope(q, k, positions, self.rope_base)
        # Boolean SDPA mask: True means key is allowed. No triangular/causal mask is used.
        attn_mask = valid_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(n, length, self.dim)
        return self.out(y) * valid_mask.unsqueeze(-1).to(y.dtype)


class LlamaBidirBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_hidden: int, dropout: float, rope_base: float, use_rope: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = BidirectionalRoPEAttention(dim, heads, dropout, rope_base, use_rope=use_rope)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), valid_mask, positions))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x * valid_mask.unsqueeze(-1).to(x.dtype)


class LlamaBidirectionalScorer(CandidateScoringModel):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__(option_chunk_size=int(cfg.get("option_chunk_size", 2)))
        self.cfg = cfg
        d = int(cfg.get("d_model", 384))
        text_dim = int(cfg.get("text_dim", 384))
        video_dim = int(cfg.get("video_dim", 3072))
        heads = int(cfg.get("num_heads", 8))
        dropout = float(cfg.get("dropout", 0.1))
        layers = int(cfg.get("num_layers", 6))
        ffn_hidden = int(cfg.get("ffn_hidden", d * 3))
        self.video_proj = nn.Linear(video_dim, d)
        self.query_proj = nn.Identity() if text_dim == d else nn.Linear(text_dim, d)
        self.option_proj = nn.Identity() if text_dim == d else nn.Linear(text_dim, d)
        self.use_xy_encoding = bool(cfg.get("use_xy_encoding", True))
        self.xy_encoder = XYPositionEncoding(d, int(cfg.get("xy_frequencies", 8))) if self.use_xy_encoding else None
        use_rope = bool(cfg.get("use_temporal_rope", True))
        self.markers = nn.Parameter(torch.randn(7, d) * 0.02)
        self.layers = nn.ModuleList([LlamaBidirBlock(d, heads, ffn_hidden, dropout, float(cfg.get("rope_base", 10000.0)), use_rope=use_rope) for _ in range(layers)])
        self.norm = RMSNorm(d)
        self.scorer = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d, 1))
        self.gradient_checkpointing = bool(cfg.get("gradient_checkpointing", False))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def score_candidates(self, expanded: dict[str, torch.Tensor]) -> torch.Tensor:
        vf = expanded["video_features"]
        vmask = expanded["video_mask"].bool()
        vpos = expanded["video_positions"].float()
        vxy = expanded["video_xy"].float()
        video = self.video_proj(vf)
        if self.xy_encoder is not None:
            video = video + self.xy_encoder(vxy)
        query = self.query_proj(expanded["query_tokens"])
        option = self.option_proj(expanded["option_tokens"])
        x, valid, positions, _, _ = assemble_sequence(
            video, query, option, vmask,
            expanded["query_mask"].bool(), expanded["option_token_mask"].bool(),
            vpos, vxy, self.markers,
        )
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(layer, x, valid, positions, use_reentrant=False)
            else:
                x = layer(x, valid, positions)
        x = self.norm(x)
        # Final token is the learned [MASK] decision token, shared across all options.
        return self.scorer(x[:, -1]).squeeze(-1)
