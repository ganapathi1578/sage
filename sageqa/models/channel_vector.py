"""Capsule/channel-vector style decision architecture (no encryption/key handling).

Each token is represented as [channels, vector_dim]. ChannelLinear acts on the channel axis,
while attention weights use vector dot products and therefore are invariant to a common orthogonal
change of basis on the vector axis. Spatial coordinates enter as scalar channel gates and relative
attention biases; temporal RoPE rotates the channel/head axis, not the vector axis.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import CandidateScoringModel, assemble_sequence


class ChannelLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [..., C, V] -> [..., C_out, V]; no additive vector bias.
        return torch.einsum("...cv,oc->...ov", x, self.weight)


class VectorRMSNorm(nn.Module):
    """Normalize by a scalar invariant over channels and vector dimensions."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=(-2, -1), keepdim=True).add(self.eps).sqrt()
        y = x / rms.to(x.dtype)
        return y * self.weight.to(dtype=x.dtype).view(*([1] * (x.ndim - 2)), -1, 1)


class VectorSquash(nn.Module):
    """Smooth radial nonlinearity; scales each vector without changing its direction."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
        gate = torch.tanh(norm) / norm
        return x * gate.to(x.dtype)


def apply_vector_rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, base: float = 10000.0):
    """RoPE along per-head channel dimension, shared across vector components.

    q/k shapes: [N,H,L,head_channels,V]. The transform acts on head_channels so the
    vector axis remains structurally distinct.
    """
    n, h, length, hd, vdim = q.shape
    if hd % 2:
        raise ValueError(f"RoPE requires an even per-head channel count, got {hd}")
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=q.device, dtype=torch.float32) / hd))
    angles = positions.to(device=q.device, dtype=torch.float32)[:, None, :, None] * inv[None, None, None, :]
    cos, sin = angles.cos()[:, :, :, :, None].to(q.dtype), angles.sin()[:, :, :, :, None].to(q.dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        paired = x.reshape(n, h, length, hd // 2, 2, vdim)
        even, odd = paired[..., 0, :], paired[..., 1, :]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-2).flatten(3, 4)
    return rotate(q), rotate(k)


class RelativeXYBias(nn.Module):
    def __init__(self, heads: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, heads, bias=False))

    def forward(self, xy: torch.Tensor, xy_valid: torch.Tensor) -> torch.Tensor:
        # [N,L,2] -> [N,H,L,L]; only visual-patch pairs receive spatial bias.
        delta = xy[:, :, None, :] - xy[:, None, :, :]
        bias = self.net(delta).permute(0, 3, 1, 2)
        pair_mask = xy_valid[:, None, :, None] & xy_valid[:, None, None, :]
        return bias * pair_mask.to(bias.dtype)


class VectorSelfAttention(nn.Module):
    def __init__(self, channels: int, vector_dim: int, heads: int, dropout: float, rope_base: float,
                 use_rope: bool = True, use_xy_bias: bool = True, attention_type: str = "vector_multihead"):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by num_heads={heads}")
        self.channels, self.vector_dim, self.heads = channels, vector_dim, heads
        self.head_channels = channels // heads
        if self.head_channels % 2:
            raise ValueError("(channels / num_heads) must be even for temporal RoPE")
        self.q_proj = ChannelLinear(channels, channels)
        self.k_proj = ChannelLinear(channels, channels)
        self.v_proj = ChannelLinear(channels, channels)
        self.o_proj = ChannelLinear(channels, channels)
        if attention_type not in {"vector_multihead", "invariant_linear"}:
            raise ValueError(f"Unknown channel-vector attention type: {attention_type}")
        self.attention_type = attention_type
        self.xy_bias = RelativeXYBias(heads) if (use_xy_bias and attention_type == "vector_multihead") else None
        self.use_rope = bool(use_rope)
        self.dropout = float(dropout)
        self.rope_base = float(rope_base)

    def forward(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        positions: torch.Tensor,
        xy: torch.Tensor,
        xy_valid: torch.Tensor,
    ) -> torch.Tensor:
        n, length, channels, vdim = x.shape
        def project(layer: nn.Module) -> torch.Tensor:
            return layer(x).reshape(n, length, self.heads, self.head_channels, vdim).permute(0, 2, 1, 3, 4)
        q, k, v = project(self.q_proj), project(self.k_proj), project(self.v_proj)
        if self.use_rope:
            q, k = apply_vector_rope(q, k, positions, self.rope_base)
        if self.attention_type == "vector_multihead":
            scores = torch.einsum("bhicv,bhjcv->bhij", q.float(), k.float()) / ((self.head_channels * vdim) ** 0.5)
            if self.xy_bias is not None:
                scores = scores + self.xy_bias(xy.float(), xy_valid).float()
            scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1).to(v.dtype)
            if self.training and self.dropout:
                attn = torch.nn.functional.dropout(attn, p=self.dropout)
            out = torch.einsum("bhij,bhjcv->bhicv", attn, v)
        else:
            # Kernelized linear attention. Q/K features are invariant vector magnitudes;
            # the flattened time dimension enters linearly rather than forming an L x L map.
            q_feature = torch.nn.functional.elu(q.float().pow(2).mean(dim=-1).add(1e-8).sqrt()) + 1.0
            k_feature = torch.nn.functional.elu(k.float().pow(2).mean(dim=-1).add(1e-8).sqrt()) + 1.0
            k_feature = k_feature * valid[:, None, :, None].to(k_feature.dtype)
            kv = torch.einsum("bhld,bhlcv->bhdcv", k_feature, v.float())
            numerator = torch.einsum("bhld,bhdcv->bhlcv", q_feature, kv)
            normalizer = torch.einsum("bhld,bhd->bhl", q_feature, k_feature.sum(dim=2)).clamp_min(1e-6)
            out = (numerator / normalizer[:, :, :, None, None]).to(v.dtype)
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(n, length, channels, vdim)
        out = self.o_proj(out)
        return out * valid[:, :, None, None].to(out.dtype)


class VectorChannelBlock(nn.Module):
    def __init__(self, channels: int, vector_dim: int, ffn_channels: int, heads: int, dropout: float, rope_base: float,
                 use_rope: bool = True, use_xy_bias: bool = True, attention_type: str = "vector_multihead"):
        super().__init__()
        self.norm1 = VectorRMSNorm(channels)
        self.attn = VectorSelfAttention(channels, vector_dim, heads, dropout, rope_base, use_rope=use_rope,
                                       use_xy_bias=use_xy_bias, attention_type=attention_type)
        self.norm2 = VectorRMSNorm(channels)
        self.w1 = ChannelLinear(channels, ffn_channels)
        self.act = VectorSquash()
        self.w2 = ChannelLinear(ffn_channels, channels)
        self.dropout = float(dropout)

    def forward(self, x, valid, positions, xy, xy_valid):
        h = x + torch.nn.functional.dropout(
            self.attn(self.norm1(x), valid, positions, xy, xy_valid),
            p=self.dropout, training=self.training,
        )
        ffn = self.w2(self.act(self.w1(self.norm2(h))))
        h = h + torch.nn.functional.dropout(ffn, p=self.dropout, training=self.training)
        return h * valid[:, :, None, None].to(h.dtype)


class ChannelVectorScorer(CandidateScoringModel):
    """EqCipher-inspired channel/vector path with the encryption mechanism removed."""
    def __init__(self, cfg: dict[str, Any]):
        super().__init__(option_chunk_size=int(cfg.get("option_chunk_size", 1)))
        self.cfg = cfg
        self.channels = int(cfg.get("channels", 384))
        self.vector_dim = int(cfg.get("vector_dim", 8))
        self.d_model = self.channels * self.vector_dim
        self.video_dim = int(cfg.get("video_dim", 3072))
        self.text_dim = int(cfg.get("text_dim", 384))
        heads = int(cfg.get("num_heads", 8))
        dropout = float(cfg.get("dropout", 0.05))
        self.video_proj = nn.Linear(self.video_dim, self.d_model)
        self.text_proj = nn.Linear(self.text_dim, self.d_model)
        # Scalar gates carry x/y position into channels without adding a fixed vector direction.
        self.use_xy_encoding = bool(cfg.get("use_xy_encoding", True))
        self.use_temporal_rope = bool(cfg.get("use_temporal_rope", True))
        self.attention_type = str(cfg.get("attention", "vector_multihead"))
        self.xy_gate = nn.Sequential(nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, self.channels)) if self.use_xy_encoding else None
        self.markers = nn.Parameter(torch.randn(7, self.channels, self.vector_dim) * 0.02)
        self.layers = nn.ModuleList([
            VectorChannelBlock(self.channels, self.vector_dim, int(cfg.get("ffn_channels", 512)), heads, dropout, float(cfg.get("rope_base", 10000.0)),
                               use_rope=self.use_temporal_rope, use_xy_bias=self.use_xy_encoding, attention_type=self.attention_type)
            for _ in range(int(cfg.get("num_layers", 6)))
        ])
        self.norm = VectorRMSNorm(self.channels)
        self.scorer = nn.Sequential(
            nn.Linear(self.channels, max(64, self.channels // 2)), nn.SiLU(), nn.Dropout(dropout), nn.Linear(max(64, self.channels // 2), 1)
        )
        self.gradient_checkpointing = bool(cfg.get("gradient_checkpointing", False))

    def score_candidates(self, expanded: dict[str, torch.Tensor]) -> torch.Tensor:
        n = expanded["video_features"].shape[0]
        vmask = expanded["video_mask"].bool()
        vxy = expanded["video_xy"].float()
        vf = self.video_proj(expanded["video_features"])
        # Position-dependent scalar channel gating; each scalar scales the 8-vector as a unit.
        vf = vf.view(n, -1, self.channels, self.vector_dim)
        if self.xy_gate is not None:
            gate = 1.0 + 0.1 * torch.tanh(self.xy_gate(vxy))
            vf = vf * gate.unsqueeze(-1)
        query = self.text_proj(expanded["query_tokens"]).view(n, -1, self.channels, self.vector_dim)
        option = self.text_proj(expanded["option_tokens"]).view(n, -1, self.channels, self.vector_dim)
        x, valid, positions, xy, xy_valid = assemble_sequence(
            vf.reshape(n, -1, self.d_model),
            query.reshape(n, -1, self.d_model),
            option.reshape(n, -1, self.d_model),
            vmask,
            expanded["query_mask"].bool(), expanded["option_token_mask"].bool(),
            expanded["video_positions"].float(), vxy, self.markers.view(7, self.d_model),
        )
        x = x.view(n, x.shape[1], self.channels, self.vector_dim)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(layer, x, valid, positions, xy, xy_valid, use_reentrant=False)
            else:
                x = layer(x, valid, positions, xy, xy_valid)
        x = self.norm(x)
        # Magnitudes are invariant to a common orthogonal basis change on vector_dim.
        mask_magnitude = x[:, -1].float().pow(2).mean(dim=-1).add(1e-8).sqrt()
        return self.scorer(mask_magnitude).squeeze(-1)
