"""Common multiple-choice scoring contract and shared sequence construction."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


MARKERS = ("video_open", "video_close", "query_open", "query_close", "option_open", "option_close", "mask")


def expand_candidate_inputs(batch: dict[str, Any], start: int, end: int) -> dict[str, torch.Tensor]:
    """Repeat shared video/question inputs for each candidate in [start, end)."""
    video = batch["video_features"]
    b, t, p, c = video.shape
    kc = end - start
    def repeat_bk(x: torch.Tensor) -> torch.Tensor:
        return x[:, None].expand(b, kc, *x.shape[1:]).reshape(b * kc, *x.shape[1:])

    frame_mask = batch["video_mask"]
    visual_mask = frame_mask[:, :, None].expand(b, t, p).reshape(b, t * p)
    frame_positions = batch["video_times"][:, :, None].expand(b, t, p).reshape(b, t * p)
    xy = batch["video_xy"][:, None].expand(b, t, p, 2).reshape(b, t * p, 2)
    return {
        "video_features": repeat_bk(video).reshape(b * kc, t * p, c),
        "video_mask": repeat_bk(visual_mask),
        "video_positions": repeat_bk(frame_positions),
        "video_xy": repeat_bk(xy),
        "query_tokens": repeat_bk(batch["query_tokens"]),
        "query_mask": repeat_bk(batch["query_mask"]),
        "option_tokens": batch["option_tokens"][:, start:end].reshape(b * kc, *batch["option_tokens"].shape[2:]),
        "option_token_mask": batch["option_token_mask"][:, start:end].reshape(b * kc, *batch["option_token_mask"].shape[2:]),
        "batch_size": torch.tensor(b, device=video.device),
        "chunk_options": torch.tensor(kc, device=video.device),
    }


def assemble_sequence(
    video_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    option_tokens: torch.Tensor,
    video_mask: torch.Tensor,
    query_mask: torch.Tensor,
    option_mask: torch.Tensor,
    video_positions: torch.Tensor,
    video_xy: torch.Tensor,
    marker_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Serialize one candidate as <video>...</video><query>...</query><option>...</option><mask>.

    Returns sequence features, valid-token mask, rotary positions, XY coordinates, and
    a mask indicating which sequence positions carry actual visual patch coordinates.
    All attention remains bidirectional; masks only hide padding keys.
    """
    n, nv, d = video_tokens.shape
    nq = query_tokens.shape[1]
    no = option_tokens.shape[1]
    device, dtype = video_tokens.device, video_tokens.dtype
    m = marker_embeddings.to(device=device, dtype=dtype)

    def marker(index: int) -> torch.Tensor:
        return m[index].view(1, 1, d).expand(n, 1, d)

    pieces = [marker(0), video_tokens, marker(1), marker(2), query_tokens, marker(3), marker(4), option_tokens, marker(5), marker(6)]
    x = torch.cat(pieces, dim=1)
    one = torch.ones((n, 1), device=device, dtype=torch.bool)
    valid = torch.cat([one, video_mask, one, one, query_mask, one, one, option_mask, one, one], dim=1)

    # Video patches at the same sampled frame share the same time coordinate. Text positions
    # continue after the video interval; spatial XY is supplied independently.
    max_vpos = video_positions.masked_fill(~video_mask, 0.0).max(dim=1).values
    video_end_pos = max_vpos + 1.0
    query_start = video_end_pos + 1.0
    qpos = query_start[:, None] + torch.arange(nq, device=device, dtype=torch.float32)[None, :] + 1.0
    query_end_pos = query_start + nq + 1.0
    option_start = query_end_pos + 1.0
    opos = option_start[:, None] + torch.arange(no, device=device, dtype=torch.float32)[None, :] + 1.0
    option_end_pos = option_start + no + 1.0
    mask_pos = option_end_pos + 1.0
    pos = torch.cat([
        torch.zeros((n, 1), device=device), video_positions.float(),
        video_end_pos[:, None], query_start[:, None], qpos,
        query_end_pos[:, None], option_start[:, None], opos,
        option_end_pos[:, None], mask_pos[:, None],
    ], dim=1)

    def zeros_xy(length: int) -> torch.Tensor:
        return torch.zeros((n, length, 2), device=device, dtype=video_xy.dtype)

    def false_mask(length: int) -> torch.Tensor:
        return torch.zeros((n, length), device=device, dtype=torch.bool)

    xy = torch.cat([
        zeros_xy(1), video_xy, zeros_xy(1), zeros_xy(1), zeros_xy(nq),
        zeros_xy(1), zeros_xy(1), zeros_xy(no), zeros_xy(1), zeros_xy(1),
    ], dim=1)
    xy_valid = torch.cat([
        false_mask(1), video_mask, false_mask(1), false_mask(1), false_mask(nq),
        false_mask(1), false_mask(1), false_mask(no), false_mask(1), false_mask(1),
    ], dim=1)
    # Defensive shape assertions make serialization mistakes fail close to the source.
    if x.shape[1] != valid.shape[1] or x.shape[1] != pos.shape[1] or x.shape[1] != xy.shape[1] or x.shape[1] != xy_valid.shape[1]:
        raise RuntimeError(f"Packed sequence alignment error: x={x.shape}, valid={valid.shape}, pos={pos.shape}, xy={xy.shape}, xy_valid={xy_valid.shape}")
    return x, valid, pos, xy, xy_valid


class CandidateScoringModel(nn.Module, ABC):
    """Model API: shared inputs -> [batch, candidates] logits.

    Candidates are scored by the same network independently. Therefore permuting candidate
    options permutes output scores in the same way (option-order equivariance) without leaking
    one candidate's text into another candidate's score.
    """
    def __init__(self, option_chunk_size: int = 1):
        super().__init__()
        self.option_chunk_size = max(1, int(option_chunk_size))

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        option_mask = batch["option_mask"].bool()
        b, k = option_mask.shape
        chunks: list[torch.Tensor] = []
        for start in range(0, k, self.option_chunk_size):
            end = min(start + self.option_chunk_size, k)
            expanded = expand_candidate_inputs(batch, start, end)
            score = self.score_candidates(expanded)
            chunks.append(score.reshape(b, end - start))
        logits = torch.cat(chunks, dim=1)
        return logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)

    @abstractmethod
    def score_candidates(self, expanded: dict[str, torch.Tensor]) -> torch.Tensor:
        """Score candidate rows in a flattened [batch * candidate_chunk] layout."""
        raise NotImplementedError

    def model_summary(self) -> dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"architecture": self.__class__.__name__, "parameters": total, "trainable_parameters": trainable}
