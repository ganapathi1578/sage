"""Explicit architecture registry."""
from __future__ import annotations

from typing import Any

from .channel_vector import ChannelVectorScorer
from .llama_bidir import LlamaBidirectionalScorer

MODEL_REGISTRY = {
    "llama_bidir": LlamaBidirectionalScorer,
    "channel_vector": ChannelVectorScorer,
}


def build_model(cfg: dict[str, Any]):
    name = str(cfg.get("name", "llama_bidir"))
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model architecture {name!r}; registered: {sorted(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[name](cfg)
    return model
