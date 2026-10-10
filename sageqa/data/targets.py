"""Target conversion for mixed single-answer and multi-answer QA."""
from __future__ import annotations

from typing import Any


def resolve_target(value: Any, option_ids: list[Any], target_cfg: dict[str, Any], record: dict[str, Any]) -> list[int]:
    """Return one multi-hot vector for every QA example.

    Scalar option indices and correct-text labels become one-hot vectors.
    Per-option label vectors may contain one *or multiple* positive options.
    This makes the data contract work for both single-answer and select-all tasks.
    """
    fmt = str(target_cfg.get("format", "auto")).lower()
    k = len(option_ids)
    if k <= 0:
        raise ValueError("An example must contain at least one answer option")

    def one_hot(index: int) -> list[int]:
        if not 0 <= index < k:
            raise ValueError(f"Target index {index} outside available option range [0, {k})")
        return [1 if i == index else 0 for i in range(k)]

    if fmt == "correct_text":
        raw_options = record.get(str(target_cfg.get("options_text_column", "options")))
        if not isinstance(raw_options, (list, tuple)) or not isinstance(value, str):
            raise ValueError("target.format=correct_text needs a text target and options_text_column list")
        matches = [i for i, option in enumerate(raw_options) if str(option) == value]
        if len(matches) != 1:
            raise ValueError(f"Correct answer text matched {len(matches)} options; expected exactly one")
        return one_hot(matches[0])

    if fmt in {"index", "auto"} and isinstance(value, (int, float)) and not isinstance(value, bool):
        raw_index = int(value)
        if float(value) != raw_index:
            raise ValueError(f"Target index must be an integer, got {value!r}")
        idx = raw_index - int(target_cfg.get("index_base", 0))
        return one_hot(idx)

    if fmt not in {"one_hot", "option_labels", "multi_hot", "auto"}:
        raise ValueError(f"Unsupported target.format={fmt!r}")
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"Target column is not a per-option list (format={fmt!r}, type={type(value).__name__}). "
            "Inspect the original labels column and set data.target.format explicitly."
        )
    if len(value) != k:
        raise ValueError(f"Label list length {len(value)} differs from option count {k}")

    positive_value = target_cfg.get("positive_value", 1)
    positive_strings = {"true", "yes", "correct", "positive"}
    target: list[int] = []
    for label in value:
        if isinstance(label, bool):
            is_positive = label
        elif isinstance(label, (int, float)):
            is_positive = label == positive_value
        elif isinstance(label, str):
            is_positive = label.strip().lower() in positive_strings or label.strip() == str(positive_value)
        else:
            is_positive = False
        target.append(1 if is_positive else 0)

    positives = sum(target)
    if positives == 0:
        raise ValueError(
            f"No positive answer labels found for {k} options; target={value!r}. "
            "Each example must have at least one correct option."
        )
    # One positive is single-answer; >1 is a valid multi-answer example.
    return target

