"""Target conversion for mixed single-choice and multi-label QA tasks."""
from __future__ import annotations

from typing import Any


_FALSE_STATES = {"false", "0", "no", "negative", "false_state", "false-state"}
_DEFAULT_MULTI_LABEL_TYPES = {"multi_label", "multi-label", "multilabel", "multi_select", "multi-select", "select_all"}


def _normalise_name(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_") if value is not None else ""


def target_mode_for_record(record: dict[str, Any], target_cfg: dict[str, Any] | None = None) -> str:
    """Resolve supervision semantics from the row's task_type, not positive count.

    A multi-label task may have zero, one, or many positive options. All other
    task types use the single-choice contract and must have exactly one positive.
    """
    cfg = target_cfg or {}
    task_type_column = str(cfg.get("task_type_column", "task_type"))
    observed = _normalise_name(record.get(task_type_column))
    configured_types = cfg.get("multi_label_task_types", sorted(_DEFAULT_MULTI_LABEL_TYPES))
    if isinstance(configured_types, str):
        configured_types = [configured_types]
    multi_types = {_normalise_name(item) for item in configured_types}
    return "multi_label" if observed in multi_types else "single_choice"


def _is_false_truth_state(value: Any, false_values: set[str]) -> bool:
    if value is False:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 0
    return _normalise_name(value) in false_values


def resolve_target(
    value: Any,
    option_ids: list[Any],
    target_cfg: dict[str, Any],
    record: dict[str, Any],
) -> list[int]:
    """Return a multi-hot target vector aligned to the candidate option order.

    ``task_type`` determines semantics. Multi-label rows may have an empty
    target, but by default such rows must also state ``truth_state=FALSE``.
    Single-choice rows must have exactly one positive target.
    """
    cfg = target_cfg or {}
    fmt = str(cfg.get("format", "auto")).lower()
    k = len(option_ids)
    if k <= 0:
        raise ValueError("An example must contain at least one answer option")
    mode = target_mode_for_record(record, cfg)

    def one_hot(index: int) -> list[int]:
        if not 0 <= index < k:
            raise ValueError(f"Target index {index} outside available option range [0, {k})")
        return [1 if i == index else 0 for i in range(k)]

    if fmt == "correct_text":
        raw_options = record.get(str(cfg.get("options_text_column", "options")))
        if not isinstance(raw_options, (list, tuple)) or not isinstance(value, str):
            raise ValueError("target.format=correct_text needs a text target and options_text_column list")
        matches = [i for i, option in enumerate(raw_options) if str(option) == value]
        if len(matches) != 1:
            raise ValueError(f"Correct answer text matched {len(matches)} options; expected exactly one")
        target = one_hot(matches[0])
    elif fmt in {"index", "auto"} and isinstance(value, (int, float)) and not isinstance(value, bool):
        raw_index = int(value)
        if float(value) != raw_index:
            raise ValueError(f"Target index must be an integer, got {value!r}")
        target = one_hot(raw_index - int(cfg.get("index_base", 0)))
    else:
        if fmt not in {"one_hot", "option_labels", "multi_hot", "auto"}:
            raise ValueError(f"Unsupported target.format={fmt!r}")
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"Target column is not a per-option list (format={fmt!r}, type={type(value).__name__}). "
                "Inspect the labels column and set data.target.format explicitly."
            )
        if len(value) != k:
            raise ValueError(f"Label list length {len(value)} differs from option count {k}")

        positive_value = cfg.get("positive_value", 1)
        positive_strings = {"true", "yes", "correct", "positive"}
        target = []
        for label in value:
            if isinstance(label, bool):
                positive = label
            elif isinstance(label, (int, float)):
                positive = label == positive_value
            elif isinstance(label, str):
                positive = label.strip().lower() in positive_strings or label.strip() == str(positive_value)
            else:
                positive = False
            target.append(1 if positive else 0)

    positives = sum(target)
    if mode == "single_choice":
        if positives != 1:
            raise ValueError(
                f"Single-choice task {record.get(str(cfg.get('task_type_column', 'task_type')))!r} "
                f"must have exactly one positive option; found {positives}. target={value!r}"
            )
        return target

    # Multi-label: zero positives is meaningful only for a known false state by default.
    if positives == 0 and bool(cfg.get("require_false_truth_state_for_empty", True)):
        truth_col = str(cfg.get("truth_state_column", "truth_state"))
        false_values = {_normalise_name(v) for v in cfg.get("false_truth_values", sorted(_FALSE_STATES))}
        truth_value = record.get(truth_col)
        if not _is_false_truth_state(truth_value, false_values):
            raise ValueError(
                f"Empty target is allowed for multi-label tasks only when {truth_col} is explicitly false; "
                f"got {truth_value!r}. task_type={record.get(str(cfg.get('task_type_column', 'task_type')))!r}"
            )
    return target
