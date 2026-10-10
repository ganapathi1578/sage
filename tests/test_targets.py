import pytest

from sageqa.data.targets import resolve_target, target_mode_for_record


CFG = {
    "format": "auto",
    "positive_value": 1,
    "task_type_column": "task_type",
    "multi_label_task_types": ["multi_label"],
    "require_false_truth_state_for_empty": True,
    "truth_state_column": "truth_state",
}


def test_single_positive_vector_is_preserved_as_one_hot():
    row = {"task_type": "single_choice"}
    assert resolve_target([0, 1, 0], [11, 12, 13], CFG, row) == [0, 1, 0]


def test_multi_positive_vector_is_supported_only_for_multilabel_tasks():
    row = {"task_type": "multi_label", "truth_state": "TRUE"}
    assert resolve_target([0, 1, 0, 1], [11, 12, 13, 14], CFG, row) == [0, 1, 0, 1]
    with pytest.raises(ValueError, match="Single-choice task.*exactly one"):
        resolve_target([0, 1, 0, 1], [11, 12, 13, 14], CFG, {"task_type": "single_choice"})


def test_scalar_index_becomes_one_hot_for_single_choice():
    assert resolve_target(2, [11, 12, 13], CFG, {"task_type": "single_choice"}) == [0, 0, 1]


def test_target_length_must_match_option_count():
    with pytest.raises(ValueError, match="differs from option count"):
        resolve_target([0, 1], [11, 12, 13], CFG, {"task_type": "single_choice"})


def test_zero_positive_is_valid_for_explicit_false_multilabel_target():
    row = {"task_type": "multi_label", "truth_state": "FALSE"}
    assert resolve_target([0, 0, 0], [11, 12, 13], CFG, row) == [0, 0, 0]


def test_zero_positive_multilabel_requires_false_truth_state_by_default():
    with pytest.raises(ValueError, match="explicitly false"):
        resolve_target([0, 0, 0], [11, 12, 13], CFG, {"task_type": "multi_label", "truth_state": "TRUE"})


def test_zero_positive_is_invalid_for_single_choice():
    with pytest.raises(ValueError, match="exactly one positive"):
        resolve_target([0, 0, 0], [11, 12, 13], CFG, {"task_type": "single_choice"})


def test_task_mode_comes_from_task_type_not_label_count():
    assert target_mode_for_record({"task_type": "multi_label"}, CFG) == "multi_label"
    assert target_mode_for_record({"task_type": "single_choice"}, CFG) == "single_choice"
