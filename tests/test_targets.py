import pytest

from sageqa.data.targets import resolve_target


def test_single_positive_vector_is_preserved_as_one_hot():
    assert resolve_target([0, 1, 0], [11, 12, 13], {"format": "auto", "positive_value": 1}, {}) == [0, 1, 0]


def test_multi_positive_vector_is_supported():
    assert resolve_target([0, 1, 0, 1], [11, 12, 13, 14], {"format": "auto", "positive_value": 1}, {}) == [0, 1, 0, 1]


def test_scalar_index_becomes_one_hot():
    assert resolve_target(2, [11, 12, 13], {"format": "auto", "index_base": 0}, {}) == [0, 0, 1]


def test_target_length_must_match_option_count():
    with pytest.raises(ValueError, match="differs from option count"):
        resolve_target([0, 1], [11, 12, 13], {"format": "auto"}, {})


def test_every_example_must_have_at_least_one_positive():
    with pytest.raises(ValueError, match="No positive answer labels"):
        resolve_target([0, 0, 0], [11, 12, 13], {"format": "auto", "positive_value": 1}, {})
