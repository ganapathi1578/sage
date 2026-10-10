import pytest
import torch

from sageqa.engine.losses import option_supervision_loss
from sageqa.engine.metrics import MetricAccumulator


def test_loss_uses_task_type_mask_and_ignores_padding():
    logits = torch.tensor([[0.2, 2.0, -0.3, -100.0], [1.4, 0.9, -0.2, 100.0]], requires_grad=True)
    targets = torch.tensor([[0, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.float32)
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)
    is_multi = torch.tensor([False, True])

    loss = option_supervision_loss(logits, targets, valid, {"target_loss": "mixed_ce_bce"}, is_multi)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[:, 3].abs().max().item() == 0.0


def test_multi_label_task_with_exactly_one_positive_still_uses_bce():
    logits = torch.tensor([[0.0, 2.0, -1.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 1.0, 0.0]])
    valid = torch.ones_like(targets, dtype=torch.bool)
    loss = option_supervision_loss(
        logits, targets, valid, {"target_loss": "mixed_ce_bce"}, torch.tensor([True])
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(loss, expected)


def test_false_multilabel_example_with_zero_positive_targets_trains_with_bce():
    logits = torch.tensor([[2.0, -1.0, 100.0]], requires_grad=True)
    targets = torch.zeros((1, 3))
    valid = torch.tensor([[True, True, False]])
    is_multi = torch.tensor([True])

    loss = option_supervision_loss(logits, targets, valid, {"target_loss": "mixed_ce_bce"}, is_multi)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, :2], targets[:, :2])
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0] > 0  # Positive logit should be pushed down for a false option.
    assert logits.grad[0, 1] > 0  # Negative logit still pushed down, but much less.
    assert logits.grad[0, 2] == 0  # Padded option has no contribution.


def test_bce_ablation_supports_zero_single_and_multiple_positives():
    logits = torch.tensor([[0.5, 1.2, -0.2], [0.7, -0.3, 0.9]], requires_grad=True)
    targets = torch.tensor([[0, 0, 0], [1, 0, 1]], dtype=torch.float32)
    valid = torch.ones_like(targets, dtype=torch.bool)
    loss = option_supervision_loss(logits, targets, valid, {"target_loss": "bce"}, torch.tensor([True, True]))
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None


def test_single_choice_rows_reject_zero_or_multiple_positive_targets():
    valid = torch.ones((1, 3), dtype=torch.bool)
    with pytest.raises(ValueError, match="exactly one positive"):
        option_supervision_loss(torch.zeros((1, 3)), torch.zeros((1, 3)), valid, {}, torch.tensor([False]))
    with pytest.raises(ValueError, match="exactly one positive"):
        option_supervision_loss(torch.zeros((1, 3)), torch.tensor([[1, 1, 0.0]]), valid, {}, torch.tensor([False]))


def test_metrics_include_zero_positive_multilabel_rows_and_empty_set_match():
    logits = torch.tensor([[-2.0, -3.0, 99.0], [2.0, -2.0, 99.0], [0.0, 2.0, -2.0]])
    targets = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=torch.float32)
    valid = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    is_multi = torch.tensor([True, True, False])
    loss = option_supervision_loss(logits, targets, valid, {}, is_multi)
    metrics = MetricAccumulator(threshold=0.5)
    metrics.update(logits, targets, valid, loss, metadata=None, is_multi_label_task=is_multi)
    out = metrics.compute()

    assert out["single_choice_examples"] == 1
    assert out["multi_label_examples"] == 2
    assert out["multi_label_empty_target_examples"] == 1
    assert out["multi_label_nonempty_target_examples"] == 1
    assert out["multi_label_exact_match"] == 0.5
    assert out["multi_label_f1"] is not None


def test_task_semantics_not_positive_count_controls_metrics():
    # First row is multi-label with exactly one positive; still counted as multilabel.
    logits = torch.tensor([[0.0, 3.0, -1.0], [3.0, 1.0, -1.0]])
    targets = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32)
    valid = torch.ones_like(targets, dtype=torch.bool)
    is_multi = torch.tensor([True, False])
    loss = option_supervision_loss(logits, targets, valid, {}, is_multi)
    metrics = MetricAccumulator(threshold=0.5)
    metrics.update(logits, targets, valid, loss, is_multi_label_task=is_multi)
    out = metrics.compute()
    assert out["single_choice_examples"] == 1
    assert out["multi_label_examples"] == 1
