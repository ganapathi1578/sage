import pytest
import torch

from sageqa.engine.losses import option_supervision_loss
from sageqa.engine.metrics import MetricAccumulator


def test_mixed_ce_bce_accepts_single_and_multi_positive_rows_and_ignores_padding():
    logits = torch.tensor([[0.2, 2.0, -0.3, -100.0], [1.4, 0.9, -0.2, 100.0]], requires_grad=True)
    targets = torch.tensor([[0, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.float32)
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)

    loss = option_supervision_loss(logits, targets, valid, {"target_loss": "mixed_ce_bce"})
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    # Padded logits are excluded from the loss graph.
    assert logits.grad[:, 3].abs().max().item() == 0.0


def test_bce_ablation_supports_single_and_multi_answer_rows():
    logits = torch.tensor([[0.5, 1.2, -0.2], [0.7, -0.3, 0.9]], requires_grad=True)
    targets = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.float32)
    valid = torch.ones_like(targets, dtype=torch.bool)
    loss = option_supervision_loss(logits, targets, valid, {"target_loss": "bce"})
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_loss_rejects_examples_without_any_correct_option():
    logits = torch.zeros(1, 3)
    targets = torch.tensor([[0.0, 0.0, 0.0]])
    valid = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    with pytest.raises(ValueError, match="no positive target"):
        option_supervision_loss(logits, targets, valid)


def test_metrics_report_single_accuracy_and_multilabel_scores():
    logits = torch.tensor([[0.0, 3.0, 1.0, -10.0], [3.0, 2.0, -1.0, 100.0]])
    targets = torch.tensor([[0, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.float32)
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)
    loss = option_supervision_loss(logits, targets, valid)

    metrics = MetricAccumulator(threshold=0.5)
    metrics.update(logits, targets, valid, loss)
    out = metrics.compute()

    assert out["examples"] == 2
    assert out["single_choice_examples"] == 1
    assert out["multi_answer_examples"] == 1
    assert out["single_choice_accuracy"] == 1.0
    assert out["multi_label_precision"] is not None
    assert out["multi_label_recall"] is not None
    assert out["multi_label_f1"] is not None
    assert 0.0 <= out["selection_score"] <= 1.0
