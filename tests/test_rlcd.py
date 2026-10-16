import unittest

import torch

from sageqa.engine.rlcd import (
    bernoulli_proper_reward,
    categorical_proper_reward,
    laya_style_rlcd_loss,
)


class RlcdObjectiveTests(unittest.TestCase):
    def test_categorical_reward_prefers_more_honest_target_probability(self):
        mask = torch.tensor([[True, True]])
        target = torch.tensor([[1.0, 0.0]])
        good = categorical_proper_reward(torch.tensor([[0.9, 0.1]]), target, mask)
        weak = categorical_proper_reward(torch.tensor([[0.6, 0.4]]), target, mask)
        self.assertGreater(float(good.item()), float(weak.item()))

    def test_bernoulli_reward_respects_positive_and_negative_targets(self):
        mask = torch.tensor([[True, True]])
        target = torch.tensor([[1.0, 0.0]])
        good = bernoulli_proper_reward(torch.tensor([[0.9, 0.1]]), target, mask)
        weak = bernoulli_proper_reward(torch.tensor([[0.6, 0.4]]), target, mask)
        self.assertGreater(float(good.item()), float(weak.item()))

    def test_mixed_rlcd_loss_has_finite_gradients_and_handles_padding(self):
        torch.manual_seed(7)
        logits = torch.tensor(
            [
                [1.0, 0.2, -0.2, -1e4],       # single-choice, 3 valid options
                [0.3, -0.4, -1e4, -1e4],      # multi-label, 2 valid options
                [-0.1, 0.8, 0.5, -0.2],       # single-choice, 4 valid options
                [0.9, -0.1, 0.2, -1e4],       # multi-label, 3 valid options
            ], requires_grad=True,
        )
        mask = torch.tensor(
            [
                [True, True, True, False],
                [True, True, False, False],
                [True, True, True, True],
                [True, True, True, False],
            ]
        )
        targets = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        multi = torch.tensor([False, True, False, True])
        loss, parts = laya_style_rlcd_loss(
            logits, targets, mask, multi,
            sigma=0.25, samples=4, w_sph=0.75, supervised_weight=1.0,
            supervised_cfg={"target_loss": "mixed_ce_bce"},
        )
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(parts["policy_gradient_loss"]))
        self.assertTrue(torch.isfinite(parts["supervised_loss"]))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_requires_at_least_two_group_samples(self):
        logits = torch.zeros(1, 2)
        target = torch.tensor([[1.0, 0.0]])
        mask = torch.ones_like(target, dtype=torch.bool)
        with self.assertRaises(ValueError):
            laya_style_rlcd_loss(
                logits, target, mask, torch.tensor([False]), samples=1,
            )


if __name__ == "__main__":
    unittest.main()
