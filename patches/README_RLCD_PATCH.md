# Sage: Laya-style RLCD patch

This is an add-on patch for the Sage source tree. Extract it from the Sage repository root; it adds new files and does not replace `scripts/train.py` or the current supervised trainer.

**RLCD here means Reinforcement Learning for Calibrated Decisions**, following the Laya approach: Gaussian exploration over candidate logits, proper-scoring rewards, a group-mean baseline, REINFORCE-style policy updates, and a supervised CE/BCE stabilizer. Sage's current candidate scorer, video-token features, text-token embeddings, option masks and target schema are retained.

Read [`docs/rlcd_laya_style.md`](docs/rlcd_laya_style.md) for the algorithm mapping, validation strategy, smoke test and launch commands. Test the objective with:

```bash
python -m unittest discover -s tests -p 'test_rlcd.py' -v
```

Do not launch training while the supervised 1M run is still using the selected GPU. Once it finishes, initialize RLCD from its `best_checkpoint.pt`, use the matching `config.resolved.yaml`, and compare models selected on validation. Keep the test split held out for final reporting.
