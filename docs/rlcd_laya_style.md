# Laya-style RLCD for Sage Video QA

This is an experimental, task-adapted implementation of **RLCD = Reinforcement Learning for Calibrated Decisions**, following the public Laya method. It is not a text-generation PPO pipeline and does not require a separate reward model.

## What is retained from Laya

- The model emits scores for a fixed set of candidate decisions.
- At training time, sample multiple Gaussian-perturbed logits per example (`group_size`, default 4).
- Convert each noisy score vector to a probability distribution.
- Reward each sampled distribution using strictly proper scoring rules (log score + spherical score).
- Subtract the per-example group-mean reward and normalize advantages.
- Apply a REINFORCE-style policy-gradient term to the Gaussian log density.
- Add a supervised loss term to stabilize the policy update.
- Anneal exploration noise from `sigma_start=0.4` to `sigma_end=0.1` across epochs.

Reference implementation and description:

- Laya model hub: https://huggingface.co/convaiinnovations/laya
- Laya training objective: https://github.com/NandhaKishorM/laya/blob/main/laya/train.py
- Laya fine-tuning notebook: https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb

Laya's hub lists its model under Apache-2.0. This Sage implementation reuses the method's publicly described algorithmic structure but retains Sage's own data pipeline, candidate scorer, target schema and metric implementation.

## Sage-specific adaptation

Sage's `LlamaBidirectionalScorer` returns `[B, K]` candidate logits using video tokens and text-token embeddings. The RLCD loss does not alter this architecture.

- `single_choice`: categorical softmax distribution over valid options; categorical log+spherical reward.
- `multi_label`: independent sigmoid/Bernoulli distribution for each valid option; mean per-option Bernoulli log+spherical reward. This extension is required because Sage labels can have zero, one, or multiple positive candidates, unlike Laya's ordinary categorical choice distribution.
- Padded options are masked out of exploration, reward, and the score-function term.
- The existing mixed CE/BCE supervised loss is added to the policy-gradient term (`supervised_weight=1.0` by default).
- Ranked probability score is intentionally not enabled: Sage's current answer candidates are not guaranteed to represent ordered ordinal levels.
- TD(lambda) is not enabled: Sage's current training items are static video-question examples, not multi-turn conversation prefixes.
- Laya's act/escalate head is not included because the current Sage benchmark has no matching supervised abstain/escalate action target.

This is a research adaptation, not a claim that the current Sage task exactly matches Laya's typed-decision benchmark. The original test split must remain held out; use validation `selection_score` to pick an RLCD checkpoint, then run test evaluation once for the selected checkpoint.

## Install the patch

Extract the patch archive at the Sage repository root so these files are added without replacing the existing trainer:

- `sageqa/engine/rlcd.py`
- `scripts/train_rlcd.py`
- `scripts/evaluate_calibration.py` — ECE, Brier score and NLL on saved predictions
- `tests/test_rlcd.py`

Then run the objective tests:

```bash
python -m unittest discover -s tests -p 'test_rlcd.py' -v
```

## Safe smoke test (no optimizer steps or checkpoint writes)

Run only when GPU memory is available. The current supervised 1M run should finish before using the same GPU.

```bash
cd ~/sage
SRC="outputs/experiments/llama_bidir_full_1m_b2_acc128_2ep_seed42_val500k"

python -m scripts.train_rlcd \
  --config "$SRC/config.resolved.yaml" \
  --init-checkpoint "$SRC/best_checkpoint.pt" \
  --scale 1m \
  --run-name rlcd_smoke_only \
  --smoke-only --smoke-batches 1 \
  --set training.rlcd.group_size=4 \
  --set training.rlcd.sigma_start=0.4
```

A smoke run requires the source checkpoint to exist. It runs a forward/backward pass and exits without stepping the optimizer or writing a checkpoint. The run-name is required by the CLI but no output directory is created in smoke mode.

## Evaluate the supervised 1M baseline first

After the supervised process exits and its final checkpoint is saved, evaluate that fixed checkpoint on the test split. This records the comparison baseline; RLCD training uses train and validation only.

```bash
cd ~/sage
SUP="outputs/experiments/llama_bidir_full_1m_b2_acc128_2ep_seed42_val500k"

# First verify that the supervised run has actually stopped.
pgrep -af 'scripts.train'
# If the supervised 1M process is still listed, wait. Do not start RLCD yet.

test -f "$SUP/best_checkpoint.pt" || {
  echo "Supervised checkpoint not found: $SUP/best_checkpoint.pt" >&2
  exit 1
}

python -m scripts.evaluate \
  --config "$SUP/config.resolved.yaml" \
  --checkpoint "$SUP/best_checkpoint.pt" \
  --split test --scale 1m \
  --predictions "$SUP/test_predictions.jsonl" \
  2>&1 | tee "$SUP/test_evaluation.log"

python -m scripts.evaluate_calibration \
  --predictions "$SUP/test_predictions.jsonl" \
  --output "$SUP/test_calibration.json"
```

## Pilot run (after supervised 1M training completes)

Use the *source run's resolved configuration* to guarantee the RLCD model architecture matches the checkpoint. The values below cap a pilot at 20,000 training microbatches (not optimizer updates); with accumulation 32 this is at most about 625 updates. This is intentionally not a full epoch on the 1M stream. Because this is a one-epoch pilot, the code holds sigma at 0.25 (the end value is not reached until later epochs); use multiple epochs for the full 0.4-to-0.1 annealing schedule.

```bash
cd ~/sage
SRC="outputs/experiments/llama_bidir_full_1m_b2_acc128_2ep_seed42_val500k"
RUN="llama_bidir_full_1m_rlcd_group4_$(date +%Y%m%d_%H%M%S)"
LOG="logs/training/${RUN}.log"
mkdir -p logs/training

test -f "$SRC/best_checkpoint.pt" || {
  echo "Wait until the supervised 1M run has written best_checkpoint.pt: $SRC" >&2
  exit 1
}

# Refuse to start while the original supervised trainer is still active.
if pgrep -af '[s]cripts.train.*--scale 1m'; then
  echo "Supervised 1M training is still running; wait for it to finish first." >&2
  exit 1
fi

# Check available GPU memory before starting:
nvidia-smi

CUDA_VISIBLE_DEVICES=2 nohup python -u -m scripts.train_rlcd \
  --config "$SRC/config.resolved.yaml" \
  --init-checkpoint "$SRC/best_checkpoint.pt" \
  --scale 1m --run-name "$RUN" \
  --max-train-batches 20000 \
  --set training.epochs=1 \
  --set training.batch_size=2 \
  --set training.gradient_accumulation_steps=32 \
  --set training.learning_rate=0.00002 \
  --set training.warmup_steps=50 \
  --set training.amp=fp16 \
  --set training.rlcd.group_size=4 \
  --set training.rlcd.sigma_start=0.25 \
  --set training.rlcd.sigma_end=0.25 \
  --set training.rlcd.w_sph=0.75 \
  --set training.rlcd.supervised_weight=1.0 \
  > "$LOG" 2>&1 &

printf 'RUN=%s\nLOG=%s\nPID=%s\n' "$RUN" "$LOG" "$!"
```

Before the pilot, verify the selected supervised checkpoint, `config.resolved.yaml`, data/feature store and validation protocol. Do not launch it while the original 1M job still uses that GPU.

## Evaluate a completed RLCD checkpoint

```bash
cd ~/sage
RUN_DIR="outputs/experiments/REPLACE_WITH_RLCD_RUN_NAME"

python -m scripts.evaluate \
  --config "$RUN_DIR/config.resolved.yaml" \
  --checkpoint "$RUN_DIR/best_checkpoint.pt" \
  --split test --scale 1m \
  --predictions "$RUN_DIR/test_predictions.jsonl" \
  2>&1 | tee "$RUN_DIR/test_evaluation.log"
```

Then compute raw-probability calibration diagnostics:

```bash
python -m scripts.evaluate_calibration \
  --predictions "$RUN_DIR/test_predictions.jsonl" \
  --output "$RUN_DIR/test_calibration.json"
```

The calibration script reports top-1 ECE / multiclass Brier / NLL for single-choice rows and option-level binary ECE / Brier / NLL for multi-label rows. It does not fit a temperature. Post-hoc per-task temperature fitting is a separate calibration stage and should use a designated calibration subset, never the test set.

Do not use test metrics to pick epochs or tune RLCD hyperparameters. Compare the selected RLCD model with the supervised baseline using the exact same test split and evaluation code. Report single-choice accuracy, multi-label F1/exact match, selection score, task-family breakdowns, and calibration metrics.


## FP16 non-finite gradient handling

The trainer checks logits and the combined objective for non-finite values before backward. If a gradient norm becomes non-finite at an optimizer boundary, it discards the accumulated gradient window and reduces the FP16 GradScaler scale by half. The skipped step does not advance the optimizer scheduler or successful `global_step`. Three consecutive non-finite gradient windows stop the run rather than silently continuing. This handles transient FP16 gradient overflow; it does not mask non-finite logits or objective values, which fail immediately with the offending microbatch number. Such a failure should be investigated rather than blindly retried.
