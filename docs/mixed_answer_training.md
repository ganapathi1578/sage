# Mixed single-choice and multi-label supervision

SageQA determines target semantics from each row's `task_type`, not merely from
how many positive labels happen to be present.

- `task_type: multi_label`: use masked binary cross-entropy for the complete
  multi-hot target vector. This applies if the row has zero, one, or several
  positive options.
- Other task types: use masked cross-entropy and require exactly one positive
  option.
- By default, an all-zero `multi_label` vector is accepted only when the row's
  `truth_state` explicitly represents false (`FALSE`, `false`, `0`, `no`, or
  `negative`). This protects against accidental empty labels on ordinary or
  unknown examples.
- Padded candidates are excluded from both objectives and all answer metrics.

The shared data contract supplies `target_labels`, `target_modes`, and the
boolean batch tensor `is_multi_label_task`. Both architectures return one logit
per option, so the architecture itself does not need to change.

## Relevant configuration

In `configs/training/base.yaml`:

```yaml
data:
  target:
    column: labels
    format: auto
    positive_value: 1
    task_type_column: task_type
    multi_label_task_types: [multi_label]
    require_false_truth_state_for_empty: true
    truth_state_column: truth_state
    false_truth_values: ["false", "0", "no", "negative"]

training:
  target_loss: mixed_ce_bce
  multi_label_threshold: 0.5
```

`mixed_ce_bce` uses cross-entropy only for single-choice rows and BCE for every
multi-label row, including zero-positive and one-positive rows. `bce` is an
optional ablation that uses BCE for every row. Use a fresh run name when
changing the objective; do not resume a checkpoint created under incompatible
loss semantics.

## Metrics

- Single-choice accuracy is computed on rows whose task type is single-choice.
- Multi-label precision/recall/F1 and exact-set match are computed on all
  `multi_label` rows, including empty targets.
- An empty predicted set exactly matches a zero-positive target.
- `multi_label_selection_score` averages multi-label micro-F1 and exact-set
  match. Including exact-set match allows correctly predicted empty answer sets
  to contribute positively even though an all-negative target has no positive
  class for F1.
- `selection_score` averages single-choice accuracy and the multi-label
  selection score when both task families are present; otherwise it uses the
  available category score.

## Validation

```bash
python -m pytest -q
python -m scripts.validate_training_data \
  --config configs/training/experiments/llama_10k.yaml \
  --scale 10k --split val --batches 4
python -m scripts.train \
  --config configs/training/experiments/llama_10k.yaml \
  --set training.max_steps_per_epoch=10 \
  --set training.epochs=1 \
  --run-name llama_bidir_mixed_targets_smoke_10k_v2
```
