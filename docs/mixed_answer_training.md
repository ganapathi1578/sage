# Mixed single-answer and multi-answer targets

The generated QA corpus intentionally includes both single-correct and multi-correct questions. Prompts such as “Select all objects present at this location” can have several positive options. The training framework must not force these rows into one target index.

## Batch contract

- `logits`: floating tensor `[B, K]`, one score per candidate.
- `target_labels`: float multi-hot tensor `[B, K]`; valid options contain 0/1, and every example has at least one positive.
- `option_mask`: boolean tensor `[B, K]`; padded candidates are false.
- `target_counts`: number of correct options in each row.

The order of the candidate options and the labels is kept intact. The models remain option-order equivariant: permuting candidate options permutes their scores in the same way.

## Loss

`training.target_loss: mixed_ce_bce` is the default. Rows with exactly one correct option use cross-entropy after invalid candidates are masked out. Rows with multiple correct options use binary cross-entropy with logits, averaged over the row's valid options. The batch loss is the mean of these per-example losses. `training.target_loss: bce` is available as an ablation using BCE for all rows.

## Metrics and checkpoint selection

- `top1_any_correct`: whether the highest-scored valid option is any correct option, averaged over all examples.
- `single_choice_accuracy`: top-1 accuracy only for rows with exactly one positive.
- `multi_label_precision`, `multi_label_recall`, `multi_label_f1`: computed on multi-answer rows by applying the configured sigmoid threshold to each valid option.
- `multi_label_exact_match`: fraction of multi-answer rows whose thresholded predicted option set exactly equals the positive target set.
- `selection_score`: mean of single-choice accuracy and multi-answer F1 when both subsets exist; if only one subset exists, uses its corresponding metric.

Tune `training.multi_label_threshold` on validation data only. The test split must not be used for threshold tuning or checkpoint selection. Thresholded F1 depends on calibration, so report top-1-any-correct and the multi-label metrics together rather than interpreting one number in isolation.

## Validation

The data adapter rejects mismatches between option count and label-vector length, non-binary labels, and examples with no correct options. Padded candidates must have zero targets and are excluded by `option_mask` from loss and metric calculations.
