# Sage streaming shuffle-buffer update

## What changes

- Keeps streaming Parquet reads and existing target conversion logic.
- Shuffles shard order and rows within each Arrow batch for training.
- Adds a bounded replacement buffer (default 16,384 rows) that mixes examples across Arrow batch boundaries.
- Uses the existing `set_epoch()` and seed formula for deterministic epoch-specific order.
- Validation/test remain sequential because `build_loaders.py` sets `shuffle=False` for non-training splits.

This is a bounded streaming shuffle, not a uniform global permutation of every row. It uses additional memory proportional to `shuffle_buffer_rows` per DataLoader worker.

## Apply from `/data/dharmendra/sage`

Back up the existing file first:

```bash
cp sageqa/data/qa_dataset.py sageqa/data/qa_dataset.py.bak.$(date +%Y%m%d_%H%M%S)
```

Copy the included file to `sageqa/data/qa_dataset.py`, then add this setting under `data:` in `configs/training/base.yaml`:

```yaml
  parquet_read_batch_rows: 2048
  shuffle_buffer_rows: 16384
```

If `parquet_read_batch_rows` is already present, add only the second line at the same indentation. The code defaults to 16,384 when the setting is omitted.

Validate syntax and the test suite:

```bash
python -m py_compile sageqa/data/qa_dataset.py
pytest -q
```

The change applies only to new training processes. An already-running process has already imported the old Python module; restart it for the update to take effect.
