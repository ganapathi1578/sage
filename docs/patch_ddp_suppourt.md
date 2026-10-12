# Sage 3-GPU DDP patch

This patch adds single-node PyTorch DistributedDataParallel (DDP) to the current Sage training code and shards the existing `QAParquetStream` deterministically across ranks.

## Included changes

- `scripts/train.py`: initialize NCCL from `torchrun` rank environment variables; report on rank 0; cleanly destroy the process group.
- `sageqa/data/build_loaders.py`: pass rank/world size into the training stream; require `num_workers=0` with DDP.
- `sageqa/data/qa_dataset.py`: every rank reproduces the same seeded global shuffled stream, then keeps only its disjoint strided subset. The global tail is trimmed so each rank has an equal number of rows divisible by local batch size times gradient accumulation.
- `sageqa/engine/trainer.py`: DDP model wrapper, `no_sync()` during gradient accumulation, global training metric reduction, rank-0-only full validation, and rank-0 checkpoint writing using unwrapped model state dicts (compatible with `scripts.evaluate`).

## Before applying

1. Stop every existing training process.
2. Back up the existing four files and ensure your working tree is clean.
3. The patch is intended for the `dev-gana` code layout whose loader uses `QAParquetStream` and `sageqa/engine/trainer.py`. Review your current diff before committing.

## Apply

From the Sage repository root, copy the four included files over their matching paths. Then run:

```bash
python -m py_compile scripts/train.py sageqa/data/build_loaders.py sageqa/data/qa_dataset.py sageqa/engine/trainer.py
pytest -q
```

A successful unit-test run does not replace the three-GPU smoke test below.

## Recommended launch

With three GPUs, per-GPU batch size 2 and accumulation 11, the global effective batch is `3 * 2 * 11 = 66` (near the target 64). Accumulation 32 would mean global effective batch 192.

```bash
RUN_NAME=llama_bidir_full_100k_b2_acc11_10ep_seed42_ddp3_globalshuffle
mkdir -p logs/training
LOG="logs/training/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

OMP_NUM_THREADS=1 nohup torchrun --standalone --nnodes=1 --nproc_per_node=3 \
  -m scripts.train \
  --config configs/training/experiments/llama_10k.yaml \
  --scale 100k \
  --set training.batch_size=2 \
  --set training.gradient_accumulation_steps=11 \
  --set training.epochs=10 \
  --set training.max_steps_per_epoch=null \
  --set training.num_workers=0 \
  --set data.shuffle_buffer_rows=100000 \
  --run-name "$RUN_NAME" \
  > "$LOG" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "logs/training/${RUN_NAME}.pid"
echo "PID=$PID"
echo "LOG=$LOG"
```

Immediately verify the log reports global train rows close to 80,000, rows/rank close to 26,664, `world_size=3`, and the correct 100k scale. Use `nvidia-smi` and confirm a Python process has allocated memory on GPUs 0, 1, and 2.

## Important behavior and limitations

- `shuffle_buffer_rows=100000` gives a full-buffer shuffle of an 80,000-row training split, but triples raw Parquet scanning/buffer memory across the three processes. Check host RAM first (`free -h`). If RAM is insufficient, use a smaller buffer (e.g. 16,384), knowing that shuffle is then approximate rather than a full permutation.
- The distributed sharder drops the global tail to ensure equal numbers of examples per rank and complete accumulation groups. With 80,000 rows, world size 3, local batch 2, accumulation 11, it uses 79,992 rows per epoch (8 rows dropped).
- Global shuffling makes task types/families likely to mix in each accumulation group, but does not guarantee exact per-family balance per optimizer update. Strict balance needs a task-stratified sampler.
- Validation is evaluated once on rank 0, then the metrics are broadcast to the other ranks.
- DDP checkpoint files save the unwrapped model state so the existing evaluation script can load them without `module.` prefixes.
- Run a brief smoke test first (e.g. `--set training.epochs=1 --set training.max_steps_per_epoch=22`, accumulation=11) and verify all three GPUs are used before starting the full 10-epoch job. Since the output directory contains checkpoints/logs once created, use a unique run name for the final run.
