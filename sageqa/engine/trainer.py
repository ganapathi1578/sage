"""Reusable supervised training loop shared by all registered architectures."""
from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import save_checkpoint, write_run_metadata
from .evaluator import evaluate, move_batch
from .metrics import MetricAccumulator
from .losses import option_supervision_loss


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _make_scheduler(optimizer, total_steps: int, cfg: dict[str, Any]):
    schedule_name = str(cfg.get("scheduler", "cosine")).lower()
    warmup = int(cfg.get("warmup_steps", 0))
    if total_steps <= 0 or schedule_name == "none":
        return None

    def lr_lambda(step: int):
        if warmup > 0 and step < warmup:
            return max(1e-8, (step + 1) / warmup)
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
        if schedule_name == "linear":
            return 1.0 - progress
        if schedule_name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unknown scheduler={schedule_name!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _sync_metric_accumulator(acc: MetricAccumulator, device: torch.device) -> None:
    """Aggregate rank-local training metrics so the epoch summary is global."""
    if not _is_distributed():
        return

    fields = (
        "examples", "loss_sum", "top1_hits", "single_examples", "single_correct",
        "multi_examples", "multi_nonempty_targets", "multi_empty_targets",
        "multi_top1_hits", "multi_exact_match", "tp", "fp", "fn",
    )
    values = torch.tensor(
        [float(getattr(acc, name)) for name in fields],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    integer_fields = set(fields) - {"loss_sum"}
    for name, value in zip(fields, values.cpu().tolist()):
        setattr(acc, name, int(round(value)) if name in integer_fields else float(value))

    local_groups = {key: dict(value) for key, value in acc.groups.items()}
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_groups)
    merged: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "top1_hits": 0})
    for rank_groups in gathered:
        for name, values_by_group in (rank_groups or {}).items():
            merged[name]["count"] += int(values_by_group.get("count", 0))
            merged[name]["top1_hits"] += int(values_by_group.get("top1_hits", 0))
    acc.groups = merged


def _rank_zero_eval(model, val_loader, device, threshold, loss_cfg):
    """Evaluate the complete validation split once, on rank 0, then share metrics."""
    if not _is_distributed():
        return evaluate(model, val_loader, device, threshold=threshold, loss_cfg=loss_cfg)

    dist.barrier()
    payload = [None]
    if dist.get_rank() == 0:
        payload[0] = evaluate(model, val_loader, device, threshold=threshold, loss_cfg=loss_cfg)
    # With NCCL, each process has already selected its local CUDA device.
    dist.broadcast_object_list(payload, src=0)
    dist.barrier()
    return payload[0]


def train(
    cfg: dict[str, Any], repo_root: Path, train_loader, train_ds, train_collator,
    val_loader, val_collator, model,
) -> Path:
    distributed = _is_distributed()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))

    seed = int(cfg.get("seed", 42))
    # Model weights were initialized from the same seed in every process;
    # rank-specific seeds now decorrelate dropout/other training randomness.
    seed_everything(seed + rank)
    tcfg = cfg["training"]

    device_name = str(tcfg.get("device", "auto"))
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device requests CUDA but CUDA is unavailable")

    base_model = model.to(device)
    if distributed:
        model = DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
    else:
        model = base_model

    out_root = Path(cfg.get("output_root", "outputs/experiments"))
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    run_name = str(cfg.get("experiment", {}).get("name", "experiment"))
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if any((run_dir / name).exists() for name in ("metrics.jsonl", "last_checkpoint.pt", "best_checkpoint.pt")) and not tcfg.get("resume_checkpoint"):
        raise RuntimeError(
            f"Run output already exists: {run_dir}. Use a new experiment.name for a fresh run, "
            "or set training.resume_checkpoint to last_checkpoint.pt to resume deliberately."
        )
    if rank == 0:
        write_run_metadata(run_dir, cfg, base_model.model_summary(), repo_root)
    if distributed:
        dist.barrier()

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(tcfg.get("learning_rate", 3e-4)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        betas=tuple(tcfg.get("betas", [0.9, 0.999])),
    )
    epochs = int(tcfg.get("epochs", 3))
    accum = max(1, int(tcfg.get("gradient_accumulation_steps", 1)))
    max_steps_per_epoch = tcfg.get("max_steps_per_epoch")
    nominal_batches = len(train_loader)
    if max_steps_per_epoch is not None:
        nominal_batches = min(nominal_batches, int(max_steps_per_epoch))
    total_updates = epochs * max(1, math.ceil(nominal_batches / accum))
    scheduler = _make_scheduler(optimizer, total_updates, tcfg)

    amp_mode = str(tcfg.get("amp", "none")).lower()
    amp_enabled = device.type == "cuda" and amp_mode in {"fp16", "bf16"}
    amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_mode == "fp16"))
    best_metric = -1.0
    start_epoch = 0
    global_step = 0
    resume = tcfg.get("resume_checkpoint")
    if resume:
        checkpoint = torch.load(Path(resume), map_location=device, weights_only=False)
        previous_cfg = checkpoint.get("config", {})
        for contract_key in ("model", "data"):
            if previous_cfg.get(contract_key) != cfg.get(contract_key):
                raise ValueError(
                    f"Cannot resume: checkpoint {contract_key} configuration differs from current config. "
                    "Use a new experiment name for a new architecture/scale/cache contract."
                )
        old_training = previous_cfg.get("training", {})
        new_training = cfg.get("training", {})
        for objective_key, legacy_default in (("target_loss", "single_cross_entropy"), ("multi_label_threshold", 0.5)):
            old_value = old_training.get(objective_key, legacy_default)
            new_value = new_training.get(objective_key, "mixed_ce_bce" if objective_key == "target_loss" else 0.5)
            if old_value != new_value:
                raise ValueError(
                    f"Cannot resume: checkpoint training.{objective_key}={old_value!r} differs from "
                    f"current value {new_value!r}. Use a new experiment name or restore the old objective."
                )
        base_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        rng = checkpoint.get("rng_state")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and rng.get("cuda") is not None and not distributed:
                torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_metric = float(checkpoint.get("best_metric", -1.0))

    metric_path = run_dir / "metrics.jsonl"
    max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
    print_every = max(0, int(cfg.get("logging", {}).get("print_every_steps", 20)))
    best_path = run_dir / "best_checkpoint.pt"
    last_path = run_dir / "last_checkpoint.pt"
    patience_value = tcfg.get("early_stopping_patience")
    patience = None if patience_value is None else max(1, int(patience_value))
    bad_epochs = 0

    for epoch in range(start_epoch, epochs):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        model.train()
        metric_threshold = float(tcfg.get("multi_label_threshold", 0.5))
        train_metrics = MetricAccumulator(threshold=metric_threshold)
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            if max_steps_per_epoch is not None and batch_idx >= int(max_steps_per_epoch):
                break
            do_step = ((batch_idx + 1) % accum == 0) or ((batch_idx + 1) == nominal_batches)
            batch = move_batch(batch, device)
            sync_context = model.no_sync() if distributed and not do_step else nullcontext()
            # DDP requires both forward and backward to be inside no_sync() on
            # accumulation microbatches that do not trigger an optimizer update.
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits = model(batch).float()
                    targets = batch["target_labels"]
                    option_mask = batch["option_mask"]
                    loss = option_supervision_loss(
                        logits, targets, option_mask, tcfg, batch.get("is_multi_label_task")
                    )
                    scaled_loss = loss / accum
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            if do_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

            train_metrics.update(
                logits.detach(), targets.detach(), option_mask.detach(), loss.detach(),
                batch.get("metadata"), batch.get("is_multi_label_task"),
            )
            if rank == 0 and print_every and (batch_idx + 1) % print_every == 0:
                running = train_metrics.compute()
                print(
                    f"  step {batch_idx + 1}/{nominal_batches} "
                    f"(rank0 shard) | loss={running['loss']:.4f} "
                    f"acc={running['accuracy']:.4f} lr={optimizer.param_groups[0]['lr']:.3g}",
                    flush=True,
                )

        _sync_metric_accumulator(train_metrics, device)
        train_result = train_metrics.compute()
        elapsed = time.perf_counter() - t0
        val_result = _rank_zero_eval(
            base_model, val_loader, device, metric_threshold, tcfg
        )

        current_selection_score = float(val_result["selection_score"])
        is_best = current_selection_score > best_metric
        if is_best:
            next_best_metric = current_selection_score
        else:
            next_best_metric = best_metric

        if rank == 0:
            record = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "seconds": elapsed,
                "train": train_result,
                "val": val_result,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            with metric_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            save_checkpoint(
                last_path, model=base_model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, epoch=epoch, global_step=global_step,
                best_metric=next_best_metric, cfg=cfg, repo_root=repo_root,
            )
            if is_best:
                save_checkpoint(
                    best_path, model=base_model, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler, epoch=epoch, global_step=global_step,
                    best_metric=next_best_metric, cfg=cfg, repo_root=repo_root,
                )
            print(
                f"epoch {epoch + 1}/{epochs} | train_top1_hit={train_result['accuracy']:.4f} "
                f"train_loss={train_result['loss']:.4f} val_top1_hit={val_result['accuracy']:.4f} "
                f"val_selection={val_result['selection_score']:.4f} val_loss={val_result['loss']:.4f} "
                f"single_acc={val_result['single_choice_accuracy']} "
                f"multi_f1={val_result['multi_label_f1']} steps={global_step} time={elapsed:.1f}s",
                flush=True,
            )

        best_metric = next_best_metric
        if is_best:
            bad_epochs = 0
        else:
            bad_epochs += 1
        if distributed:
            # Do not begin a new epoch while rank 0 is still writing checkpoints.
            dist.barrier()
        if patience is not None and bad_epochs >= patience:
            if rank == 0:
                print(f"Early stopping after {bad_epochs} validation epochs without improving selection score.")
            break

    train_collator.close()
    val_collator.close()
    if rank == 0:
        print(f"Best checkpoint: {best_path}")
    return best_path
