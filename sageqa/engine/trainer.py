"""Reusable supervised training loop shared by all registered architectures."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import save_checkpoint, write_run_metadata
from .evaluator import evaluate, move_batch
from .metrics import MetricAccumulator


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


def train(cfg: dict[str, Any], repo_root: Path, train_loader, train_ds, train_collator, val_loader, val_collator, model) -> Path:
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    tcfg = cfg["training"]
    device_name = str(tcfg.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device requests CUDA but CUDA is unavailable")
    model.to(device)
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
    write_run_metadata(run_dir, cfg, model.model_summary(), repo_root)

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
        model.load_state_dict(checkpoint["model"])
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
            if torch.cuda.is_available() and rng.get("cuda") is not None:
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
        train_metrics = MetricAccumulator()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        seen_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            if max_steps_per_epoch is not None and batch_idx >= int(max_steps_per_epoch):
                break
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(batch).float()
                targets = batch["target_index"]
                loss = F.cross_entropy(logits, targets)
                scaled_loss = loss / accum
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            do_step = ((batch_idx + 1) % accum == 0) or (batch_idx + 1 == nominal_batches)
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
            train_metrics.update(logits.detach(), targets.detach(), loss.detach(), batch.get("metadata"))
            seen_batches += 1
            if print_every and (batch_idx + 1) % print_every == 0:
                running = train_metrics.compute()
                print(f"  step {batch_idx + 1}/{nominal_batches} | loss={running['loss']:.4f} "
                      f"acc={running['accuracy']:.4f} lr={optimizer.param_groups[0]['lr']:.3g}", flush=True)
        train_result = train_metrics.compute()
        val_result = evaluate(model, val_loader, device)
        elapsed = time.perf_counter() - t0
        record = {
            "epoch": epoch + 1, "global_step": global_step, "seconds": elapsed,
            "train": train_result, "val": val_result,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        with metric_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        save_checkpoint(last_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        epoch=epoch, global_step=global_step, best_metric=max(best_metric, val_result["accuracy"]), cfg=cfg, repo_root=repo_root)
        is_best = val_result["accuracy"] > best_metric
        if is_best:
            best_metric = float(val_result["accuracy"])
            bad_epochs = 0
            save_checkpoint(best_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=epoch, global_step=global_step, best_metric=best_metric, cfg=cfg, repo_root=repo_root)
        else:
            bad_epochs += 1
        print(f"epoch {epoch + 1}/{epochs} | train_acc={train_result['accuracy']:.4f} train_loss={train_result['loss']:.4f} "
              f"val_acc={val_result['accuracy']:.4f} val_loss={val_result['loss']:.4f} steps={global_step} time={elapsed:.1f}s")
        if patience is not None and bad_epochs >= patience:
            print(f"Early stopping after {bad_epochs} validation epochs without improving accuracy.")
            break
    train_collator.close()
    val_collator.close()
    print(f"Best checkpoint: {best_path}")
    return best_path
