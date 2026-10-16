#!/usr/bin/env python3
"""Fine-tune a Sage candidate scorer with Laya-style RLCD.

Single-process version first; this reuses Sage's streaming feature loaders, model
registry, existing mixed CE/BCE labels, evaluator, metrics, and checkpoint format.
It intentionally does not share or interfere with `scripts.train`.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from sageqa.config import load_config
from sageqa.data.build_loaders import build_loader
from sageqa.engine.checkpoint import save_checkpoint, write_run_metadata
from sageqa.engine.evaluator import evaluate, move_batch
from sageqa.engine.metrics import MetricAccumulator
from sageqa.engine.rlcd import laya_style_rlcd_loss
from sageqa.engine.trainer import _make_scheduler, seed_everything
from sageqa.models.registry import build_model


def parse_value(value: str):
    return yaml.safe_load(value)


def apply_override(cfg: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be KEY=VALUE, got {expression!r}")
    key, raw = expression.split("=", 1)
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        current = node.setdefault(part, {})
        if not isinstance(current, dict):
            raise ValueError(f"Override path crosses a non-mapping value: {key}")
        node = current
    node[parts[-1]] = parse_value(raw)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device requests CUDA but CUDA is unavailable")
    return device


def _check_checkpoint_contract(cfg: dict[str, Any], ckpt: dict[str, Any], path: Path) -> None:
    source_cfg = ckpt.get("config", {})
    if source_cfg.get("model") != cfg.get("model"):
        raise ValueError(
            "The initializer checkpoint's model configuration differs from the requested model.\n"
            f"checkpoint model={source_cfg.get('model')}\nrequested model={cfg.get('model')}\n"
            "Use the resolved config saved next to that checkpoint; do not change model depth/width."
        )
    source_data = source_cfg.get("data", {})
    target_data = cfg.get("data", {})
    for key in ("target", "columns", "video_store", "text_token_store_root"):
        if key in source_data and source_data.get(key) != target_data.get(key):
            raise ValueError(
                f"Checkpoint/data contract mismatch for data.{key}: "
                f"checkpoint={source_data.get(key)!r}, requested={target_data.get(key)!r}. "
                "Do not continue until the data contract is verified."
            )
    if "model" not in ckpt:
        raise ValueError(f"Checkpoint {path} does not contain a 'model' state dict")


def _close_collators(*items) -> None:
    for item in items:
        if item is not None and hasattr(item, "close"):
            try:
                item.close()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Prefer source run's config.resolved.yaml")
    parser.add_argument("--init-checkpoint", required=True, help="Supervised best_checkpoint.pt")
    parser.add_argument("--scale", help="Optional override of data.scale (e.g. 1m)")
    parser.add_argument("--run-name", required=True, help="New, isolated RLCD output run name")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--max-updates", type=int, default=None, help="Optional optimizer-update cap")
    parser.add_argument(
        "--max-train-batches", type=int, default=None,
        help="Optional per-epoch MICRO-BATCH cap; intended for short pilots, not final runs",
    )
    parser.add_argument(
        "--smoke-only", action="store_true",
        help="Forward/backward a few batches to verify the RLCD objective; no optimizer/checkpoint writes",
    )
    parser.add_argument("--smoke-batches", type=int, default=1)
    args = parser.parse_args()

    if args.max_updates is not None and args.max_updates < 1:
        parser.error("--max-updates must be positive")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("--max-train-batches must be positive")
    if args.smoke_batches < 1:
        parser.error("--smoke-batches must be positive")

    cfg, repo_root = load_config(args.config)
    for expr in args.set:
        apply_override(cfg, expr)
    if args.scale:
        cfg.setdefault("data", {})["scale"] = args.scale
    cfg.setdefault("experiment", {})["name"] = args.run_name
    cfg.setdefault("training", {})
    tcfg = cfg["training"]
    rcfg = tcfg.setdefault("rlcd", {})
    rcfg.setdefault("group_size", 4)
    rcfg.setdefault("sigma_start", 0.4)
    rcfg.setdefault("sigma_end", 0.1)
    rcfg.setdefault("w_sph", 0.75)
    rcfg.setdefault("supervised_weight", 1.0)

    group_size = int(rcfg["group_size"])
    if group_size < 2:
        raise ValueError("training.rlcd.group_size must be >= 2")
    if float(rcfg["sigma_start"]) <= 0 or float(rcfg["sigma_end"]) <= 0:
        raise ValueError("training.rlcd sigma values must be > 0")
    epochs = int(tcfg.get("epochs", 1))
    if epochs < 1:
        raise ValueError("training.epochs must be >= 1")

    init_path = Path(args.init_checkpoint).expanduser().resolve()
    if not init_path.is_file():
        raise FileNotFoundError(f"Initializer checkpoint not found: {init_path}")
    checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
    _check_checkpoint_contract(cfg, checkpoint, init_path)

    out_root = Path(cfg.get("output_root", "outputs/experiments"))
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    run_dir = out_root / args.run_name
    if not args.smoke_only:
        if any((run_dir / n).exists() for n in ("metrics.jsonl", "last_checkpoint.pt", "best_checkpoint.pt")):
            raise FileExistsError(f"RLCD output already exists: {run_dir}; use a new --run-name")
        run_dir.mkdir(parents=True, exist_ok=True)
    rcfg["init_checkpoint"] = str(init_path)
    rcfg["method"] = "laya_style_rlcd"
    rcfg["group_size"] = group_size

    device = _resolve_device(str(tcfg.get("device", "auto")))
    seed_everything(int(cfg.get("seed", 42)))
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    train_loader = train_ds = train_collator = None
    val_loader = val_collator = None
    try:
        train_loader, train_ds, train_collator = build_loader(cfg, repo_root, "train")
        if args.smoke_only:
            amp_mode = str(tcfg.get("amp", "none")).lower()
            amp_enabled = device.type == "cuda" and amp_mode in {"fp16", "bf16"}
            amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
            model.train()
            n_seen = 0
            for batch in train_loader:
                batch = move_batch(batch, device)
                model.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits = model(batch)
                loss, parts = laya_style_rlcd_loss(
                    logits.float(), batch["target_labels"], batch["option_mask"],
                    batch["is_multi_label_task"],
                    sigma=float(rcfg["sigma_start"]), samples=group_size,
                    w_sph=float(rcfg["w_sph"]),
                    supervised_weight=float(rcfg["supervised_weight"]),
                    supervised_cfg=tcfg,
                )
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("max_grad_norm", 1.0)))
                if not torch.isfinite(loss.detach()) or not torch.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite RLCD loss or gradients during smoke test")
                n_seen += 1
                print(
                    f"smoke_batch={n_seen} total={loss.item():.5f} "
                    f"policy_gradient={parts['policy_gradient_loss'].item():.5f} "
                    f"supervised={parts['supervised_loss'].item():.5f} "
                    f"grad_norm={float(grad_norm):.5f}", flush=True,
                )
                if n_seen >= args.smoke_batches:
                    break
            if n_seen == 0:
                raise RuntimeError("The training loader yielded no batches")
            print("RLCD smoke test passed; no optimizer step or checkpoint was written.")
            return 0

        val_loader, _, val_collator = build_loader(cfg, repo_root, "val")
        run_dir.mkdir(parents=True, exist_ok=True)
        write_run_metadata(run_dir, cfg, model.model_summary(), repo_root)

        if len(train_loader) <= 0:
            raise RuntimeError("Training loader has no batches")
        nominal_batches = len(train_loader)
        config_cap = tcfg.get("max_steps_per_epoch")
        if config_cap is not None:
            nominal_batches = min(nominal_batches, int(config_cap))
        if args.max_train_batches is not None:
            nominal_batches = min(nominal_batches, args.max_train_batches)
        if nominal_batches <= 0:
            raise ValueError("No training batches remain after applying batch caps")

        accum = max(1, int(tcfg.get("gradient_accumulation_steps", 1)))
        updates_per_epoch = math.ceil(nominal_batches / accum)
        total_updates = max(1, epochs * updates_per_epoch)
        if args.max_updates is not None:
            total_updates = min(total_updates, args.max_updates)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=float(tcfg.get("learning_rate", 2e-5)),
            weight_decay=float(tcfg.get("weight_decay", 0.01)),
            betas=tuple(tcfg.get("betas", [0.9, 0.999])),
        )
        scheduler = _make_scheduler(optimizer, total_updates, tcfg)
        amp_mode = str(tcfg.get("amp", "none")).lower()
        amp_enabled = device.type == "cuda" and amp_mode in {"fp16", "bf16"}
        amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_mode == "fp16"))
        threshold = float(tcfg.get("multi_label_threshold", 0.5))
        print_every = max(0, int(cfg.get("logging", {}).get("print_every_steps", 20)))
        best_metric = -1.0
        bad_epochs = 0
        patience_value = tcfg.get("early_stopping_patience")
        patience = None if patience_value is None else max(1, int(patience_value))
        metric_path = run_dir / "metrics.jsonl"
        best_path = run_dir / "best_checkpoint.pt"
        last_path = run_dir / "last_checkpoint.pt"
        global_step = 0
        stop_training = False
        skipped_updates = 0
        consecutive_bad_updates = 0
        max_consecutive_bad_updates = max(
            1, int(tcfg.get("max_consecutive_nonfinite_updates", 3))
        )

        print(f"Repo root: {repo_root}")
        print(f"RLCD run: {args.run_name} | init={init_path}")
        print(f"Scale={cfg['data'].get('scale')} | batches/epoch={nominal_batches} | "
              f"accumulation={accum} | planned_updates={total_updates} | device={device}")
        print(f"RLCD group_size={group_size} sigma={rcfg['sigma_start']}->{rcfg['sigma_end']} "
              f"w_sph={rcfg['w_sph']} supervised_weight={rcfg['supervised_weight']}")

        for epoch in range(epochs):
            if hasattr(train_ds, "set_epoch"):
                train_ds.set_epoch(epoch)
            model.train()
            start_time = time.perf_counter()
            train_metrics = MetricAccumulator(threshold=threshold)
            sums = {"total": 0.0, "pg": 0.0, "supervised": 0.0, "reward": 0.0}
            seen = 0
            optimizer.zero_grad(set_to_none=True)
            epoch_sigma = float(rcfg["sigma_start"]) + (
                float(rcfg["sigma_end"]) - float(rcfg["sigma_start"])
            ) * epoch / max(1, epochs - 1)

            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= nominal_batches:
                    break
                batch = move_batch(batch, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits = model(batch)
                processed = batch_idx + 1
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(
                        f"Non-finite model logits at microbatch {processed}; "
                        "check the batch/features and FP16 forward pass"
                    )
                loss, parts = laya_style_rlcd_loss(
                    logits.float(), batch["target_labels"], batch["option_mask"],
                    batch["is_multi_label_task"], sigma=epoch_sigma, samples=group_size,
                    w_sph=float(rcfg["w_sph"]),
                    supervised_weight=float(rcfg["supervised_weight"]),
                    supervised_cfg=tcfg,
                )
                if not torch.isfinite(loss.detach()):
                    raise FloatingPointError(
                        f"Non-finite RLCD objective at microbatch {processed}; "
                        f"pg={float(parts['policy_gradient_loss'])}, "
                        f"supervised={float(parts['supervised_loss'])}"
                    )

                window_start = ((processed - 1) // accum) * accum
                window_size = min(accum, nominal_batches - window_start)
                scaled_loss = loss / max(1, window_size)
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                do_step = (processed % accum == 0) or (processed == nominal_batches)
                if do_step:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(tcfg.get("max_grad_norm", 1.0))
                    )
                    if not torch.isfinite(grad_norm):
                        # Do not pass non-finite gradients into AdamW. With FP16 AMP,
                        # lower the GradScaler scale explicitly and skip this update;
                        # its current behavior would otherwise raise before the scaler
                        # can recover. We also handle the rare non-AMP case by discarding
                        # the bad accumulation window, and stop if failures persist.
                        old_scale = scaler.get_scale() if scaler.is_enabled() else None
                        optimizer.zero_grad(set_to_none=True)
                        if scaler.is_enabled():
                            scaler.update(new_scale=max(float(old_scale) / 2.0, 1.0))
                            new_scale = scaler.get_scale()
                        else:
                            new_scale = None
                        skipped_updates += 1
                        consecutive_bad_updates += 1
                        print(
                            f"WARNING: skipped non-finite gradient update; "
                            f"attempted_step={global_step + skipped_updates} "
                            f"global_step={global_step} grad_norm={float(grad_norm)} "
                            f"amp_scale={old_scale}->{new_scale} "
                            f"consecutive_skips={consecutive_bad_updates}/"
                            f"{max_consecutive_bad_updates}",
                            flush=True,
                        )
                        if consecutive_bad_updates >= max_consecutive_bad_updates:
                            raise FloatingPointError(
                                f"{consecutive_bad_updates} consecutive non-finite gradient updates; "
                                "stopping rather than silently training through instability"
                            )
                    else:
                        if scaler.is_enabled():
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if scheduler is not None:
                            scheduler.step()
                        global_step += 1
                        consecutive_bad_updates = 0

                train_metrics.update(
                    logits.detach().float(), batch["target_labels"], batch["option_mask"],
                    loss.detach(), batch.get("metadata"), batch.get("is_multi_label_task"),
                )
                sums["total"] += float(loss.detach())
                sums["pg"] += float(parts["policy_gradient_loss"])
                sums["supervised"] += float(parts["supervised_loss"])
                sums["reward"] += float(parts["reward_mean"])
                seen += 1
                if print_every and (processed % print_every == 0):
                    print(
                        f"  microbatch {processed}/{nominal_batches} global_step={global_step} "
                        f"loss={loss.item():.4f} pg={parts['policy_gradient_loss'].item():.4f} "
                        f"sup={parts['supervised_loss'].item():.4f} "
                        f"reward={parts['reward_mean'].item():.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.3g}", flush=True,
                    )

                if args.max_updates is not None and global_step >= args.max_updates and do_step:
                    stop_training = True
                    break

            if seen == 0:
                raise RuntimeError("No training batches were consumed")
            train_result = train_metrics.compute()
            val_result = evaluate(
                model, val_loader, device,
                threshold=threshold,
                loss_cfg=tcfg,
            )
            selection = float(val_result["selection_score"])
            is_best = selection > best_metric
            if is_best:
                best_metric = selection
                bad_epochs = 0
            else:
                bad_epochs += 1
            elapsed = time.perf_counter() - start_time
            denom = max(1, seen)
            record = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "seconds": elapsed,
                "sigma": epoch_sigma,
                "rlcd": {
                    "loss": sums["total"] / denom,
                    "policy_gradient_loss": sums["pg"] / denom,
                    "supervised_loss": sums["supervised"] / denom,
                    "reward_mean": sums["reward"] / denom,
                    "group_size": group_size,
                    "skipped_nonfinite_updates": skipped_updates,
                },
                "train": train_result,
                "val": val_result,
            }
            with metric_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            save_checkpoint(
                last_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch, global_step=global_step, best_metric=best_metric,
                cfg=cfg, repo_root=repo_root,
            )
            if is_best:
                save_checkpoint(
                    best_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    epoch=epoch, global_step=global_step, best_metric=best_metric,
                    cfg=cfg, repo_root=repo_root,
                )
            print(
                f"epoch {epoch + 1}/{epochs} | sigma={epoch_sigma:.4f} "
                f"rlcd_loss={sums['total']/denom:.4f} pg_loss={sums['pg']/denom:.4f} "
                f"sup_loss={sums['supervised']/denom:.4f} "
                f"train_top1_hit={train_result['accuracy']:.4f} "
                f"val_top1_hit={val_result['accuracy']:.4f} "
                f"val_selection={selection:.4f} val_loss={val_result['loss']:.4f} "
                f"single_acc={val_result['single_choice_accuracy']} "
                f"multi_f1={val_result['multi_label_f1']} steps={global_step} "
                f"time={elapsed:.1f}s best={best_metric:.4f}", flush=True,
            )
            if (patience is not None and bad_epochs >= patience) or stop_training:
                break

        print(f"Best checkpoint: {best_path}")
        print(f"Training complete. Metrics: {metric_path}")
        return 0
    finally:
        _close_collators(train_collator, val_collator)


if __name__ == "__main__":
    raise SystemExit(main())
