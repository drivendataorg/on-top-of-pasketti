from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import polars as pl
import torch
import wandb
from ema_pytorch import EMA
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from src.train.train_CTC import _get_group_lr, _get_lr_logs, _l2_norm_params
from src.utils.score import score_ipa_cer, score_wer


class _CyclingIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


def _run_validation_task(
    model: torch.nn.Module,
    dataloader,
    criterion: torch.nn.Module,
    task: str,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[float, float, Dict[str, Any]]:
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_refs = []
    all_utterance_ids = []
    all_child_ids = []

    pbar = tqdm(dataloader, desc=f"Validating[{task}]", leave=True)
    for batch in pbar:
        if batch is None:
            continue

        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
            log_probs, logits, _ = model(input_features, attention_mask=attention_mask, task=task)
            output_lengths = model.get_output_lengths(input_lengths)
            loss = criterion(log_probs, labels, output_lengths, target_lengths)

        total_loss += float(loss.item())
        pbar.set_postfix({"loss": f"{float(loss.item()):.4f}"})

        decoded = model.decode(logits.detach().cpu(), task=task)
        refs = []
        for lbl in labels.detach().cpu():
            pad_id = model.phon_vocab.pad_token_id if task == "phon" else model.word_vocab.pad_token_id
            valid = lbl[lbl != pad_id].tolist()
            tokenizer = model.phon_vocab if task == "phon" else model.word_vocab
            refs.append(tokenizer.decode(valid))

        all_preds.extend(decoded)
        all_refs.extend(refs)
        all_utterance_ids.extend(batch["utterance_ids"])
        all_child_ids.extend(batch["child_ids"])

    avg_loss = total_loss / max(len(dataloader), 1)
    if task == "phon":
        metric = score_ipa_cer(actual=all_refs, predicted=all_preds)
    else:
        # jiwer raises when any reference is empty after normalization.
        valid_refs = []
        valid_preds = []
        dropped = 0
        for ref, pred in zip(all_refs, all_preds):
            if str(ref).strip() == "":
                dropped += 1
                continue
            valid_refs.append(ref)
            valid_preds.append(pred)

        if dropped > 0:
            print(f"[MTL][word_val] Dropped {dropped} empty-reference samples before WER computation.")

        if not valid_refs:
            # Fallback when all refs are empty after preprocessing.
            metric = 1.0
        else:
            metric = score_wer(actual=valid_refs, predicted=valid_preds)

    return avg_loss, metric, {
        "utterance_id": all_utterance_ids,
        "child_id": all_child_ids,
        "ground_truth": all_refs,
        "prediction": all_preds,
        "task": [task] * len(all_preds),
    }


def train_mtl_model(
    cfg: DictConfig,
    model: torch.nn.Module,
    loaders: Dict[str, Any],
    criterion_phon: torch.nn.Module,
    criterion_word: torch.nn.Module,
    ema: EMA | None,
    output_dir: str,
    fold: int,
    step_offset: int = 0,
    epoch_offset: int = 0,
    device: torch.device | None = None,
):
    env_device = os.environ.get("TRAINING_DEVICE")
    if env_device is not None:
        device = torch.device(env_device)
    elif device is None:
        device = torch.device(cfg.device)

    model = model.to(device)
    criterion_phon = criterion_phon.to(device)
    criterion_word = criterion_word.to(device)
    ema = ema.to(device) if ema is not None else None

    weight_decay = cfg.training.weight_decay
    backbone_lr = cfg.training.backbone_lr
    head_lr = cfg.training.head_lr
    lla_lr = cfg.training.lla_lr if model.LLA else None

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "head" not in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "head" in n]

    if model.LLA and lla_lr is None:
        raise ValueError("LLA is enabled but training.lla_lr is not set in config.")
    if model.LLA:
        backbone_params = [p for p in backbone_params if p is not model.layer_weights]
        head_params = [p for p in head_params if p is not model.layer_weights]

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
        {"params": head_params, "lr": head_lr, "name": "head"},
    ]
    if model.LLA:
        param_groups.append({"params": [model.layer_weights], "lr": lla_lr, "name": "lla_weights"})

    OptimizerClass = hydra.utils.get_class(cfg.training.optimizer._target_)
    optimizer = OptimizerClass(param_groups, lr=backbone_lr, weight_decay=weight_decay)

    use_amp = cfg.training.mixed_precision
    if use_amp and device.type == "cuda":
        if os.environ.get("USER") in {"epochvipc3", "epochvipc8"}:
            amp_dtype = torch.bfloat16
            scaler = None
        else:
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")
    else:
        amp_dtype = torch.float32
        scaler = None
    print(f"Using AMP: {use_amp} with dtype {amp_dtype} on device {device}")

    steps_per_epoch = 2 * len(loaders["phon_train"])
    # print(f"length of word loader: {len(loaders["word_train"])}")
    # print(f"length of phon loader: {len(loaders["phon_train"])}")
    # print(len(loaders["word_train"]))
    # print(len(loaders["phon_train"]))

    total_steps = int(cfg.training.num_epochs) * steps_per_epoch

    scheduler_cfg = cfg.scheduler
    scheduler_head = None
    scheduler_backbone = None
    if "backbone" in scheduler_cfg and "head" in scheduler_cfg:
        scheduler_backbone = instantiate(
            scheduler_cfg.backbone,
            optimizer=optimizer,
            total_steps=total_steps,
            group_names=["backbone"],
        )
        scheduler_head = instantiate(
            scheduler_cfg.head,
            optimizer=optimizer,
            total_steps=total_steps,
            group_names=["head"],
        )
    else:
        scheduler_head = instantiate(scheduler_cfg, optimizer=optimizer, total_steps=total_steps)

    word_iter = _CyclingIterator(loaders["word_train"])
    phon_iter = _CyclingIterator(loaders["phon_train"])

    best_val_per = float("inf")
    early_stopping_counter = 0
    base_output_dir = Path(output_dir).parent
    oof_path = base_output_dir / "oof_predictions_best.parquet"

    last_logged_step = step_offset
    last_epoch_run = 0

    for epoch in range(1, int(cfg.training.num_epochs) + 1):
        model.train()
        pbar = tqdm(range(steps_per_epoch), desc="Training[MTL]", leave=True)

        total_loss = 0.0
        word_loss_sum = 0.0
        phon_loss_sum = 0.0
        word_steps = 0
        phon_steps = 0

        for local_step in pbar:
            task = "word" if local_step % 2 == 0 else "phon"
            batch = word_iter.next() if task == "word" else phon_iter.next()
            if batch is None:
                continue

            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
                log_probs, _, _ = model(input_features, attention_mask=attention_mask, task=task)
                output_lengths = model.get_output_lengths(input_lengths)
                criterion = criterion_word if task == "word" else criterion_phon
                loss = criterion(log_probs, labels, output_lengths, target_lengths)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler_head is not None:
                scheduler_head.step()
            if scheduler_backbone is not None:
                scheduler_backbone.step()
            if ema is not None:
                ema.update()

            loss_val = float(loss.item())
            total_loss += loss_val
            if task == "word":
                word_loss_sum += loss_val
                word_steps += 1
            else:
                phon_loss_sum += loss_val
                phon_steps += 1

            last_logged_step += 1
            pbar.set_postfix({"loss": f"{loss_val:.4f}", "task": task})

            grad_norm = float(total_norm.item()) if torch.is_tensor(total_norm) else float(total_norm)
            backbone_grad_norm = _l2_norm_params(backbone_params, use_grad=True)
            head_grad_norm = _l2_norm_params(head_params, use_grad=True)
            weight_norm_backbone = _l2_norm_params(backbone_params, use_grad=False)
            backbone_lr_current = _get_group_lr(optimizer, "backbone")
            update_ratio_backbone = (backbone_lr_current * backbone_grad_norm) / (weight_norm_backbone + 1e-12)

            wandb.log(
                {
                    "train/step_loss": loss_val,
                    "train/step_task": 0 if task == "word" else 1,
                    "opt/grad_norm": grad_norm,
                    "opt/backbone_grad_norm": backbone_grad_norm,
                    "opt/head_grad_norm": head_grad_norm,
                    "opt/update_ratio_backbone": update_ratio_backbone,
                    **_get_lr_logs(optimizer),
                },
                step=last_logged_step,
            )

        train_loss = total_loss / max(steps_per_epoch, 1)
        train_word_loss = word_loss_sum / max(word_steps, 1)
        train_phon_loss = phon_loss_sum / max(phon_steps, 1)

        eval_model = model
        backup = None
        if ema is not None:
            backup = model
            eval_model = ema.ema_model

        with torch.no_grad():
            val_phon_loss, val_phon_per, phon_results = _run_validation_task(
                model=eval_model,
                dataloader=loaders["phon_val"],
                criterion=criterion_phon,
                task="phon",
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            val_word_loss, val_word_wer, word_results = _run_validation_task(
                model=eval_model,
                dataloader=loaders["word_val"],
                criterion=criterion_word,
                task="word",
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

        if backup is not None:
            model = backup
            model.to(device)
            del backup
            gc.collect()
            torch.cuda.empty_cache()

        wandb.log(
            {
                "epoch": epoch_offset + epoch,
                "train/loss": train_loss,
                "train/word_loss": train_word_loss,
                "train/phon_loss": train_phon_loss,
                "val/phon_loss": val_phon_loss,
                "val/word_loss": val_word_loss,
                "val/phon_per": val_phon_per,
                "val/word_wer": val_word_wer,
                **_get_lr_logs(optimizer),
            },
            step=last_logged_step,
        )

        print(
            f"Epoch {epoch}/{int(cfg.training.num_epochs)} | "
            f"train_loss={train_loss:.4f} word_loss={train_word_loss:.4f} phon_loss={train_phon_loss:.4f} | "
            f"val_phon_per={val_phon_per:.4f} val_word_wer={val_word_wer:.4f}"
        )

        last_epoch_run = epoch
        if val_phon_per < best_val_per:
            best_val_per = val_phon_per
            early_stopping_counter = 0

            checkpoint_path = f"{output_dir}/best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_per": best_val_per,
                    "val_word_wer": val_word_wer,
                    "ema_state_dict": ema.state_dict() if ema is not None else None,
                },
                checkpoint_path,
            )

            df_new_oof = pl.concat([
                pl.DataFrame(phon_results).with_columns(pl.lit(fold).alias("fold")),
                pl.DataFrame(word_results).with_columns(pl.lit(fold).alias("fold")),
            ])
            if oof_path.exists():
                df_existing = pl.read_parquet(oof_path)
                df_existing = df_existing.filter(pl.col("fold") != fold)
                df_oof = pl.concat([df_existing, df_new_oof])
            else:
                df_oof = df_new_oof
            df_oof.write_parquet(oof_path)
        else:
            early_stopping_counter += 1

        if early_stopping_counter > int(cfg.training.early_stopping_patience):
            print("Early Stopping triggered...")
            break

    return best_val_per, last_logged_step, last_epoch_run
