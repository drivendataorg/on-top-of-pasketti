import gc
import numpy as np

import torch
import hydra
from typing import Dict, Any, List
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.utils import instantiate
import wandb
from typing import Tuple
import polars as pl
from pathlib import Path
import os
import torch.nn.functional as F
from src.utils.score import score_ipa_cer

from ema_pytorch import EMA


def _trim_logits_batch(logits: torch.Tensor, output_lengths: torch.Tensor) -> List[torch.Tensor]:
    """Trim padded encoder frames so saved emissions only contain real timesteps."""
    logits_cpu = logits.detach().cpu()
    lengths_cpu = output_lengths.detach().cpu().tolist()
    return [seq_logits[: int(seq_len)].contiguous() for seq_logits, seq_len in zip(logits_cpu, lengths_cpu)]


def _build_logits_payload(
    trimmed_logits: List[torch.Tensor],
    utterance_ids: List[str],
) -> Dict[str, np.ndarray]:
    lengths = np.asarray([seq.shape[0] for seq in trimmed_logits], dtype=np.int32)
    offsets = np.zeros(len(lengths), dtype=np.int64)
    if len(lengths) > 1:
        offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)

    if trimmed_logits:
        packed_logits = torch.cat([seq.to(dtype=torch.float16) for seq in trimmed_logits], dim=0).numpy()
    else:
        packed_logits = np.empty((0, 0), dtype=np.float16)

    return {
        "logits": packed_logits,
        "offsets": offsets,
        "lengths": lengths,
        "utterance_ids": np.asarray(utterance_ids),
    }



def _get_lr_logs(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    logs: Dict[str, float] = {}
    for idx, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group_{idx}")
        logs[f"lr/{name}"] = group["lr"]
    return logs


def _l2_norm_params(params: List[torch.nn.Parameter], use_grad: bool) -> float:
    if not params:
        return 0.0
    total = torch.zeros((), device=params[0].device)
    for p in params:
        t = p.grad if use_grad else p
        if t is None:
            continue
        total = total + t.detach().pow(2).sum()
    return float(torch.sqrt(total).item()) if total.numel() else 0.0


def _get_group_lr(optimizer: torch.optim.Optimizer, group_name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return float(group.get("lr", 0.0))
    return float(optimizer.param_groups[0].get("lr", 0.0)) if optimizer.param_groups else 0.0


def train_epoch(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    criterion: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    ema: EMA | None,
    backbone_params: List[torch.nn.Parameter],
    head_params: List[torch.nn.Parameter],
    device: torch.device, 
    step_offset: int,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler_backbone: torch.optim.lr_scheduler.LRScheduler | None = None,
    scheduler_head: torch.optim.lr_scheduler.LRScheduler | None = None,
    step_backbone_scheduler: bool = True,
    clip_max_norm: float = 1.0,
) -> Tuple[float, float, Dict[str, float], int]:
    """Trains the model for a single epoch."""
    model.train()
    total_loss = 0.0
    total_loss_ctc = 0.0
    total_loss_ce = 0.0
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Training", leave=True)

    # Containers to collect all decoded preds and labels from forward pass
    all_trimmed_logits = []
    all_gt_phonetic_texts = []
    all_age_buckets = []
    
    for batch_idx, batch in enumerate(pbar):
        # 1. Move everything to device
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # 2. Forward pass with Mixed Precision
        optimizer.zero_grad()

        
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type=="cuda"):
            output = model(input_features, attention_mask=attention_mask)
            log_probs = output[0]
            logits = output[1]
            if model.LLA:
                norm_weights = output[2]
            output_lengths = model.get_output_lengths(input_lengths)
            
            if getattr(model, "enable_age_head", False):
                age_logits = output[3]
                age_labels_tensor = batch["age_labels"].to(device)
                loss_ce = F.cross_entropy(age_logits, age_labels_tensor, ignore_index=-100)
                loss_ctc = criterion(log_probs, labels, output_lengths, target_lengths)
                loss = loss_ctc + (model.age_head_lambda * loss_ce)
                loss_dict = {"loss_ctc": loss_ctc.item(), "loss_ce": loss_ce.item(), "loss_total": loss.item()}
            else:
                loss = criterion(log_probs, labels, output_lengths, target_lengths)
                loss_dict = {"loss_ctc": loss.item(), "loss_total": loss.item(), "loss_ce": 0.0}
            
        # 3. Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm) 
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm) # clipping grads in place
            optimizer.step()

        # Step the LR schedulers every batch
        if scheduler_head is not None:
            scheduler_head.step()
        if step_backbone_scheduler and scheduler_backbone is not None:
            scheduler_backbone.step()
        # Update EMA
        if ema is not None:
            ema.update()

        # 4. Logging
        loss_val = loss.item()
        total_loss += loss_val
        total_loss_ce += loss_dict["loss_ce"]
        total_loss_ctc += loss_dict["loss_ctc"]
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        trimmed_logits = _trim_logits_batch(logits, output_lengths)

        # Update containers with decoded preds and labels from batch
        all_trimmed_logits.extend(trimmed_logits)
        all_gt_phonetic_texts.extend(batch["phonetic_text"])
        all_age_buckets.extend(batch["age_buckets"])

        global_step = step_offset + (batch_idx + 1)

        # Cheap optimizer dynamics
        grad_norm = float(total_norm.item()) if torch.is_tensor(total_norm) else float(total_norm)
        backbone_grad_norm = _l2_norm_params(backbone_params, use_grad=True)
        head_grad_norm = _l2_norm_params(head_params, use_grad=True)
        weight_norm_backbone = _l2_norm_params(backbone_params, use_grad=False)
        weight_norm_ctc = _l2_norm_params(head_params, use_grad=False)
        backbone_lr_current = _get_group_lr(optimizer, "backbone")
        update_ratio_backbone = (backbone_lr_current * backbone_grad_norm) / (weight_norm_backbone + 1e-12)
        clip_trigger = float(grad_norm > clip_max_norm)

        lr_logs = _get_lr_logs(optimizer)
        wandb.log(
            {
                "train/step_loss": loss_val,
                "opt/grad_norm": grad_norm,
                "opt/backbone_grad_norm": backbone_grad_norm,
                "opt/ctc_head_grad_norm": head_grad_norm,
                "opt/weight_norm_backbone": weight_norm_backbone,
                "opt/weight_norm_ctc": weight_norm_ctc,
                "opt/update_ratio_backbone": update_ratio_backbone,
                "opt/clip_trigger": clip_trigger,
                **lr_logs,
            },
            step=global_step,
        )

        if getattr(model, "enable_age_head", False):
            wandb.log({
                "train/loss_ctc": loss_dict["loss_ctc"],
                "train/loss_age_ce": loss_dict["loss_ce"],
                "train/loss_total": loss_dict["loss_total"],
            }, step=global_step)

        # print(f"this batch labels: {batch['phonetic_text']}")
        # print(f"this batch decoded: {decoded}")
    if model.LLA:
        # 1. Detach from graph, move to CPU, and squeeze out the 1x1x1 dimensions
        flat_weights = norm_weights.detach().cpu().squeeze().tolist()
        
        # 2. Format each weight to 3 decimal places for clean reading
        # Layer 0 is the CNN output, Layers 1-12 are the Transformer layers
        formatted_weights = [f"L{i}: {w:.3f}" for i, w in enumerate(flat_weights)]
        
        # 3. Print as a single, clean line
        print(f"Layer Weights: {', '.join(formatted_weights)}")
    #apply the decoder to all logits at once (after the epoch) to get the final decoded predictions for PER calculation
    # Note: This assumes that the model's forward method returns raw logits as
    # the second output when in training mode, which we can decode here.
    all_decoded_preds = model.decoder(all_trimmed_logits)

    # Calculate average epoch loss over all batches
    avg_loss = total_loss / len(dataloader)
    avg_loss_ctc = total_loss_ctc / len(dataloader)
    avg_loss_ce = total_loss_ce / len(dataloader)

    # Calculate train_PER over the entire epoch
    train_per = score_ipa_cer(actual=all_gt_phonetic_texts, predicted=all_decoded_preds)

    # Calculate train_PER per age bucket
    train_per_by_bucket = per_by_bucket(
        actual=all_gt_phonetic_texts,
        predicted=all_decoded_preds,
        buckets=all_age_buckets,
    )
        
    last_step = step_offset + len(dataloader)
    return avg_loss, train_per, train_per_by_bucket, last_step, avg_loss_ce, avg_loss_ctc


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    criterion: torch.nn.Module, 
    ema: EMA | None,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[float, float, Dict[str, Any], Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluates the model on the validation set."""
    # If EMA is enabled, use the EMA weights for evaluation
    backup = None
    if ema is not None:
        backup = model # Keep a backup of the original model
        model = ema.ema_model
        # ema.store(model.parameters())
        # ema.copy_to(model.parameters())

    model.eval()
    total_loss = 0.0
    total_loss_ctc = 0.0
    total_loss_ce = 0.0
    
    pbar = tqdm(dataloader, desc="Validating", leave=True)

    # Containers to collect all data
    all_trimmed_logits = []
    all_gt_phonetic_texts = []
    all_utterance_ids = []
    all_child_ids = []
    all_age_buckets = []
    all_output_lengths = []

    for batch in pbar:
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Forward pass
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type=="cuda"):
            output = model(input_features, attention_mask=attention_mask)
            log_probs = output[0]
            logits = output[1]
            output_lengths = model.get_output_lengths(input_lengths)
            
            if getattr(model, "enable_age_head", False):
                age_logits = output[3]
                age_labels_tensor = batch["age_labels"].to(device)
                loss_ce = F.cross_entropy(age_logits, age_labels_tensor, ignore_index=-100)
                loss_ctc = criterion(log_probs, labels, output_lengths, target_lengths)
                loss = loss_ctc + (model.age_head_lambda * loss_ce)
                loss_dict = {"loss_ctc": loss_ctc.item(), "loss_ce": loss_ce.item(), "loss_total": loss.item()}
            else:
                loss = criterion(log_probs, labels, output_lengths, target_lengths)
                loss_dict = {"loss_ctc": loss.item(), "loss_total": loss.item(), "loss_ce": 0.0}
            
        loss_val = loss.item()
        total_loss += loss_val
        total_loss_ctc += loss_dict["loss_ctc"]
        total_loss_ce += loss_dict["loss_ce"]
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        trimmed_logits = _trim_logits_batch(logits, output_lengths)

        # Update containers
        all_trimmed_logits.extend(trimmed_logits)
        all_gt_phonetic_texts.extend(batch["phonetic_text"])
        all_utterance_ids.extend(batch["utterance_ids"])
        all_child_ids.extend(batch["child_ids"])
        all_age_buckets.extend(batch["age_buckets"])
        all_output_lengths.extend(int(length) for length in output_lengths.detach().cpu().tolist())

    # Apply the decoder to all collected logits at once
    all_decoded_preds = model.decoder(all_trimmed_logits)

    # Calculate average epoch loss over all batches
    avg_loss = total_loss / len(dataloader)
    avg_loss_ctc = total_loss_ctc / len(dataloader)
    avg_loss_ce = total_loss_ce / len(dataloader)

    # Calculate val_PER over the entire epoch
    val_per = score_ipa_cer(actual=all_gt_phonetic_texts, predicted=all_decoded_preds)

    # Calculate val_PER per age bucket
    val_per_by_bucket = per_by_bucket(
        actual=all_gt_phonetic_texts,
        predicted=all_decoded_preds,
        buckets=all_age_buckets,
    )

    # Pack up the results to save later
    val_results = {
        "utterance_id": all_utterance_ids,
        "child_id": all_child_ids,
        "ground_truth": all_gt_phonetic_texts,
        "prediction": all_decoded_preds,
        "output_length": all_output_lengths,
    }
    val_logits_payload = _build_logits_payload(all_trimmed_logits, all_utterance_ids)
        
    # If EMA was used, restore original weights
    if ema is not None:
        model = backup
        model.to(device)
        del backup
        gc.collect()
        torch.cuda.empty_cache()
    return avg_loss, val_per, val_results, val_per_by_bucket, val_logits_payload, avg_loss_ce, avg_loss_ctc


def per_by_bucket(
    actual: List[str],
    predicted: List[str],
    buckets: List[str],
) -> Dict[str, float]:
    """Compute PER per bucket (e.g., age buckets)."""
    bucket_to_indices: Dict[str, List[int]] = {}
    for idx, bucket in enumerate(buckets):
        bucket_to_indices.setdefault(bucket, []).append(idx)

    per_by_bucket: Dict[str, float] = {}
    for bucket, indices in bucket_to_indices.items():
        bucket_actual = [actual[i] for i in indices]
        bucket_pred = [predicted[i] for i in indices]
        per_by_bucket[bucket] = score_ipa_cer(actual=bucket_actual, predicted=bucket_pred)

    return per_by_bucket


def train_CTC_model(
    cfg: DictConfig,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    ema: EMA | None,
    output_dir: str,
    fold: int, 
    step_offset: int = 0,
    epoch_offset: int = 0,
    device: torch.device | None = None,
):
    """Main training orchestration function."""
    # 1. Setup Device

    env_device = os.environ.get("TRAINING_DEVICE")
    if env_device is not None:
        device = torch.device(env_device)
    elif device is None:
        device = torch.device(cfg.device)
    print(f"Training on device: {device}")
    model = model.to(device)
    criterion = criterion.to(device)
    ema = ema.to(device) if ema is not None else None
    
    # 2. Setup Optimizer and AMP Scaler
    weight_decay = cfg.training.weight_decay
    backbone_lr = cfg.training.backbone_lr
    head_lr = cfg.training.head_lr
    lla_lr = cfg.training.lla_lr if model.LLA else None
    clip_max_norm = float(cfg.training.max_grad_norm)

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "head" not in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "head" in n]
    if model.LLA and lla_lr is None:
        raise ValueError("LLA is enabled but training.lla_lr is not set in config.")


    #remove the layer_weights from both param groups if LLA is enabled, since we'll give it its own group and lr
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
        if (os.environ.get("USER") == "epochvipc3" or os.environ.get("USER") == "epochvipc8") and False:
            amp_dtype = torch.bfloat16
            scaler = None
        else:
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")
    else:
        amp_dtype = torch.float32
        scaler = None
    print(f"Mixed precision: {use_amp} | dtype: {amp_dtype} | scaler: {scaler is not None}")

    # 2b. Setup LR Scheduler(s) (stepped every batch)
    total_steps = cfg.training.num_epochs * len(train_loader)
    freeze_backbone_epochs = int(cfg.training.freeze_backbone_epochs)
    backbone_steps = max(0, cfg.training.num_epochs - freeze_backbone_epochs) * len(train_loader)

    scheduler_backbone = None
    scheduler_head = None
    scheduler_cfg = cfg.scheduler
    has_backbone = "backbone" in scheduler_cfg
    has_head = "head" in scheduler_cfg
    
    # if differential scheduler, init two schedulers.
    if has_backbone or has_head:
        if not (has_backbone and has_head):
            raise ValueError("scheduler.backbone and scheduler.head must both be set for differential scheduling")
        if backbone_steps <= 0:
            raise ValueError("freeze_backbone_epochs must be < num_epochs for differential scheduling")

        scheduler_backbone = instantiate(
            scheduler_cfg.backbone,
            optimizer=optimizer,
            total_steps=backbone_steps,
            group_names=["backbone"],
        )

        scheduler_head = instantiate(
            scheduler_cfg.head,
            optimizer=optimizer,
            total_steps=total_steps,
            group_names=["head"],
        )
    else:
        scheduler_head = instantiate(
            scheduler_cfg,
            optimizer=optimizer,
            total_steps=total_steps,
        )

    epochs = cfg.training.num_epochs
    best_val_per = float('inf')
    early_stopping_counter = 0
    save_val_logits_best = bool(cfg.logging.get("save_val_logits_best", True))
    train_on_all = cfg.data.get("train_on_all_data", False)
    
    # Setup the unified OOF path (steps up one directory from output_dir/fold_X)
    base_output_dir = Path(output_dir).parent
    oof_path = base_output_dir / "oof_predictions_best.parquet"

    if freeze_backbone_epochs > 0:
        print(f"Freezing backbone parameters for the first {freeze_backbone_epochs} epochs.")
        for p in backbone_params:
            p.requires_grad = False
    
    # 3. Epoch Loop
    print("\n--- Starting Training Loop ---")

    last_logged_step = step_offset
    last_epoch_run = 0
    for epoch in range(1, epochs + 1):

        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
            print("Unfreezing backbone parameters.")
            for p in backbone_params:
                p.requires_grad = True

        print(f"\nEpoch {epoch}/{epochs}")
        
        train_loss, train_per, train_per_by_bucket, last_logged_step, train_loss_ce, train_loss_ctc = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            ema=ema,
            backbone_params=backbone_params,
            head_params=head_params,
            device=device,
            step_offset=last_logged_step,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            scheduler_backbone=scheduler_backbone,
            scheduler_head=scheduler_head,
            step_backbone_scheduler=epoch >= freeze_backbone_epochs + 1,
            clip_max_norm=clip_max_norm,
        )
        if train_on_all:
            # No validation — save checkpoint every epoch
            val_loss, val_per, val_loss_ce, val_loss_ctc = 0.0, 0.0, 0.0, 0.0
            val_per_by_bucket = {}
        else:
            val_loss, val_per, val_results, val_per_by_bucket, val_logits_payload, val_loss_ce, val_loss_ctc = validate_epoch(
                model,
                val_loader,
                criterion,
                ema,
                device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

        train_bucket_logs = {f"train/per_age/{k}": v for k, v in train_per_by_bucket.items()}
        val_bucket_logs = {f"val/per_age/{k}": v for k, v in val_per_by_bucket.items()}
        lr_logs = _get_lr_logs(optimizer)

        log_dict = {
            "epoch": epoch_offset + epoch,
            "train/loss": train_loss,
            "train/loss_total": train_loss,
            "train/loss_ctc": train_loss_ctc,
            "train/per": train_per,
            **lr_logs,
            **train_bucket_logs,
        }

        if not train_on_all:
            log_dict.update({
                "val/loss": val_loss,
                "val/loss_total": val_loss,
                "val/loss_ctc": val_loss_ctc,
                "val/per": val_per,
                **val_bucket_logs,
            })

        if getattr(model, "enable_age_head", False):
            if not train_on_all:
                log_dict["val/loss_age_ce"] = val_loss_ce
            log_dict["train/loss_age_ce"] = train_loss_ce

        wandb.log(log_dict, step=last_logged_step)

        if train_on_all:
            print(f"Train Loss: {train_loss:.4f} | Train PER: {train_per:.4f} | (no val)")
        else:
            print(
                f"Train Loss: {train_loss:.4f} | Train PER: {train_per:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val PER: {val_per:.4f}"
            )
        last_epoch_run = epoch
        # if train_per_by_bucket:
        #     train_bucket_str = " | ".join(
        #         f"{k}: {v:.4f}" for k, v in sorted(train_per_by_bucket.items())
        #     )
        #     print(f"Train PER by age_bucket: {train_bucket_str}")
        # if val_per_by_bucket:
        #     val_bucket_str = " | ".join(
        #         f"{k}: {v:.4f}" for k, v in sorted(val_per_by_bucket.items())
        #     )
        #     print(f"Val PER by age_bucket: {val_bucket_str}")
        
        if train_on_all:
            # No validation — save checkpoint every epoch (overwrite)
            checkpoint_path = f"{output_dir}/best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_per': 0.0,
                'ema_state_dict': ema.state_dict() if ema is not None else None,
            }, checkpoint_path)
            print(f"Checkpoint saved (epoch {epoch}) → {checkpoint_path}")

        elif val_per < best_val_per:
            best_val_per = val_per
            early_stopping_counter = 0

            checkpoint_path = f"{output_dir}/best_model.pth"
            logits_path = Path(output_dir) / "val_logits_best.npz"

            # Save the model state dict
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                'val_per': best_val_per,
                'ema_state_dict': ema.state_dict() if ema is not None else None,
            }, checkpoint_path)

            # Create DF and add the fold column
            df_new_oof = pl.DataFrame(val_results).with_columns(pl.lit(fold).alias("fold"))

            if oof_path.exists():
                df_existing = pl.read_parquet(oof_path)
                # Drop rows from the CURRENT fold so we can overwrite our own previous best
                df_existing = df_existing.filter(pl.col("fold") != fold)
                # Combine the remaining folds with our new best predictions
                df_oof = pl.concat([df_existing, df_new_oof])
            else:
                df_oof = df_new_oof

            df_oof.write_parquet(oof_path)
            if save_val_logits_best:
                np.savez_compressed(logits_path, **val_logits_payload)
                print(
                    "New best validation PER! "
                    f"Model and preds saved to {checkpoint_path.split('speech_phonetic_track/', 1)[1]} "
                    f"and logits saved to {logits_path.relative_to(Path.cwd())}"
                )
            else:
                print(
                    "New best validation PER! "
                    f"Model and preds saved to {checkpoint_path.split('speech_phonetic_track/', 1)[1]}"
                )


        else:
            early_stopping_counter += 1

        if not train_on_all and early_stopping_counter > cfg.training.early_stopping_patience:
            print("Early Stopping triggered...")
            break
            
        guard_epochs = cfg.training.get("guard_epochs", None)
        guard_per = cfg.training.get("guard_PER", None)
        if not train_on_all and guard_epochs is not None and guard_per is not None:
            if epoch >= guard_epochs and val_per >= guard_per:
                print(f"Guard triggered: PER {val_per:.4f} >= {guard_per} after {epoch} epochs. Stopping training.")
                break
            
    print("\n--- Training Complete ---")
    return best_val_per, last_logged_step, last_epoch_run
