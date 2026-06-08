from __future__ import annotations

# import os


import os
os.environ["NPY_DISABLE_CPU_FEATURES"] = "AVX512F"
import numpy as np


import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf
import wandb

from preprocessing.dataset import prepare_dl_dataset
from preprocessing.dataset_mtl import prepare_mtl_datasets
from preprocessing.get_json import get_data_from_json
from train.train_CTC import train_CTC_model
from train.train_CTC_mtl import train_mtl_model

from ema_pytorch import EMA


def _set_training_epochs(cfg: DictConfig, num_epochs: int) -> None:
    cfg.training.num_epochs = int(num_epochs)


def _build_pretraining_stages(cfg: DictConfig):
    pre_cfg = cfg.get("pretraining", None)
    if not pre_cfg or not pre_cfg.get("enabled", False):
        return []

    stages = []
    for idx, group in enumerate(pre_cfg.get("groups", []), start=1):
        jsonl_paths = [str(p) for p in group.get("jsonl_paths", []) if p]
        if not jsonl_paths:
            continue
        stage_name = group.get("name", f"group_{idx}")
        stages.append({
            "name": stage_name,
            "jsonl_paths": jsonl_paths,
            "num_epochs": int(pre_cfg.get("num_epochs", 1)),
        })
    return stages


def _resolve_ssl_pretrained_name(cfg: DictConfig, fold: int) -> str | None:
    ssl_init_cfg = cfg.get("ssl_init", None)
    if not ssl_init_cfg or not ssl_init_cfg.get("enabled", False):
        return None

    explicit_path = ssl_init_cfg.get("pretrained_name", None)
    if explicit_path:
        path = Path(str(explicit_path)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"ssl_init.pretrained_name path does not exist: {path}")
        return str(path)

    run_dir = ssl_init_cfg.get("run_dir", None)
    if not run_dir:
        raise ValueError(
            "ssl_init.enabled=true requires either ssl_init.pretrained_name "
            "or ssl_init.run_dir."
        )

    run_dir_path = Path(str(run_dir)).expanduser().resolve()
    if not run_dir_path.exists():
        raise FileNotFoundError(f"ssl_init.run_dir does not exist: {run_dir_path}")

    fold_index = fold + 1
    use_per_fold = bool(ssl_init_cfg.get("use_per_fold_backbone", True))
    backbone_subdir = str(ssl_init_cfg.get("backbone_subdir", "backbone_pretrained"))

    if use_per_fold:
        candidate = run_dir_path / f"fold_{fold_index}" / backbone_subdir
    else:
        candidate = run_dir_path / backbone_subdir

    if not candidate.exists():
        raise FileNotFoundError(
            f"Resolved SSL backbone path does not exist for fold {fold_index}: {candidate}"
        )

    return str(candidate)




@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    run_name = None
    run_counter = None
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # --- Setup Output Directory & Naming ---
    # if cfg.logging.get("use_random_name", False):
    #     run_name, run_counter = generate_unique_name()
    #     renamed_output_dir = output_dir.with_name(f"{output_dir.name}_{run_name}")
    #     output_dir.rename(renamed_output_dir)
    #     output_dir = renamed_output_dir

    # print("Loaded Hydra config:")
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    if run_name is not None:
        print(f"Run name: {run_name} (counter={run_counter})")
    print(f"Output dir: {output_dir}")

    # --- Initialize Weights & Biases ---
    wandb_tmp_dir = Path(tempfile.gettempdir()) / "wandb"
    wandb_tmp_dir.mkdir(parents=True, exist_ok=True)

    wandb_mode = "online" if cfg.logging.get("use_wandb", False) else "disabled"
    run = wandb.init(
        project=cfg.logging.get("project_name", "speech_phonetic_track"),
        entity=cfg.logging.get("wandb_entity", None),
        mode=wandb_mode,
        config=resolved_cfg,
        name=run_name,
        dir=str(wandb_tmp_dir),
    )
    # Use epoch as the x-axis for epoch-level metrics without changing the global step.
    wandb.define_metric("epoch", hidden=True)
    wandb.define_metric("train/loss", step_metric="epoch")
    wandb.define_metric("train/per", step_metric="epoch")
    wandb.define_metric("val/loss", step_metric="epoch")
    wandb.define_metric("val/per", step_metric="epoch")
    wandb.define_metric("train/per_age/*", step_metric="epoch")
    wandb.define_metric("val/per_age/*", step_metric="epoch")

    # --- Device Selection ---
    # TRAINING_DEVICE env var overrides config (e.g. TRAINING_DEVICE=cuda:1)
    device_str = os.environ.get("TRAINING_DEVICE", cfg.get("device", "cuda:0"))
    import torch
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # --- Cross Validation Loop ---
    n_splits = cfg.cv.get("n_splits", 5)
    
    fold_val_pers = [] 
    if cfg.cv.holdout: 
        n_splits = 1 # override for holdout test set
    wandb_step_offset = 0
    wandb_epoch_offset = 0
    for fold in range(n_splits):
        print(f"\n{'='*40}")
        print(f"Starting Fold {fold + 1}/{n_splits}")
        print(f"{'='*40}\n")
        
        mtl_enabled = bool(cfg.get("mtl", {}).get("enabled", False))

        # Optional SSL initialization: set fold-specific pretrained backbone path.
        # This must happen BEFORE dataset preparation so Whisper feature extraction
        # matches the backbone's expected mel bin count.
        ssl_pretrained_name = _resolve_ssl_pretrained_name(cfg, fold=fold)
        if ssl_pretrained_name is not None:
            if cfg.model.get('pretrained_name', None) is not None:
                cfg.model.pretrained_name = ssl_pretrained_name
            else:
                cfg.model.whisper_model_id = ssl_pretrained_name
            print(f"Using SSL-initialized backbone for fold {fold + 1}: {ssl_pretrained_name}")

        # 1. Prepare Fold Data
        if mtl_enabled:
            loaders, dataset_info = prepare_mtl_datasets(cfg, fold=fold)
            train_loader = val_loader = None
        else:
            train_loader, val_loader, dataset_info = prepare_dl_dataset(cfg, fold=fold)

        # Build pretraining stage metadata once per fold.
        pretraining_stages = _build_pretraining_stages(cfg)
        
        # 2a. Build decoder (injects runtime objects like tokenizer)
        # decoder = _build_decoder(cfg, dataset_info)
        if mtl_enabled:
            phon_decoder = instantiate(
                cfg.model.decoder,
                tokenizer=dataset_info.get("phon_tokenizer"),
            )
            word_decoder = instantiate(
                cfg.model.decoder,
                tokenizer=dataset_info.get("word_tokenizer"),
            )
        else:
            decoder = instantiate(
                cfg.model.decoder,
                tokenizer=dataset_info.get("tokenizer"),
            )

        # 2b. Initialize Model
        if mtl_enabled:
            model = instantiate(
                cfg.model,
                phoneme_vocab_size=dataset_info["phon_vocab_size"],
                word_vocab_size=dataset_info["word_vocab_size"],
                phon_vocab=dataset_info["phon_tokenizer"],
                word_vocab=dataset_info["word_tokenizer"],
                decoder=None,
                phon_decoder=phon_decoder,
                word_decoder=word_decoder,
            )
        else:
            model = instantiate(
                cfg.model,
                vocab_size=dataset_info["vocab_size"],
                vocab=dataset_info["tokenizer"],
                decoder=decoder,
            )
        
        # 3. Initialize CTC Loss
        if mtl_enabled:
            criterion_phon = instantiate(
                cfg.loss,
                blank_token_id=dataset_info["phon_blank_token_id"],
            )
            criterion_word = instantiate(
                cfg.loss,
                blank_token_id=dataset_info["word_blank_token_id"],
            )
            criterion = None
        else:
            criterion = instantiate(
                cfg.loss,
                blank_token_id=dataset_info["blank_token_id"]
            )
        
        # 4. Create fold-specific output directory
        fold_output_dir = output_dir / f"fold_{fold+1}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        if cfg.ema.enabled:
            ema = EMA(
                model,
                beta=cfg.ema.decay,
                update_after_step=cfg.ema.update_after_step,
                update_every=cfg.ema.update_every,
            )
            ema = ema.to(device)
            print(f"Initialized EMA with decay={cfg.ema.decay}, update_after_step={cfg.ema.update_after_step}, update_every={cfg.ema.update_every}")
        else:
            ema = None


        # 5. Optional staged pretraining on external groups.
        original_main_epochs = int(cfg.training.get("num_epochs", 1))
        if mtl_enabled and pretraining_stages:
            print("MTL mode enabled: skipping external staged pretraining for this run.")
        elif pretraining_stages:
            print(f"Running {len(pretraining_stages)} pretraining stage(s) before main training.")

            for stage_idx, stage in enumerate(pretraining_stages, start=1):
                print(
                    f"\n[Pretraining Stage {stage_idx}/{len(pretraining_stages)}] "
                    f"{stage['name']} | epochs={stage['num_epochs']}"
                )
                stage_data = get_data_from_json(
                    cfg,
                    inference=False,
                    pretraining=True,
                    jsonl_paths=stage["jsonl_paths"],
                )
                if not stage_data:
                    print(f"Skipping stage '{stage['name']}' because no valid utterances were found.")
                    continue

                stage_train_loader, stage_val_loader, _ = prepare_dl_dataset(
                    cfg,
                    fold=fold,
                    inference=False,
                    data_override=stage_data,
                )
                _set_training_epochs(cfg, stage["num_epochs"])
                stage_output_dir = fold_output_dir / f"pretraining_stage_{stage_idx}_{stage['name']}"
                stage_output_dir.mkdir(parents=True, exist_ok=True)

                _, wandb_step_offset, stage_epochs_run = train_CTC_model(
                    cfg=cfg,
                    model=model,
                    train_loader=stage_train_loader,
                    val_loader=stage_val_loader,
                    criterion=criterion,
                    ema=ema,
                    output_dir=str(stage_output_dir),
                    fold=fold,
                    step_offset=wandb_step_offset,
                    epoch_offset=wandb_epoch_offset,
                    device=device,
                )
                wandb_epoch_offset += stage_epochs_run

        # 6. Main training on competition data.
        _set_training_epochs(cfg, original_main_epochs)
        if mtl_enabled:
            best_val_per, wandb_step_offset, last_epoch_run = train_mtl_model(
                cfg=cfg,
                model=model,
                loaders=loaders,
                criterion_phon=criterion_phon,
                criterion_word=criterion_word,
                ema=ema,
                output_dir=str(fold_output_dir),
                fold=fold,
                step_offset=wandb_step_offset,
                epoch_offset=wandb_epoch_offset,
                device=device,
            )
        else:
            best_val_per, wandb_step_offset, last_epoch_run = train_CTC_model(
                cfg=cfg,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                ema=ema,
                output_dir=str(fold_output_dir),
                fold=fold,
                step_offset=wandb_step_offset,
                epoch_offset=wandb_epoch_offset,
                device=device,
            )
        wandb_epoch_offset += last_epoch_run
        
        fold_val_pers.append(best_val_per)
        
        wandb.log({f"cv/fold_{fold}_best_val_per": best_val_per})

    # --- Summary Logging ---
    mean_val_per = np.mean(fold_val_pers)
    print(f"\nCross Validation Complete! Mean Val PER: {mean_val_per:.4f}")

    metrics = {
        "cv/mean_val_per": mean_val_per, 
    }

    wandb.log(metrics)
    
    if run is not None and run_name is not None:
        run.summary["run/name"] = run_name
        run.summary["run/output_dir"] = str(output_dir)
        run.summary["cv/mean_val_per"] = mean_val_per 

    if run is not None:
        run.finish()

    # --- Cleanup ---
    local_wandb_dir = PROJECT_ROOT / "wandb" # Ensure PROJECT_ROOT is defined
    if local_wandb_dir.exists():
        shutil.rmtree(local_wandb_dir)

if __name__ == "__main__":
    main()
