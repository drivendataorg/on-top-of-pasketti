"""Factory for word track training models, datasets, and collate functions."""
from loguru import logger


def create_model(cfg):
    model_type = cfg.model.type

    if model_type == "qwen":
        from src.models.word.qwen import QwenASRModule
        lora_cfg = cfg.get("lora", {})
        lora_rank = lora_cfg.get("rank", 0)
        checkpoint = cfg.model.get("checkpoint", None)
        if checkpoint and lora_rank > 0:
            logger.info(f"Loading base Qwen from {checkpoint}, then applying LoRA")
            model = QwenASRModule.load_from_checkpoint(
                checkpoint, map_location="cpu",
                lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
                warmup=cfg.optimizer.warmup,
            )
            model.apply_lora(
                rank=lora_rank,
                alpha=lora_cfg.get("alpha", 16),
                target_modules=lora_cfg.get("target_modules", None),
            )
            return model
        elif checkpoint:
            logger.info(f"Loading finetuned Qwen from {checkpoint}")
            model = QwenASRModule.load_from_checkpoint(
                checkpoint, map_location="cpu",
                lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
                warmup=cfg.optimizer.warmup,
            )
            return model
        return QwenASRModule(
            model_name=cfg.model.name,
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            warmup=cfg.optimizer.warmup,
            freeze_audio_encoder=cfg.model.freeze_audio_encoder,
            freeze_lm_layers=cfg.model.freeze_lm_layers,
            batch_audio_tower=cfg.model.get("batch_audio_tower", True),
            lora_rank=lora_rank,
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_target_modules=lora_cfg.get("target_modules", None),
            kl_alpha=cfg.get("distillation", {}).get("kl_alpha", 0.0),
            kl_alpha_min=cfg.get("distillation", {}).get("kl_alpha_min", 0.0),
        )
    else:
        raise ValueError(f"Unknown word track model type: {model_type}")


def create_dataset(entries, model, cfg, train=True):
    model_type = cfg.model.type

    if model_type == "qwen":
        from src.models.word.qwen import QwenASRDataset
        use_prompts = cfg.model.get("use_prompts", False)
        augment_cfg = cfg.get("augment", None) if train else None
        return QwenASRDataset(entries, model.processor, model.text_prompt, use_prompts=use_prompts, augment_cfg=augment_cfg)
    else:
        raise ValueError(f"Unknown word track model type: {model_type}")


def get_collate_fn(model, cfg):
    model_type = cfg.model.type

    if model_type == "qwen":
        from src.models.word.qwen import collate_fn
        return collate_fn
    else:
        raise ValueError(f"Unknown word track model type: {model_type}")
