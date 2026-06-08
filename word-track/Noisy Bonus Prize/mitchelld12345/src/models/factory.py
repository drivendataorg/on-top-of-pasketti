"""Factory for building ASR models from config."""
from pathlib import Path

from loguru import logger


def load_model_from_config(cfg):
    model_type = cfg.type
    model_name = cfg.name
    checkpoint = getattr(cfg, "checkpoint", None)

    if model_type == "qwen":
        from src.models.qwen_model import QwenASR
        model = QwenASR(model_name=model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if checkpoint:
        checkpoint = str(checkpoint)
        if checkpoint.endswith(".pt"):
            model.load_finetuned_weights(checkpoint)
            logger.info(f"Loaded finetuned weights from {checkpoint}")
        elif checkpoint.endswith(".ckpt"):
            model.load_checkpoint_weights(checkpoint)
            logger.info(f"Loaded checkpoint weights from {checkpoint}")
        else:
            raise ValueError(f"Unknown checkpoint format: {checkpoint}")

    logger.info(f"Loaded {model_type} model: {model_name}")
    return model
