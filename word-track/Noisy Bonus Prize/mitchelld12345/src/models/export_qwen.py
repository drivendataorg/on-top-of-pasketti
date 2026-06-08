"""Export a fine-tuned Qwen3-ASR checkpoint to safetensors format for inference."""
import argparse
import shutil
from pathlib import Path

import torch
from loguru import logger

from dirtygit import check
check()


def export_model(checkpoint_path, output_dir, model_name="Qwen/Qwen3-ASR-1.7B"):
    from qwen_asr import Qwen3ASRModel

    logger.info(f"Loading pretrained {model_name} on CPU")
    wrapper = Qwen3ASRModel.from_pretrained(
        model_name, device_map="cpu", max_new_tokens=2048, dtype=torch.bfloat16,
    )
    thinker = wrapper.model.thinker
    processor = wrapper.processor
    del wrapper

    logger.info(f"Loading finetuned weights from {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    del ckpt

    thinker_sd = {}
    for k, v in state_dict.items():
        if k.startswith("_base_thinker."):
            continue
        if k.startswith("thinker."):
            thinker_sd[k[len("thinker."):]] = v
        else:
            thinker_sd[k] = v

    missing, unexpected = thinker.load_state_dict(thinker_sd, strict=False)
    if missing:
        logger.warning(f"{len(missing)} missing keys")
    if unexpected:
        logger.warning(f"{len(unexpected)} unexpected keys")
    logger.info(f"Loaded {len(thinker_sd)} params into thinker")

    if hasattr(thinker, "generation_config") and thinker.generation_config is not None:
        thinker.generation_config.temperature = None

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    thinker.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info(f"Saved model to {output_dir} ({sum(f.stat().st_size for f in output_dir.iterdir()) / 1e9:.1f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-ASR checkpoint to safetensors")
    parser.add_argument("checkpoint", type=Path, help="Path to PyTorch Lightning .ckpt file")
    parser.add_argument("--output", type=Path, default=Path("exported_model"), help="Output directory (default: exported_model)")
    parser.add_argument("--model_name", default="Qwen/Qwen3-ASR-1.7B", help="Base model name on HuggingFace (default: Qwen/Qwen3-ASR-1.7B)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    export_model(args.checkpoint, args.output, args.model_name)


if __name__ == "__main__":
    main()
