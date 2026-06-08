from pathlib import Path
from typing import Annotated

import torch
import typer
from loguru import logger
from peft import PeftModel
from transformers import AutoModel, AutoProcessor, GenerationConfig

# Side-effect: register Qwen3ASR into AutoConfig/AutoModel/AutoProcessor
from csrc_qwen.qwen_asr.inference.qwen3_asr import Qwen3ASRModel as _  # noqa: F401

app = typer.Typer()


@app.command()
def main(
    base_model: Annotated[str, typer.Argument(help="ベースモデルのパス")],
    adapter_path: Annotated[str, typer.Argument(help="LoRA アダプターのパス")],
    output_path: Annotated[str, typer.Argument(help="マージ済みモデルの出力先")],
) -> None:
    """LoRA アダプターをベースモデルにマージして保存する。"""
    logger.info(f"Loading base model: {base_model}")
    model = AutoModel.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging adapter into base model")
    model = model.merge_and_unload()

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged model to: {out}")
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.save_pretrained(out, safe_serialization=True)

    logger.info(f"Saving processor to: {out}")
    processor = AutoProcessor.from_pretrained(base_model, fix_mistral_regex=True)
    processor.save_pretrained(out)

    logger.info("Done")


if __name__ == "__main__":
    app()
