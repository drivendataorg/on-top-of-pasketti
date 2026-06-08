import re
import shutil
from pathlib import Path
from typing import Annotated, Any

import torch
import typer
from csrc.manifest import load_manifest
from loguru import logger
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import GenerationConfig, Trainer, TrainerCallback, TrainingArguments

from csrc_qwen.config import TrainConfig, load_config, save_config
from csrc_qwen.dataset import (
    Qwen3ASRDataCollator,
    Qwen3ASRManifestDataset,
    build_prefix_text,
)
from csrc_qwen.qwen_asr.inference.qwen3_asr import Qwen3ASRModel

app = typer.Typer()

def patch_outer_forward(model: torch.nn.Module) -> None:
    """outer class に forward() を追加し、thinker.forward() に委譲する。"""
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. Your qwen3_asr model may be incompatible.",
        )

    def forward(  # noqa: PLR0913
        self: Any,  # noqa: ANN401
        input_ids: Any = None,  # noqa: ANN401
        attention_mask: Any = None,  # noqa: ANN401
        input_features: Any = None,  # noqa: ANN401
        feature_attention_mask: Any = None,  # noqa: ANN401
        labels: Any = None,  # noqa: ANN401
        **kwargs: Any,
    ) -> Any:  # noqa: ANN401
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True  # type: ignore


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        if hasattr(self.data_collator, "training"):
            self.data_collator.training = False
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            if hasattr(self.data_collator, "training"):
                self.data_collator.training = True


def copy_required_hf_files(src_dir: str, dst_dir: str) -> None:
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    src = Path(src_dir)
    for fn in required:
        src_file = src / fn
        if src_file.exists():
            shutil.copy2(src_file, dst / fn)


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str) -> None:
        self.base_model_path = base_model_path

    def on_save(
        self,
        args: TrainingArguments,
        state: Any,  # noqa: ANN401
        control: Any,  # noqa: ANN401
        **kwargs: Any,
    ) -> Any:  # noqa: ANN401
        if args.process_index != 0:
            return control

        ckpt_dir = str(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        if not Path(ckpt_dir).is_dir():
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_required_hf_files(self.base_model_path, ckpt_dir)
        return control


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> str | None:
    output_path = Path(output_dir)
    if not output_dir or not output_path.is_dir():
        return None
    best_step = None
    best_path = None
    for entry in output_path.iterdir():
        m = _CKPT_RE.match(entry.name)
        if not m:
            continue
        step = int(m.group(1))
        if entry.is_dir() and (best_step is None or step > best_step):
            best_step = step
            best_path = str(entry)
    return best_path


def run_training(cfg: TrainConfig) -> None:
    use_bf16 = cfg.precision == "bf16" and torch.cuda.is_available()

    logger.info(f"Loading model: {cfg.model_name_or_path}")
    load_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16 if use_bf16 else torch.float16,
        "device_map": None,
        "attn_implementation": "flash_attention_2",
    }
    if cfg.lora.use_qlora:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
        load_kwargs["device_map"] = "auto"
        logger.info("QLoRA enabled: loading model in 4-bit NF4")
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        cfg.model_name_or_path,
        **load_kwargs,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    # Validate
    if not cfg.train_audio_encoder and cfg.train_projector:
        raise ValueError("train_projector=True requires train_audio_encoder=True")

    # Apply LoRA
    exclude_modules = None
    if not cfg.train_audio_encoder:
        exclude_modules = ".*audio_tower.*"

    target_modules = list(cfg.lora.target_modules)
    if cfg.train_projector:
        target_modules.extend(["proj1", "proj2"])

    peft_config = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Gradient checkpointing
    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Build prefix_text
    prefix_text = build_prefix_text(processor, cfg.prompt)
    logger.info(f"prefix_text: {prefix_text!r}")

    # Build datasets
    train_entries = load_manifest(cfg.data.train_manifest)
    logger.info(f"Train entries: {len(train_entries)}")
    train_dataset = Qwen3ASRManifestDataset(train_entries, prefix_text)

    val_dataset = None
    if cfg.data.val_manifest and Path(cfg.data.val_manifest).exists():
        val_entries = load_manifest(cfg.data.val_manifest)
        logger.info(f"Val entries: {len(val_entries)}")
        val_dataset = Qwen3ASRManifestDataset(val_entries, prefix_text)

    collator = Qwen3ASRDataCollator(processor=processor, augment_cfg=cfg.augment)

    # WandB
    report_to = "none"
    if cfg.wandb.enabled:
        import wandb  # noqa: PLC0415

        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=cfg.model_dump(mode="json"),
        )
        report_to = "wandb"

    # Training arguments
    output_dir = str(cfg.output_dir)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size or cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        optim=cfg.optim.optim_name,
        num_train_epochs=cfg.max_epochs,
        lr_scheduler_type=cfg.scheduler.name,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        logging_steps=10,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=cfg.eval_steps if val_dataset else None,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        save_safetensors=True,
        load_best_model_at_end=cfg.load_best_model_at_end if val_dataset else False,
        metric_for_best_model=cfg.metric_for_best_model if val_dataset else None,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=report_to,
        eval_delay=10000,
    )

    # Save config
    save_config(cfg, Path(output_dir) / "config.yaml")

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=cfg.model_name_or_path)],
    )

    # Resume
    resume_from = find_latest_checkpoint(output_dir)
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save final adapter
    final_dir = str(Path(output_dir) / "final_adapter")
    model.save_pretrained(final_dir)
    copy_required_hf_files(cfg.model_name_or_path, final_dir)
    logger.info(f"Final adapter saved to: {final_dir}")


@app.command()
def main(
    config: Annotated[Path, typer.Argument(help="YAML設定ファイルパス")],
) -> None:
    """Qwen3-ASR LoRA ファインチューニングを実行する。"""
    cfg = load_config(config)
    run_training(cfg)


if __name__ == "__main__":
    app()
