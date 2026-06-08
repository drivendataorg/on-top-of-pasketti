"""Fine-tune Qwen3-ASR on children's speech data for the Word track.

This is a wrapper around the official Qwen3-ASR SFT training script,
adapted for the competition use case. It fine-tunes the model on
children's speech data to improve WER on the target domain.

Usage (full dataset, single GPU):
    python word_track/train.py \
        --model_path Qwen/Qwen3-ASR-1.7B \
        --train_file ./train.jsonl \
        --output_dir ./checkpoints \
        --batch_size 16 \
        --grad_acc 4 \
        --lr 2e-5 \
        --epochs 3
"""

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from qwen_asr import Qwen3ASRModel
from transformers import (EarlyStoppingCallback, GenerationConfig, Trainer,
                          TrainerCallback, TrainingArguments)


# ── Monkey-patch: expose thinker.forward at the top level ──────────────
def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    def forward(self, input_ids=None, attention_mask=None,
                input_features=None, feature_attention_mask=None,
                labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels, **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


# ── Checkpoint utilities ───────────────────────────────────────────────
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step, best_path = None, None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step, best_path = step, path
    return best_path


def load_audio(path: str, sr: int = 16000, max_duration: float = 30.0):
    """Load audio, preferring pre-cached .npy file over decoding flac/wav."""
    npy_path = path.rsplit(".", 1)[0] + ".npy" if "." in path else path + ".npy"
    max_samples = int(sr * max_duration)
    if os.path.exists(npy_path):
        wav = np.load(npy_path)
        if len(wav) > max_samples:
            wav = wav[:max_samples]
        return wav
    try:
        wav, _ = librosa.load(path, sr=sr, mono=True, duration=max_duration)
        if len(wav) < sr * 2:
            # Pad very short audio to at least 2 seconds for audio encoder compatibility
            wav = np.pad(wav, (0, sr * 2 - len(wav)), mode='constant')
        return wav
    except Exception as e:
        logger.warning(f"Failed to load audio {path}: {e}, returning 2s silence")
        return np.zeros(sr * 2, dtype=np.float32)


def _cache_one_audio(args):
    """Module-level function for ProcessPoolExecutor (must be picklable)."""
    path, sr = args
    npy_path = path.rsplit(".", 1)[0] + ".npy" if "." in path else path + ".npy"
    wav, _ = librosa.load(path, sr=sr, mono=True)
    np.save(npy_path, wav)


def precache_audio(jsonl_paths: List[str], sr: int = 16000, num_workers: int = 8):
    """Pre-decode audio files to .npy for fast training-time loading.

    Skips files that already have a .npy cache. Uses multiprocessing
    to saturate CPU cores during the one-time preprocessing step.
    """
    import json
    from concurrent.futures import ProcessPoolExecutor, as_completed

    audio_paths = set()
    for jsonl_path in jsonl_paths:
        if not jsonl_path:
            continue
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    audio_paths.add(json.loads(line)["audio"])

    # Filter out already-cached
    to_cache = []
    for p in audio_paths:
        npy_path = p.rsplit(".", 1)[0] + ".npy" if "." in p else p + ".npy"
        if not os.path.exists(npy_path):
            to_cache.append(p)

    if not to_cache:
        logger.info("All audio files already cached as .npy, skipping precache")
        return

    logger.info(f"Pre-caching {len(to_cache)} audio files to .npy (workers={num_workers})...")

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_cache_one_audio, (p, sr)): p for p in to_cache}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
            except Exception as e:
                logger.warning(f"Failed to cache {futures[fut]}: {e}")
            if done % 2000 == 0:
                logger.info(f"  cached {done}/{len(to_cache)}")

    logger.info(f"Pre-cache complete: {done} files")


# ── Data processing ───────────────────────────────────────────────────
def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        prefix_msgs = build_prefix_messages(prompt, None)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt, "audio": ex["audio"],
            "target": ex["text"], "prefix_text": prefix_text,
        }
    return _preprocess


@dataclass
class DataCollatorForQwen3ASR:
    processor: Any
    sampling_rate: int = 16000
    max_length: int = 0  # 0 = no truncation

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        truncation = self.max_length > 0
        trunc_kwargs = {"max_length": self.max_length} if truncation else {}

        full_inputs = self.processor(
            text=full_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=truncation,
            **trunc_kwargs,
        )
        prefix_inputs = self.processor(
            text=prefix_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=truncation,
            **trunc_kwargs,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


def copy_hf_files(src_dir: str, dst_dir: str):
    """Copy config/tokenizer files so checkpoints are directly loadable."""
    os.makedirs(dst_dir, exist_ok=True)
    for fn in [
        "config.json", "generation_config.json", "preprocessor_config.json",
        "processor_config.json", "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json",
    ]:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class InferableCheckpointCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        copy_hf_files(self.base_model_path, ckpt_dir)
        return control


class EvalLoggingCallback(TrainerCallback):
    """Explicitly print eval metrics so they don't get swallowed by tqdm."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and state.is_world_process_zero:
            step = state.global_step
            filtered = {k: v for k, v in metrics.items() if not k.startswith("eval_runtime")}
            logger.info(f"[Eval @ step {step}] {filtered}")


class ProgressLogCallback(TrainerCallback):
    """Print a progress line every `logging_steps` to replace tqdm in log files."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        step = state.global_step
        total = state.max_steps
        pct = step / total * 100 if total else 0
        parts = [f"step {step}/{total} ({pct:.1f}%)"]
        for k in ("loss", "grad_norm", "learning_rate", "eval_loss"):
            if k in logs:
                parts.append(f"{k}={logs[k]:.6g}")
        if "epoch" in logs:
            parts.append(f"epoch={logs['epoch']:.3f}")
        logger.info(" | ".join(parts))



# ── CLI ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Fine-tuning for Children's Speech")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=0,
                   help="Eval batch size. 0=same as batch_size.")
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gradient_checkpointing", type=int, default=0,
                   help="1=enable gradient checkpointing to save VRAM.")
    p.add_argument("--early_stopping_patience", type=int, default=0,
                   help="Stop after N evals with no improvement. 0=disabled.")
    p.add_argument("--early_stopping_threshold", type=float, default=0.0,
                   help="Minimum change to qualify as an improvement.")
    p.add_argument("--load_best_model_at_end", type=int, default=0,
                   help="1=load best model at end (requires eval).")
    p.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)
    p.add_argument("--precache_workers", type=int, default=16,
                   help="Workers for pre-caching audio to .npy (0 to skip)")
    p.add_argument("--max_length", type=int, default=0,
                   help="Max token length for truncation. 0=no truncation.")
    return p.parse_args()


def main():
    args = parse_args()

    # Determine local rank for multi-GPU (DDP)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0

    if is_main_process:
        logger.info(f"Fine-tuning Qwen3-ASR: {args.model_path}")
        logger.info(f"Train file: {args.train_file}")
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            logger.info(f"Multi-GPU training: {world_size} GPUs")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    # Pre-cache audio to .npy (only on main process, others wait)
    if args.precache_workers > 0:
        if is_main_process:
            precache_audio(
                jsonl_paths=[args.train_file, args.eval_file],
                sr=16000,
                num_workers=args.precache_workers,
            )
        # Barrier: wait for main process to finish caching
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # Auto-append model name to output_dir
    model_name = os.path.basename(args.model_path.rstrip("/"))
    args.output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Detect flash-attn availability
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        if is_main_process:
            logger.info("FlashAttention 2 detected, enabling it")
    except ImportError:
        attn_impl = None
        if is_main_process:
            logger.info("FlashAttention 2 not found, using default attention")

    # Load model
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
        **({"attn_implementation": attn_impl} if attn_impl else {}),
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    # Load dataset
    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file

    raw_ds = load_dataset("json", data_files=data_files)
    ds = raw_ds.map(make_preprocess_fn(processor), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForQwen3ASR(processor=processor, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_prefetch_factor=2 if args.num_workers > 0 else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps" if args.eval_file else "no",
        eval_steps=args.save_steps if args.eval_file else None,
        load_best_model_at_end=bool(args.load_best_model_at_end) and bool(args.eval_file),
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=False,
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        disable_tqdm=True,
    )

    callbacks = [
        InferableCheckpointCallback(base_model_path=args.model_path),
        EvalLoggingCallback(),
        ProgressLogCallback(),
    ]
    if args.early_stopping_patience > 0 and args.eval_file:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        ))

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=callbacks,
    )

    # Resume logic
    resume_from = (args.resume_from or "").strip()
    if not resume_from and args.resume == 1:
        resume_from = find_latest_checkpoint(args.output_dir) or ""

    if resume_from:
        if is_main_process:
            logger.info(f"Resuming from: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    if is_main_process:
        if ds.get("validation") is not None:
            metrics = trainer.evaluate()
            logger.info(f"Final eval metrics: {metrics}")
        logger.success("Training complete.")


if __name__ == "__main__":
    main()