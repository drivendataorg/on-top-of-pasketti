"""Evaluate Qwen3-ASR WER on a test set.

Usage:
    python src/eval_wer.py \
        --model_path model/ \
        --eval_file data/utterance_metadata.jsonl
"""

import argparse
import json

import torch
from jiwer import wer, cer
from loguru import logger
from pathlib import Path
from qwen_asr import Qwen3ASRModel
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer({})


def normalize_text(text: str) -> str:
    return _normalizer(text)


def parse_target_text(text_field: str) -> str:
    if "<asr_text>" in text_field:
        return text_field.split("<asr_text>", 1)[1]
    return text_field


def main():
    p = argparse.ArgumentParser("Evaluate Qwen3-ASR WER")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--eval_file", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    args = p.parse_args()

    with open(args.eval_file, "r") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Eval samples: {len(samples)}")

    logger.info(f"Loading model: {args.model_path}")
    model = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        max_new_tokens=args.max_new_tokens,
    )
    model.model.generation_config.pad_token_id = model.processor.tokenizer.pad_token_id

    references = []
    hypotheses = []

    data_dir = Path("data")
    for i, sample in enumerate(samples):
        audio_path = str(data_dir / sample["audio_path"])
        ref_text = normalize_text(parse_target_text(sample["orthographic_text"]))

        results = model.transcribe(audio=audio_path, language="English")
        hyp_text = normalize_text(results[0].text)

        references.append(ref_text)
        hypotheses.append(hyp_text)

        if (i + 1) % 100 == 0:
            logger.info(f"[{i+1}/{len(samples)}]")

    overall_wer = wer(references, hypotheses)
    overall_cer = cer(references, hypotheses)

    logger.info("=" * 50)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Samples: {len(samples)}")
    logger.info(f"WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    logger.info(f"CER: {overall_cer:.4f} ({overall_cer*100:.2f}%)")


if __name__ == "__main__":
    main()
