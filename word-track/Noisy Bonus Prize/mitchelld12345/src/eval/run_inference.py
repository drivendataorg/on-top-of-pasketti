"""Standalone inference script for Qwen3-ASR. Runs batched greedy decoding on a set of utterances."""
import argparse
import json
import time
from pathlib import Path

import librosa
import torch
from loguru import logger


BATCH_DUR = 300


def load_model(model_dir):
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRThinkerForConditionalGeneration
    from transformers import AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Qwen3ASRThinkerForConditionalGeneration.from_pretrained(
        str(model_dir), device_map=device, dtype=dtype,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    return model, processor


def build_text_prompt(processor):
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    base = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return base + "language English<asr_text>"


def transcribe_batch(model, processor, audios, text_prompt):
    from qwen_asr.inference.utils import detect_and_fix_repetitions

    max_dur = max(len(a) / 16000 for a in audios)
    if max_dur < 1.5: max_tokens = 16
    elif max_dur < 3: max_tokens = 24
    elif max_dur < 5: max_tokens = 32
    elif max_dur < 10: max_tokens = 48
    elif max_dur < 15: max_tokens = 72
    elif max_dur < 30: max_tokens = 128
    else: max_tokens = 192

    inputs = processor(
        text=[text_prompt] * len(audios),
        audio=audios,
        return_tensors="pt",
        padding=True,
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point()
        else v.to(device=device)
        for k, v in inputs.items()
    }
    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_tokens)

    if hasattr(generated, "sequences"):
        generated = generated.sequences

    preds = processor.batch_decode(
        generated[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [detect_and_fix_repetitions(p) for p in preds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.jsonl"))
    args = parser.parse_args()

    t0 = time.time()

    logger.info(f"Loading model from {args.model_dir}")
    model, processor = load_model(args.model_dir)
    text_prompt = build_text_prompt(processor)
    logger.info(f"Model loaded in {time.time() - t0:.0f}s")

    with open(args.input) as f:
        entries = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(entries)} utterances from {args.input}")

    dur_map = {}
    for e in entries:
        path = args.audio_dir / f"{e['utterance_id']}.flac"
        if path.exists():
            import soundfile as sf
            info = sf.info(str(path))
            dur_map[e["utterance_id"]] = info.duration

    indexed = [(i, e, dur_map.get(e["utterance_id"], 5.0)) for i, e in enumerate(entries)]
    indexed.sort(key=lambda x: x[2])

    batches = []
    current, current_dur = [], 0
    for orig_idx, entry, dur in indexed:
        if current and current_dur + dur > BATCH_DUR:
            batches.append(current)
            current, current_dur = [], 0
        current.append((orig_idx, entry, dur))
        current_dur += dur
    if current:
        batches.append(current)

    logger.info(f"Predicting in {len(batches)} batches")

    predictions = {}
    for i, batch in enumerate(batches):
        audios = []
        for _, entry, _ in batch:
            path = str(args.audio_dir / f"{entry['utterance_id']}.flac")
            audio, _ = librosa.load(path, sr=16000, dtype="float32", mono=True)
            audios.append(audio)

        try:
            texts = transcribe_batch(model, processor, audios, text_prompt)
            for (_, entry, _), text in zip(batch, texts):
                predictions[entry["utterance_id"]] = text
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                logger.warning(f"OOM on batch {i}, falling back to single-utterance decoding")
                for _, entry, _ in batch:
                    path = str(args.audio_dir / f"{entry['utterance_id']}.flac")
                    audio, _ = librosa.load(path, sr=16000, dtype="float32", mono=True)
                    texts = transcribe_batch(model, processor, [audio], text_prompt)
                    predictions[entry["utterance_id"]] = texts[0]
            else:
                raise

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            logger.info(f"  {i+1}/{len(batches)} batches, {len(predictions)}/{len(entries)} done, {elapsed:.0f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for entry in entries:
            uid = entry["utterance_id"]
            out = {"utterance_id": uid, "orthographic_text": predictions.get(uid, "")}
            f.write(json.dumps(out) + "\n")

    elapsed = time.time() - t0
    logger.info(f"Saved {len(entries)} predictions to {args.output} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
