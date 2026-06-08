"""Generate predictions from a model config. Supports both submission format and manifest JSONL."""
import json
from pathlib import Path

import dirtygit
from loguru import logger
from omegaconf import OmegaConf

from src.config import CONFIG_DIR, load_config, setup_logging
from src.data.utils import load_jsonl, save_jsonl, load_audio
from src.models.factory import load_model_from_config
from src.paths import RAW_AUDIO_DIR, SUBMISSION_FORMAT_A, SUBMISSION_FORMAT_B, SUBMISSIONS_DIR

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "word" / "eval.yaml"


def parse_config():
    cfg = load_config(DEFAULT_CONFIG)
    return cfg


def predict_submission(model, cfg):
    submission_format = cfg.eval.get("submission_format", "a")
    format_path = SUBMISSION_FORMAT_A if submission_format == "a" else SUBMISSION_FORMAT_B
    submission = load_jsonl(format_path)

    batch_size = cfg.eval.get("batch_size", 16)
    predictions = {}
    for i in range(0, len(submission), batch_size):
        batch_entries = submission[i:i + batch_size]
        audio_paths = [RAW_AUDIO_DIR / f"{e['utterance_id']}.flac" for e in batch_entries]
        audios = [load_audio(p)[0] for p in audio_paths]
        texts = model.inference_batch(audios)
        for entry, text in zip(batch_entries, texts):
            predictions[entry["utterance_id"]] = text
        if (i // batch_size + 1) % 50 == 0:
            logger.info(f"{len(predictions)}/{len(submission)} done")

    for entry in submission:
        entry["orthographic_text"] = predictions.get(entry["utterance_id"], "")

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    output = cfg.eval.get("output", None)
    if output:
        output = Path(output)
    else:
        output = SUBMISSIONS_DIR / f"submission_{submission_format}.jsonl"
    save_jsonl(submission, output)
    logger.info(f"Saved {len(submission)} predictions to {output}")


def predict_manifest(model, cfg):
    manifest_path = Path(cfg.eval.manifest)
    audio_dir = manifest_path.parent / "audio"
    entries = load_jsonl(manifest_path)
    logger.info(f"Loaded {len(entries):,} entries from {manifest_path}")

    output_path = Path(cfg.eval.get("output", None) or manifest_path.parent / "predictions.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if existing_ids:
            logger.info(f"Resuming: skipping {len(existing_ids):,} already-predicted samples")

    remaining = [e for e in entries if e["id"] not in existing_ids]
    logger.info(f"Remaining: {len(remaining):,}")

    batch_size = cfg.eval.get("batch_size", 16)
    out_file = open(output_path, "a")
    predicted = 0

    try:
        for i in range(0, len(remaining), batch_size):
            batch_entries = remaining[i:i + batch_size]
            audio_paths = [audio_dir / f"{e['id']}.flac" for e in batch_entries]
            missing = [p for p in audio_paths if not p.exists()]
            if missing:
                batch_entries = [e for e, p in zip(batch_entries, audio_paths) if p.exists()]
                audio_paths = [p for p in audio_paths if p.exists()]
            if not batch_entries:
                continue

            audios = [load_audio(p)[0] for p in audio_paths]

            if hasattr(model, '_transcribe_paths_with_scores'):
                import tempfile
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_paths = []
                    for j, audio in enumerate(audios):
                        p = Path(tmp_dir) / f"{j}.wav"
                        import soundfile as sf_write
                        sf_write.write(str(p), audio, 16000)
                        tmp_paths.append(str(p))
                    results = model._transcribe_paths_with_scores(tmp_paths)
                texts = [r[0] for r in results]
                scores = [r[1] for r in results]
                n_tokens_list = [r[2] for r in results]
            else:
                texts = model.inference_batch(audios)
                scores = [None] * len(texts)
                n_tokens_list = [None] * len(texts)

            for entry, text, score, n_tokens in zip(batch_entries, texts, scores, n_tokens_list):
                record = {**entry, "predicted_text": text}
                if score is not None:
                    record["score"] = score
                    record["n_tokens"] = n_tokens
                    record["score_per_token"] = score / n_tokens if n_tokens > 0 else 0.0
                out_file.write(json.dumps(record) + "\n")
                predicted += 1

            if (i // batch_size + 1) % 50 == 0:
                out_file.flush()
                logger.info(f"{predicted:,}/{len(remaining):,} done")
    finally:
        out_file.close()

    logger.info(f"Saved {predicted:,} predictions to {output_path}")


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    logger.info(f"config:\n{OmegaConf.to_yaml(cfg)}")

    model = load_model_from_config(cfg.model)

    if cfg.eval.get("manifest", None):
        predict_manifest(model, cfg)
    else:
        predict_submission(model, cfg)


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = parse_config()
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
