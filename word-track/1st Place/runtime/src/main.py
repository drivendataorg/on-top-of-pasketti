import json

# import subprocess
import time

# import zipfile
from pathlib import Path

import yaml
from loguru import logger

DATA_DIR = Path("data")
SUBMISSION_DIR = Path("submission")
SMOKE_TEST_THRESHOLD = 10_000

from lib.config import Config  # noqa: E402
from lib.csrc.normalize import normalize_orthographic  # noqa: E402
from lib.model import ASRModel, load_model  # noqa: E402
from tqdm import tqdm  # noqa: E402


def load_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open() as f:
        items = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(items)} utterances from {manifest_path}")
    return items


def _adaptive_batch_size(max_duration: float, base_batch_size: int) -> int:
    """音声の最大duration に応じてバッチサイズを縮小する。
    Conformer attentionはシーケンス長の2乗でメモリを消費するため、
    長い音声が含まれるバッチではサイズを小さくする。
    """
    if max_duration > 120:
        return 1
    if max_duration > 60:
        return min(2, base_batch_size)
    if max_duration > 30:
        return min(4, base_batch_size)
    if max_duration > 15:
        return min(8, base_batch_size)
    return base_batch_size


def transcribe(
    model: ASRModel,
    items: list[dict],
    batch_size: int,
    model_type: str = "parakeet",
    disable_tqdm: bool = False,
) -> dict[str, str]:
    predictions = {}
    i = 0
    miniters = max(1, len(items) // 10)
    pbar = tqdm(
        total=len(items),
        desc="Transcribing",
        disable=disable_tqdm,
        miniters=miniters,
        maxinterval=float("inf"),
    )
    while i < len(items):
        max_dur = items[i]["audio_duration_sec"]  # items are sorted by duration desc
        effective_bs = batch_size if model_type == "qwen" else _adaptive_batch_size(max_dur, batch_size)
        batch = items[i : i + effective_bs]
        audio_paths = [DATA_DIR / item["audio_path"] for item in batch]
        texts = model.predict_batch(audio_paths, batch_size=effective_bs)
        for item, text in zip(batch, texts, strict=True):
            predictions[item["utterance_id"]] = text
        pbar.update(len(batch))
        i += effective_bs
    pbar.close()
    return predictions


def write_submission(predictions: dict[str, str]) -> None:
    submission_format_path = DATA_DIR / "submission_format.jsonl"
    submission_path = SUBMISSION_DIR / "submission.jsonl"
    with submission_format_path.open() as fr, submission_path.open("w") as fw:
        for line in fr:
            item = json.loads(line)
            item["orthographic_text"] = normalize_orthographic(predictions[item["utterance_id"]])
            fw.write(json.dumps(item) + "\n")
    logger.success(f"Wrote submission to {submission_path}")


def main() -> None:
    src_root = Path(__file__).parent.resolve()

    with (src_root / "config.yaml").open() as f:
        config = Config(**yaml.safe_load(f))

    model = load_model(config.model)

    items = load_manifest(DATA_DIR / "utterance_metadata.jsonl")

    is_smoke = len(items) < SMOKE_TEST_THRESHOLD

    t0 = time.monotonic()
    predictions = transcribe(
        model,
        items,
        batch_size=config.model.batch_size,
        model_type=config.model.type,
        disable_tqdm=is_smoke,
    )
    write_submission(predictions)
    elapsed = time.monotonic() - t0
    logger.info(f"Elapsed: {elapsed:.1f}s ({elapsed / 60:.1f}min)")


if __name__ == "__main__":
    main()
