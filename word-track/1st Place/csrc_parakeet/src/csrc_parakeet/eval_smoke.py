"""smoke_test.csv に対してモデルを推論し WER を計算するスクリプト。

Usage:
    # ベースライン (HuggingFace)
    uv run csrc_parakeet/src/csrc_parakeet/eval_smoke.py \
        nvidia/parakeet-tdt-0.6b-v3 \
        input/csrc-input/smoke_test.csv \
        -o output/smoke_base.csv \
        --audio-tar input/csrc-input/audio_part.tar

    # 微調整モデル (.nemo)
    uv run csrc_parakeet/src/csrc_parakeet/eval_smoke.py \
        output/parakeet-level2-exp000/checkpoints/best_model.nemo \
        input/csrc-input/smoke_test.csv \
        -o output/smoke_finetuned.csv \
        --audio-tar input/csrc-input/audio_part.tar
"""

import csv
import json
import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Annotated, cast

import jiwer
import nemo.collections.asr as nemo_asr
import typer
from csrc.metric import score_wer
from csrc.normalize import normalize_orthographic
from loguru import logger
from nemo.utils import logging as nemo_logging
from tqdm import tqdm

app = typer.Typer()

nemo_logging.setLevel(logging.WARNING)


def load_smoke_csv(csv_path: Path) -> list[dict]:
    """smoke_test.csv を読み込む。"""
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    logger.info(f"Loaded {len(rows)} rows from {csv_path}")
    return rows


def extract_audio_from_tar(
    tar_path: Path,
    utterance_ids: set[str],
    output_dir: Path,
) -> dict[str, Path]:
    """tarから必要な音声ファイルのみ抽出する。"""
    audio_map: dict[str, Path] = {}
    with tarfile.open(tar_path, "r") as tar:
        for member in tqdm(tar.getmembers(), desc="Scanning tar"):
            stem = Path(member.name).stem
            if stem in utterance_ids:
                tar.extract(member, output_dir)
                audio_map[stem] = output_dir / member.name
    logger.info(f"Extracted {len(audio_map)}/{len(utterance_ids)} audio files")
    missing = utterance_ids - set(audio_map.keys())
    if missing:
        logger.warning(f"Missing {len(missing)} audio files: {list(missing)[:5]}...")
    return audio_map


def load_model(model_path: str) -> nemo_asr.models.ASRModel:
    """モデルをロードする。.nemoファイルまたはHuggingFace名。"""
    p = Path(model_path)
    if p.is_file() and p.suffix == ".nemo":
        logger.info(f"Restoring from .nemo: {p}")
        return nemo_asr.models.ASRModel.restore_from(str(p))
    else:
        logger.info(f"Loading from pretrained: {model_path}")
        return nemo_asr.models.ASRModel.from_pretrained(model_path)


def run_inference(
    model: nemo_asr.models.ASRModel,
    audio_paths: list[str],
    batch_size: int = 16,
) -> list[str]:
    """推論を実行する。"""
    predictions: list[str] = []
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Inference"):
        batch = audio_paths[i : i + batch_size]
        hypotheses = model.transcribe(batch, batch_size=len(batch), verbose=False)
        predictions.extend([h.text for h in hypotheses])
    return predictions


@app.command()
def main(
    model_path: Annotated[str, typer.Argument(help="モデルパス (.nemo or HuggingFace名)")],
    smoke_csv: Annotated[Path, typer.Argument(help="smoke_test.csv")],
    output_path: Annotated[Path, typer.Option("-o", "--output", help="出力CSVパス")] = Path("output/smoke_eval.csv"),
    audio_tar: Annotated[Path | None, typer.Option("--audio-tar", help="音声tarファイル")] = None,
    audio_dir: Annotated[Path | None, typer.Option("--audio-dir", help="展開済み音声ディレクトリ")] = None,
    batch_size: Annotated[int, typer.Option("-b", "--batch-size")] = 16,
) -> None:
    """smoke_test.csv に対してモデルを推論し WER を計算する。"""
    rows = load_smoke_csv(smoke_csv)
    utterance_ids = {row["utterance_id"] for row in rows}

    # 音声ファイルのマッピング
    if audio_dir is not None:
        # 展開済みディレクトリから探す
        audio_map: dict[str, Path] = {}
        for uid in utterance_ids:
            for ext in [".mp3", ".flac", ".wav"]:
                candidate = audio_dir / f"{uid}{ext}"
                if candidate.exists():
                    audio_map[uid] = candidate
                    break
        logger.info(f"Found {len(audio_map)}/{len(utterance_ids)} audio files in {audio_dir}")
    elif audio_tar is not None:
        # tarから抽出
        tmp_dir = Path(tempfile.mkdtemp(prefix="smoke_audio_"))
        logger.info(f"Extracting audio to {tmp_dir}")
        audio_map = extract_audio_from_tar(audio_tar, utterance_ids, tmp_dir)
    else:
        typer.echo("--audio-tar or --audio-dir のいずれかを指定してください", err=True)
        raise typer.Exit(1)

    # 音声が見つかった行のみ処理
    valid_rows = [r for r in rows if r["utterance_id"] in audio_map]
    logger.info(f"Processing {len(valid_rows)} utterances")

    # モデルロード
    model = load_model(model_path)

    # 推論
    audio_paths = [str(audio_map[r["utterance_id"]]) for r in valid_rows]
    predictions = run_inference(model, audio_paths, batch_size=batch_size)

    # WER計算
    references = [r["text"] for r in valid_rows]

    # 行単位WER/CER
    row_wers: list[float] = []
    row_cers: list[float] = []
    for ref, pred in zip(references, predictions, strict=True):
        norm_pred = normalize_orthographic(pred)
        row_wers.append(jiwer.wer(ref, norm_pred))
        row_cers.append(cast("float", jiwer.cer(ref, norm_pred)))

    # Corpus WER (公式スコアリングと同じ)
    corpus_wer = score_wer(references, predictions)
    logger.info(f"Corpus WER: {corpus_wer:.4f} ({len(valid_rows)} utterances)")
    print(f"Corpus WER: {corpus_wer:.4f}")

    # CSV出力
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["utterance_id", "duration", "text", "prediction", "normalized_prediction", "wer", "cer"],
        )
        writer.writeheader()
        for row, pred, wer, cer in zip(valid_rows, predictions, row_wers, row_cers, strict=True):
            writer.writerow(
                {
                    "utterance_id": row["utterance_id"],
                    "duration": row["audio_duration_sec"],
                    "text": row["text"],
                    "prediction": pred,
                    "normalized_prediction": normalize_orthographic(pred),
                    "wer": f"{wer:.4f}",
                    "cer": f"{cer:.4f}",
                },
            )

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    app()
