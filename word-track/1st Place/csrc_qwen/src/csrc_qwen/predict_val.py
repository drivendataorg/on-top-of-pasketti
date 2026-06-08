import csv
from pathlib import Path
from typing import Annotated, cast

import jiwer
import torch
import typer
from csrc.manifest import load_manifest
from csrc.metric import score_wer
from csrc.normalize import normalize_orthographic
from loguru import logger
from tqdm import tqdm

from csrc_qwen.qwen_asr.inference.qwen3_asr import ASRTranscription, Qwen3ASRModel

app = typer.Typer()


def _extract_ref(text: str) -> str:
    """manifestのtextフィールドから<asr_text>以降を抽出する。"""
    return text.split("<asr_text>", 1)[1] if "<asr_text>" in text else text


@app.command()
def main(  # noqa: PLR0913
    manifest: Annotated[Path, typer.Argument(help="入力JSONL（audio_filepath必須）")],
    output_dir: Annotated[Path, typer.Argument(help="出力ディレクトリ")],
    model: Annotated[str, typer.Argument(help="モデルパス（ベースモデル or マージ済み）")],
    batch_size: Annotated[int, typer.Option("--batch-size", help="推論バッチサイズ")] = 32,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", help="最大生成トークン数")] = 300,
    max_samples: Annotated[int | None, typer.Option("--max-samples", help="先頭N件のみ推論")] = None,
) -> None:
    """Qwen3-ASR 推論スクリプト。"""
    # モデルロード
    logger.info(f"Loading model: {model}")
    asr = Qwen3ASRModel.LLM(
        model=str(model),
        max_inference_batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    # マニフェスト読み込み
    entries = load_manifest(manifest)
    logger.info(f"Loaded {len(entries)} entries from {manifest}")

    if max_samples is not None:
        entries = entries[:max_samples]
        logger.info(f"Truncated to {len(entries)} entries")

    # 推論実行（バッチごとにprogress bar表示）
    audio_paths = [e["audio_filepath"] for e in entries]
    logger.info(f"Running transcription on {len(audio_paths)} files (max_new_tokens={max_new_tokens})...")
    results: list[ASRTranscription] = []
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Transcribing"):
        batch_paths = audio_paths[i : i + batch_size]
        batch_results = asr.transcribe(audio=batch_paths, language="English")
        results.extend(batch_results)

    # リファレンス取得・行単位WER/CER計算
    references = [_extract_ref(e.get("text", "")) for e in entries]
    predictions = [r.text for r in results]

    row_wers: list[float] = []
    row_cers: list[float] = []
    for ref, pred in zip(references, predictions, strict=True):
        norm_ref = normalize_orthographic(ref)
        norm_pred = normalize_orthographic(pred)
        row_wers.append(jiwer.wer(norm_ref, norm_pred))
        row_cers.append(cast("float", jiwer.cer(norm_ref, norm_pred)))

    # 全体WER
    corpus_wer = score_wer(references, predictions)
    logger.info(f"Corpus WER: {corpus_wer:.4f}")
    print(f"Corpus WER: {corpus_wer:.4f}")

    # CSV出力
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "val_pred.csv"
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["utterance_id", "duration", "text", "prediction", "language", "wer", "cer"],
        )
        writer.writeheader()
        for entry, result, wer, cer in zip(entries, results, row_wers, row_cers, strict=True):
            writer.writerow(
                {
                    "utterance_id": entry.get("utterance_id", ""),
                    "duration": entry.get("duration", ""),
                    "text": _extract_ref(entry.get("text", "")),
                    "prediction": result.text,
                    "language": result.language,
                    "wer": f"{wer:.4f}",
                    "cer": f"{cer:.4f}",
                },
            )

    logger.info(f"Results saved to {output_path} ({len(entries)} rows)")


if __name__ == "__main__":
    app()
