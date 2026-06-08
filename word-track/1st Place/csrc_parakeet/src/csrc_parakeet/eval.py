import json
import logging
from pathlib import Path
from typing import Annotated

import nemo.collections.asr as nemo_asr
import typer
from csrc.manifest import load_manifest
from csrc.metric import score_wer
from loguru import logger
from nemo.utils import logging as nemo_logging
from tqdm import tqdm

app = typer.Typer()

nemo_logging.setLevel(logging.WARNING)


def run_inference(
    model: nemo_asr.models.ASRModel,
    manifest_entries: list[dict],
    batch_size: int = 4,
) -> list[str]:
    """manifestの音声ファイルに対して推論を実行する。"""
    audio_paths = [e["audio_filepath"] for e in manifest_entries]
    predictions = []

    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Inference"):
        batch = audio_paths[i : i + batch_size]
        hypotheses = model.transcribe(batch, batch_size=len(batch), verbose=False)
        predictions.extend([h.text for h in hypotheses])

    return predictions


@app.command()
def main(
    model_path: Annotated[Path, typer.Argument(help=".nemoモデルファイル")],
    manifest_path: Annotated[Path, typer.Argument(help="NeMo manifest JSONL")],
    batch_size: Annotated[int, typer.Option("-b", "--batch-size")] = 4,
    output_path: Annotated[Path | None, typer.Option("--output", "-o", help="推論結果の出力先JSONL")] = None,
) -> None:
    """ファインチューニング済みモデルの評価（推論 + WER計算）。"""
    # モデルロード
    logger.info(f"Loading model from {model_path}")
    model = nemo_asr.models.ASRModel.restore_from(str(model_path))

    # Manifest読み込み
    entries = load_manifest(manifest_path)
    logger.info(f"Loaded {len(entries)} entries from manifest")

    # 推論
    predictions = run_inference(model, entries, batch_size=batch_size)

    # 参照テキスト取得
    references = [e["text"] for e in entries]

    # WER計算
    wer = score_wer(references, predictions)
    logger.info(f"WER: {wer:.4f}")
    print(f"WER: {wer:.4f}")

    # 結果出力
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for entry, pred in zip(entries, predictions, strict=True):
                result = {
                    "audio_filepath": entry["audio_filepath"],
                    "reference": entry["text"],
                    "prediction": pred,
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    app()
