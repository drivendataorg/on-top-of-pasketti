import csv
import logging
from pathlib import Path
from typing import Annotated, cast

import jiwer
import nemo.collections.asr as nemo_asr
import typer
from csrc.ipa_normalize import load_reverse_dict, reverse_ipa
from csrc.manifest import load_manifest
from csrc.metric import score_wer
from csrc.normalize import normalize_orthographic
from loguru import logger
from nemo.utils import logging as nemo_logging
from omegaconf import OmegaConf
from tqdm import tqdm

app = typer.Typer()

nemo_logging.setLevel(logging.WARNING)


def _safe_text(hyp) -> str:  # noqa: ANN001
    """仮説からテキストを取得し、None/空の場合は空文字列を返す。"""
    text = hyp.text if hyp.text is not None else ""
    return text


def apply_beam_decoding(
    model: nemo_asr.models.ASRModel,
    beam_size: int = 4,
    search_type: str = "default",
) -> None:
    """beam search デコーディングに切り替える。"""
    cfg = OmegaConf.create(
        {
            "strategy": "beam",
            "beam": {
                "beam_size": beam_size,
                "search_type": search_type,
                "score_norm": True,
                "return_best_hypothesis": True,
            },
        },
    )
    model.change_decoding_strategy(cfg, verbose=True)
    logger.info(f"Decoding strategy changed to beam (beam_size={beam_size})")


def run_inference(
    model: nemo_asr.models.ASRModel,
    manifest_entries: list[dict],
    batch_size: int = 4,
) -> list[str]:
    """manifestの音声ファイルに対して推論を実行する。"""
    audio_paths = [e["audio_filepath"] for e in manifest_entries]
    predictions: list[str] = []

    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Inference"):
        batch = audio_paths[i : i + batch_size]
        hypotheses = model.transcribe(batch, batch_size=len(batch), verbose=False)  # type: ignore
        predictions.extend([_safe_text(h) for h in hypotheses])

    return predictions


@app.command()
def main(  # noqa: PLR0913
    model_path: Annotated[Path, typer.Argument(help=".nemoモデルファイル")],
    manifest_path: Annotated[Path, typer.Argument(help="NeMo manifest JSONL")],
    output_dir: Annotated[Path, typer.Argument(help="出力ディレクトリ")],
    batch_size: Annotated[int, typer.Option("-b", "--batch-size")] = 8,
    beam_size: Annotated[int, typer.Option("--beam-size", help="beam search サイズ (0=greedy)")] = 0,
    reverse_dict: Annotated[Path | None, typer.Option("--reverse-dict", help="IPA逆変換辞書JSONパス")] = None,
) -> None:
    """val_manifestに対して推論を実行し、結果をCSVで出力する。"""
    # モデルロード
    logger.info(f"Loading model from {model_path}")
    model = nemo_asr.models.ASRModel.restore_from(str(model_path))

    if beam_size > 0:
        apply_beam_decoding(model, beam_size=beam_size)

    # Manifest読み込み
    entries = load_manifest(manifest_path)
    logger.info(f"Loaded {len(entries)} entries from manifest")

    # 推論
    predictions = run_inference(model, entries, batch_size=batch_size)

    # IPA逆変換
    if reverse_dict is not None:
        rev_dict = load_reverse_dict(reverse_dict)
        predictions = [reverse_ipa(p, rev_dict) for p in predictions]
        logger.info(f"Applied IPA reverse mapping ({len(rev_dict)} entries)")

    # 行単位WER/CER計算（normalize_orthographicで正規化後に算出）
    references = [e["text"] for e in entries]
    row_wers: list[float] = []
    row_cers: list[float] = []
    for ref, pred in zip(references, predictions, strict=True):
        norm_ref = normalize_orthographic(ref)
        norm_pred = normalize_orthographic(pred)
        row_wers.append(jiwer.wer(norm_ref, norm_pred))
        row_cers.append(cast("float", jiwer.cer(norm_ref, norm_pred)))

    # 全体WER (正規化済み)
    corpus_wer = score_wer(references, predictions)
    logger.info(f"Corpus WER: {corpus_wer:.4f}")
    print(f"Corpus WER: {corpus_wer:.4f}")

    # CSV出力
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "val_pred.csv"
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["utterance_id", "duration", "text", "prediction", "wer", "cer"],
        )
        writer.writeheader()
        for entry, pred, wer, cer in zip(entries, predictions, row_wers, row_cers, strict=True):
            writer.writerow(
                {
                    "utterance_id": entry["utterance_id"],
                    "duration": entry["duration"],
                    "text": entry["text"],
                    "prediction": pred,
                    "wer": f"{wer:.4f}",
                    "cer": f"{cer:.4f}",
                },
            )

    logger.info(f"Results saved to {output_path} ({len(entries)} rows)")


if __name__ == "__main__":
    app()
