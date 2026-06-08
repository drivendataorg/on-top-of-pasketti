import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from csrc.filters import apply_filter
from csrc.ipa_normalize import normalize_ipa_to_ascii
from loguru import logger

from csrc_parakeet.config import load_config

app = typer.Typer()

MAX_DURATION_SEC = 30.0


def df_to_manifest(
    df: pd.DataFrame, output_path: Path, audio_dir: Path | None = None, ipa_normalize: bool = False,
) -> Path:
    """DataFrameをNeMo manifest JSONL形式で書き出す。

    audio_dir が None の場合は df["_audio_dir"] カラムから行ごとに取得する。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing = 0
    with output_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            row_audio_dir = audio_dir if audio_dir is not None else Path(row["_audio_dir"])
            audio_path = row_audio_dir / f"{row['utterance_id']}.mp3"

            if not audio_path.exists():
                missing += 1
                continue

            # pipe区切りのtextからpipeを除去（スペースは保持）
            text = str(row["text"]).replace(" | ", " ").strip()
            # 連続スペースを正規化
            text = " ".join(text.split())

            if ipa_normalize:
                text = normalize_ipa_to_ascii(text)

            entry = {
                "audio_filepath": str(audio_path),
                "text": text,
                "duration": float(row["audio_duration_sec"]) if "audio_duration_sec" in row.index else 0.0,
                "utterance_id": str(row["utterance_id"]),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    if missing > 0:
        logger.warning(f"Skipped {missing} entries due to missing audio files")
    logger.info(f"Wrote {written}/{len(df)} entries to {output_path}")
    return output_path


@app.command()
def train(
    config: Annotated[Path, typer.Argument(help="YAML設定ファイルパス")],
) -> None:
    """設定YAMLからtrain用NeMo manifest (JSONL) を作成する。"""
    cfg = load_config(config)

    dfs: list[pd.DataFrame] = []
    for source in cfg.data.train_sources:
        df = pd.read_csv(source.csv)
        if source.cer_threshold is not None or source.wer_threshold is not None:
            df = apply_filter(df, cer_thresholds=source.cer_threshold or {}, wer_thresholds=source.wer_threshold)
        if source.max_text_len is not None:
            text_len = df["text"].astype(str).str.len()
            dropped = df[text_len > source.max_text_len]
            if len(dropped) > 0:
                logger.info(
                    f"Dropping {len(dropped)} samples with text_len > {source.max_text_len} from {source.csv.name}:"
                )
                for _, row in dropped.iterrows():
                    logger.info(f"  {row['utterance_id']} (text_len={len(str(row['text']))})")
            df = df[text_len <= source.max_text_len].reset_index(drop=True)
        if source.max_duration_sec is not None:
            duration = df["audio_duration_sec"]
            dropped = df[duration > source.max_duration_sec]
            if len(dropped) > 0:
                logger.info(
                    f"Dropping {len(dropped)} samples with duration > {source.max_duration_sec}s from {source.csv.name}:"
                )
                for _, row in dropped.iterrows():
                    logger.info(f"  {row['utterance_id']} (duration={row['audio_duration_sec']:.1f}s)")
            df = df[duration <= source.max_duration_sec].reset_index(drop=True)
        df["_audio_dir"] = str(source.audio_dir)
        dfs.append(df)

    train_df = pd.concat(dfs, ignore_index=True)
    train_df = train_df.sample(frac=1.0).reset_index(drop=True)
    logger.info(f"Train samples: {len(train_df)}")

    train_manifest = df_to_manifest(
        train_df, cfg.data.train_manifest, ipa_normalize=cfg.ipa_normalize.enabled,
    )
    logger.info(f"Train manifest: {train_manifest}")


@app.command()
def val(
    val_csv: Annotated[Path, typer.Argument(help="Validation CSVファイルパス")],
    audio_dir: Annotated[Path, typer.Argument(help="音声ファイルディレクトリ")],
    output: Annotated[Path, typer.Argument(help="出力manifestパス")],
) -> None:
    """Validation用NeMo manifest (JSONL) を作成する。フィルタリングなし。"""
    val_df = pd.read_csv(val_csv)
    logger.info(f"Val samples: {len(val_df)}")

    val_manifest = df_to_manifest(val_df, output, audio_dir=audio_dir)
    logger.info(f"Val manifest: {val_manifest}")


if __name__ == "__main__":
    app()
