from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from csrc.filters import apply_filter
from csrc.manifest import write_manifest
from loguru import logger

from csrc_qwen.config import load_config

app = typer.Typer()

ASR_TARGET_PREFIX = "language English<asr_text>"


def _find_audio_path(audio_dir: Path, utterance_id: str) -> Path | None:
    for ext in (".mp3",):
        p = audio_dir / f"{utterance_id}{ext}"
        if p.exists():
            return p
    return None


def _df_to_entries(
    df: pd.DataFrame,
    audio_dir: Path | None = None,
    max_duration_sec: float | None = None,
) -> list[dict]:
    entries = []
    missing = 0
    skipped_duration = 0

    for _, row in df.iterrows():
        row_audio_dir = audio_dir if audio_dir is not None else Path(row["_audio_dir"])
        audio_path = _find_audio_path(row_audio_dir, str(row["utterance_id"]))

        if audio_path is None:
            missing += 1
            continue

        duration = float(row["audio_duration_sec"]) if "audio_duration_sec" in row.index else 0.0
        if max_duration_sec is not None and duration > max_duration_sec:
            skipped_duration += 1
            continue

        text = str(row["text"]).replace(" | ", " ").strip()
        text = " ".join(text.split())
        text = f"{ASR_TARGET_PREFIX}{text}"

        entries.append({
            "audio_filepath": str(audio_path),
            "text": text,
            "duration": duration,
            "utterance_id": str(row["utterance_id"]),
            "source": str(row["_source_name"]) if "_source_name" in row.index else "unknown",
        })

    if missing > 0:
        logger.warning(f"Skipped {missing} entries due to missing audio files")
    if skipped_duration > 0:
        logger.info(f"Skipped {skipped_duration} entries exceeding {max_duration_sec}s")
    logger.info(f"Created {len(entries)}/{len(df)} entries")
    return entries


@app.command()
def train(
    config: Annotated[Path, typer.Argument(help="YAML設定ファイルパス")],
) -> None:
    """設定YAMLからtrain用 manifest (JSONL) を作成する。"""
    cfg = load_config(config)

    dfs: list[pd.DataFrame] = []
    for source in cfg.data.train_sources:
        df = pd.read_csv(source.csv)
        if source.cer_threshold is not None or source.wer_threshold is not None:
            df = apply_filter(
                df,
                cer_thresholds=source.cer_threshold or {},
                wer_thresholds=source.wer_threshold,
                filter_mode=source.filter_mode,
            )
        df["_audio_dir"] = str(source.audio_dir)
        df["_source_name"] = source.source_name
        dfs.append(df)

    train_df = pd.concat(dfs, ignore_index=True)
    train_df = train_df.sample(frac=1.0).reset_index(drop=True)
    logger.info(f"Train samples: {len(train_df)}")

    entries = _df_to_entries(train_df, max_duration_sec=cfg.data.max_duration_sec)
    manifest_path = write_manifest(entries, cfg.data.train_manifest)
    logger.info(f"Train manifest: {manifest_path}")


@app.command()
def val(
    val_csv: Annotated[Path, typer.Argument(help="Validation CSVファイルパス")],
    audio_dir: Annotated[Path, typer.Argument(help="音声ファイルディレクトリ")],
    output: Annotated[Path, typer.Argument(help="出力manifestパス")],
) -> None:
    """Validation用 manifest (JSONL) を作成する。フィルタリングなし。"""
    val_df = pd.read_csv(val_csv)
    logger.info(f"Val samples: {len(val_df)}")

    entries = _df_to_entries(val_df, audio_dir=audio_dir)
    manifest_path = write_manifest(entries, output)
    logger.info(f"Val manifest: {manifest_path}")


if __name__ == "__main__":
    app()
