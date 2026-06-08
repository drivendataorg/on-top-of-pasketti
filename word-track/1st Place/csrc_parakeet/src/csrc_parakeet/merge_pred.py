from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from loguru import logger

app = typer.Typer()

MERGE_COLS = ["utterance_id", "prediction", "wer", "cer"]


@app.command()
def merge(
    pred_csv: Annotated[Path, typer.Option(help="予測 CSV (train_pred.csv)")],
    input_train_csv: Annotated[
        Path, typer.Option(help="input/csrc-processed-input/train.csv")
    ],
    talkbank_train_csv: Annotated[
        Path, typer.Option(help="input/csrc-processed-talkbank/train.csv")
    ],
    output_input_csv: Annotated[Path, typer.Option(help="input 側マージ結果の出力先")],
    output_talkbank_csv: Annotated[
        Path, typer.Option(help="talkbank 側マージ結果の出力先")
    ],
) -> None:
    """予測 CSV の wer/cer/prediction を各 train.csv にマージして保存する。"""
    pred_df = pd.read_csv(pred_csv)[MERGE_COLS]
    logger.info(f"Loaded predictions: {len(pred_df)} rows from {pred_csv}")

    for src, dst in [
        (input_train_csv, output_input_csv),
        (talkbank_train_csv, output_talkbank_csv),
    ]:
        base = pd.read_csv(src)
        merged = base.merge(pred_df, on="utterance_id", how="left")
        matched = merged["wer"].notna().sum()
        logger.info(
            f"{src.name}: {len(base)} rows, matched {matched} predictions -> {dst}"
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(dst, index=False)


if __name__ == "__main__":
    app()
