import shutil
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from tqdm import tqdm

app = typer.Typer()


@app.command()
def merge(
    target_dir: Annotated[Path, typer.Argument(help="target_dir")],
    output_dir: Annotated[Path, typer.Argument(help="出力ディレクトリ")],
) -> None:
    base_csv = target_dir.joinpath("train_word_transcripts.csv")
    fa_csv = target_dir.joinpath("forced_align/train_word_transcripts.csv")
    base_audio_dir = target_dir.joinpath("audio")
    fa_audio_dir = target_dir.joinpath("forced_align/audio")

    base_df = pd.read_csv(base_csv)
    fa_df = pd.read_csv(fa_csv)

    split_ids = set(fa_df["original_utterance_id"].astype(str))
    before = len(base_df)
    base_df = base_df[~base_df["utterance_id"].astype(str).isin(split_ids)].reset_index(drop=True)
    typer.echo(f"Dropped {before - len(base_df)} rows from base CSV (replaced by forced-align segments)")

    base_sources = [base_audio_dir.joinpath(f"{uid}.mp3") for uid in base_df["utterance_id"].astype(str)]
    fa_sources = [fa_audio_dir.joinpath(f"{uid}.mp3") for uid in fa_df["utterance_id"].astype(str)]

    missing_base = [p for p in base_sources if not p.exists()]
    missing_fa = [p for p in fa_sources if not p.exists()]
    assert not missing_base, f"{len(missing_base)} base audio missing, e.g. {missing_base[:3]}"
    assert not missing_fa, f"{len(missing_fa)} forced-align audio missing, e.g. {missing_fa[:3]}"

    output_dir.mkdir(parents=True, exist_ok=True)
    out_audio_dir = output_dir.joinpath("audio")
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    for src in tqdm(base_sources + fa_sources, desc="Copying audio"):
        shutil.copy2(src, out_audio_dir.joinpath(src.name))

    merged = pd.concat([base_df, fa_df], ignore_index=True)
    output_csv = output_dir.joinpath("train.csv")
    merged.to_csv(output_csv, index=False)
    typer.echo(f"Written {len(merged)} rows to {output_csv}")
    typer.echo(f"Copied {len(base_sources) + len(fa_sources)} audio files to {out_audio_dir}")


if __name__ == "__main__":
    app()
