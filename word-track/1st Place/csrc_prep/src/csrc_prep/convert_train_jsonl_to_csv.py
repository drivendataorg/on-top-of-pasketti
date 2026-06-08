import tarfile
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

app = typer.Typer()

COLUMNS = ["utterance_id", "child_id", "session_id", "age_bucket", "orthographic_text"]


@app.command()
def main(
    target_dir: Annotated[Path, typer.Argument(help="target_dir")],
) -> None:
    _df = pd.read_json(target_dir.joinpath("train_word_transcripts_filtered.jsonl"), lines=True)
    _df = _df[COLUMNS]
    audio_duration_df = pd.read_csv(target_dir.joinpath("audio_duration.csv"))

    audio_tar_path = target_dir.joinpath("audio.tar")
    with tarfile.open(audio_tar_path, "r") as tar:
        tar_utterance_ids = {
            Path(name).stem
            for name in tar.getnames()
            if name.endswith(".mp3")
        }

    before = len(_df)
    _df = _df[_df["utterance_id"].isin(tar_utterance_ids)].reset_index(drop=True)
    missing = before - len(_df)
    typer.echo(f"Filtered by audio.tar intersection: {before} -> {len(_df)} (missing: {missing})")

    df = _df.merge(audio_duration_df, on="utterance_id", how="left")
    df = df[df["audio_duration_sec"] < 1000].reset_index(drop=True)
    df = df.rename(columns={"orthographic_text": "text"})

    df.to_csv(target_dir.joinpath("train_word_transcripts.csv"), index=False)


if __name__ == "__main__":
    app()
