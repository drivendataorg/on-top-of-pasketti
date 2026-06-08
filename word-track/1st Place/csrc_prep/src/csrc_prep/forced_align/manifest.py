import json
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer()


@app.command()
def generate_manifest(target_dir: Path) -> None:
    """train_concat.csvからNeMo manifest形式のJSONLファイルを生成する.

    CSVにはid, textカラムが必要。audio_filepathはaudio_dir/id{audio_ext}として生成される。
    """
    df = pd.read_csv(target_dir.joinpath("train_word_transcripts.csv"))
    audio_dir = target_dir.joinpath("audio")

    if "utterance_id" not in df.columns or "text" not in df.columns or "audio_duration_sec" not in df.columns:
        msg = "csv_path is not valid"
        raise typer.BadParameter(msg)

    output_dir = target_dir.joinpath("forced_align")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir.joinpath("forced_align_manifest.jsonl")

    skipped = 0
    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            audio_filepath = audio_dir / f"{row['utterance_id']}.mp3"
            if not audio_filepath.exists():
                skipped += 1
                continue
            if row["audio_duration_sec"] < 30.0:
                # 30sec未満の音声はforced alignしない
                continue
            entry = {
                "utterance_id": row["utterance_id"],
                "audio_filepath": str(audio_filepath.resolve()),
                "text": str(row["text"]),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"Manifest generated: {output_path} ({written} entries, {skipped} skipped)")


if __name__ == "__main__":
    app()
