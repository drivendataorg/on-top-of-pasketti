"""文字起こしJSONディレクトリをCSVに変換し、CER/WERメトリクスを付与する。"""

import json
from pathlib import Path
from typing import Annotated

import jiwer
import pandas as pd
import typer
from csrc.normalize import normalize_orthographic
from tqdm import tqdm

app = typer.Typer()


@app.command()
def convert(
    transcript_dir: Annotated[Path, typer.Argument(help="文字起こしJSONが格納されたディレクトリ")],
    reference_csv: Annotated[
        Path,
        typer.Argument(help="ラベルテキストとaudio_duration_secを含むCSVファイル (utterance_id列が必要)"),
    ],
    output_csv: Annotated[Path, typer.Argument(help="出力CSVのパス")],
    audio_dir: Annotated[Path, typer.Argument(help="音声ファイルが格納されたディレクトリ")],
) -> None:
    """文字起こしJSONディレクトリとリファレンスCSVからメトリクスCSVを生成する。"""
    df = pd.read_csv(reference_csv).set_index("utterance_id")

    json_paths = sorted(transcript_dir.glob("*.json"))
    if not json_paths:
        typer.echo(f"JSONファイルが見つかりません: {transcript_dir}", err=True)
        raise typer.Exit(1)

    records = []
    missing = []

    for path in tqdm(json_paths, desc="Converting"):
        uid = path.stem
        if uid not in df.index:
            missing.append(uid)
            continue

        row = df.loc[uid]
        raw_label: str = row["text"]
        label = raw_label.replace(" | ", " ")

        data = json.loads(path.read_text(encoding="utf-8"))
        pred = normalize_orthographic(data["text"])
        text_normalized = normalize_orthographic(label)

        audio_duration_sec: float = float(row["audio_duration_sec"])
        label_no_space = label.replace(" ", "")

        records.append(
            {
                "utterance_id": uid,
                "child_id": row.get("child_id"),
                "session_id": row.get("session_id"),
                "age_bucket": row.get("age_bucket"),
                "text": label,
                "text_normalized": text_normalized,
                "pred": pred,
                "audio_duration_sec": audio_duration_sec,
                "char_per_sec": len(label_no_space) / audio_duration_sec if audio_duration_sec > 0 else 0.0,
                "cer": jiwer.cer(text_normalized, pred),
                "wer": jiwer.wer(text_normalized, pred),
                "audio_filepath": audio_dir / f"{uid}.mp3",
            },
        )

    if missing:
        typer.echo(f"Warning: {len(missing)} IDがCSVに見つかりませんでした: {missing[:5]}", err=True)

    metric_df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metric_df.to_csv(output_csv, index=False)
    typer.echo(f"{len(metric_df)} 件を出力しました: {output_csv}")
    if len(metric_df) > 0:
        typer.echo(f"  平均CER: {metric_df['cer'].mean():.4f}  平均WER: {metric_df['wer'].mean():.4f}")


if __name__ == "__main__":
    app()
