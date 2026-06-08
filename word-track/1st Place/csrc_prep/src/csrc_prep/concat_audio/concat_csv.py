"""child_id, session_id をユニークとして label CSV をconcatするスクリプト."""

import hashlib
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from csrc_prep.concat_audio.filters import FilterLevel, apply_filter

MAX_DURATION_SEC = 30.0


def concat_csv(
    input_paths: list[str],
    output_path: Path,
    filter_level: FilterLevel = FilterLevel.NONE,
) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in input_paths], ignore_index=True)
    df = df.sort_values(["child_id", "session_id", "utterance_id"]).reset_index(drop=True)

    df = apply_filter(df, filter_level)

    rows: list[dict] = []
    for (child_id, session_id), group in df.groupby(["child_id", "session_id"]):  # type: ignore
        age_bucket = group.iloc[0]["age_bucket"]
        concat_idx = 0
        current_duration = 0.0
        current_texts: list[str] = []
        current_preds: list[str] = []
        current_audio_paths: list[str] = []
        current_ids: list[str] = []
        current_cer: list[float] = []
        current_wer: list[float] = []
        for _, row in group.iterrows():
            dur = row["audio_duration_sec"]

            if current_duration + dur > MAX_DURATION_SEC and current_ids:
                total_duration = round(current_duration, 3)
                concat_text = " | ".join(current_texts)
                rows.append(
                    {
                        "id": hashlib.md5(f"{child_id}_{session_id}_{concat_idx}".encode()).hexdigest()[:10],
                        "child_id": child_id,
                        "session_id": session_id,
                        "age_bucket": age_bucket,
                        "source_utterance_ids": ",".join(current_ids),
                        "num_utterances": len(current_ids),
                        "audio_duration_sec": total_duration,
                        "char_per_sec": round(len(concat_text.replace(" | ", "")) / total_duration, 3)
                        if total_duration > 0
                        else 0.0,
                        "text": concat_text,
                        "pred": " | ".join(current_preds),
                        "cer": round(sum(current_cer) / len(current_cer), 6) if current_cer else None,
                        "wer": round(sum(current_wer) / len(current_wer), 6) if current_wer else None,
                        "audio_filepaths": ",".join(current_audio_paths),
                    },
                )
                concat_idx += 1
                current_duration = 0.0
                current_texts = []
                current_preds = []
                current_audio_paths = []
                current_ids = []
                current_cer = []
                current_wer = []

            current_ids.append(str(row["utterance_id"]))
            current_audio_paths.append(str(row["audio_filepath"]))
            current_duration += dur
            current_texts.append(str(row["text"]))
            current_preds.append(str(row["pred"]))
            if pd.notna(row["cer"]):
                current_cer.append(float(row["cer"]))
            if pd.notna(row["wer"]):
                current_wer.append(float(row["wer"]))

        if current_ids:
            total_duration = round(current_duration, 3)
            concat_text = " | ".join(current_texts)
            rows.append(
                {
                    "id": hashlib.md5(f"{child_id}_{session_id}_{concat_idx}".encode()).hexdigest()[:10],
                    "child_id": child_id,
                    "session_id": session_id,
                    "age_bucket": age_bucket,
                    "source_utterance_ids": ",".join(current_ids),
                    "num_utterances": len(current_ids),
                    "audio_duration_sec": total_duration,
                    "char_per_sec": round(len(concat_text.replace(" | ", "")) / total_duration, 3)
                    if total_duration > 0
                    else 0.0,
                    "text": concat_text,
                    "pred": " | ".join(current_preds),
                    "cer": round(sum(current_cer) / len(current_cer), 6) if current_cer else None,
                    "wer": round(sum(current_wer) / len(current_wer), 6) if current_wer else None,
                    "audio_filepaths": ",".join(current_audio_paths),
                },
            )

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} concat entries to {output_path}")
    return result


def main(
    input_paths: Annotated[list[Path], typer.Argument(..., help="入力CSVファイルパス（複数指定可」")],
    output_path: Annotated[Path, typer.Option("-o", "--output", help="出力CSVファイルパス")],
    filter_level: Annotated[
        FilterLevel,
        typer.Option(
            "--filter",
            "-f",
            help="フィルタリングレベル (none/level1/level2)",
        ),
    ] = FilterLevel.NONE,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_csv([str(p) for p in input_paths], output_path, filter_level=filter_level)


if __name__ == "__main__":
    typer.run(main)
