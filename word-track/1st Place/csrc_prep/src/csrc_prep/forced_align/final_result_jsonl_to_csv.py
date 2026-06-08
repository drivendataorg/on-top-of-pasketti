import csv
import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


@app.command()
def convert(
    target_dir: Annotated[Path, typer.Argument(help="target_dir")],
) -> None:
    """final_result.json を train_word_transcripts.csv と同じ形式に変換する."""
    final_result_path = target_dir.joinpath("forced_align/final_result.jsonl")
    train_csv_path = target_dir.joinpath("train_word_transcripts.csv")
    output_csv_path = target_dir.joinpath("forced_align/train_word_transcripts.csv")

    # utterance_id -> (child_id, session_id, age_bucket) のマッピングを構築
    utterance_meta: dict[str, dict[str, str]] = {}
    with train_csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            utterance_meta[row["utterance_id"]] = {
                "child_id": row["child_id"],
                "session_id": row["session_id"],
                "age_bucket": row["age_bucket"],
            }

    # JSONL を読み込んでCSVに変換
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    missing_ids: set[str] = set()
    written = 0

    skipped = 0
    with output_csv_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(
            [
                "utterance_id",
                "child_id",
                "session_id",
                "age_bucket",
                "text",
                "audio_duration_sec",
                "original_utterance_id",
            ],
        )

        with final_result_path.open(encoding="utf-8") as in_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                utterance_id = entry["utterance_id"]
                meta = utterance_meta.get(utterance_id)
                if meta is None:
                    missing_ids.add(utterance_id)
                    continue

                if entry["duration"] > 30.0:
                    print(f"Skipped {entry['id']} (audio_duration_sec > 30.0)")
                    skipped += 1
                    continue

                # id をそのまま utterance_id として使う（セグメント単位のID）
                writer.writerow(
                    [
                        entry["id"],
                        meta["child_id"],
                        meta["session_id"],
                        meta["age_bucket"],
                        entry["text"],
                        round(entry["duration"], 3),
                        entry["utterance_id"],
                    ],
                )
                written += 1

    print(f"Written {written} rows to {output_csv_path}")
    if missing_ids:
        print(f"Warning: {len(missing_ids)} utterance_ids not found in train CSV: {sorted(missing_ids)[:5]}...")

    print(f"Skipped {skipped} rows (audio_duration_sec > 30.0)")


if __name__ == "__main__":
    app()
