"""Decode smoke test log to recover utterance_ids and analyze distribution.

Usage:
    uv run csrc/src/csrc/decode_smoke_log.py logs/smoke_log.txt \
        --train-csv input/csrc-input/train_transcript_result.csv \
        -o output/smoke_test_utterances.csv
"""

import base64
import re
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

app = typer.Typer()


def extract_b64_from_log(log_path: Path) -> str:
    """Extract base64 payload between [SMOKE_IDS_BEGIN] and [SMOKE_IDS_END]."""
    text = log_path.read_text()
    lines = text.splitlines()

    collecting = False
    b64_parts: list[str] = []
    for line in lines:
        if "[SMOKE_IDS_BEGIN]" in line:
            collecting = True
            continue
        if "[SMOKE_IDS_END]" in line:
            break
        if collecting:
            # Strip any whitespace; the line should be pure base64
            cleaned = line.strip()
            # Skip lines that look like loguru prefixes without payload
            if cleaned and re.match(r"^[A-Za-z0-9+/=]+$", cleaned):
                b64_parts.append(cleaned)

    if not b64_parts:
        raise ValueError("No base64 payload found between [SMOKE_IDS_BEGIN] and [SMOKE_IDS_END]")

    return "".join(b64_parts)


def decode_utterance_ids(b64: str) -> list[str]:
    """Decode base64 string to list of utterance_ids (U_ + 16 hex chars)."""
    raw = base64.b64decode(b64)
    if len(raw) % 8 != 0:
        raise ValueError(f"Decoded bytes length {len(raw)} is not a multiple of 8")

    ids = []
    for i in range(0, len(raw), 8):
        chunk = raw[i : i + 8]
        ids.append("U_" + chunk.hex())
    return ids


@app.command()
def main(
    log_path: Annotated[Path, typer.Argument(help="Path to the smoke test log file")],
    train_csv: Annotated[
        Path | None,
        typer.Option("--train-csv", help="Path to train_transcript_result.csv for LEFT JOIN"),
    ] = None,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Output CSV path")] = None,
) -> None:
    """Decode utterance_ids from smoke test log and analyze distribution."""
    # 1. Extract and decode
    b64 = extract_b64_from_log(log_path)
    utterance_ids = decode_utterance_ids(b64)
    typer.echo(f"Decoded {len(utterance_ids)} utterance_ids from log")

    smoke_df = pd.DataFrame({"utterance_id": utterance_ids})

    # 2. LEFT JOIN with train metadata if provided
    if train_csv is not None:
        train_df = pd.read_csv(train_csv)
        typer.echo(f"Loaded {len(train_df)} rows from {train_csv}")

        # Deduplicate train_df to one row per utterance_id
        meta_cols = ["utterance_id"]
        meta_cols.extend(
            col
            for col in ["child_id", "session_id", "age_bucket", "text", "audio_duration_sec"]
            if col in train_df.columns
        )
        train_meta = train_df[meta_cols].drop_duplicates(subset=["utterance_id"])

        merged = smoke_df.merge(train_meta, on="utterance_id", how="left")
        matched = merged["age_bucket"].notna().sum() if "age_bucket" in merged.columns else 0
        typer.echo(f"Match rate: {matched}/{len(merged)} ({matched / len(merged) * 100:.1f}%)")

        # Distribution summary
        if "age_bucket" in merged.columns:
            typer.echo("\n--- age_bucket distribution ---")
            dist = merged["age_bucket"].value_counts(dropna=False).sort_index()
            for bucket, cnt in dist.items():
                pct = cnt / len(merged) * 100
                typer.echo(f"  {bucket}: {cnt} ({pct:.1f}%)")

        if "child_id" in merged.columns:
            typer.echo(f"\nUnique children: {merged['child_id'].nunique()}")

        if "audio_duration_sec" in merged.columns:
            dur = merged["audio_duration_sec"]
            typer.echo(
                f"\naudio_duration: mean={dur.mean():.2f}s, "
                f"median={dur.median():.2f}s, "
                f"min={dur.min():.2f}s, max={dur.max():.2f}s, "
                f"total={dur.sum() / 3600:.2f}h",
            )
    else:
        merged = smoke_df

    # 3. Save output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output, index=False)
        typer.echo(f"\nSaved to {output}")


if __name__ == "__main__":
    app()
