import tarfile
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from tqdm import tqdm

app = typer.Typer()

SMOKE_COLUMNS = [
    "utterance_id",
    "child_id",
    "session_id",
    "age_bucket",
    "text",
    "audio_duration_sec",
]


def _split_by_ids(
    df: pd.DataFrame, ids: set[str], label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df["utterance_id"].astype(str).isin(ids)
    matched = df[mask].reset_index(drop=True)
    remaining = df[~mask].reset_index(drop=True)
    typer.echo(f"{label}: {len(matched)} matched / {len(remaining)} remaining")
    return matched, remaining


def _extract_audio(tar_path: Path, ids: set[str], audio_out_dir: Path) -> int:
    extracted = 0
    with tarfile.open(tar_path) as tar:
        for member in tqdm(tar, desc=f"Extract {tar_path.name}"):
            if not member.isfile():
                continue
            stem = Path(member.name).stem
            if stem not in ids:
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            dest = audio_out_dir.joinpath(Path(member.name).name)
            dest.write_bytes(fileobj.read())
            extracted += 1
    return extracted


@app.command()
def main(
    a_jsonl: Annotated[Path, typer.Argument(help="submission_format jsonl (A)")],
    b_jsonl: Annotated[Path, typer.Argument(help="train metadata jsonl (B)")],
    c_jsonl: Annotated[Path, typer.Argument(help="train metadata jsonl (C)")],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="smoke.csv / audio の出力先 (既定: A の親)"),
    ] = None,
) -> None:
    out_dir = output_dir if output_dir is not None else a_jsonl.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = out_dir.joinpath("audio")
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    a_df = pd.read_json(a_jsonl, lines=True)
    smoke_ids: set[str] = set(a_df["utterance_id"].astype(str))
    typer.echo(f"Loaded {len(smoke_ids)} utterance_ids from {a_jsonl}")

    b_df = pd.read_json(b_jsonl, lines=True)
    c_df = pd.read_json(c_jsonl, lines=True)

    b_matched, b_remaining = _split_by_ids(b_df, smoke_ids, b_jsonl.name)
    c_matched, c_remaining = _split_by_ids(c_df, smoke_ids, c_jsonl.name)

    smoke_df = pd.concat([b_matched, c_matched], ignore_index=True)
    missing = smoke_ids - set(smoke_df["utterance_id"].astype(str))
    if missing:
        typer.echo(
            f"WARNING: {len(missing)} utterance_ids not found in B/C, "
            f"e.g. {list(missing)[:3]}",
        )

    smoke_df = smoke_df.rename(columns={"orthographic_text": "text"})
    smoke_df = smoke_df[[c for c in SMOKE_COLUMNS if c in smoke_df.columns]]
    smoke_csv = out_dir.joinpath("smoke.csv")
    smoke_df.to_csv(smoke_csv, index=False)
    typer.echo(f"Written {len(smoke_df)} rows to {smoke_csv}")

    b_out = b_jsonl.with_name(f"{b_jsonl.stem}_filtered.jsonl")
    c_out = c_jsonl.with_name(f"{c_jsonl.stem}_filtered.jsonl")
    b_remaining.to_json(b_out, orient="records", lines=True, force_ascii=False)
    c_remaining.to_json(c_out, orient="records", lines=True, force_ascii=False)
    typer.echo(f"Written filtered B to {b_out}")
    typer.echo(f"Written filtered C to {c_out}")

    total_extracted = 0
    for jsonl_path, matched_df in ((b_jsonl, b_matched), (c_jsonl, c_matched)):
        tar_path = jsonl_path.parent.joinpath("audio.tar")
        if not tar_path.exists():
            typer.echo(f"WARNING: {tar_path} not found, skipping")
            continue
        ids = set(matched_df["utterance_id"].astype(str))
        if not ids:
            continue
        total_extracted += _extract_audio(tar_path, ids, audio_out_dir)

    typer.echo(f"Done. {total_extracted} audio files copied to {audio_out_dir}")


if __name__ == "__main__":
    app()
