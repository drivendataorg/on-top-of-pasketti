"""ノイズ音声ディレクトリからNeMo JSONL形式のマニフェストを生成する。

NeMo Lhotseのnoise_pathは拡張子で読み込み方法が変わる:
- .jsonl → Lhotse CutSet として直接パース
- .json  → NeMo manifest として LazyNeMoIterator で変換
NeMo形式 + .json拡張子で出力することで、NeMo内部で正しくCutに変換される。
"""

import json
from pathlib import Path
from typing import Annotated

import typer
from lhotse import Recording
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    audio_dir: Annotated[Path, typer.Argument(help="ノイズ音声ファイルのディレクトリ")],
    output_dir: Annotated[Path, typer.Argument(help="出力ディレクトリ")],
) -> None:
    """ノイズ音声ディレクトリからNeMo JSONL形式のマニフェストを生成する。"""
    audio_files = sorted(
        p for p in audio_dir.iterdir() if p.is_file() and p.suffix in {".mp3", ".wav", ".flac", ".ogg"}
    )
    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        raise typer.Exit(code=1)

    output_path = output_dir / "noise_manifest.json"

    count = 0
    with output_path.open("w") as f:
        for audio_path in audio_files:
            rec = Recording.from_file(audio_path)
            entry = {
                "audio_filepath": str(audio_path.resolve()),
                "duration": round(rec.duration, 3),
            }
            f.write(json.dumps(entry) + "\n")
            count += 1

    logger.info(f"Wrote {count} entries to {output_path}")


if __name__ == "__main__":
    app()
