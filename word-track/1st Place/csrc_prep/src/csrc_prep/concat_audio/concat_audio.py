import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from tqdm import tqdm

app = typer.Typer()


def _process_row(
    _id: str,
    audio_paths: list[str],
    margin_ms: int,
    output_dir: str,
) -> str | None:
    """1行分のaudio concat処理."""
    if not audio_paths:
        return f"Warning: no audio for {_id}"

    output_path = Path(output_dir) / f"{_id}.mp3"

    if margin_ms == 0:
        # ストリームコピー (再エンコードなし)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for path in audio_paths:
                f.write(f"file '{Path(path.strip()).resolve()}'\n")
            list_path = f.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", str(output_path)],
                capture_output=True,
                check=True,
            )
        finally:
            Path(list_path).unlink()
    else:
        # 無音挿入あり: ffmpeg filter_complexで結合
        inputs: list[str] = []
        filter_parts: list[str] = []
        for i, path in enumerate(audio_paths):
            inputs.extend(["-i", str(Path(path.strip()).resolve())])
            filter_parts.append(f"[{i}:a]")

        # 各音声の間に無音を挿入
        n = len(audio_paths)
        filter_str = ""
        for i in range(n):
            filter_str += f"[{i}:a]"
            if i < n - 1:
                # adelayの代わりにapad+atrimで無音生成
                silence_sec = margin_ms / 1000.0
                filter_str = ""
                # anullsrcで無音を生成してconcat
                filter_inputs = []
                for j in range(n):
                    filter_inputs.append(f"[{j}:a]")
                    if j < n - 1:
                        filter_inputs.append(f"[s{j}]")

                silence_filters = []
                for j in range(n - 1):
                    silence_filters.append(  # noqa
                        f"anullsrc=r=16000:cl=mono[s{j}_raw];[s{j}_raw]atrim=0:{silence_sec}[s{j}]",
                    )

                filter_str = (
                    ";".join(silence_filters) + ";" + "".join(filter_inputs) + f"concat=n={2 * n - 1}:v=0:a=1[out]"
                )
                break

        cmd = [*inputs, "-filter_complex", filter_str, "-map", "[out]", str(output_path)]
        subprocess.run(
            ["ffmpeg", "-y", *cmd],
            capture_output=True,
            check=True,
        )

    return None


@app.command()
def concat_audio(
    csv_path: Annotated[str, typer.Argument(help="concat_train.csv のパス")],
    output_dir: Annotated[str, typer.Argument(help="出力ディレクトリのパス")],
    margin_ms: Annotated[int, typer.Option("-m", "--margin-ms", help="各音声間に挿入する無音の長さ（ミリ秒）")] = 0,
    n_jobs: Annotated[int, typer.Option("-j", "--n-jobs", help="並列ワーカー数（デフォルト: CPU数）")] = 0,
    debug: Annotated[bool, typer.Option("--debug", help="最初の1件だけ逐次実行してデバッグ出力")] = False,
) -> None:
    """CSVの audio_filepaths カラムに記載されたパスのMP3をconcatして出力する."""
    df = pd.read_csv(csv_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 各行のタスクを準備し、全パスの存在を事前検証
    tasks: list[tuple[str, list[str]]] = []
    missing_all: list[str] = []
    for _, row in df.iterrows():
        _id = str(row["id"])
        paths = [p.strip() for p in str(row["audio_filepaths"]).split(",")]
        missing = [p for p in paths if not Path(p).exists()]
        missing_all.extend(missing)
        tasks.append((_id, paths))

    if missing_all:
        for m in missing_all:
            print(f"Error: file not found: {m}")
        raise typer.Exit(code=1)

    total_files = sum(len(p) for _, p in tasks)
    print(f"All {total_files} audio files verified.")

    workers = n_jobs if n_jobs > 0 else os.cpu_count() or 4
    print(f"Processing {len(tasks)} rows with {workers} workers ...")

    if margin_ms == 0:
        print("Using ffmpeg concat demuxer (no re-encoding).")
    else:
        print(f"Using ffmpeg filter_complex with {margin_ms}ms silence margin.")

    if debug:
        _id, paths = tasks[0]
        print(f"[DEBUG] id={_id}, paths={paths}")
        result = _process_row(_id, paths, margin_ms, output_dir)
        if result:
            print(result)
        else:
            output_path = Path(output_dir) / f"{_id}.mp3"
            print(f"[DEBUG] output exists: {output_path.exists()}")
            if output_path.exists():
                print(f"[DEBUG] output size: {output_path.stat().st_size} bytes")
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_row, _id, paths, margin_ms, output_dir): _id for _id, paths in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Concatenating"):
            warn = future.result()
            if warn:
                print(warn)

    print(f"Done. Output saved to {output_dir}")


if __name__ == "__main__":
    app()
