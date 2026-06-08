import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def convert_flac_to_mp3(args: tuple[Path, Path, Path]) -> Path | None:
    """
    Convert a single FLAC file to MP3.

    Args:
        args: Tuple of (input_file, extract_dir, output_dir)

    Returns:
        Path to the converted MP3 file, or None if skipped due to error
    """
    input_file, extract_dir, output_dir = args

    # 相対パスを保持
    rel_path = input_file.relative_to(extract_dir)
    output_file = output_dir / rel_path.with_suffix(".mp3")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # ffmpegで変換: mono, 16kHz, VBR (quality 4)
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(input_file),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-q:a",
            "4",
            "-y",
            str(output_file),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        output_file.unlink(missing_ok=True)
        return None

    # ffprobeで出力MP3を検証
    probe = subprocess.run(  # noqa
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if probe.returncode != 0 or probe.stdout.strip() != b"1":
        output_file.unlink(missing_ok=True)
        return None

    return output_file


def process_directory(
    audio_dir: Path,
    temp_output_dir: Path,
    parallel_jobs: int,
) -> int:
    """
    Process a directory: convert all FLAC files to MP3.

    Args:
        audio_dir: Path to directory containing FLAC files
        temp_output_dir: Temporary directory for converted MP3s
        parallel_jobs: Number of parallel conversion jobs

    Returns:
        Number of files converted
    """
    print(f"\nProcessing directory {audio_dir}...")

    # FLACファイルを検索
    flac_files = list(audio_dir.rglob("*.flac")) + list(audio_dir.rglob("*.FLAC"))

    if not flac_files:
        print(f"  Warning: No FLAC files found in {audio_dir}")
        return 0

    print(f"  Found {len(flac_files)} FLAC files")
    print("  Converting to MP3 (16kHz, VBR)...")

    # 並列変換
    conversion_args = [(flac_file, audio_dir, temp_output_dir) for flac_file in flac_files]
    skipped = []

    with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
        futures = {executor.submit(convert_flac_to_mp3, args): args for args in conversion_args}

        with tqdm(total=len(flac_files), desc=f"  {audio_dir.name}", unit="file") as pbar:
            for future in as_completed(futures):
                args = futures[future]
                try:
                    result = future.result()
                    if result is None:
                        skipped.append(args[0])
                except Exception as e:
                    skipped.append(args[0])
                    print(f"\n  Error converting {args[0]}: {e}")
                pbar.update(1)

    if skipped:
        print(f"  Skipped {len(skipped)} files due to errors:")
        for f in skipped:
            print(f"    {f}")

    return len(flac_files) - len(skipped)


def process_zip_file(
    zip_path: Path,
    temp_extract_dir: Path,
    temp_output_dir: Path,
    parallel_jobs: int,
) -> int:
    """
    Process a single ZIP file: extract and convert all FLAC files.

    Args:
        zip_path: Path to input ZIP file
        temp_extract_dir: Temporary directory for extraction
        temp_output_dir: Temporary directory for converted MP3s
        parallel_jobs: Number of parallel conversion jobs

    Returns:
        Number of files converted
    """
    print(f"\nProcessing {zip_path.name}...")

    # ZIPを解凍
    extract_dir = temp_extract_dir / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)

    print("  Extracting...")
    subprocess.run(
        ["unzip", "-q", str(zip_path), "-d", str(extract_dir)],
        check=True,
    )

    return process_directory(extract_dir, temp_output_dir, parallel_jobs)


def create_tar(source_dir: Path, output_file: Path) -> None:
    """
    Create a tar archive from a directory.

    Args:
        source_dir: Directory to archive
        output_file: Output tar file path
    """
    print(f"\nCreating {output_file.name}...")

    with tarfile.open(output_file, "w") as tar:
        # 進捗表示付きで追加
        files = list(source_dir.rglob("*"))
        files = [f for f in files if f.is_file()]

        with tqdm(total=len(files), desc="  Archiving", unit="file") as pbar:
            for file_path in files:
                arcname = file_path.relative_to(source_dir)
                tar.add(file_path, arcname=arcname)
                pbar.update(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FLAC files to MP3 and create a single tar")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input ZIP files or directories containing FLAC files",
    )
    parser.add_argument(
        "output_tar",
        type=Path,
        help="Output tar file",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: CPU count)",
    )

    args = parser.parse_args()

    parallel_jobs = args.jobs or os.cpu_count() or 4

    # 入力パスの存在確認
    for input_path in args.inputs:
        if not input_path.exists():
            print(f"Error: {input_path} not found")
            return

    # 必要なコマンドの確認
    has_zips = any(p.is_file() for p in args.inputs)
    required_cmds = ["ffmpeg"]
    if has_zips:
        required_cmds.append("unzip")
    for cmd in required_cmds:
        if not shutil.which(cmd):
            print(f"Error: {cmd} is not installed")
            return

    # 一時ディレクトリ作成
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        temp_extract_dir = temp_path / "extracted"
        temp_output_dir = temp_path / "converted"
        temp_extract_dir.mkdir()
        temp_output_dir.mkdir()

        total_converted = 0

        for input_path in args.inputs:
            if input_path.is_dir():
                converted = process_directory(
                    input_path,
                    temp_output_dir,
                    parallel_jobs,
                )
            else:
                converted = process_zip_file(
                    input_path,
                    temp_extract_dir,
                    temp_output_dir,
                    parallel_jobs,
                )
            total_converted += converted

        if total_converted == 0:
            print("\nError: No files were converted")
            return

        # tarを作成
        create_tar(temp_output_dir, args.output_tar)

    print("\nConversion complete!")
    print(f"Output: {args.output_tar.resolve()}")
    print(f"Total files converted: {total_converted}")


if __name__ == "__main__":
    main()
