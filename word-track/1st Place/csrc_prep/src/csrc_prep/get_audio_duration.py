import argparse
import csv
import io
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import cast

from mutagen.mp3 import MP3
from tqdm import tqdm


def get_duration(path: Path) -> tuple[str, float]:
    audio = MP3(path)
    return (path.stem, cast("float", audio.info.length))  # type: ignore


def get_duration_from_bytes(item: tuple[str, bytes]) -> tuple[str, float]:
    name, data = item
    audio = MP3(io.BytesIO(data))
    return (Path(name).stem, cast("float", audio.info.length))  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=Path)
    parser.add_argument("-w", "--workers", type=int, default=None)
    args = parser.parse_args()

    source_dir = Path(args.target_dir) / "audio.tar"

    if tarfile.is_tarfile(source_dir):
        with tarfile.open(source_dir) as tar:
            items = []
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".mp3"):
                    f = tar.extractfile(member)
                    if f is not None:
                        items.append((member.name, f.read()))
        if not items:
            print(f"No mp3 files found in {source_dir}")
            return
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(get_duration_from_bytes, item) for item in items]
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                results.append(future.result())  # noqa
    else:
        files = sorted(source_dir.glob("*.mp3"))
        if not files:
            print(f"No mp3 files found in {source_dir}")
            return
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(get_duration, f) for f in files]
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                results.append(future.result())

    results.sort(key=lambda x: x[0])

    output_path = Path(args.target_dir) / "audio_duration.csv"
    with Path(output_path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utterance_id", "audio_duration_sec"])
        writer.writerows(results)

    print(f"Wrote {len(results)} entries to {output_path}")


if __name__ == "__main__":
    main()
