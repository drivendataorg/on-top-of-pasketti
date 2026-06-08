"""Speed Perturbation によるデータ拡張。

0.9x / 1.1x の速度摂動を適用して音声ファイルを生成し、拡張済みmanifestを出力する。
- 0.9x: 再生速度 0.9 → 遅い・低ピッチ（子供のゆっくりした発話を模倣）
- 1.1x: 再生速度 1.1 → 速い・高ピッチ（より幼い子供の声を模倣）
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
from typing import Annotated

import numpy as np
import scipy.signal
import soundfile as sf
import typer
from loguru import logger
from tqdm import tqdm

app = typer.Typer()

DEFAULT_SPEED_FACTORS = [0.9, 1.1]


def _speed_ratio(speed_factor: float) -> tuple[int, int]:
    """速度係数を有理数 (up, down) に変換する。resample_poly 用。"""
    frac = Fraction(speed_factor).limit_denominator(100)
    # resample_poly(audio, up, down) は len*up/down にリサンプル
    # speed_factor=1.1 → 短くしたい → new_len = len/1.1 → up=10, down=11
    return frac.denominator, frac.numerator


def speed_perturb(audio: np.ndarray, up: int, down: int) -> np.ndarray:
    """速度摂動を適用する。resample_poly で高速リサンプリング。"""
    return scipy.signal.resample_poly(audio, up, down).astype(audio.dtype)


def _process_one(args: tuple[str, str, float, int, int]) -> tuple[str, str, float, float]:
    """1ファイルを処理する（マルチプロセス用）。既存ファイルはスキップ。"""
    audio_path, out_path, speed_factor, up, down = args
    if Path(out_path).exists():
        info = sf.info(out_path)
        return audio_path, out_path, info.duration, speed_factor
    audio, sr = sf.read(audio_path)
    perturbed = speed_perturb(audio, up, down)
    new_duration = len(perturbed) / sr
    sf.write(out_path, perturbed, sr)
    return audio_path, out_path, new_duration, speed_factor


@app.command()
def main(
    manifest_path: Annotated[Path, typer.Argument(help="入力manifest (JSONL)")],
    output_dir: Annotated[Path, typer.Argument(help="摂動音声の出力ディレクトリ")],
    speed_factors: Annotated[list[float], typer.Option("--speed", help="速度摂動係数")] = DEFAULT_SPEED_FACTORS,
    include_original: Annotated[bool, typer.Option(help="元の音声もmanifestに含める")] = True,
    num_workers: Annotated[int, typer.Option("-j", "--num-workers", help="並列ワーカー数")] = 0,
) -> None:
    """Speed Perturbation を適用して拡張manifestを生成する。"""
    output_audio_dir = output_dir / "audio_speed_perturbed"
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = output_dir / "speed_perturbed_manifest.json"

    if num_workers <= 0:
        num_workers = os.cpu_count() or 4

    with manifest_path.open() as f:
        entries = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Loaded {len(entries)} entries from {manifest_path}")
    logger.info(f"Speed factors: {speed_factors}, workers: {num_workers}")

    # stem → entry の逆引きdict（O(1)ルックアップ用）
    stem_to_entry = {Path(e["audio_filepath"]).stem: e for e in entries}

    # 全タスクを構築: (audio_path, out_path, speed_factor, up, down, uid, sp_label)
    tasks: list[tuple[str, str, float, int, int, str, str]] = []
    for speed_factor in speed_factors:
        up, down = _speed_ratio(speed_factor)
        sp_label = f"sp{speed_factor:.1f}".replace(".", "")
        for entry in entries:
            stem = Path(entry["audio_filepath"]).stem
            out_path = str(output_audio_dir / f"{stem}_{sp_label}.mp3")
            tasks.append((entry["audio_filepath"], out_path, speed_factor, up, down, entry["utterance_id"], sp_label))

    logger.info(f"Processing {len(tasks)} files with {num_workers} workers...")

    # マルチプロセスで並列実行
    results: list[tuple[str, str, float, float]] = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_one, t[:5]): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Speed perturb"):
            results.append(future.result())  # noqa

    # audio_path → (uid, sp_label) の逆引き
    path_to_meta = {t[1]: (t[5], t[6]) for t in tasks}

    # manifest 構築
    augmented_entries: list[dict] = []
    if include_original:
        augmented_entries.extend(entries)

    for _audio_path, out_path, new_duration, _speed_factor in results:
        uid, sp_label = path_to_meta[out_path]
        augmented_entries.append(
            {
                "audio_filepath": out_path,
                "text": stem_to_entry[Path(_audio_path).stem]["text"],
                "duration": round(new_duration, 3),
                "utterance_id": f"{uid}_{sp_label}",
            },
        )

    with output_manifest.open("w", encoding="utf-8") as f:
        for entry in augmented_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    n_original = len(entries) if include_original else 0
    n_augmented = len(augmented_entries) - n_original
    logger.info(
        f"Wrote {len(augmented_entries)} entries to {output_manifest} (original={n_original}, augmented={n_augmented})",
    )


if __name__ == "__main__":
    app()
