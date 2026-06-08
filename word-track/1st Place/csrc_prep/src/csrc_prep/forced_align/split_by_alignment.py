import io
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from pydub import AudioSegment
from tqdm import tqdm

app = typer.Typer()


@dataclass
class WordInfo:
    start: float
    duration: float
    word: str

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Segment:
    start: float
    end: float
    words: list[str]

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return " ".join(self.words)


def parse_ctm_words(ctm_path: Path) -> list[WordInfo]:
    """word-level CTMファイルをパースする.

    CTMフォーマット: file_id channel start duration word ...
    """
    words: list[WordInfo] = []
    with ctm_path.open() as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                continue
            start = float(parts[2])
            duration = float(parts[3])
            word = parts[4]
            words.append(WordInfo(start=start, duration=duration, word=word))
    return words


def segment_words(words: list[WordInfo], max_duration_sec: float) -> list[Segment]:
    """単語列をmax_duration_sec以下のセグメントに分割する.

    なるべくmax_duration_secに近づけるよう貪欲に詰める。
    分割ポイントは無音ギャップが最大の位置を選ぶ。
    """
    if not words:
        return []

    segments: list[Segment] = []
    seg_start_idx = 0

    while seg_start_idx < len(words):
        seg_start_time = words[seg_start_idx].start

        # max_durationに収まる最後の単語を探す
        last_valid_idx = seg_start_idx
        for i in range(seg_start_idx, len(words)):
            seg_end_time = words[i].end
            if seg_end_time - seg_start_time > max_duration_sec and i > seg_start_idx:
                break
            last_valid_idx = i

        # 残りの単語が全部収まる場合
        if last_valid_idx == len(words) - 1:
            seg = Segment(
                start=seg_start_time,
                end=words[last_valid_idx].end,
                words=[w.word for w in words[seg_start_idx : last_valid_idx + 1]],
            )
            segments.append(seg)
            break

        # 分割ポイントを決める: seg_start_idx+1 ~ last_valid_idx の間で
        # 最大の無音ギャップを持つ位置で切る
        best_split_idx = last_valid_idx
        best_gap = -1.0

        for i in range(seg_start_idx + 1, last_valid_idx + 1):
            gap = words[i].start - words[i - 1].end
            if gap > best_gap:
                best_gap = gap
                best_split_idx = i - 1  # この単語までをセグメントに含める

        seg = Segment(
            start=seg_start_time,
            end=words[best_split_idx].end,
            words=[w.word for w in words[seg_start_idx : best_split_idx + 1]],
        )
        segments.append(seg)
        seg_start_idx = best_split_idx + 1

    return segments


def _load_manifest_file_ids(manifest_path: Path) -> list[str]:
    """manifestからfile_idリストを読み込む."""
    entries: list[dict] = []
    with manifest_path.open() as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            entries.append(json.loads(stripped))
    return [Path(e["audio_filepath"]).stem for e in entries]


def _load_audio_files(audio_dir: Path, target_ids: set[str]) -> dict[str, bytes]:
    """tarからtarget_idsに該当するaudioをメモリに読み込む."""
    audio_map: dict[str, bytes] = {}
    for audio_path in audio_dir.glob("*.mp3"):
        if audio_path.stem in target_ids:
            audio_map[audio_path.stem] = audio_path.read_bytes()
    return audio_map


def _export_segment(
    audio: AudioSegment,
    seg: Segment,
    seg_id: str,
    output_dir: Path,
    margin_start_sec: float,
    margin_end_sec: float,
) -> dict:
    """1セグメントを切り出してファイルに書き出し、manifest entryを返す."""
    audio_duration_ms = len(audio)
    start_ms = max(0, int((seg.start - margin_start_sec) * 1000))
    end_ms = min(audio_duration_ms, int((seg.end + margin_end_sec) * 1000))

    chunk = audio[start_ms:end_ms]
    out_path = output_dir / f"{seg_id}.mp3"

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        chunk.export(tmp.name, format="mp3")
        Path(tmp.name).rename(out_path)

    return {
        "id": seg_id,
        "utterance_id": seg_id.rsplit("_", 1)[0],
        "audio_filepath": str(out_path.resolve()),
        "text": seg.text,
        "duration": round(len(chunk) / 1000, 3),
        "original_start": round(seg.start, 3),
        "original_end": round(seg.end, 3),
    }


@app.command()
def split(
    target_dir: Annotated[Path, typer.Argument(help="target_dir")],
    max_duration_sec: Annotated[float, typer.Option("--max-duration", help="セグメントの最大長(秒)")] = 30.0,
    margin_start_sec: Annotated[
        float,
        typer.Option("--margin-start", help="セグメント前方に追加するマージン(秒)"),
    ] = 0.05,
    margin_end_sec: Annotated[float, typer.Option("--margin-end", help="セグメント末尾に追加するマージン(秒)")] = 0.35,
) -> None:
    """Forced Alignmentのword CTMを使って長い音声を30秒以下に分割する."""
    manifest_path = target_dir.joinpath("forced_align/forced_align_manifest.jsonl")
    ctm_words_dir = target_dir.joinpath("forced_align/ctm/words")
    output_dir = target_dir.joinpath("forced_align/audio")
    output_manifest = target_dir.joinpath("forced_align/final_result.jsonl")

    audio_dir = target_dir.joinpath("audio")
    file_ids = _load_manifest_file_ids(manifest_path)
    print(f"Manifest: {len(file_ids)} files to split")

    print(f"Loading audio from {audio_dir} ...")
    audio_map = _load_audio_files(audio_dir, set(file_ids))
    print(f"Loaded {len(audio_map)} audio files from tar")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    total_segments = 0
    with output_manifest.open("w", encoding="utf-8") as mf:
        for utterance_id in tqdm(file_ids, desc="Splitting"):
            ctm_path = ctm_words_dir / f"{utterance_id}.ctm"
            if not ctm_path.exists() or utterance_id not in audio_map:
                continue

            words = parse_ctm_words(ctm_path)
            if not words:
                continue

            segments = segment_words(words, max_duration_sec - margin_start_sec - margin_end_sec)
            audio = AudioSegment.from_mp3(io.BytesIO(audio_map[utterance_id]))

            for seg_idx, seg in enumerate(segments):
                seg_id = f"{utterance_id}_{seg_idx:03d}"
                entry = _export_segment(audio, seg, seg_id, output_dir, margin_start_sec, margin_end_sec)
                mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total_segments += 1

    print(f"Done. {total_segments} segments written to {output_dir}")
    print(f"Manifest: {output_manifest}")


if __name__ == "__main__":
    app()
