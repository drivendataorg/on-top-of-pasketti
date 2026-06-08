#!/usr/bin/env python3
"""
Diarization Auditor (pyannote)
===============================================================================
Runs pyannote speaker diarization on a stratified sample from training data to
estimate how often utterances contain multiple speakers.

Loads:
  - data/train_phon_transcripts.jsonl
  - data/train_phon_transcripts_talkbank.jsonl

Outputs:
  - Speaker-count distribution (1, 2, 3+ speakers)
  - Per-utterance diarization summary JSON
  - Multi-speaker sample JSON (num_speakers > 1)

Usage:
  uv run exploration/diarization_auditor.py --sample-size 250

Notes:
  - Uses CUDA when available and logs active device.
  - If pyannote model access is gated, set HF_TOKEN in env.
"""

import argparse
import dataclasses
import importlib
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.get_json import get_data_from_json
torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cuda.matmul.fp32_precision = 'ieee'
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


def _patch_torchaudio_for_pyannote() -> None:
    """Patch compatibility shims required by pyannote with newer torchaudio builds."""
    if not hasattr(torchaudio, "AudioMetaData"):
        @dataclasses.dataclass
        class AudioMetaData:
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int | None = None
            encoding: str | None = None

        torchaudio.AudioMetaData = AudioMetaData

    if not hasattr(torchaudio, "list_audio_backends"):
        def _list_audio_backends() -> list[str]:
            return ["ffmpeg", "soundfile"]

        torchaudio.list_audio_backends = _list_audio_backends

    if "torchaudio.backend.common" not in sys.modules:
        backend_mod = ModuleType("torchaudio.backend")
        common_mod = ModuleType("torchaudio.backend.common")
        common_mod.AudioMetaData = torchaudio.AudioMetaData
        backend_mod.common = common_mod
        sys.modules["torchaudio.backend"] = backend_mod
        sys.modules["torchaudio.backend.common"] = common_mod


def _patch_torch_serialization_for_pyannote() -> None:
    """Allowlist classes required by older pyannote checkpoints on PyTorch >=2.6."""
    safe_globals: list[object] = []

    try:
        from torch.torch_version import TorchVersion

        safe_globals.append(TorchVersion)
    except Exception:
        pass

    # Pyannote checkpoint classes commonly referenced during torch.load
    # when weights_only=True (PyTorch >=2.6 default).
    try:
        from pyannote.audio.core.task import Specifications

        safe_globals.append(Specifications)
    except Exception:
        pass

    try:
        from pyannote.audio.core.model import Specifications as ModelSpecifications

        safe_globals.append(ModelSpecifications)
    except Exception:
        pass

    if safe_globals:
        try:
            torch.serialization.add_safe_globals(safe_globals)
        except Exception:
            pass


def _allowlist_global_from_error(exc: Exception) -> bool:
    """Parse unsupported GLOBAL from torch error text and allowlist it dynamically."""
    message = str(exc)
    match = re.search(r"Unsupported global:\s+GLOBAL\s+([\w\.]+)", message)
    if not match:
        return False

    dotted = match.group(1)
    parts = dotted.split(".")
    if len(parts) < 2:
        return False

    module_name = ".".join(parts[:-1])
    attr_name = parts[-1]

    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, attr_name)
        torch.serialization.add_safe_globals([obj])
        logger.info(f"Allowlisted torch safe global: {dotted}")
        return True
    except Exception:
        return False


_patch_torchaudio_for_pyannote()
_patch_torch_serialization_for_pyannote()
from pyannote.audio import Pipeline


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "exploration" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATHS = [
    DATA_DIR / "train_phon_transcripts.jsonl",
    DATA_DIR / "train_phon_transcripts_talkbank.jsonl",
]

DEFAULT_SAMPLE_SIZE = 250
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MIN_AUDIO_DURATION_SEC = 0.0


def load_jsonl_data(paths: list[Path]) -> list[dict[str, Any]]:
    """Load and deduplicate JSONL rows by utterance_id."""
    seen_ids: set[str] = set()
    all_data: list[dict[str, Any]] = []

    for path in paths:
        if not path.exists():
            logger.warning(f"Path does not exist: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                item = json.loads(raw)
                uid = item.get("utterance_id")
                if uid and uid not in seen_ids:
                    seen_ids.add(uid)
                    item["source"] = path.stem
                    all_data.append(item)

    logger.info(f"Loaded {len(all_data)} unique utterances from {len(paths)} files")
    return all_data


def resolve_existing_path(path_str: str) -> Path:
    raw_path = Path(path_str).expanduser()
    candidates = [raw_path, PROJECT_ROOT / raw_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def resolve_run_dir(path_str: str | Path) -> Path:
    requested = resolve_existing_path(str(path_str))
    hydra_config = requested / ".hydra" / "config.yaml"
    if hydra_config.exists():
        return requested
    if requested.name.startswith("fold_") and (requested.parent / ".hydra" / "config.yaml").exists():
        return requested.parent
    raise FileNotFoundError(
        "Could not resolve run dir for val split. Provide a run output dir with .hydra/config.yaml"
    )


def load_val_data_from_run(run_path: str | Path, fold: int) -> list[dict[str, Any]]:
    if fold < 1:
        raise ValueError("--fold must be >= 1")

    run_dir = resolve_run_dir(run_path)
    config_path = run_dir / ".hydra" / "config.yaml"
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(config_path), resolve=False))
    fold_index = fold - 1

    all_data = get_data_from_json(cfg, inference=False)
    _, val_data = instantiate(cfg.cv.splitter)(all_data=all_data, fold=fold_index)
    logger.info(f"Loaded val subset from run={run_dir.name}, fold={fold}: {len(val_data)} utterances")
    return val_data


def resolve_audio_path(item: dict[str, Any]) -> Path | None:
    """Resolve audio file path from various possible locations."""
    audio_path = item.get("audio_path", "")

    candidates = [
        DATA_DIR / "audio" / Path(audio_path).name,
        DATA_DIR / "audio" / audio_path,
        DATA_DIR / Path(audio_path),
        PROJECT_ROOT / "data" / audio_path,
        PROJECT_ROOT / audio_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def stratified_sample(data: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    """Sample approximately uniformly across source datasets."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in data:
        source = item.get("source", "unknown")
        by_source[source].append(item)

    logger.info(f"Data distribution by source: { {k: len(v) for k, v in by_source.items()} }")

    samples: list[dict[str, Any]] = []
    sources = list(by_source.keys())
    if not sources:
        return samples

    items_per_source = max(1, sample_size // len(sources))

    for source in sources:
        source_data = by_source[source]
        #only select from samples longer than MIN_AUDIO_DURATION_SEC
        source_data = [item for item in source_data if item.get("audio_duration_sec", 0) >= MIN_AUDIO_DURATION_SEC]
        n = min(items_per_source, len(source_data))
        samples.extend(random.sample(source_data, k=n))

    logger.info(f"Selected {len(samples)} samples for diarization")
    return samples


def run_diarization(pipeline: Pipeline, audio_path: Path) -> dict[str, Any]:
    """Run pyannote diarization for one utterance."""
    try:
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        waveform = waveform.to(DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=False):
            diarization = pipeline({"waveform": waveform, "sample_rate": sr})

        speakers: set[str] = set()
        segments_per_speaker: dict[str, int] = defaultdict(int)
        speaker_time_sec: dict[str, float] = defaultdict(float)
        segments: list[dict[str, Any]] = []
        intervals: list[tuple[float, float, str]] = []
        for idx, (turn, track, speaker) in enumerate(diarization.itertracks(yield_label=True)):
            speaking_time = turn.end - turn.start
            if speaking_time < 0.4:
                continue

            speaker_name = str(speaker)
            speakers.add(speaker_name)
            segments_per_speaker[speaker_name] += 1
            speaker_time_sec[speaker_name] += float(speaking_time)
            intervals.append((float(turn.start), float(turn.end), speaker_name))
            segments.append(
                {
                    "segment_id": f"seg_{idx:04d}",
                    "track_id": str(track),
                    "speaker_id": speaker_name,
                    "start_sec": float(turn.start),
                    "end_sec": float(turn.end),
                    "duration_sec": float(speaking_time),
                }
            )

        total_speech_sec = float(sum(speaker_time_sec.values()))
        speaker_fraction = {
            spk: (dur / total_speech_sec if total_speech_sec > 0 else 0.0)
            for spk, dur in speaker_time_sec.items()
        }

        dominant_speaker_id = None
        dominant_speaker_fraction = 0.0
        if speaker_time_sec:
            dominant_speaker_id = max(speaker_time_sec, key=speaker_time_sec.get)
            dominant_speaker_fraction = speaker_fraction.get(dominant_speaker_id, 0.0)

        # Approximate overlap (time where 2+ speakers are active simultaneously).
        events: list[tuple[float, int]] = []
        for start, end, _ in intervals:
            events.append((start, 1))
            events.append((end, -1))
        events.sort(key=lambda x: (x[0], -x[1]))

        overlap_speech_sec = 0.0
        active = 0
        for i in range(len(events) - 1):
            t, delta = events[i]
            active += delta
            next_t = events[i + 1][0]
            if next_t > t and active >= 2:
                overlap_speech_sec += (next_t - t)

        overlap_ratio = (overlap_speech_sec / total_speech_sec) if total_speech_sec > 0 else 0.0

        for seg in segments:
            seg["is_short"] = seg["duration_sec"] < 0.8

        return {
            "success": True,
            "num_speakers": int(len(speakers)),
            "segments_per_speaker": dict(segments_per_speaker),
            "speaker_time_sec": dict(speaker_time_sec),
            "speaker_fraction": speaker_fraction,
            "dominant_speaker_id": dominant_speaker_id,
            "dominant_speaker_fraction": float(dominant_speaker_fraction),
            "total_speech_sec": total_speech_sec,
            "overlap_speech_sec": float(overlap_speech_sec),
            "overlap_ratio": float(overlap_ratio),
            "segments": segments,
            "audio_duration": float(waveform.shape[-1] / sr),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "success": False,
            "num_speakers": None,
            "segments_per_speaker": {},
            "speaker_time_sec": {},
            "speaker_fraction": {},
            "dominant_speaker_id": None,
            "dominant_speaker_fraction": 0.0,
            "total_speech_sec": 0.0,
            "overlap_speech_sec": 0.0,
            "overlap_ratio": 0.0,
            "segments": [],
            "audio_duration": None,
            "error": str(e),
        }


def _build_pipeline(model_id: str, hf_token: str | None) -> tuple[Pipeline, str]:
    """Load pyannote diarization pipeline and move it to selected device."""
    candidates = [model_id]

    last_exc: Exception | None = None
    for candidate in candidates:
        retries = 8
        for attempt in range(1, retries + 1):
            try:
                pipeline = Pipeline.from_pretrained(candidate, use_auth_token=hf_token)
                pipeline.to(torch.device(DEVICE))
                return pipeline, candidate
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _allowlist_global_from_error(exc):
                    logger.info(f"Retrying model load after safe-global patch ({attempt}/{retries})")
                    continue
                logger.warning(f"Failed to load model '{candidate}': {exc}")
                break

    raise RuntimeError(
        "Could not load any pyannote diarization model. "
        "Set HF_TOKEN and accept model conditions on Hugging Face for the requested model."
    ) from last_exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pyannote diarization audit on train JSONLs")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--model-id", type=str, default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument(
        "--subset",
        type=str,
        choices=["all", "val"],
        default="all",
        help="Dataset scope to audit: all training rows or only validation fold rows.",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default=None,
        help="Run output dir used to rebuild val split when --subset val.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="1-based fold to use for --subset val (default: 1).",
    )
    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN", None)

    logger.info("=" * 80)
    logger.info("PYANNOTE DIARIZATION AUDITOR")
    logger.info("=" * 80)
    logger.info(f"Device: {DEVICE}")
    if DEVICE.startswith("cuda"):
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Requested model: {args.model_id}")

    if args.subset == "val":
        if not args.eval_output_dir:
            raise ValueError("--eval-output-dir is required when --subset val")
        all_data = load_val_data_from_run(args.eval_output_dir, args.fold)
    else:
        all_data = load_jsonl_data(JSONL_PATHS)

    samples = stratified_sample(all_data, args.sample_size)

    logger.info("Loading pyannote pipeline...")
    try:
        pipeline, resolved_model = _build_pipeline(model_id=args.model_id, hf_token=hf_token)
        logger.info(f"Loaded pyannote model: {resolved_model}")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load pyannote pipeline.")
        logger.error("If model access is gated: set HF_TOKEN and accept model terms on Hugging Face.")
        logger.error(str(exc))
        return

    logger.info(f"Processing {len(samples)} samples...")
    results: list[dict[str, Any]] = []
    speaker_counts: dict[int, int] = defaultdict(int)
    multi_speaker_samples: list[dict[str, Any]] = []
    errors = 0

    for idx, item in enumerate(samples):
        if (idx + 1) % max(1, len(samples) // 10) == 0:
            logger.info(f"  Processed {idx + 1}/{len(samples)}")

        audio_path = resolve_audio_path(item)
        if not audio_path:
            logger.warning(f"Could not resolve audio path for {item.get('utterance_id')}")
            errors += 1
            continue

        analysis_result = run_diarization(pipeline, audio_path)

        if not analysis_result["success"]:
            errors += 1
            logger.debug(f"Error on {item.get('utterance_id')}: {analysis_result['error']}")
            continue

        num_speakers = int(analysis_result["num_speakers"])
        speaker_counts[num_speakers] += 1

        result_item = {
            "utterance_id": item.get("utterance_id"),
            "source": item.get("source"),
            "age_bucket": item.get("age_bucket"),
            "label": item.get("phonetic_text"),
            "phonetic_text": item.get("phonetic_text"),
            "audio_duration_sec": item.get("audio_duration_sec"),
            "num_speakers": num_speakers,
            "segments_per_speaker": analysis_result.get("segments_per_speaker", {}),
            "speaker_time_sec": analysis_result.get("speaker_time_sec", {}),
            "speaker_fraction": analysis_result.get("speaker_fraction", {}),
            "dominant_speaker_id": analysis_result.get("dominant_speaker_id"),
            "dominant_speaker_fraction": analysis_result.get("dominant_speaker_fraction", 0.0),
            "total_speech_sec": analysis_result.get("total_speech_sec", 0.0),
            "overlap_speech_sec": analysis_result.get("overlap_speech_sec", 0.0),
            "overlap_ratio": analysis_result.get("overlap_ratio", 0.0),
            "segments": analysis_result.get("segments", []),
        }
        results.append(result_item)

        if num_speakers > 1:
            multi_speaker_samples.append(result_item)

    logger.info("")
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)

    successful = len(results)
    total = len(samples)
    logger.info(f"Successfully processed: {successful}/{total} ({100 * successful / max(1, total):.1f}%)")
    if errors > 0:
        logger.info(f"Errors: {errors}")

    logger.info("")
    logger.info("Speaker Count Distribution:")
    for num_speakers in sorted(speaker_counts.keys()):
        count = speaker_counts[num_speakers]
        pct = 100 * count / max(1, successful)
        logger.info(f"  {num_speakers} speaker(s): {count:5d} ({pct:5.1f}%)")

    candidate_count = len(multi_speaker_samples)
    candidate_pct = 100 * candidate_count / max(1, successful)
    logger.info("")
    logger.info(f"Multi-speaker samples (>1 speaker): {candidate_count} ({candidate_pct:.1f}%)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"diarization_results_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "sample_size": len(samples),
                    "successful": successful,
                    "errors": errors,
                    "device": str(DEVICE),
                    "method": resolved_model,
                    "subset": args.subset,
                    "fold": args.fold if args.subset == "val" else None,
                    "eval_output_dir": args.eval_output_dir if args.subset == "val" else None,
                },
                "speaker_distribution": dict(speaker_counts),
                "multi_speaker_count": candidate_count,
                "multi_speaker_percentage": candidate_pct,
                "results": results,
            },
            f,
            indent=2,
        )

    logger.info(f"Full results saved to: {output_file}")

    if multi_speaker_samples:
        multi_speaker_file = OUTPUT_DIR / f"multi_speaker_samples_{timestamp}.json"
        with open(multi_speaker_file, "w", encoding="utf-8") as f:
            json.dump(multi_speaker_samples, f, indent=2)
        logger.info(f"Multi-speaker samples saved to: {multi_speaker_file}")

    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
