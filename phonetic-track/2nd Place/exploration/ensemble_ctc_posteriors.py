from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.score import score_ipa_cer

DEFAULT_RUN_A = PROJECT_ROOT / "outputs/2026-03-23/18-40-44_japanese-where_merch-259"
DEFAULT_RUN_B = PROJECT_ROOT / "outputs/submit-day/20-44-36_lying-just_one_more_run_bro-301"
DEFAULT_DECODER_CONFIG = PROJECT_ROOT / "configs/decoder/beam_search_simple.yaml"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "exploration/reports/whisper_hubert_posterior_fusion.parquet"


@dataclass(frozen=True)
class PackedLattice:
    label: str
    logits: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    utterance_ids: np.ndarray
    spans_by_id: dict[str, tuple[int, int]]

    @classmethod
    def load(cls, label: str, path: Path) -> PackedLattice:
        with np.load(path, allow_pickle=False) as payload:
            logits = payload["logits"]
            offsets = payload["offsets"]
            lengths = payload["lengths"]
            utterance_ids = payload["utterance_ids"]

        spans_by_id = {
            str(utterance_id): (int(offset), int(length))
            for utterance_id, offset, length in zip(utterance_ids.tolist(), offsets.tolist(), lengths.tolist())
        }
        return cls(
            label=label,
            logits=logits,
            offsets=offsets,
            lengths=lengths,
            utterance_ids=utterance_ids,
            spans_by_id=spans_by_id,
        )

    def sequence(self, utterance_id: str) -> np.ndarray:
        offset, length = self.spans_by_id[utterance_id]
        return self.logits[offset : offset + length]


@dataclass(frozen=True)
class EnsembleRunConfig:
    run_a: Path
    run_b: Path
    label_a: str
    label_b: str
    fold: int
    decoder_config_path: Path
    fusion: str
    weight_a: float
    weight_b: float
    temperature_a: float
    temperature_b: float
    blank_scale_a: float
    blank_scale_b: float
    decode_batch_size: int
    output_path: Path | None


@dataclass(frozen=True)
class EnsembleRunResult:
    result_df: pl.DataFrame
    model_a_per: float | None
    model_b_per: float | None
    ensemble_per: float | None
    trimmed_counter: Counter[str]
    decoder_target: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load two saved CTC validation lattices, align them by utterance_id, "
            "fuse posteriors, and decode an ensemble prediction."
        )
    )
    parser.add_argument("--run-a", default=str(DEFAULT_RUN_A), help="First run directory.")
    parser.add_argument("--run-b", default=str(DEFAULT_RUN_B), help="Second run directory.")
    parser.add_argument("--label-a", default="whisper", help="Short label for run A.")
    parser.add_argument("--label-b", default="hubert", help="Short label for run B.")
    parser.add_argument("--fold", type=int, default=1, help="1-based fold number. Defaults to 1.")
    parser.add_argument(
        "--decoder-config",
        default=str(DEFAULT_DECODER_CONFIG),
        help=(
            "Decoder config to instantiate with the tokenizer from run A. "
            "Defaults to configs/decoder/beam_search_simple.yaml."
        ),
    )
    parser.add_argument(
        "--fusion",
        choices=("arithmetic", "log_linear"),
        default="arithmetic",
        help="Posterior fusion rule. Defaults to arithmetic mean in probability space.",
    )
    parser.add_argument(
        "--weight-a",
        type=float,
        default=0.5,
        help="Weight for run A. If --weight-b is omitted, run B gets the remaining weight.",
    )
    parser.add_argument(
        "--weight-b",
        type=float,
        default=None,
        help="Optional weight for run B. If omitted, uses 1 - weight_a.",
    )
    parser.add_argument(
        "--temperature-a",
        type=float,
        default=1.0,
        help="Temperature for run A before posterior fusion.",
    )
    parser.add_argument(
        "--temperature-b",
        type=float,
        default=1.0,
        help="Temperature for run B before posterior fusion.",
    )
    parser.add_argument(
        "--blank-scale-a",
        type=float,
        default=1.0,
        help="Multiplier applied to run A's blank posterior before renormalization.",
    )
    parser.add_argument(
        "--blank-scale-b",
        type=float,
        default=1.0,
        help="Multiplier applied to run B's blank posterior before renormalization.",
    )
    parser.add_argument(
        "--decode-batch-size",
        type=int,
        default=256,
        help="How many fused utterances to decode per batch.",
    )
    parser.add_argument(
        "--output-parquet",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to save ensemble predictions and metadata.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    raw_path = Path(path_str).expanduser()
    for candidate in (raw_path, PROJECT_ROOT / raw_path):
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / raw_path).resolve()


def build_artifact_paths(run_dir: Path, fold: int) -> tuple[Path, Path, Path]:
    fold_dir = run_dir / f"fold_{fold}"
    logits_path = fold_dir / "val_logits_best.npz"
    oof_path = run_dir / "oof_predictions_best.parquet"
    config_path = run_dir / ".hydra" / "config.yaml"

    if not logits_path.exists():
        raise FileNotFoundError(f"Missing logits cache: {logits_path}")
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing OOF predictions: {oof_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    return logits_path, oof_path, config_path


def normalize_weights(weight_a: float, weight_b: float | None) -> tuple[float, float]:
    weight_b = (1.0 - weight_a) if weight_b is None else weight_b
    total = weight_a + weight_b
    if total <= 0:
        raise ValueError("Fusion weights must sum to a positive value.")
    return weight_a / total, weight_b / total


def stable_log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32, copy=False)
    logits = logits - logits.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    return logits - logsumexp


def load_decoder(run_config_path: Path, decoder_config_path: Path) -> tuple[Any, Any, Any, DictConfig]:
    run_cfg = OmegaConf.load(run_config_path)
    tokenizer = instantiate(run_cfg.tokenizer)
    decoder_cfg = OmegaConf.load(decoder_config_path)
    decoder = instantiate(decoder_cfg, tokenizer=tokenizer)
    return run_cfg, tokenizer, decoder, decoder_cfg


def renormalize_log_probs(log_values: np.ndarray) -> np.ndarray:
    log_values = log_values.astype(np.float32, copy=False)
    log_values = log_values - log_values.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(log_values).sum(axis=-1, keepdims=True))
    return log_values - logsumexp


def apply_blank_scale(log_probs: np.ndarray, blank_id: int, blank_scale: float) -> np.ndarray:
    if blank_scale <= 0:
        raise ValueError("Blank scale must be > 0.")
    if math.isclose(blank_scale, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        return log_probs

    adjusted = np.array(log_probs, copy=True)
    adjusted[..., blank_id] += math.log(blank_scale)
    return renormalize_log_probs(adjusted)


def load_prediction_table(oof_path: Path, label: str) -> pl.DataFrame:
    df = pl.read_parquet(oof_path)
    rename_map = {"prediction": f"{label}_prediction"}
    keep_columns = ["utterance_id", f"{label}_prediction"]
    if "ground_truth" in df.columns:
        rename_map["ground_truth"] = f"{label}_ground_truth"
        keep_columns.append(f"{label}_ground_truth")
    if "child_id" in df.columns:
        keep_columns.append("child_id")
    return df.rename(rename_map).select(keep_columns)


def build_metadata_table(
    predictions_a: pl.DataFrame,
    predictions_b: pl.DataFrame,
    label_a: str,
    label_b: str,
) -> pl.DataFrame:
    join_columns = ["utterance_id", f"{label_b}_prediction"]
    ground_truth_b = f"{label_b}_ground_truth"
    if ground_truth_b in predictions_b.columns:
        join_columns.append(ground_truth_b)

    joined = predictions_a.join(predictions_b.select(join_columns), on="utterance_id", how="inner")

    ground_truth_a = f"{label_a}_ground_truth"
    if ground_truth_a in joined.columns and ground_truth_b in joined.columns:
        ground_truth_match = joined.select((pl.col(ground_truth_a) == pl.col(ground_truth_b)).all()).item()
        if not ground_truth_match:
            raise ValueError("Ground-truth strings do not match across the two OOF prediction files.")
        joined = joined.drop(ground_truth_b)

    if ground_truth_a in joined.columns:
        joined = joined.rename({ground_truth_a: "ground_truth"})
    elif ground_truth_b in joined.columns:
        joined = joined.rename({ground_truth_b: "ground_truth"})
    else:
        joined = joined.with_columns(pl.lit(None).alias("ground_truth"))

    if "child_id" not in joined.columns:
        joined = joined.with_columns(pl.lit(None).alias("child_id"))

    return joined.sort("utterance_id")


def align_and_fuse_sequence(
    sequence_a: np.ndarray,
    sequence_b: np.ndarray,
    label_a: str,
    label_b: str,
    weight_a: float,
    weight_b: float,
    temperature_a: float,
    temperature_b: float,
    blank_id: int,
    blank_scale_a: float,
    blank_scale_b: float,
    fusion: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    frames_a = int(sequence_a.shape[0])
    frames_b = int(sequence_b.shape[0])
    delta = frames_a - frames_b

    if abs(delta) > 1:
        raise ValueError(
            f"Expected frame delta <= 1 after utterance alignment, got {delta} "
            f"for {label_a}={frames_a}, {label_b}={frames_b}."
        )

    trimmed_from = "none"
    if delta == 1:
        sequence_a = sequence_a[:-1]
        trimmed_from = label_a
    elif delta == -1:
        sequence_b = sequence_b[:-1]
        trimmed_from = label_b

    log_probs_a = stable_log_softmax(sequence_a / temperature_a)
    log_probs_b = stable_log_softmax(sequence_b / temperature_b)
    log_probs_a = apply_blank_scale(log_probs_a, blank_id=blank_id, blank_scale=blank_scale_a)
    log_probs_b = apply_blank_scale(log_probs_b, blank_id=blank_id, blank_scale=blank_scale_b)

    if fusion == "arithmetic":
        posterior_a = np.exp(log_probs_a)
        posterior_b = np.exp(log_probs_b)
        fused_posteriors = (weight_a * posterior_a) + (weight_b * posterior_b)
        fused_emissions = np.log(np.clip(fused_posteriors, 1e-12, None))
    else:
        fused_emissions = (weight_a * log_probs_a) + (weight_b * log_probs_b)

    fused_emissions = np.ascontiguousarray(fused_emissions.astype(np.float32, copy=False))
    return torch.from_numpy(fused_emissions), {
        f"{label_a}_frames": frames_a,
        f"{label_b}_frames": frames_b,
        "ensemble_frames": int(fused_emissions.shape[0]),
        "frame_delta": delta,
        "trimmed_from": trimmed_from,
    }


def decode_in_batches(
    utterance_ids: list[str],
    fused_sequences: list[torch.Tensor],
    decoder: Any,
    decode_batch_size: int,
) -> dict[str, str]:
    predictions_by_id: dict[str, str] = {}
    for start_idx in tqdm(range(0, len(utterance_ids), decode_batch_size), desc="Decoding ensemble"):
        end_idx = min(len(utterance_ids), start_idx + decode_batch_size)
        batch_ids = utterance_ids[start_idx:end_idx]
        batch_sequences = fused_sequences[start_idx:end_idx]
        batch_predictions = decoder(batch_sequences)
        for utterance_id, prediction in zip(batch_ids, batch_predictions):
            predictions_by_id[utterance_id] = prediction
    return predictions_by_id


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    args = parse_args()
    run_config = EnsembleRunConfig(
        run_a=resolve_path(args.run_a),
        run_b=resolve_path(args.run_b),
        label_a=args.label_a,
        label_b=args.label_b,
        fold=args.fold,
        decoder_config_path=resolve_path(args.decoder_config),
        fusion=args.fusion,
        weight_a=args.weight_a,
        weight_b=(1.0 - args.weight_a) if args.weight_b is None else args.weight_b,
        temperature_a=args.temperature_a,
        temperature_b=args.temperature_b,
        blank_scale_a=args.blank_scale_a,
        blank_scale_b=args.blank_scale_b,
        decode_batch_size=args.decode_batch_size,
        output_path=resolve_path(args.output_parquet),
    )
    run_result = run_ensemble(run_config)
    result_df = run_result.result_df
    model_a_per = run_result.model_a_per
    model_b_per = run_result.model_b_per
    ensemble_per = run_result.ensemble_per
    trimmed_counter = run_result.trimmed_counter

    print_section("Runs")
    print(f"{args.label_a}: {run_config.run_a}")
    print(f"{args.label_b}: {run_config.run_b}")
    print(f"fold: {run_config.fold}")
    print(f"decoder_config: {run_config.decoder_config_path}")

    print_section("Fusion")
    print(f"fusion: {run_config.fusion}")
    print(f"{args.label_a}_weight: {result_df[f'{args.label_a}_weight'][0]:.4f}")
    print(f"{args.label_b}_weight: {result_df[f'{args.label_b}_weight'][0]:.4f}")
    print(f"{args.label_a}_temperature: {run_config.temperature_a:.4f}")
    print(f"{args.label_b}_temperature: {run_config.temperature_b:.4f}")
    print(f"{args.label_a}_blank_scale: {run_config.blank_scale_a:.4f}")
    print(f"{args.label_b}_blank_scale: {run_config.blank_scale_b:.4f}")

    print_section("Alignment")
    print(f"utterances: {result_df.height}")
    print(f"trimmed_from_{args.label_a}: {trimmed_counter.get(args.label_a, 0)}")
    print(f"trimmed_from_{args.label_b}: {trimmed_counter.get(args.label_b, 0)}")
    print(f"trimmed_from_none: {trimmed_counter.get('none', 0)}")
    print(f"max_abs_frame_delta_before_trim: {int(result_df['frame_delta'].abs().max())}")

    if ensemble_per is not None:
        print_section("PER")
        print(f"{args.label_a}: {model_a_per:.5f}")
        print(f"{args.label_b}: {model_b_per:.5f}")
        print(f"ensemble: {ensemble_per:.5f}")
        print(f"delta_vs_{args.label_a}: {ensemble_per - model_a_per:+.5f}")
        print(f"delta_vs_{args.label_b}: {ensemble_per - model_b_per:+.5f}")

    print_section("Saved Predictions")
    print(run_config.output_path)


def run_ensemble(run_config: EnsembleRunConfig) -> EnsembleRunResult:
    weight_a, weight_b = normalize_weights(run_config.weight_a, run_config.weight_b)

    logits_a_path, oof_a_path, config_a_path = build_artifact_paths(run_config.run_a, run_config.fold)
    logits_b_path, oof_b_path, _ = build_artifact_paths(run_config.run_b, run_config.fold)

    lattice_a = PackedLattice.load(run_config.label_a, logits_a_path)
    lattice_b = PackedLattice.load(run_config.label_b, logits_b_path)

    if lattice_a.logits.shape[1] != lattice_b.logits.shape[1]:
        raise ValueError(
            f"Vocab mismatch: {run_config.label_a} has {lattice_a.logits.shape[1]} columns, "
            f"{run_config.label_b} has {lattice_b.logits.shape[1]}."
        )

    _, tokenizer, decoder, decoder_cfg = load_decoder(config_a_path, run_config.decoder_config_path)
    predictions_a = load_prediction_table(oof_a_path, run_config.label_a)
    predictions_b = load_prediction_table(oof_b_path, run_config.label_b)
    metadata_df = build_metadata_table(predictions_a, predictions_b, run_config.label_a, run_config.label_b)

    utterance_ids = metadata_df["utterance_id"].to_list()
    fused_sequences: list[torch.Tensor] = []
    alignment_rows: list[dict[str, Any]] = []
    trimmed_counter: Counter[str] = Counter()

    for utterance_id in tqdm(utterance_ids, desc="Fusing posteriors"):
        fused_sequence, alignment_info = align_and_fuse_sequence(
            sequence_a=lattice_a.sequence(utterance_id),
            sequence_b=lattice_b.sequence(utterance_id),
            label_a=run_config.label_a,
            label_b=run_config.label_b,
            weight_a=weight_a,
            weight_b=weight_b,
            temperature_a=run_config.temperature_a,
            temperature_b=run_config.temperature_b,
            blank_id=tokenizer.blank_token_id,
            blank_scale_a=run_config.blank_scale_a,
            blank_scale_b=run_config.blank_scale_b,
            fusion=run_config.fusion,
        )
        fused_sequences.append(fused_sequence)
        alignment_rows.append({"utterance_id": utterance_id, **alignment_info})
        trimmed_counter[alignment_info["trimmed_from"]] += 1

    predictions_by_id = decode_in_batches(
        utterance_ids=utterance_ids,
        fused_sequences=fused_sequences,
        decoder=decoder,
        decode_batch_size=run_config.decode_batch_size,
    )

    alignment_df = pl.DataFrame(alignment_rows)
    ensemble_predictions = [predictions_by_id[utterance_id] for utterance_id in utterance_ids]
    result_df = metadata_df.join(alignment_df, on="utterance_id", how="left").with_columns(
        pl.Series("ensemble_prediction", ensemble_predictions),
        pl.lit(run_config.fusion).alias("fusion"),
        pl.lit(weight_a).alias(f"{run_config.label_a}_weight"),
        pl.lit(weight_b).alias(f"{run_config.label_b}_weight"),
        pl.lit(run_config.temperature_a).alias(f"{run_config.label_a}_temperature"),
        pl.lit(run_config.temperature_b).alias(f"{run_config.label_b}_temperature"),
        pl.lit(run_config.blank_scale_a).alias(f"{run_config.label_a}_blank_scale"),
        pl.lit(run_config.blank_scale_b).alias(f"{run_config.label_b}_blank_scale"),
        pl.lit(str(decoder_cfg._target_)).alias("decoder_target"),
    )

    references = result_df["ground_truth"].to_list() if result_df["ground_truth"].dtype == pl.String else []
    if references:
        model_a_per = score_ipa_cer(references, result_df[f"{run_config.label_a}_prediction"].to_list())
        model_b_per = score_ipa_cer(references, result_df[f"{run_config.label_b}_prediction"].to_list())
        ensemble_per = score_ipa_cer(references, result_df["ensemble_prediction"].to_list())
    else:
        model_a_per = None
        model_b_per = None
        ensemble_per = None

    if run_config.output_path is not None:
        run_config.output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_parquet(run_config.output_path)

    return EnsembleRunResult(
        result_df=result_df,
        model_a_per=model_a_per,
        model_b_per=model_b_per,
        ensemble_per=ensemble_per,
        trimmed_counter=trimmed_counter,
        decoder_target=str(decoder_cfg._target_),
    )


if __name__ == "__main__":
    main()
