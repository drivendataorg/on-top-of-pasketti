from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import polars as pl
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exploration.ensemble_ctc_posteriors import (
    DEFAULT_DECODER_CONFIG,
    DEFAULT_RUN_A,
    DEFAULT_RUN_B,
    PackedLattice,
    align_and_fuse_sequence,
    build_artifact_paths,
    build_metadata_table,
    decode_in_batches,
    load_decoder,
    load_prediction_table,
    print_section,
    resolve_path,
)
from src.utils.score import score_ipa_cer

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "exploration/reports/whisper_hubert_smooth_weight_ensemble.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a length-routed posterior ensemble where the Whisper/HuBERT "
            "weights are chosen by a smooth tanh curve over utterance frame length."
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
        help="Decoder config used to decode the fused emissions.",
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
        "--crossover",
        type=float,
        default=95.0,
        help="Frame length where the weighting crosses through 50/50.",
    )
    parser.add_argument(
        "--max-skew",
        type=float,
        default=0.10,
        help="Maximum shift away from 50/50. Must be < 0.5.",
    )
    parser.add_argument(
        "--steepness",
        type=float,
        default=20.0,
        help="Controls how quickly the tanh transition changes with length.",
    )
    parser.add_argument(
        "--decode-batch-size",
        type=int,
        default=256,
        help="How many utterances to decode per batch.",
    )
    parser.add_argument(
        "--output-parquet",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to save predictions and per-utterance routing metadata.",
    )
    return parser.parse_args()


def get_ensemble_weights_smooth(
    frame_length: float,
    crossover: float = 95.0,
    max_skew: float = 0.10,
    steepness: float = 20.0,
) -> tuple[float, float]:
    """
    Returns (whisper_weight, hubert_weight) using a smooth tanh transition.
    """
    shift = max_skew * math.tanh((frame_length - crossover) / steepness)
    hubert_weight = 0.5 + shift
    whisper_weight = 1.0 - hubert_weight
    return whisper_weight, hubert_weight


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.max_skew < 0.5):
        raise ValueError("--max-skew must be in [0.0, 0.5).")
    if args.steepness <= 0:
        raise ValueError("--steepness must be > 0.")

    run_a = resolve_path(args.run_a)
    run_b = resolve_path(args.run_b)
    decoder_config_path = resolve_path(args.decoder_config)
    output_path = resolve_path(args.output_parquet)

    logits_a_path, oof_a_path, config_a_path = build_artifact_paths(run_a, args.fold)
    logits_b_path, oof_b_path, _ = build_artifact_paths(run_b, args.fold)

    lattice_a = PackedLattice.load(args.label_a, logits_a_path)
    lattice_b = PackedLattice.load(args.label_b, logits_b_path)
    if lattice_a.logits.shape[1] != lattice_b.logits.shape[1]:
        raise ValueError(
            f"Vocab mismatch: {args.label_a} has {lattice_a.logits.shape[1]} columns, "
            f"{args.label_b} has {lattice_b.logits.shape[1]}."
        )

    _, tokenizer, decoder, decoder_cfg = load_decoder(config_a_path, decoder_config_path)
    predictions_a = load_prediction_table(oof_a_path, args.label_a)
    predictions_b = load_prediction_table(oof_b_path, args.label_b)
    metadata_df = build_metadata_table(predictions_a, predictions_b, args.label_a, args.label_b)

    utterance_ids = metadata_df["utterance_id"].to_list()
    fused_sequences: list[torch.Tensor] = []
    alignment_rows: list[dict[str, object]] = []
    trimmed_counter: Counter[str] = Counter()
    whisper_weights: list[float] = []
    hubert_weights: list[float] = []

    for utterance_id in tqdm(utterance_ids, desc="Fusing smooth-routed posteriors"):
        sequence_a = lattice_a.sequence(utterance_id)
        sequence_b = lattice_b.sequence(utterance_id)
        aligned_frame_length = min(int(sequence_a.shape[0]), int(sequence_b.shape[0]))
        whisper_weight, hubert_weight = get_ensemble_weights_smooth(
            frame_length=float(aligned_frame_length),
            crossover=args.crossover,
            max_skew=args.max_skew,
            steepness=args.steepness,
        )

        fused_sequence, alignment_info = align_and_fuse_sequence(
            sequence_a=sequence_a,
            sequence_b=sequence_b,
            label_a=args.label_a,
            label_b=args.label_b,
            weight_a=whisper_weight,
            weight_b=hubert_weight,
            temperature_a=args.temperature_a,
            temperature_b=args.temperature_b,
            blank_id=tokenizer.blank_token_id,
            blank_scale_a=1.0,
            blank_scale_b=1.0,
            fusion="arithmetic",
        )

        fused_sequences.append(fused_sequence)
        whisper_weights.append(whisper_weight)
        hubert_weights.append(hubert_weight)
        alignment_rows.append(
            {
                "utterance_id": utterance_id,
                **alignment_info,
                f"{args.label_a}_weight": whisper_weight,
                f"{args.label_b}_weight": hubert_weight,
                "routing_frame_length": aligned_frame_length,
            }
        )
        trimmed_counter[alignment_info["trimmed_from"]] += 1

    predictions_by_id = decode_in_batches(
        utterance_ids=utterance_ids,
        fused_sequences=fused_sequences,
        decoder=decoder,
        decode_batch_size=args.decode_batch_size,
    )

    alignment_df = pl.DataFrame(alignment_rows)
    result_df = metadata_df.join(alignment_df, on="utterance_id", how="left").with_columns(
        pl.Series("ensemble_prediction", [predictions_by_id[utterance_id] for utterance_id in utterance_ids]),
        pl.lit("arithmetic").alias("fusion"),
        pl.lit(args.temperature_a).alias(f"{args.label_a}_temperature"),
        pl.lit(args.temperature_b).alias(f"{args.label_b}_temperature"),
        pl.lit(args.crossover).alias("crossover"),
        pl.lit(args.max_skew).alias("max_skew"),
        pl.lit(args.steepness).alias("steepness"),
        pl.lit(str(decoder_cfg._target_)).alias("decoder_target"),
    )

    references = result_df["ground_truth"].to_list() if result_df["ground_truth"].dtype == pl.String else []
    if references:
        model_a_per = score_ipa_cer(references, result_df[f"{args.label_a}_prediction"].to_list())
        model_b_per = score_ipa_cer(references, result_df[f"{args.label_b}_prediction"].to_list())
        ensemble_per = score_ipa_cer(references, result_df["ensemble_prediction"].to_list())
    else:
        model_a_per = None
        model_b_per = None
        ensemble_per = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(output_path)

    print_section("Runs")
    print(f"{args.label_a}: {run_a}")
    print(f"{args.label_b}: {run_b}")
    print(f"fold: {args.fold}")
    print(f"decoder_config: {decoder_config_path}")

    print_section("Smooth Routing")
    print(f"crossover: {args.crossover:.2f}")
    print(f"max_skew: {args.max_skew:.4f}")
    print(f"steepness: {args.steepness:.2f}")
    print(f"mean_{args.label_a}_weight: {sum(whisper_weights) / len(whisper_weights):.5f}")
    print(f"mean_{args.label_b}_weight: {sum(hubert_weights) / len(hubert_weights):.5f}")
    print(f"min_{args.label_a}_weight: {min(whisper_weights):.5f}")
    print(f"max_{args.label_a}_weight: {max(whisper_weights):.5f}")
    print(f"min_{args.label_b}_weight: {min(hubert_weights):.5f}")
    print(f"max_{args.label_b}_weight: {max(hubert_weights):.5f}")

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
    print(output_path)


if __name__ == "__main__":
    main()
