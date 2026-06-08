from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.score import normalize_ipa

DEFAULT_RUN_A = PROJECT_ROOT / "outputs/submit-day/05-56-13_chatgpted-speedrun-207"
# DEFAULT_RUN_B = PROJECT_ROOT / "outputs/submit-day/20-44-36_lying-just_one_more_run_bro-301"
DEFAULT_RUN_B = PROJECT_ROOT / "outputs/2026-03-27/02-47-10_best-Geert-297"

@dataclass(frozen=True)
class PackedLattice:
    label: str
    path: Path
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
            path=path,
            logits=logits,
            offsets=offsets,
            lengths=lengths,
            utterance_ids=utterance_ids,
            spans_by_id=spans_by_id,
        )

    def sequence(self, utterance_id: str) -> np.ndarray:
        offset, length = self.spans_by_id[utterance_id]
        return self.logits[offset : offset + length]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe two CTC validation runs for lattice alignment, prediction correlation, "
            "and complementary strengths/weaknesses."
        )
    )
    parser.add_argument("--run-a", default=str(DEFAULT_RUN_A), help="First run directory.")
    parser.add_argument("--run-b", default=str(DEFAULT_RUN_B), help="Second run directory.")
    parser.add_argument("--label-a", default="whisper", help="Short label for run A.")
    parser.add_argument("--label-b", default="hubert", help="Short label for run B.")
    parser.add_argument("--fold", type=int, default=1, help="1-based fold number. Defaults to 1.")
    parser.add_argument(
        "--min-char-count",
        type=int,
        default=500,
        help="Minimum reference-char support for strengths/weaknesses slices.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many rows to print for example/error tables.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory to save parquet outputs with detailed comparison tables.",
    )
    return parser.parse_args()


def resolve_run_dir(path_str: str) -> Path:
    raw_path = Path(path_str).expanduser()
    for candidate in (raw_path, PROJECT_ROOT / raw_path):
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / raw_path).resolve()


def build_artifact_paths(run_dir: Path, fold: int) -> tuple[Path, Path]:
    fold_dir = run_dir / f"fold_{fold}"
    logits_path = fold_dir / "val_logits_best.npz"
    oof_path = run_dir / "oof_predictions_best.parquet"
    if not logits_path.exists():
        raise FileNotFoundError(f"Missing logits cache: {logits_path}")
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing OOF predictions: {oof_path}")
    return logits_path, oof_path


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            substitution_cost = 0 if char_a == char_b else 1
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def char_error_rate(reference: str, hypothesis: str) -> float:
    return edit_distance(reference, hypothesis) / max(len(reference), 1)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32, copy=False)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def softmax_frame_cosine(a_logits: np.ndarray, b_logits: np.ndarray) -> np.ndarray:
    a_probs = softmax(a_logits)
    b_probs = softmax(b_logits)
    dots = (a_probs * b_probs).sum(axis=-1)
    a_norms = np.linalg.norm(a_probs, axis=-1)
    b_norms = np.linalg.norm(b_probs, axis=-1)
    return dots / np.clip(a_norms * b_norms, 1e-12, None)


def load_predictions(oof_path: Path, label: str) -> pl.DataFrame:
    df = pl.read_parquet(oof_path)
    required_columns = {"utterance_id", "prediction"}
    if not required_columns.issubset(df.columns):
        missing = sorted(required_columns - set(df.columns))
        raise ValueError(f"Missing columns in {oof_path}: {missing}")
    rename_map = {"prediction": f"{label}_prediction"}
    keep_columns = ["utterance_id", f"{label}_prediction"]
    if "ground_truth" in df.columns:
        rename_map["ground_truth"] = f"{label}_ground_truth"
        keep_columns.append(f"{label}_ground_truth")
    if "child_id" in df.columns:
        keep_columns.append("child_id")
    return df.rename(rename_map).select(keep_columns)


def build_alignment_df(
    lattice_a: PackedLattice,
    lattice_b: PackedLattice,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    utterance_ids_a = set(lattice_a.spans_by_id)
    utterance_ids_b = set(lattice_b.spans_by_id)
    common_utterance_ids = sorted(utterance_ids_a & utterance_ids_b)

    frame_lengths_a: list[int] = []
    frame_lengths_b: list[int] = []
    min_frame_lengths: list[int] = []
    frame_argmax_agreements: list[float] = []
    frame_prob_cosines: list[float] = []
    frame_deltas: list[int] = []

    total_frames = 0
    total_argmax_matches = 0
    total_prob_cosine = 0.0

    for utterance_id in common_utterance_ids:
        sequence_a = lattice_a.sequence(utterance_id)
        sequence_b = lattice_b.sequence(utterance_id)

        frames_a = int(sequence_a.shape[0])
        frames_b = int(sequence_b.shape[0])
        min_frames = min(frames_a, frames_b)
        clipped_a = sequence_a[:min_frames]
        clipped_b = sequence_b[:min_frames]

        argmax_matches = clipped_a.argmax(axis=-1) == clipped_b.argmax(axis=-1)
        argmax_agreement = float(argmax_matches.mean()) if min_frames else 1.0
        cosine_values = softmax_frame_cosine(clipped_a, clipped_b) if min_frames else np.ones(1, dtype=np.float32)
        cosine_mean = float(cosine_values.mean())

        frame_lengths_a.append(frames_a)
        frame_lengths_b.append(frames_b)
        min_frame_lengths.append(min_frames)
        frame_argmax_agreements.append(argmax_agreement)
        frame_prob_cosines.append(cosine_mean)
        frame_deltas.append(frames_a - frames_b)

        total_frames += min_frames
        total_argmax_matches += int(argmax_matches.sum())
        total_prob_cosine += float(cosine_values.sum())

    alignment_df = pl.DataFrame(
        {
            "utterance_id": common_utterance_ids,
            f"{lattice_a.label}_frames": frame_lengths_a,
            f"{lattice_b.label}_frames": frame_lengths_b,
            "min_aligned_frames": min_frame_lengths,
            "frame_delta": frame_deltas,
            "frame_argmax_agreement": frame_argmax_agreements,
            "frame_prob_cosine_mean": frame_prob_cosines,
        }
    )

    abs_deltas = np.abs(np.asarray(frame_deltas, dtype=np.int32))
    summary = {
        "shared_utterances": len(common_utterance_ids),
        f"{lattice_a.label}_only_utterances": len(utterance_ids_a - utterance_ids_b),
        f"{lattice_b.label}_only_utterances": len(utterance_ids_b - utterance_ids_a),
        "same_utterance_order": bool(np.array_equal(lattice_a.utterance_ids, lattice_b.utterance_ids)),
        "frame_length_pearson": pearson_corr(
            np.asarray(frame_lengths_a, dtype=np.float32),
            np.asarray(frame_lengths_b, dtype=np.float32),
        ),
        "frame_length_equal_rate": float((abs_deltas == 0).mean()),
        "frame_abs_delta_mean": float(abs_deltas.mean()),
        "frame_abs_delta_median": float(np.median(abs_deltas)),
        "frame_abs_delta_p95": float(np.quantile(abs_deltas, 0.95)),
        "frame_abs_delta_max": int(abs_deltas.max()),
        "frame_delta_mean_signed": float(np.mean(frame_deltas)),
        "total_aligned_frames": total_frames,
        "frame_argmax_agreement": total_argmax_matches / max(total_frames, 1),
        "frame_prob_cosine_mean": total_prob_cosine / max(total_frames, 1),
        "utterance_argmax_agreement_mean": float(np.mean(frame_argmax_agreements)),
        "utterance_argmax_agreement_median": float(np.median(frame_argmax_agreements)),
        "utterance_argmax_agreement_p10": float(np.quantile(frame_argmax_agreements, 0.10)),
        "utterance_argmax_agreement_p90": float(np.quantile(frame_argmax_agreements, 0.90)),
    }
    return alignment_df, summary


def build_comparison_df(
    predictions_a: pl.DataFrame,
    predictions_b: pl.DataFrame,
    alignment_df: pl.DataFrame,
    label_a: str,
    label_b: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    ground_truth_column_a = f"{label_a}_ground_truth"
    ground_truth_column_b = f"{label_b}_ground_truth"

    join_columns = ["utterance_id", f"{label_b}_prediction"]
    if ground_truth_column_b in predictions_b.columns:
        join_columns.append(ground_truth_column_b)

    joined = predictions_a.join(predictions_b.select(join_columns), on="utterance_id", how="inner")
    joined = joined.join(alignment_df, on="utterance_id", how="inner")

    if ground_truth_column_a not in joined.columns and ground_truth_column_b not in joined.columns:
        raise ValueError("Need at least one run to provide ground-truth strings in the OOF parquet.")

    if ground_truth_column_a in joined.columns and ground_truth_column_b in joined.columns:
        ground_truth_match = joined.select((pl.col(ground_truth_column_a) == pl.col(ground_truth_column_b)).all()).item()
        if not ground_truth_match:
            raise ValueError("Ground-truth strings do not match across the two OOF prediction files.")
        joined = joined.drop(ground_truth_column_b)

    ground_truth_column = ground_truth_column_a if ground_truth_column_a in joined.columns else ground_truth_column_b
    joined = joined.rename({ground_truth_column: "ground_truth"})

    if "child_id" not in joined.columns:
        joined = joined.with_columns(pl.lit(None).alias("child_id"))

    joined = joined.sort("utterance_id")

    utterance_ids = joined["utterance_id"].to_list()
    child_ids = joined["child_id"].to_list()
    references = joined["ground_truth"].to_list()
    predictions_a_list = joined[f"{label_a}_prediction"].to_list()
    predictions_b_list = joined[f"{label_b}_prediction"].to_list()
    frames_a = joined[f"{label_a}_frames"].to_list()
    frames_b = joined[f"{label_b}_frames"].to_list()
    min_aligned_frames = joined["min_aligned_frames"].to_list()
    frame_deltas = joined["frame_delta"].to_list()
    frame_argmax_agreement = joined["frame_argmax_agreement"].to_list()
    frame_prob_cosine_mean = joined["frame_prob_cosine_mean"].to_list()

    normalized_references: list[str] = []
    normalized_predictions_a: list[str] = []
    normalized_predictions_b: list[str] = []
    reference_lengths: list[int] = []
    cer_a: list[float] = []
    cer_b: list[float] = []
    pair_cer: list[float] = []
    predictions_equal: list[bool] = []
    exact_a: list[bool] = []
    exact_b: list[bool] = []
    winners: list[str] = []

    for reference, prediction_a, prediction_b in zip(references, predictions_a_list, predictions_b_list):
        normalized_reference = normalize_ipa(reference)
        normalized_prediction_a = normalize_ipa(prediction_a)
        normalized_prediction_b = normalize_ipa(prediction_b)

        error_a = char_error_rate(normalized_reference, normalized_prediction_a)
        error_b = char_error_rate(normalized_reference, normalized_prediction_b)
        pairwise_error = char_error_rate(normalized_prediction_a, normalized_prediction_b)

        normalized_references.append(normalized_reference)
        normalized_predictions_a.append(normalized_prediction_a)
        normalized_predictions_b.append(normalized_prediction_b)
        reference_lengths.append(len(normalized_reference))
        cer_a.append(error_a)
        cer_b.append(error_b)
        pair_cer.append(pairwise_error)
        predictions_equal.append(normalized_prediction_a == normalized_prediction_b)
        exact_a.append(error_a == 0.0)
        exact_b.append(error_b == 0.0)

        if error_a < error_b:
            winners.append(label_a)
        elif error_b < error_a:
            winners.append(label_b)
        else:
            winners.append("tie")

    comparison_df = pl.DataFrame(
        {
            "utterance_id": utterance_ids,
            "child_id": child_ids,
            "ground_truth": references,
            f"{label_a}_prediction": predictions_a_list,
            f"{label_b}_prediction": predictions_b_list,
            "normalized_ground_truth": normalized_references,
            f"normalized_{label_a}_prediction": normalized_predictions_a,
            f"normalized_{label_b}_prediction": normalized_predictions_b,
            "reference_length": reference_lengths,
            f"{label_a}_cer": cer_a,
            f"{label_b}_cer": cer_b,
            "prediction_pair_cer": pair_cer,
            "predictions_equal": predictions_equal,
            f"{label_a}_exact": exact_a,
            f"{label_b}_exact": exact_b,
            "winner": winners,
            f"{label_a}_frames": frames_a,
            f"{label_b}_frames": frames_b,
            "min_aligned_frames": min_aligned_frames,
            "frame_delta": frame_deltas,
            "frame_argmax_agreement": frame_argmax_agreement,
            "frame_prob_cosine_mean": frame_prob_cosine_mean,
            "cer_delta": (np.asarray(cer_a) - np.asarray(cer_b)).tolist(),
        }
    )

    cer_a_array = np.asarray(cer_a, dtype=np.float32)
    cer_b_array = np.asarray(cer_b, dtype=np.float32)
    summary = {
        "num_utterances": comparison_df.height,
        "prediction_exact_match_rate": float(np.mean(predictions_equal)),
        "prediction_pair_cer_mean": float(np.mean(pair_cer)),
        "prediction_pair_cer_median": float(np.median(pair_cer)),
        "prediction_pair_cer_p95": float(np.quantile(pair_cer, 0.95)),
        f"{label_a}_mean_utterance_cer": float(np.mean(cer_a)),
        f"{label_b}_mean_utterance_cer": float(np.mean(cer_b)),
        "per_utterance_cer_pearson": pearson_corr(cer_a_array, cer_b_array),
        f"{label_a}_wins": int(np.sum(cer_a_array < cer_b_array)),
        f"{label_b}_wins": int(np.sum(cer_b_array < cer_a_array)),
        "ties": int(np.sum(cer_a_array == cer_b_array)),
        "both_exact": int(np.sum(np.asarray(exact_a) & np.asarray(exact_b))),
        f"{label_a}_exact_only": int(np.sum(np.asarray(exact_a) & ~np.asarray(exact_b))),
        f"{label_b}_exact_only": int(np.sum(np.asarray(exact_b) & ~np.asarray(exact_a))),
        "neither_exact": int(np.sum(~np.asarray(exact_a) & ~np.asarray(exact_b))),
    }
    return comparison_df, summary


def quantile_bucket_summary(
    df: pl.DataFrame,
    bucket_column: str,
    label_a: str,
    label_b: str,
) -> pl.DataFrame:
    values = df[bucket_column].to_numpy()
    quartiles = np.quantile(values, [0.25, 0.50, 0.75])
    bounds = [-np.inf, *quartiles.tolist(), np.inf]
    rows: list[dict[str, Any]] = []

    cer_a = df[f"{label_a}_cer"].to_numpy()
    cer_b = df[f"{label_b}_cer"].to_numpy()

    for idx in range(4):
        lower = bounds[idx]
        upper = bounds[idx + 1]
        mask = (values > lower) & (values <= upper)
        rows.append(
            {
                "bucket": f"Q{idx + 1}",
                "range": f"({lower:.0f}, {upper:.0f}]",
                "n": int(mask.sum()),
                f"{label_a}_mean_cer": float(cer_a[mask].mean()),
                f"{label_b}_mean_cer": float(cer_b[mask].mean()),
                f"{label_a}_wins": int((cer_a[mask] < cer_b[mask]).sum()),
                f"{label_b}_wins": int((cer_b[mask] < cer_a[mask]).sum()),
            }
        )
    return pl.DataFrame(rows)


def reference_char_strengths(
    df: pl.DataFrame,
    label_a: str,
    label_b: str,
    min_char_count: int,
) -> pl.DataFrame:
    normalized_references = df["normalized_ground_truth"].to_list()
    cer_a = df[f"{label_a}_cer"].to_numpy()
    cer_b = df[f"{label_b}_cer"].to_numpy()

    candidate_chars = sorted(set("".join(normalized_references)) - {" "})
    rows: list[dict[str, Any]] = []
    for char in candidate_chars:
        mask = np.asarray([char in reference for reference in normalized_references], dtype=bool)
        support = int(mask.sum())
        if support < min_char_count:
            continue
        rows.append(
            {
                "char": char,
                "support": support,
                f"{label_a}_mean_cer": float(cer_a[mask].mean()),
                f"{label_b}_mean_cer": float(cer_b[mask].mean()),
                "cer_delta": float((cer_a[mask] - cer_b[mask]).mean()),
                f"{label_a}_wins": int((cer_a[mask] < cer_b[mask]).sum()),
                f"{label_b}_wins": int((cer_b[mask] < cer_a[mask]).sum()),
            }
        )
    return pl.DataFrame(rows).sort("cer_delta")


def extreme_examples(
    df: pl.DataFrame,
    label_a: str,
    label_b: str,
    top_k: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    best_for_a = df.sort("cer_delta").head(top_k)
    best_for_b = df.sort("cer_delta", descending=True).head(top_k)
    columns = [
        "utterance_id",
        "child_id",
        "ground_truth",
        f"{label_a}_prediction",
        f"{label_b}_prediction",
        f"{label_a}_cer",
        f"{label_b}_cer",
        "cer_delta",
        f"{label_a}_frames",
        f"{label_b}_frames",
        "frame_argmax_agreement",
        "frame_prob_cosine_mean",
    ]
    return best_for_a.select(columns), best_for_b.select(columns)


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def print_key_values(summary: dict[str, Any]) -> None:
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def maybe_save_tables(
    save_dir: str | None,
    comparison_df: pl.DataFrame,
    char_df: pl.DataFrame,
    best_for_a: pl.DataFrame,
    best_for_b: pl.DataFrame,
    label_a: str,
    label_b: str,
) -> None:
    if save_dir is None:
        return

    output_dir = resolve_run_dir(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{label_a}_vs_{label_b}"
    comparison_path = output_dir / f"{stem}_per_utterance.parquet"
    char_path = output_dir / f"{stem}_reference_char_slices.parquet"
    best_a_path = output_dir / f"{stem}_best_for_{label_a}.parquet"
    best_b_path = output_dir / f"{stem}_best_for_{label_b}.parquet"

    comparison_df.write_parquet(comparison_path)
    char_df.write_parquet(char_path)
    best_for_a.write_parquet(best_a_path)
    best_for_b.write_parquet(best_b_path)

    print_section("Saved Tables")
    print(comparison_path)
    print(char_path)
    print(best_a_path)
    print(best_b_path)


def main() -> None:
    args = parse_args()
    run_a = resolve_run_dir(args.run_a)
    run_b = resolve_run_dir(args.run_b)

    logits_a_path, oof_a_path = build_artifact_paths(run_a, args.fold)
    logits_b_path, oof_b_path = build_artifact_paths(run_b, args.fold)

    lattice_a = PackedLattice.load(args.label_a, logits_a_path)
    lattice_b = PackedLattice.load(args.label_b, logits_b_path)
    predictions_a = load_predictions(oof_a_path, args.label_a)
    predictions_b = load_predictions(oof_b_path, args.label_b)

    alignment_df, alignment_summary = build_alignment_df(lattice_a, lattice_b)
    comparison_df, prediction_summary = build_comparison_df(
        predictions_a=predictions_a,
        predictions_b=predictions_b,
        alignment_df=alignment_df,
        label_a=args.label_a,
        label_b=args.label_b,
    )
    reference_char_df = reference_char_strengths(
        comparison_df,
        label_a=args.label_a,
        label_b=args.label_b,
        min_char_count=args.min_char_count,
    )
    reference_len_quartiles = quantile_bucket_summary(
        comparison_df,
        bucket_column="reference_length",
        label_a=args.label_a,
        label_b=args.label_b,
    )
    frame_len_quartiles = quantile_bucket_summary(
        comparison_df,
        bucket_column=f"{args.label_a}_frames",
        label_a=args.label_a,
        label_b=args.label_b,
    )
    best_for_a, best_for_b = extreme_examples(
        comparison_df,
        label_a=args.label_a,
        label_b=args.label_b,
        top_k=args.top_k,
    )

    print_section("Runs")
    print(f"{args.label_a}: {run_a}")
    print(f"{args.label_b}: {run_b}")
    print(f"fold: {args.fold}")
    print(f"{args.label_a} logits: {logits_a_path}")
    print(f"{args.label_b} logits: {logits_b_path}")

    print_section("Frame Alignment")
    print_key_values(alignment_summary)

    print_section("Prediction Correlation")
    print_key_values(prediction_summary)

    print_section("Reference Length Quartiles")
    print(reference_len_quartiles)

    print_section("Frame Length Quartiles")
    print(frame_len_quartiles)

    print_section("Reference Characters Best For Run A")
    print(reference_char_df.head(args.top_k))

    print_section("Reference Characters Best For Run B")
    print(reference_char_df.tail(args.top_k))

    print_section(f"Examples Where {args.label_a} Beats {args.label_b}")
    print(best_for_a)

    print_section(f"Examples Where {args.label_b} Beats {args.label_a}")
    print(best_for_b)

    maybe_save_tables(
        save_dir=args.save_dir,
        comparison_df=comparison_df,
        char_df=reference_char_df,
        best_for_a=best_for_a,
        best_for_b=best_for_b,
        label_a=args.label_a,
        label_b=args.label_b,
    )


if __name__ == "__main__":
    main()
