#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.score import normalize_ipa, score_ipa_cer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate simple utterance-level ensemble rules on a saved llm-fusion parquet. "
            "This is useful for checking whether rule-based selectors beat majority vote "
            "before spending more time on LLM prompting."
        )
    )
    parser.add_argument("parquet_path", help="Path to a saved llm-fusion parquet file.")
    return parser.parse_args()


def normalize_prediction(text: str | None) -> str:
    return normalize_ipa((text or "").strip())


def weighted_vote(
    row: dict[str, object],
    prediction_columns: list[str],
    weights: dict[str, float],
) -> str:
    scores: dict[str, float] = {}
    first_raw: dict[str, str] = {}
    first_idx: dict[str, int] = {}

    for idx, column in enumerate(prediction_columns):
        raw = str(row[column] or "").strip()
        key = normalize_prediction(raw)
        if not key:
            continue
        scores[key] = scores.get(key, 0.0) + weights[column]
        if key not in first_raw:
            first_raw[key] = raw
            first_idx[key] = idx

    if not scores:
        return str(row[prediction_columns[0]] or "").strip()

    best_key = max(scores, key=lambda key: (scores[key], -first_idx[key]))
    return first_raw[best_key]


def detect_prediction_columns(df: pl.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column.endswith("_prediction")
        and column not in {"majority_vote_prediction", "llm_fusion_prediction"}
    ]


def per_for_column(df: pl.DataFrame, column: str) -> float:
    return score_ipa_cer(df["ground_truth"].to_list(), df[column].to_list())


def summarize_two_way_disagreements(
    df: pl.DataFrame,
    prediction_columns: list[str],
) -> list[tuple[str, int, float, float]]:
    summary: dict[tuple[tuple[int, ...], int], dict[str, int]] = {}

    for row in df.iter_rows(named=True):
        normalized = [normalize_prediction(str(row[column] or "")) for column in prediction_columns]
        unique_nonempty = {text for text in normalized if text}
        if len(unique_nonempty) != 2:
            continue

        counts = Counter(normalized)
        majority_norm = max(counts, key=counts.get)
        agreeing = tuple(idx + 1 for idx, text in enumerate(normalized) if text == majority_norm)
        lone_model = next(idx + 1 for idx, text in enumerate(normalized) if text != majority_norm)
        key = (agreeing, lone_model)
        stats = summary.setdefault(key, {"n": 0, "majority_correct": 0, "lone_correct": 0})
        ground_truth = normalize_prediction(str(row["ground_truth"] or ""))

        stats["n"] += 1
        stats["majority_correct"] += int(majority_norm == ground_truth)
        stats["lone_correct"] += int(normalized[lone_model - 1] == ground_truth)

    rows = []
    for (agreeing, lone_model), stats in sorted(summary.items(), key=lambda item: item[1]["n"], reverse=True):
        n = stats["n"]
        rows.append(
            (
                f"{agreeing} vs {lone_model}",
                n,
                stats["majority_correct"] / n,
                stats["lone_correct"] / n,
            )
        )
    return rows


def main() -> None:
    args = parse_args()
    df = pl.read_parquet(args.parquet_path)
    prediction_columns = detect_prediction_columns(df)
    if "ground_truth" not in df.columns:
        raise ValueError("The parquet must contain a ground_truth column to evaluate PER.")
    if not prediction_columns:
        raise ValueError("Could not find any base model prediction columns.")

    model_pers = {column: per_for_column(df, column) for column in prediction_columns}
    best_model_column = min(model_pers, key=model_pers.get)
    inverse_per_weights = {column: 1.0 / per_value for column, per_value in model_pers.items()}
    max_per = max(model_pers.values())
    margin_weights = {column: max_per - per_value for column, per_value in model_pers.items()}

    strategies: dict[str, list[str]] = {
        "majority_vote": df["majority_vote_prediction"].to_list(),
    }
    if "llm_fusion_prediction" in df.columns:
        strategies["llm_fusion"] = df["llm_fusion_prediction"].to_list()
    strategies["best_single_model"] = df[best_model_column].to_list()
    strategies["weighted_inverse_per"] = [
        weighted_vote(row, prediction_columns, inverse_per_weights)
        for row in df.iter_rows(named=True)
    ]
    strategies["weighted_margin"] = [
        weighted_vote(row, prediction_columns, margin_weights)
        for row in df.iter_rows(named=True)
    ]
    strategies["majority_for_2way_best_model_for_3way"] = [
        str(row["majority_vote_prediction"] or "")
        if int(row["num_unique_hypotheses"]) < 3
        else str(row[best_model_column] or "")
        for row in df.iter_rows(named=True)
    ]
    if "llm_fusion_prediction" in df.columns:
        strategies["majority_for_2way_llm_for_3way"] = [
            str(row["majority_vote_prediction"] or "")
            if int(row["num_unique_hypotheses"]) < 3
            else str(row["llm_fusion_prediction"] or "")
            for row in df.iter_rows(named=True)
        ]

    print("Model PER")
    print("---------")
    for column, per_value in sorted(model_pers.items(), key=lambda item: item[1]):
        print(f"{column}: {per_value:.5f}")

    print()
    print("Rule Sweep")
    print("----------")
    results: list[tuple[str, float]] = []
    references = df["ground_truth"].to_list()
    for name, predictions in strategies.items():
        results.append((name, score_ipa_cer(references, predictions)))
    for name, per_value in sorted(results, key=lambda item: item[1]):
        print(f"{name}: {per_value:.5f}")

    print()
    print("Two-Way Disagreements")
    print("---------------------")
    print("pair_agrees n majority_exact lone_exact")
    for pair_name, n, majority_rate, lone_rate in summarize_two_way_disagreements(df, prediction_columns):
        print(f"{pair_name:12} {n:4d} {majority_rate:.3f} {lone_rate:.3f}")


if __name__ == "__main__":
    main()
