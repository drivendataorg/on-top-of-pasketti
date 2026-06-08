import polars as pl
from pathlib import Path

def analyze_oof_predictions(parquet_path: str):
    print(f"Loading predictions from: {parquet_path}\n")
    
    # Load the data
    df = pl.read_parquet(parquet_path)
    
    # 1. Basic Overall Stats
    total_utterances = df.height
    unique_children = df["child_id"].n_unique()
    exact_matches = df.filter(pl.col("ground_truth") == pl.col("prediction")).height
    
    print("=== OVERALL SUMMARY ===")
    print(f"Total Utterances: {total_utterances}")
    print(f"Unique Children:  {unique_children}")
    print(f"Exact Matches:    {exact_matches} ({(exact_matches / total_utterances) * 100:.2f}%)")
    
    # 2. Add length columns and absolute differences
    df = df.with_columns([
        pl.col("ground_truth").str.len_chars().alias("gt_len"),
        pl.col("prediction").str.len_chars().alias("pred_len"),
    ])
    
    df = df.with_columns([
        (pl.col("pred_len") - pl.col("gt_len")).alias("len_diff"),
        (pl.col("pred_len") - pl.col("gt_len")).abs().alias("abs_len_diff")
    ])

    # Basic length stats
    avg_gt_len = df["gt_len"].mean()
    avg_pred_len = df["pred_len"].mean()
    print(f"Avg Ground Truth Length: {avg_gt_len:.1f} chars")
    print(f"Avg Prediction Length:   {avg_pred_len:.1f} chars\n")

    # 3. Analyze the Worst Mismatches (Hallucinations or Deletions)
    print("=== TOP 5 WORST LENGTH MISMATCHES ===")
    worst_mismatches = df.sort("abs_len_diff", descending=True).head(5)
    
    # Set Polars to print wider strings so we can see the full phonemes
    with pl.Config(fmt_str_lengths=100):
        print(worst_mismatches.select(["utterance_id", "gt_len", "pred_len", "ground_truth", "prediction"]))
        
    # 4. Empty Predictions Check
    empty_preds = df.filter(pl.col("pred_len") == 0).height
    if empty_preds > 0:
        print(f"\nWARNING: Model predicted an empty string for {empty_preds} utterances.")

    # 5. Child-Level Grouping
    print("\n=== TOP 5 HARDEST CHILDREN (By Average Length Mismatch) ===")
    child_stats = (
        df.group_by("child_id")
        .agg([
            pl.len().alias("utterance_count"),
            pl.col("abs_len_diff").mean().alias("avg_len_error"),
            (pl.col("ground_truth") == pl.col("prediction")).sum().alias("exact_match_count")
        ])
        .filter(pl.col("utterance_count") >= 5) # Filter out kids with very little data
        .sort("avg_len_error", descending=True)
    )
    print(child_stats.head(5))


if __name__ == "__main__":
    file_path = "/home/epochvipc8/repos/speech_phonetic_track/outputs/2026-03-02/16-59-19_robust-trust_the_llm-60/fold_1/oof_predictions_best.parquet"
    
    if Path(file_path).exists():
        analyze_oof_predictions(file_path)
    else:
        print(f"File not found: {file_path}")