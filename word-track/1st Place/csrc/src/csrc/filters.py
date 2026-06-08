import pandas as pd


def apply_filter(
    df: pd.DataFrame,
    cer_thresholds: dict[str, float | None],
    wer_thresholds: dict[str, float | None] | None = None,
    filter_mode: str = "and",
) -> pd.DataFrame:
    """年齢別CER/WER閾値でフィルタを適用して DataFrame を返す.

    Args:
        df: utterance単位のDataFrame。少なくとも以下のカラムを含むこと:
            - cer: Character Error Rate
            - age_bucket: 年齢バケット ("3-4", "5-7", "8-11")
            - wer: Word Error Rate (wer_thresholds使用時)
        cer_thresholds: age_bucketごとのCER閾値。Noneはフィルタなし。
        wer_thresholds: age_bucketごとのWER閾値。Noneはフィルタなし。
            未指定の場合はCER閾値のみ適用。
        filter_mode: "and" (両方満たす) or "or" (どちらか満たす)。

    Returns:
        フィルタ適用後の DataFrame
    """
    original_count = len(df)
    mask = pd.Series(False, index=df.index)

    # CER/WER両方のキーを統合してバケット一覧を作成
    all_buckets = set(cer_thresholds.keys())
    if wer_thresholds is not None:
        all_buckets |= set(wer_thresholds.keys())

    for bucket in sorted(all_buckets):
        cer_threshold = cer_thresholds.get(bucket)
        wer_threshold = wer_thresholds.get(bucket) if wer_thresholds is not None else None
        bucket_mask = df["age_bucket"] == bucket
        bucket_count = bucket_mask.sum()

        passed = bucket_mask
        cer_cond = df["cer"] <= cer_threshold if cer_threshold is not None else None
        wer_cond = df["wer"] <= wer_threshold if wer_threshold is not None else None

        if cer_cond is not None and wer_cond is not None:
            combined = (cer_cond | wer_cond) if filter_mode == "or" else (cer_cond & wer_cond)
            passed = passed & combined
        elif cer_cond is not None:
            passed = passed & cer_cond
        elif wer_cond is not None:
            passed = passed & wer_cond

        filtered_count = bucket_count - passed.sum()
        filter_desc = f"cer<={cer_threshold}"
        if wer_thresholds is not None:
            filter_desc += f", wer<={wer_threshold}"

        mask = mask | passed
        print(
            f"Filter(age={bucket}, {filter_desc}): {bucket_count:,} -> {passed.sum():,} "
            f"(-{filtered_count:,}, {filtered_count / bucket_count * 100:.1f}%)"
            if bucket_count > 0
            else f"Filter(age={bucket}, {filter_desc}): 0 samples",
        )

    known_buckets = all_buckets
    unknown_mask = ~df["age_bucket"].isin(known_buckets)
    if unknown_mask.any():
        mask = mask | unknown_mask
        print(f"Filter(age=other): {unknown_mask.sum():,} samples passed (no filter)")

    df = df[mask].reset_index(drop=True)

    filtered_count = original_count - len(df)
    print(
        f"Filter(total): {original_count:,} -> {len(df):,} "
        f"(-{filtered_count:,}, {filtered_count / original_count * 100:.1f}%)",
    )

    return df
