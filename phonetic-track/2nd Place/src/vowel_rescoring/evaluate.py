#!/usr/bin/env python3
"""
Evaluate K=5 CV XGBoost phoneme rescoring on the full OOF set.

Loads all 5 fold models per pair, averages their predicted probabilities,
and applies the correction.  Since each utterance was held out from exactly
one fold's training, the predictions are honest OOF estimates.

Usage
-----
    uv run python src/vowel_rescoring/evaluate.py
"""

import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from collections import defaultdict

import numpy as np
import polars as pl
import torchaudio
import xgboost as xgb
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.preprocessing.phoneme_tokenizer import PhonemeTokenizer
from src.utils.score import score_ipa_cer
from src.vowel_rescoring.pipeline import (
    CONFUSABLE_PAIRS,
    FEATURE_NAMES,
    _pair_idx_of,
    _get_context,
    get_alignment,
    extract_features,
    build_feature_vector,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

RUNS = [
    {
        "name": "252 (Liam)",
        "logits": PROJECT_ROOT / "outputs/submit-day/18-00-56_annoying-Liam-252/fold_1/val_logits_best.npz",
        "parquet": PROJECT_ROOT / "outputs/submit-day/18-00-56_annoying-Liam-252/oof_predictions_best.parquet",
    },
    {
        "name": "367 (tyfus-brain)",
        "logits": PROJECT_ROOT / "outputs/2026-04-04/07-29-58_tyfus-brain-367/fold_1/val_logits_best.npz",
        "parquet": PROJECT_ROOT / "outputs/2026-04-04/07-29-58_tyfus-brain-367/oof_predictions_best.parquet",
    },
    {
        "name": "368 (smelly-addiction)",
        "logits": PROJECT_ROOT / "outputs/2026-04-04/09-58-24_smelly-addiction-368/fold_1/val_logits_best.npz",
        "parquet": PROJECT_ROOT / "outputs/2026-04-04/09-58-24_smelly-addiction-368/oof_predictions_best.parquet",
    },
]

AUDIO_DIR   = PROJECT_ROOT / "data/audio"
MODEL_DIR   = PROJECT_ROOT / "outputs/vowel_rescoring"

N_FOLDS = 5
SR = 16_000
N_WORKERS = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(MODEL_DIR / "eval_report.txt", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

TOK = PhonemeTokenizer()


# ─── Load K-fold models ─────────────────────────────────────────────────────

def load_kfold_models() -> dict[int, list[xgb.XGBClassifier]]:
    """Load all fold models per pair. Returns {pair_idx: [model_fold0, ..., model_fold4]}."""
    models: dict[int, list[xgb.XGBClassifier]] = {}
    for pair_idx, (ph0, ph1) in enumerate(CONFUSABLE_PAIRS):
        fold_models = []
        for fold_k in range(N_FOLDS):
            path = MODEL_DIR / f"xgb_pair{pair_idx}_fold{fold_k}.json"
            if not path.exists():
                break
            m = xgb.XGBClassifier()
            m.load_model(str(path))
            fold_models.append(m)
        if fold_models:
            models[pair_idx] = fold_models
            log.info("Loaded %d fold models for pair %d (%s vs %s)",
                     len(fold_models), pair_idx, ph0, ph1)
        else:
            log.warning("No models found for pair %d (%s vs %s)", pair_idx, ph0, ph1)
    return models


def ensemble_predict(models: list[xgb.XGBClassifier], X: np.ndarray) -> np.ndarray:
    """Average probabilities from K fold models, return class predictions."""
    probs = np.mean([m.predict_proba(X) for m in models], axis=0)
    return np.argmax(probs, axis=1)


# ─── Per-utterance worker ────────────────────────────────────────────────────

def _correct_utterance(
    utt_id: str,
    ctc_prediction: str,
    logits: np.ndarray,
    kfold_models: dict[int, list[xgb.XGBClassifier]],
) -> tuple[str, dict]:
    """Apply ensemble rescoring to one utterance."""
    audio_path = AUDIO_DIR / f"{utt_id}.flac"
    if not audio_path.exists():
        return ctc_prediction, {}

    try:
        waveform, file_sr = torchaudio.load(str(audio_path))
        if file_sr != SR:
            waveform = torchaudio.functional.resample(waveform, file_sr, SR)
        audio = waveform.mean(0).numpy()
    except Exception:
        return ctc_prediction, {}

    alignment = get_alignment(logits, ctc_prediction, TOK)
    if not alignment:
        return ctc_prediction, {}

    phonemes = [seg["phoneme"] for seg in alignment]
    flips: dict[tuple[str, str], int] = defaultdict(int)

    for i, seg in enumerate(alignment):
        ph = seg["phoneme"]
        pair_idx = _pair_idx_of(ph)
        if pair_idx < 0 or pair_idx not in kfold_models:
            continue

        prev_ph, next_ph = _get_context(alignment, i)

        feats = extract_features(
            audio=audio,
            sr=SR,
            start_sec=seg["start_sec"],
            end_sec=seg["end_sec"],
            logits=logits,
            start_frame=seg["start_frame"],
            end_frame=seg["end_frame"],
            pair_idx=pair_idx,
            tokenizer=TOK,
        )

        feat_vec = build_feature_vector(feats, prev_ph, next_ph, pair_idx, TOK)

        # Average probabilities from all K fold models
        probs = np.mean(
            [m.predict_proba(feat_vec.reshape(1, -1)) for m in kfold_models[pair_idx]],
            axis=0,
        )
        xgb_pred = int(np.argmax(probs, axis=1)[0])
        predicted_ph = CONFUSABLE_PAIRS[pair_idx][xgb_pred]

        if predicted_ph != ph:
            phonemes[i] = predicted_ph
            flips[(ph, predicted_ph)] += 1

    return "".join(phonemes), dict(flips)


# ─── Main ─────────────────────────────────────────────────────────────────────

def evaluate_run(
    run_name: str,
    logits_path: Path,
    parquet_path: Path,
    kfold_models: dict[int, list[xgb.XGBClassifier]],
):
    """Evaluate rescoring on a single CTC run."""
    log.info("\n" + "=" * 70)
    log.info("  RUN: %s", run_name)
    log.info("=" * 70)

    # ── Load logits ──────────────────────────────────────────────────────────
    npz = np.load(str(logits_path), allow_pickle=True)
    all_logits  = npz["logits"]
    offsets     = npz["offsets"]
    lengths     = npz["lengths"]
    npz_utt_ids = npz["utterance_ids"]

    logit_map = {
        uid: (int(offsets[k]), int(lengths[k]))
        for k, uid in enumerate(npz_utt_ids)
    }

    # ── Load OOF predictions + ground truth ──────────────────────────────────
    oof_df = pl.read_parquet(str(parquet_path))
    utt_ids  = oof_df["utterance_id"].to_list()
    pred_map = dict(zip(utt_ids, oof_df["prediction"].to_list()))
    gt_map   = dict(zip(utt_ids, oof_df["ground_truth"].to_list()))

    valid_ids = [uid for uid in utt_ids if uid in logit_map]
    log.info("Evaluating on %d utterances.", len(valid_ids))

    # ── Apply correction in parallel ─────────────────────────────────────────
    corrected_preds: dict[str, str] = {}
    all_flips: dict[tuple[str, str], int] = defaultdict(int)

    futures = {}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for uid in valid_ids:
            off, L = logit_map[uid]
            logits_utt = all_logits[off:off + L]
            pred = pred_map[uid]
            fut = pool.submit(_correct_utterance, uid, pred, logits_utt, kfold_models)
            futures[fut] = uid

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{run_name}"):
            uid = futures[fut]
            try:
                corrected, flips = fut.result()
                corrected_preds[uid] = corrected
                for k, v in flips.items():
                    all_flips[k] += v
            except Exception as exc:
                log.warning("Worker failed for %s: %s", uid, exc)
                corrected_preds[uid] = pred_map[uid]

    # ── Compute PER ──────────────────────────────────────────────────────────
    refs_all      = [gt_map[uid]   for uid in valid_ids]
    baseline_all  = [pred_map[uid] for uid in valid_ids]
    corrected_all = [corrected_preds.get(uid, pred_map[uid]) for uid in valid_ids]

    per_baseline  = score_ipa_cer(refs_all, baseline_all)
    per_corrected = score_ipa_cer(refs_all, corrected_all)
    delta         = per_corrected - per_baseline
    total_flips   = sum(all_flips.values())

    log.info("Baseline  PER : %.4f", per_baseline)
    log.info("Corrected PER : %.4f", per_corrected)
    log.info("Delta PER     : %+.4f   (total flips: %d)", delta, total_flips)

    log.info("\nPer-direction flip counts:")
    for (orig, corr), count in sorted(all_flips.items(), key=lambda x: -x[1]):
        log.info("  %s → %s : %d", orig, corr, count)

    return {"name": run_name, "baseline": per_baseline, "corrected": per_corrected, "delta": delta}


def main():
    kfold_models = load_kfold_models()
    if not kfold_models:
        log.error("No models loaded. Run train_xgb.py first.")
        sys.exit(1)

    results = []
    for run in RUNS:
        if not run["logits"].exists():
            log.warning("Skipping %s — logits not found", run["name"])
            continue
        if not run["parquet"].exists():
            log.warning("Skipping %s — parquet not found", run["name"])
            continue
        r = evaluate_run(run["name"], run["logits"], run["parquet"], kfold_models)
        results.append(r)

    # ── Summary table ────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info("%-30s %8s %8s %8s", "Run", "Baseline", "Rescored", "Delta")
    log.info("-" * 60)
    for r in results:
        log.info("%-30s %8.4f %8.4f %+8.4f", r["name"], r["baseline"], r["corrected"], r["delta"])

    log.info("\nDone.")


if __name__ == "__main__":
    main()
