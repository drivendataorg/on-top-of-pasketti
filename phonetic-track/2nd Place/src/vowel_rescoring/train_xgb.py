#!/usr/bin/env python3
"""
Train XGBoost classifiers to rescore confusable phoneme pairs (K=5 CV).

Strategy
--------
* Extract features for ALL utterances in a single parallel pass.
* K=5 fold CV at utterance level: for each fold, train on 4/5, predict
  on the held-out 1/5.  Collect OOF predictions for honest PER eval.
* Save 5 models per pair for inference (average probabilities).

Usage
-----
    uv run python src/vowel_rescoring/train_xgb.py

Outputs
-------
    outputs/vowel_rescoring/xgb_pair{P}_fold{K}.json  (5 models × 7 pairs)
    outputs/vowel_rescoring/training_report.txt
"""

import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import numpy as np
import polars as pl
import torchaudio
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import xgboost as xgb
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.preprocessing.phoneme_tokenizer import PhonemeTokenizer
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

LOGITS_NPZ   = PROJECT_ROOT / "outputs/submit-day/18-00-56_annoying-Liam-252/fold_1/val_logits_best.npz"
OOF_PARQUET  = PROJECT_ROOT / "outputs/submit-day/18-00-56_annoying-Liam-252/oof_predictions_best.parquet"
AUDIO_DIR    = PROJECT_ROOT / "data/audio"
OUT_DIR      = PROJECT_ROOT / "outputs/vowel_rescoring"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 16_000
N_WORKERS = 30      # ThreadPoolExecutor; 32 cores available
N_FOLDS = 5
RANDOM_SEED = 42

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(OUT_DIR / "training_report.txt", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

TOK = PhonemeTokenizer()


# ─── Per-utterance worker ─────────────────────────────────────────────────────

def _process_utterance(
    utt_id: str,
    ground_truth: str,
    logits: np.ndarray,
) -> list[dict]:
    """
    Force-align the ground truth and extract features for every confusable phoneme.

    Returns a list of sample dicts with keys:
        utt_id, pair_idx, feature_vec (float32 ndarray), label (int 0 or 1)
    """
    audio_path = AUDIO_DIR / f"{utt_id}.flac"
    if not audio_path.exists():
        return []

    try:
        waveform, file_sr = torchaudio.load(str(audio_path))
        if file_sr != SR:
            waveform = torchaudio.functional.resample(waveform, file_sr, SR)
        audio = waveform.mean(0).numpy()
    except Exception:
        return []

    alignment = get_alignment(logits, ground_truth, TOK)
    if not alignment:
        return []

    samples = []
    for i, seg in enumerate(alignment):
        ph = seg["phoneme"]
        pair_idx = _pair_idx_of(ph)
        if pair_idx < 0:
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

        pair = CONFUSABLE_PAIRS[pair_idx]
        label = 0 if ph == pair[0] else 1

        samples.append({
            "utt_id":      utt_id,
            "pair_idx":    pair_idx,
            "feature_vec": feat_vec,
            "label":       label,
        })

    return samples


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Load logits ──────────────────────────────────────────────────────────
    log.info("Loading OOF logits from %s", LOGITS_NPZ)
    npz = np.load(str(LOGITS_NPZ), allow_pickle=True)
    all_logits  = npz["logits"]
    offsets     = npz["offsets"]
    lengths     = npz["lengths"]
    npz_utt_ids = npz["utterance_ids"]

    logit_map: dict[str, tuple[int, int]] = {
        uid: (int(offsets[k]), int(lengths[k]))
        for k, uid in enumerate(npz_utt_ids)
    }
    log.info("Loaded logits for %d utterances.", len(logit_map))

    # ── Load OOF ground truth ────────────────────────────────────────────────
    log.info("Loading OOF parquet from %s", OOF_PARQUET)
    oof_df = pl.read_parquet(str(OOF_PARQUET))
    utt_ids = [uid for uid in oof_df["utterance_id"].to_list() if uid in logit_map]
    gt_map  = dict(zip(oof_df["utterance_id"].to_list(), oof_df["ground_truth"].to_list()))
    log.info("Using %d utterances with logits + ground truth.", len(utt_ids))

    # ── Step 1: Extract features for ALL utterances (single pass) ────────────
    log.info("Extracting features for all utterances…")
    all_samples: list[dict] = []
    futures = {}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for uid in utt_ids:
            off, L = logit_map[uid]
            logits_utt = all_logits[off:off + L]
            gt = gt_map[uid]
            fut = pool.submit(_process_utterance, uid, gt, logits_utt)
            futures[fut] = uid

        for fut in tqdm(as_completed(futures), total=len(futures), desc="features"):
            try:
                all_samples.extend(fut.result())
            except Exception as exc:
                log.warning("Worker failed for %s: %s", futures[fut], exc)

    log.info("Total samples extracted: %d", len(all_samples))

    # ── Index samples by utterance ID for fold splitting ─────────────────────
    utt_to_samples: dict[str, list[dict]] = {}
    for s in all_samples:
        utt_to_samples.setdefault(s["utt_id"], []).append(s)

    # Only keep utterances that produced at least one sample
    utt_ids_with_samples = sorted(utt_to_samples.keys())
    log.info("Utterances with confusable phonemes: %d", len(utt_ids_with_samples))

    # ── Step 2: K-fold CV ────────────────────────────────────────────────────
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    utt_arr = np.array(utt_ids_with_samples)

    # Collect OOF predictions per pair: {pair_idx: {sample_index: pred_label}}
    # We'll use a flat list approach: store (pair_idx, label, oof_pred) per sample
    oof_labels = {p: [] for p in range(len(CONFUSABLE_PAIRS))}
    oof_preds  = {p: [] for p in range(len(CONFUSABLE_PAIRS))}

    for fold_k, (train_idx, val_idx) in enumerate(kf.split(utt_arr)):
        train_uids = set(utt_arr[train_idx])
        val_uids   = set(utt_arr[val_idx])

        log.info("Fold %d/%d: train=%d utts, val=%d utts",
                 fold_k + 1, N_FOLDS, len(train_uids), len(val_uids))

        # Split samples by pair and train/val
        train_by_pair: dict[int, tuple[list, list]] = {}
        val_by_pair:   dict[int, tuple[list, list]] = {}

        for uid in train_uids:
            for s in utt_to_samples.get(uid, []):
                p = s["pair_idx"]
                train_by_pair.setdefault(p, ([], []))
                train_by_pair[p][0].append(s["feature_vec"])
                train_by_pair[p][1].append(s["label"])

        for uid in val_uids:
            for s in utt_to_samples.get(uid, []):
                p = s["pair_idx"]
                val_by_pair.setdefault(p, ([], []))
                val_by_pair[p][0].append(s["feature_vec"])
                val_by_pair[p][1].append(s["label"])

        # Train one model per pair for this fold
        for pair_idx, (ph0, ph1) in enumerate(CONFUSABLE_PAIRS):
            if pair_idx not in train_by_pair:
                log.warning("Fold %d, pair %d (%s/%s): no training samples.", fold_k+1, pair_idx, ph0, ph1)
                continue

            X_tr = np.stack(train_by_pair[pair_idx][0])
            y_tr = np.array(train_by_pair[pair_idx][1], dtype=np.int32)

            n0, n1 = int((y_tr == 0).sum()), int((y_tr == 1).sum())
            scale_pos_weight = n0 / n1 if n1 > 0 else 1.0

            model = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                nthread=32,
            )

            # Fit with val set for early stopping monitoring (not used for stopping, just logging)
            if pair_idx in val_by_pair:
                X_ev = np.stack(val_by_pair[pair_idx][0])
                y_ev = np.array(val_by_pair[pair_idx][1], dtype=np.int32)
                model.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)

                # Collect OOF predictions
                y_pred = model.predict(X_ev)
                oof_labels[pair_idx].extend(y_ev.tolist())
                oof_preds[pair_idx].extend(y_pred.tolist())
            else:
                model.fit(X_tr, y_tr, verbose=False)

            # Save model
            save_path = OUT_DIR / f"xgb_pair{pair_idx}_fold{fold_k}.json"
            model.save_model(str(save_path))

        log.info("Fold %d/%d models saved.", fold_k + 1, N_FOLDS)

    # ── Step 3: Report OOF classification metrics per pair ───────────────────
    log.info("\n" + "=" * 70)
    log.info("OOF CLASSIFICATION REPORTS (honest, held-out predictions)")
    log.info("=" * 70)

    for pair_idx, (ph0, ph1) in enumerate(CONFUSABLE_PAIRS):
        if not oof_labels[pair_idx]:
            log.warning("Pair %d (%s/%s): no OOF predictions.", pair_idx, ph0, ph1)
            continue

        y_true = np.array(oof_labels[pair_idx])
        y_pred = np.array(oof_preds[pair_idx])

        report = classification_report(
            y_true, y_pred,
            target_names=[ph0, ph1],
            digits=4,
        )
        log.info("\nPair %d (%s vs %s) – %d OOF samples:\n%s",
                 pair_idx, ph0, ph1, len(y_true), report)

        # Feature importances from last fold as a representative
        last_model_path = OUT_DIR / f"xgb_pair{pair_idx}_fold{N_FOLDS - 1}.json"
        if last_model_path.exists():
            m = xgb.XGBClassifier()
            m.load_model(str(last_model_path))
            importances = dict(zip(FEATURE_NAMES, m.feature_importances_))
            ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            log.info("Feature importances (pair %d, last fold):", pair_idx)
            for feat, imp in ranked:
                log.info("  %-25s %.4f", feat, imp)

    log.info("\nTraining complete. Models saved to %s", OUT_DIR)
    log.info("Model files: xgb_pair{P}_fold{K}.json  (P=0..%d, K=0..%d)",
             len(CONFUSABLE_PAIRS) - 1, N_FOLDS - 1)


if __name__ == "__main__":
    main()
