"""
Prosodic vowel rescoring pipeline.

Corrects confusable vowel pairs from a CTC 1-best output using
duration, pitch, energy, and acoustic confidence features fed into
per-pair XGBoost classifiers.

Confusable pairs
----------------
  Pair 0 – ə vs ʌ   (schwa vs wedge)
  Pair 1 – ɪ vs i   (lax-i vs tense-i)
  Pair 2 – ɛ vs æ   (dress vs trap)
  Pair 3 – ɔ vs ɑ   (thought vs lot)
  Pair 4 – e vs ɛ   (face vs dress)
  Pair 5 – ɹ vs ɚ   (approximant-r vs rhoticized schwa)
  Pair 6 – d vs ð   (stop vs dental fricative)

Pipeline steps
--------------
  1. get_alignment          – CTC forced alignment → phoneme timestamps
  2. extract_features       – prosodic + acoustic features for one segment
  3. correct_sequence       – apply XGBoost to flip confusable vowels
  4. post_process_utterance – end-to-end wrapper
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
import warnings as _warnings
# Suppress torchaudio 2.9 deprecation noise for forced_align
_warnings.filterwarnings("ignore", message=".*forced_align.*deprecated.*", category=UserWarning)
import torchaudio.functional as F_ta
import librosa

# ─── Constants ────────────────────────────────────────────────────────────────

# WavLM-large feature extractor: 7 conv layers with effective stride 320 at 16 kHz
# → 16 000 / 320 = 50 frames per second
WAVLM_FRAME_RATE: float = 50.0

SIL_TOKEN: str = "<SIL>"
SIL_ID: int = -1  # sentinel used in feature vector for missing context

# (class-0 phoneme, class-1 phoneme) for each pair
CONFUSABLE_PAIRS: list[tuple[str, str]] = [
    ("ə", "ʌ"),  # pair 0 – schwa vs wedge
    ("ɪ", "i"),  # pair 1 – lax-i vs tense-i
    ("ɛ", "æ"),  # pair 2 – dress vs trap
    ("ɔ", "ɑ"),  # pair 3 – thought vs lot
    ("e", "ɛ"),  # pair 4 – face vs dress
    ("ɹ", "ɚ"),  # pair 5 – approximant-r vs rhoticized schwa
    ("d", "ð"),  # pair 6 – stop vs dental fricative
]

_CONFUSABLE_SET: frozenset[str] = frozenset(
    ph for pair in CONFUSABLE_PAIRS for ph in pair
)

# Feature column order shared between training and inference
FEATURE_NAMES: list[str] = [
    "duration_ms",
    "mean_pitch_hz",
    "mean_energy_rms",
    "ctc_logit_class0",   # mean log-prob of pair's class-0 phoneme over segment
    "ctc_logit_class1",   # mean log-prob of pair's class-1 phoneme over segment
    "ctc_pred_enc",       # 0 if model favours class-0, 1 if it favours class-1
]


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _pair_idx_of(phoneme: str) -> int:
    """Return index into CONFUSABLE_PAIRS for *phoneme*, or -1 if not confusable."""
    for idx, (a, b) in enumerate(CONFUSABLE_PAIRS):
        if phoneme in (a, b):
            return idx
    return -1


def _phoneme_id(phoneme: str, tokenizer) -> int:
    """Vocab ID for *phoneme*, or SIL_ID for the <SIL> boundary token."""
    if phoneme == SIL_TOKEN:
        return SIL_ID
    return tokenizer.vocab.get(phoneme, SIL_ID)


def _get_context(alignment: list[dict], idx: int) -> tuple[str, str]:
    """
    Return (prev_phoneme, next_phoneme) skipping space tokens.
    Pads with SIL_TOKEN at utterance boundaries.
    """
    prev_ph = SIL_TOKEN
    for j in range(idx - 1, -1, -1):
        if alignment[j]["phoneme"] != " ":
            prev_ph = alignment[j]["phoneme"]
            break

    next_ph = SIL_TOKEN
    for j in range(idx + 1, len(alignment)):
        if alignment[j]["phoneme"] != " ":
            next_ph = alignment[j]["phoneme"]
            break

    return prev_ph, next_ph


def _log_softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted - logsumexp


# ─── Step 1 : Forced alignment ───────────────────────────────────────────────

def get_alignment(
    logits: np.ndarray,
    phoneme_sequence: str,
    tokenizer,
    frame_rate: float = WAVLM_FRAME_RATE,
) -> list[dict]:
    """
    CTC forced alignment of a phoneme string against raw emission logits.

    Uses ``torchaudio.functional.forced_align`` which implements an O(T·S)
    Viterbi search that guarantees the returned path decodes exactly to
    *phoneme_sequence*.

    Parameters
    ----------
    logits:
        Raw (pre-softmax) model logits, shape ``(T, V)``.
    phoneme_sequence:
        Character-level IPA string as produced by the CTC decoder, e.g.
        ``"hɛloʊ wɚld"``.  Spaces count as valid tokens.
    tokenizer:
        ``PhonemeTokenizer`` with ``.vocab``, ``.blank_token_id``, and
        ``.pad_token_id`` attributes.
    frame_rate:
        Emission frames per second.  50 for WavLM-large (default).

    Returns
    -------
    list[dict]
        One dict per aligned phoneme token (including spaces) with keys:
        ``phoneme``, ``start_frame``, ``end_frame``,
        ``start_sec``, ``end_sec``.
        Returns an empty list if alignment is impossible or fails.
    """
    blank_id = tokenizer.blank_token_id

    # Encode every character; skip blank / pad artefacts from the tokenizer
    target_ids: list[int] = [
        tid
        for tid in tokenizer(phoneme_sequence)
        if tid not in (blank_id, tokenizer.pad_token_id)
    ]
    if not target_ids:
        return []

    T, _ = logits.shape
    S = len(target_ids)

    # CTC minimum-length constraint: T ≥ 2·S − 1
    if T < 2 * S - 1:
        return []

    log_probs = torch.nn.functional.log_softmax(
        torch.from_numpy(logits.astype(np.float32)), dim=-1
    )  # (T, V)

    try:
        paths, _ = F_ta.forced_align(
            log_probs.unsqueeze(0),                   # (1, T, V)
            torch.tensor([target_ids], dtype=torch.long),  # (1, S)
            input_lengths=torch.tensor([T]),
            target_lengths=torch.tensor([S]),
            blank=blank_id,
        )
        path: list[int] = paths.squeeze(0).tolist()  # length T
    except Exception:
        return []

    # ── Decode path → phoneme segments ───────────────────────────────────────
    # Merge consecutive frames with the same non-blank token into one segment.
    # A blank frame ends the current segment; a different non-blank token
    # transitions directly without requiring a blank.
    segments: list[dict] = []
    current_tok: int | None = None
    seg_start: int = 0

    for t, tok in enumerate(path):
        if tok == blank_id:
            # Blank frame → close any open segment
            if current_tok is not None:
                segments.append({
                    "phoneme":     tokenizer.decode([current_tok]),
                    "start_frame": seg_start,
                    "end_frame":   t,
                    "start_sec":   seg_start / frame_rate,
                    "end_sec":     t / frame_rate,
                })
                current_tok = None
        else:
            # Non-blank frame
            if tok != current_tok:
                # New (or first) phoneme: close previous segment if any
                if current_tok is not None:
                    segments.append({
                        "phoneme":     tokenizer.decode([current_tok]),
                        "start_frame": seg_start,
                        "end_frame":   t,
                        "start_sec":   seg_start / frame_rate,
                        "end_sec":     t / frame_rate,
                    })
                current_tok = tok
                seg_start = t
            # else: same token continues – nothing to do

    # Close the last open segment
    if current_tok is not None:
        segments.append({
            "phoneme":     tokenizer.decode([current_tok]),
            "start_frame": seg_start,
            "end_frame":   T,
            "start_sec":   seg_start / frame_rate,
            "end_sec":     T / frame_rate,
        })

    return segments


# ─── Step 2 : Feature extraction ─────────────────────────────────────────────

def extract_features(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    logits: np.ndarray,
    start_frame: int,
    end_frame: int,
    pair_idx: int,
    tokenizer,
) -> dict:
    """
    Extract prosodic and acoustic-confidence features for one phoneme segment.

    Parameters
    ----------
    audio:
        1-D float32 waveform at *sr* Hz.
    sr:
        Sample rate (16 000 for this project).
    start_sec, end_sec:
        Temporal extent of the phoneme in seconds.
    logits:
        Raw logits for the full utterance, shape ``(T, V)``.
    start_frame, end_frame:
        Corresponding frame indices into *logits*.
    pair_idx:
        0 for the ə/ʌ pair, 1 for the ɪ/i pair.
    tokenizer:
        ``PhonemeTokenizer``.

    Returns
    -------
    dict
        Keys: ``duration_ms``, ``mean_pitch_hz``, ``mean_energy_rms``,
        ``ctc_logit_class0``, ``ctc_logit_class1``.
    """
    pair = CONFUSABLE_PAIRS[pair_idx]
    id0, id1 = tokenizer.vocab[pair[0]], tokenizer.vocab[pair[1]]

    # ── Duration ──────────────────────────────────────────────────────────────
    duration_ms = float((end_sec - start_sec) * 1000.0)

    # ── Audio segment ─────────────────────────────────────────────────────────
    s_samp = int(start_sec * sr)
    e_samp = int(end_sec * sr)
    segment = audio[s_samp:e_samp].astype(np.float32)

    # Guard: yin needs at least frame_length samples
    min_samples = 2048
    if len(segment) < min_samples:
        pad = np.zeros(min_samples - len(segment), dtype=np.float32)
        segment = np.concatenate([segment, pad])

    # ── Fundamental frequency (F0) via YIN ───────────────────────────────────
    # librosa.yin is ~10× faster than pyin (no HMM smoothing) and sufficient
    # for training/inference since pitch contributes <2% to XGBoost decisions.
    # fmin/fmax cover children's speech range (C2 ≈ 65 Hz, C7 ≈ 2093 Hz).
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0 = librosa.yin(
                segment,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
                frame_length=512,
            )
        valid_f0 = f0[(f0 > librosa.note_to_hz("C2")) & (f0 < librosa.note_to_hz("C7"))]
        mean_pitch_hz = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0
    except Exception:
        mean_pitch_hz = 0.0

    # ── RMS energy ────────────────────────────────────────────────────────────
    try:
        rms = librosa.feature.rms(y=segment)[0]
        mean_energy_rms = float(np.mean(rms))
    except Exception:
        mean_energy_rms = 0.0

    # ── CTC log-probability evidence for each class ───────────────────────────
    # Use the raw logits over the aligned frames so that these features are
    # independent of the alignment used (GT or predicted).
    seg_logits = logits[start_frame:end_frame]
    if len(seg_logits) == 0:
        # Fallback: grab the nearest single frame
        fb = max(0, min(start_frame, logits.shape[0] - 1))
        seg_logits = logits[fb:fb + 1]

    log_probs_seg = _log_softmax_np(seg_logits.astype(np.float32))
    ctc_logit_class0 = float(np.mean(log_probs_seg[:, id0]))
    ctc_logit_class1 = float(np.mean(log_probs_seg[:, id1]))

    return {
        "duration_ms":      duration_ms,
        "mean_pitch_hz":    mean_pitch_hz,
        "mean_energy_rms":  mean_energy_rms,
        "ctc_logit_class0": ctc_logit_class0,
        "ctc_logit_class1": ctc_logit_class1,
    }


def build_feature_vector(
    feats: dict,
    prev_phoneme: str,
    next_phoneme: str,
    pair_idx: int,
    tokenizer,
) -> np.ndarray:
    """
    Assemble a fixed-length feature vector for XGBoost.

    ctc_pred_enc is derived directly from the CTC logit evidence (which class
    has higher mean log-probability), making it consistent between training
    (GT-aligned) and inference (prediction-aligned).

    Returns float32 array of length ``len(FEATURE_NAMES)``.
    """
    ctc_pred_enc = 0 if feats["ctc_logit_class0"] >= feats["ctc_logit_class1"] else 1

    return np.array([
        feats["duration_ms"],
        feats["mean_pitch_hz"],
        feats["mean_energy_rms"],
        feats["ctc_logit_class0"],
        feats["ctc_logit_class1"],
        ctc_pred_enc,
    ], dtype=np.float32)


# ─── Step 3 : Sequence correction ────────────────────────────────────────────

def correct_sequence(
    alignment: list[dict],
    audio: np.ndarray,
    sr: int,
    logits: np.ndarray,
    xgb_models: dict,
    tokenizer,
) -> str:
    """
    Walk the aligned sequence and apply XGBoost to flip confusable vowels.

    Parameters
    ----------
    alignment:
        Output of ``get_alignment``.
    audio:
        1-D float32 waveform at *sr* Hz.
    sr:
        Sample rate.
    logits:
        Raw logits for the full utterance, shape ``(T, V)``.
    xgb_models:
        ``{pair_idx: trained XGBClassifier}``.  Missing keys are skipped.
    tokenizer:
        ``PhonemeTokenizer``.

    Returns
    -------
    str
        Corrected phoneme string (same format as the input to
        ``get_alignment``).
    """
    if not alignment:
        return ""

    # Work on a mutable copy of the phoneme labels
    phonemes = [seg["phoneme"] for seg in alignment]

    for i, seg in enumerate(alignment):
        ph = seg["phoneme"]
        pair_idx = _pair_idx_of(ph)
        if pair_idx < 0 or pair_idx not in xgb_models:
            continue

        prev_ph, next_ph = _get_context(alignment, i)

        feats = extract_features(
            audio=audio,
            sr=sr,
            start_sec=seg["start_sec"],
            end_sec=seg["end_sec"],
            logits=logits,
            start_frame=seg["start_frame"],
            end_frame=seg["end_frame"],
            pair_idx=pair_idx,
            tokenizer=tokenizer,
        )

        feat_vec = build_feature_vector(feats, prev_ph, next_ph, pair_idx, tokenizer)
        xgb_pred = int(xgb_models[pair_idx].predict(feat_vec.reshape(1, -1))[0])
        predicted_ph = CONFUSABLE_PAIRS[pair_idx][xgb_pred]

        if predicted_ph != ph:
            phonemes[i] = predicted_ph

    return "".join(phonemes)


# ─── Step 4 : End-to-end wrapper ─────────────────────────────────────────────

def post_process_utterance(
    audio: np.ndarray,
    ctc_logits: np.ndarray,
    greedy_string: str,
    xgb_models: dict,
    tokenizer,
    sr: int = 16_000,
) -> str:
    """
    End-to-end vowel rescoring pipeline for one utterance.

    Parameters
    ----------
    audio:
        1-D float32 waveform, already resampled to *sr* Hz.
    ctc_logits:
        Raw emission logits from the CTC model, shape ``(T, V)``.
    greedy_string:
        1-best phoneme string from the CTC decoder.
    xgb_models:
        ``{pair_idx: trained XGBClassifier}`` – 0 for ə/ʌ, 1 for ɪ/i.
    tokenizer:
        ``PhonemeTokenizer``.
    sr:
        Audio sample rate (default 16 000 Hz).

    Returns
    -------
    str
        Corrected phoneme string.  Falls back to *greedy_string* if
        forced alignment fails (e.g. utterance too short).
    """
    alignment = get_alignment(ctc_logits, greedy_string, tokenizer)
    if not alignment:
        return greedy_string

    return correct_sequence(
        alignment=alignment,
        audio=audio,
        sr=sr,
        logits=ctc_logits,
        xgb_models=xgb_models,
        tokenizer=tokenizer,
    )
