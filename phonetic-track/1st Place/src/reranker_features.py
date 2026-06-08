"""Feature computation for tree reranker inference.

Shared between offline evaluation (ensemble-fold0.py) and Docker inference
(submit2.py). Given in-memory CTC logprobs from multiple models, builds
a DataFrame of (uid, candidate, features) for tree model prediction.
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import logging

_log = logging.getLogger(__name__)


def _is_ctc_score_feature(feature_name):
    feature_name = str(feature_name)
    if 'ctc_score' in feature_name:
        return True
    if feature_name.startswith('word_score_diff_'):
        return True
    if feature_name == 'word_primary_diff_mean':
        return True
    if feature_name.endswith('_score_ctc_rank_gap'):
        return True
    if feature_name.endswith('_vs_ctc_best_gap'):
        return True
    return False


def _normalize_candidate_text(text, normalize_text_fn=None):
    if text is None:
        return ''
    if isinstance(text, float) and np.isnan(text):
        return ''
    text = str(text)
    if callable(normalize_text_fn):
        try:
            text = normalize_text_fn(text)
        except Exception:
            text = str(text)
    return str(text).strip()


def _get_model_blank_id(model_name, log_probs, blank_id, model_ctc_meta=None):
    meta = (model_ctc_meta or {}).get(model_name) or {}
    if meta.get('blank_id') is not None:
        return int(meta['blank_id'])
    if meta.get('blank_last', False):
        return int(log_probs.shape[-1] - 1)
    return int(blank_id)


def _encode_candidate_text_for_model(cand_text, model_name, char_to_id=None, model_ctc_meta=None):
    meta = (model_ctc_meta or {}).get(model_name) or {}
    text_to_ids = meta.get('text_to_ids')
    if callable(text_to_ids):
        try:
            return [int(x) for x in (text_to_ids(cand_text) or [])]
        except Exception:
            return []
    if char_to_id is not None:
        return [char_to_id[ch] for ch in cand_text if ch in char_to_id]
    return []


def _decode_nbest_hyps_for_model(log_probs, model_name, blank_id, beam_width, nbest,
                                 id_to_char=None, model_ctc_meta=None,
                                 normalize_text_fn=None):
    from src.ctc_decode import prefix_beam_search_nbest

    meta = (model_ctc_meta or {}).get(model_name) or {}
    decode_ids_to_text = meta.get('decode_ids_to_text')
    model_blank_id = _get_model_blank_id(model_name, log_probs, blank_id, model_ctc_meta)
    if callable(decode_ids_to_text):
        raw_hyps = prefix_beam_search_nbest(
            log_probs, model_blank_id, beam_width, nbest=nbest, id_to_char=None)
        hyps = []
        for score, token_ids in raw_hyps:
            try:
                text = decode_ids_to_text(token_ids)
            except Exception:
                text = ''
            text = _normalize_candidate_text(text, normalize_text_fn)
            if text:
                hyps.append((score, text))
        return hyps
    hyps = prefix_beam_search_nbest(
        log_probs, model_blank_id, beam_width, nbest=nbest, id_to_char=id_to_char)
    return [(_score, _normalize_candidate_text(text, normalize_text_fn)) for _score, text in hyps]


# ---- CTC Force Alignment (Viterbi) ----

def ctc_force_align(log_probs, token_ids, blank=0):
    """Viterbi CTC force alignment: find best frame-to-token assignment.

    Returns per-token info: (token_confidence, duration_in_frames) for each token,
    plus the blank_frame_ratio over the whole sequence.

    Args:
        log_probs: (T, V) numpy float32 log-probabilities.
        token_ids: list[int], label sequence (no blanks).
        blank: blank token id.

    Returns:
        dict with keys:
            'token_confidences': list[float] - per-token average log-prob
            'token_durations': list[int] - frames assigned to each token
            'blank_frame_ratio': float - fraction of frames assigned to blank
    """
    T, V = log_probs.shape
    L = len(token_ids)

    if L == 0:
        return {
            'token_confidences': [],
            'token_durations': [],
            'blank_frame_ratio': 1.0,
        }

    # Build CTC label sequence: b t0 b t1 b ... tL-1 b  (length 2L+1)
    S = 2 * L + 1
    labels = [blank] * S
    for i, tid in enumerate(token_ids):
        labels[2 * i + 1] = tid

    NEG_INF = -1e30
    dp = np.full((T, S), NEG_INF, dtype=np.float64)
    bt = np.full((T, S), -1, dtype=np.int32)

    dp[0, 0] = log_probs[0, labels[0]]
    if S > 1:
        dp[0, 1] = log_probs[0, labels[1]]

    for t in range(1, T):
        for s in range(S):
            lbl = labels[s]
            emit = float(log_probs[t, lbl])

            best_prev = dp[t - 1, s]
            best_s = s
            if s >= 1 and dp[t - 1, s - 1] > best_prev:
                best_prev = dp[t - 1, s - 1]
                best_s = s - 1
            if s >= 2 and labels[s] != blank and labels[s] != labels[s - 2]:
                if dp[t - 1, s - 2] > best_prev:
                    best_prev = dp[t - 1, s - 2]
                    best_s = s - 2

            if best_prev > NEG_INF:
                dp[t, s] = best_prev + emit
                bt[t, s] = best_s

    # Backtrack
    if dp[T - 1, S - 1] >= dp[T - 1, S - 2]:
        s = S - 1
    else:
        s = S - 2

    path = [s]
    for t in range(T - 1, 0, -1):
        s = bt[t, s]
        path.append(s)
    path.reverse()

    frame_assignments = [labels[s] for s in path]
    n_blank = sum(1 for f in frame_assignments if f == blank)
    blank_ratio = n_blank / T

    token_conf = [[] for _ in range(L)]
    token_dur = [0] * L
    for t, s in enumerate(path):
        if s % 2 == 1:
            tok_idx = s // 2
            token_conf[tok_idx].append(float(log_probs[t, labels[s]]))
            token_dur[tok_idx] += 1

    avg_conf = []
    for i in range(L):
        if token_conf[i]:
            avg_conf.append(float(np.mean(token_conf[i])))
        else:
            avg_conf.append(NEG_INF)

    return {
        'token_confidences': avg_conf,
        'token_durations': token_dur,
        'blank_frame_ratio': blank_ratio,
    }


# ---- IPA Character Sets ----
_IPA_VOWELS = set('eiouɑæɐɔəɚɛɪʊʌ')
_IPA_CONSONANTS = set('bcdfghjklmnprstvwxzçðŋɟɫɬɹɾʁʃʒʔʝθχʧʤ')


# ---- Parallel Infrastructure ----
_MP_BUILD = {}


def _pool_init():
    """Worker initializer: reduce thread oversubscription."""
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def _build_rows_for_uid(uid):
    """Build feature rows for a single uid. Reads shared state from _MP_BUILD."""
    import numpy as np
    import editdistance
    from src.ctc_decode import prefix_beam_search_nbest, ctc_force_score_batch
    from collections import Counter as _Counter

    B = _MP_BUILD
    model_names = B['model_names']
    score_model_names = B['score_model_names']
    candidate_model_names = B['candidate_model_names']
    all_eval_preds = B.get('all_eval_preds', {})
    all_logprobs = B['all_logprobs']
    nbest = B['nbest']
    beam_width = B['beam_width']
    blank_id = B['blank_id']
    id_to_char = B['id_to_char']
    char_to_id = B['char_to_id']
    model_ctc_meta = B.get('model_ctc_meta', {})
    lm = B['lm']
    no_lm_feats = B['no_lm_feats']
    feat_text = B['feat_text']
    feat_ipa = B['feat_ipa']
    feat_ctc_stats = B['feat_ctc_stats']
    feat_align = B['feat_align']
    feat_logprob_proxy = B['feat_logprob_proxy']
    feat_audio = B['feat_audio']
    feat_consensus = B['feat_consensus']
    normalize_text_fn = B.get('normalize_text_fn')

    rows = []
    # Step 1: Collect N-best from score models + eval preds from all models
    per_model_hyps = {}
    candidate_set = set()
    for mn in candidate_model_names:
        hyps = []
        if mn in all_logprobs and uid in all_logprobs[mn]:
            lp = all_logprobs[mn][uid].astype(np.float32)
            hyps = _decode_nbest_hyps_for_model(
                lp, mn, blank_id, beam_width, nbest,
                id_to_char=id_to_char, model_ctc_meta=model_ctc_meta,
                normalize_text_fn=normalize_text_fn)
        per_model_hyps[mn] = hyps
        # Add greedy eval prediction as candidate (for all models including TDT-only)
        if all_eval_preds and mn in all_eval_preds and uid in all_eval_preds[mn]:
            eval_text = _normalize_candidate_text(all_eval_preds[mn][uid], normalize_text_fn)
            if eval_text:
                candidate_set.add(eval_text)
        for _score, text in hyps:
            if text:
                candidate_set.add(text)

    candidates = list(candidate_set)
    if not candidates:
        return []

    # Step 2: CTC force-score all candidates under score models only
    all_token_ids = []
    for cand_text in candidates:
        token_ids = _encode_candidate_text_for_model(
            cand_text, score_model_names[0], char_to_id=char_to_id, model_ctc_meta=model_ctc_meta)
        all_token_ids.append(token_ids)

    ctc_scores = {}
    n_frames_per_model = {}
    for mn in score_model_names:
        lp = all_logprobs[mn][uid].astype(np.float32)
        n_frames_per_model[mn] = lp.shape[0]
        model_blank_id = _get_model_blank_id(mn, lp, blank_id, model_ctc_meta)
        token_ids = [
            _encode_candidate_text_for_model(
                cand_text, mn, char_to_id=char_to_id, model_ctc_meta=model_ctc_meta)
            for cand_text in candidates
        ]
        ctc_scores[mn] = ctc_force_score_batch(lp, token_ids, blank=model_blank_id)
        if mn == score_model_names[0]:
            all_token_ids = token_ids

    # CTC force alignment + logprob proxy need ref model's logprobs
    if feat_align or feat_logprob_proxy:
        ref_mn = score_model_names[0]
        ref_lp = all_logprobs[ref_mn][uid].astype(np.float32)
        ref_blank_id = _get_model_blank_id(ref_mn, ref_lp, blank_id, model_ctc_meta)

    if feat_align:
        align_results = []
        for tids in all_token_ids:
            align_results.append(ctc_force_align(ref_lp, tids, blank=ref_blank_id))

    # Per-model beam rank
    beam_ranks = {}
    best_per_model = {}
    for mn in model_names:
        hyps = per_model_hyps[mn]
        ranks = {}
        for rank, (_s, text) in enumerate(hyps):
            ranks[text] = rank
        # For models without beam search results, use eval pred at rank 0
        if not hyps and all_eval_preds and mn in all_eval_preds:
            primary_text = _normalize_candidate_text(all_eval_preds[mn].get(uid, ''), normalize_text_fn)
            if primary_text:
                ranks[primary_text] = 0
        beam_ranks[mn] = ranks
        best_per_model[mn] = hyps[0][1] if hyps else (
            _normalize_candidate_text(all_eval_preds.get(mn, {}).get(uid, ''), normalize_text_fn)
            if all_eval_preds else '')

    # LM scores
    if not no_lm_feats:
        if lm is not None:
            lm_scores = []
            for cand_text in candidates:
                lm_sc = 0.0
                ctx = ''
                for ch in cand_text:
                    lm_sc += lm.score(ctx, ch)
                    ctx += ch
                lm_sc += lm.score(ctx, '$')
                lm_scores.append(lm_sc)
        else:
            lm_scores = [0.0] * len(candidates)

    n_frames = max(n_frames_per_model.values())

    # Logprob proxy features (utterance-level, computed once per uid)
    if feat_logprob_proxy:
        _lp_probs = np.exp(ref_lp)
        _frame_entropy = -np.sum(_lp_probs * ref_lp, axis=1)
        _utt_entropy_mean = float(np.mean(_frame_entropy))
        _utt_entropy_std = float(np.std(_frame_entropy))
        _utt_entropy_max = float(np.max(_frame_entropy))
        _utt_blank_prob_mean = float(np.mean(_lp_probs[:, blank_id]))
        _utt_top1_prob_mean = float(np.mean(np.max(_lp_probs, axis=1)))
        del _lp_probs

    # Build feature row for each candidate
    for ci, cand_text in enumerate(candidates):
        row = {'uid': uid, 'candidate_text': cand_text}

        # CTC scores from score models only
        scores_arr = np.array([ctc_scores[mn][ci] for mn in score_model_names], dtype=np.float32)
        row['ctc_score_mean'] = float(np.mean(scores_arr))
        row['ctc_score_std'] = float(np.std(scores_arr))
        row['ctc_score_min'] = float(np.min(scores_arr))
        row['ctc_score_max'] = float(np.max(scores_arr))
        row['ctc_score_range'] = float(np.max(scores_arr) - np.min(scores_arr))
        row['n_score_models'] = len(score_model_names)

        # Per-model CTC score columns: real score if available, mean as fallback
        for mn in model_names:
            if mn in ctc_scores:
                row[f'ctc_score_{mn}'] = ctc_scores[mn][ci]
            else:
                row[f'ctc_score_{mn}'] = row['ctc_score_mean']

        # Text length features
        text_len = len(cand_text)
        row['text_len'] = text_len
        row['n_frames'] = n_frames
        row['char_per_frame'] = text_len / max(n_frames, 1)

        # Beam rank features
        n_models_has = 0
        for mn in model_names:
            rank = beam_ranks[mn].get(cand_text, -1)
            row[f'beam_rank_{mn}'] = rank
            if rank >= 0:
                n_models_has += 1
        row['n_models_has'] = n_models_has

        # Edit distance to each model's 1-best
        for mn in model_names:
            best_text = best_per_model[mn]
            row[f'edit_dist_to_best_{mn}'] = editdistance.eval(cand_text, best_text)

        n_spaces = cand_text.count(' ')

        # LM score
        if not no_lm_feats:
            row['lm_score'] = lm_scores[ci]
            row['lm_score_per_char'] = lm_scores[ci] / max(text_len, 1)
            row['lm_score_per_word'] = lm_scores[ci] / max(n_spaces + 1, 1)

        # Text analysis features
        if feat_text:
            row['n_spaces'] = n_spaces
            row['n_words'] = n_spaces + 1
            words = cand_text.split(' ') if cand_text else ['']
            word_lens = [len(w) for w in words]
            row['avg_word_len'] = text_len / max(n_spaces + 1, 1)
            row['max_word_len'] = max(word_lens) if word_lens else 0
            row['min_word_len'] = min(word_lens) if word_lens else 0
            row['n_unique_chars'] = len(set(cand_text))
            if text_len > 0:
                _ch_counts = _Counter(cand_text)
                _probs = np.array(list(_ch_counts.values()), dtype=np.float64) / text_len
                row['char_entropy'] = float(-np.sum(_probs * np.log(_probs + 1e-12)))
            else:
                row['char_entropy'] = 0.0
            max_repeat = 1
            cur_repeat = 1
            for k in range(1, text_len):
                if cand_text[k] == cand_text[k - 1]:
                    cur_repeat += 1
                    if cur_repeat > max_repeat:
                        max_repeat = cur_repeat
                else:
                    cur_repeat = 1
            row['max_char_repeat'] = max_repeat if text_len > 0 else 0
            row['unique_char_ratio'] = len(set(cand_text)) / max(text_len, 1)
            n_repeats = sum(1 for k in range(1, text_len) if cand_text[k] == cand_text[k - 1])
            row['repeat_char_ratio'] = n_repeats / max(text_len - 1, 1)

        # IPA phonetic features
        if feat_ipa:
            n_vowels = sum(1 for ch in cand_text if ch in _IPA_VOWELS)
            n_consonants = sum(1 for ch in cand_text if ch in _IPA_CONSONANTS)
            row['n_vowels'] = n_vowels
            row['n_consonants'] = n_consonants
            row['vowel_ratio'] = n_vowels / max(text_len, 1)
            row['consonant_ratio'] = n_consonants / max(text_len, 1)
            row['vc_ratio'] = n_vowels / max(n_consonants, 1)
            row['n_length_marks'] = cand_text.count('ː')

        # CTC distribution features
        if feat_ctc_stats:
            row['ctc_score_median'] = float(np.median(scores_arr))
            abs_mean = abs(float(np.mean(scores_arr)))
            row['ctc_score_cv'] = float(np.std(scores_arr)) / max(abs_mean, 1e-8)
            if len(scores_arr) >= 3:
                from scipy.stats import skew as _skew, kurtosis as _kurtosis
                row['ctc_score_skew'] = float(_skew(scores_arr))
                row['ctc_score_kurtosis'] = float(_kurtosis(scores_arr))
            else:
                row['ctc_score_skew'] = 0.0
                row['ctc_score_kurtosis'] = 0.0
            q75, q25 = np.percentile(scores_arr, [75, 25])
            row['ctc_score_iqr'] = float(q75 - q25)

        # CTC alignment features
        if feat_align:
            aln = align_results[ci]
            row['blank_frame_ratio'] = aln['blank_frame_ratio']
            tok_conf = aln['token_confidences']
            tok_dur = aln['token_durations']
            if tok_conf:
                row['avg_frame_confidence'] = float(np.mean(tok_conf))
                row['min_phoneme_confidence'] = float(np.min(tok_conf))
                row['max_phoneme_confidence'] = float(np.max(tok_conf))
                row['std_phoneme_confidence'] = float(np.std(tok_conf))
            else:
                row['avg_frame_confidence'] = -30.0
                row['min_phoneme_confidence'] = -30.0
                row['max_phoneme_confidence'] = -30.0
                row['std_phoneme_confidence'] = 0.0
            if tok_dur:
                dur_arr = np.array(tok_dur, dtype=np.float64)
                row['phoneme_dur_mean'] = float(np.mean(dur_arr))
                row['phoneme_dur_std'] = float(np.std(dur_arr))
                row['phoneme_dur_min'] = float(np.min(dur_arr))
                row['phoneme_dur_max'] = float(np.max(dur_arr))
                dur_z = (dur_arr - np.mean(dur_arr)) / max(float(np.std(dur_arr)), 1e-8)
                row['phoneme_dur_zscore_max'] = float(np.max(np.abs(dur_z)))
                row['single_frame_phoneme_ratio'] = float(np.sum(dur_arr <= 1)) / max(len(dur_arr), 1)
            else:
                row['phoneme_dur_mean'] = 0.0
                row['phoneme_dur_std'] = 0.0
                row['phoneme_dur_min'] = 0.0
                row['phoneme_dur_max'] = 0.0
                row['phoneme_dur_zscore_max'] = 0.0
                row['single_frame_phoneme_ratio'] = 0.0

        # Logprob proxy features (utterance-level)
        if feat_logprob_proxy:
            row['entropy_mean'] = _utt_entropy_mean
            row['entropy_std'] = _utt_entropy_std
            row['entropy_max'] = _utt_entropy_max
            row['blank_prob_mean'] = _utt_blank_prob_mean
            row['top1_prob_mean'] = _utt_top1_prob_mean
            row['model_ctc_std'] = float(np.std(scores_arr))
            _ranks_for_ent = []
            for mn in score_model_names:
                mn_scores = ctc_scores[mn]
                _r = sorted(range(len(mn_scores)), key=lambda x: mn_scores[x], reverse=True)
                _ranks_for_ent.append(_r.index(ci))
            _ranks_arr = np.array(_ranks_for_ent, dtype=np.float64)
            row['model_rank_std'] = float(np.std(_ranks_arr))
            _model_best_idx = int(np.argmax(scores_arr))
            for mi, mn in enumerate(score_model_names):
                row[f'is_best_model_{mn}'] = 1 if mi == _model_best_idx else 0
            # Non-score models always 0
            for mn in model_names:
                if mn not in score_model_names:
                    row[f'is_best_model_{mn}'] = 0

        # Audio / speaking rate features
        if feat_audio:
            duration_sec = n_frames * 0.04  # 40ms per subsampled frame
            row['duration_sec'] = duration_sec
            row['chars_per_sec'] = text_len / max(duration_sec, 0.01)
            row['words_per_sec'] = (n_spaces + 1) / max(duration_sec, 0.01)

        # Beam rank aggregate features
        valid_ranks = [beam_ranks[mn].get(cand_text, -1) for mn in model_names]
        valid_ranks_pos = [r for r in valid_ranks if r >= 0]
        row['beam_rank_mean'] = np.mean(valid_ranks_pos) if valid_ranks_pos else nbest
        row['beam_rank_min'] = min(valid_ranks_pos) if valid_ranks_pos else nbest
        row['beam_rank_max'] = max(valid_ranks_pos) if valid_ranks_pos else nbest
        if feat_consensus:
            row['n_models_in_top3'] = sum(1 for r in valid_ranks if 0 <= r < 3)

        # Pairwise edit distance / consensus
        if feat_consensus:
            pairwise_sum = 0
            n_pairs = 0
            for oi, other_text in enumerate(candidates):
                if oi != ci:
                    pairwise_sum += editdistance.eval(cand_text, other_text)
                    n_pairs += 1
            row['mean_pairwise_edit_dist'] = pairwise_sum / max(n_pairs, 1)
            mean_cer_to_others = 0.0
            if n_pairs > 0 and text_len > 0:
                mean_cer_to_others = pairwise_sum / (n_pairs * text_len)
            row['consensus_score'] = 1.0 - min(mean_cer_to_others, 1.0)
            row['n_exact_best'] = sum(1 for mn in model_names if best_per_model[mn] == cand_text)

        rows.append(row)
    return rows


def build_reranker_features(all_logprobs, model_names, nbest=10, beam_width=10,
                            lm=None, verbose=False,
                            feat_text=False, feat_ipa=False,
                            feat_ctc_stats=False, no_ctc_score_feats=False,
                            feat_audio=False,
                            feat_consensus=False, feat_mbr=False,
                            feat_group_ext=False,
                            feat_align=False, feat_logprob_proxy=False,
                            no_lm_feats=False, n_workers=0,
                            all_eval_preds=None,
                            model_ctc_meta=None,
                            normalize_text_fn=None):
    """Build feature DataFrame for tree reranker from in-memory CTC logprobs.

    Supports partial logprobs: not all models need to have CTC logprobs.
    Models without logprobs contribute only greedy predictions as candidates
    (via all_eval_preds), matching offline ensemble_feats.py behavior.

    Args:
        all_logprobs: dict {model_name: {uid: numpy (T, V) float32/16}}
                      May contain only a subset of model_names (score models).
        model_names: list of ALL model name strings (order matters for features)
        nbest: number of N-best candidates per model
        beam_width: beam width for beam search
        lm: optional CharNgramLM for language model scores
        verbose: print progress
        feat_text..feat_logprob_proxy: feature group flags (False=skip)
        feat_mbr: if True, add heavier MBR reranker features
        no_ctc_score_feats: if True, exclude CTC-score-derived feature columns
        no_lm_feats: if True, skip LM score features entirely
        all_eval_preds: dict {model_name: {uid: text}} greedy predictions for
                        all models (including those without logprobs). Required
                        when not all model_names have entries in all_logprobs.

    Returns:
        df: DataFrame with uid, candidate_text, and feature columns
        feat_cols: list of feature column names
    """
    from src.ctc_decode import prefix_beam_search_nbest, ctc_force_score_batch
    from src.models.base import IPA_ID_TO_CHAR, IPA_CTC_BLANK
    import editdistance

    blank_id = IPA_CTC_BLANK
    id_to_char = IPA_ID_TO_CHAR
    char_to_id = {ch: cid for cid, ch in id_to_char.items()}

    # Split models into score models (have logprobs) and candidate models (all)
    score_model_names = [mn for mn in model_names if mn in all_logprobs and all_logprobs[mn]]
    candidate_model_names = list(model_names)
    assert score_model_names, 'tree_reranker requires at least one model with CTC logprobs'

    if all_eval_preds is None:
        all_eval_preds = {}
    if len(score_model_names) < len(model_names) and not all_eval_preds:
        raise ValueError(
            f'Only {len(score_model_names)}/{len(model_names)} models have CTC logprobs, '
            f'but all_eval_preds not provided for candidate generation from non-CTC models')

    # Tree reranking only requires score-model logprobs. Candidate-only models
    # contribute optional greedy-text candidates and may legitimately emit empty
    # strings for some utterances; those empties should not exclude the uid from
    # tree scoring altogether.
    uid_sets = [set(all_logprobs[mn].keys()) for mn in score_model_names]
    common_uids = set.intersection(*uid_sets)
    uids_list = sorted(common_uids)
    total = len(uids_list)

    if verbose:
        _log.debug(f'Score models (CTC logprobs): {len(score_model_names)}/{len(model_names)}')
        if len(score_model_names) < len(model_names):
            non_score = [mn for mn in model_names if mn not in score_model_names]
            _log.debug(f'Candidate-only models (greedy preds): {non_score}')

    import time
    t0 = time.time()

    # Populate shared state for worker function
    global _MP_BUILD
    _MP_BUILD.clear()
    _MP_BUILD.update({
        'model_names': model_names,
        'score_model_names': score_model_names,
        'candidate_model_names': candidate_model_names,
        'all_eval_preds': all_eval_preds,
        'all_logprobs': all_logprobs,
        'nbest': nbest,
        'beam_width': beam_width,
        'blank_id': blank_id,
        'id_to_char': id_to_char,
        'char_to_id': char_to_id,
        'model_ctc_meta': model_ctc_meta or {},
        'lm': lm,
        'no_lm_feats': no_lm_feats,
        'feat_text': feat_text,
        'feat_ipa': feat_ipa,
        'feat_ctc_stats': feat_ctc_stats,
        'feat_align': feat_align,
        'feat_logprob_proxy': feat_logprob_proxy,
        'feat_audio': feat_audio,
        'feat_consensus': feat_consensus,
        'normalize_text_fn': normalize_text_fn,
    })

    # Determine parallelism
    if n_workers == 0:
        n_workers = min(os.cpu_count() or 1, 16)
    use_parallel = n_workers > 1 and total >= 64

    rows = []
    if use_parallel:
        if verbose:
            print(f'  Building features in parallel: {total} uids, {n_workers} workers')
        ctx = mp.get_context('fork')
        pool = ctx.Pool(n_workers, initializer=_pool_init)
        try:
            chunksize = max(1, total // (n_workers * 4))
            done = 0
            for uid_rows in pool.imap_unordered(_build_rows_for_uid, uids_list, chunksize=chunksize):
                rows.extend(uid_rows)
                done += 1
        finally:
            pool.close()
            pool.join()
    else:
        for idx, uid in enumerate(uids_list):
            uid_rows = _build_rows_for_uid(uid)
            rows.extend(uid_rows)
    if verbose:
        elapsed = time.time() - t0
        rate = total / elapsed if elapsed > 0 else 0
        print(f'  Built {len(rows)} candidate rows for {total} utterances in {elapsed:.1f}s ({rate:.1f} utt/s)')

    df = pd.DataFrame(rows)
    if df.empty:
        return df, []

    # ---- Group-relative features ----
    _add_group_relative_features(df, model_names, nbest,
                                  feat_text=feat_text, feat_consensus=feat_consensus,
                                  feat_mbr=feat_mbr,
                                  feat_group_ext=feat_group_ext, no_lm_feats=no_lm_feats)

    # Define feature columns
    exclude = {'uid', 'candidate_text'}
    feat_cols = [c for c in df.columns if c not in exclude]
    if no_ctc_score_feats:
        feat_cols = [c for c in feat_cols if not _is_ctc_score_feature(c)]
    return df, feat_cols


def _add_group_relative_features(df, model_names, nbest,
                                  feat_text=False, feat_consensus=False, feat_mbr=False,
                                  feat_group_ext=False, no_lm_feats=False):
    """Add group-relative features computed per uid group."""
    import editdistance

    # Per-model CTC: rank + per_char (always), zscore (group_ext)
    for mn in model_names:
        col = f'ctc_score_{mn}'
        df[f'{col}_rank'] = df.groupby('uid')[col].rank(ascending=False, method='min')
        df[f'{col}_per_char'] = df[col] / df['text_len'].clip(lower=1)
        if feat_group_ext:
            grp_mn = df.groupby('uid')[col]
            df[f'{col}_zscore'] = (df[col] - grp_mn.transform('mean')) / grp_mn.transform('std').clip(lower=1e-8)

    # CTC mean score group features (always: rank, zscore, diff_from_best)
    df['ctc_score_mean_rank'] = df.groupby('uid')['ctc_score_mean'].rank(ascending=False, method='min')
    grp = df.groupby('uid')['ctc_score_mean']
    df['ctc_score_mean_zscore'] = (df['ctc_score_mean'] - grp.transform('mean')) / grp.transform('std').clip(lower=1e-8)
    df['ctc_score_diff_from_best'] = df['ctc_score_mean'] - grp.transform('max')

    if feat_group_ext:
        df['ctc_score_mean_pct'] = df.groupby('uid')['ctc_score_mean'].rank(pct=True)
        df['ctc_score_diff_from_min'] = df['ctc_score_mean'] - grp.transform('min')
        df['ctc_score_diff_from_group_mean'] = df['ctc_score_mean'] - grp.transform('mean')
        df['ctc_score_diff_from_group_median'] = df['ctc_score_mean'] - grp.transform('median')
        _grp_min = grp.transform('min')
        _grp_range = (grp.transform('max') - _grp_min).clip(lower=1e-8)
        df['ctc_score_minmax_norm'] = (df['ctc_score_mean'] - _grp_min) / _grp_range

    df['ctc_score_mean_per_char'] = df['ctc_score_mean'] / df['text_len'].clip(lower=1)

    if feat_group_ext:
        for mn in model_names:
            col = f'ctc_score_{mn}'
            grp_mn = df.groupby('uid')[col]
            df[f'{col}_diff_from_median'] = df[col] - grp_mn.transform('median')

    for mn in model_names:
        df[f'is_best_{mn}'] = (df[f'beam_rank_{mn}'] == 0).astype(int)
    df['n_models_is_best'] = sum(df[f'is_best_{mn}'] for mn in model_names)

    # Text length group features
    grp_len = df.groupby('uid')['text_len']
    df['text_len_diff_from_median'] = df['text_len'] - grp_len.transform('median')
    if feat_text:
        df['text_len_rank'] = grp_len.rank(method='min')
        df['text_len_zscore'] = (df['text_len'] - grp_len.transform('mean')) / grp_len.transform('std').clip(lower=1e-8)

    df['n_candidates'] = df.groupby('uid')['uid'].transform('count')

    edit_cols = [f'edit_dist_to_best_{mn}' for mn in model_names]
    df['mean_edit_dist_to_best'] = df[edit_cols].mean(axis=1)
    df['min_edit_dist_to_best'] = df[edit_cols].min(axis=1)
    if feat_group_ext:
        df['max_edit_dist_to_best'] = df[edit_cols].max(axis=1)

    # LM score group features
    if not no_lm_feats and feat_group_ext:
        grp_lm = df.groupby('uid')['lm_score']
        df['lm_score_rank'] = grp_lm.rank(ascending=False, method='min')
        df['lm_score_zscore'] = (df['lm_score'] - grp_lm.transform('mean')) / grp_lm.transform('std').clip(lower=1e-8)
        df['lm_score_diff_from_best'] = df['lm_score'] - grp_lm.transform('max')
        df['lm_score_pct'] = grp_lm.rank(pct=True)

    if feat_consensus:
        grp_ped = df.groupby('uid')['mean_pairwise_edit_dist']
        df['mean_pairwise_edit_dist_rank'] = grp_ped.rank(method='min')

    if feat_text and 'n_spaces' in df.columns:
        grp_sp = df.groupby('uid')['n_spaces']
        df['n_spaces_diff_from_median'] = df['n_spaces'] - grp_sp.transform('median')

    # MBR-related features
    if feat_mbr:
        def _mbr_features(group):
            texts = group['candidate_text'].values
            n = len(texts)
            if n <= 1:
                group['is_mbr_selected'] = 1
                group['edit_dist_to_mbr'] = 0
                return group
            avg_cer = np.zeros(n)
            for i in range(n):
                total = 0
                for j in range(n):
                    if i != j:
                        ri = texts[i].strip()
                        rj = texts[j].strip()
                        if ri:
                            total += editdistance.eval(ri, rj) / len(ri)
                avg_cer[i] = total / (n - 1)
            mbr_idx = int(np.argmin(avg_cer))
            mbr_text = texts[mbr_idx]
            group['is_mbr_selected'] = (group['candidate_text'] == mbr_text).astype(int)
            group['edit_dist_to_mbr'] = group['candidate_text'].apply(
                lambda t: editdistance.eval(t, mbr_text))
            return group

        df = df.groupby('uid', group_keys=False).apply(_mbr_features)
    return df
