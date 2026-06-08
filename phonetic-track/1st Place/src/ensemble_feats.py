#!/usr/bin/env python3

import json
import ast
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd


_MP_BUILD = {}
_MODEL_CTC_META_FACTORY = None
_TARGET_ERROR_FN = None
# Optional overrides for word-track: when set, used for consensus/pairwise/MBR features
# instead of raw char-level editdistance.  Set to a callable(str, str) -> int or float.
_PAIRWISE_EDIT_DIST_FN = None   # override for pairwise / consensus distance
_PAIRWISE_EDIT_DIST_NORM = None # override for consensus_score normalizer: callable(dist, n_pairs, cand_text) -> float
_MBR_DIST_FN = None             # override for MBR feature distance: callable(ref_str, hyp_str) -> float (0..1)
_EDIT_DIST_TO_BEST_FN = None    # override for edit_dist_to_best_{mn} feature
_POST_BUILD_HOOK = None         # callable(df, model_names, feat_consensus) -> df; for adding extra features


def _should_use_tqdm(verbose=True, progress_mode=None):
  if not verbose:
    return False
  progress_mode = str(progress_mode or os.environ.get('ENSEMBLE_PROGRESS', 'auto')).lower()
  if progress_mode == 'tqdm':
    return True
  if progress_mode in ('log', 'none'):
    return False
  if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS') or os.environ.get('BUILD_BUILDID'):
    return False
  return bool(sys.stdout.isatty() and sys.stderr.isatty())


def _iter_with_progress(iterable, total, desc, verbose=True, progress_mode=None):
  progress_mode = str(progress_mode or os.environ.get('ENSEMBLE_PROGRESS', 'auto')).lower()
  if not verbose or progress_mode == 'none':
    for item in iterable:
      yield item
    return

  if _should_use_tqdm(verbose=verbose, progress_mode=progress_mode):
    from tqdm import tqdm
    yield from tqdm(iterable, total=total, desc=desc, dynamic_ncols=True,
                    mininterval=0.5, leave=True,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    return

  milestones = [1, 2, 5] + list(range(10, 101, 10))
  thresholds = []
  for pct in milestones:
    if total <= 0:
      break
    thresholds.append(max(1, int(round(total * pct / 100.0))))
  thresholds = sorted(set(thresholds + ([total] if total > 0 else [])))
  next_idx = 0
  start_time = time.time()
  for index, item in enumerate(iterable, start=1):
    yield item
    if next_idx >= len(thresholds) or index < thresholds[next_idx]:
      continue
    while next_idx < len(thresholds) and index >= thresholds[next_idx]:
      next_idx += 1
    elapsed = time.time() - start_time
    rate = index / elapsed if elapsed > 0 else 0.0
    eta = (total - index) / rate if rate > 0 else 0.0
    pct = (100.0 * index / total) if total > 0 else 0.0
    print(f'{desc}: {index}/{total} ({pct:.0f}%, {rate:.1f} it/s, ETA {eta:.0f}s)', flush=True)


def _greedy_ctc_decode_numpy(log_probs, blank_id=0, id_to_char=None):
  token_ids = np.argmax(log_probs, axis=-1)
  decoded = []
  prev = -1
  for token_id in token_ids:
    if token_id != prev:
      if token_id != blank_id:
        decoded.append(int(token_id))
      prev = token_id

  if id_to_char is not None:
    return ''.join(id_to_char.get(c, '') for c in decoded)
  return str(decoded)


def _char_cer(normalize_ipa, ref, hyp):
  ref = normalize_ipa(ref).strip()
  hyp = normalize_ipa(hyp).strip()
  if not ref:
    return 0.0 if not hyp else 1.0
  import editdistance
  return editdistance.eval(ref, hyp) / len(ref)


def _score_lm_candidate(lm, cand_text):
  if lm is None:
    return 0.0
  if hasattr(lm, 'score_text'):
    return float(lm.score_text(cand_text))
  lm_sc = 0.0
  ctx = ''
  for ch in cand_text:
    lm_sc += lm.score(ctx, ch)
    ctx += ch
  lm_sc += lm.score(ctx, '$')
  return float(lm_sc)


def _normalize_candidate_text(normalize_ipa, text):
  if text is None:
    return ''
  if isinstance(text, float) and np.isnan(text):
    return ''
  return normalize_ipa(str(text)).strip()


def _parse_serialized_text_list(normalize_ipa, value):
  if value is None:
    return []
  if isinstance(value, float) and np.isnan(value):
    return []
  if isinstance(value, (list, tuple)):
    items = value
  else:
    text = str(value).strip()
    if not text:
      return []
    try:
      items = json.loads(text)
    except Exception:
      try:
        items = ast.literal_eval(text)
      except Exception:
        return []
  if not isinstance(items, (list, tuple)):
    return []
  out = []
  for item in items:
    norm = _normalize_candidate_text(normalize_ipa, item)
    if norm:
      out.append(norm)
  return out


def _get_model_ctc_meta(model_names, all_logprobs):
  if callable(_MODEL_CTC_META_FACTORY):
    try:
      meta = _MODEL_CTC_META_FACTORY(model_names, all_logprobs)
      return meta or {}
    except Exception as e:
      print(f'WARNING: model CTC meta factory failed: {e}')
      return {}
  return {}


def _get_target_error(normalize_ipa, ref, hyp):
  if callable(_TARGET_ERROR_FN):
    return float(_TARGET_ERROR_FN(ref, hyp))
  return _char_cer(normalize_ipa, ref, hyp)


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


def _decode_nbest_hyps_for_model(prefix_beam_search_nbest, log_probs, model_name,
                                 blank_id, beam_width, nbest, id_to_char,
                                 normalize_ipa, model_ctc_meta=None):
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
      text = _normalize_candidate_text(normalize_ipa, text)
      if text:
        hyps.append((score, text))
    return hyps
  hyps = prefix_beam_search_nbest(
      log_probs, model_blank_id, beam_width, nbest=nbest, id_to_char=id_to_char)
  return [(_score, _normalize_candidate_text(normalize_ipa, text)) for _score, text in hyps]


def _safe_float(value):
  if value is None:
    return np.nan
  if isinstance(value, str):
    value = value.strip()
    if not value:
      return np.nan
  try:
    result = float(value)
  except (TypeError, ValueError):
    return np.nan
  return result if np.isfinite(result) else np.nan


def _ctc_force_align(log_probs, token_ids, blank=0):
  T, _ = log_probs.shape
  L = len(token_ids)

  if L == 0:
    return {
        'token_confidences': [],
        'token_durations': [],
        'blank_frame_ratio': 1.0,
        'frame_assignments': [blank] * T,
    }

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
      'frame_assignments': frame_assignments,
  }


def load_word_labels(word_label_file, word_label_col='', verbose=True):
  if not word_label_file:
    return {}

  def _pick_uid(row):
    for key in ('utterance_id', 'id', 'uid'):
      value = row.get(key)
      if value is not None and str(value).strip():
        return str(value).strip()
    return None

  def _pick_word(row):
    candidates = []
    if word_label_col:
      candidates.append(word_label_col)
    candidates.extend([
        'orthographic_text', 'word_text', 'text', 'transcript', 'transcription',
        'label_text', 'word_label', 'orthography',
    ])
    for key in candidates:
      value = row.get(key)
      if value is not None and str(value).strip():
        return str(value).strip()
    return ''

  labels = {}
  if word_label_file.endswith('.csv'):
    df = pd.read_csv(word_label_file)
    for row in df.to_dict('records'):
      uid = _pick_uid(row)
      if uid is None:
        continue
      labels[uid] = _pick_word(row)
  else:
    with open(word_label_file) as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        row = json.loads(line)
        uid = _pick_uid(row)
        if uid is None:
          continue
        labels[uid] = _pick_word(row)

  if verbose:
    print(f'  Loaded word labels: {len(labels)} from {word_label_file}')
  return labels


def _build_eval_uid_set_mismatch_message(eval_uid_infos):
  if not eval_uid_infos:
    return 'Ensemble eval-set mismatch detected, but no eval UID info was collected.'

  uid_sets = [info['uids'] for info in eval_uid_infos.values()]
  common_uids = set.intersection(*uid_sets) if uid_sets else set()
  union_uids = set.union(*uid_sets) if uid_sets else set()
  lines = [
      'Ensemble model eval-set mismatch detected. Selected models do not share the same eval UID set.',
      'This usually means train/eval directories were mixed, for example model vs model.eval.',
      f'Common utterances: {len(common_uids)} / union {len(union_uids)}',
      'Per-model eval coverage:',
  ]
  for model_name, info in eval_uid_infos.items():
    source_counts = info.get('source_counts') or {}
    source_desc = ', '.join(f'{key or "<empty>"}={value}' for key, value in sorted(source_counts.items()))
    if not source_desc:
      source_desc = 'n/a'
    only_here = len(info['uids'] - common_uids)
    missing_from_model = len(common_uids - info['uids'])
    lines.append(
        f'  - {model_name}: rows={info["n_rows"]}, uids={info["n_uids"]}, '
        f'only_here={only_here}, missing_common={missing_from_model}, sources=[{source_desc}]')

  suspicious_pairs = []
  names = list(eval_uid_infos.keys())
  name_set = set(names)
  for name in names:
    if name.endswith('.eval'):
      base = name[:-5]
      if base in name_set:
        suspicious_pairs.append((base, name))
    else:
      eval_name = f'{name}.eval'
      if eval_name in name_set:
        suspicious_pairs.append((name, eval_name))
  if suspicious_pairs:
    rendered = ', '.join(f'{left} vs {right}' for left, right in sorted(set(suspicious_pairs)))
    lines.append(f'Suspicious train/eval name pairs: {rendered}')

  lines.append('If this mismatch is intentional, rerun with --allow_eval_mismatch.')
  return '\n'.join(lines)


def _validate_eval_uid_sets(eval_uid_infos, allow_mismatch=False, verbose=True):
  if not eval_uid_infos:
    return set()
  uid_sets = [info['uids'] for info in eval_uid_infos.values()]
  if all(uid_set == uid_sets[0] for uid_set in uid_sets[1:]):
    return uid_sets[0]

  message = _build_eval_uid_set_mismatch_message(eval_uid_infos)
  if allow_mismatch:
    if verbose:
      print(f'WARNING: {message}')
    return set.intersection(*uid_sets)
  raise ValueError(message)


def _build_rows_for_uid(uid):
  S = _MP_BUILD
  model_names = S['model_names']
  candidate_model_names = S['candidate_model_names']
  score_model_names = S['score_model_names']
  all_eval_preds = S['all_eval_preds']
  all_eval_nbest_texts = S['all_eval_nbest_texts']
  all_logprobs = S['all_logprobs']
  blank_id = S['blank_id']
  beam_width = S['beam_width']
  nbest = S['nbest']
  id_to_char = S['id_to_char']
  char_to_id = S['char_to_id']
  lm = S['lm']
  aux_lms = S.get('aux_lms', {})
  no_lm_feats = S['no_lm_feats']
  gold = S['gold']
  meta = S['meta']
  normalize_ipa = S['normalize_ipa']
  prefix_beam_search_nbest = S['prefix_beam_search_nbest']
  ctc_force_score_batch = S['ctc_force_score_batch']
  feat_text = S['feat_text']
  feat_ipa = S['feat_ipa']
  feat_ctc_stats = S['feat_ctc_stats']
  feat_audio = S['feat_audio']
  feat_consensus = S['feat_consensus']
  feat_group_ext = S['feat_group_ext']
  feat_align = S['feat_align']
  feat_logprob_proxy = S['feat_logprob_proxy']
  feat_tdtctc_compare = S['feat_tdtctc_compare']
  feat_dual = S['feat_dual']
  feat_word = S['feat_word']
  feat_aux_meta = S['feat_aux_meta']
  feat_word_label = S['feat_word_label']
  tdt_eval_nbest = int(S.get('tdt_eval_nbest', 0) or 0)
  all_word_logprobs = S['all_word_logprobs']
  word_head_types = S['word_head_types']
  all_aux_meta = S['all_aux_meta']
  aux_meta_info = S['aux_meta_info']
  all_dual_head_preds = S['all_dual_head_preds']
  word_labels = S['word_labels']
  infer_mode = S.get('infer_mode', False)
  model_ctc_meta = S.get('model_ctc_meta', {})
  _word_id_to_char = S['_word_id_to_char']
  _word_blank_id = S['_word_blank_id']
  _ipa_convert = S['_ipa_convert']

  has_score_models = bool(score_model_names)
  feat_ctc_stats = bool(feat_ctc_stats and has_score_models)
  feat_align = bool(feat_align and has_score_models)
  feat_logprob_proxy = bool(feat_logprob_proxy and has_score_models)

  per_model_hyps = {}
  candidate_set = set()
  for mn in candidate_model_names:
    hyps = []
    if mn in all_logprobs and uid in all_logprobs[mn]:
      lp = all_logprobs[mn][uid].astype(np.float32)
      hyps = _decode_nbest_hyps_for_model(
          prefix_beam_search_nbest, lp, mn, blank_id, beam_width, nbest,
          id_to_char=id_to_char, normalize_ipa=normalize_ipa,
          model_ctc_meta=model_ctc_meta)
    per_model_hyps[mn] = hyps
    primary_text = _normalize_candidate_text(normalize_ipa, all_eval_preds.get(mn, {}).get(uid, ''))
    if primary_text:
      candidate_set.add(primary_text)
    eval_nbest_texts = all_eval_nbest_texts.get(mn, {}).get(uid, [])[:tdt_eval_nbest] if tdt_eval_nbest > 0 else []
    for text in eval_nbest_texts:
      if text:
        candidate_set.add(text)
    for _score, text in hyps:
      if text:
        candidate_set.add(text)
    dual_info = all_dual_head_preds.get(mn, {}).get(uid) if feat_dual else None
    if dual_info is not None:
      ctc_text = _normalize_candidate_text(normalize_ipa, dual_info.get('pred_ctc', ''))
      tdt_text = _normalize_candidate_text(normalize_ipa, dual_info.get('pred_tdt', ''))
      if ctc_text:
        candidate_set.add(ctc_text)
      if tdt_text:
        candidate_set.add(tdt_text)

  candidates = list(candidate_set)
  if not candidates:
    return []

  all_token_ids = [
      _encode_candidate_text_for_model(cand_text, score_model_names[0], char_to_id, model_ctc_meta)
      for cand_text in candidates
  ] if score_model_names else []

  # Filter score models to those that have logprobs for this uid
  uid_score_models = [mn for mn in score_model_names if uid in all_logprobs.get(mn, {})]
  has_uid_score_models = bool(uid_score_models)

  ctc_scores = {}
  n_frames_per_model = {}
  for mn in uid_score_models:
    lp = all_logprobs[mn][uid].astype(np.float32)
    n_frames_per_model[mn] = lp.shape[0]
    model_blank_id = _get_model_blank_id(mn, lp, blank_id, model_ctc_meta)
    token_ids = [
        _encode_candidate_text_for_model(cand_text, mn, char_to_id, model_ctc_meta)
        for cand_text in candidates
    ]
    ctc_scores[mn] = ctc_force_score_batch(lp, token_ids, blank=model_blank_id)
    if mn == score_model_names[0]:
      all_token_ids = token_ids

  if (feat_align or feat_logprob_proxy) and uid_score_models:
    ref_mn = uid_score_models[0]
    ref_lp = all_logprobs[ref_mn][uid].astype(np.float32)
    ref_blank_id = _get_model_blank_id(ref_mn, ref_lp, blank_id, model_ctc_meta)

  if feat_align and uid_score_models:
    align_results = []
    for tids in all_token_ids:
      align_results.append(_ctc_force_align(ref_lp, tids, blank=ref_blank_id))

  beam_ranks = {}
  best_per_model = {}
  for mn in model_names:
    hyps = per_model_hyps[mn]
    ranks = {}
    for rank, (_s, text) in enumerate(hyps):
      ranks[text] = rank
    eval_nbest_texts = all_eval_nbest_texts.get(mn, {}).get(uid, [])[:tdt_eval_nbest] if tdt_eval_nbest > 0 else []
    for rank, text in enumerate(eval_nbest_texts):
      if text and text not in ranks:
        ranks[text] = rank
    if not hyps:
      primary_text = _normalize_candidate_text(normalize_ipa, all_eval_preds.get(mn, {}).get(uid, ''))
      if primary_text:
        ranks[primary_text] = 0
    beam_ranks[mn] = ranks
    best_per_model[mn] = hyps[0][1] if hyps else _normalize_candidate_text(
        normalize_ipa, all_eval_preds.get(mn, {}).get(uid, ''))

  if not no_lm_feats:
    lm_scores = []
    if lm is not None:
      for cand_text in candidates:
        lm_scores.append(_score_lm_candidate(lm, cand_text))
    else:
      lm_scores = [0.0] * len(candidates)
    aux_lm_scores = {
        lm_name: [_score_lm_candidate(extra_lm, cand_text) for cand_text in candidates]
        for lm_name, extra_lm in aux_lms.items()
    }

  word_ctc_scores = {}
  aux_greedy_texts = {}
  if feat_word:
    for mn in all_word_logprobs:
      if uid not in all_word_logprobs[mn]:
        continue
      ht = word_head_types[mn]
      aux_lp = all_word_logprobs[mn][uid].astype(np.float32)
      if ht == 'pseudo_ipa':
        word_ctc_scores[mn] = ctc_force_score_batch(aux_lp, all_token_ids, blank=blank_id)
      elif ht in ('word_ctc', 'word_ctc_bpe'):
        if ht == 'word_ctc' and _word_id_to_char is not None:
          aux_greedy_texts[mn] = _greedy_ctc_decode_numpy(
              aux_lp, blank_id=_word_blank_id, id_to_char=_word_id_to_char)

  n_frames = max(n_frames_per_model.values()) if n_frames_per_model else np.nan

  if feat_logprob_proxy and uid_score_models:
    _lp_probs = np.exp(ref_lp)
    _frame_entropy = -np.sum(_lp_probs * ref_lp, axis=1)
    _utt_entropy_mean = float(np.mean(_frame_entropy))
    _utt_entropy_std = float(np.std(_frame_entropy))
    _utt_entropy_max = float(np.max(_frame_entropy))
    _utt_blank_prob_mean = float(np.mean(_lp_probs[:, blank_id]))
    _utt_top1_prob_mean = float(np.mean(np.max(_lp_probs, axis=1)))
    del _lp_probs

  gold_text = gold.get(uid, '')
  item_meta = meta.get(uid, {})
  source = item_meta.get('source', '')
  child_id = item_meta.get('child_id', '')
  age_bucket = item_meta.get('age_bucket', '')
  raw_audio_duration_sec = _safe_float(item_meta.get('audio_duration_sec', np.nan))
  word_text = word_labels.get(uid, '').strip()
  word_text_norm = word_text.lower().strip()
  word_ipa = ''
  if feat_word_label and word_text_norm and _ipa_convert is not None:
    try:
      word_ipa = _ipa_convert(word_text_norm).replace('*', '').strip()
    except Exception:
      word_ipa = ''

  tdtctc_row_base = {}
  if feat_tdtctc_compare:
    ctc_len_values = []
    tdt_len_values = []
    ctc_space_values = []
    tdt_space_values = []
    len_gap_values = []
    len_gap_abs_values = []
    space_gap_values = []
    space_gap_abs_values = []
    edit_dist_values = []
    edit_dist_norm_values = []
    same_text_values = []
    same_len_values = []
    same_space_values = []
    tdt_longer_values = []
    tdt_more_space_values = []

    for mn in model_names:
      dual_info = all_dual_head_preds.get(mn, {}).get(uid)
      prefix = f'tdtctc_'
      if dual_info is None:
        tdtctc_row_base[f'{prefix}has_dual_{mn}'] = 0
        tdtctc_row_base[f'{prefix}ctc_len_{mn}'] = -1
        tdtctc_row_base[f'{prefix}tdt_len_{mn}'] = -1
        tdtctc_row_base[f'{prefix}ctc_spaces_{mn}'] = -1
        tdtctc_row_base[f'{prefix}tdt_spaces_{mn}'] = -1
        tdtctc_row_base[f'{prefix}len_gap_{mn}'] = 0
        tdtctc_row_base[f'{prefix}len_gap_abs_{mn}'] = -1
        tdtctc_row_base[f'{prefix}spaces_gap_{mn}'] = 0
        tdtctc_row_base[f'{prefix}spaces_gap_abs_{mn}'] = -1
        tdtctc_row_base[f'{prefix}edit_dist_{mn}'] = -1
        tdtctc_row_base[f'{prefix}edit_dist_norm_{mn}'] = -1.0
        tdtctc_row_base[f'{prefix}same_text_{mn}'] = 0
        tdtctc_row_base[f'{prefix}same_len_{mn}'] = 0
        tdtctc_row_base[f'{prefix}same_spaces_{mn}'] = 0
        tdtctc_row_base[f'{prefix}tdt_longer_{mn}'] = 0
        tdtctc_row_base[f'{prefix}tdt_more_spaces_{mn}'] = 0
        continue

      pred_ctc = _normalize_candidate_text(normalize_ipa, dual_info.get('pred_ctc', ''))
      pred_tdt = _normalize_candidate_text(normalize_ipa, dual_info.get('pred_tdt', dual_info.get('pred_primary', '')))
      ctc_len = len(pred_ctc)
      tdt_len = len(pred_tdt)
      ctc_spaces = pred_ctc.count(' ')
      tdt_spaces = pred_tdt.count(' ')
      len_gap = tdt_len - ctc_len
      spaces_gap = tdt_spaces - ctc_spaces
      edit_dist = editdistance.eval(pred_tdt, pred_ctc)
      edit_dist_norm = edit_dist / max(max(ctc_len, tdt_len), 1)
      same_text = int(pred_tdt == pred_ctc)
      same_len = int(tdt_len == ctc_len)
      same_spaces = int(tdt_spaces == ctc_spaces)
      tdt_longer = int(tdt_len > ctc_len)
      tdt_more_spaces = int(tdt_spaces > ctc_spaces)

      tdtctc_row_base[f'{prefix}has_dual_{mn}'] = 1
      tdtctc_row_base[f'{prefix}ctc_len_{mn}'] = ctc_len
      tdtctc_row_base[f'{prefix}tdt_len_{mn}'] = tdt_len
      tdtctc_row_base[f'{prefix}ctc_spaces_{mn}'] = ctc_spaces
      tdtctc_row_base[f'{prefix}tdt_spaces_{mn}'] = tdt_spaces
      tdtctc_row_base[f'{prefix}len_gap_{mn}'] = len_gap
      tdtctc_row_base[f'{prefix}len_gap_abs_{mn}'] = abs(len_gap)
      tdtctc_row_base[f'{prefix}spaces_gap_{mn}'] = spaces_gap
      tdtctc_row_base[f'{prefix}spaces_gap_abs_{mn}'] = abs(spaces_gap)
      tdtctc_row_base[f'{prefix}edit_dist_{mn}'] = edit_dist
      tdtctc_row_base[f'{prefix}edit_dist_norm_{mn}'] = edit_dist_norm
      tdtctc_row_base[f'{prefix}same_text_{mn}'] = same_text
      tdtctc_row_base[f'{prefix}same_len_{mn}'] = same_len
      tdtctc_row_base[f'{prefix}same_spaces_{mn}'] = same_spaces
      tdtctc_row_base[f'{prefix}tdt_longer_{mn}'] = tdt_longer
      tdtctc_row_base[f'{prefix}tdt_more_spaces_{mn}'] = tdt_more_spaces

      ctc_len_values.append(ctc_len)
      tdt_len_values.append(tdt_len)
      ctc_space_values.append(ctc_spaces)
      tdt_space_values.append(tdt_spaces)
      len_gap_values.append(len_gap)
      len_gap_abs_values.append(abs(len_gap))
      space_gap_values.append(spaces_gap)
      space_gap_abs_values.append(abs(spaces_gap))
      edit_dist_values.append(edit_dist)
      edit_dist_norm_values.append(edit_dist_norm)
      same_text_values.append(same_text)
      same_len_values.append(same_len)
      same_space_values.append(same_spaces)
      tdt_longer_values.append(tdt_longer)
      tdt_more_space_values.append(tdt_more_spaces)

    n_dual_models = len(ctc_len_values)
    tdtctc_row_base['n_tdtctc_models'] = n_dual_models
    if n_dual_models:
      tdtctc_row_base['tdtctc_ctc_len_mean'] = float(np.mean(ctc_len_values))
      tdtctc_row_base['tdtctc_tdt_len_mean'] = float(np.mean(tdt_len_values))
      tdtctc_row_base['tdtctc_ctc_spaces_mean'] = float(np.mean(ctc_space_values))
      tdtctc_row_base['tdtctc_tdt_spaces_mean'] = float(np.mean(tdt_space_values))
      tdtctc_row_base['tdtctc_len_gap_mean'] = float(np.mean(len_gap_values))
      tdtctc_row_base['tdtctc_len_gap_abs_mean'] = float(np.mean(len_gap_abs_values))
      tdtctc_row_base['tdtctc_len_gap_abs_max'] = float(np.max(len_gap_abs_values))
      tdtctc_row_base['tdtctc_spaces_gap_mean'] = float(np.mean(space_gap_values))
      tdtctc_row_base['tdtctc_spaces_gap_abs_mean'] = float(np.mean(space_gap_abs_values))
      tdtctc_row_base['tdtctc_spaces_gap_abs_max'] = float(np.max(space_gap_abs_values))
      tdtctc_row_base['tdtctc_edit_dist_mean'] = float(np.mean(edit_dist_values))
      tdtctc_row_base['tdtctc_edit_dist_norm_mean'] = float(np.mean(edit_dist_norm_values))
      tdtctc_row_base['tdtctc_same_text_count'] = int(np.sum(same_text_values))
      tdtctc_row_base['tdtctc_same_text_frac'] = float(np.mean(same_text_values))
      tdtctc_row_base['tdtctc_same_len_count'] = int(np.sum(same_len_values))
      tdtctc_row_base['tdtctc_same_len_frac'] = float(np.mean(same_len_values))
      tdtctc_row_base['tdtctc_same_spaces_count'] = int(np.sum(same_space_values))
      tdtctc_row_base['tdtctc_same_spaces_frac'] = float(np.mean(same_space_values))
      tdtctc_row_base['tdtctc_tdt_longer_count'] = int(np.sum(tdt_longer_values))
      tdtctc_row_base['tdtctc_tdt_longer_frac'] = float(np.mean(tdt_longer_values))
      tdtctc_row_base['tdtctc_tdt_more_spaces_count'] = int(np.sum(tdt_more_space_values))
      tdtctc_row_base['tdtctc_tdt_more_spaces_frac'] = float(np.mean(tdt_more_space_values))
    else:
      tdtctc_row_base['tdtctc_ctc_len_mean'] = -1.0
      tdtctc_row_base['tdtctc_tdt_len_mean'] = -1.0
      tdtctc_row_base['tdtctc_ctc_spaces_mean'] = -1.0
      tdtctc_row_base['tdtctc_tdt_spaces_mean'] = -1.0
      tdtctc_row_base['tdtctc_len_gap_mean'] = 0.0
      tdtctc_row_base['tdtctc_len_gap_abs_mean'] = -1.0
      tdtctc_row_base['tdtctc_len_gap_abs_max'] = -1.0
      tdtctc_row_base['tdtctc_spaces_gap_mean'] = 0.0
      tdtctc_row_base['tdtctc_spaces_gap_abs_mean'] = -1.0
      tdtctc_row_base['tdtctc_spaces_gap_abs_max'] = -1.0
      tdtctc_row_base['tdtctc_edit_dist_mean'] = -1.0
      tdtctc_row_base['tdtctc_edit_dist_norm_mean'] = -1.0
      tdtctc_row_base['tdtctc_same_text_count'] = 0
      tdtctc_row_base['tdtctc_same_text_frac'] = 0.0
      tdtctc_row_base['tdtctc_same_len_count'] = 0
      tdtctc_row_base['tdtctc_same_len_frac'] = 0.0
      tdtctc_row_base['tdtctc_same_spaces_count'] = 0
      tdtctc_row_base['tdtctc_same_spaces_frac'] = 0.0
      tdtctc_row_base['tdtctc_tdt_longer_count'] = 0
      tdtctc_row_base['tdtctc_tdt_longer_frac'] = 0.0
      tdtctc_row_base['tdtctc_tdt_more_spaces_count'] = 0
      tdtctc_row_base['tdtctc_tdt_more_spaces_frac'] = 0.0

  import editdistance
  rows = []
  for ci, cand_text in enumerate(candidates):
    row = {
        'uid': uid,
        'candidate_text': cand_text,
        'source': source,
        'child_id': child_id,
        'age_bucket': age_bucket,
    }
    if feat_tdtctc_compare:
      row.update(tdtctc_row_base)

    scores_arr = np.array([ctc_scores[mn][ci] for mn in uid_score_models], dtype=np.float32)
    if has_uid_score_models:
      row['ctc_score_mean'] = float(np.mean(scores_arr))
      row['ctc_score_std'] = float(np.std(scores_arr))
      row['ctc_score_min'] = float(np.min(scores_arr))
      row['ctc_score_max'] = float(np.max(scores_arr))
      row['ctc_score_range'] = float(np.max(scores_arr) - np.min(scores_arr))
      row['n_score_models'] = len(uid_score_models)

      for mn in model_names:
        if mn in ctc_scores:
          row[f'ctc_score_{mn}'] = ctc_scores[mn][ci]
        else:
          row[f'ctc_score_{mn}'] = np.nan

    text_len = len(cand_text)
    n_spaces = cand_text.count(' ')
    row['text_len'] = text_len
    row['n_frames'] = n_frames
    row['char_per_frame'] = (text_len / n_frames) if np.isfinite(n_frames) and n_frames > 0 else np.nan

    n_models_has = 0
    for mn in model_names:
      rank = beam_ranks[mn].get(cand_text, -1)
      row[f'beam_rank_{mn}'] = rank
      if rank >= 0:
        n_models_has += 1
    row['n_models_has'] = n_models_has

    _ed_best_fn = _EDIT_DIST_TO_BEST_FN
    for mn in model_names:
      best_text = best_per_model[mn]
      row[f'edit_dist_to_best_{mn}'] = (
          _ed_best_fn(cand_text, best_text) if callable(_ed_best_fn)
          else editdistance.eval(cand_text, best_text)
      )

    if feat_dual or feat_tdtctc_compare:
      dual_ctc_hits = 0
      dual_tdt_hits = 0
      dual_primary_hits = 0
      cand_ctc_len_diff_values = []
      cand_ctc_space_diff_values = []
      cand_ctc_edit_dist_values = []
      cand_ctc_edit_dist_norm_values = []
      cand_tdt_len_diff_values = []
      cand_tdt_space_diff_values = []
      cand_tdt_edit_dist_values = []
      cand_tdt_edit_dist_norm_values = []
      for mn in model_names:
        dual_info = all_dual_head_preds.get(mn, {}).get(uid)
        if dual_info is None:
          if feat_dual:
            row[f'is_dual_ctc_{mn}'] = 0
            row[f'is_dual_tdt_{mn}'] = 0
            row[f'is_dual_primary_{mn}'] = 0
            row[f'dual_heads_agree_{mn}'] = 0
            row[f'dual_len_gap_{mn}'] = -1
          if feat_tdtctc_compare:
            row[f'tdtctc_cand_ctc_len_diff_{mn}'] = -1
            row[f'tdtctc_cand_ctc_space_diff_{mn}'] = -1
            row[f'tdtctc_cand_ctc_edit_dist_{mn}'] = -1
            row[f'tdtctc_cand_ctc_edit_dist_norm_{mn}'] = -1.0
            row[f'tdtctc_cand_tdt_len_diff_{mn}'] = -1
            row[f'tdtctc_cand_tdt_space_diff_{mn}'] = -1
            row[f'tdtctc_cand_tdt_edit_dist_{mn}'] = -1
            row[f'tdtctc_cand_tdt_edit_dist_norm_{mn}'] = -1.0
          continue
        pred_ctc = _normalize_candidate_text(normalize_ipa, dual_info.get('pred_ctc', ''))
        pred_tdt = _normalize_candidate_text(normalize_ipa, dual_info.get('pred_tdt', dual_info.get('pred_primary', '')))
        pred_primary = _normalize_candidate_text(normalize_ipa, dual_info.get('pred_primary', ''))
        is_dual_ctc = int(cand_text == pred_ctc)
        is_dual_tdt = int(cand_text == pred_tdt)
        is_dual_primary = int(cand_text == pred_primary)
        if feat_dual:
          row[f'is_dual_ctc_{mn}'] = is_dual_ctc
          row[f'is_dual_tdt_{mn}'] = is_dual_tdt
          row[f'is_dual_primary_{mn}'] = is_dual_primary
          row[f'dual_heads_agree_{mn}'] = int(dual_info.get('pred_heads_agree', 0))
          row[f'dual_len_gap_{mn}'] = int(dual_info.get('pred_dual_len_gap', -1))
        if feat_tdtctc_compare:
          cand_ctc_len_diff = abs(text_len - len(pred_ctc))
          cand_ctc_space_diff = abs(n_spaces - pred_ctc.count(' '))
          cand_ctc_edit_dist = editdistance.eval(cand_text, pred_ctc)
          cand_ctc_edit_dist_norm = cand_ctc_edit_dist / max(max(text_len, len(pred_ctc)), 1)
          cand_tdt_len_diff = abs(text_len - len(pred_tdt))
          cand_tdt_space_diff = abs(n_spaces - pred_tdt.count(' '))
          cand_tdt_edit_dist = editdistance.eval(cand_text, pred_tdt)
          cand_tdt_edit_dist_norm = cand_tdt_edit_dist / max(max(text_len, len(pred_tdt)), 1)

          row[f'tdtctc_cand_ctc_len_diff_{mn}'] = cand_ctc_len_diff
          row[f'tdtctc_cand_ctc_space_diff_{mn}'] = cand_ctc_space_diff
          row[f'tdtctc_cand_ctc_edit_dist_{mn}'] = cand_ctc_edit_dist
          row[f'tdtctc_cand_ctc_edit_dist_norm_{mn}'] = cand_ctc_edit_dist_norm
          row[f'tdtctc_cand_tdt_len_diff_{mn}'] = cand_tdt_len_diff
          row[f'tdtctc_cand_tdt_space_diff_{mn}'] = cand_tdt_space_diff
          row[f'tdtctc_cand_tdt_edit_dist_{mn}'] = cand_tdt_edit_dist
          row[f'tdtctc_cand_tdt_edit_dist_norm_{mn}'] = cand_tdt_edit_dist_norm

          cand_ctc_len_diff_values.append(cand_ctc_len_diff)
          cand_ctc_space_diff_values.append(cand_ctc_space_diff)
          cand_ctc_edit_dist_values.append(cand_ctc_edit_dist)
          cand_ctc_edit_dist_norm_values.append(cand_ctc_edit_dist_norm)
          cand_tdt_len_diff_values.append(cand_tdt_len_diff)
          cand_tdt_space_diff_values.append(cand_tdt_space_diff)
          cand_tdt_edit_dist_values.append(cand_tdt_edit_dist)
          cand_tdt_edit_dist_norm_values.append(cand_tdt_edit_dist_norm)
        if feat_dual:
          dual_ctc_hits += is_dual_ctc
          dual_tdt_hits += is_dual_tdt
          dual_primary_hits += is_dual_primary
      if feat_dual:
        row['n_dual_ctc_hits'] = dual_ctc_hits
        row['n_dual_tdt_hits'] = dual_tdt_hits
        row['n_dual_primary_hits'] = dual_primary_hits
      if feat_tdtctc_compare:
        if cand_ctc_len_diff_values:
          row['tdtctc_cand_ctc_len_diff_mean'] = float(np.mean(cand_ctc_len_diff_values))
          row['tdtctc_cand_ctc_len_diff_min'] = float(np.min(cand_ctc_len_diff_values))
          row['tdtctc_cand_ctc_space_diff_mean'] = float(np.mean(cand_ctc_space_diff_values))
          row['tdtctc_cand_ctc_space_diff_min'] = float(np.min(cand_ctc_space_diff_values))
          row['tdtctc_cand_ctc_edit_dist_mean'] = float(np.mean(cand_ctc_edit_dist_values))
          row['tdtctc_cand_ctc_edit_dist_min'] = float(np.min(cand_ctc_edit_dist_values))
          row['tdtctc_cand_ctc_edit_dist_norm_mean'] = float(np.mean(cand_ctc_edit_dist_norm_values))
          row['tdtctc_cand_ctc_edit_dist_norm_min'] = float(np.min(cand_ctc_edit_dist_norm_values))
        else:
          row['tdtctc_cand_ctc_len_diff_mean'] = -1.0
          row['tdtctc_cand_ctc_len_diff_min'] = -1.0
          row['tdtctc_cand_ctc_space_diff_mean'] = -1.0
          row['tdtctc_cand_ctc_space_diff_min'] = -1.0
          row['tdtctc_cand_ctc_edit_dist_mean'] = -1.0
          row['tdtctc_cand_ctc_edit_dist_min'] = -1.0
          row['tdtctc_cand_ctc_edit_dist_norm_mean'] = -1.0
          row['tdtctc_cand_ctc_edit_dist_norm_min'] = -1.0
        if cand_tdt_len_diff_values:
          row['tdtctc_cand_tdt_len_diff_mean'] = float(np.mean(cand_tdt_len_diff_values))
          row['tdtctc_cand_tdt_len_diff_min'] = float(np.min(cand_tdt_len_diff_values))
          row['tdtctc_cand_tdt_space_diff_mean'] = float(np.mean(cand_tdt_space_diff_values))
          row['tdtctc_cand_tdt_space_diff_min'] = float(np.min(cand_tdt_space_diff_values))
          row['tdtctc_cand_tdt_edit_dist_mean'] = float(np.mean(cand_tdt_edit_dist_values))
          row['tdtctc_cand_tdt_edit_dist_min'] = float(np.min(cand_tdt_edit_dist_values))
          row['tdtctc_cand_tdt_edit_dist_norm_mean'] = float(np.mean(cand_tdt_edit_dist_norm_values))
          row['tdtctc_cand_tdt_edit_dist_norm_min'] = float(np.min(cand_tdt_edit_dist_norm_values))
        else:
          row['tdtctc_cand_tdt_len_diff_mean'] = -1.0
          row['tdtctc_cand_tdt_len_diff_min'] = -1.0
          row['tdtctc_cand_tdt_space_diff_mean'] = -1.0
          row['tdtctc_cand_tdt_space_diff_min'] = -1.0
          row['tdtctc_cand_tdt_edit_dist_mean'] = -1.0
          row['tdtctc_cand_tdt_edit_dist_min'] = -1.0
          row['tdtctc_cand_tdt_edit_dist_norm_mean'] = -1.0
          row['tdtctc_cand_tdt_edit_dist_norm_min'] = -1.0

    if not no_lm_feats:
      row['lm_score'] = lm_scores[ci]
      row['lm_score_per_char'] = lm_scores[ci] / max(text_len, 1)
      row['lm_score_per_word'] = lm_scores[ci] / max(n_spaces + 1, 1)
      for lm_name, scores in aux_lm_scores.items():
        prefix = f'{lm_name}_lm_score'
        row[prefix] = scores[ci]
        row[f'{prefix}_per_char'] = scores[ci] / max(text_len, 1)
        row[f'{prefix}_per_word'] = scores[ci] / max(n_spaces + 1, 1)

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
        ch_counts = Counter(cand_text)
        probs = np.array(list(ch_counts.values()), dtype=np.float64) / text_len
        row['char_entropy'] = float(-np.sum(probs * np.log(probs + 1e-12)))
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

    if feat_ipa:
      ipa_vowels = set('eiouɑæɐɔəɚɛɪʊʌ')
      ipa_consonants = set('bcdfghjklmnprstvwxzçðŋɟɫɬɹɾʁʃʒʔʝθχʧʤ')
      n_vowels = sum(1 for ch in cand_text if ch in ipa_vowels)
      n_consonants = sum(1 for ch in cand_text if ch in ipa_consonants)
      row['n_vowels'] = n_vowels
      row['n_consonants'] = n_consonants
      row['vowel_ratio'] = n_vowels / max(text_len, 1)
      row['consonant_ratio'] = n_consonants / max(text_len, 1)
      row['vc_ratio'] = n_vowels / max(n_consonants, 1)
      row['n_length_marks'] = cand_text.count('ː')

    if feat_ctc_stats and has_uid_score_models:
      row['ctc_score_median'] = float(np.median(scores_arr))
      abs_mean = abs(float(np.mean(scores_arr)))
      row['ctc_score_cv'] = float(np.std(scores_arr)) / max(abs_mean, 1e-8)
      if len(scores_arr) >= 3:
        from scipy.stats import kurtosis, skew
        row['ctc_score_skew'] = float(skew(scores_arr))
        row['ctc_score_kurtosis'] = float(kurtosis(scores_arr))
      else:
        row['ctc_score_skew'] = 0.0
        row['ctc_score_kurtosis'] = 0.0
      q75, q25 = np.percentile(scores_arr, [75, 25])
      row['ctc_score_iqr'] = float(q75 - q25)

    if feat_align and uid_score_models:
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

    if feat_logprob_proxy and uid_score_models:
      row['entropy_mean'] = _utt_entropy_mean
      row['entropy_std'] = _utt_entropy_std
      row['entropy_max'] = _utt_entropy_max
      row['blank_prob_mean'] = _utt_blank_prob_mean
      row['top1_prob_mean'] = _utt_top1_prob_mean
      row['model_ctc_std'] = float(np.std(scores_arr))
      ranks_for_ent = []
      for mn in uid_score_models:
        mn_scores = ctc_scores[mn]
        ranks = sorted(range(len(mn_scores)), key=lambda x: mn_scores[x], reverse=True)
        ranks_for_ent.append(ranks.index(ci))
      ranks_arr = np.array(ranks_for_ent, dtype=np.float64)
      row['model_rank_std'] = float(np.std(ranks_arr))
      model_best_idx = int(np.argmax(scores_arr))
      model_best_name = uid_score_models[model_best_idx]
      for mi, mn in enumerate(model_names):
        row[f'is_best_model_{mn}'] = 1 if mn == model_best_name else 0

    if feat_audio:
      duration_sec = n_frames * 0.04 if np.isfinite(n_frames) and n_frames > 0 else np.nan
      row['duration_sec'] = duration_sec
      row['chars_per_sec'] = text_len / max(duration_sec, 0.01) if np.isfinite(duration_sec) else np.nan
      row['words_per_sec'] = (n_spaces + 1) / max(duration_sec, 0.01) if np.isfinite(duration_sec) else np.nan
      row['audio_duration_sec'] = raw_audio_duration_sec
      row['has_audio_duration_sec'] = int(np.isfinite(raw_audio_duration_sec) and raw_audio_duration_sec > 0)
      if row['has_audio_duration_sec']:
        row['chars_per_audio_sec'] = text_len / max(raw_audio_duration_sec, 0.01)
        row['words_per_audio_sec'] = (n_spaces + 1) / max(raw_audio_duration_sec, 0.01)
        row['audio_minus_frame_duration_sec'] = raw_audio_duration_sec - duration_sec if np.isfinite(duration_sec) else np.nan
        row['audio_to_frame_duration_ratio'] = raw_audio_duration_sec / max(duration_sec, 0.01) if np.isfinite(duration_sec) else np.nan
      else:
        row['chars_per_audio_sec'] = np.nan
        row['words_per_audio_sec'] = np.nan
        row['audio_minus_frame_duration_sec'] = np.nan
        row['audio_to_frame_duration_ratio'] = np.nan

    if feat_word:
      word_ipa_scores = []
      for mn in word_ctc_scores:
        sc = word_ctc_scores[mn][ci]
        row[f'word_ctc_score_{mn}'] = sc
        word_ipa_scores.append(sc)
        if mn in ctc_scores:
          row[f'word_score_diff_{mn}'] = ctc_scores[mn][ci] - sc
      if word_ipa_scores:
        row['word_ctc_score_mean'] = float(np.mean(word_ipa_scores))
        row['word_ctc_score_std'] = float(np.std(word_ipa_scores))
        if 'ctc_score_mean' in row:
          row['word_primary_diff_mean'] = row['ctc_score_mean'] - row['word_ctc_score_mean']

      for mn in aux_greedy_texts:
        aux_word_text = aux_greedy_texts[mn]
        aux_word_text_norm = aux_word_text.strip()
        row[f'word_head_len_{mn}'] = len(aux_word_text_norm)
        row[f'word_head_spaces_{mn}'] = aux_word_text_norm.count(' ')
        row[f'word_head_words_{mn}'] = (aux_word_text_norm.count(' ') + 1) if aux_word_text_norm else 0
        row[f'word_head_len_diff_{mn}'] = abs(text_len - len(aux_word_text_norm))
        row[f'word_edit_dist_raw_{mn}'] = editdistance.eval(cand_text, aux_word_text)
        row[f'word_edit_dist_norm_{mn}'] = editdistance.eval(cand_text, aux_word_text) / max(text_len, 1)
        if _ipa_convert is not None and aux_word_text.strip():
          pseudo_ipa = _ipa_convert(aux_word_text).replace('*', '')
          row[f'word_head_ipa_len_{mn}'] = len(pseudo_ipa)
          row[f'word_head_ipa_spaces_{mn}'] = pseudo_ipa.count(' ')
          row[f'word_head_ipa_len_diff_{mn}'] = abs(text_len - len(pseudo_ipa))
          row[f'word_edit_dist_ipa_{mn}'] = editdistance.eval(cand_text, pseudo_ipa)
          row[f'word_edit_dist_ipa_norm_{mn}'] = editdistance.eval(cand_text, pseudo_ipa) / max(text_len, 1)
        else:
          row[f'word_head_ipa_len_{mn}'] = -1
          row[f'word_head_ipa_spaces_{mn}'] = -1
          row[f'word_head_ipa_len_diff_{mn}'] = -1

    if feat_aux_meta and all_aux_meta:
      age_scores = []
      domain_probs = []
      nchars_preds = []
      nspaces_preds = []
      ordinal_age_probs = {
          '5plus': [],
          '8plus': [],
          '12plus': [],
      }
      class_age_probs = {}
      for mn in all_aux_meta:
        if uid not in all_aux_meta[mn]:
          continue
        pred = all_aux_meta[mn][uid]
        mi = aux_meta_info[mn]
        if mi['has_age'] and 'age_logits' in pred:
          age_logits = pred['age_logits']
          if mi['age_mode'] == 'ordinal':
            age_probs = 1.0 / (1.0 + np.exp(-age_logits))
            row[f'aux_age_prob_5plus_{mn}'] = float(age_probs[0])
            row[f'aux_age_prob_8plus_{mn}'] = float(age_probs[1])
            row[f'aux_age_prob_12plus_{mn}'] = float(age_probs[2])
            ordinal_age_probs['5plus'].append(float(age_probs[0]))
            ordinal_age_probs['8plus'].append(float(age_probs[1]))
            ordinal_age_probs['12plus'].append(float(age_probs[2]))
            age_score = float(np.mean(age_probs))
          elif mi['age_mode'] == 'classify':
            exp_l = np.exp(age_logits - np.max(age_logits))
            probs = exp_l / exp_l.sum()
            age_score = float(np.dot(probs, np.arange(len(probs))))
            for k in range(len(probs)):
              row[f'aux_age_class{k}_prob_{mn}'] = float(probs[k])
              class_age_probs.setdefault(k, []).append(float(probs[k]))
          else:
            age_score = float(age_logits)
          row[f'aux_age_score_{mn}'] = age_score
          age_scores.append(age_score)
        if mi['has_domain'] and 'domain_logit' in pred:
          domain_logit = pred['domain_logit']
          domain_prob = 1.0 / (1.0 + np.exp(-domain_logit))
          row[f'aux_domain_prob_dd_{mn}'] = float(domain_prob)
          domain_probs.append(float(domain_prob))
        if mi.get('has_nchars') and 'nchars_pred' in pred:
          nchars_log = float(pred['nchars_pred'])
          nchars_hat = float(np.expm1(nchars_log))
          row[f'aux_nchars_log_{mn}'] = nchars_log
          row[f'aux_nchars_pred_{mn}'] = nchars_hat
          row[f'aux_nchars_diff_{mn}'] = abs(text_len - nchars_hat)
          nchars_preds.append(nchars_hat)
        if mi.get('has_nspaces') and 'nspaces_pred' in pred:
          nspaces_log = float(pred['nspaces_pred'])
          nspaces_hat = float(np.expm1(nspaces_log))
          row[f'aux_nspaces_log_{mn}'] = nspaces_log
          row[f'aux_nspaces_pred_{mn}'] = nspaces_hat
          row[f'aux_nspaces_diff_{mn}'] = abs(n_spaces - nspaces_hat)
          nspaces_preds.append(nspaces_hat)
      if age_scores:
        row['aux_age_score_mean'] = float(np.mean(age_scores))
      for suffix, values in ordinal_age_probs.items():
        if values:
          row[f'aux_age_prob_{suffix}_mean'] = float(np.mean(values))
      for k, values in sorted(class_age_probs.items()):
        if values:
          row[f'aux_age_class{k}_prob_mean'] = float(np.mean(values))
      if domain_probs:
        row['aux_domain_prob_dd_mean'] = float(np.mean(domain_probs))
      if nchars_preds:
        row['aux_nchars_pred_mean'] = float(np.mean(nchars_preds))
        row['aux_nchars_diff_mean'] = abs(text_len - row['aux_nchars_pred_mean'])
      if nspaces_preds:
        row['aux_nspaces_pred_mean'] = float(np.mean(nspaces_preds))
        row['aux_nspaces_diff_mean'] = abs(n_spaces - row['aux_nspaces_pred_mean'])

    if feat_word_label:
      row['has_word_label'] = int(bool(word_text_norm))
      row['word_label_text'] = word_text
      row['word_label_ipa'] = word_ipa
      row['word_label_len'] = len(word_text_norm)
      row['word_label_spaces'] = word_text_norm.count(' ')
      row['word_label_words'] = row['word_label_spaces'] + 1 if word_text_norm else 0
      if word_text_norm:
        row['word_label_len_diff'] = abs(text_len - len(word_text_norm))
      if word_ipa:
        row['word_label_ipa_len'] = len(word_ipa)
        row['word_label_ipa_spaces'] = word_ipa.count(' ')
        row['word_label_ipa_len_diff'] = abs(text_len - len(word_ipa))
        row['word_label_ipa_edit_dist'] = editdistance.eval(cand_text, word_ipa)
        row['word_label_ipa_edit_dist_norm'] = row['word_label_ipa_edit_dist'] / max(text_len, 1)
        row['word_label_ipa_space_diff'] = abs(n_spaces - word_ipa.count(' '))
        row['word_label_ipa_same_len'] = int(text_len == len(word_ipa))
        row['word_label_ipa_same_spaces'] = int(n_spaces == word_ipa.count(' '))
        row['word_label_ipa_exact_match'] = int(cand_text == word_ipa)
      else:
        row['word_label_ipa_len'] = -1
        row['word_label_ipa_spaces'] = -1
        row['word_label_ipa_len_diff'] = -1
        row['word_label_ipa_edit_dist'] = -1
        row['word_label_ipa_edit_dist_norm'] = -1.0
        row['word_label_ipa_space_diff'] = -1
        row['word_label_ipa_same_len'] = 0
        row['word_label_ipa_same_spaces'] = 0
        row['word_label_ipa_exact_match'] = 0

    valid_ranks = [beam_ranks[mn].get(cand_text, -1) for mn in model_names]
    valid_ranks_pos = [r for r in valid_ranks if r >= 0]
    row['beam_rank_mean'] = np.mean(valid_ranks_pos) if valid_ranks_pos else nbest
    row['beam_rank_min'] = min(valid_ranks_pos) if valid_ranks_pos else nbest
    row['beam_rank_max'] = max(valid_ranks_pos) if valid_ranks_pos else nbest
    if feat_consensus:
      row['n_models_in_top3'] = sum(1 for r in valid_ranks if 0 <= r < 3)

    if feat_consensus:
      _pw_ed_fn = _PAIRWISE_EDIT_DIST_FN
      _pw_norm_fn = _PAIRWISE_EDIT_DIST_NORM
      pairwise_sum = 0
      n_pairs = 0
      for oi, other_text in enumerate(candidates):
        if oi != ci:
          pairwise_sum += (
              _pw_ed_fn(cand_text, other_text) if callable(_pw_ed_fn)
              else editdistance.eval(cand_text, other_text)
          )
          n_pairs += 1
      row['mean_pairwise_edit_dist'] = pairwise_sum / max(n_pairs, 1)
      if callable(_pw_norm_fn):
        mean_cer_to_others = _pw_norm_fn(pairwise_sum, n_pairs, cand_text) if n_pairs > 0 else 0.0
      else:
        mean_cer_to_others = 0.0
        if n_pairs > 0 and text_len > 0:
          mean_cer_to_others = pairwise_sum / (n_pairs * text_len)
      row['consensus_score'] = 1.0 - min(mean_cer_to_others, 1.0)
      row['n_exact_best'] = sum(1 for mn in model_names if best_per_model[mn] == cand_text)

    row['target_cer'] = np.nan if infer_mode else _get_target_error(normalize_ipa, gold_text, cand_text)
    rows.append(row)

  return rows


def build_reranker_dataset(model_names, get_model_dir, get_eval_csv,
                           prefix_beam_search_nbest, ctc_force_score_batch,
                           normalize_ipa, id_to_char, blank_id,
                           nbest=10, beam_width=10,
                           lm=None, aux_lms=None, verbose=True,
                           feat_text=False, feat_ipa=False,
                           feat_ctc_stats=False, feat_audio=False,
                           feat_consensus=False, feat_mbr=False,
                           feat_group_ext=False,
                           feat_align=False, feat_logprob_proxy=False,
                           feat_tdtctc_compare=False,
                           feat_dual=False,
                           feat_word=False, feat_aux=None, feat_aux_meta=False,
                           feat_word_label=False,
                           word_label_file='', word_label_col='',
                           tdt_eval_nbest=0,
                           no_lm_feats=False, n_workers=0,
                           require_gold=True,
                           allow_eval_set_mismatch=False,
                           model_max_dur=None):
  import torch

  feat_word = feat_word or bool(feat_aux)

  all_eval_preds = {}
  all_eval_nbest_texts = {}
  eval_uid_infos = {}
  for mn in model_names:
    eval_csv = get_eval_csv(get_model_dir(mn))
    eval_df = pd.read_csv(eval_csv)
    eval_uid_infos[mn] = {
        'uids': set(eval_df['utterance_id'].astype(str)),
        'n_rows': int(len(eval_df)),
        'n_uids': int(eval_df['utterance_id'].nunique()),
        'source_counts': eval_df['source'].fillna('').astype(str).value_counts().to_dict() if 'source' in eval_df.columns else {},
    }
    pred_col = 'pred' if 'pred' in eval_df.columns else 'text'
    all_eval_preds[mn] = {
      uid: _normalize_candidate_text(normalize_ipa, text)
      for uid, text in zip(eval_df['utterance_id'], eval_df[pred_col].fillna(''))
    }
    all_eval_nbest_texts[mn] = {}
    if int(tdt_eval_nbest or 0) > 0 and 'pred_nbest_texts' in eval_df.columns:
      for uid, texts in zip(eval_df['utterance_id'], eval_df['pred_nbest_texts']):
        parsed = _parse_serialized_text_list(normalize_ipa, texts)
        if parsed:
          all_eval_nbest_texts[mn][uid] = parsed[:int(tdt_eval_nbest)]
      if verbose and all_eval_nbest_texts[mn]:
        print(f'  Loaded eval nbest texts for {mn}: {len(all_eval_nbest_texts[mn])} utterances, topk={int(tdt_eval_nbest)}')

  all_logprobs = {}
  for mn in model_names:
    lp_path = get_model_dir(mn) / 'ctc_logprobs.pt'
    if lp_path.exists():
      all_logprobs[mn] = torch.load(str(lp_path), map_location='cpu', weights_only=False)
      if verbose:
        print(f'  Loaded {len(all_logprobs[mn])} utterances from {mn}')
    elif verbose:
      print(f'  No ctc_logprobs for {mn}; using primary/dual predictions as reranker candidates only')

  score_model_names = [mn for mn in model_names if mn in all_logprobs]
  candidate_model_names = list(model_names)
  if verbose and not score_model_names:
    print('  No models with ctc_logprobs.pt found; build pure candidate/text/audio/TDT reranker features only')

  all_word_logprobs = {}
  word_head_types = {}
  if feat_word:
    for mn in model_names:
      word_path = get_model_dir(mn) / 'ctc_logprobs_word.pt'
      aux_path = get_model_dir(mn) / 'ctc_logprobs_aux.pt'
      load_path = word_path if word_path.exists() else aux_path
      if load_path.exists():
        aux_data = torch.load(str(load_path), map_location='cpu', weights_only=False)
        all_word_logprobs[mn] = aux_data['logprobs']
        word_head_types[mn] = aux_data['meta']['head_type']
        if verbose:
          print(f'  Loaded word ({aux_data["meta"]["head_type"]}) logprobs for {mn}')
    assert all_word_logprobs, (
        'feat_word=True but no ctc_logprobs_word.pt found for any model. '
        'Either set --feat_word=False or train models with --save_word_head_preds --save_logprobs.')

  all_aux_meta = {}
  aux_meta_info = {}
  if feat_aux_meta:
    for mn in model_names:
      mp = get_model_dir(mn) / 'aux_meta_preds.pt'
      if mp.exists():
        data = torch.load(str(mp), map_location='cpu', weights_only=False)
        all_aux_meta[mn] = data['preds']
        aux_meta_info[mn] = data['meta']
        if verbose:
          mi = data['meta']
          print(f'  Loaded aux meta preds for {mn}: age={mi["has_age"]}(mode={mi["age_mode"]}), domain={mi["has_domain"]}')
    if verbose and not all_aux_meta:
      print('  WARNING: feat_aux_meta=True but no aux_meta_preds.pt found for any model')

  all_dual_head_preds = {}
  if feat_dual or feat_tdtctc_compare:
    for mn in model_names:
      dual_path = get_model_dir(mn) / 'dual_head_preds.pt'
      if dual_path.exists():
        dual_data = torch.load(str(dual_path), map_location='cpu', weights_only=False)
        all_dual_head_preds[mn] = dual_data.get('preds', {})
        if verbose:
          print(f'  Loaded dual-head preds for {mn}: {len(all_dual_head_preds[mn])} utterances')

  word_labels = load_word_labels(word_label_file, word_label_col=word_label_col, verbose=verbose) if feat_word_label else {}

  word_id_to_char = None
  word_blank_id = 0
  ipa_convert = None
  if feat_word and any(ht in ('word_ctc', 'word_ctc_bpe') for ht in word_head_types.values()):
    from src.models.base import WORD_CTC_BLANK, WORD_ID_TO_CHAR
    word_id_to_char = WORD_ID_TO_CHAR
    word_blank_id = WORD_CTC_BLANK
  if feat_word or feat_word_label:
    try:
      from eng_to_ipa import convert as ipa_convert
    except ImportError:
      if verbose and (feat_word or feat_word_label):
        print('  WARNING: eng_to_ipa not installed, IPA-conversion-derived features will be limited')

  uid_sets = [set(all_eval_preds[mn].keys()) for mn in candidate_model_names]
  uid_sets.extend(set(all_logprobs[mn].keys()) for mn in score_model_names)
  common_uids = _validate_eval_uid_sets(
      eval_uid_infos,
      allow_mismatch=allow_eval_set_mismatch,
      verbose=verbose,
  )
  char_to_id = {ch: cid for cid, ch in id_to_char.items()} if id_to_char else None
  model_ctc_meta = _get_model_ctc_meta(model_names, all_logprobs)

  eval_csv = get_eval_csv(get_model_dir(model_names[0]))
  gold_df = pd.read_csv(eval_csv)
  if require_gold:
    assert 'label' in gold_df.columns, f'Missing label column in {eval_csv}'
    gold = dict(zip(gold_df['utterance_id'], gold_df['label'].fillna('')))
  else:
    gold = {}
  meta = {}
  for _, row in gold_df.iterrows():
    meta[row['utterance_id']] = {
        'source': row.get('source', ''),
        'age_bucket': row.get('age_bucket', ''),
        'child_id': row.get('child_id', ''),
        'session_id': row.get('session_id', ''),
        'audio_path': row.get('audio_path', ''),
        'audio_duration_sec': row.get('audio_duration_sec', ''),
    }

  # ---- Apply per-model max audio duration filter on logprobs ----
  if model_max_dur:
    _dur_map = {}
    for _, _row in gold_df.iterrows():
      _dur_map[_row['utterance_id']] = _safe_float(_row.get('audio_duration_sec', np.nan))
    for mn, max_dur in model_max_dur.items():
      if mn not in all_logprobs:
        continue
      orig_count = len(all_logprobs[mn])
      all_logprobs[mn] = {
          uid: lp for uid, lp in all_logprobs[mn].items()
          if _dur_map.get(uid, float('inf')) <= max_dur
      }
      removed = orig_count - len(all_logprobs[mn])
      if verbose:
        print(f'  Duration filter ({max_dur:.1f}s): {mn} {orig_count} -> {len(all_logprobs[mn])} '
              f'utterances ({removed} removed)')

  uids_list = sorted(common_uids)
  total = len(uids_list)
  t0 = time.time()

  global _MP_BUILD
  _MP_BUILD = {
      'model_names': model_names,
      'candidate_model_names': candidate_model_names,
      'score_model_names': score_model_names,
      'all_eval_preds': all_eval_preds,
      'all_eval_nbest_texts': all_eval_nbest_texts,
      'all_logprobs': all_logprobs,
      'blank_id': blank_id,
      'beam_width': beam_width,
      'nbest': nbest,
      'id_to_char': id_to_char,
      'char_to_id': char_to_id,
      'model_ctc_meta': model_ctc_meta,
      'lm': lm,
      'aux_lms': aux_lms or {},
      'no_lm_feats': no_lm_feats,
      'gold': gold,
      'meta': meta,
      'normalize_ipa': normalize_ipa,
      'prefix_beam_search_nbest': prefix_beam_search_nbest,
      'ctc_force_score_batch': ctc_force_score_batch,
      'feat_text': feat_text,
      'feat_ipa': feat_ipa,
      'feat_ctc_stats': feat_ctc_stats,
      'feat_audio': feat_audio,
      'feat_consensus': feat_consensus,
      'feat_group_ext': feat_group_ext,
      'feat_align': feat_align,
      'feat_logprob_proxy': feat_logprob_proxy,
      'feat_tdtctc_compare': feat_tdtctc_compare,
      'feat_dual': feat_dual,
      'feat_word': feat_word,
      'feat_aux_meta': feat_aux_meta,
      'feat_word_label': feat_word_label,
      'tdt_eval_nbest': int(tdt_eval_nbest or 0),
      'all_word_logprobs': all_word_logprobs,
      'word_head_types': word_head_types,
      'all_aux_meta': all_aux_meta,
      'aux_meta_info': aux_meta_info,
      'all_dual_head_preds': all_dual_head_preds,
      'word_labels': word_labels,
        'infer_mode': not require_gold,
      '_word_id_to_char': word_id_to_char,
      '_word_blank_id': word_blank_id,
      '_ipa_convert': ipa_convert,
  }

  if n_workers <= 0:
    import os
    n_workers = min(os.cpu_count() or 1, 16)

  rows = []
  if n_workers > 1:
    import multiprocessing as mp
    ctx = mp.get_context('fork')

    def _pool_init():
      import os
      import torch
      torch.set_num_threads(1)
      os.environ['OMP_NUM_THREADS'] = '1'
      os.environ['MKL_NUM_THREADS'] = '1'

    if verbose:
      print(f'  Using {n_workers} workers for parallel feature building')
    chunksize = max(1, total // (n_workers * 4))
    with ctx.Pool(n_workers, initializer=_pool_init) as pool:
      for uid_rows in _iter_with_progress(pool.imap_unordered(_build_rows_for_uid, uids_list, chunksize=chunksize),
                                          total=total, desc='  Building features', verbose=verbose):
        rows.extend(uid_rows)
  else:
    for uid in _iter_with_progress(uids_list, total=total, desc='  Building features', verbose=verbose):
      rows.extend(_build_rows_for_uid(uid))

  _MP_BUILD = {}

  if verbose:
    elapsed = time.time() - t0
    print(f'\r  Built {len(rows)} candidate rows for {total} utterances in {elapsed:.1f}s')

  df = pd.DataFrame(rows)

  has_ctc_scores = 'ctc_score_mean' in df.columns
  if has_ctc_scores:
    for mn in model_names:
      col = f'ctc_score_{mn}'
      if col not in df.columns:
        continue
      df[f'{col}_rank'] = df.groupby('uid')[col].rank(ascending=False, method='min')
      df[f'{col}_per_char'] = df[col] / df['text_len'].clip(lower=1)
      if feat_group_ext:
        grp_mn = df.groupby('uid')[col]
        df[f'{col}_zscore'] = (df[col] - grp_mn.transform('mean')) / grp_mn.transform('std').clip(lower=1e-8)

    df['ctc_score_mean_rank'] = df.groupby('uid')['ctc_score_mean'].rank(ascending=False, method='min')
    grp = df.groupby('uid')['ctc_score_mean']
    df['ctc_score_mean_zscore'] = (df['ctc_score_mean'] - grp.transform('mean')) / grp.transform('std').clip(lower=1e-8)
    df['ctc_score_diff_from_best'] = df['ctc_score_mean'] - grp.transform('max')

    if feat_group_ext:
      df['ctc_score_mean_pct'] = df.groupby('uid')['ctc_score_mean'].rank(pct=True)
      df['ctc_score_diff_from_min'] = df['ctc_score_mean'] - grp.transform('min')
      df['ctc_score_diff_from_group_mean'] = df['ctc_score_mean'] - grp.transform('mean')
      df['ctc_score_diff_from_group_median'] = df['ctc_score_mean'] - grp.transform('median')
      grp_min = grp.transform('min')
      grp_range = (grp.transform('max') - grp_min).clip(lower=1e-8)
      df['ctc_score_minmax_norm'] = (df['ctc_score_mean'] - grp_min) / grp_range

    df['ctc_score_mean_per_char'] = df['ctc_score_mean'] / df['text_len'].clip(lower=1)

    if feat_group_ext:
      for mn in model_names:
        col = f'ctc_score_{mn}'
        if col not in df.columns:
          continue
        grp_mn = df.groupby('uid')[col]
        df[f'{col}_diff_from_median'] = df[col] - grp_mn.transform('median')

  if feat_word and 'word_ctc_score_mean' in df.columns:
    grp_word = df.groupby('uid')['word_ctc_score_mean']
    df['word_ctc_score_mean_rank'] = grp_word.rank(ascending=False, method='min')
    df['word_ctc_score_diff_from_best'] = df['word_ctc_score_mean'] - grp_word.transform('max')
    for mn in word_head_types:
      col = f'word_ctc_score_{mn}'
      if col in df.columns:
        df[f'{col}_rank'] = df.groupby('uid')[col].rank(ascending=False, method='min')

  for mn in model_names:
    df[f'is_best_{mn}'] = (df[f'beam_rank_{mn}'] == 0).astype(int)
  df['n_models_is_best'] = sum(df[f'is_best_{mn}'] for mn in model_names)

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

  if not no_lm_feats and feat_group_ext:
    grp_lm = df.groupby('uid')['lm_score']
    df['lm_score_rank'] = grp_lm.rank(ascending=False, method='min')
    df['lm_score_zscore'] = (df['lm_score'] - grp_lm.transform('mean')) / grp_lm.transform('std').clip(lower=1e-8)
    df['lm_score_diff_from_best'] = df['lm_score'] - grp_lm.transform('max')
    df['lm_score_pct'] = grp_lm.rank(pct=True)
    aux_lm_score_cols = [c for c in df.columns if c.endswith('_lm_score') and c != 'lm_score']
    for col in aux_lm_score_cols:
      grp_aux_lm = df.groupby('uid')[col]
      df[f'{col}_rank'] = grp_aux_lm.rank(ascending=False, method='min')
      df[f'{col}_zscore'] = (df[col] - grp_aux_lm.transform('mean')) / grp_aux_lm.transform('std').clip(lower=1e-8)
      df[f'{col}_diff_from_best'] = df[col] - grp_aux_lm.transform('max')
      df[f'{col}_pct'] = grp_aux_lm.rank(pct=True)

  if feat_consensus:
    grp_ped = df.groupby('uid')['mean_pairwise_edit_dist']
    df['mean_pairwise_edit_dist_rank'] = grp_ped.rank(method='min')

  if feat_text and 'n_spaces' in df.columns:
    grp_sp = df.groupby('uid')['n_spaces']
    df['n_spaces_diff_from_median'] = df['n_spaces'] - grp_sp.transform('median')

  if feat_consensus or feat_mbr:
    import editdistance

    _mbr_dist_override = _MBR_DIST_FN
    _pw_ed_override = _PAIRWISE_EDIT_DIST_FN

    def _mbr_features(group):
      texts = group['candidate_text'].values
      n = len(texts)
      if n <= 1:
        group['is_mbr_selected'] = 1
        group['edit_dist_to_mbr'] = 0
        return group
      avg_cer = np.zeros(n)
      if callable(_mbr_dist_override):
        for i in range(n):
          total = 0
          for j in range(n):
            if i != j:
              total += _mbr_dist_override(texts[j], texts[i])
          avg_cer[i] = total / (n - 1)
      else:
        for i in range(n):
          total = 0
          for j in range(n):
            if i != j:
              ri = normalize_ipa(texts[i]).strip()
              rj = normalize_ipa(texts[j]).strip()
              if ri:
                total += editdistance.eval(ri, rj) / len(ri)
          avg_cer[i] = total / (n - 1)
      mbr_idx = int(np.argmin(avg_cer))
      mbr_text = texts[mbr_idx]
      group['is_mbr_selected'] = (group['candidate_text'] == mbr_text).astype(int)
      if callable(_pw_ed_override):
        group['edit_dist_to_mbr'] = group['candidate_text'].apply(lambda t: _pw_ed_override(t, mbr_text))
      else:
        group['edit_dist_to_mbr'] = group['candidate_text'].apply(lambda t: editdistance.eval(t, mbr_text))
      return group

    df = df.groupby('uid', group_keys=False).apply(_mbr_features)

  # Post-build hook (e.g. word-track adds word-level edit distance features)
  if callable(_POST_BUILD_HOOK):
    df = _POST_BUILD_HOOK(df, model_names, feat_consensus)

  exclude = {
      'uid', 'candidate_text', 'source', 'target_cer', 'child_id', 'age_bucket',
      'word_label_text', 'word_label_ipa',
  }
  feat_cols = [c for c in df.columns if c not in exclude]

  return df, feat_cols, gold, meta


def build_reranker_infer_dataset(model_names, get_model_dir, get_eval_csv,
                                 prefix_beam_search_nbest, ctc_force_score_batch,
                                 normalize_ipa, id_to_char, blank_id,
                                 nbest=10, beam_width=10,
                                 lm=None, aux_lms=None, verbose=True,
                                 feat_text=False, feat_ipa=False,
                                 feat_ctc_stats=False, feat_audio=False,
                                 feat_consensus=False, feat_mbr=False,
                                 feat_group_ext=False,
                                 feat_align=False, feat_logprob_proxy=False,
                                 feat_tdtctc_compare=False,
                                 feat_dual=False,
                                 feat_word=False, feat_aux=None, feat_aux_meta=False,
                                 feat_word_label=False,
                                 word_label_file='', word_label_col='',
                                 no_lm_feats=False, n_workers=0,
                                 model_max_dur=None):
  df, feat_cols, _, meta = build_reranker_dataset(
      model_names=model_names,
      get_model_dir=get_model_dir,
      get_eval_csv=get_eval_csv,
      prefix_beam_search_nbest=prefix_beam_search_nbest,
      ctc_force_score_batch=ctc_force_score_batch,
      normalize_ipa=normalize_ipa,
      id_to_char=id_to_char,
      blank_id=blank_id,
      nbest=nbest,
      beam_width=beam_width,
      lm=lm,
      aux_lms=aux_lms,
      verbose=verbose,
      feat_text=feat_text,
      feat_ipa=feat_ipa,
      feat_ctc_stats=feat_ctc_stats,
      feat_audio=feat_audio,
      feat_consensus=feat_consensus,
      feat_mbr=feat_mbr,
      feat_group_ext=feat_group_ext,
      feat_align=feat_align,
      feat_logprob_proxy=feat_logprob_proxy,
      feat_tdtctc_compare=feat_tdtctc_compare,
      feat_dual=feat_dual,
      feat_word=feat_word,
      feat_aux=feat_aux,
      feat_aux_meta=feat_aux_meta,
      feat_word_label=feat_word_label,
      word_label_file=word_label_file,
      word_label_col=word_label_col,
      no_lm_feats=no_lm_feats,
      n_workers=n_workers,
      require_gold=False,
      model_max_dur=model_max_dur,
  )
  return df, feat_cols, meta