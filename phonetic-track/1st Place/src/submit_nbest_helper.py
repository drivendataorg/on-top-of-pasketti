#!/usr/bin/env python3

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np


def _diag(msg):
  print(f'[submit-nbest] {msg}', flush=True)


def _normalize_ipa_mbr(s):
  import re
  import unicodedata
  s = unicodedata.normalize('NFC', s)
  s = re.sub(r'\s+', ' ', s).strip()
  return s


def _mbr_cer(ref, hyp):
  r = _normalize_ipa_mbr(ref).strip()
  h = _normalize_ipa_mbr(hyp).strip()
  if not r:
    return 0.0 if not h else 1.0
  import editdistance
  return editdistance.eval(r, h) / len(r)


def _mbr_select(candidates):
  if len(candidates) == 1:
    return candidates[0]
  n = len(candidates)
  avg_cer = []
  for i in range(n):
    total_cer = sum(_mbr_cer(candidates[j], candidates[i]) for j in range(n) if j != i)
    avg_cer.append(total_cer / (n - 1))
  best_idx = int(min(range(n), key=lambda i: avg_cer[i]))
  return candidates[best_idx]


_STATE = {}


def _pool_init():
  try:
    import torch
    torch.set_num_threads(1)
    if hasattr(torch, 'set_num_interop_threads'):
      torch.set_num_interop_threads(1)
  except Exception:
    pass


def _maybe_log_progress(desc, idx, total, start_time, interval):
  if idx % interval != 0 and idx != total:
    return
  elapsed = time.time() - start_time
  rate = idx / elapsed if elapsed > 0 else 0.0
  eta = (total - idx) / rate if rate > 0 else 0.0
  print(f'[submit-nbest] {desc}: {idx}/{total} ({rate:.1f} utt/s, ETA {eta:.0f}s)', flush=True)


def _build_candidates_single_uid(uid):
  state = _STATE
  candidate_set = set()
  raw_count = 0
  for model_name in state['candidate_model_names']:
    if model_name in state['all_logprobs'] and not state['skip_ctc_candidate_models'].get(model_name, False):
      log_probs = state['all_logprobs'][model_name][uid].astype(np.float32)
      hyps = state['prefix_beam_search_nbest'](
          log_probs,
          state['blank_id'],
          state['beam_width'],
          nbest=state['nbest'],
          id_to_char=state['id_to_char'])
      raw_count += len(hyps)
      for _score, text in hyps:
        candidate_set.add(text)

    pred = str(state['all_model_pred_map'].get(uid, {}).get(model_name, '') or '')
    if pred:
      raw_count += 1
      candidate_set.add(pred)

  candidates = list(candidate_set)
  return uid, candidates, raw_count, len(candidates)


def _rescore_single_uid(uid):
  state = _STATE
  candidates = state['candidate_lists'][uid]
  if not candidates:
    return uid, ''
  if len(candidates) == 1:
    return uid, candidates[0]

  all_token_ids = []
  for cand_text in candidates:
    token_ids = [state['char_to_id'][ch] for ch in cand_text if ch in state['char_to_id']]
    all_token_ids.append(token_ids)

  avg_scores = np.zeros(len(candidates), dtype=np.float64)
  for model_name in state['model_names']:
    log_probs = state['all_logprobs'][model_name][uid].astype(np.float32)
    scores = state['ctc_force_score_batch'](log_probs, all_token_ids, blank=state['blank_id'])
    for i, score in enumerate(scores):
      avg_scores[i] += score
  avg_scores /= len(state['model_names'])
  best_idx = int(np.argmax(avg_scores))
  return uid, candidates[best_idx]


def main():
  parser = argparse.ArgumentParser(description='Isolated CPU helper for online N-best rescoring')
  parser.add_argument('--input', required=True, help='Pickle payload path')
  parser.add_argument('--output', required=True, help='Pickle result path')
  args = parser.parse_args()

  input_path = Path(args.input)
  output_path = Path(args.output)
  with input_path.open('rb') as f:
    payload = pickle.load(f)

  from ctc_decode import prefix_beam_search_nbest, ctc_force_score_batch
  from models.base import IPA_CTC_BLANK, IPA_ID_TO_CHAR

  all_logprobs = payload['all_logprobs']
  all_model_preds = payload['all_model_preds']
  all_model_pred_map = payload['all_model_pred_map']
  skip_ctc_candidate_models = payload['skip_ctc_candidate_models']
  nbest = int(payload['nbest'])
  beam_width = int(payload['beam_width'])

  model_names = list(all_logprobs.keys())
  candidate_model_names = list(model_names)
  for pred_map in all_model_pred_map.values():
    for model_name in pred_map.keys():
      if model_name not in candidate_model_names:
        candidate_model_names.append(model_name)

  uid_sets = [set(all_logprobs[mn].keys()) for mn in model_names]
  common_uids = set.intersection(*uid_sets)
  uids_list = sorted(common_uids)
  total = len(uids_list)

  blank_id = IPA_CTC_BLANK
  id_to_char = IPA_ID_TO_CHAR
  char_to_id = {ch: cid for cid, ch in id_to_char.items()}

  n_workers = payload.get('n_workers', 0) or 0
  if n_workers <= 0:
    n_workers = min(os.cpu_count() or 1, 16)
  n_workers = max(1, n_workers)
  can_parallel = n_workers > 1 and total >= 512
  chunksize = max(4, min(32, total // max(n_workers * 16, 1))) if can_parallel else 1

  _diag(f'Loaded payload: {len(model_names)} scoring models, {len(candidate_model_names)} candidate models, {total} utterances, workers={n_workers}, chunksize={chunksize}')

  global _STATE
  _STATE = {
      'candidate_model_names': candidate_model_names,
      'all_logprobs': all_logprobs,
      'all_model_pred_map': all_model_pred_map,
      'skip_ctc_candidate_models': skip_ctc_candidate_models,
      'blank_id': blank_id,
      'beam_width': beam_width,
      'nbest': nbest,
      'id_to_char': id_to_char,
      'char_to_id': char_to_id,
      'prefix_beam_search_nbest': prefix_beam_search_nbest,
      'ctc_force_score_batch': ctc_force_score_batch,
  }

  candidate_lists = {}
  n_cands_total = 0
  n_unique_total = 0
  build_interval = max(10000, total // 5)
  build_start = time.time()
  if can_parallel:
    import multiprocessing as mp
    ctx = mp.get_context('fork')
    with ctx.Pool(n_workers, initializer=_pool_init) as pool:
      for idx, (uid, candidates, raw_count, unique_count) in enumerate(
          pool.imap_unordered(_build_candidates_single_uid, uids_list, chunksize=chunksize), start=1):
        candidate_lists[uid] = candidates
        n_cands_total += raw_count
        n_unique_total += unique_count
        _maybe_log_progress('Build candidates', idx, total, build_start, build_interval)
  else:
    _pool_init()
    for idx, uid in enumerate(uids_list, start=1):
      uid, candidates, raw_count, unique_count = _build_candidates_single_uid(uid)
      candidate_lists[uid] = candidates
      n_cands_total += raw_count
      n_unique_total += unique_count
      _maybe_log_progress('Build candidates', idx, total, build_start, build_interval)

  _STATE = {
      'model_names': model_names,
      'all_logprobs': all_logprobs,
      'candidate_lists': candidate_lists,
      'blank_id': blank_id,
      'char_to_id': char_to_id,
      'ctc_force_score_batch': ctc_force_score_batch,
  }

  _diag('Build candidates done; starting exact CTC rescoring')
  predictions = {}
  score_interval = max(2000, total // 10)
  score_start = time.time()
  if can_parallel:
    import multiprocessing as mp
    ctx = mp.get_context('fork')
    with ctx.Pool(n_workers, initializer=_pool_init) as pool:
      for idx, (uid, pred) in enumerate(
          pool.imap_unordered(_rescore_single_uid, uids_list, chunksize=chunksize), start=1):
        predictions[uid] = pred
        _maybe_log_progress('N-best rescore', idx, total, score_start, score_interval)
  else:
    for idx, uid in enumerate(uids_list, start=1):
      uid, pred = _rescore_single_uid(uid)
      predictions[uid] = pred
      _maybe_log_progress('N-best rescore', idx, total, score_start, score_interval)

  n_fallback = 0
  for uid, candidates in all_model_preds.items():
    if uid not in predictions:
      if len(set(candidates)) <= 1:
        predictions[uid] = candidates[0] if candidates else ''
      else:
        predictions[uid] = _mbr_select(candidates)
      n_fallback += 1

  result = {
      'predictions': predictions,
      'avg_candidates_raw': (n_cands_total / total) if total else 0.0,
      'avg_candidates_unique': (n_unique_total / total) if total else 0.0,
      'fallback_count': n_fallback,
      'elapsed': time.time() - build_start,
  }
  with output_path.open('wb') as f:
    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
  _diag(f'Helper done: {total} utterances in {result["elapsed"]:.1f}s')


if __name__ == '__main__':
  main()