#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   eval.py
#        \author   chenghuige  
#          \date   2025-02-13
#   \Description   Shared evaluation for both Phonetic (CER) and Word (WER) tracks.
#                  Metric function dispatched via FLAGS.score_metric.
#                  Official scoring from metric/score.py (runtime image).
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import csv
import numpy as np
from gezi.common import * 
from src.config import *
from src.preprocess import get_tokenizer

from metric.score import score_ipa_cer, score_wer, normalize_ipa

SCORE_FNS = {
  'ipa_cer': score_ipa_cer,
  'wer': score_wer,
}


def _age_group_series(age_bucket_series):
  """Map age_bucket column to coarse groups: '3-4' or '5+'.
  
  Handles string values like '3-4', '5-7', '8-12', or numeric-like '3', '5'.
  Returns a pd.Series with values '3-4', '5+', or '' (unknown).
  """
  import re

  def _map(val):
    if pd.isna(val) or str(val).strip() == '':
      return ''
    s = str(val).strip()
    # Direct match
    if s in ('3-4',):
      return '3-4'
    if s in ('5+', '5-7', '5-12', '8-12', '6-8', '6-12'):
      return '5+'
    # Try to extract first number
    m = re.match(r'(\d+)', s)
    if m:
      first_num = int(m.group(1))
      return '3-4' if first_num <= 4 else '5+'
    return ''

  return age_bucket_series.map(_map)


def _age_fine_group_series(age_bucket_series):
  """Map age_bucket to fine-grained groups: '3-4', '5-7', '8-11', '12+'."""
  import re

  def _map(val):
    if pd.isna(val) or str(val).strip() == '':
      return ''
    s = str(val).strip()
    if s in ('3-4',):
      return '3-4'
    if s in ('5-7',):
      return '5-7'
    if s in ('8-11', '8-12'):
      return '8-11'
    if s in ('12+',):
      return '12+'
    m = re.match(r'(\d+)', s)
    if m:
      n = int(m.group(1))
      if n <= 4: return '3-4'
      if n <= 7: return '5-7'
      if n <= 11: return '8-11'
      return '12+'
    return ''

  return age_bucket_series.map(_map)


def _filter_empty_refs(targets, preds):
  """Filter out (target, pred) pairs where target is empty after normalization."""
  if FLAGS.score_metric == 'wer':
    from metric.score import EnglishTextNormalizer, english_spelling_normalizer
    normalizer = EnglishTextNormalizer(english_spelling_normalizer)
    return [(t, p) for t, p in zip(targets, preds) if normalizer(t).strip()]
  elif FLAGS.score_metric == 'ipa_cer':
    return [(t, p) for t, p in zip(targets, preds) if normalize_ipa(t).strip()]
  else:
    return list(zip(targets, preds))


def _trim_and_decode(token_ids_2d):
  """Decode token ids to strings. Trims trailing pad/eos tokens BEFORE
  calling batch_decode so the tokenizer does minimal work.

  token_ids_2d: np.ndarray of shape (N, max_len), may contain -100 or pad_id
  Returns: list of N decoded strings
  """
  tokenizer = get_tokenizer(FLAGS.backbone)

  # NeMo backbone: tokenizer is None, use NeMo's own SentencePiece tokenizer
  if tokenizer is None:
    try:
      from src.preprocess import _is_nemo_backbone
      if _is_nemo_backbone(FLAGS.backbone):
        from src.models.nemo import _load_nemo_model, BACKBONES
        import gezi
        nemo_tok = gezi.get('nemo_tokenizer')
        if nemo_tok is None:
          # Try to get from loaded model
          model = gezi.get('model')
          if model is not None and hasattr(model, '_nemo_tokenizer'):
            nemo_tok = model._nemo_tokenizer
        if nemo_tok is not None:
          pad_id = 0
          token_ids_2d = np.where(token_ids_2d == -100, pad_id, token_ids_2d).astype(np.int64)
          texts = []
          for row in token_ids_2d:
            pad_positions = np.where(row == pad_id)[0]
            end = int(pad_positions[0]) if len(pad_positions) > 0 else len(row)
            ids = row[:end].tolist()
            texts.append(nemo_tok.ids_to_text(ids) if ids else '')
          return texts
    except Exception:
      pass
    # Fallback: treat as raw IDs, return empty strings
    return [''] * len(token_ids_2d)

  pad_id = tokenizer.pad_token_id or 0
  eos_id = tokenizer.eos_token_id

  # Ensure integer dtype (model output may be float); replace -100 with pad
  token_ids_2d = np.where(token_ids_2d == -100, pad_id, token_ids_2d).astype(np.int64)

  # Trim each sequence at the first pad token to avoid decoding padding
  trimmed = []
  for row in token_ids_2d:
    # Find first pad position
    pad_positions = np.where(row == pad_id)[0]
    end = int(pad_positions[0]) if len(pad_positions) > 0 else len(row)
    trimmed.append(row[:end].tolist())

  # batch_decode on trimmed (much shorter) sequences
  texts = tokenizer.batch_decode(trimmed, skip_special_tokens=True)
  return texts


def _decode_ipa_char_ids(char_ids_2d):
  """Decode character-level IPA IDs to text strings.
  Used when CTC outputs IPA char indices (from IPA_CHAR_VOCAB)."""
  from src.models.base import IPA_ID_TO_CHAR, IPA_CTC_BLANK
  results = []
  for row in char_ids_2d:
    chars = []
    for cid in row:
      cid = int(cid)
      if cid == IPA_CTC_BLANK:  # blank / pad
        continue
      ch = IPA_ID_TO_CHAR.get(cid)
      if ch is not None:
        chars.append(ch)
    results.append(''.join(chars))
  return results


def decode_predictions(res, model=None):
  """Decode model output to text strings — single shared entry point.

  Handles all decode paths:
    1. res['pred_texts'] already set (NeMo native CTC, or base _generate_ctc char-level)
    2. CTC char-level IPA: decode IPA char IDs (0-52) via _decode_ipa_char_ids
    3. BPE token IDs: decode via tokenizer (_trim_and_decode / decode_ids)

  Args:
    res: dict from model.forward(), must contain 'pred' and optionally 'pred_texts'
    model: the model instance (used to check _ctc_char_level)
  Returns:
    list of decoded text strings
  """
  import torch
  # Path 1: direct text predictions (set by _generate_ctc char-level or nemo_native_ctc)
  if 'pred_texts' in res and res['pred_texts']:
    return list(res['pred_texts'])

  pred_ids = res['pred']
  if isinstance(pred_ids, torch.Tensor):
    pred_ids = pred_ids.cpu().numpy()
  pred_ids = np.asarray(pred_ids)
  if pred_ids.ndim == 1:
    pred_ids = pred_ids[np.newaxis, :]

  # Path 2: CTC char-level IPA (check model flag or global)
  is_ctc_char = False
  if model is not None and getattr(model, '_ctc_char_level', False):
    is_ctc_char = True
  elif gezi.get('ctc_char_level') or gezi.get('s2s_ipa_chars'):
    is_ctc_char = True
  if is_ctc_char:
    return _decode_ipa_char_ids(pred_ids)

  # Path 3: BPE token IDs → tokenizer decode
  return _trim_and_decode(pred_ids)


# keep backward compatibility
def decode_ids(token_ids):
  """Decode token ids (numpy array) to strings."""
  token_ids = np.asarray(token_ids)
  if token_ids.ndim == 1:
    token_ids = token_ids[np.newaxis, :]
  return _trim_and_decode(token_ids)


def eval_df(df):
  """Core evaluation: compute metric from a DataFrame with 'pred' and 'label' columns."""
  metrics = OrderedDict()

  # Filter out rows with empty labels (e.g. some ext data has no transcription)
  mask = df['label'].apply(lambda x: bool(str(x).strip()))
  if mask.sum() < len(df):
    ic(f'eval_df: filtered {(~mask).sum()} empty-label rows')
    df = df[mask]

  preds = df['pred'].tolist()
  targets = df['label'].tolist()

  if not preds or not targets:
    metrics['score'] = 1.0
    return metrics

  # Filter out pairs where reference becomes empty after normalization
  # (e.g. punctuation-only text normalised to "")
  if FLAGS.score_metric == 'wer':
    from metric.score import EnglishTextNormalizer, english_spelling_normalizer
    normalizer = EnglishTextNormalizer(english_spelling_normalizer)
    filtered = [(t, p) for t, p in zip(targets, preds) if normalizer(t).strip()]
    if len(filtered) < len(targets):
      ic(f'eval_df: filtered {len(targets) - len(filtered)} empty-after-normalize rows')
      if not filtered:
        metrics['score'] = 1.0
        metrics['n_samples'] = 0
        return metrics
      targets, preds = zip(*filtered)
      targets, preds = list(targets), list(preds)
  elif FLAGS.score_metric == 'ipa_cer':
    filtered = [(t, p) for t, p in zip(targets, preds) if normalize_ipa(t).strip()]
    if len(filtered) < len(targets):
      ic(f'eval_df: filtered {len(targets) - len(filtered)} empty-after-normalize rows')
      if not filtered:
        metrics['score'] = 1.0
        metrics['n_samples'] = 0
        return metrics
      targets, preds = zip(*filtered)
      targets, preds = list(targets), list(preds)

  score_fn = SCORE_FNS[FLAGS.score_metric]
  metrics['score'] = score_fn(targets, preds)
  metrics['n_samples'] = len(preds)

  # Per-source breakdown (DD vs ext) when eval_ext or eval_add_ext is enabled
  if (getattr(FLAGS, 'eval_ext', False) or getattr(FLAGS, 'eval_add_ext', False)) and 'source' in df.columns:
    for src in ['dd', 'ext']:
      src_mask = df['source'] == src
      if src_mask.sum() == 0:
        continue
      src_preds = df.loc[src_mask, 'pred'].tolist()
      src_targets = df.loc[src_mask, 'label'].tolist()
      src_filtered = _filter_empty_refs(src_targets, src_preds)
      if src_filtered:
        st, sp = zip(*src_filtered)
        metrics[f'score/{src}'] = score_fn(list(st), list(sp))
    # Weighted score: macro-average DD and EXT scores with eval_ext_weight.
    # This replaces the corpus-level micro-average (which is dominated by the
    # larger source) with a controlled weighted average.
    # score = (score_dd + w * score_ext) / (1 + w)
    if 'score/dd' in metrics and 'score/ext' in metrics:
      w = getattr(FLAGS, 'eval_ext_weight', 1.0)
      metrics['score'] = (metrics['score/dd'] + w * metrics['score/ext']) / (1.0 + w)
      metrics['score/raw'] = score_fn(targets, preds)  # keep unweighted corpus-level for reference

  # Helper: compute score for a subset, with source breakdown and weighted aggregation
  # consistent with the top-level score logic (weighted DD/EXT average).
  has_source = 'source' in df.columns and (
    getattr(FLAGS, 'eval_ext', False) or getattr(FLAGS, 'eval_add_ext', False))
  ext_weight = getattr(FLAGS, 'eval_ext_weight', 1.0)

  def _bucket_score(mask, name, extra_metrics=None):
    """Compute score/{name}, score/dd/{name}, score/ext/{name}.
    score/{name} uses same weighted logic as top-level score when DD+EXT present."""
    if mask.sum() == 0:
      return
    bk_preds = df.loc[mask, 'pred'].tolist()
    bk_targets = df.loc[mask, 'label'].tolist()
    bk_filtered = _filter_empty_refs(bk_targets, bk_preds)
    if not bk_filtered:
      return
    bt, bp = zip(*bk_filtered)
    raw_score = score_fn(list(bt), list(bp))
    if extra_metrics:
      for k, v in extra_metrics.items():
        metrics[k] = v
    # source breakdown
    if has_source:
      for src in ['dd', 'ext']:
        src_mask = mask & (df['source'] == src)
        if src_mask.sum() == 0:
          continue
        sp = df.loc[src_mask, 'pred'].tolist()
        st = df.loc[src_mask, 'label'].tolist()
        sf = _filter_empty_refs(st, sp)
        if sf:
          sft, sfp = zip(*sf)
          metrics[f'score/{src}/{name}'] = score_fn(list(sft), list(sfp))
      # weighted average consistent with top-level score
      dd_key, ext_key = f'score/dd/{name}', f'score/ext/{name}'
      if dd_key in metrics and ext_key in metrics:
        metrics[f'score/{name}'] = (metrics[dd_key] + ext_weight * metrics[ext_key]) / (1.0 + ext_weight)
        return
    # no source split or only one source: use raw corpus-level score
    metrics[f'score/{name}'] = raw_score

  # ---- Age-group breakdown: score/3-4, score/5+ ----
  if 'age_bucket' in df.columns:
    age_group = _age_group_series(df['age_bucket'])
    for grp in ['3-4', '5+']:
      _bucket_score(age_group == grp, grp)

  # ---- Fine-grained age breakdown: score/5-7, score/8-11, score/12+ ----
  if 'age_bucket' in df.columns:
    age_fine = _age_fine_group_series(df['age_bucket'])
    for grp in ['5-7', '8-11', '12+']:
      _bucket_score(age_fine == grp, grp)

  # ---- Word-count bucket breakdown: score/1w, score/2-3w, ... ----
  # "words" = space-separated tokens in the label (IPA phonological words or orthographic words)
  _WORD_BUCKETS = [(1, 1, '1w'), (2, 3, '2-3w'), (4, 6, '4-6w'), (7, 10, '7-10w'), (11, 9999, '11+w')]
  label_wc = df['label'].apply(lambda x: len(str(x).split()))
  for lo, hi, bname in _WORD_BUCKETS:
    _bucket_score((label_wc >= lo) & (label_wc <= hi), bname)

  # ---- Duration bucket breakdown: score/0-1s, score/1-2s, ... ----
  if 'audio_duration_sec' in df.columns:
    dur = df['audio_duration_sec'].astype(float)
    _DUR_BUCKETS = [(0, 1, '0-1s'), (1, 2, '1-2s'), (2, 5, '2-5s'), (5, 1e9, '5+s')]
    for lo, hi, dname in _DUR_BUCKETS:
      dk_mask = (dur >= lo) & (dur < hi)
      n_filtered = len(_filter_empty_refs(
        df.loc[dk_mask, 'label'].tolist(), df.loc[dk_mask, 'pred'].tolist())) if dk_mask.sum() > 0 else 0
      _bucket_score(dk_mask, dname, extra_metrics={f'n/{dname}': n_filtered} if n_filtered else None)

  return metrics


def _to_2d_array(arr):
  """Convert a list of variable-length sequences to a padded 2D numpy array."""
  if isinstance(arr, np.ndarray) and arr.ndim >= 2:
    return arr
  # Try fast path first
  try:
    out = np.asarray(arr)
    if out.dtype == object:
      raise ValueError
    return out
  except (ValueError, TypeError):
    pass
  # Ragged: pad to longest
  maxlen = max(len(row) for row in arr)
  padded = np.full((len(arr), maxlen), fill_value=-100, dtype=np.int64)
  for i, row in enumerate(arr):
    n = len(row)
    if n > 0:
      padded[i, :n] = row[:n] if isinstance(row, np.ndarray) else np.asarray(row[:n])
  return padded


def evaluate(y_true, y_pred, x=None, other=None, **kwargs):
  """
  Called by mt.fit after all eval batches.
  y_true: label token ids, y_pred: generated token ids.
  """
  t0 = time.time()
  metrics = OrderedDict()

  y_pred = _to_2d_array(y_pred)
  y_true = _to_2d_array(y_true)

  if y_pred.ndim < 2:
    metrics['score'] = 1.0
    if gz.get('timer'):
      metrics['elapsed'] = gz.get('timer').elapsed_minutes()
    return metrics

  t1 = time.time()
  # For some NeMo / custom decode paths (e.g. TDT 53-vocab), model.forward()
  # already produced decoded texts. Use them directly to avoid an expensive and
  # potentially wrong token-id -> tokenizer decode roundtrip.
  eval_pred_texts = gz.get('eval_pred_texts', None)
  if eval_pred_texts and len(eval_pred_texts) >= len(y_pred):
    all_preds = eval_pred_texts[:len(y_pred)]
    gz.set('eval_pred_texts', None)
  else:
    # Decode predictions using shared logic
    _res = {'pred': y_pred}  # wrap for decode_predictions
    all_preds = decode_predictions(_res, model=gezi.get('model'))
    if eval_pred_texts:
      gz.set('eval_pred_texts', None)

  model = gezi.get('model')
  is_ipa_direct = False
  if model is not None and getattr(model, '_ctc_char_level', False):
    is_ipa_direct = True
  elif gezi.get('ctc_char_level') or gezi.get('s2s_ipa_chars'):
    is_ipa_direct = True
  # For phonetic track with seq2seq models (ctc_weight=0):
  # predictions are English text, need to convert to IPA
  if (FLAGS.score_metric == 'ipa_cer' 
      and getattr(FLAGS, 'ipa_method', '') == 'eng_to_ipa'
      and not is_ipa_direct):
    try:
      import eng_to_ipa as ipa
      all_preds = [ipa.convert(t.strip()).replace('*', '') for t in all_preds]
    except ImportError:
      logger.warning('eng_to_ipa not installed, skipping IPA conversion')
  t2 = time.time()

  # For NeMo backbone (tokenizer=None), label_texts are accumulated in gezi
  # during forward() passes.  Use them directly instead of decoding token IDs
  # (which would be all -100 placeholders).
  eval_label_texts = gz.get('eval_label_texts', None)
  if eval_label_texts and len(eval_label_texts) >= len(all_preds):
    all_targets = eval_label_texts[:len(all_preds)]
    gz.set('eval_label_texts', None)  # reset for next eval round
    logger.info(f'  [eval] Using {len(all_targets)} accumulated label_texts (tokenizer=None / NeMo path)')
  else:
    all_targets = _trim_and_decode(y_true)
    # Reset any partial accumulation
    if eval_label_texts:
      gz.set('eval_label_texts', None)
  t3 = time.time()

  df = pd.DataFrame({
    'pred': all_preds,
    'label': all_targets,
  })

  # Attach metadata columns from eval dataset for fine-grained metrics & analysis
  eval_orig = gz.get('eval_df')
  if eval_orig is not None and len(eval_orig) == len(df):
    for col in ['utterance_id', 'child_id', 'session_id', 'audio_path',
                'audio_duration_sec', 'age_bucket', 'source']:
      if col in eval_orig.columns:
        df[col] = eval_orig[col].values

  eval_dual_head_preds = gz.get('eval_dual_head_preds', None)
  if eval_dual_head_preds and len(eval_dual_head_preds) >= len(df):
    dual_df = pd.DataFrame(eval_dual_head_preds[:len(df)])
    for col in dual_df.columns:
      df[col] = dual_df[col].values
    gz.set('eval_dual_head_preds', None)
  elif eval_dual_head_preds:
    gz.set('eval_dual_head_preds', None)

  eval_primary_decode_meta = gz.get('eval_primary_decode_meta', None)
  if eval_primary_decode_meta and len(eval_primary_decode_meta) >= len(df):
    meta_df = pd.DataFrame(eval_primary_decode_meta[:len(df)])
    for col in meta_df.columns:
      df[col] = meta_df[col].values
    gz.set('eval_primary_decode_meta', None)
  elif eval_primary_decode_meta:
    gz.set('eval_primary_decode_meta', None)

  eval_word_pred_texts = gz.get('eval_word_pred_texts', None)
  eval_word_head_type = gz.get('eval_word_head_type', None)
  if eval_word_pred_texts and len(eval_word_pred_texts) >= len(df):
    word_texts = list(eval_word_pred_texts[:len(df)])
    df['pred_word'] = word_texts
    df['pred_word_len'] = [len(str(text)) for text in word_texts]
    df['pred_word_head_type'] = eval_word_head_type or ''
    df['pred_word_agree'] = (df['pred_word'] == df['pred']).astype(int)
    gz.set('eval_word_pred_texts', None)
  elif eval_word_pred_texts:
    gz.set('eval_word_pred_texts', None)

  # Always save eval.csv so that downstream analysis / save_best can use it
  eval_csv = f'{FLAGS.model_dir}/eval.csv'
  df.to_csv(eval_csv, index=False, escapechar='\\', quoting=csv.QUOTE_MINIMAL)
  logger.info(f'  Saved eval results to {eval_csv} ({len(df)} rows)')

  oof_file = f'{FLAGS.model_dir}/oof.pkl'
  df.to_pickle(oof_file)

  # Save CTC logprobs for offline ensemble (prob/logits averaging)
  eval_ctc_logprobs = gz.get('eval_ctc_logprobs', None)
  if eval_ctc_logprobs and len(eval_ctc_logprobs) >= len(df):
    import torch
    uid_col = df['utterance_id'].values if 'utterance_id' in df.columns else [f'sample_{i}' for i in range(len(df))]
    logprobs_dict = {}
    for i, uid in enumerate(uid_col[:len(eval_ctc_logprobs)]):
      logprobs_dict[uid] = eval_ctc_logprobs[i]
    lp_path = f'{FLAGS.model_dir}/ctc_logprobs.pt'
    torch.save(logprobs_dict, lp_path)
    size_mb = os.path.getsize(lp_path) / 1e6
    logger.info(f'  Saved CTC logprobs to {lp_path} ({len(logprobs_dict)} utterances, {size_mb:.1f} MB)')
    gz.set('eval_ctc_logprobs', None)  # reset for next eval round

  eval_ctc_logprobs_word = gz.get('eval_ctc_logprobs_word', None)
  if eval_ctc_logprobs_word and len(eval_ctc_logprobs_word) >= len(df):
    import torch
    uid_col_word_lp = df['utterance_id'].values if 'utterance_id' in df.columns else [f'sample_{i}' for i in range(len(df))]
    word_logprobs_dict = {}
    for i, uid in enumerate(uid_col_word_lp[:len(eval_ctc_logprobs_word)]):
      word_logprobs_dict[uid] = eval_ctc_logprobs_word[i]
    word_lp_path = f'{FLAGS.model_dir}/ctc_logprobs_word.pt'
    torch.save({
        'logprobs': word_logprobs_dict,
        'meta': {'head_type': eval_word_head_type or ''},
    }, word_lp_path)
    size_mb = os.path.getsize(word_lp_path) / 1e6
    logger.info(
        f'  Saved word CTC logprobs to {word_lp_path} '
        f'({len(word_logprobs_dict)} utterances, {size_mb:.1f} MB)')
    gz.set('eval_ctc_logprobs_word', None)
  elif eval_ctc_logprobs_word:
    gz.set('eval_ctc_logprobs_word', None)

  dual_cols = {'pred_ctc', 'pred_tdt', 'pred_tdt_score', 'pred_primary', 'pred_primary_method',
               'pred_heads_agree', 'pred_ctc_len', 'pred_tdt_len', 'pred_dual_len_gap'}
  dual_export_cols = [c for c in df.columns if c in dual_cols]
  if dual_export_cols:
    import torch
    uid_col_dual = df['utterance_id'].values if 'utterance_id' in df.columns else [f'sample_{i}' for i in range(len(df))]
    dual_pred_dict = {}
    for i, uid in enumerate(uid_col_dual):
      dual_pred_dict[uid] = {col: df.iloc[i][col] for col in dual_export_cols}
    dual_path = f'{FLAGS.model_dir}/dual_head_preds.pt'
    torch.save({
        'preds': dual_pred_dict,
        'meta': {
            'columns': dual_export_cols,
            'decode_method': getattr(FLAGS, 'decode_method', 'auto'),
        }
    }, dual_path)
    size_mb = os.path.getsize(dual_path) / 1e6
    logger.info(f'  Saved dual-head preds to {dual_path} ({len(dual_pred_dict)} utterances, {size_mb:.1f} MB)')

  word_pred_cols = {'pred_word', 'pred_word_len', 'pred_word_head_type', 'pred_word_agree'}
  word_export_cols = [c for c in df.columns if c in word_pred_cols]
  if word_export_cols:
    import torch
    uid_col_word_pred = df['utterance_id'].values if 'utterance_id' in df.columns else [f'sample_{i}' for i in range(len(df))]
    word_pred_dict = {}
    for i, uid in enumerate(uid_col_word_pred):
      word_pred_dict[uid] = {col: df.iloc[i][col] for col in word_export_cols}
    word_pred_path = f'{FLAGS.model_dir}/word_head_preds.pt'
    torch.save({
        'preds': word_pred_dict,
        'meta': {
            'columns': word_export_cols,
            'head_type': eval_word_head_type or '',
        }
    }, word_pred_path)
    size_mb = os.path.getsize(word_pred_path) / 1e6
    logger.info(f'  Saved word-head preds to {word_pred_path} ({len(word_pred_dict)} utterances, {size_mb:.1f} MB)')

  if eval_word_head_type is not None:
    gz.set('eval_word_head_type', None)

  # Save auxiliary meta predictions for tree reranker (aux_meta_preds.pt)
  _aux_age_raw = gz.get('eval_aux_age_logits', None)
  _aux_dom_raw = gz.get('eval_aux_domain_logits', None)
  _aux_nchars_raw = gz.get('eval_aux_nchars_pred', None)
  _aux_nspaces_raw = gz.get('eval_aux_nspaces_pred', None)
  if _aux_age_raw or _aux_dom_raw or _aux_nchars_raw or _aux_nspaces_raw:
    import torch as _torch_save
    uid_col_aux = df['utterance_id'].values if 'utterance_id' in df.columns else [f'sample_{i}' for i in range(len(df))]
    aux_meta_preds = {}
    if _aux_age_raw:
      age_all = _torch_save.cat(_aux_age_raw, dim=0)[:len(df)].float()
    if _aux_dom_raw:
      dom_all = _torch_save.cat(_aux_dom_raw, dim=0)[:len(df)].float()
    if _aux_nchars_raw:
      nchars_all = _torch_save.cat(_aux_nchars_raw, dim=0)[:len(df)].float()
    if _aux_nspaces_raw:
      nspaces_all = _torch_save.cat(_aux_nspaces_raw, dim=0)[:len(df)].float()
    for i, uid in enumerate(uid_col_aux[:len(df)]):
      pred = {}
      if _aux_age_raw and i < len(age_all):
        pred['age_logits'] = age_all[i].cpu().numpy()
      if _aux_dom_raw and i < len(dom_all):
        pred['domain_logit'] = float(dom_all[i].cpu())
      if _aux_nchars_raw and i < len(nchars_all):
        pred['nchars_pred'] = float(nchars_all[i].cpu())
      if _aux_nspaces_raw and i < len(nspaces_all):
        pred['nspaces_pred'] = float(nspaces_all[i].cpu())
      if pred:
        aux_meta_preds[uid] = pred
    if aux_meta_preds:
      age_mode = getattr(FLAGS, 'aux_age_mode', None)
      has_age = _aux_age_raw is not None and len(_aux_age_raw) > 0
      has_domain = _aux_dom_raw is not None and len(_aux_dom_raw) > 0
      has_nchars = _aux_nchars_raw is not None and len(_aux_nchars_raw) > 0
      has_nspaces = _aux_nspaces_raw is not None and len(_aux_nspaces_raw) > 0
      meta_info = {
          'age_mode': age_mode,
          'has_age': has_age,
          'has_domain': has_domain,
          'has_nchars': has_nchars,
          'has_nspaces': has_nspaces,
      }
      meta_path = f'{FLAGS.model_dir}/aux_meta_preds.pt'
      _torch_save.save({'preds': aux_meta_preds, 'meta': meta_info}, meta_path)
      size_mb = os.path.getsize(meta_path) / 1e6
      n_age = sum(1 for v in aux_meta_preds.values() if 'age_logits' in v)
      n_dom = sum(1 for v in aux_meta_preds.values() if 'domain_logit' in v)
      n_nchars = sum(1 for v in aux_meta_preds.values() if 'nchars_pred' in v)
      n_nspaces = sum(1 for v in aux_meta_preds.values() if 'nspaces_pred' in v)
      logger.info(
          f'  Saved aux meta preds to {meta_path} '
          f'({n_age} age, {n_dom} domain, {n_nchars} nchars, {n_nspaces} nspaces, {size_mb:.1f} MB)')

  t4 = time.time()
  metrics = eval_df(df)
  t5 = time.time()

  # ---- Auxiliary metadata metrics (age accuracy / domain AUC) ----
  import torch as _torch
  eval_aux_age = gz.get('eval_aux_age_logits', None)
  if eval_aux_age and len(eval_aux_age) > 0:
    try:
      age_logits = _torch.cat(eval_aux_age, dim=0)[:len(df)].float()
      gz.set('eval_aux_age_logits', None)
      if 'age_bucket' in df.columns:
        # 4-class: 3-4=0, 5-7=1, 8-11=2, 12+=3
        age_fine = _age_fine_group_series(df['age_bucket'])
        _AGE_CLASS_MAP = {'3-4': 0, '5-7': 1, '8-11': 2, '12+': 3}
        valid_mask = age_fine.isin(_AGE_CLASS_MAP.keys())
        if valid_mask.sum() > 0:
          age_true = age_fine[valid_mask].map(_AGE_CLASS_MAP).values
          age_logits_valid = age_logits[valid_mask.values]
          age_mode = getattr(FLAGS, 'aux_age_mode', 'classify')
          if age_mode == 'classify':
            age_pred = age_logits_valid.argmax(dim=-1).numpy()
          elif age_mode == 'ordinal':
            # Ordinal: count exceeded thresholds → class index
            age_pred = (_torch.sigmoid(age_logits_valid) > 0.5).sum(dim=-1).numpy()
          else:  # regress
            # Denormalize from [0,1] back to age midpoints: x * 9.5 + 3.5
            vals = age_logits_valid.squeeze(-1).numpy() * 9.5 + 3.5
            age_pred = np.digitize(vals, bins=[4.75, 7.75, 11.25])
          age_acc = (age_pred == age_true).mean()
          metrics['aux/age_acc'] = float(age_acc)
          
          # MAE: use representative midpoint ages for each class
          _AGE_MIDPOINTS = np.array([3.5, 6.0, 9.5, 13.0])
          age_true_mid = _AGE_MIDPOINTS[age_true]
          if age_mode == 'classify':
            # Soft prediction: probability-weighted midpoint age
            probs = _torch.softmax(age_logits_valid, dim=-1).numpy()  # (N, 4)
            age_pred_mid = probs @ _AGE_MIDPOINTS
          elif age_mode == 'ordinal':
            # Expected class from cumulative sigmoid, then midpoint
            cum_probs = _torch.sigmoid(age_logits_valid).numpy()  # (N, 3)
            expected_class = cum_probs.sum(axis=-1)  # continuous in [0, 3]
            age_pred_mid = np.interp(expected_class, np.arange(4), _AGE_MIDPOINTS)
          else:  # regress
            # Denormalize from [0,1] back to age midpoints: x * 9.5 + 3.5
            age_pred_mid = age_logits_valid.squeeze(-1).numpy() * 9.5 + 3.5
          age_mae = np.abs(age_pred_mid - age_true_mid).mean()
          metrics['aux/age_mae'] = float(age_mae)
          logger.info(f'  [aux] age acc: {age_acc:.4f}, mae: {age_mae:.3f} ({valid_mask.sum()} samples, 4-class)')
    except Exception as e:
      logger.warning(f'  [aux] age eval failed: {e}')
      gz.set('eval_aux_age_logits', None)

  eval_aux_domain = gz.get('eval_aux_domain_logits', None)
  if eval_aux_domain and len(eval_aux_domain) > 0:
    try:
      domain_logits = _torch.cat(eval_aux_domain, dim=0)[:len(df)]
      gz.set('eval_aux_domain_logits', None)
      if 'source' in df.columns:
        domain_true = (df['source'] == 'dd').astype(int).values
        domain_probs = _torch.sigmoid(domain_logits).float().numpy().flatten()
        # AUC requires both classes
        if len(set(domain_true)) > 1:
          from sklearn.metrics import roc_auc_score
          domain_auc = roc_auc_score(domain_true, domain_probs)
          metrics['aux/domain_auc'] = float(domain_auc)
          logger.info(f'  [aux] domain AUC: {domain_auc:.4f} ({len(domain_true)} samples)')
        else:
          # Only one class in eval set: compute accuracy instead
          domain_pred = (domain_probs > 0.5).astype(int)
          domain_acc = (domain_pred == domain_true).mean()
          metrics['aux/domain_acc'] = float(domain_acc)
    except Exception as e:
      logger.warning(f'  [aux] domain eval failed: {e}')
      gz.set('eval_aux_domain_logits', None)

  # ---- aux nchars / nspaces MAE ----
  import math as _math
  for aux_name, label_col in [('nchars', 'label'), ('nspaces', 'label')]:
    eval_preds = gz.get(f'eval_aux_{aux_name}_pred', None)
    if eval_preds and len(eval_preds) > 0:
      try:
        preds = _torch.cat(eval_preds, dim=0)[:len(df)].float().numpy()
        gz.set(f'eval_aux_{aux_name}_pred', None)
        # Compute ground truth from label text
        labels = df[label_col].values
        if aux_name == 'nchars':
          gt = [_math.log1p(len(str(t))) for t in labels]
        else:  # nspaces
          gt = [_math.log1p(str(t).count(' ')) for t in labels]
        import numpy as _np
        gt = _np.array(gt, dtype=_np.float32)
        mae = float(_np.mean(_np.abs(preds - gt)))
        metrics[f'aux/{aux_name}_mae'] = mae
        logger.info(f'  [aux] {aux_name} MAE: {mae:.4f} (log1p scale, {len(gt)} samples)')
      except Exception as e:
        logger.warning(f'  [aux] {aux_name} eval failed: {e}')
        gz.set(f'eval_aux_{aux_name}_pred', None)

  if gz.get('timer'):
    metrics['elapsed'] = gz.get('timer').elapsed_minutes()

  # show examples
  n_show = min(3, len(all_preds))
  for i in range(n_show):
    logger.info(f'  [pred] {all_preds[i][:120]}')
    logger.info(f'  [gold] {all_targets[i][:120]}')
    if FLAGS.score_metric == 'ipa_cer':
      logger.info(f'  [pred_norm] {normalize_ipa(all_preds[i])[:120]}')
      logger.info(f'  [gold_norm] {normalize_ipa(all_targets[i])[:120]}')
    logger.info('')

  logger.info(f'  [eval timing] to_2d={t1-t0:.1f}s  decode_pred={t2-t1:.1f}s  '
              f'decode_target={t3-t2:.1f}s  score={t5-t4:.1f}s  '
              f'total={t5-t0:.1f}s  (n={len(all_preds)})')

  return metrics
