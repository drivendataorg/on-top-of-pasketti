#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   ctc_decode.py
#        \author   chenghuige
#          \date   2025-02-17
#   \Description   CTC prefix beam search + optional n-gram LM.
#                  Pure Python/PyTorch — no external C dependencies (kenlm etc.)
#                  so it can be bundled in a DrivenData submission zip.
#
#  Usage:
#    # Greedy (default, same as argmax + collapse)
#    texts = ctc_decode(log_probs, blank=0, beam_width=1)
#
#    # Beam search without LM
#    texts = ctc_decode(log_probs, blank=0, beam_width=10, id_to_char=ID_TO_CHAR)
#
#    # Beam search with character n-gram LM
#    lm = CharNgramLM.load('lm.json')
#    texts = ctc_decode(log_probs, blank=0, beam_width=10,
#                       id_to_char=ID_TO_CHAR, lm=lm, lm_weight=0.5)
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import json
import math
from collections import defaultdict

import torch


# =====================================================================
#  Character-level N-gram Language Model (pure Python, JSON serialised)
# =====================================================================

class CharNgramLM:
  """Simple smoothed character n-gram LM for IPA sequences.
  
  Uses stupid-backoff (Brants et al. 2007) — no normalisation needed,
  just stores raw log-probabilities at each n-gram order and backs off
  with a penalty factor (default 0.4).
  
  Model file is a small JSON dict:
    { "order": 5,
      "backoff": 0.4,
      "vocab": ["a", "b", ...],
      "counts": { "": {"a": 100, "b": 50, ...},        # unigram
                  "a": {"b": 30, ...},                   # bigram given "a"
                  "ab": {"c": 10, ...},                  # trigram given "ab"
                  ... } }
  """

  def __init__(self, order=5, backoff_weight=0.4):
    self.order = order
    self.backoff_weight = backoff_weight
    self.vocab = set()
    # counts[context_str] = {next_char: count}
    self.counts = defaultdict(lambda: defaultdict(int))
    self._log_probs = {}  # cached: context -> {char: log_prob}
    self._total = {}      # cached: context -> total_count

  def train(self, texts):
    """Train on a list of IPA text strings."""
    for text in texts:
      # Pad with BOS markers
      padded = '^' * (self.order - 1) + text + '$'
      for ch in text:
        self.vocab.add(ch)
      for i in range(len(padded) - 1):
        for n in range(1, self.order + 1):
          if i - n + 1 < 0:
            continue
          context = padded[i - n + 1: i]
          next_ch = padded[i]
          self.counts[context][next_ch] += 1
    self._build_cache()

  def _build_cache(self):
    """Pre-compute log probabilities from counts."""
    self._log_probs = {}
    self._total = {}
    for ctx, char_counts in self.counts.items():
      total = sum(char_counts.values())
      self._total[ctx] = total
      self._log_probs[ctx] = {
          ch: math.log(cnt / total) for ch, cnt in char_counts.items()
      }

  def score(self, context, next_char):
    """Return log probability of next_char given context string.
    Uses stupid backoff: if full context not found, shorten by 1 and
    multiply probability by backoff_weight."""
    # Try from longest context down to empty
    for k in range(min(len(context), self.order - 1), -1, -1):
      ctx = context[-k:] if k > 0 else ''
      if ctx in self._log_probs and next_char in self._log_probs[ctx]:
        # Apply backoff penalty for each level we backed off
        backed_off = min(len(context), self.order - 1) - k
        penalty = math.log(self.backoff_weight) * backed_off if backed_off > 0 else 0.0
        return self._log_probs[ctx][next_char] + penalty
    # Unknown character — return a very low score
    return -20.0

  def save(self, path):
    """Save LM to JSON file."""
    data = {
        'type': 'char',
        'order': self.order,
        'backoff': self.backoff_weight,
        'vocab': sorted(self.vocab),
        'counts': {ctx: dict(cc) for ctx, cc in self.counts.items()},
    }
    with open(path, 'w', encoding='utf-8') as f:
      json.dump(data, f, ensure_ascii=False, indent=1)

  @classmethod
  def load(cls, path):
    """Load LM from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
      data = json.load(f)
    lm = cls(order=data['order'], backoff_weight=data.get('backoff', 0.4))
    lm.vocab = set(data['vocab'])
    lm.counts = defaultdict(lambda: defaultdict(int))
    for ctx, cc in data['counts'].items():
      for ch, cnt in cc.items():
        lm.counts[ctx][ch] = cnt
    lm._build_cache()
    return lm


class WordNgramLM:
  """Simple smoothed word n-gram LM for sentence scoring.

  Stores counts keyed by token context and supports both whole-sentence scoring
  for reranker features and a coarse streaming `score(context, next_char)` API so
  existing shallow-fusion call sites remain compatible.
  """

  def __init__(self, order=3, backoff_weight=0.4):
    self.order = order
    self.backoff_weight = backoff_weight
    self.vocab = set()
    self.counts = defaultdict(lambda: defaultdict(int))
    self._log_probs = {}
    self._total = {}
    self._backoff = {}

  def _ctx_key(self, tokens):
    return '\t'.join(tokens)

  def _split(self, text):
    text = ' '.join(str(text or '').strip().split())
    return text.split(' ') if text else []

  def train(self, texts):
    for text in texts:
      tokens = self._split(text)
      padded = ['<s>'] * (self.order - 1) + tokens + ['</s>']
      for tok in tokens:
        self.vocab.add(tok)
      for i in range(len(padded)):
        for n in range(1, self.order + 1):
          if i - n + 1 < 0:
            continue
          context = padded[i - n + 1:i]
          next_tok = padded[i]
          self.counts[self._ctx_key(context)][next_tok] += 1
    self._build_cache()

  def _build_cache(self):
    self._log_probs = {}
    self._total = {}
    for ctx, token_counts in self.counts.items():
      total = sum(token_counts.values())
      self._total[ctx] = total
      self._log_probs[ctx] = {
          tok: math.log(cnt / total) for tok, cnt in token_counts.items()
      }

  def score_tokens(self, context_tokens, next_token):
    max_k = min(len(context_tokens), self.order - 1)
    arpa_backoff = 0.0
    for k in range(max_k, -1, -1):
      ctx = self._ctx_key(context_tokens[-k:] if k > 0 else [])
      if ctx in self._log_probs and next_token in self._log_probs[ctx]:
        if self._backoff:
          return self._log_probs[ctx][next_token] + arpa_backoff
        backed_off = max_k - k
        penalty = math.log(self.backoff_weight) * backed_off if backed_off > 0 else 0.0
        return self._log_probs[ctx][next_token] + penalty
      if self._backoff and k > 0:
        arpa_backoff += self._backoff.get(ctx, 0.0)
    return -20.0

  def score_text(self, text):
    tokens = self._split(text)
    context = ['<s>'] * (self.order - 1)
    score = 0.0
    for tok in tokens:
      score += self.score_tokens(context, tok)
      context.append(tok)
    score += self.score_tokens(context, '</s>')
    return score

  def score(self, context, next_char):
    """Coarse streaming adapter for char-based shallow fusion call sites."""
    context = str(context or '')
    if next_char == '$':
      stripped = context.strip()
      if not stripped:
        return self.score_tokens(['<s>'] * (self.order - 1), '</s>')
      if context.endswith(' '):
        prev_tokens = self._split(stripped)
        return self.score_tokens(prev_tokens, '</s>')
      tokens = self._split(stripped)
      current = tokens[-1]
      prev_tokens = ['<s>'] * (self.order - 1) + tokens[:-1]
      return self.score_tokens(prev_tokens, current) + self.score_tokens(prev_tokens + [current], '</s>')
    if next_char != ' ':
      return 0.0
    stripped = context.strip()
    if not stripped or context.endswith(' '):
      return 0.0
    tokens = self._split(stripped)
    current = tokens[-1]
    prev_tokens = ['<s>'] * (self.order - 1) + tokens[:-1]
    return self.score_tokens(prev_tokens, current)

  def save(self, path):
    data = {
        'type': 'word',
        'order': self.order,
        'backoff': self.backoff_weight,
        'vocab': sorted(self.vocab),
        'counts': {ctx: dict(cc) for ctx, cc in self.counts.items()},
    }
    with open(path, 'w', encoding='utf-8') as f:
      json.dump(data, f, ensure_ascii=False, indent=1)

  @classmethod
  def load(cls, path):
    with open(path, 'r', encoding='utf-8') as f:
      data = json.load(f)
    lm = cls(order=data['order'], backoff_weight=data.get('backoff', 0.4))
    lm.vocab = set(data['vocab'])
    lm.counts = defaultdict(lambda: defaultdict(int))
    for ctx, cc in data['counts'].items():
      for tok, cnt in cc.items():
        lm.counts[ctx][tok] = cnt
    lm._build_cache()
    return lm

  @classmethod
  def load_arpa(cls, path):
    lm = cls(order=1, backoff_weight=0.4)
    lm.counts = defaultdict(lambda: defaultdict(int))
    lm._log_probs = {}
    lm._total = {}
    lm._backoff = {}

    current_order = 0
    max_order = 1
    with _open_lm_text(path) as f:
      for raw_line in f:
        line = raw_line.strip()
        if not line:
          continue
        if line.startswith('ngram '):
          try:
            order_value = int(line.split()[1].split('=')[0])
            max_order = max(max_order, order_value)
          except Exception:
            continue
          continue
        if line.startswith('\\end\\'):
          break
        if line.startswith('\\') and line.endswith('-grams:'):
          try:
            current_order = int(line[1:].split('-', 1)[0])
          except Exception as exc:
            raise ValueError(f'Invalid ARPA section header: {line}') from exc
          max_order = max(max_order, current_order)
          continue
        if line.startswith('\\'):
          current_order = 0
          continue
        if current_order <= 0:
          continue

        parts = line.split()
        if len(parts) < current_order + 1:
          continue

        log10_prob = float(parts[0])
        tokens = parts[1:1 + current_order]
        backoff_log10 = float(parts[1 + current_order]) if len(parts) > current_order + 1 else None
        ctx_tokens = tokens[:-1]
        next_token = tokens[-1]
        ctx_key = lm._ctx_key(ctx_tokens)
        lm._log_probs.setdefault(ctx_key, {})[next_token] = log10_prob * math.log(10.0)
        if next_token not in ('<s>', '</s>'):
          lm.vocab.add(next_token)
        if backoff_log10 is not None:
          full_key = lm._ctx_key(tokens)
          lm._backoff[full_key] = backoff_log10 * math.log(10.0)

    lm.order = max_order
    return lm


def _open_lm_text(path):
  if str(path).lower().endswith('.gz'):
    return gzip.open(path, 'rt', encoding='utf-8')
  return open(path, 'r', encoding='utf-8')


def _sniff_lm_format(path):
  with _open_lm_text(path) as f:
    for raw_line in f:
      line = raw_line.strip()
      if not line:
        continue
      if line.startswith('{') or line.startswith('['):
        return 'json'
      if line.startswith('\\data\\') or line.startswith('ngram '):
        return 'arpa'
      break
  raise ValueError(f'Unable to determine LM format for: {path}')


def load_ngram_lm(path):
  lm_format = _sniff_lm_format(path)
  if lm_format == 'arpa':
    return WordNgramLM.load_arpa(path)

  with _open_lm_text(path) as f:
    data = json.load(f)
  lm_type = data.get('type', 'char')
  if lm_type == 'word':
    return WordNgramLM.load(path)
  return CharNgramLM.load(path)


# =====================================================================
#  CTC Prefix Beam Search
# =====================================================================

def _greedy_decode_batch(log_probs, blank):
  """Fast greedy CTC decode (argmax + collapse). Returns list of id-lists.
  
  Uses single .cpu().numpy() transfer + numpy vectorized ops instead of
  per-element .item() calls. Eliminates ~1500*B GPU syncs per batch.
  """
  import numpy as np
  pred_ids = log_probs.argmax(dim=-1).cpu().numpy()  # (B, T) — one transfer
  results = []
  for i in range(pred_ids.shape[0]):
    seq = pred_ids[i]
    # Vectorized: find positions where value changes
    mask = np.empty(len(seq), dtype=bool)
    mask[0] = True
    mask[1:] = seq[1:] != seq[:-1]
    collapsed = seq[mask]
    results.append([int(c) for c in collapsed if c != blank])
  return results


# ---- fast log-add for scalars and arrays ----
_NEG_INF = float('-inf')

def _log_add(a, b):
  """Numerically stable log(exp(a) + exp(b))."""
  if a == _NEG_INF:
    return b
  if b == _NEG_INF:
    return a
  if a > b:
    return a + math.log1p(math.exp(b - a))
  else:
    return b + math.log1p(math.exp(a - b))


def _prefix_beam_search_np(log_probs_np, blank, beam_width,
                            topk_ids=None, topk_vals=None,
                            id_to_char=None, lm=None, lm_weight=0.0):
  """High-performance CTC prefix beam search.
  
  Optimised for small vocab (IPA, V~53) with beam_width 3-10.
  All hot-path operations inlined to avoid Python function-call overhead.
  
  log_probs_np: (T, V) numpy float32 array of log probabilities
  topk_ids:     (T, K) pre-computed top-K indices (optional)
  topk_vals:    (T, K) pre-computed top-K values  (optional)
  Returns: list of character IDs (best hypothesis)
  """
  T, V = log_probs_np.shape
  _inf = float('-inf')
  _log1p = math.log1p
  _exp = math.exp
  use_lm = lm is not None and lm_weight > 0 and id_to_char is not None
  # Pre-build id→char list for fast lookup in LM scoring
  if use_lm:
    _lm_score = lm.score  # cache method reference
    _lm_w = lm_weight
    # Build prefix→context_string cache to avoid repeated join
    _ctx_cache = {}
    def _get_ctx(prefix):
      if prefix in _ctx_cache:
        return _ctx_cache[prefix]
      ctx = ''.join(id_to_char.get(c, '') for c in prefix)
      _ctx_cache[prefix] = ctx
      return ctx

  # Pre-convert entire log_probs to Python list-of-lists for fastest indexing
  # (avoids numpy scalar creation overhead in the inner loop)
  lp_lists = log_probs_np.tolist()  # list[T] of list[V] of float

  # Determine per-frame candidate lists — one-time pre-computation.
  # For small vocab (V <= beam*8), iterate all with dynamic pruning.
  # For larger vocab, use pre-computed top-K.
  use_topk = topk_ids is not None
  if not use_topk and V > beam_width * 8:
    import numpy as np
    K = min(beam_width * 4, V)
    part_idx = np.argpartition(-log_probs_np, K, axis=-1)[:, :K]
    rows = np.arange(T)[:, None]
    vals = log_probs_np[rows, part_idx]
    sort_idx = np.argsort(-vals, axis=-1)
    topk_ids = np.take_along_axis(part_idx, sort_idx, axis=-1)
    topk_vals = np.take_along_axis(vals, sort_idx, axis=-1)
    use_topk = True

  # Pre-compute per-frame filtered candidate list: list of (cid, clp) tuples.
  # Uses dynamic threshold: keep tokens within 5 log-units of frame max.
  # This moves ALL pruning logic out of the inner beam loop.
  PROB_DELTA = 5.0
  frame_cands = []  # list[T] of list[(cid, clp)]
  frame_blps = []   # list[T] of float
  if use_topk:
    tk_ids_lists = topk_ids.tolist()
    tk_vals_lists = topk_vals.tolist()
    for t in range(T):
      frame_blps.append(lp_lists[t][blank])
      ids_t = tk_ids_lists[t]
      vals_t = tk_vals_lists[t]
      mx = vals_t[0] if vals_t else -999.0
      thresh = mx - PROB_DELTA
      cands = []
      for ki in range(len(ids_t)):
        cid = ids_t[ki]
        clp = vals_t[ki]
        if clp < thresh:
          break  # sorted, rest are smaller
        if cid != blank:
          cands.append((cid, clp))
      frame_cands.append(cands)
  else:
    for t in range(T):
      frame = lp_lists[t]
      frame_blps.append(frame[blank])
      mx = max(frame)
      thresh = mx - PROB_DELTA
      cands = []
      for cid in range(V):
        if cid == blank:
          continue
        clp = frame[cid]
        if clp >= thresh:
          cands.append((cid, clp))
      frame_cands.append(cands)

  # Beam state: dict  prefix_tuple -> [p_blank, p_non_blank]
  beams = {(): [0.0, _inf]}

  for t in range(T):
    blp = frame_blps[t]
    cands = frame_cands[t]
    new_beams = {}  # regular dict (no defaultdict lambda overhead)

    for prefix, scores in beams.items():
      p_b = scores[0]
      p_nb = scores[1]

      # Inline log_add for p_total
      if p_b == _inf:
        p_total = p_nb
      elif p_nb == _inf:
        p_total = p_b
      elif p_b > p_nb:
        p_total = p_b + _log1p(_exp(p_nb - p_b))
      else:
        p_total = p_nb + _log1p(_exp(p_b - p_nb))

      # --- extend with blank ---
      val_b = p_total + blp
      if prefix in new_beams:
        ob = new_beams[prefix][0]
        # Inline log_add
        if ob == _inf:
          new_beams[prefix][0] = val_b
        elif val_b == _inf:
          pass
        elif ob > val_b:
          new_beams[prefix][0] = ob + _log1p(_exp(val_b - ob))
        else:
          new_beams[prefix][0] = val_b + _log1p(_exp(ob - val_b))
      else:
        new_beams[prefix] = [val_b, _inf]

      # --- extend with non-blank candidates (pre-filtered) ---
      last = prefix[-1] if prefix else -1

      for cid, clp in cands:
        if cid == last:
          # Repeated char: extend from blank only (new label)
          new_pf = prefix + (cid,)
          # LM bonus for emitting this char as a new label token
          if use_lm:
            ch_str = id_to_char.get(cid, '')
            ctx_str = _get_ctx(prefix)
            lm_bonus = _lm_w * _lm_score(ctx_str, ch_str)
          else:
            lm_bonus = 0.0
          val = p_b + clp + lm_bonus
          if new_pf in new_beams:
            onb = new_beams[new_pf][1]
            if onb == _inf:
              new_beams[new_pf][1] = val
            elif val > onb:
              new_beams[new_pf][1] = val + _log1p(_exp(onb - val))
            else:
              new_beams[new_pf][1] = onb + _log1p(_exp(val - onb))
          else:
            new_beams[new_pf] = [_inf, val]
          # Continue existing char (no new char appended)
          val2 = p_nb + clp
          entry = new_beams[prefix]
          onb2 = entry[1]
          if onb2 == _inf:
            entry[1] = val2
          elif val2 > onb2:
            entry[1] = val2 + _log1p(_exp(onb2 - val2))
          else:
            entry[1] = onb2 + _log1p(_exp(val2 - onb2))
        else:
          new_pf = prefix + (cid,)
          # LM bonus for emitting a new (different) character
          if use_lm:
            ch_str = id_to_char.get(cid, '')
            ctx_str = _get_ctx(prefix)
            lm_bonus = _lm_w * _lm_score(ctx_str, ch_str)
          else:
            lm_bonus = 0.0
          val = p_total + clp + lm_bonus
          if new_pf in new_beams:
            onb = new_beams[new_pf][1]
            if onb == _inf:
              new_beams[new_pf][1] = val
            elif val > onb:
              new_beams[new_pf][1] = val + _log1p(_exp(onb - val))
            else:
              new_beams[new_pf][1] = onb + _log1p(_exp(val - onb))
          else:
            new_beams[new_pf] = [_inf, val]

    # Prune to beam_width
    if len(new_beams) > beam_width:
      scored = []
      for pf, (pb, pnb) in new_beams.items():
        if pb == _inf:
          total = pnb
        elif pnb == _inf:
          total = pb
        elif pb > pnb:
          total = pb + _log1p(_exp(pnb - pb))
        else:
          total = pnb + _log1p(_exp(pb - pnb))
        scored.append((total, pf, pb, pnb))
      scored.sort(reverse=True)
      new_beams = {s[1]: [s[2], s[3]] for s in scored[:beam_width]}

    beams = new_beams
    # Periodically clear LM context cache to prevent memory bloat
    if use_lm and t % 50 == 0:
      _ctx_cache.clear()

  # Score all final hypotheses
  scored_hyps = []
  for pf, (pb, pnb) in beams.items():
    if pb == _inf:
      sc = pnb
    elif pnb == _inf:
      sc = pb
    elif pb > pnb:
      sc = pb + _log1p(_exp(pnb - pb))
    else:
      sc = pnb + _log1p(_exp(pb - pnb))
    scored_hyps.append((sc, pf))
  scored_hyps.sort(reverse=True)

  if not scored_hyps:
    return []
  return list(scored_hyps[0][1])


def prefix_beam_search_nbest(log_probs_np, blank, beam_width, nbest=5,
                              id_to_char=None, lm=None, lm_weight=0.0):
  """CTC prefix beam search returning top-N hypotheses with scores.

  Same algorithm as _prefix_beam_search_np but returns multiple candidates.

  Args:
    log_probs_np: (T, V) numpy float32 log-probabilities.
    blank: blank token id.
    beam_width: beam size (should be >= nbest).
    nbest: number of top hypotheses to return.
    id_to_char: optional dict for converting ids to chars.
    lm, lm_weight: optional LM.

  Returns:
    list of (score, token_id_list) tuples, sorted by descending score.
    If id_to_char is given, returns (score, text_string) tuples instead.
  """
  import numpy as np
  if not isinstance(log_probs_np, np.ndarray):
    log_probs_np = np.asarray(log_probs_np, dtype=np.float32)
  if log_probs_np.dtype != np.float32:
    log_probs_np = log_probs_np.astype(np.float32)

  # Use wider beam to get better nbest diversity
  actual_beam = max(beam_width, nbest * 2)

  # Run the standard beam search (which now builds scored_hyps internally)
  # We re-use the core logic but need the full beam — call internal fn with wider beam
  # and then extract top-N from the scored_hyps.
  # To avoid code duplication, we replicate the final scoring from _prefix_beam_search_np
  # by running it with the wider beam and extracting the beam state.

  T, V = log_probs_np.shape
  _inf = float('-inf')
  _log1p = math.log1p
  _exp = math.exp
  use_lm = lm is not None and lm_weight > 0 and id_to_char is not None

  if use_lm:
    _lm_score = lm.score
    _lm_w = lm_weight
    _ctx_cache = {}
    def _get_ctx(prefix):
      if prefix in _ctx_cache:
        return _ctx_cache[prefix]
      ctx = ''.join(id_to_char.get(c, '') for c in prefix)
      _ctx_cache[prefix] = ctx
      return ctx

  lp_lists = log_probs_np.tolist()

  # Per-frame candidate pre-computation
  PROB_DELTA = 5.0
  frame_cands = []
  frame_blps = []
  for t in range(T):
    frame = lp_lists[t]
    frame_blps.append(frame[blank])
    mx = max(frame)
    thresh = mx - PROB_DELTA
    cands = []
    for cid in range(V):
      if cid == blank:
        continue
      clp = frame[cid]
      if clp >= thresh:
        cands.append((cid, clp))
    frame_cands.append(cands)

  beams = {(): [0.0, _inf]}

  for t in range(T):
    blp = frame_blps[t]
    cands = frame_cands[t]
    new_beams = {}

    for prefix, scores in beams.items():
      p_b = scores[0]
      p_nb = scores[1]

      if p_b == _inf:
        p_total = p_nb
      elif p_nb == _inf:
        p_total = p_b
      elif p_b > p_nb:
        p_total = p_b + _log1p(_exp(p_nb - p_b))
      else:
        p_total = p_nb + _log1p(_exp(p_b - p_nb))

      val_b = p_total + blp
      if prefix in new_beams:
        ob = new_beams[prefix][0]
        if ob == _inf:
          new_beams[prefix][0] = val_b
        elif val_b == _inf:
          pass
        elif ob > val_b:
          new_beams[prefix][0] = ob + _log1p(_exp(val_b - ob))
        else:
          new_beams[prefix][0] = val_b + _log1p(_exp(ob - val_b))
      else:
        new_beams[prefix] = [val_b, _inf]

      last = prefix[-1] if prefix else -1

      for cid, clp in cands:
        if cid == last:
          new_pf = prefix + (cid,)
          if use_lm:
            ch_str = id_to_char.get(cid, '')
            ctx_str = _get_ctx(prefix)
            lm_bonus = _lm_w * _lm_score(ctx_str, ch_str)
          else:
            lm_bonus = 0.0
          val = p_b + clp + lm_bonus
          if new_pf in new_beams:
            onb = new_beams[new_pf][1]
            if onb == _inf:
              new_beams[new_pf][1] = val
            elif val > onb:
              new_beams[new_pf][1] = val + _log1p(_exp(onb - val))
            else:
              new_beams[new_pf][1] = onb + _log1p(_exp(val - onb))
          else:
            new_beams[new_pf] = [_inf, val]
          val2 = p_nb + clp
          entry = new_beams[prefix]
          onb2 = entry[1]
          if onb2 == _inf:
            entry[1] = val2
          elif val2 > onb2:
            entry[1] = val2 + _log1p(_exp(onb2 - val2))
          else:
            entry[1] = onb2 + _log1p(_exp(val2 - onb2))
        else:
          new_pf = prefix + (cid,)
          if use_lm:
            ch_str = id_to_char.get(cid, '')
            ctx_str = _get_ctx(prefix)
            lm_bonus = _lm_w * _lm_score(ctx_str, ch_str)
          else:
            lm_bonus = 0.0
          val = p_total + clp + lm_bonus
          if new_pf in new_beams:
            onb = new_beams[new_pf][1]
            if onb == _inf:
              new_beams[new_pf][1] = val
            elif val > onb:
              new_beams[new_pf][1] = val + _log1p(_exp(onb - val))
            else:
              new_beams[new_pf][1] = onb + _log1p(_exp(val - onb))
          else:
            new_beams[new_pf] = [_inf, val]

    if len(new_beams) > actual_beam:
      scored = []
      for pf, (pb, pnb) in new_beams.items():
        if pb == _inf:
          total = pnb
        elif pnb == _inf:
          total = pb
        elif pb > pnb:
          total = pb + _log1p(_exp(pnb - pb))
        else:
          total = pnb + _log1p(_exp(pb - pnb))
        scored.append((total, pf, pb, pnb))
      scored.sort(reverse=True)
      new_beams = {s[1]: [s[2], s[3]] for s in scored[:actual_beam]}

    beams = new_beams
    if use_lm and t % 50 == 0:
      _ctx_cache.clear()

  # Score and sort all final hypotheses
  scored_hyps = []
  for pf, (pb, pnb) in beams.items():
    if pb == _inf:
      sc = pnb
    elif pnb == _inf:
      sc = pb
    elif pb > pnb:
      sc = pb + _log1p(_exp(pnb - pb))
    else:
      sc = pnb + _log1p(_exp(pb - pnb))
    scored_hyps.append((sc, pf))
  scored_hyps.sort(reverse=True)

  results = []
  for sc, pf in scored_hyps[:nbest]:
    ids = list(pf)
    if id_to_char is not None:
      text = ''.join(id_to_char.get(c, '') for c in ids)
      results.append((sc, text))
    else:
      results.append((sc, ids))
  return results


def ctc_force_score(log_probs_np, token_ids, blank=0):
  """Compute exact CTC log-probability of a label sequence under given logprobs.

  Uses torch.nn.functional.ctc_loss (C++/CUDA optimized) for speed.

  Args:
    log_probs_np: (T, V) numpy float32 log-probabilities.
    token_ids: list of int, the label sequence (no blanks).
    blank: blank token id.

  Returns:
    float: log p(token_ids | log_probs).  Returns -inf if impossible.
  """
  import numpy as np
  if not isinstance(log_probs_np, np.ndarray):
    log_probs_np = np.asarray(log_probs_np, dtype=np.float32)
  if log_probs_np.dtype != np.float32:
    log_probs_np = log_probs_np.astype(np.float32)

  T, V = log_probs_np.shape
  L = len(token_ids)

  if L == 0:
    return float(np.sum(log_probs_np[:, blank]))

  # torch.ctc_loss expects: log_probs (T, 1, V), targets (1, L)
  lp_t = torch.from_numpy(log_probs_np).unsqueeze(1)  # (T, 1, V)
  targets = torch.tensor([token_ids], dtype=torch.long)  # (1, L)
  input_lengths = torch.tensor([T], dtype=torch.long)
  target_lengths = torch.tensor([L], dtype=torch.long)

  # ctc_loss returns NLL = -log p, so negate to get log p
  nll = torch.nn.functional.ctc_loss(
      lp_t, targets, input_lengths, target_lengths,
      blank=blank, reduction='none', zero_infinity=True)
  return float(-nll.item())


def ctc_force_score_batch(log_probs_np, candidates_token_ids, blank=0):
  """Score multiple candidate label sequences under the same logprobs.

  Batched version of ctc_force_score for efficiency.

  Args:
    log_probs_np: (T, V) numpy float32 log-probabilities.
    candidates_token_ids: list of list of int (each is a label sequence).
    blank: blank token id.

  Returns:
    list of float: log p for each candidate.
  """
  import numpy as np
  if not isinstance(log_probs_np, np.ndarray):
    log_probs_np = np.asarray(log_probs_np, dtype=np.float32)
  if log_probs_np.dtype != np.float32:
    log_probs_np = log_probs_np.astype(np.float32)

  T, V = log_probs_np.shape
  N = len(candidates_token_ids)
  if N == 0:
    return []

  # Separate empty and non-empty candidates
  blank_score = float(np.sum(log_probs_np[:, blank]))
  non_empty_indices = [i for i, ids in enumerate(candidates_token_ids) if ids]

  scores = [0.0] * N
  # Handle empty candidates
  for i in range(N):
    if not candidates_token_ids[i]:
      scores[i] = blank_score

  if non_empty_indices:
    # Pack only non-empty candidates for ctc_loss
    ne_count = len(non_empty_indices)
    lp_t = torch.from_numpy(log_probs_np).unsqueeze(1).expand(T, ne_count, V).contiguous()  # (T, ne_count, V)
    input_lengths = torch.full((ne_count,), T, dtype=torch.long)

    all_targets = []
    target_lengths = []
    for i in non_empty_indices:
      ids = candidates_token_ids[i]
      all_targets.extend(ids)
      target_lengths.append(len(ids))

    targets = torch.tensor(all_targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    nlls = torch.nn.functional.ctc_loss(
        lp_t, targets, input_lengths, target_lengths,
        blank=blank, reduction='none', zero_infinity=True)

    ne_scores = (-nlls).tolist()
    for j, i in enumerate(non_empty_indices):
      scores[i] = ne_scores[j]

  return scores


# Keep old name as alias for compatibility
def _prefix_beam_search_single(log_probs_t, blank, beam_width,
                                id_to_char=None, lm=None, lm_weight=0.0):
  """Wrapper: accepts torch tensor, calls numpy implementation."""
  import numpy as np
  if isinstance(log_probs_t, torch.Tensor):
    lp_np = log_probs_t.numpy() if log_probs_t.is_cpu else log_probs_t.cpu().numpy()
  else:
    lp_np = np.asarray(log_probs_t)
  return _prefix_beam_search_np(lp_np, blank, beam_width,
                                 id_to_char=id_to_char, lm=lm, lm_weight=lm_weight)


def ctc_decode(log_probs, blank, beam_width=1,
               id_to_char=None, lm=None, lm_weight=0.0):
  """CTC decode a batch of log-probability sequences.
  
  Args:
    log_probs: (B, T, V) tensor — output of log_softmax
    blank: blank token index
    beam_width: 1 for greedy, >1 for prefix beam search
    id_to_char: dict {int: str} for char-level decoding (optional)
    lm: CharNgramLM instance (optional)
    lm_weight: LM interpolation weight
    
  Returns:
    If id_to_char is provided: list of decoded strings
    Otherwise: list of token-ID lists
  """
  import numpy as np
  if beam_width <= 1:
    id_lists = _greedy_decode_batch(log_probs, blank)
  else:
    B, T, V = log_probs.shape
    # Single GPU→CPU transfer for the whole batch
    lp_np_all = log_probs.detach().cpu().numpy()  # (B, T, V)

    # Serial beam search — ProcessPoolExecutor removed because
    # fork+pickle overhead dominates in Docker with limited CPU cores.
    id_lists = []
    for i in range(B):
      ids = _prefix_beam_search_np(
          lp_np_all[i], blank, beam_width,
          id_to_char=id_to_char, lm=lm, lm_weight=lm_weight)
      id_lists.append(ids)

  # Convert to strings if id_to_char provided
  if id_to_char is not None:
    return [''.join(id_to_char.get(c, '') for c in ids) for ids in id_lists]
  return id_lists
