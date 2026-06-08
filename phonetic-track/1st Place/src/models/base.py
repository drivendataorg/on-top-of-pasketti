#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   base.py
#        \author   chenghuige
#          \date   2025-02-16
#   \Description   Base ASR model for Pasketti.
#                  ctc_weight controls loss mix (orthogonal to encoder choice):
#                    0   -> pure seq2seq
#                    0~1 -> hybrid  (1-w)*s2s + w*ctc
#                    1   -> pure CTC encoder-only
#                  Subclasses implement _build_encoder() and _build_decoder().
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

from gezi.common import *
from src.config import *
from src.preprocess import get_processor, get_tokenizer, tokenize_text, is_native_ctc_char_tokenizer
from src.ctc_decode import ctc_decode, ctc_force_score_batch, CharNgramLM, prefix_beam_search_nbest

import torch.nn.functional as F


# ================= IPA vocabulary constraint =================

# Complete valid IPA character set (from competition metric/score.py)
VALID_IPA_CHARS = {
    ' ',
    # Consonants (ASCII)
    'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm',
    'n', 'p', 'r', 's', 't', 'v', 'w', 'x', 'z',
    # Vowels (ASCII)
    'e', 'i', 'o', 'u',
    # IPA vowels
    'ɑ', 'æ', 'ɐ', 'ɔ', 'ə', 'ɚ', 'ɛ', 'ɪ', 'ʊ', 'ʌ',
    # IPA consonants
    'ç', 'ð', 'ŋ', 'ɟ', 'ɫ', 'ɬ', 'ɹ', 'ɾ', 'ʁ', 'ʃ', 'ʒ', 'ʔ', 'ʝ', 'θ', 'χ',
    # Affricate ligatures
    'ʧ', 'ʤ',
    # Length mark
    'ː',
}

# Character-level IPA vocabulary for CTC
# blank at index 0; real IPA characters at indices 1..N
IPA_CHAR_LIST = sorted(VALID_IPA_CHARS)
IPA_CHAR_VOCAB = ['<blank>'] + IPA_CHAR_LIST
IPA_CHAR_TO_ID = {ch: idx for idx, ch in enumerate(IPA_CHAR_VOCAB) if ch != '<blank>'}
IPA_ID_TO_CHAR = {idx: ch for idx, ch in enumerate(IPA_CHAR_VOCAB) if ch != '<blank>'}
IPA_CTC_BLANK = 0
IPA_CTC_VOCAB_SIZE = len(IPA_CHAR_VOCAB)

# ---- Word character-level CTC vocabulary ----
# For word auxiliary CTC: lowercase a-z + space + apostrophe
WORD_CHAR_SET = set("abcdefghijklmnopqrstuvwxyz '")
WORD_CHAR_LIST = sorted(WORD_CHAR_SET)
WORD_CHAR_VOCAB = ['<blank>'] + WORD_CHAR_LIST
WORD_CHAR_TO_ID = {ch: idx for idx, ch in enumerate(WORD_CHAR_VOCAB) if ch != '<blank>'}
WORD_ID_TO_CHAR = {idx: ch for idx, ch in enumerate(WORD_CHAR_VOCAB) if ch != '<blank>'}
WORD_CTC_BLANK = 0
WORD_CTC_VOCAB_SIZE = len(WORD_CHAR_VOCAB)  # 29: blank + 26 letters + space + apostrophe

# ---- IPA phonetic feature table for label smoothing ----
# Each IPA char → (voicing, manner, place, height, backness, rounding)
# Categorical features encoded as strings; similarity = fraction of shared features.
# 'x' = not applicable (consonant has no vowel features and vice versa).
# blank and space get unique singleton features → only self-similar.
_IPA_FEATURES = {
  # --- Consonants: (voicing, manner, place, x, x, x) ---
  'p': ('voiceless', 'stop',      'bilabial',      'x','x','x'),
  'b': ('voiced',    'stop',      'bilabial',      'x','x','x'),
  't': ('voiceless', 'stop',      'alveolar',      'x','x','x'),
  'd': ('voiced',    'stop',      'alveolar',      'x','x','x'),
  'k': ('voiceless', 'stop',      'velar',         'x','x','x'),
  'g': ('voiced',    'stop',      'velar',         'x','x','x'),
  'ʔ': ('voiceless', 'stop',      'glottal',       'x','x','x'),
  'c': ('voiceless', 'stop',      'palatal',       'x','x','x'),
  'ɟ': ('voiced',    'stop',      'palatal',       'x','x','x'),
  'f': ('voiceless', 'fricative', 'labiodental',   'x','x','x'),
  'v': ('voiced',    'fricative', 'labiodental',   'x','x','x'),
  'θ': ('voiceless', 'fricative', 'dental',        'x','x','x'),
  'ð': ('voiced',    'fricative', 'dental',        'x','x','x'),
  's': ('voiceless', 'fricative', 'alveolar',      'x','x','x'),
  'z': ('voiced',    'fricative', 'alveolar',      'x','x','x'),
  'ʃ': ('voiceless', 'fricative', 'postalveolar',  'x','x','x'),
  'ʒ': ('voiced',    'fricative', 'postalveolar',  'x','x','x'),
  'ç': ('voiceless', 'fricative', 'palatal',       'x','x','x'),
  'ʝ': ('voiced',    'fricative', 'palatal',       'x','x','x'),
  'x': ('voiceless', 'fricative', 'velar',         'x','x','x'),
  'χ': ('voiceless', 'fricative', 'uvular',        'x','x','x'),
  'ʁ': ('voiced',    'fricative', 'uvular',        'x','x','x'),
  'h': ('voiceless', 'fricative', 'glottal',       'x','x','x'),
  'ʧ': ('voiceless', 'affricate', 'postalveolar',  'x','x','x'),
  'ʤ': ('voiced',    'affricate', 'postalveolar',  'x','x','x'),
  'm': ('voiced',    'nasal',     'bilabial',      'x','x','x'),
  'n': ('voiced',    'nasal',     'alveolar',      'x','x','x'),
  'ŋ': ('voiced',    'nasal',     'velar',         'x','x','x'),
  'l': ('voiced',    'lateral',   'alveolar',      'x','x','x'),
  'ɫ': ('voiced',    'lateral',   'alveolar',      'x','x','x'),  # dark l
  'ɬ': ('voiceless', 'lateral_fricative', 'alveolar', 'x','x','x'),
  'ɹ': ('voiced',    'approximant','postalveolar',  'x','x','x'),
  'r': ('voiced',    'trill',     'alveolar',      'x','x','x'),
  'ɾ': ('voiced',    'tap',       'alveolar',      'x','x','x'),
  'w': ('voiced',    'glide',     'bilabial',      'x','x','x'),
  'j': ('voiced',    'glide',     'palatal',       'x','x','x'),
  # --- Vowels: (voiced, vowel, x, height, backness, rounding) ---
  'i': ('voiced', 'vowel', 'x', 'high',    'front',   'unrounded'),
  'ɪ': ('voiced', 'vowel', 'x', 'near-high','front',  'unrounded'),
  'e': ('voiced', 'vowel', 'x', 'mid',     'front',   'unrounded'),
  'ɛ': ('voiced', 'vowel', 'x', 'open-mid','front',   'unrounded'),
  'æ': ('voiced', 'vowel', 'x', 'near-open','front',  'unrounded'),
  'ɑ': ('voiced', 'vowel', 'x', 'open',    'back',    'unrounded'),
  'ɔ': ('voiced', 'vowel', 'x', 'open-mid','back',    'rounded'),
  'o': ('voiced', 'vowel', 'x', 'mid',     'back',    'rounded'),
  'ʊ': ('voiced', 'vowel', 'x', 'near-high','back',   'rounded'),
  'u': ('voiced', 'vowel', 'x', 'high',    'back',    'rounded'),
  'ə': ('voiced', 'vowel', 'x', 'mid',     'central', 'unrounded'),
  'ɚ': ('voiced', 'vowel', 'x', 'mid',     'central', 'unrounded'),  # rhotacized schwa
  'ɐ': ('voiced', 'vowel', 'x', 'near-open','central','unrounded'),
  'ʌ': ('voiced', 'vowel', 'x', 'open-mid','back',    'unrounded'),
  # --- Special ---
  'ː': ('x_length', 'x_length', 'x_length', 'x_length', 'x_length', 'x_length'),
  ' ': ('x_space',  'x_space',  'x_space',  'x_space',  'x_space',  'x_space'),
}


def _build_phonetic_similarity_matrix():
  """Build (V, V) phonetic similarity matrix for IPA CTC vocabulary.
  
  S[i,j] = fraction of shared features between char i and char j.
  Blank (idx 0) and chars without features get 0 similarity to all others.
  Diagonal is always 1.0.
  
  Returns:
    torch.FloatTensor of shape (IPA_CTC_VOCAB_SIZE, IPA_CTC_VOCAB_SIZE)
  """
  V = IPA_CTC_VOCAB_SIZE
  S = torch.zeros(V, V)
  
  for i, ch_i in enumerate(IPA_CHAR_VOCAB):
    feat_i = _IPA_FEATURES.get(ch_i)
    if feat_i is None:
      S[i, i] = 1.0
      continue
    for j, ch_j in enumerate(IPA_CHAR_VOCAB):
      feat_j = _IPA_FEATURES.get(ch_j)
      if feat_j is None:
        continue
      # Only count features where at least one side is non-'x' (meaningful).
      # This avoids inflating similarity from shared placeholder slots.
      meaningful = [(a, b) for a, b in zip(feat_i, feat_j)
                    if not (a.startswith('x') and b.startswith('x'))]
      if not meaningful:
        continue
      matches = sum(1 for a, b in meaningful if a == b)
      S[i, j] = matches / len(meaningful)
  
  # Ensure diagonal = 1
  S.fill_diagonal_(1.0)
  return S


# Lazy-initialized singleton (built on first use)
_PHONETIC_SIM_MATRIX = None

def get_phonetic_similarity_matrix():
  """Get the cached phonetic similarity matrix."""
  global _PHONETIC_SIM_MATRIX
  if _PHONETIC_SIM_MATRIX is None:
    _PHONETIC_SIM_MATRIX = _build_phonetic_similarity_matrix()
  return _PHONETIC_SIM_MATRIX


def _normalize_word_text(s):
  """Normalize word text for character-level CTC: lowercase, keep only a-z/space/apostrophe."""
  s = s.lower().strip()
  # Replace right single quotation mark (\x92, \u2019) with ASCII apostrophe
  s = s.replace('\x92', "'").replace('\u2019', "'").replace('\u2018', "'")
  # Keep only valid chars
  s = ''.join(ch for ch in s if ch in WORD_CHAR_SET)
  # Collapse whitespace
  import re as _re2
  s = _re2.sub(r'\s+', ' ', s).strip()
  return s


_WER_NORMALIZER = None
_WER_NORMALIZER_FAILED = False


def _get_wer_normalizer():
  global _WER_NORMALIZER, _WER_NORMALIZER_FAILED
  if _WER_NORMALIZER is not None:
    return _WER_NORMALIZER
  if _WER_NORMALIZER_FAILED:
    return None
  try:
    from metric.score import EnglishTextNormalizer, english_spelling_normalizer
    _WER_NORMALIZER = EnglishTextNormalizer(english_spelling_normalizer)
    return _WER_NORMALIZER
  except Exception:
    _WER_NORMALIZER_FAILED = True
    return None


def _normalize_wer_text(text):
  text = str(text or '')
  normalizer = _get_wer_normalizer()
  if normalizer is not None:
    text = normalizer(text)
  text = text.replace('\u2019', "'").replace('\u2018', "'").strip().lower()
  return ' '.join(text.split())


def _single_utterance_wer(ref, hyp):
  import editdistance

  ref_words = _normalize_wer_text(ref).split()
  hyp_words = _normalize_wer_text(hyp).split()
  if not ref_words:
    return 0.0 if not hyp_words else 1.0
  return editdistance.eval(ref_words, hyp_words) / len(ref_words)


# Minimal IPA normalizer (mirrors metric/score.py normalize_ipa)
import re as _re
import string as _string
from unicodedata import normalize as _unicode_normalize

_IPA_TRANS = str.maketrans({
    **{"ẽ": "e", "ĩ": "i", "õ": "o", "ũ": "u"},          # nasal decompose
    **{ord("\u035c"): None, ord("\u0361"): None,            # tie bars
       ord("\u02c8"): None, ord("\u02cc"): None,            # stress marks
       ord("\u0303"): None},                                # combining tilde
    **{ord("ɝ"): "ɚ"},                                     # rhotic normalize
    **{c: None for c in _string.punctuation},               # punctuation
})
_IPA_SPACE_RE = _re.compile(r"\s+")


def _normalize_ipa(s):
  """Normalize IPA text to match competition scoring (VALID_IPA_CHARS only)."""
  s = _unicode_normalize("NFC", s)
  s = s.translate(_IPA_TRANS)
  s = s.replace("tʃ", "ʧ").replace("dʒ", "ʤ")
  s = _IPA_SPACE_RE.sub(" ", s).strip()
  return s


def build_ipa_allowed_ids(tokenizer):
  """Scan tokenizer vocab to find all token IDs whose surface text
  consists entirely of VALID_IPA_CHARS. Also keep ALL special tokens
  (eos, pad, bos, language, task, timestamps, etc.) so that the
  decoder prompt and generation control tokens are never masked."""
  allowed = set()
  # Always allow ALL special tokens (covers decoder prompt IDs like
  # <|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>, etc.)
  if hasattr(tokenizer, 'all_special_ids'):
    allowed.update(tokenizer.all_special_ids)
  for sid in [tokenizer.eos_token_id, tokenizer.pad_token_id,
              getattr(tokenizer, 'bos_token_id', None)]:
    if sid is not None:
      allowed.add(sid)

  vocab = tokenizer.get_vocab()  # {str: int}
  for token_str, token_id in vocab.items():
    # Whisper BPE tokens may start with 'Ġ' (== space) — strip it
    clean = token_str.replace('Ġ', ' ').replace('▁', ' ').strip()
    if not clean:  # whitespace-only tokens are ok
      allowed.add(token_id)
      continue
    if all(ch in VALID_IPA_CHARS for ch in clean):
      allowed.add(token_id)

  return allowed


class IPALogitsProcessor:
  """Zero out (set to -inf) logits for tokens not in the IPA allowed set.
  Compatible with transformers LogitsProcessor interface."""

  def __init__(self, allowed_ids, vocab_size):
    self.allowed_ids = set(allowed_ids)
    self._tok_vocab_size = vocab_size  # tokenizer base vocab size
    self._mask = None
    self._mask_size = 0

  def _get_mask(self, vocab_size, device):
    if self._mask is None or self._mask_size != vocab_size:
      mask = torch.ones(vocab_size, dtype=torch.bool)
      for tid in self.allowed_ids:
        if tid < vocab_size:
          mask[tid] = False
      # Extra tokens beyond base vocab are special — always allow
      if hasattr(self, '_tok_vocab_size') and vocab_size > self._tok_vocab_size:
        mask[self._tok_vocab_size:] = False
      self._mask = mask
      self._mask_size = vocab_size
    return self._mask.to(device)

  def __call__(self, input_ids, scores):
    mask = self._get_mask(scores.size(-1), scores.device)
    scores[:, mask] = float('-inf')
    return scores


class CTCHead(nn.Module):
  """Linear CTC head on top of encoder output."""

  def __init__(self, encoder_dim, vocab_size, dropout=0.1):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.proj = nn.Linear(encoder_dim, vocab_size)

  def forward(self, x):
    return self.proj(self.dropout(x))  # (B, T, V)


# ================= AED Decoder (Scheme 2) =================
# Vocabulary: IPA chars (1..52) + <blank/sos>=0 + <eos>=53 → 54 classes
AED_SOS_ID = IPA_CTC_BLANK        # reuse blank=0 as <sos>
AED_EOS_ID = IPA_CTC_VOCAB_SIZE   # 53 — one beyond CTC vocab
AED_VOCAB_SIZE = IPA_CTC_VOCAB_SIZE + 1  # 54


class AEDDecoder(nn.Module):
  """Lightweight Transformer AED (Attention-based Encoder-Decoder) for IPA.

  Input:  encoder output (B, T, enc_dim)
  Output: logits (B, L, vocab_size)
  """

  def __init__(self, encoder_dim, d_model=256, nhead=4, num_layers=2,
               vocab_size=AED_VOCAB_SIZE, dropout=0.1, max_pos=1024):
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.max_pos = max_pos
    self.eos_id = vocab_size - 1  # <eos> = last index
    self.sos_id = 0               # <blank/sos>

    # Encoder projection (encoder_dim → d_model)
    self.enc_proj = nn.Linear(encoder_dim, d_model)

    # Token embedding + learned positional encoding
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_enc = nn.Embedding(max_pos, d_model)
    self.embed_dropout = nn.Dropout(dropout)

    # Transformer decoder layers
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
        dropout=dropout, batch_first=True)
    self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    # Output projection
    self.output_proj = nn.Linear(d_model, vocab_size)

  def forward(self, enc_out, tgt_ids, enc_lengths=None):
    """Teacher-forced forward.
    enc_out:  (B, T, enc_dim)
    tgt_ids:  (B, L) decoder input IDs  [sos, c1, c2, ...]
    Returns:  logits (B, L, V)
    """
    B, L = tgt_ids.shape

    # Project encoder
    memory = self.enc_proj(enc_out)  # (B, T, d_model)

    # Encoder padding mask
    memory_key_padding_mask = None
    if enc_lengths is not None:
      T = memory.shape[1]
      rng = torch.arange(T, device=memory.device).unsqueeze(0)
      memory_key_padding_mask = rng >= enc_lengths.unsqueeze(1)  # True=pad

    # Embed targets
    positions = torch.arange(L, device=tgt_ids.device).unsqueeze(0)
    tgt_emb = self.embedding(tgt_ids) + self.pos_enc(positions)
    tgt_emb = self.embed_dropout(tgt_emb)

    # Causal mask
    causal_mask = nn.Transformer.generate_square_subsequent_mask(
        L, device=tgt_ids.device)

    # Decode
    dec_out = self.decoder(
        tgt=tgt_emb, memory=memory,
        tgt_mask=causal_mask,
        memory_key_padding_mask=memory_key_padding_mask)

    return self.output_proj(dec_out)  # (B, L, V)

  @torch.no_grad()
  def generate(self, enc_out, enc_lengths=None, max_len=256, beam_size=1,
               length_penalty=0.0):
    """Autoregressive decoding → IPA char IDs (B, L).
    
    beam_size=1: greedy decode (original behaviour).
    beam_size>1: beam search — keeps top-k hypotheses per sample.
    length_penalty: α for length normalisation  score / len^α  (0 = off).
    """
    if beam_size <= 1:
      return self._greedy_decode(enc_out, enc_lengths, max_len)
    return self._beam_search_decode(enc_out, enc_lengths, max_len,
                                    beam_size, length_penalty)

  # ---- greedy decode (unchanged) ----
  @torch.no_grad()
  def _greedy_decode(self, enc_out, enc_lengths, max_len):
    B = enc_out.shape[0]
    device = enc_out.device
    memory = self.enc_proj(enc_out)

    memory_key_padding_mask = None
    if enc_lengths is not None:
      T = memory.shape[1]
      rng = torch.arange(T, device=device).unsqueeze(0)
      memory_key_padding_mask = rng >= enc_lengths.unsqueeze(1)

    # Start with SOS
    generated = torch.full((B, 1), self.sos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
      L = generated.shape[1]
      positions = torch.arange(L, device=device).unsqueeze(0)
      tgt_emb = self.embedding(generated) + self.pos_enc(positions)
      causal_mask = nn.Transformer.generate_square_subsequent_mask(
          L, device=device)

      dec_out = self.decoder(
          tgt=tgt_emb, memory=memory,
          tgt_mask=causal_mask,
          memory_key_padding_mask=memory_key_padding_mask)

      next_token = self.output_proj(dec_out[:, -1, :]).argmax(dim=-1)  # (B,)
      next_token = next_token.masked_fill(finished, 0)
      finished = finished | (next_token == self.eos_id)
      generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
      if finished.all():
        break

    # Remove SOS, truncate at EOS
    result = generated[:, 1:]
    for i in range(B):
      eos_pos = (result[i] == self.eos_id).nonzero(as_tuple=False)
      if len(eos_pos) > 0:
        result[i, eos_pos[0]:] = 0
    return result

  # ---- beam search decode ----
  @torch.no_grad()
  def _beam_search_decode(self, enc_out, enc_lengths, max_len,
                          beam_size, length_penalty):
    """Per-sample beam search.  Processes each sample independently to
    keep memory bounded (batch dim is usually small for ASR)."""
    B = enc_out.shape[0]
    device = enc_out.device
    all_results = []

    for b in range(B):
      # Slice single sample: (1, T, D)
      mem_b = self.enc_proj(enc_out[b:b+1])
      mask_b = None
      if enc_lengths is not None:
        T = mem_b.shape[1]
        rng = torch.arange(T, device=device).unsqueeze(0)
        mask_b = rng >= enc_lengths[b:b+1].unsqueeze(1)

      # Each beam: (token_ids_list, cumulative_log_prob)
      beams = [([self.sos_id], 0.0)]
      finished_beams = []

      for _ in range(max_len):
        candidates = []
        # Expand memory for all live beams
        n_live = len(beams)
        if n_live == 0:
          break
        mem_exp = mem_b.expand(n_live, -1, -1)          # (n_live, T, d)
        mask_exp = mask_b.expand(n_live, -1) if mask_b is not None else None

        # Build decoder input from all beams
        max_seq = max(len(bm[0]) for bm in beams)
        dec_in = torch.full((n_live, max_seq), self.sos_id,
                            dtype=torch.long, device=device)
        for i, (toks, _) in enumerate(beams):
          dec_in[i, :len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)

        L = dec_in.shape[1]
        positions = torch.arange(L, device=device).unsqueeze(0)
        tgt_emb = self.embedding(dec_in) + self.pos_enc(positions)
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=device)

        dec_out = self.decoder(
            tgt=tgt_emb, memory=mem_exp,
            tgt_mask=causal,
            memory_key_padding_mask=mask_exp)

        logits = self.output_proj(dec_out[:, -1, :])        # (n_live, V)
        log_probs = torch.log_softmax(logits, dim=-1)       # (n_live, V)

        topk_lp, topk_id = log_probs.topk(beam_size, dim=-1)  # (n_live, beam)

        for i, (toks, score) in enumerate(beams):
          for k in range(beam_size):
            tok = topk_id[i, k].item()
            new_score = score + topk_lp[i, k].item()
            new_toks = toks + [tok]
            if tok == self.eos_id:
              # Length normalisation (exclude SOS and EOS from length)
              seq_len = max(len(new_toks) - 2, 1)
              norm_score = new_score / (seq_len ** length_penalty) if length_penalty > 0 else new_score
              finished_beams.append((new_toks, norm_score))
            else:
              candidates.append((new_toks, new_score))

        # Keep top beam_size candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]

        # Early stop: best finished beam beats all live beams
        if finished_beams and beams:
          best_fin = max(fb[1] for fb in finished_beams)
          best_live = beams[0][1]
          if best_fin >= best_live:
            break

      # Pick best hypothesis
      if not finished_beams:
        # No beam finished with EOS — take best live beam
        finished_beams = beams if beams else [([self.sos_id, 0], 0.0)]
      best_hyp = max(finished_beams, key=lambda x: x[1])[0]

      # Remove SOS, truncate at EOS
      ids = best_hyp[1:]  # drop SOS
      if self.eos_id in ids:
        ids = ids[:ids.index(self.eos_id)]
      all_results.append(ids)

    # Pad to uniform length
    max_out = max(len(r) for r in all_results) if all_results else 0
    result = torch.zeros(B, max(max_out, 1), dtype=torch.long, device=device)
    for i, ids in enumerate(all_results):
      if ids:
        result[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return result


# ================= Custom RNNT Decoder (Scheme 3) =================

class CustomRNNTDecoder(nn.Module):
  """Lightweight custom RNNT decoder for IPA character-level prediction.

  Prediction network: Embedding + LSTM
  Joint network:      Linear projections + ReLU + output
  Loss:               torchaudio RNNT loss

  Also used by Scheme 1 (rnnt_reuse) with pretrained LSTM weights.
  """

  def __init__(self, encoder_dim, vocab_size=IPA_CTC_VOCAB_SIZE,
               pred_dim=256, pred_layers=1, joint_dim=256):
    super().__init__()
    self.vocab_size = vocab_size
    self.blank_id = 0
    self.pred_dim = pred_dim
    self.encoder_dim = encoder_dim

    # Prediction network
    self.pred_embedding = nn.Embedding(vocab_size, pred_dim)
    self.pred_rnn = nn.LSTM(pred_dim, pred_dim, num_layers=pred_layers,
                            batch_first=True)

    # Joint network
    self.enc_proj = nn.Linear(encoder_dim, joint_dim)
    self.pred_proj = nn.Linear(pred_dim, joint_dim)
    self.joint_out = nn.Sequential(
        nn.ReLU(),
        nn.Linear(joint_dim, vocab_size),
    )

  def forward(self, enc_out, targets, enc_lengths, target_lengths):
    """Compute RNNT loss.
    enc_out:        (B, T, enc_dim)
    targets:        (B, U) IPA char IDs (no blank, no SOS — raw label)
    enc_lengths:    (B,)
    target_lengths: (B,)
    Returns: per-sample loss (B,)
    """
    B = targets.shape[0]
    T = enc_out.shape[1]
    U = targets.shape[1]  # max target length in batch
    device = targets.device

    # Clamp encoder lengths to actual tensor size, min=1 to avoid zero-length
    enc_lengths = enc_lengths.clamp(min=1, max=T)

    # Handle edge case: all-empty targets → return zero loss
    if target_lengths.sum() == 0:
      return torch.zeros(B, device=device, dtype=torch.float32, requires_grad=True)

    active_mask = target_lengths > 0
    if not bool(active_mask.all()):
      active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
      active_loss = self.forward(
          enc_out.index_select(0, active_idx),
          targets.index_select(0, active_idx),
          enc_lengths.index_select(0, active_idx),
          target_lengths.index_select(0, active_idx),
      )
      loss = torch.zeros(B, device=device, dtype=active_loss.dtype)
      loss = loss.index_copy(0, active_idx, active_loss)
      return loss

    # torchaudio rnnt_loss requires EXACT match:
    #   logits.size(1) == max(logit_lengths)
    #   logits.size(2) == max(target_lengths) + 1
    # So we must trim enc_out to max(enc_lengths) and targets to max(target_lengths).
    T_max = int(enc_lengths.max().item())
    U_max = int(target_lengths.max().item())
    if T_max < T:
      enc_out = enc_out[:, :T_max, :]
      T = T_max
    if U_max < U:
      targets = targets[:, :U_max]
      U = U_max

    # Prediction network: prepend SOS (blank=0)
    sos = torch.zeros(B, 1, dtype=torch.long, device=device)
    targets_sos = torch.cat([sos, targets], dim=1)  # (B, U+1)

    embedded = self.pred_embedding(targets_sos)  # (B, U+1, pred_dim)
    packed = nn.utils.rnn.pack_padded_sequence(
        embedded, (target_lengths + 1).cpu().clamp(min=1),
        batch_first=True, enforce_sorted=False)
    pred_out, _ = self.pred_rnn(packed)
    # Force output length = U+1 to match targets dimension exactly
    pred_out, _ = nn.utils.rnn.pad_packed_sequence(
        pred_out, batch_first=True, total_length=U + 1)  # (B, U+1, pred_dim)

    # Joint network: (B, T, 1, joint) + (B, 1, U+1, joint) → (B, T, U+1, V)
    enc_proj = self.enc_proj(enc_out).unsqueeze(2)       # (B, T, 1, joint)
    pred_proj = self.pred_proj(pred_out).unsqueeze(1)    # (B, 1, U+1, joint)
    logits = self.joint_out(enc_proj + pred_proj)        # (B, T, U+1, V)

    # RNNT loss
    # fused_log_softmax=True: torchaudio applies log_softmax internally on raw logits
    try:
      from torchaudio.functional import rnnt_loss
    except ImportError:
      raise ImportError(
          'torchaudio is required for custom RNNT loss (--s2s_decoder=rnnt_custom). '
          'Install with: pip install torchaudio')

    loss = rnnt_loss(
        logits.float(), targets.int(),
        enc_lengths.int(), target_lengths.int(),
        blank=self.blank_id, reduction='none',
        fused_log_softmax=True)

    # Normalize by target length
    return loss / target_lengths.float().clamp(min=1)

  @torch.no_grad()
  def greedy_decode(self, enc_out, enc_lengths=None, max_symbols_per_step=10):
    """RNNT greedy decoding → IPA char IDs (B, L)."""
    B, T, D = enc_out.shape
    device = enc_out.device

    if enc_lengths is None:
      enc_lengths = torch.full((B,), T, dtype=torch.long, device=device)

    all_decoded = []
    for b in range(B):
      decoded = []
      h = torch.zeros(self.pred_rnn.num_layers, 1, self.pred_dim, device=device)
      c = torch.zeros(self.pred_rnn.num_layers, 1, self.pred_dim, device=device)
      last_token = torch.zeros(1, 1, dtype=torch.long, device=device)  # SOS=0

      for t in range(int(enc_lengths[b])):
        enc_proj_t = self.enc_proj(enc_out[b, t])  # (joint_dim,)

        for _ in range(max_symbols_per_step):
          embedded = self.pred_embedding(last_token)  # (1, 1, pred_dim)
          pred_t, (h, c) = self.pred_rnn(embedded, (h, c))
          pred_proj_t = self.pred_proj(pred_t.squeeze(0).squeeze(0))  # (joint,)
          logits = self.joint_out(enc_proj_t + pred_proj_t)  # (V,)

          pred_id = logits.argmax().item()
          if pred_id == self.blank_id:
            break
          decoded.append(pred_id)
          last_token = torch.tensor([[pred_id]], dtype=torch.long, device=device)

      all_decoded.append(decoded)

    # Pad to uniform length
    max_len = max((len(d) for d in all_decoded), default=1)
    max_len = max(max_len, 1)
    result = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, d in enumerate(all_decoded):
      if d:
        result[i, :len(d)] = torch.tensor(d, dtype=torch.long)
    return result

  @torch.no_grad()
  def beam_decode(self, enc_out, enc_lengths=None, beam_size=5,
                  max_symbols_per_step=10):
    """RNNT beam search decoding → IPA char IDs (B, L).

    Each beam tracks (decoded_tokens, cumulative_log_prob, h_state, c_state).
    At each encoder time step, beams expand by emitting symbols or blank.
    """
    B, T, D = enc_out.shape
    device = enc_out.device

    if enc_lengths is None:
      enc_lengths = torch.full((B,), T, dtype=torch.long, device=device)

    all_decoded = []
    for b in range(B):
      # Initial beam: empty sequence, log_prob=0, fresh LSTM states
      h0 = torch.zeros(self.pred_rnn.num_layers, 1, self.pred_dim, device=device)
      c0 = torch.zeros(self.pred_rnn.num_layers, 1, self.pred_dim, device=device)
      last_tok0 = torch.zeros(1, 1, dtype=torch.long, device=device)
      # Beam: (tokens, log_prob, h, c, last_token)
      beams = [([], 0.0, h0, c0, last_tok0)]

      for t in range(int(enc_lengths[b])):
        enc_proj_t = self.enc_proj(enc_out[b, t])  # (joint_dim,)
        # At each time step, iteratively expand beams by emitting symbols,
        # then advance to next time step when blank is emitted.
        next_beams = []
        active = list(beams)

        for _ in range(max_symbols_per_step):
          if not active:
            break
          expanded = []
          for (toks, score, h, c, last_tok) in active:
            embedded = self.pred_embedding(last_tok)
            pred_t, (h_new, c_new) = self.pred_rnn(embedded, (h, c))
            pred_proj_t = self.pred_proj(pred_t.squeeze(0).squeeze(0))
            logits = self.joint_out(enc_proj_t + pred_proj_t)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Blank → advance to next time step (keep old LSTM state)
            blank_score = score + log_probs[self.blank_id].item()
            next_beams.append((toks, blank_score, h, c, last_tok))

            # Non-blank symbols → stay at this time step with new state
            topk_lp, topk_id = log_probs.topk(min(beam_size, self.vocab_size))
            for k in range(topk_lp.shape[0]):
              sym = topk_id[k].item()
              if sym == self.blank_id:
                continue
              sym_score = score + topk_lp[k].item()
              new_tok = torch.tensor([[sym]], dtype=torch.long, device=device)
              expanded.append((toks + [sym], sym_score,
                               h_new.clone(), c_new.clone(), new_tok))

          # Active beams for next symbol emission round = top-k expanded
          expanded.sort(key=lambda x: x[1], reverse=True)
          active = expanded[:beam_size]

        # Merge all candidates from this time step
        all_candidates = next_beams + active
        # De-duplicate by token sequence, keep best score
        best_by_seq = {}
        for cand in all_candidates:
          key = tuple(cand[0])
          if key not in best_by_seq or cand[1] > best_by_seq[key][1]:
            best_by_seq[key] = cand
        beams = sorted(best_by_seq.values(), key=lambda x: x[1], reverse=True)[:beam_size]

      # Best beam
      best = max(beams, key=lambda x: x[1])
      all_decoded.append(best[0])

    # Pad to uniform length
    max_len = max((len(d) for d in all_decoded), default=1)
    max_len = max(max_len, 1)
    result = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, d in enumerate(all_decoded):
      if d:
        result[i, :len(d)] = torch.tensor(d, dtype=torch.long)
    return result


def _tdt_forward_score_pytorch(logits, targets, enc_lengths, target_lengths,
                               durations, blank_id, sigma=0.0):
  """Pure PyTorch implementation of TDT forward algorithm (score only, no grads).

  Computes the negative log-likelihood for each sample in the batch, identical
  to NeMo's compute_tdt_alphas_kernel + compute_costs_data but without Numba.
  Vectorized over the U dimension on each anti-diagonal for GPU efficiency.

  Args:
    logits:         (B, T, U+1, V+D)  raw joint network output
    targets:        (B, U)             label indices (no blank, no SOS)
    enc_lengths:    (B,)
    target_lengths: (B,)
    durations:      list of ints, e.g. [0, 1, 2, 3, 4]
    blank_id:       int, blank token index
    sigma:          float, logit under-normalization weight

  Returns:
    costs: (B,) negative log-likelihood per sample (positive values, lower = better fit)
  """
  B, T, U1, VD = logits.shape
  U = U1 - 1  # U+1 prediction positions (including SOS position)
  num_dur = len(durations)
  V = VD - num_dur

  # Split into label logits and duration logits
  label_logits = logits[..., :V]           # (B, T, U+1, V)
  dur_logits = logits[..., V:]              # (B, T, U+1, D)

  # Log-softmax over vocab and duration dims
  log_probs = torch.log_softmax(label_logits.float(), dim=-1)   # (B, T, U+1, V)
  log_dur = torch.log_softmax(dur_logits.float(), dim=-1)       # (B, T, U+1, D)

  # Pre-extract blank log-probs for all positions: (B, T, U+1)
  log_probs_blank = log_probs[:, :, :, blank_id]

  INF = 1e10
  device = logits.device

  # alphas: (B, T, U+1) — forward variable in log-space
  alphas = torch.full((B, T, U1), -INF, device=device, dtype=torch.float32)
  alphas[:, 0, 0] = 0.0

  # Pre-gather label log-probs for emit transitions: log_probs_emit[b, t, u] = log_probs[b, t, u, targets[b, u]]
  # targets is (B, U), we need (B, T, U)
  tgt_expanded = targets.unsqueeze(1).expand(B, T, U)  # (B, T, U)
  # log_probs at u positions 0..U-1 (the label at position u feeds into emit to u+1)
  log_probs_emit = log_probs[:, :, :U, :].gather(3, tgt_expanded.unsqueeze(3)).squeeze(3)  # (B, T, U)

  # Wavefront diagonal sweep: for each anti-diagonal n = t + u
  for n in range(1, T + U):
    u_min = max(0, n - T + 1)
    u_max = min(U, n)
    n_cells = u_max - u_min + 1
    if n_cells <= 0:
      continue

    # u indices on this diagonal
    u_idx = torch.arange(u_min, u_max + 1, device=device)  # (n_cells,)
    t_idx = n - u_idx  # (n_cells,)

    # Collect all transition terms for this diagonal
    all_terms = []  # list of (B, n_cells) tensors

    # === Blank transitions: from (t-d, u) for each duration d>0 ===
    for d_idx, d in enumerate(durations):
      if d == 0:
        continue
      # Source: t_src = t - d, u_src = u (same)
      t_src = t_idx - d  # (n_cells,)
      valid = (t_src >= 0)  # (n_cells,)
      if not valid.any():
        break  # durations ascending, rest will also fail

      # For invalid entries, clamp to 0 (will be masked out)
      t_src_c = t_src.clamp(min=0)
      # alpha(t-d, u): index alphas[B, t_src, u_idx]
      prev = alphas[:, t_src_c, u_idx]  # (B, n_cells)
      lp_blank = log_probs_blank[:, t_src_c, u_idx]  # (B, n_cells)
      lp_dur = log_dur[:, t_src_c, u_idx, d_idx]  # (B, n_cells)
      term = prev + lp_blank - sigma + lp_dur  # (B, n_cells)
      # Mask invalid
      term[:, ~valid] = -INF
      all_terms.append(term)

    # === Emit transitions: from (t-d, u-1) for each duration d>=0 ===
    # Only valid when u > 0
    emit_mask = (u_idx > 0)  # (n_cells,)
    if emit_mask.any():
      u_src = u_idx - 1  # (n_cells,) — source u position
      for d_idx, d in enumerate(durations):
        t_src = t_idx - d  # (n_cells,)
        valid = (t_src >= 0) & emit_mask  # (n_cells,)
        if not valid.any():
          break

        t_src_c = t_src.clamp(min=0)
        u_src_c = u_src.clamp(min=0)
        prev = alphas[:, t_src_c, u_src_c]  # (B, n_cells)
        # log_probs_emit[b, t_src, u_src] = log_probs[b, t_src, u_src, targets[b, u_src]]
        lp_label = log_probs_emit[:, t_src_c, u_src_c]  # (B, n_cells)
        lp_dur = log_dur[:, t_src_c, u_src_c, d_idx]  # (B, n_cells)
        term = prev + lp_label - sigma + lp_dur  # (B, n_cells)
        term[:, ~valid] = -INF
        all_terms.append(term)

    if all_terms:
      stacked = torch.stack(all_terms, dim=0)  # (num_terms, B, n_cells)
      new_alpha = torch.logsumexp(stacked, dim=0)  # (B, n_cells)
      alphas[:, t_idx, u_idx] = new_alpha

  # Terminal: sum over durations d>=1 of alpha(T-d, U) + logp(blank at T-d, U) - sigma + logp_dur
  costs = torch.full((B,), INF, device=device, dtype=torch.float32)
  for b in range(B):
    Tb = int(enc_lengths[b].item())
    Ub = int(target_lengths[b].item())
    terms_b = []
    for d_idx, d in enumerate(durations):
      if d == 0:
        continue
      if d > Tb:
        break
      t_src = Tb - d
      u_src = Ub
      val = (alphas[b, t_src, u_src]
             + log_probs_blank[b, t_src, u_src]
             - sigma
             + log_dur[b, t_src, u_src, d_idx])
      terms_b.append(val)
    if terms_b:
      costs[b] = -torch.logsumexp(torch.stack(terms_b), dim=0)
    else:
      costs[b] = INF
  return costs


class CustomTDTDecoder(nn.Module):
  """TDT decoder for IPA character-level prediction — exact replica of NeMo TDT
  architecture with duration prediction, only vocab changed to IPA.

  Prediction network: Embedding + LSTM (weights copied from NeMo TDT backbone)
  Joint network:      Linear projections + ReLU + Dropout + output (vocab + durations)
  Loss:               NeMo TDTLossNumba (RNNT + duration prediction)

  Used by --s2s_decoder=tdt_reuse.
  """

  def __init__(self, encoder_dim, vocab_size=IPA_CTC_VOCAB_SIZE,
               pred_dim=640, pred_layers=2, joint_dim=640,
               durations=None, sigma=0.02, omega=0.1, dropout=0.2,
               blank_id=0, padding_idx='auto', pred_embedding=None,
               pred_rnn=None, enc_proj=None, pred_proj=None, joint_out=None):
    super().__init__()
    if durations is None:
      durations = [0, 1, 2, 3, 4]
    self.vocab_size = vocab_size
    self.blank_id = int(blank_id)
    self.pred_dim = pred_dim
    self.encoder_dim = encoder_dim
    self.durations = durations
    self.num_durations = len(durations)

    if padding_idx == 'auto':
      padding_idx = vocab_size - 1

    # Prediction network (matches NeMo RNNTDecoder structure)
    self.pred_embedding = pred_embedding or nn.Embedding(
        vocab_size, pred_dim, padding_idx=padding_idx)
    self.pred_rnn = pred_rnn or nn.LSTM(
        pred_dim, pred_dim, num_layers=pred_layers,
        batch_first=True, dropout=dropout if pred_layers > 1 else 0.0)

    # Joint network (matches NeMo RNNTJoint structure)
    self.enc_proj = enc_proj or nn.Linear(encoder_dim, joint_dim)
    self.pred_proj = pred_proj or nn.Linear(pred_dim, joint_dim)
    # Output: vocab (token logits) + num_durations (duration logits)
    self.joint_out = joint_out or nn.Sequential(
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(joint_dim, vocab_size + self.num_durations),
    )

    # TDT loss from NeMo
    from nemo.collections.asr.losses.rnnt import TDTLossNumba
    self.tdt_loss = TDTLossNumba(
        blank=self.blank_id,
        durations=durations,
        reduction='none',
        sigma=sigma,
        omega=omega,
    )

  def forward(self, enc_out, targets, enc_lengths, target_lengths):
    """Compute TDT loss.
    enc_out:        (B, T, enc_dim)
    targets:        (B, U) IPA char IDs (no blank, no SOS — raw label)
    enc_lengths:    (B,)
    target_lengths: (B,)
    Returns: per-sample loss (B,)
    """
    B = targets.shape[0]
    T = enc_out.shape[1]
    U = targets.shape[1]
    device = targets.device

    enc_lengths = enc_lengths.clamp(min=1, max=T)

    if target_lengths.sum() == 0:
      return torch.zeros(B, device=device, requires_grad=True)

    T_max = int(enc_lengths.max().item())
    U_max = int(target_lengths.max().item())
    if T_max < T:
      enc_out = enc_out[:, :T_max, :]
      T = T_max
    if U_max < U:
      targets = targets[:, :U_max]
      U = U_max

    targets = targets.contiguous()
    per_sample_loss = self._compute_per_sample_loss(
        enc_out, targets, enc_lengths, target_lengths)
    return per_sample_loss / target_lengths.float().clamp(min=1)

  def _compute_per_sample_loss(self, enc_out, targets, enc_lengths, target_lengths):
    B = targets.shape[0]
    T = enc_out.shape[1]
    U = targets.shape[1]
    device = targets.device

    enc_lengths = enc_lengths.clamp(min=1, max=T)

    if target_lengths.sum() == 0:
      return torch.zeros(B, device=device, dtype=torch.float32, requires_grad=True)

    active_mask = target_lengths > 0
    if not bool(active_mask.all()):
      active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
      active_loss = self._compute_per_sample_loss(
          enc_out.index_select(0, active_idx),
          targets.index_select(0, active_idx),
          enc_lengths.index_select(0, active_idx),
          target_lengths.index_select(0, active_idx),
      )
      loss = torch.zeros(B, device=device, dtype=active_loss.dtype)
      loss = loss.index_copy(0, active_idx, active_loss)
      return loss

    T_max = int(enc_lengths.max().item())
    U_max = int(target_lengths.max().item())
    if T_max < T:
      enc_out = enc_out[:, :T_max, :]
    if U_max < U:
      targets = targets[:, :U_max].contiguous()
      U = U_max

    sos = torch.zeros(B, 1, dtype=torch.long, device=device)
    targets_sos = torch.cat([sos, targets], dim=1)

    embedded = self.pred_embedding(targets_sos)
    packed = nn.utils.rnn.pack_padded_sequence(
        embedded, (target_lengths + 1).cpu().clamp(min=1),
        batch_first=True, enforce_sorted=False)
    pred_out, _ = self.pred_rnn(packed)
    pred_out, _ = nn.utils.rnn.pad_packed_sequence(
        pred_out, batch_first=True, total_length=U + 1)

    enc_proj = self.enc_proj(enc_out).unsqueeze(2)
    pred_proj = self.pred_proj(pred_out).unsqueeze(1)
    logits = self.joint_out(enc_proj + pred_proj)

    return self.tdt_loss(
        logits.float(), targets.long().contiguous(),
        enc_lengths.long(), target_lengths.long(),
    )

  def _compute_per_sample_loss_pytorch(self, enc_out, targets, enc_lengths, target_lengths):
    """Same as _compute_per_sample_loss but uses pure PyTorch forward algorithm
    instead of NeMo TDTLossNumba (Numba CUDA JIT). This avoids Numba JIT
    compilation issues in restricted Docker environments."""
    B = targets.shape[0]
    T = enc_out.shape[1]
    U = targets.shape[1]
    device = targets.device

    enc_lengths = enc_lengths.clamp(min=1, max=T)

    if target_lengths.sum() == 0:
      return torch.zeros(B, device=device, dtype=torch.float32)

    active_mask = target_lengths > 0
    if not bool(active_mask.all()):
      active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
      active_loss = self._compute_per_sample_loss_pytorch(
          enc_out.index_select(0, active_idx),
          targets.index_select(0, active_idx),
          enc_lengths.index_select(0, active_idx),
          target_lengths.index_select(0, active_idx),
      )
      loss = torch.zeros(B, device=device, dtype=active_loss.dtype)
      loss = loss.index_copy(0, active_idx, active_loss)
      return loss

    T_max = int(enc_lengths.max().item())
    U_max = int(target_lengths.max().item())
    if T_max < T:
      enc_out = enc_out[:, :T_max, :]
    if U_max < U:
      targets = targets[:, :U_max].contiguous()
      U = U_max

    sos = torch.zeros(B, 1, dtype=torch.long, device=device)
    targets_sos = torch.cat([sos, targets], dim=1)

    embedded = self.pred_embedding(targets_sos)
    packed = nn.utils.rnn.pack_padded_sequence(
        embedded, (target_lengths + 1).cpu().clamp(min=1),
        batch_first=True, enforce_sorted=False)
    pred_out, _ = self.pred_rnn(packed)
    pred_out, _ = nn.utils.rnn.pad_packed_sequence(
        pred_out, batch_first=True, total_length=U + 1)

    enc_proj = self.enc_proj(enc_out).unsqueeze(2)
    pred_proj = self.pred_proj(pred_out).unsqueeze(1)
    logits = self.joint_out(enc_proj + pred_proj)
    
    if not hasattr(self, '_compiled_tdt_pytorch'):
      try:
        import torch._dynamo as _dummy_dynamo
        self._compiled_tdt_pytorch = torch.compile(_tdt_forward_score_pytorch)
        self._compiled_tdt_warmup = False
      except Exception:
        self._compiled_tdt_pytorch = _tdt_forward_score_pytorch
        
    return self._compiled_tdt_pytorch(
        logits, targets.long().contiguous(),
        enc_lengths.long(), target_lengths.long(),
        durations=self.durations, blank_id=self.blank_id, sigma=self.tdt_loss.sigma,
    )

  def _compute_per_sample_loss_numba_cpu(self, enc_out, targets, enc_lengths, target_lengths):
    B = targets.shape[0]
    T = enc_out.shape[1]
    U = targets.shape[1]
    device = targets.device

    enc_lengths = enc_lengths.clamp(min=1, max=T)

    if target_lengths.sum() == 0:
      return torch.zeros(B, device=device, dtype=torch.float32)

    active_mask = target_lengths > 0
    if not bool(active_mask.all()):
      active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
      active_loss = self._compute_per_sample_loss_numba_cpu(
          enc_out.index_select(0, active_idx),
          targets.index_select(0, active_idx),
          enc_lengths.index_select(0, active_idx),
          target_lengths.index_select(0, active_idx),
      )
      loss = torch.zeros(B, device=device, dtype=active_loss.dtype)
      loss = loss.index_copy(0, active_idx, active_loss)
      return loss

    T_max = int(enc_lengths.max().item())
    U_max = int(target_lengths.max().item())
    if T_max < T:
      enc_out = enc_out[:, :T_max, :]
    if U_max < U:
      targets = targets[:, :U_max].contiguous()
      U = U_max

    sos = torch.zeros(B, 1, dtype=torch.long, device=device)
    targets_sos = torch.cat([sos, targets], dim=1)

    embedded = self.pred_embedding(targets_sos)
    packed = nn.utils.rnn.pack_padded_sequence(
        embedded, (target_lengths + 1).cpu().clamp(min=1),
        batch_first=True, enforce_sorted=False)
    pred_out, _ = self.pred_rnn(packed)
    pred_out, _ = nn.utils.rnn.pad_packed_sequence(
        pred_out, batch_first=True, total_length=U + 1)

    enc_proj = self.enc_proj(enc_out).unsqueeze(2)
    pred_proj = self.pred_proj(pred_out).unsqueeze(1)
    logits = self.joint_out(enc_proj + pred_proj)
    
    from src.models.tdt_numba_cpu import tdt_forward_score_numba_cpu
    return tdt_forward_score_numba_cpu(
        logits, targets.long().contiguous(),
        enc_lengths.long(), target_lengths.long(),
        durations=self.durations, blank_id=self.blank_id, sigma=self.tdt_loss.sigma,
    )

  @torch.no_grad()
  def score_targets(self, enc_out, targets, enc_lengths, target_lengths,
                    normalize=True, method=None):
    """Score fixed target sequences under the TDT head.

    Returns larger-is-better log scores. By default this is the negative
    length-normalized TDT loss; set normalize=False for raw total sequence score.

    method:
      None / 'auto' — use FLAGS.tdt_score_method (default 'numba')
      'numba'        — NeMo TDTLossNumba via Numba CUDA JIT (fastest)
      'exact'        — pure PyTorch forward algorithm (marginal log-prob, slower)
      'numba_cpu'    — Numba CPU exact forward algorithm (fast, safe in docker)
      'forced_align' — forced greedy alignment (Viterbi approx, much faster)
    """
    if method is None or method == 'auto':
      method = str(getattr(FLAGS, 'tdt_score_method', 'numba') or 'numba')

    if method == 'forced_align':
      return self.score_targets_forced_align(
          enc_out, targets, enc_lengths, target_lengths, normalize=normalize)

    if method == 'numba':
      # Use Numba CUDA JIT (NeMo TDTLossNumba) — fastest path
      per_sample_loss = self._compute_per_sample_loss(
          enc_out, targets, enc_lengths, target_lengths)
    elif method == 'numba_cpu':
      # Safe fast CPU path
      per_sample_loss = self._compute_per_sample_loss_numba_cpu(
          enc_out, targets, enc_lengths, target_lengths)
    else:
      # 'exact': pure PyTorch forward algorithm (no Numba dependency)
      per_sample_loss = self._compute_per_sample_loss_pytorch(
          enc_out, targets, enc_lengths, target_lengths)

    if normalize:
      per_sample_loss = per_sample_loss / target_lengths.float().clamp(min=1)
    return -per_sample_loss

  @torch.no_grad()
  def score_token_ids(self, enc_out, targets, enc_lengths=None, target_lengths=None,
                      normalize=True):
    """Score padded token-id targets under the TDT head."""
    B, T, _ = enc_out.shape
    device = enc_out.device
    if enc_lengths is None:
      enc_lengths = torch.full((B,), T, dtype=torch.long, device=device)
    else:
      enc_lengths = enc_lengths.to(device).long().clamp(min=1, max=T)

    targets = targets.to(device).long()
    if target_lengths is None:
      target_lengths = (targets != self.blank_id).sum(dim=1)
    else:
      target_lengths = target_lengths.to(device).long().clamp(min=0)

    if targets.numel() == 0:
      return torch.zeros(B, dtype=enc_out.dtype, device=device)

    return self.score_targets(enc_out, targets, enc_lengths, target_lengths, normalize=normalize)

  @torch.no_grad()
  def score_targets_forced_align(self, enc_out, targets, enc_lengths,
                                 target_lengths, normalize=True):
    """Score targets via forced greedy alignment — O(T+U) steps, vectorized.

    Instead of the full TDT forward algorithm (O(T*U*D) with a huge T×U joint
    grid), this method:
      1. Runs the prediction LSTM on the full target sequence at once (batched).
      2. Pre-computes encoder and predictor projections.
      3. Walks a forced greedy alignment in lockstep across all B samples,
         batching joint network evaluations at each step.

    Returns larger-is-better log scores (same convention as score_targets).
    Scores are a Viterbi-like approximation — the log-probability of the best
    single alignment path rather than marginalised over all paths.
    """
    B, T, _ = enc_out.shape
    device = enc_out.device

    enc_lengths = enc_lengths.long().clamp(min=1, max=T)
    target_lengths = target_lengths.long().clamp(min=0)

    Tb_max = int(enc_lengths.max().item())
    Ub_max = int(target_lengths.max().item())
    if Ub_max == 0:
      return torch.zeros(B, device=device, dtype=torch.float32)

    # 1. Run prediction LSTM on full target sequence (batched, efficient)
    sos = torch.zeros(B, 1, dtype=torch.long, device=device)
    targets_clip = targets[:, :Ub_max].contiguous()
    targets_sos = torch.cat([sos, targets_clip], dim=1)  # (B, Ub_max+1)
    embedded = self.pred_embedding(targets_sos)
    packed = nn.utils.rnn.pack_padded_sequence(
        embedded, (target_lengths + 1).cpu().clamp(min=1),
        batch_first=True, enforce_sorted=False)
    pred_out, _ = self.pred_rnn(packed)
    pred_out, _ = nn.utils.rnn.pad_packed_sequence(
        pred_out, batch_first=True, total_length=Ub_max + 1)  # (B, Ub_max+1, pred_dim)

    # 2. Pre-compute encoder and predictor projections
    enc_proj = self.enc_proj(enc_out[:, :Tb_max, :])  # (B, Tb_max, joint_dim)
    pred_proj = self.pred_proj(pred_out)               # (B, Ub_max+1, joint_dim)

    # 3. Vectorized forced greedy alignment — all samples in parallel
    t_pos = torch.zeros(B, dtype=torch.long, device=device)
    u_pos = torch.zeros(B, dtype=torch.long, device=device)
    scores = torch.zeros(B, device=device, dtype=torch.float32)
    finished = (target_lengths <= 0) | (enc_lengths <= 0)

    max_steps = Tb_max + Ub_max + 100
    for _ in range(max_steps):
      active = ~finished
      if not active.any():
        break
      active_idx = active.nonzero(as_tuple=False).squeeze(-1)
      n_active = active_idx.numel()

      # Gather projections at current (t, u) for active samples
      a_t = t_pos[active_idx].clamp(max=Tb_max - 1)
      a_u = u_pos[active_idx].clamp(max=Ub_max)
      enc_at_t = enc_proj[active_idx, a_t]   # (n_active, joint_dim)
      pred_at_u = pred_proj[active_idx, a_u]  # (n_active, joint_dim)

      # Batched joint evaluation
      joint_out = self.joint_out(enc_at_t + pred_at_u)  # (n_active, V+D)
      tok_logits = joint_out[:, :self.vocab_size]
      dur_logits = joint_out[:, self.vocab_size:]
      log_tok = torch.log_softmax(tok_logits.float(), dim=-1)   # (n_active, V)
      log_dur = torch.log_softmax(dur_logits.float(), dim=-1)   # (n_active, D)

      dur_idx = dur_logits.argmax(dim=-1)  # (n_active,)
      dur_vals = torch.tensor(self.durations, device=device, dtype=torch.long)[dur_idx]
      dur_lp = log_dur.gather(1, dur_idx.unsqueeze(1)).squeeze(1)

      # Determine per-sample action: emit vs blank
      a_u_for_tgt = a_u.clamp(max=Ub_max - 1)
      tgt_ids = targets_clip[active_idx, a_u_for_tgt]  # (n_active,)
      target_lp = log_tok.gather(1, tgt_ids.unsqueeze(1)).squeeze(1)
      blank_lp = log_tok[:, self.blank_id]

      can_emit = u_pos[active_idx] < target_lengths[active_idx]
      do_emit = can_emit & (target_lp >= blank_lp)

      # Score accumulation: emit or blank
      step_token_lp = torch.where(do_emit, target_lp, blank_lp)
      scores[active_idx] += step_token_lp + dur_lp

      # State transitions
      # Emit: u += 1, t += dur if dur > 0
      emit_idx = active_idx[do_emit]
      if emit_idx.numel() > 0:
        u_pos[emit_idx] += 1
        emit_dur = dur_vals[do_emit]
        advance_emit = emit_dur > 0
        if advance_emit.any():
          t_pos[emit_idx[advance_emit]] += emit_dur[advance_emit]

      # Blank (including samples that finished emitting): t += max(dur, 1)
      blank_mask = ~do_emit
      blank_idx = active_idx[blank_mask]
      if blank_idx.numel() > 0:
        blank_dur = dur_vals[blank_mask].clamp(min=1)
        t_pos[blank_idx] += blank_dur

      finished = finished | (t_pos >= enc_lengths)

    if normalize:
      scores = scores / target_lengths.float().clamp(min=1)
    return scores

  @torch.no_grad()
  def greedy_decode(self, enc_out, enc_lengths=None, max_symbols_per_step=10,
                    max_decode_len=None):
    """Greedy TDT decoding with duration-aware frame skipping → IPA char IDs (B, L)."""
    B, T, _ = enc_out.shape
    device = enc_out.device

    if enc_lengths is None:
      enc_lengths = torch.full((B,), T, dtype=torch.long, device=device)
    enc_lengths = enc_lengths.long().clamp(min=0, max=T)
    max_decode_len = int(max_decode_len or getattr(FLAGS, 'max_new_tokens', 256))
    max_decode_len = max(max_decode_len, 1)

    state_dtype = enc_out.dtype
    h = torch.zeros(self.pred_rnn.num_layers, B, self.pred_dim,
            device=device, dtype=state_dtype)
    c = torch.zeros(self.pred_rnn.num_layers, B, self.pred_dim,
            device=device, dtype=state_dtype)
    last_token = torch.zeros(B, 1, dtype=torch.long, device=device)  # SOS=0
    t = torch.zeros(B, dtype=torch.long, device=device)
    out_lens = torch.zeros(B, dtype=torch.long, device=device)
    result = torch.zeros(B, max_decode_len, dtype=torch.long, device=device)
    duration_values = torch.tensor(self.durations, dtype=torch.long, device=device)

    finished = enc_lengths <= 0
    while True:
      finished = finished | (t >= enc_lengths) | (out_lens >= max_decode_len)
      active_idx = (~finished).nonzero(as_tuple=False).squeeze(-1)
      if active_idx.numel() == 0:
        break

      current_idx = active_idx

      for _ in range(max_symbols_per_step):
        if current_idx.numel() == 0:
          break

        current_t = t[current_idx]
        enc_proj_t = self.enc_proj(enc_out[current_idx, current_t])

        embedded = self.pred_embedding(last_token[current_idx])
        pred_t, (new_h, new_c) = self.pred_rnn(
            embedded,
            (h[:, current_idx, :].contiguous(), c[:, current_idx, :].contiguous()))
        if h.dtype != new_h.dtype:
          h = h.to(new_h.dtype)
        if c.dtype != new_c.dtype:
          c = c.to(new_c.dtype)

        pred_proj_t = self.pred_proj(pred_t.squeeze(1))
        joint = self.joint_out(enc_proj_t + pred_proj_t)  # (A, V+num_dur)

        token_logits = joint[:, :self.vocab_size]
        dur_logits = joint[:, self.vocab_size:]
        pred_ids = token_logits.argmax(dim=-1)
        dur_argmax = dur_logits.argmax(dim=-1)
        pred_durs = duration_values[dur_argmax]

        blank_mask = pred_ids == self.blank_id
        nonblank_mask = ~blank_mask

        if nonblank_mask.any():
          emit_idx = current_idx[nonblank_mask]
          emit_pos = out_lens[emit_idx]
          result[emit_idx, emit_pos] = pred_ids[nonblank_mask]
          out_lens[emit_idx] += 1
          last_token[emit_idx, 0] = pred_ids[nonblank_mask]
          h[:, emit_idx, :] = new_h[:, nonblank_mask, :]
          c[:, emit_idx, :] = new_c[:, nonblank_mask, :]

        if blank_mask.any():
          blank_idx = current_idx[blank_mask]
          t[blank_idx] += pred_durs[blank_mask].clamp_min(1)

        advance_mask = nonblank_mask & (pred_durs > 0)
        if advance_mask.any():
          advance_idx = current_idx[advance_mask]
          t[advance_idx] += pred_durs[advance_mask]

        keep_mask = nonblank_mask & (pred_durs == 0)
        if keep_mask.any():
          keep_idx = current_idx[keep_mask]
          keep_mask = out_lens[keep_idx] < max_decode_len
          current_idx = keep_idx[keep_mask]
        else:
          current_idx = current_idx[:0]

      # Match NeMo TDT greedy semantics: if a sample keeps emitting
      # duration=0 symbols up to max_symbols_per_step on the same frame,
      # force-advance one encoder step to avoid getting stuck in local loops.
      if current_idx.numel() > 0:
        t[current_idx] += 1

      finished = finished | (t >= enc_lengths) | (out_lens >= max_decode_len)

    max_len = int(out_lens.max().item()) if out_lens.numel() else 0
    max_len = max(max_len, 1)
    result = result[:, :max_len]
    return result

  @torch.no_grad()
  def beam_decode(self, enc_out, enc_lengths=None, beam_size=4, nbest=None,
                  max_symbols_per_step=10, max_decode_len=None):
    """Optional lightweight beam search for N-best export only."""
    B, T, _ = enc_out.shape
    device = enc_out.device

    if enc_lengths is None:
      enc_lengths = torch.full((B,), T, dtype=torch.long, device=device)
    enc_lengths = enc_lengths.long().clamp(min=0, max=T)
    max_decode_len = int(max_decode_len or getattr(FLAGS, 'max_new_tokens', 256))
    max_decode_len = max(max_decode_len, 1)
    beam_size = max(int(beam_size or 1), 1)
    nbest = max(int(nbest or beam_size), 1)

    hyps = []
    for i in range(B):
      hyps.append(
          self._beam_decode_single(
              enc_out[i:i + 1],
              int(enc_lengths[i].item()),
              beam_size=beam_size,
              nbest=nbest,
              max_symbols_per_step=max_symbols_per_step,
              max_decode_len=max_decode_len,
          )
      )
    return hyps

  @torch.no_grad()
  def _beam_decode_single(self, enc_out, enc_length, beam_size, nbest,
                          max_symbols_per_step, max_decode_len):
    device = enc_out.device
    state_dtype = enc_out.dtype
    duration_values = torch.tensor(self.durations, dtype=torch.long, device=device)
    enc_proj_all = self.enc_proj(enc_out[:, :max(enc_length, 1), :]).squeeze(0)

    def _new_state(score, t, tokens, last_token, h, c, same_frame_steps):
      return {
          'score': float(score),
          't': int(t),
          'tokens': tuple(tokens),
          'last_token': int(last_token),
          'h': h,
          'c': c,
          'same_frame_steps': int(same_frame_steps),
      }

    init_h = torch.zeros(self.pred_rnn.num_layers, 1, self.pred_dim,
                         device=device, dtype=state_dtype)
    init_c = torch.zeros(self.pred_rnn.num_layers, 1, self.pred_dim,
                         device=device, dtype=state_dtype)
    beam = [_new_state(0.0, 0, (), 0, init_h, init_c, 0)]
    max_steps = max(enc_length + max_decode_len * (max_symbols_per_step + 1) + 4, 1)

    for _ in range(max_steps):
      candidates = []
      all_finished = True
      for state in beam:
        if state['t'] >= enc_length or len(state['tokens']) >= max_decode_len:
          candidates.append(state)
          continue
        all_finished = False

        if state['same_frame_steps'] >= max_symbols_per_step:
          candidates.append(_new_state(
              state['score'],
              state['t'] + 1,
              state['tokens'],
              state['last_token'],
              state['h'],
              state['c'],
              0,
          ))
          continue

        enc_proj_t = enc_proj_all[state['t']].unsqueeze(0)
        last_token = torch.tensor([[state['last_token']]], dtype=torch.long, device=device)
        embedded = self.pred_embedding(last_token)
        pred_t, (new_h, new_c) = self.pred_rnn(embedded, (state['h'], state['c']))
        pred_proj_t = self.pred_proj(pred_t.squeeze(1))
        joint = self.joint_out(enc_proj_t + pred_proj_t).squeeze(0)

        token_logp = torch.log_softmax(joint[:self.vocab_size].float(), dim=-1)
        dur_logp = torch.log_softmax(joint[self.vocab_size:].float(), dim=-1)

        top_token_k = min(self.vocab_size, max(beam_size, nbest))
        top_token_scores, top_token_ids = torch.topk(token_logp, k=top_token_k)
        token_candidates = [(int(tid.item()), float(ts.item()))
                            for tid, ts in zip(top_token_ids, top_token_scores)]
        if self.blank_id not in {tid for tid, _ in token_candidates}:
          token_candidates.append((self.blank_id, float(token_logp[self.blank_id].item())))

        for token_id, token_score in token_candidates:
          for dur_idx in range(self.num_durations):
            duration = int(duration_values[dur_idx].item())
            score = state['score'] + token_score + float(dur_logp[dur_idx].item())
            if token_id == self.blank_id:
              candidates.append(_new_state(
                  score,
                  state['t'] + max(duration, 1),
                  state['tokens'],
                  state['last_token'],
                  state['h'],
                  state['c'],
                  0,
              ))
            else:
              tokens = state['tokens'] + (token_id,)
              next_t = state['t'] + duration if duration > 0 else state['t']
              next_same_frame_steps = 0 if duration > 0 else state['same_frame_steps'] + 1
              candidates.append(_new_state(
                  score,
                  next_t,
                  tokens,
                  token_id,
                  new_h,
                  new_c,
                  next_same_frame_steps,
              ))

      beam = sorted(candidates, key=lambda x: x['score'], reverse=True)[:beam_size]
      if all_finished or all((state['t'] >= enc_length or len(state['tokens']) >= max_decode_len)
                             for state in beam):
        break

    dedup = {}
    for state in beam:
      tokens = state['tokens']
      if tokens not in dedup or state['score'] > dedup[tokens]['score']:
        dedup[tokens] = state
    ranked = sorted(dedup.values(), key=lambda x: x['score'], reverse=True)
    return [list(state['tokens']) for state in ranked[:nbest]]


# ---- Self-contained attention pooling for aux heads (no external deps) ----

class _LinearAttentionPool(nn.Module):
  """Linear attention pooling: Dense -> softmax -> weighted sum.
  Equivalent to melt.layers.LinearAttentionPooling but in pure PyTorch.
  """
  def __init__(self, dim):
    super().__init__()
    self.score = nn.Linear(dim, 1)

  def forward(self, x, lengths=None):
    """
    Args:
      x: (B, T, D) encoder hidden states
      lengths: (B,) valid lengths (optional)
    Returns:
      (B, D) pooled representation
    """
    logits = self.score(x)  # (B, T, 1)
    if lengths is not None:
      mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
      logits = logits.masked_fill(mask.unsqueeze(-1), float('-inf'))
    alpha = torch.softmax(logits, dim=1)  # (B, T, 1)
    return (x * alpha).sum(dim=1)  # (B, D)


class _NonLinearAttentionPool(nn.Module):
  """Non-linear attention pooling: FFN(hidden->1) -> softmax -> weighted sum.
  Equivalent to melt.layers.NonLinearAttentionPooling but in pure PyTorch.
  """
  def __init__(self, dim):
    super().__init__()
    self.ffn = nn.Sequential(
      nn.Linear(dim, dim),
      nn.ReLU(),
      nn.Linear(dim, 1),
    )

  def forward(self, x, lengths=None):
    logits = self.ffn(x)  # (B, T, 1)
    if lengths is not None:
      mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
      logits = logits.masked_fill(mask.unsqueeze(-1), float('-inf'))
    alpha = torch.softmax(logits, dim=1)  # (B, T, 1)
    return (x * alpha).sum(dim=1)  # (B, D)


class BaseASRModel(nn.Module):
  """Base class for Pasketti ASR models.

  ctc_weight controls loss (independent of encoder type):
    0   -> pure seq2seq
    0~1 -> hybrid  loss = (1-w)*s2s + w*ctc
    1   -> pure CTC encoder-only

  Subclasses must implement:
    _encode(input_features) -> (B, T, encoder_dim)
    _s2s_forward(input_features, labels, encoder_hidden_states) -> loss
    _s2s_generate(input_features, encoder_hidden_states) -> token_ids (B, L)
    save_pretrained(path)
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.processor = get_processor(FLAGS.backbone)
    self.tokenizer = get_tokenizer(FLAGS.backbone)

    # If word_tokenizer=nemo/parakeet but backbone is non-NeMo (e.g. wavlm),
    # we need to load the NeMo SP tokenizer separately for the CTC head.
    word_tok = getattr(FLAGS, 'word_tokenizer', None)
    if (self.tokenizer is None
        and word_tok is not None
        and not hasattr(self, '_nemo_tokenizer')):
      _wt = word_tok.lower()
      _is_nemo_tok = (_wt in ('nemo', 'parakeet')
                      or any(k in _wt for k in ['parakeet', 'fastconformer', 'canary', 'conformer']))
      if _is_nemo_tok:
        from src.preprocess import load_nemo_tokenizer
        _nemo_bb = word_tok if word_tok not in ('nemo', 'parakeet') else 'parakeet-ctc-0.6b'
        logger.info('Loading standalone NeMo tokenizer for word_tokenizer=%s (backbone=%s)',
                     word_tok, FLAGS.backbone)
        _model_dir = getattr(FLAGS, "model_dir", "") or ""
        self._nemo_tokenizer = load_nemo_tokenizer(_nemo_bb, model_dir=_model_dir)
        logger.info('NeMo tokenizer loaded: vocab_size=%s', self._nemo_tokenizer.vocab_size)

    # Helper: pad_token_id that works even when tokenizer is None (NeMo)
    self._pad_token_id = (self.tokenizer.pad_token_id
                          if self.tokenizer is not None and self.tokenizer.pad_token_id is not None
                          else 0)

    self.ctc_weight = getattr(FLAGS, 'ctc_weight', 0.0)
    self.use_ctc = self.ctc_weight > 0
    self.ctc_only = self.ctc_weight >= 1.0

    # Default blank id before the concrete CTC head is built.
    # For HF BPE CTC we reserve one extra output class and move blank to the
    # last index in _init_ctc_head(); for char-level CTC blank stays at 0.
    self.ctc_blank_id = 0

    # IPA constrained decoding & training (for phonetic track)
    self.constrain_ipa = getattr(FLAGS, 'constrain_ipa', False)
    self._ipa_logits_processor = None
    self._ipa_mask = None          # (V,) bool, True = masked (non-IPA)
    self._ipa_allowed_ids = None
    if self.constrain_ipa and self.tokenizer is not None:
      allowed = build_ipa_allowed_ids(self.tokenizer)
      self._ipa_allowed_ids = allowed
      self._ipa_logits_processor = IPALogitsProcessor(
          allowed, self.tokenizer.vocab_size)
      # Mask will be built lazily in _mask_logits_ipa using actual logits dim
      # (tokenizer.vocab_size may < model config.vocab_size for Whisper)
      self._ipa_vocab_mask = None
      logger.info(f'IPA constrained vocab: {len(allowed)} allowed tokens '
                   f'out of {self.tokenizer.vocab_size} (training + decoding)')

    # CTC beam search LM (loaded lazily on first decode)
    self._ctc_lm = None
    self._ctc_lm_loaded = False

    # InterCTC: initialized by _init_inter_ctc() in subclass
    self._inter_ctc_enabled = False
    self._inter_hidden_states = []
    self._inter_ctc_layers = []

    # CTC layer fusion: initialized by _init_ctc_layer_fusion() in subclass
    self._ctc_layer_fusion_enabled = False
    self._ctc_fusion_hidden_states = []

    # Native CTC char tokenizer detection (e.g. HuBERT, Wav2Vec2-ft)
    # Wav2Vec2CTCTokenizer.decode() merges consecutive tokens by default;
    # since our ctc_decode already handles blank/repeat removal, we need
    # group_tokens=False to avoid double-decoding.
    self._is_native_ctc_char_tok = is_native_ctc_char_tokenizer(self.tokenizer)

    self.eval_keys = ['id']

  def save_model_meta(self, path, model_type, encoder_dim, **extra):
    """Save model_meta.json for submit.py auto-detection.
    
    Called at the end of save_pretrained() in each subclass.
    This makes the saved model directly loadable by submit.py
    without needing export_model.py.
    """
    import json as _json
    meta = {
        'model_type': model_type,
        'model_name': getattr(FLAGS, 'backbone', ''),
        'encoder_dim': encoder_dim,
        'eval_batch_size': getattr(FLAGS, 'eval_batch_size', 8),
    }
    if self.use_ctc and hasattr(self, 'ctc_head'):
      meta['ctc_vocab_size'] = self.ctc_head.proj.out_features
    meta.update(extra)
    meta_path = os.path.join(path, 'model_meta.json')
    with open(meta_path, 'w') as f:
      _json.dump(meta, f, indent=2)
    logger.info(f'model_meta.json saved to {meta_path}')

  def _tokenizer_batch_decode(self, token_ids, **kwargs):
    """Wrapper for tokenizer.batch_decode that handles CTC char tokenizers.

    Wav2Vec2CTCTokenizer.decode() merges consecutive tokens by default
    (group_tokens=True). Since our ctc_decode already removes blanks and
    collapses repeats, we must pass group_tokens=False to avoid double
    decoding (e.g. "HELLO" → "HELO").
    """
    if self.tokenizer is None:
      return [''] * (len(token_ids) if hasattr(token_ids, '__len__') else 1)
    if self._is_native_ctc_char_tok:
      return self.tokenizer.batch_decode(token_ids, group_tokens=False, **kwargs)
    return self.tokenizer.batch_decode(token_ids, **kwargs)

  def _decode_ctc_token_seqs_to_texts(self, token_seqs):
    """Decode CTC token-id sequences using the same tokenizer path as the main CTC head."""
    token_seqs = [list(seq) for seq in token_seqs]
    if self._ctc_char_level:
      return [''.join(IPA_ID_TO_CHAR.get(int(token_id), '') for token_id in seq)
              for seq in token_seqs]
    if self.tokenizer is not None:
      return self._tokenizer_batch_decode(token_seqs, skip_special_tokens=True)
    nemo_tok = getattr(self, '_nemo_tokenizer', None)
    if nemo_tok is not None:
      return [nemo_tok.ids_to_text(seq) if seq else '' for seq in token_seqs]
    return [''] * len(token_seqs)

  def _init_ctc_head(self, encoder_dim):
    """Initialize CTC head. Call from subclass __init__ after encoder is built.
    
    When constrain_ipa=True, uses character-level IPA vocabulary (~50 chars)
    instead of the full BPE vocabulary for CTC.
    """
    # Track char-level IPA mode even without CTC head (e.g. TDT-only)
    if not self.use_ctc:
      self._ctc_char_level = bool(self.constrain_ipa)

    if self.use_ctc:
      ctc_dropout = getattr(FLAGS, 'ctc_dropout', 0.1)

      if self.constrain_ipa:
        # Character-level IPA CTC: ~53 classes instead of 50k BPE tokens
        self._ctc_char_level = True
        if self.ctc_only:
          gz.set('ctc_char_level', True)
        self.ctc_head = CTCHead(encoder_dim, IPA_CTC_VOCAB_SIZE, dropout=ctc_dropout)
        self.ctc_loss_fn = nn.CTCLoss(blank=IPA_CTC_BLANK, reduction='none',
                                      zero_infinity=True)
        mode = 'ctc_only' if self.ctc_only else 'hybrid'
        logger.info(f'CTC head ({mode}, char-level IPA): dim={encoder_dim}, '
                     f'vocab={IPA_CTC_VOCAB_SIZE} ({len(IPA_CHAR_LIST)} chars + blank), '
                     f'ctc_weight={self.ctc_weight}')
      else:
        self._ctc_char_level = False
        tokenizer = self.tokenizer
        using_nemo_tokenizer = False
        if tokenizer is None:
          tokenizer = getattr(self, '_nemo_tokenizer', None)
          using_nemo_tokenizer = tokenizer is not None
        assert tokenizer is not None, (
            'Non-IPA CTC requires a tokenizer. '
            'Expected HF tokenizer or NeMo tokenizer fallback.')
        vocab_size = getattr(tokenizer, 'vocab_size', None)
        if vocab_size is None:
          tokenizer_impl = getattr(tokenizer, 'tokenizer', None)
          vocab_size = getattr(tokenizer_impl, 'vocab_size', None)
        assert vocab_size is not None, (
            'Failed to infer vocab size for non-IPA CTC tokenizer')
        # Reserve one extra class for the CTC blank at the last index.
        # HF tokenizer IDs and NeMo SentencePiece IDs both occupy
        # [0, orig_vocab_size-1], so CTCLoss needs an additional class.
        orig_vocab_size = vocab_size
        vocab_size = orig_vocab_size + 1
        self.ctc_blank_id = vocab_size - 1
        self.ctc_head = CTCHead(encoder_dim, vocab_size, dropout=ctc_dropout)
        self.ctc_loss_fn = nn.CTCLoss(blank=self.ctc_blank_id, reduction='none',
                                      zero_infinity=True)
        mode = 'ctc_only' if self.ctc_only else 'hybrid'
        logger.info(f'CTC head ({mode}): dim={encoder_dim}, '
                     f'vocab_size={vocab_size} (tokenizer={orig_vocab_size} + blank), '
                     f'ctc_weight={self.ctc_weight}, '
                     f'blank_id={self.ctc_blank_id}')

    self._init_word_ctc_head(encoder_dim)

  def _init_word_ctc_head(self, encoder_dim):
    """Initialize auxiliary word/pseudo-IPA CTC head independent of main CTC.

    This keeps --tdt and --tdt_only behavior aligned: changing ctc_weight only
    affects the primary IPA head, not whether the auxiliary word branch uses CTC
    or falls back to S2S.
    """
    if not getattr(FLAGS, 'word_ctc', False):
      return

    ctc_dropout = getattr(FLAGS, 'ctc_dropout', 0.1)
    self._word_ctc_bpe = getattr(FLAGS, 'word_ctc_bpe', False)
    self._pseudo_ipa_ctc = getattr(FLAGS, 'pseudo_ipa_ctc', False)

    if self._pseudo_ipa_ctc:
      # Pseudo-IPA CTC: same 53-class IPA vocab as primary CTC head.
      self._word_ctc_blank = IPA_CTC_BLANK
      self.word_ctc_head = CTCHead(encoder_dim, IPA_CTC_VOCAB_SIZE, dropout=ctc_dropout)
      self.word_ctc_loss_fn = nn.CTCLoss(blank=IPA_CTC_BLANK, reduction='none',
                                         zero_infinity=True)
      logger.info(f'Pseudo-IPA CTC head (auxiliary): dim={encoder_dim}, '
                   f'vocab={IPA_CTC_VOCAB_SIZE} (same as primary IPA CTC)')
    elif self._word_ctc_bpe:
      # BPE-level word CTC: use backbone tokenizer vocab.
      add_blank = bool(getattr(FLAGS, 'word_ctc_bpe_add_blank', False))
      self._word_ctc_bpe_add_blank = add_blank
      self._word_ctc_bpe_legacy_blank0 = not add_blank
      nemo_tok = getattr(self, '_nemo_tokenizer', None)
      if nemo_tok is not None:
        word_ctc_vocab = nemo_tok.vocab_size + (1 if add_blank else 0)
      elif self.tokenizer is not None:
        word_ctc_vocab = self.tokenizer.vocab_size + (1 if add_blank else 0)
      else:
        # Offline fallback: tokenizer unavailable (e.g. wavlm phonetic track).
        # Use whisper-large-v3 default vocab size so checkpoint weights load
        # correctly.  The BPE head is auxiliary and not used for IPA scoring.
        word_ctc_vocab = 50257 + (1 if add_blank else 0)
        if add_blank:
          logger.info('word_ctc_bpe: tokenizer unavailable, using default vocab_size=50258 (50257 + blank)')
        else:
          logger.info('word_ctc_bpe: tokenizer unavailable, using legacy default vocab_size=50257')
      # Backward compatibility: old wav2vec2-family checkpoints used a shared
      # vocab where id 0 is reserved as the CTC blank. In that legacy mode we
      # keep tokenizer ids unshifted and require real text pieces to avoid id 0.
      self._word_ctc_blank = 0
      self.word_ctc_head = CTCHead(encoder_dim, word_ctc_vocab, dropout=ctc_dropout)
      self.word_ctc_loss_fn = nn.CTCLoss(blank=self._word_ctc_blank, reduction='none',
                                         zero_infinity=True)
      logger.info(f'Word CTC head (auxiliary, BPE): dim={encoder_dim}, '
                   f'vocab={word_ctc_vocab}, add_blank={add_blank}, '
                   f'blank_id={self._word_ctc_blank}')
    else:
      # Char-level word CTC: 29 classes (a-z + space + apostrophe + blank).
      self._word_ctc_blank = WORD_CTC_BLANK
      self.word_ctc_head = CTCHead(encoder_dim, WORD_CTC_VOCAB_SIZE, dropout=ctc_dropout)
      self.word_ctc_loss_fn = nn.CTCLoss(blank=self._word_ctc_blank, reduction='none',
                                         zero_infinity=True)
      logger.info(f'Word CTC head (auxiliary, char): dim={encoder_dim}, '
                   f'vocab={WORD_CTC_VOCAB_SIZE} ({len(WORD_CHAR_LIST)} chars + blank)')

  def _init_inter_ctc(self, encoder_dim, num_encoder_layers):
    """Initialize InterCTC heads for intermediate encoder layers.
    
    Call from subclass __init__ after encoder is built and _init_ctc_head is done.
    Creates separate CTC heads for each specified intermediate layer.
    """
    self._inter_ctc_enabled = False
    if not self.use_ctc or not getattr(FLAGS, 'inter_ctc', False):
      return
    
    # Determine which layers to apply InterCTC
    layer_strs = getattr(FLAGS, 'inter_ctc_layers', [])
    if layer_strs:
      inter_layers = [int(x) for x in layer_strs]
    else:
      # Auto-select: layer at 1/2 depth
      inter_layers = [num_encoder_layers // 2]
    
    # Validate layer indices
    inter_layers = [l for l in inter_layers if 0 <= l < num_encoder_layers]
    if not inter_layers:
      logger.warning(f'InterCTC: no valid layers in {layer_strs} '
                     f'(encoder has {num_encoder_layers} layers)')
      return
    
    self._inter_ctc_layers = inter_layers
    self._inter_ctc_enabled = True
    
    # Create a CTC head for each intermediate layer (shared vocab with main CTC)
    ctc_dropout = getattr(FLAGS, 'ctc_dropout', 0.1)
    if self.constrain_ipa:
      vocab_size = IPA_CTC_VOCAB_SIZE
    else:
      vocab_size = self.ctc_head.proj.out_features
    
    self.inter_ctc_heads = nn.ModuleList([
      CTCHead(encoder_dim, vocab_size, dropout=ctc_dropout)
      for _ in inter_layers
    ])
    
    logger.info(f'InterCTC enabled: layers={inter_layers}, '
                f'weight={getattr(FLAGS, "inter_ctc_weight", 0.3)}, '
                f'num_heads={len(inter_layers)}')

  def _init_ctc_layer_fusion(self, num_encoder_layers):
    """Initialize learnable weighted layer fusion for CTC head.
    
    When --ctc_layer_fusion specifies encoder layer indices, the CTC head
    receives a *learned weighted average* of those layers instead of only
    the last encoder layer. Learns one scalar weight per layer (softmax-
    normalised), similar to ELMo / wav2vec2 weighted_layer_sum.
    
    Call from subclass __init__ after encoder is built.
    """
    self._ctc_layer_fusion_enabled = False
    self._ctc_fusion_layers = []
    self._ctc_fusion_hidden_states = []  # populated by _encode()

    if not self.use_ctc:
      return
    
    # ctc_fusion_last_n takes priority over ctc_layer_fusion
    last_n = getattr(FLAGS, 'ctc_fusion_last_n', None)
    if last_n is not None:
      assert last_n > 0, f'ctc_fusion_last_n must be positive, got {last_n}'
      assert last_n <= num_encoder_layers, (
          f'ctc_fusion_last_n={last_n} > num_encoder_layers={num_encoder_layers}')
      fusion_layers = list(range(num_encoder_layers - last_n, num_encoder_layers))
    else:
      layer_strs = getattr(FLAGS, 'ctc_layer_fusion', [])
      if not layer_strs:
        return
      fusion_layers = [int(x) for x in layer_strs]
    # Validate: allow negative indexing (e.g. -1 = last, -2 = second-to-last)
    resolved = []
    for idx in fusion_layers:
      if idx < 0:
        idx = num_encoder_layers + idx
      assert 0 <= idx < num_encoder_layers, (
          f'ctc_layer_fusion index {idx} out of range [0, {num_encoder_layers})')
      resolved.append(idx)

    self._ctc_fusion_layers = sorted(set(resolved))
    self._ctc_layer_fusion_enabled = True

    # Learnable weights: one per layer, initialised to equal weights
    n = len(self._ctc_fusion_layers)
    self.ctc_fusion_weights = nn.Parameter(torch.zeros(n))  # softmax(zeros) = uniform

    logger.info(f'CTC layer fusion: layers={self._ctc_fusion_layers}, '
                f'num_layers={n} (learnable weighted sum)')

  def _fuse_ctc_layers(self):
    """Compute weighted sum of cached intermediate layers for CTC head.
    
    Returns: (B, T, D) fused representation, or None if fusion is disabled
    or no hidden states are cached.
    """
    if not self._ctc_layer_fusion_enabled:
      return None
    hidden_states = self._ctc_fusion_hidden_states
    if not hidden_states or len(hidden_states) != len(self._ctc_fusion_layers):
      return None
    
    # hidden_states: list of (B, T, D)
    weights = F.softmax(self.ctc_fusion_weights, dim=0)  # (n,)
    stacked = torch.stack(hidden_states, dim=0)  # (n, B, T, D)
    fused = (weights[:, None, None, None] * stacked).sum(dim=0)  # (B, T, D)
    return fused

  def _init_aux_heads(self, encoder_dim):
    """Initialize auxiliary classification heads for metadata prediction.
    
    - Age head: predicts coarse age group (3-4 vs 5+)
      - classify: 2-class softmax, CE loss
      - ordinal: scalar sigmoid, BCE loss (threshold 0.5)
      - regress: scalar, MSE loss (target 3.5 / 6.0)
    - Domain head: predicts DD(1) vs EXT(0), scalar sigmoid, BCE loss
    
    Pooling: mean / linear_att / nonlinear_att (controlled by --aux_pool).
    """
    self._aux_age = getattr(FLAGS, 'aux_age_weight', 0) > 0
    self._aux_domain = getattr(FLAGS, 'aux_domain_weight', 0) > 0
    self._aux_age_mode = getattr(FLAGS, 'aux_age_mode', 'classify')
    self._aux_pool_method = getattr(FLAGS, 'aux_pool', 'mean')
    
    # Build pooling layer (shared by age & domain heads)
    if self._aux_age or self._aux_domain:
      if self._aux_pool_method == 'linear_att':
        self.aux_pool = _LinearAttentionPool(encoder_dim)
        logger.info(f'Aux pooling: linear_att (dim={encoder_dim})')
      elif self._aux_pool_method == 'nonlinear_att':
        self.aux_pool = _NonLinearAttentionPool(encoder_dim)
        logger.info(f'Aux pooling: nonlinear_att (dim={encoder_dim})')
      else:
        self.aux_pool = None  # use simple mean pooling
    
    # 4 age classes: 3-4=0, 5-7=1, 8-11=2, 12+=3
    self._age_num_classes = 4
    if self._aux_age:
      if self._aux_age_mode == 'classify':
        n_out = self._age_num_classes  # 4
        self.aux_age_head = nn.Linear(encoder_dim, n_out)
      elif self._aux_age_mode == 'ordinal':
        n_out = self._age_num_classes - 1  # 3 cumulative thresholds
        self.aux_age_head = nn.Linear(encoder_dim, n_out)
      else:  # regress
        n_out = 1
        self.aux_age_head = nn.Linear(encoder_dim, 1)
      logger.info(f'Aux age head: mode={self._aux_age_mode}, '
                  f'weight={getattr(FLAGS, "aux_age_weight", 0)}, '
                  f'pool={self._aux_pool_method}, '
                  f'out={n_out}, classes={self._age_num_classes} (3-4/5-7/8-11/12+)')
    
    if self._aux_domain:
      self.aux_domain_head = nn.Linear(encoder_dim, 1)
      logger.info(f'Aux domain head: weight={getattr(FLAGS, "aux_domain_weight", 0)}, '
                  f'pool={self._aux_pool_method}, DD=1, EXT=0')

    # ---- Length prediction heads: log(1+n_ipa_chars) and log(1+n_spaces) ----
    self._aux_nchars = getattr(FLAGS, 'aux_nchars_weight', 0) > 0
    self._aux_nspaces = getattr(FLAGS, 'aux_nspaces_weight', 0) > 0
    if self._aux_nchars or self._aux_nspaces:
      # Ensure pooling layer is built (may already exist from age/domain)
      if not (self._aux_age or self._aux_domain):
        if self._aux_pool_method == 'linear_att':
          self.aux_pool = _LinearAttentionPool(encoder_dim)
        elif self._aux_pool_method == 'nonlinear_att':
          self.aux_pool = _NonLinearAttentionPool(encoder_dim)
        else:
          self.aux_pool = None
    if self._aux_nchars:
      self.aux_nchars_head = nn.Linear(encoder_dim, 1)
      logger.info(f'Aux nchars head: weight={getattr(FLAGS, "aux_nchars_weight", 0)}, '
                  f'target=log(1+n_ipa_chars), MSE loss')
    if self._aux_nspaces:
      self.aux_nspaces_head = nn.Linear(encoder_dim, 1)
      logger.info(f'Aux nspaces head: weight={getattr(FLAGS, "aux_nspaces_weight", 0)}, '
                  f'target=log(1+n_spaces), MSE loss')

  def _compute_aux_losses(self, enc_out, inputs):
    """Compute auxiliary age/domain losses from mean-pooled encoder output.
    
    Args:
      enc_out: (B, T, D) encoder hidden states
      inputs: batch dict with age_label, age_mask, domain_label, domain_mask
    
    Returns:
      dict with 'aux_age_loss', 'aux_domain_loss' (B,) per-sample losses,
      and 'aux_age_logits', 'aux_domain_logits' for eval/reranker.
    """
    res = {}
    
    # Pool encoder output → (B, D)
    enc_len = getattr(self, '_last_enc_len', None)
    aux_pool = getattr(self, 'aux_pool', None)
    if aux_pool is not None:
      pooled = aux_pool(enc_out, enc_len)  # attention pooling handles masking internally
    elif enc_len is not None:
      B, T, D = enc_out.shape
      mask = torch.arange(T, device=enc_out.device).unsqueeze(0) < enc_len.unsqueeze(1)  # (B, T)
      masked = enc_out * mask.unsqueeze(-1).float()
      pooled = masked.sum(dim=1) / enc_len.unsqueeze(1).float().clamp(min=1)  # (B, D)
    else:
      pooled = enc_out.mean(dim=1)  # (B, D)
    
    if self._aux_age:
      # Always produce logits (needed for eval metrics even without labels)
      age_logits = self.aux_age_head(pooled)  # classify:(B,4), ordinal:(B,3), regress:(B,1)
      if self._aux_age_mode == 'regress':
        age_logits = age_logits.squeeze(-1)  # (B,)
      res['aux_age_logits'] = age_logits.detach()
      
      age_label = inputs.get('age_label', None)
      age_mask = inputs.get('age_mask', None)
      if age_label is not None and age_mask is not None:
        age_label = age_label.to(pooled.device)
        age_mask = age_mask.to(pooled.device)
        
        if self._aux_age_mode == 'classify':
          targets = age_label.long()  # class indices 0-3
          loss = F.cross_entropy(age_logits, targets, reduction='none')  # (B,)
        elif self._aux_age_mode == 'ordinal':
          # Ordinal encoding: class k → thresholds [1]*k + [0]*(K-1-k)
          K = self._age_num_classes - 1  # 3 thresholds
          targets = age_label.long()  # class indices 0-3
          ordinal_targets = torch.zeros(targets.shape[0], K, device=pooled.device)
          for k in range(K):
            ordinal_targets[:, k] = (targets > k).float()
          loss = F.binary_cross_entropy_with_logits(
              age_logits, ordinal_targets, reduction='none').mean(dim=-1)  # (B,)
        else:  # regress
          loss = F.mse_loss(age_logits, age_label, reduction='none')
        
        res['aux_age_loss'] = loss * age_mask  # (B,) zero for unknown age
        res['aux_age_mask'] = age_mask
    
    if self._aux_domain:
      logits = self.aux_domain_head(pooled).squeeze(-1)  # (B,)
      res['aux_domain_logits'] = logits.detach()
      domain_label = inputs.get('domain_label', None)
      domain_mask = inputs.get('domain_mask', None)
      if domain_label is not None and domain_mask is not None:
        domain_label = domain_label.to(pooled.device)
        domain_mask = domain_mask.to(pooled.device)
        loss = F.binary_cross_entropy_with_logits(logits, domain_label, reduction='none')
        res['aux_domain_loss'] = loss * domain_mask
        res['aux_domain_mask'] = domain_mask
    
    # ---- Length prediction: log(1+n_ipa_chars) and log(1+n_spaces) ----
    if self._aux_nchars:
      nchars_pred = self.aux_nchars_head(pooled).squeeze(-1)  # (B,)
      res['aux_nchars_pred'] = nchars_pred.detach()
      nchars_label = inputs.get('nchars_label', None)
      if nchars_label is not None:
        nchars_label = nchars_label.to(pooled.device)
        nchars_mask = inputs.get('nchars_mask', torch.ones(nchars_label.shape[0], device=pooled.device)).to(pooled.device)
        loss = F.smooth_l1_loss(nchars_pred, nchars_label, reduction='none')  # (B,)
        res['aux_nchars_loss'] = loss * nchars_mask
        res['aux_nchars_mask'] = nchars_mask

    if self._aux_nspaces:
      nspaces_pred = self.aux_nspaces_head(pooled).squeeze(-1)  # (B,)
      res['aux_nspaces_pred'] = nspaces_pred.detach()
      nspaces_label = inputs.get('nspaces_label', None)
      if nspaces_label is not None:
        nspaces_label = nspaces_label.to(pooled.device)
        nspaces_mask = inputs.get('nspaces_mask', torch.ones(nspaces_label.shape[0], device=pooled.device)).to(pooled.device)
        loss = F.smooth_l1_loss(nspaces_pred, nspaces_label, reduction='none')  # (B,)
        res['aux_nspaces_loss'] = loss * nspaces_mask
        res['aux_nspaces_mask'] = nspaces_mask
    
    return res

  def _compute_inter_ctc_losses(self, inter_hidden_states, labels):
    """Compute CTC losses for intermediate encoder layers.
    
    Args:
      inter_hidden_states: list of (B, T, D) tensors from intermediate layers
      labels: same labels used for main CTC
    
    Returns:
      inter_ctc_loss: (B,) averaged InterCTC loss across all layers
    """
    if not self._inter_ctc_enabled or not inter_hidden_states:
      return None
    
    all_losses = []
    for i, (hidden, head) in enumerate(zip(inter_hidden_states, self.inter_ctc_heads)):
      # Reuse same CTC loss computation logic
      ctc_logits = head(hidden)  # (B, T, V)
      log_probs = F.log_softmax(ctc_logits.float(), dim=-1)
      B, T, V = log_probs.shape
      log_probs_t = log_probs.transpose(0, 1)  # (T, B, V)
      
      enc_len = getattr(self, '_last_enc_len', None)
      if enc_len is not None:
        input_lengths = enc_len.to(log_probs.device).clamp(max=T)
      else:
        input_lengths = torch.full((B,), T, dtype=torch.long, device=log_probs.device)
      
      if self._ctc_char_level:
        # Char-level IPA: decode labels → text → char IDs
        texts = self._decode_labels_to_texts(labels)
        char_seqs = []
        for text in texts:
          text = _normalize_ipa(text)
          seq = [IPA_CHAR_TO_ID[ch] for ch in text if ch in IPA_CHAR_TO_ID]
          char_seqs.append(seq)
        target_lengths = torch.tensor([len(s) for s in char_seqs],
                                       dtype=torch.long, device=log_probs.device)
        if target_lengths.sum() == 0:
          continue
        targets_flat = torch.tensor([c for s in char_seqs for c in s],
                                     dtype=torch.long, device=log_probs.device)
        loss = self.ctc_loss_fn(log_probs_t, targets_flat, input_lengths, target_lengths)
        loss = loss / target_lengths.clamp(min=1).float()
      else:
        # BPE-level CTC
        pad_id = self.tokenizer.pad_token_id if self.tokenizer else 0
        target_mask = (labels != -100) & (labels != pad_id)
        target_lengths = target_mask.sum(dim=-1)
        targets_flat = torch.cat([labels[i_][target_mask[i_]] for i_ in range(B)])
        loss = self.ctc_loss_fn(log_probs_t, targets_flat, input_lengths, target_lengths)
        loss = loss / target_lengths.clamp(min=1).float()
      
      all_losses.append(loss)
    
    if not all_losses:
      return None
    
    # Average across all InterCTC layers → (B,)
    inter_loss = torch.stack(all_losses, dim=0).mean(dim=0)
    return inter_loss

  # ---- Custom S2S decoder initialization (Schemes 1/2/3) ----

  def _init_s2s_decoder(self, encoder_dim):
    """Initialize custom S2S decoder based on FLAGS.s2s_decoder.

    Call from subclass __init__ after encoder is built (alongside _init_ctc_head).
    Schemes:
      native       – subclass handles everything (default, no change)
      aed          – Transformer AED decoder (Scheme 2)
      rnnt_reuse   – subclass handles separately (Scheme 1, needs backbone LSTM)
      tdt_scratch  – scratch TDT decoder (tokenizer vocab when available)
      rnnt_custom  – custom RNNT from scratch (Scheme 3)
    """
    decoder_type = getattr(FLAGS, 's2s_decoder', 'native')
    self._s2s_decoder_type = decoder_type

    if getattr(FLAGS, 'word_tdt_pseudo_ipa', False):
      assert decoder_type == 'tdt_reuse', \
        '--word_tdt_pseudo_ipa requires --s2s_decoder=tdt_reuse'
    if getattr(FLAGS, 'word_tdt_mixed', False):
      assert decoder_type == 'tdt_reuse', \
        '--word_tdt_mixed requires --s2s_decoder=tdt_reuse'

    if decoder_type in ('native', 'rnnt_reuse', 'tdt_reuse'):
      # native: subclass _s2s_forward/_s2s_generate unchanged
      # rnnt_reuse/tdt_reuse: subclass (nemo.py) handles separately after this call
      if decoder_type in ('rnnt_reuse', 'tdt_reuse') and not self.constrain_ipa:
        logger.warning(f'{decoder_type} without --constrain_ipa is unusual — '
                       'custom decoders use IPA char-level vocabulary.')
      return

    if decoder_type == 'aed':
      vocab_size = getattr(FLAGS, 'aed_vocab_size', 0)
      if vocab_size == 0:
        vocab_size = AED_VOCAB_SIZE if self.constrain_ipa else self.tokenizer.vocab_size
      d_model = getattr(FLAGS, 'aed_dim', 256)
      nhead = getattr(FLAGS, 'aed_heads', 4)
      num_layers = getattr(FLAGS, 'aed_layers', 2)
      dropout = getattr(FLAGS, 'aed_dropout', 0.1)
      self.aed_decoder = AEDDecoder(
          encoder_dim=encoder_dim, d_model=d_model, nhead=nhead,
          num_layers=num_layers, vocab_size=vocab_size, dropout=dropout)
      if self.constrain_ipa:
        gz.set('s2s_ipa_chars', True)
      logger.info(f'AED decoder (Scheme 2): dim={d_model}, heads={nhead}, '
                   f'layers={num_layers}, vocab={vocab_size}')

    elif decoder_type == 'rnnt_custom':
      vocab_size = getattr(FLAGS, 'rnnt_vocab_size', 0)
      if vocab_size == 0:
        vocab_size = IPA_CTC_VOCAB_SIZE if self.constrain_ipa else self.tokenizer.vocab_size
      pred_dim = getattr(FLAGS, 'rnnt_pred_dim', 256)
      pred_layers = getattr(FLAGS, 'rnnt_pred_layers', 1)
      joint_dim = getattr(FLAGS, 'rnnt_joint_dim', 256)
      self.rnnt_decoder = CustomRNNTDecoder(
          encoder_dim=encoder_dim, vocab_size=vocab_size,
          pred_dim=pred_dim, pred_layers=pred_layers, joint_dim=joint_dim)
      if self.constrain_ipa:
        gz.set('s2s_ipa_chars', True)
      logger.info(f'Custom RNNT decoder (Scheme 3): pred_dim={pred_dim}, '
                   f'layers={pred_layers}, joint_dim={joint_dim}, vocab={vocab_size}')

    elif decoder_type == 'tdt_scratch':
      durations = [int(d) for d in getattr(FLAGS, 'tdt_durations', ['0', '1', '2', '3', '4'])]
      sigma = getattr(FLAGS, 'tdt_sigma', 0.02)
      omega = getattr(FLAGS, 'tdt_omega', 0.1)
      pred_dim = 640
      pred_layers = 2
      joint_dim = 640
      dropout = 0.2

      tokenizer = None if self.constrain_ipa else self._get_main_tdt_tokenizer()
      if tokenizer is not None:
        base_vocab_size = self._get_text_tokenizer_vocab_size(tokenizer)
        vocab_size = base_vocab_size + 1
        blank_id = 0
        padding_idx = None
        self._tdt_uses_tokenizer_vocab = True
        logger.info(
            'Scratch TDT main decoder: tokenizer-vocab mode '
            f'(tokenizer={type(tokenizer).__name__}, base_vocab={base_vocab_size}, '
            f'model_vocab={vocab_size}, blank_id={blank_id})')
      else:
        vocab_size = IPA_CTC_VOCAB_SIZE
        blank_id = 0
        padding_idx = 'auto'
        self._tdt_uses_tokenizer_vocab = False
        gz.set('s2s_ipa_chars', True)
        logger.info(
            'Scratch TDT main decoder: IPA-char mode '
            f'(vocab={vocab_size}, blank_id={blank_id})')

      self.tdt_decoder = CustomTDTDecoder(
          encoder_dim=encoder_dim,
          vocab_size=vocab_size,
          pred_dim=pred_dim,
          pred_layers=pred_layers,
          joint_dim=joint_dim,
          durations=durations,
          sigma=sigma,
          omega=omega,
          dropout=dropout,
          blank_id=blank_id,
          padding_idx=padding_idx,
      )
      logger.info(
          'Scratch TDT decoder initialized: '
          f'pred_dim={pred_dim}, layers={pred_layers}, joint_dim={joint_dim}, '
          f'durations={durations}, sigma={sigma}, omega={omega}, dropout={dropout}')

    else:
      raise ValueError(f'Unknown s2s_decoder: {decoder_type!r}. '
                       f'Choose from: native, aed, rnnt_reuse, tdt_reuse, tdt_scratch, rnnt_custom.')

  def _log_params(self):
    num_params = sum(p.numel() for p in self.parameters())
    num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
    logger.info(f'Model params: {num_params:,}, trainable: {num_trainable:,}')

    word_weight = getattr(FLAGS, 'word_weight', 0.0)
    if word_weight <= 0:
      return

    if hasattr(self, 'word_ctc_head'):
      if getattr(self, '_pseudo_ipa_ctc', False):
        word_mode = 'ctc:pseudo_ipa'
      elif getattr(self, '_word_ctc_bpe', False):
        word_mode = 'ctc:bpe'
      else:
        word_mode = 'ctc:char'
    elif getattr(FLAGS, 'word_tdt_mixed', False):
      word_mode = 'tdt:mixed_bpe:half_shared'
    elif getattr(FLAGS, 'word_tdt_pseudo_ipa', False):
      if getattr(FLAGS, 'word_tdt_half_share_decoder', False):
        word_mode = 'tdt:pseudo_ipa:half_shared'
      else:
        share = self._share_word_tdt_decoder()
        word_mode = f'tdt:pseudo_ipa:{"shared" if share else "separate"}'
    else:
      s2s_type = getattr(self, '_s2s_decoder_type', 'native')
      word_mode = f's2s:{s2s_type}'

    logger.info(
        'Word auxiliary branch: '
        f'mode={word_mode}, '
        f'weight={word_weight}, '
        f'use_cross_labels={getattr(FLAGS, "use_cross_labels", False)}, '
        f'word_ctc={getattr(FLAGS, "word_ctc", False)}, '
        f'word_ctc_head={hasattr(self, "word_ctc_head")}')

  def _share_word_tdt_decoder(self):
    share = getattr(FLAGS, 'word_tdt_share_decoder', None)
    if share is None:
      return True
    return bool(share)

  def _is_tdt_decoder_type(self, decoder_type=None):
    decoder_type = decoder_type or getattr(self, '_s2s_decoder_type', 'native')
    return decoder_type in ('tdt_reuse', 'tdt_scratch')

  def _get_text_tokenizer_vocab_size(self, tokenizer):
    vocab_size = getattr(tokenizer, 'vocab_size', None)
    if vocab_size is not None:
      return int(vocab_size)
    tokenizer_impl = getattr(tokenizer, 'tokenizer', None)
    if tokenizer_impl is not None and hasattr(tokenizer_impl, 'get_vocab_size'):
      return int(tokenizer_impl.get_vocab_size())
    raise ValueError(f'Failed to infer vocab size from tokenizer {type(tokenizer).__name__}')

  def _get_main_tdt_tokenizer(self):
    tokenizer = getattr(self, '_nemo_tokenizer', None)
    if tokenizer is not None:
      return tokenizer
    return self.tokenizer

  def _main_tdt_uses_tokenizer_vocab(self):
    return bool(getattr(self, '_tdt_uses_tokenizer_vocab', False))

  def _half_share_word_tdt_decoder(self):
    return bool(getattr(FLAGS, 'word_tdt_half_share_decoder', False))

  def _get_word_tdt_decoder(self):
    if getattr(FLAGS, 'word_tdt_mixed', False):
      assert hasattr(self, 'word_tdt_decoder'), \
        'word_tdt_mixed requires word_tdt_decoder to be initialized'
      return self.word_tdt_decoder
    if self._half_share_word_tdt_decoder():
      assert hasattr(self, 'word_tdt_decoder'), \
        'half-shared word TDT requested but word_tdt_decoder is not initialized'
      return self.word_tdt_decoder
    if self._share_word_tdt_decoder():
      assert hasattr(self, 'tdt_decoder'), 'shared word TDT requires main tdt_decoder'
      return self.tdt_decoder
    assert hasattr(self, 'word_tdt_decoder'), \
      'separate word TDT requested but word_tdt_decoder is not initialized'
    return self.word_tdt_decoder

  def _get_tdt_target_length_limit(self, device=None):
    limit = getattr(FLAGS, 'max_label_tokens', 0) or 0
    limit = int(limit) if limit else None
    if device is not None and getattr(device, 'type', None) == 'cuda' and torch.cuda.is_available():
      props = torch.cuda.get_device_properties(device)
      max_threads = int(getattr(props, 'max_threads_per_block', 1024))
      kernel_limit = max(max_threads - 1, 1)
      limit = kernel_limit if limit is None else min(limit, kernel_limit)
    return limit

  def _get_tdt_enc_lengths(self, enc_out):
    B = enc_out.shape[0]
    device = enc_out.device
    enc_lengths = getattr(self, '_last_enc_len', None)
    if enc_lengths is None:
      return torch.full((B,), enc_out.shape[1], dtype=torch.long, device=device)
    return enc_lengths.to(device).clamp(min=1, max=enc_out.shape[1])

  def _get_word_tdt_target_lengths(self, raw_texts):
    assert raw_texts is not None and len(raw_texts) > 0, \
      'word_tdt_pseudo_ipa requires non-empty word_label_texts'
    device = getattr(self, '_last_enc_out', None)
    device = device.device if device is not None else None
    char_seqs = self._texts_to_ipa_char_ids(
        raw_texts,
        max_len=self._get_tdt_target_length_limit(device),
    )
    return torch.tensor([len(seq) for seq in char_seqs], dtype=torch.float32, device=device)

  def _texts_to_word_tdt_token_ids(self, texts, max_len=None):
    assert hasattr(self, '_nemo_tokenizer') and self._nemo_tokenizer is not None, \
      'word_tdt_mixed requires a NeMo tokenizer'
    token_seqs = []
    for text in texts:
      ids = self._nemo_tokenizer.text_to_ids(text)
      if not ids:
        ids = [0]
      seq = [token_id + 1 for token_id in ids]
      if max_len is not None and len(seq) > max_len:
        if not getattr(self, '_word_tdt_target_trunc_warned', False):
          logger.warning(
              f'word_tdt_mixed target length exceeds kernel-safe limit {max_len}; '
              'truncating word BPE targets for TDT aux loss')
          self._word_tdt_target_trunc_warned = True
        seq = seq[:max_len]
      token_seqs.append(seq)
    return token_seqs

  def _texts_to_main_tdt_token_ids(self, texts, max_len=None):
    tokenizer = self._get_main_tdt_tokenizer()
    assert tokenizer is not None, 'tdt_scratch tokenizer-vocab mode requires a tokenizer'
    token_seqs = []
    for text in texts:
      if hasattr(tokenizer, 'text_to_ids'):
        ids = tokenizer.text_to_ids(text)
      else:
        ids = tokenizer.encode(text, add_special_tokens=False)
      if not ids:
        ids = [0]
      seq = [int(token_id) + 1 for token_id in ids]
      if max_len is not None and len(seq) > max_len:
        if not getattr(self, '_tdt_target_trunc_warned', False):
          logger.warning(
              f'main TDT target length exceeds kernel-safe limit {max_len}; '
              'truncating tokenizer targets for scratch TDT')
          self._tdt_target_trunc_warned = True
        seq = seq[:max_len]
      token_seqs.append(seq)
    return token_seqs

  def _decode_main_tdt_token_seqs_to_texts(self, token_seqs):
    if not self._main_tdt_uses_tokenizer_vocab():
      return [''.join(IPA_ID_TO_CHAR.get(int(token_id), '') for token_id in seq)
              for seq in token_seqs]

    tokenizer = self._get_main_tdt_tokenizer()
    assert tokenizer is not None, 'tdt_scratch tokenizer-vocab mode requires a tokenizer'
    raw_token_seqs = [[int(token_id) - 1 for token_id in seq if int(token_id) > 0]
                      for seq in token_seqs]
    if tokenizer is self.tokenizer and self.tokenizer is not None:
      return self._tokenizer_batch_decode(raw_token_seqs, skip_special_tokens=True)
    return [tokenizer.ids_to_text(seq) if seq else '' for seq in raw_token_seqs]

  def _get_word_tdt_mixed_target_lengths(self, raw_texts):
    assert raw_texts is not None and len(raw_texts) > 0, \
      'word_tdt_mixed requires non-empty word_label_texts'
    device = getattr(self, '_last_enc_out', None)
    device = device.device if device is not None else None
    token_seqs = self._texts_to_word_tdt_token_ids(
        raw_texts,
        max_len=self._get_tdt_target_length_limit(device),
    )
    return torch.tensor([len(seq) for seq in token_seqs], dtype=torch.float32, device=device)

  def _word_tdt_pseudo_ipa_forward(self, enc_out, raw_texts):
    assert getattr(FLAGS, 'word_tdt_pseudo_ipa', False), 'word_tdt_pseudo_ipa must be enabled'
    assert getattr(self, '_s2s_decoder_type', 'native') == 'tdt_reuse', \
      'word_tdt_pseudo_ipa requires --s2s_decoder=tdt_reuse'
    assert raw_texts is not None and len(raw_texts) > 0, \
      'word_tdt_pseudo_ipa requires word_label_texts'

    B = enc_out.shape[0]
    device = enc_out.device
    char_seqs = self._texts_to_ipa_char_ids(
      raw_texts,
      max_len=self._get_tdt_target_length_limit(device),
    )

    max_u = max(len(s) for s in char_seqs) if char_seqs else 1
    targets = torch.zeros(B, max_u, dtype=torch.long, device=device)
    target_lengths = torch.zeros(B, dtype=torch.long, device=device)
    for i, seq in enumerate(char_seqs):
      L = len(seq)
      if L > 0:
        targets[i, :L] = torch.tensor(seq, dtype=torch.long, device=device)
      target_lengths[i] = L

    enc_lengths = self._get_tdt_enc_lengths(enc_out)

    decoder = self._get_word_tdt_decoder()
    return decoder(enc_out, targets, enc_lengths, target_lengths)

  def _word_tdt_mixed_forward(self, enc_out, raw_texts):
    assert getattr(FLAGS, 'word_tdt_mixed', False), 'word_tdt_mixed must be enabled'
    assert getattr(self, '_s2s_decoder_type', 'native') == 'tdt_reuse', \
      'word_tdt_mixed requires --s2s_decoder=tdt_reuse'
    assert raw_texts is not None and len(raw_texts) > 0, \
      'word_tdt_mixed requires word_label_texts'

    B = enc_out.shape[0]
    device = enc_out.device
    token_seqs = self._texts_to_word_tdt_token_ids(
        raw_texts,
        max_len=self._get_tdt_target_length_limit(device),
    )

    max_u = max(len(seq) for seq in token_seqs) if token_seqs else 1
    targets = torch.zeros(B, max_u, dtype=torch.long, device=device)
    target_lengths = torch.zeros(B, dtype=torch.long, device=device)
    for i, seq in enumerate(token_seqs):
      L = len(seq)
      if L > 0:
        targets[i, :L] = torch.tensor(seq, dtype=torch.long, device=device)
      target_lengths[i] = L

    enc_lengths = self._get_tdt_enc_lengths(enc_out)
    decoder = self._get_word_tdt_decoder()
    return decoder(enc_out, targets, enc_lengths, target_lengths)

  # ---- Subclass interface (must override) ----

  def _encode(self, input_features, attention_mask=None):
    """Encode audio features -> (B, T, encoder_dim)."""
    raise NotImplementedError

  def _s2s_forward(self, input_features, labels, encoder_hidden_states, attention_mask=None):
    """Seq2seq forward pass -> loss scalar."""
    raise NotImplementedError

  def _s2s_generate(self, input_features, encoder_hidden_states, attention_mask=None):
    """Seq2seq generate -> token_ids (B, L).
    Subclass should call self._get_logits_processors() to apply IPA constraint."""
    raise NotImplementedError

  def _get_logits_processors(self):
    """Return list of logits processors for constrained generation."""
    if self._ipa_logits_processor is not None:
      return [self._ipa_logits_processor]
    return None

  # ---- Custom S2S decoder helpers ----

  def _decode_labels_to_texts(self, labels, raw_texts=None):
    """Decode label tensor to text strings.
    
    Prefer raw label_texts (avoids lossy BPE roundtrip) for both NeMo and Whisper.
    Falls back to tokenizer.batch_decode() only when label_texts unavailable.
    """
    label_texts = raw_texts if raw_texts is not None else getattr(self, '_current_label_texts', None)
    if label_texts and any(label_texts):
      return list(label_texts)
    # Fallback: BPE decode (only when label_texts not available)
    if self.tokenizer is None:
      raise RuntimeError('No label_texts and no tokenizer — cannot decode labels')
    clean = labels.clone()
    clean[clean == -100] = self.tokenizer.pad_token_id or 0
    return self._tokenizer_batch_decode(clean.long().cpu().numpy(),
                                        skip_special_tokens=True)

  def _labels_to_ipa_char_ids(self, labels, raw_texts=None, max_len=None):
    """Convert BPE label tensor → list of IPA char-ID sequences.
    Shared by AED, custom RNNT, and rnnt_reuse decoders.
    """
    texts = self._decode_labels_to_texts(labels, raw_texts=raw_texts)
    return self._texts_to_ipa_char_ids(texts, max_len=max_len)

  def _texts_to_ipa_char_ids(self, texts, max_len=None):
    """Convert text strings → list of IPA char-ID sequences."""
    char_seqs = []
    for text in texts:
      text = _normalize_ipa(text)
      seq = [IPA_CHAR_TO_ID[ch] for ch in text if ch in IPA_CHAR_TO_ID]
      if max_len is not None and len(seq) > max_len:
        if not getattr(self, '_tdt_target_trunc_warned', False):
          logger.warning(
              'Truncating TDT target sequence to fit kernel/label limit: '
              f'orig_len={len(seq)} max_len={max_len}')
          self._tdt_target_trunc_warned = True
        seq = seq[:max_len]
      char_seqs.append(seq)
    return char_seqs

  @torch.no_grad()
  def score_tdt_texts(self, enc_out, texts, enc_lengths=None, normalize=True):
    """Score candidate IPA texts with the TDT head.

    If enc_out batch size is 1 and multiple candidate texts are provided,
    the encoder output is broadcast so one utterance can score an n-best list.
    Returns larger-is-better log scores.
    """
    assert hasattr(self, 'tdt_decoder'), 'score_tdt_texts requires a TDT decoder'
    assert texts is not None and len(texts) > 0, 'texts must be a non-empty list'

    if self._main_tdt_uses_tokenizer_vocab():
      target_seqs = self._texts_to_main_tdt_token_ids(texts)
    else:
      target_seqs = self._texts_to_ipa_char_ids(texts)
    n_texts = len(target_seqs)
    batch_size = enc_out.shape[0]

    if batch_size == 1 and n_texts > 1:
      enc_out = enc_out.expand(n_texts, -1, -1)
      if enc_lengths is not None:
        enc_lengths = enc_lengths.expand(n_texts)
    else:
      assert batch_size == n_texts, (
          f'enc_out batch ({batch_size}) must match len(texts) ({n_texts}), '
          'or enc_out batch must be 1 for broadcasting')

    device = enc_out.device
    max_u = max(len(seq) for seq in target_seqs) if target_seqs else 1
    targets = torch.zeros(n_texts, max_u, dtype=torch.long, device=device)
    target_lengths = torch.zeros(n_texts, dtype=torch.long, device=device)
    for i, seq in enumerate(target_seqs):
      length = len(seq)
      if length > 0:
        targets[i, :length] = torch.tensor(seq, dtype=torch.long, device=device)
      target_lengths[i] = length

    if enc_lengths is None:
      enc_lengths = torch.full((n_texts,), enc_out.shape[1], dtype=torch.long, device=device)
    else:
      enc_lengths = enc_lengths.to(device).long().view(-1)
      if enc_lengths.numel() == 1 and n_texts > 1:
        enc_lengths = enc_lengths.expand(n_texts)

    return self.tdt_decoder.score_targets(
        enc_out, targets, enc_lengths, target_lengths, normalize=normalize)

  @torch.no_grad()
  def score_ctc_texts(self, enc_out, texts, ctc_logits=None, enc_lengths=None):
    """Score candidate IPA texts with the CTC head for a single utterance.

    Returns exact CTC forward scores for one acoustic input against many text
    candidates. Larger is better.
    """
    assert self.use_ctc and getattr(self, '_ctc_char_level', False), (
        'score_ctc_texts requires a character-level CTC head')
    assert enc_out.shape[0] == 1, 'score_ctc_texts currently expects enc_out batch size = 1'
    assert texts is not None and len(texts) > 0, 'texts must be a non-empty list'

    if ctc_logits is None:
      fused = self._fuse_ctc_layers()
      ctc_input = fused if fused is not None else enc_out
      ctc_logits = self.ctc_head(ctc_input)

    if getattr(FLAGS, 'ctc_decode_fp32', False):
      log_probs = F.log_softmax(ctc_logits.float(), dim=-1)
    else:
      log_probs = F.log_softmax(ctc_logits, dim=-1)

    if enc_lengths is None:
      enc_lengths = getattr(self, '_last_enc_len', None)
    if enc_lengths is not None:
      B, T, V = log_probs.shape
      lengths = enc_lengths.to(log_probs.device).clamp(max=T)
      range_t = torch.arange(T, device=log_probs.device).unsqueeze(0)
      mask = range_t >= lengths.unsqueeze(1)
      blank_row = torch.full((V,), float('-inf'), device=log_probs.device)
      blank_row[IPA_CTC_BLANK] = 0.0
      log_probs = log_probs.clone()
      log_probs[mask] = blank_row

    cand_token_ids = self._texts_to_ipa_char_ids(texts)
    scores = ctc_force_score_batch(
        log_probs[0].float().cpu().numpy(), cand_token_ids, blank=IPA_CTC_BLANK)
    return torch.tensor(scores, dtype=torch.float32, device=enc_out.device)

  def _dispatch_s2s_forward(self, input_features, labels, enc_out,
                            attention_mask=None, raw_texts=None):
    """Route S2S training to appropriate decoder."""
    dt = getattr(self, '_s2s_decoder_type', 'native')
    if dt == 'native':
      return self._s2s_forward(input_features, labels, enc_out,
                               attention_mask=attention_mask)
    elif dt == 'aed':
      return self._aed_s2s_forward(enc_out, labels, raw_texts=raw_texts)
    elif dt in ('rnnt_custom', 'rnnt_reuse'):
      return self._rnnt_custom_s2s_forward(enc_out, labels, raw_texts=raw_texts)
    elif dt in ('tdt_reuse', 'tdt_scratch'):
      return self._tdt_reuse_s2s_forward(enc_out, labels, raw_texts=raw_texts)
    raise ValueError(f'Unknown s2s_decoder: {dt}')

  def _dispatch_s2s_generate(self, input_features, enc_out,
                             attention_mask=None):
    """Route S2S generation to appropriate decoder."""
    dt = getattr(self, '_s2s_decoder_type', 'native')
    if dt == 'native':
      return self._s2s_generate(input_features, enc_out,
                                attention_mask=attention_mask)
    elif dt == 'aed':
      return self._aed_generate(enc_out)
    elif dt in ('rnnt_custom', 'rnnt_reuse'):
      return self._rnnt_custom_generate(enc_out)
    elif dt in ('tdt_reuse', 'tdt_scratch'):
      return self._tdt_reuse_generate(enc_out)
    raise ValueError(f'Unknown s2s_decoder: {dt}')

  # ---- AED (Scheme 2) forward / generate ----

  def _aed_s2s_forward(self, enc_out, labels, raw_texts=None):
    """AED teacher-forced cross-entropy → per-sample loss (B,).
    
    With --aed_scheduled_sampling > 0, randomly replaces some teacher-forced
    tokens with the model's own greedy predictions (reduces exposure bias).
    """
    char_seqs = self._labels_to_ipa_char_ids(labels, raw_texts=raw_texts)
    B = enc_out.shape[0]
    device = enc_out.device

    # Truncate sequences that exceed pos_enc capacity
    pos_limit = self.aed_decoder.max_pos - 1  # reserve 1 for EOS
    char_seqs = [s[:pos_limit] for s in char_seqs]

    # Build decoder inputs [SOS, c1, ..., cN] and targets [c1, ..., cN, EOS]
    max_len = max(len(s) for s in char_seqs) + 1  # +1 for EOS
    dec_input = torch.zeros(B, max_len, dtype=torch.long, device=device)
    dec_target = torch.full((B, max_len), -100, dtype=torch.long, device=device)

    for i, seq in enumerate(char_seqs):
      L = len(seq)
      if L > 0:
        dec_input[i, 1:L + 1] = torch.tensor(seq, dtype=torch.long)
        dec_target[i, :L] = torch.tensor(seq, dtype=torch.long)
      dec_target[i, L] = self.aed_decoder.eos_id

    # ---- Scheduled sampling: mix teacher-forcing with model predictions ----
    ss_prob = getattr(FLAGS, 'scheduled_sampling', 0.0)
    if ss_prob > 0 and self.training and max_len > 1:
      enc_lengths = getattr(self, '_last_enc_len', None)
      memory = self.aed_decoder.enc_proj(enc_out)
      memory_kpm = None
      if enc_lengths is not None:
        T = memory.shape[1]
        rng = torch.arange(T, device=device).unsqueeze(0)
        memory_kpm = rng >= enc_lengths.unsqueeze(1)

      # Step-by-step: for positions 1..max_len-1, decide whether to use
      # teacher token or model's own prediction from the previous step.
      mixed_input = dec_input.clone()
      for t in range(1, max_len):
        # Decode up to position t-1 to get prediction for position t's input
        if torch.rand(1).item() < ss_prob:
          L_cur = t
          positions = torch.arange(L_cur, device=device).unsqueeze(0)
          tgt_emb = self.aed_decoder.embedding(mixed_input[:, :L_cur]) + \
                    self.aed_decoder.pos_enc(positions)
          tgt_emb = self.aed_decoder.embed_dropout(tgt_emb)
          causal = nn.Transformer.generate_square_subsequent_mask(L_cur, device=device)
          dec_out = self.aed_decoder.decoder(
              tgt=tgt_emb, memory=memory,
              tgt_mask=causal,
              memory_key_padding_mask=memory_kpm)
          pred_t = self.aed_decoder.output_proj(dec_out[:, -1, :]).argmax(dim=-1)
          mixed_input[:, t] = pred_t
      dec_input = mixed_input

    enc_lengths = getattr(self, '_last_enc_len', None)
    logits = self.aed_decoder(enc_out, dec_input, enc_lengths=enc_lengths)

    V = logits.shape[-1]
    label_smoothing = getattr(FLAGS, 'label_smoothing', 0.0)
    per_token = F.cross_entropy(
        logits.view(-1, V), dec_target.view(-1),
        ignore_index=-100, reduction='none',
        label_smoothing=label_smoothing,
    ).view(B, max_len)
    mask = (dec_target != -100).float()

    # ---- Cheap batch-level S2S metric: teacher-forced argmax → text ----
    # Almost free: logits are already computed, just argmax + ID→char lookup.
    if not self.training:
      with torch.no_grad():
        pred_ids = logits.argmax(dim=-1)  # (B, max_len)
        s2s_texts = []
        for i in range(B):
          chars = []
          for t in range(max_len):
            if dec_target[i, t].item() == -100:
              break
            cid = pred_ids[i, t].item()
            ch = IPA_ID_TO_CHAR.get(cid)
            if ch is not None:
              chars.append(ch)
          s2s_texts.append(''.join(chars))
        self._s2s_pred_texts = s2s_texts

    return (per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

  def _aed_generate(self, enc_out):
    """AED autoregressive decode → padded IPA char IDs (B, max_new_tokens)."""
    enc_lengths = getattr(self, '_last_enc_len', None)
    max_len = getattr(FLAGS, 'max_new_tokens', 256)
    beam_size = getattr(FLAGS, 'num_beams', 1)
    length_penalty = getattr(FLAGS, 'length_penalty', 1.0)
    result = self.aed_decoder.generate(enc_out, enc_lengths=enc_lengths,
                                       max_len=max_len,
                                       beam_size=beam_size,
                                       length_penalty=length_penalty)
    B = enc_out.shape[0]
    cur = result.shape[1]
    if cur < max_len:
      pad = torch.zeros(B, max_len - cur, dtype=torch.long, device=enc_out.device)
      result = torch.cat([result, pad], dim=1)
    else:
      result = result[:, :max_len]
    return result

  # ---- Custom / reuse RNNT (Schemes 1 & 3) forward / generate ----

  def _rnnt_custom_s2s_forward(self, enc_out, labels, raw_texts=None):
    """Custom RNNT loss → per-sample loss (B,)."""
    char_seqs = self._labels_to_ipa_char_ids(labels, raw_texts=raw_texts)
    B = enc_out.shape[0]
    device = enc_out.device

    max_u = max(len(s) for s in char_seqs) if char_seqs else 1
    targets = torch.zeros(B, max_u, dtype=torch.long, device=device)
    target_lengths = torch.zeros(B, dtype=torch.long, device=device)
    for i, seq in enumerate(char_seqs):
      L = len(seq)
      if L > 0:
        targets[i, :L] = torch.tensor(seq, dtype=torch.long)
      target_lengths[i] = L

    enc_lengths = getattr(self, '_last_enc_len', None)
    if enc_lengths is None:
      enc_lengths = torch.full((B,), enc_out.shape[1],
                               dtype=torch.long, device=device)
    else:
      enc_lengths = enc_lengths.to(device).clamp(max=enc_out.shape[1])
    return self.rnnt_decoder(enc_out, targets, enc_lengths, target_lengths)

  def _rnnt_custom_generate(self, enc_out):
    """Custom RNNT decode → padded IPA char IDs (B, max_new_tokens).
    Uses greedy or beam search depending on --num_beams."""
    enc_lengths = getattr(self, '_last_enc_len', None)
    max_len = getattr(FLAGS, 'max_new_tokens', 256)
    beam_size = getattr(FLAGS, 'num_beams', 1)
    if beam_size > 1:
      result = self.rnnt_decoder.beam_decode(enc_out, enc_lengths,
                                             beam_size=beam_size)
    else:
      result = self.rnnt_decoder.greedy_decode(enc_out, enc_lengths)
    B = enc_out.shape[0]
    cur = result.shape[1]
    if cur < max_len:
      pad = torch.zeros(B, max_len - cur, dtype=torch.long, device=enc_out.device)
      result = torch.cat([result, pad], dim=1)
    else:
      result = result[:, :max_len]
    return result

  # ---- TDT-reuse (Scheme 4) forward / generate ----

  def _tdt_reuse_s2s_forward(self, enc_out, labels, raw_texts=None):
    """TDT loss with duration prediction → per-sample loss (B,)."""
    if self._main_tdt_uses_tokenizer_vocab():
      texts = self._decode_labels_to_texts(labels, raw_texts=raw_texts)
      char_seqs = self._texts_to_main_tdt_token_ids(
          texts,
          max_len=self._get_tdt_target_length_limit(enc_out.device),
      )
    else:
      char_seqs = self._labels_to_ipa_char_ids(
          labels,
          raw_texts=raw_texts,
          max_len=self._get_tdt_target_length_limit(enc_out.device),
      )
    B = enc_out.shape[0]
    device = enc_out.device

    max_u = max(len(s) for s in char_seqs) if char_seqs else 1
    targets = torch.zeros(B, max_u, dtype=torch.long, device=device)
    target_lengths = torch.zeros(B, dtype=torch.long, device=device)
    for i, seq in enumerate(char_seqs):
      L = len(seq)
      if L > 0:
        targets[i, :L] = torch.tensor(seq, dtype=torch.long)
      target_lengths[i] = L

    enc_lengths = getattr(self, '_last_enc_len', None)
    if enc_lengths is None:
      enc_lengths = torch.full((B,), enc_out.shape[1],
                               dtype=torch.long, device=device)
    else:
      enc_lengths = enc_lengths.to(device).clamp(max=enc_out.shape[1])
    return self.tdt_decoder(enc_out, targets, enc_lengths, target_lengths)

  def _score_tdt_candidate_groups(self, enc_out, enc_lengths, candidate_groups,
                                  normalize=True):
    """Batch-score grouped TDT token candidates for the current batch."""
    assert hasattr(self, 'tdt_decoder'), '_score_tdt_candidate_groups requires a TDT decoder'
    batch_size = enc_out.shape[0]
    assert len(candidate_groups) == batch_size, (
        f'candidate_groups size mismatch: {len(candidate_groups)} vs batch {batch_size}')

    flat_owner = []
    flat_seqs = []
    dedup_groups = []
    for sample_idx, group in enumerate(candidate_groups):
      seen = set()
      dedup = []
      for seq in group:
        seq_t = tuple(int(token_id) for token_id in seq if int(token_id) != 0)
        if seq_t in seen:
          continue
        seen.add(seq_t)
        dedup.append(seq_t)
      dedup_groups.append(dedup)
      for seq_t in dedup:
        flat_owner.append(sample_idx)
        flat_seqs.append(seq_t)

    rows = [[] for _ in range(batch_size)]
    if not flat_seqs:
      return rows

    device = enc_out.device
    owner_idx = torch.tensor(flat_owner, dtype=torch.long, device=device)
    max_u = max(max((len(seq) for seq in flat_seqs), default=0), 1)
    targets = torch.zeros(len(flat_seqs), max_u, dtype=torch.long, device=device)
    target_lengths = torch.zeros(len(flat_seqs), dtype=torch.long, device=device)
    for idx, seq in enumerate(flat_seqs):
      if seq:
        targets[idx, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
      target_lengths[idx] = len(seq)

    enc_rep = enc_out.index_select(0, owner_idx)
    enc_len_rep = enc_lengths.index_select(0, owner_idx)
    scores = self.tdt_decoder.score_targets(
        enc_rep,
        targets,
        enc_len_rep,
        target_lengths,
        normalize=normalize,
    ).detach().cpu().tolist()

    for sample_idx, seq_t, score in zip(flat_owner, flat_seqs, scores):
      rows[sample_idx].append((seq_t, float(score)))
    return rows

  def _tdt_reuse_generate(self, enc_out):
    """TDT decode with duration-aware frame skipping → padded IPA char IDs (B, max_new_tokens)."""
    enc_lengths = getattr(self, '_last_enc_len', None)
    max_len = getattr(FLAGS, 'max_new_tokens', 256)
    result = self.tdt_decoder.greedy_decode(
        enc_out,
        enc_lengths,
        max_decode_len=max_len)
    B = enc_out.shape[0]
    cur = result.shape[1]
    if cur < max_len:
      pad = torch.zeros(B, max_len - cur, dtype=torch.long, device=enc_out.device)
      result = torch.cat([result, pad], dim=1)
    else:
      result = result[:, :max_len]
    # Convert internal TDT token IDs → text strings for pred_texts
    greedy_token_seqs = []
    for i in range(B):
      token_ids = []
      for j in range(result.shape[1]):
        cid = result[i, j].item()
        if cid == 0:
          break
        token_ids.append(cid)
      greedy_token_seqs.append(token_ids)
    texts = self._decode_main_tdt_token_seqs_to_texts(greedy_token_seqs)
    self._last_pred_texts = texts
    self._last_decode_meta = None

    want_score = bool(getattr(FLAGS, 'save_pred_score', False))
    want_nbest = int(getattr(FLAGS, 'save_pred_nbest', 0) or 0)
    if want_score or want_nbest > 0:
      if enc_lengths is None:
        enc_lengths = torch.full((B,), enc_out.shape[1], dtype=torch.long, device=enc_out.device)
      else:
        enc_lengths = enc_lengths.to(enc_out.device).long().clamp(min=0, max=enc_out.shape[1])

      candidate_groups = [[tuple(seq)] for seq in greedy_token_seqs]
      if want_nbest > 0:
        beam_size = max(int(getattr(FLAGS, 'num_beams', 1) or 1), want_nbest)
        beam_candidates = self.tdt_decoder.beam_decode(
            enc_out,
            enc_lengths,
            beam_size=beam_size,
            nbest=want_nbest,
            max_decode_len=max_len,
        )
        for i in range(B):
          candidate_groups[i].extend(beam_candidates[i])

      scored_groups = self._score_tdt_candidate_groups(
          enc_out,
          enc_lengths,
          candidate_groups,
          normalize=True,
      )

      rows = []
      for i in range(B):
        greedy_seq = tuple(greedy_token_seqs[i])
        score_map = {seq: score for seq, score in scored_groups[i]}
        row = {}
        if want_score:
          row['pred_score'] = score_map.get(greedy_seq, np.nan)
        if want_nbest > 0:
          ranked = sorted(scored_groups[i], key=lambda x: x[1], reverse=True)[:want_nbest]
          row['pred_nbest_texts'] = self._decode_main_tdt_token_seqs_to_texts(
              [list(seq) for seq, _ in ranked])
          row['pred_nbest_scores'] = [float(score) for _, score in ranked]
        rows.append(row)
      self._last_decode_meta = rows
    return result

  # ---- IPA-masked loss helpers ----

  def _mask_logits_ipa(self, logits):
    """Set non-IPA token logits to -inf. Also keeps blank token for CTC.
    logits: (B, T, V) or (B, L, V). Returns same shape."""
    if not self.constrain_ipa:
      return logits
    V = logits.size(-1)
    tok_V = self.tokenizer.vocab_size  # e.g. 50258
    # Lazily build / rebuild mask to match actual model vocab size
    if self._ipa_vocab_mask is None or self._ipa_vocab_mask.size(0) != V:
      mask = torch.ones(V, dtype=torch.bool, device=logits.device)
      for tid in self._ipa_allowed_ids:
        if tid < V:
          mask[tid] = False
      # Extra tokens beyond tokenizer vocab (50258..51864) are special
      # (language, task, timestamp, etc.) — always allow them
      if V > tok_V:
        mask[tok_V:] = False
      self._ipa_vocab_mask = mask
    mask = self._ipa_vocab_mask.to(logits.device)
    logits = logits.clone()
    logits[:, :, mask] = float('-inf')
    return logits

  def _compute_s2s_loss(self, logits, labels):
    """Compute seq2seq cross-entropy loss, optionally with IPA masking.
    logits: (B, L, V)  labels: (B, L)
    Returns per-sample loss (B,) for weighted aggregation."""
    if self.constrain_ipa:
      logits = self._mask_logits_ipa(logits)
    # Shift: predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    B, L, V = shift_logits.shape
    label_smoothing = getattr(FLAGS, 'label_smoothing', 0.0)
    per_token = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction='none',
        label_smoothing=label_smoothing,
    ).view(B, L)
    # Mean over valid tokens per sample → (B,)
    mask = (shift_labels != -100).float()
    per_sample = (per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    return per_sample

  # ---- Shared CTC logic ----

  def _compute_ctc_entropy(self, log_probs_btv):
    """Compute per-sample mean entropy from CTC log-probabilities.

    Args:
      log_probs_btv: (B, T, V) log-softmax output (already normalised).
    Returns:
      (B,) per-sample mean entropy across time frames, masking padding.
    """
    if getattr(FLAGS, 'ctc_entropy_reg', 0.0) <= 0:
      return None
    p = log_probs_btv.exp()
    ent = -(p * log_probs_btv).sum(dim=-1)  # (B, T) per-frame entropy
    enc_len = getattr(self, '_last_enc_len', None)
    if enc_len is not None:
      T = ent.shape[1]
      mask = torch.arange(T, device=ent.device).unsqueeze(0) < enc_len.unsqueeze(1).to(ent.device)
      ent = (ent * mask).sum(dim=1) / enc_len.float().to(ent.device).clamp(min=1)
    else:
      ent = ent.mean(dim=1)  # (B,)
    return ent

  def _apply_phonetic_smoothing(self, log_probs, alpha):
    """Apply phonetic label smoothing to CTC log-probabilities.
    
    Blurs the CTC output distribution toward phonetically similar phones:
      p_smooth = (1 - α) * p + α * (p @ S)
    where S is a row-normalized phonetic similarity matrix (diagonal zeroed,
    so only off-diagonal mass spreads).
    
    Args:
      log_probs: (B, T, V) log-softmax output
      alpha: smoothing weight in [0, 1)
    Returns:
      (B, T, V) smoothed log-probabilities
    """
    S = get_phonetic_similarity_matrix().to(log_probs.device)  # (V, V)
    # Zero diagonal so smoothing only spreads to other tokens
    S_off = S.clone()
    S_off.fill_diagonal_(0)
    # Row-normalize off-diagonal similarities
    row_sums = S_off.sum(dim=1, keepdim=True).clamp(min=1e-8)
    S_norm = S_off / row_sums  # (V, V), each row sums to ~1
    
    probs = log_probs.exp()  # (B, T, V)
    # (B, T, V) @ (V, V) → (B, T, V): redistribute probability to similar phones
    smooth_probs = (1.0 - alpha) * probs + alpha * torch.matmul(probs, S_norm)
    return torch.log(smooth_probs.clamp(min=1e-10))

  def _compute_ctc_loss(self, enc_out, labels):
    # Use fused multi-layer representation if layer fusion is enabled
    fused = self._fuse_ctc_layers()
    ctc_input = fused if fused is not None else enc_out
    ctc_logits = self.ctc_head(ctc_input)  # (B, T, V)

    if self._ctc_char_level:
      # Character-level IPA CTC: decode BPE labels → text → char indices
      # CTC loss requires float32 (not supported for Half on CUDA)
      log_probs = F.log_softmax(ctc_logits.float(), dim=-1)
      B, T, V = log_probs.shape

      # Phonetic label smoothing: blur CTC output toward similar phones
      alpha = getattr(FLAGS, 'phonetic_label_smoothing', 0.0)
      if alpha > 0:
        log_probs = self._apply_phonetic_smoothing(log_probs, alpha)

      log_probs = log_probs.transpose(0, 1)  # (T, B, V)

      # Use actual encoder lengths when available (NeMo variable-length),
      # fall back to max T (Whisper fixed-length).
      enc_len = getattr(self, '_last_enc_len', None)
      if enc_len is not None:
        input_lengths = enc_len.to(log_probs.device).clamp(max=T)
      else:
        input_lengths = torch.full((B,), T, dtype=torch.long, device=log_probs.device)

      # Decode labels to text, then normalize IPA
      texts = self._decode_labels_to_texts(labels)
      char_seqs = []
      for text in texts:
        text = _normalize_ipa(text)
        seq = [IPA_CHAR_TO_ID[ch] for ch in text if ch in IPA_CHAR_TO_ID]
        char_seqs.append(seq)

      target_lengths = torch.tensor([len(s) for s in char_seqs],
                                     dtype=torch.long, device=log_probs.device)
      if target_lengths.sum() == 0:
        return torch.zeros(B, device=log_probs.device, requires_grad=True), ctc_logits, target_lengths
      targets_flat = torch.tensor([c for s in char_seqs for c in s],
                                   dtype=torch.long, device=log_probs.device)

      ctc_loss = self.ctc_loss_fn(log_probs, targets_flat, input_lengths, target_lengths)
      # Normalize per sample by target length to match old reduction='mean' behavior
      ctc_loss = ctc_loss / target_lengths.clamp(min=1).float()
      # Compute per-sample entropy for entropy regularization
      self._ctc_entropy = self._compute_ctc_entropy(log_probs.transpose(0, 1))  # back to (B,T,V)
      return ctc_loss, ctc_logits, target_lengths

    # Standard BPE CTC (no IPA constraint)
    # CTC loss requires float32 (not supported for Half on CUDA)
    log_probs = F.log_softmax(ctc_logits.float(), dim=-1)
    B, T, V = log_probs.shape
    log_probs = log_probs.transpose(0, 1)  # (T, B, V)

    # Use actual encoder lengths when available (NeMo variable-length),
    # fall back to max T (Whisper fixed-length).
    enc_len = getattr(self, '_last_enc_len', None)
    if enc_len is not None:
      input_lengths = enc_len.to(log_probs.device).clamp(max=T)
    else:
      input_lengths = torch.full((B,), T, dtype=torch.long, device=log_probs.device)

    if self.tokenizer is None:
      label_texts = getattr(self, '_current_label_texts', None)
      nemo_tok = getattr(self, '_nemo_tokenizer', None)
      assert nemo_tok is not None, (
          'NeMo non-IPA CTC requires _nemo_tokenizer for label encoding')
      if label_texts and any(label_texts):
        token_seqs = []
        for text in label_texts:
          ids = nemo_tok.text_to_ids(text) if text else []
          token_seqs.append(ids)
        target_lengths = torch.tensor([len(seq) for seq in token_seqs],
                                      dtype=torch.long, device=log_probs.device)
        if target_lengths.sum() == 0:
          return torch.zeros(B, device=log_probs.device, requires_grad=True), ctc_logits, target_lengths
        targets_flat = torch.tensor([token for seq in token_seqs for token in seq],
                                    dtype=torch.long, device=log_probs.device)
      else:
        target_lengths = torch.zeros(B, dtype=torch.long, device=log_probs.device)
        return torch.zeros(B, device=log_probs.device, requires_grad=True), ctc_logits, target_lengths
    else:
      pad_id = self.tokenizer.pad_token_id if self.tokenizer else 0
      target_mask = (labels != -100) & (labels != pad_id)
      target_lengths = target_mask.sum(dim=-1)
      targets_flat = torch.cat([labels[i][target_mask[i]] for i in range(B)])

    ctc_loss = self.ctc_loss_fn(log_probs, targets_flat, input_lengths, target_lengths)
    # Normalize per sample by target length to match old reduction='mean' behavior
    ctc_loss = ctc_loss / target_lengths.clamp(min=1).float()
    # Compute per-sample entropy for entropy regularization
    self._ctc_entropy = self._compute_ctc_entropy(log_probs.transpose(0, 1))  # back to (B,T,V)
    return ctc_loss, ctc_logits, target_lengths

  def _compute_mcer_loss(self, ctc_logits, labels):
    """Compute MCER (Minimum Character Error Rate) loss via REINFORCE.

    Steps:
      1. CTC beam search (no_grad) -> N-best hypotheses per sample
      2. Compute CER for each hypothesis vs reference (no_grad)
      3. Force-score each hypothesis using DIFFERENTIABLE CTC forward
         on the original ctc_logits (gradients flow here!)
      4. REINFORCE: loss = -sum_j log P(h_j) * (CER_j - baseline)

    Returns (B,) per-sample MCER loss.
    """
    import jiwer
    from metric.score import normalize_ipa as _mcer_normalize_ipa

    mcer_nbest = getattr(FLAGS, 'mcer_nbest', 8)
    mcer_beam = getattr(FLAGS, 'mcer_beam_size', 16)

    B, T, V = ctc_logits.shape
    # Differentiable log-probs for force-scoring (gradients flow through here)
    log_probs_diff = F.log_softmax(ctc_logits.float(), dim=-1)  # (B, T, V)

    # No-grad log-probs for beam search (numpy)
    with torch.no_grad():
      log_probs_np = log_probs_diff.detach().cpu().numpy()

      # Mask padded frames for variable-length encoders
      enc_len = getattr(self, '_last_enc_len', None)
      if enc_len is not None:
        lengths = enc_len.cpu().clamp(max=T).numpy()
      else:
        lengths = None

    # Also mask diff log_probs for correct CTC force-scoring
    if enc_len is not None:
      len_t = enc_len.to(log_probs_diff.device).clamp(max=T)
      range_t = torch.arange(T, device=log_probs_diff.device).unsqueeze(0)
      pad_mask = range_t >= len_t.unsqueeze(1)  # (B, T)
      blank_id = IPA_CTC_BLANK if self._ctc_char_level else self.ctc_blank_id
      # For masked positions: set blank=0, all others=-inf (differentiable via where)
      blank_row = torch.full((V,), float('-inf'), device=log_probs_diff.device)
      blank_row[blank_id] = 0.0
      log_probs_diff = torch.where(
          pad_mask.unsqueeze(-1).expand_as(log_probs_diff),
          blank_row.unsqueeze(0).unsqueeze(0).expand_as(log_probs_diff),
          log_probs_diff)
      # Also apply to numpy for beam search
      import numpy as np
      for b_idx in range(B):
        if lengths is not None and lengths[b_idx] < T:
          L = int(lengths[b_idx])
          log_probs_np[b_idx, L:, :] = -1e9
          log_probs_np[b_idx, L:, blank_id] = 0.0

    # Decode reference labels
    ref_texts = self._decode_labels_to_texts(labels)
    ref_texts_norm = [_mcer_normalize_ipa(t) for t in ref_texts]

    blank = IPA_CTC_BLANK if self._ctc_char_level else self.ctc_blank_id
    id_to_char = IPA_ID_TO_CHAR if self._ctc_char_level else None

    # Per-sample MCER loss (differentiable via CTC force-scoring)
    mcer_losses = []
    for i in range(B):
      ref_norm = ref_texts_norm[i]
      if not ref_norm.strip():
        mcer_losses.append(torch.tensor(0.0, device=ctc_logits.device))
        continue

      # N-best beam search (no_grad, CPU numpy)
      hyps = prefix_beam_search_nbest(
          log_probs_np[i], blank=blank, beam_width=mcer_beam,
          nbest=mcer_nbest, id_to_char=id_to_char)

      if not hyps:
        mcer_losses.append(torch.tensor(0.0, device=ctc_logits.device))
        continue

      # Convert hypotheses to text + token IDs
      hyp_token_ids = []
      hyp_texts_norm = []
      for score, hyp in hyps:
        if isinstance(hyp, str):
          text = hyp
          ids = [IPA_CHAR_TO_ID[ch] for ch in text if ch in IPA_CHAR_TO_ID]
        else:
          ids = list(hyp)
          text = ''.join(IPA_ID_TO_CHAR.get(c, '') for c in ids)
        hyp_token_ids.append(ids)
        hyp_texts_norm.append(_mcer_normalize_ipa(text))

      # Compute CER for each hypothesis (no grad, pure reward signal)
      with torch.no_grad():
        cers = []
        for hn in hyp_texts_norm:
          cers.append(jiwer.cer(ref_norm, hn) if ref_norm else (0.0 if not hn else 1.0))
        cers_t = torch.tensor(cers, dtype=torch.float32, device=ctc_logits.device)
        baseline = cers_t.mean()
        advantage = cers_t - baseline  # (N,)

      # Differentiable CTC force-scoring: -log P(hyp|x) via F.ctc_loss
      # Filter out empty hypotheses
      valid = [(j, ids) for j, ids in enumerate(hyp_token_ids) if ids]
      if not valid:
        mcer_losses.append(torch.tensor(0.0, device=ctc_logits.device))
        continue

      valid_idx, valid_ids = zip(*valid)
      n_valid = len(valid_idx)
      # Expand single-sample log_probs to n_valid copies: (T, N, V)
      lp_i = log_probs_diff[i].unsqueeze(1).expand(T, n_valid, V)  # (T, N, V)
      input_lengths = torch.full((n_valid,), T if lengths is None else int(lengths[i]) if lengths is not None else T,
                                  dtype=torch.long, device=ctc_logits.device)
      if enc_len is not None:
        input_lengths[:] = enc_len[i].clamp(max=T).long()

      all_targets = []
      target_lengths = []
      for ids in valid_ids:
        all_targets.extend(ids)
        target_lengths.append(len(ids))
      targets_t = torch.tensor(all_targets, dtype=torch.long, device=ctc_logits.device)
      target_lengths_t = torch.tensor(target_lengths, dtype=torch.long, device=ctc_logits.device)

      # CTC loss gives -log P(label|x), so nll_j = -log P(h_j|x)
      nll = F.ctc_loss(lp_i, targets_t, input_lengths, target_lengths_t,
                        blank=blank, reduction='none', zero_infinity=True)  # (N,)

      # REINFORCE: loss = sum_j nll_j * advantage_j (stop gradient on advantage)
      # nll_j = -log P(h_j), gradient of nll_j w.r.t. theta pushes model
      # to increase P for low-CER hypotheses and decrease P for high-CER ones
      adv_valid = advantage[list(valid_idx)]  # (n_valid,)
      # Normalize advantage to unit variance for stable gradients
      if n_valid > 1:
        adv_std = adv_valid.std().clamp(min=1e-6)
        adv_valid = adv_valid / adv_std

      mcer_loss_i = (nll * adv_valid.detach()).mean()
      mcer_losses.append(mcer_loss_i)

    # Stack: some are tensors with grad, some are scalar 0s
    return torch.stack(mcer_losses)  # (B,)

  def _compute_mwer_loss(self, ctc_logits, labels):
    """Compute MWER (Minimum Word Error Rate) loss via REINFORCE.

    Designed for non-IPA CTC models used on the word track. The loss is kept
    fully opt-in through mwer_* flags so existing training runs are unchanged.
    """
    mwer_nbest = getattr(FLAGS, 'mwer_nbest', 8)
    mwer_beam = getattr(FLAGS, 'mwer_beam_size', 16)

    B, T, V = ctc_logits.shape
    log_probs_diff = F.log_softmax(ctc_logits.float(), dim=-1)

    with torch.no_grad():
      log_probs_np = log_probs_diff.detach().cpu().numpy()
      enc_len = getattr(self, '_last_enc_len', None)
      if enc_len is not None:
        lengths = enc_len.cpu().clamp(max=T).numpy()
      else:
        lengths = None

    if enc_len is not None:
      len_t = enc_len.to(log_probs_diff.device).clamp(max=T)
      range_t = torch.arange(T, device=log_probs_diff.device).unsqueeze(0)
      pad_mask = range_t >= len_t.unsqueeze(1)
      blank_id = self.ctc_blank_id
      blank_row = torch.full((V,), float('-inf'), device=log_probs_diff.device)
      blank_row[blank_id] = 0.0
      log_probs_diff = torch.where(
          pad_mask.unsqueeze(-1).expand_as(log_probs_diff),
          blank_row.unsqueeze(0).unsqueeze(0).expand_as(log_probs_diff),
          log_probs_diff)
      for b_idx in range(B):
        if lengths is not None and lengths[b_idx] < T:
          L = int(lengths[b_idx])
          log_probs_np[b_idx, L:, :] = -1e9
          log_probs_np[b_idx, L:, blank_id] = 0.0

    ref_texts = self._decode_labels_to_texts(labels)
    blank = self.ctc_blank_id

    mwer_losses = []
    for i in range(B):
      ref_text = ref_texts[i]
      if not _normalize_wer_text(ref_text):
        mwer_losses.append(torch.tensor(0.0, device=ctc_logits.device))
        continue

      hyps = prefix_beam_search_nbest(
          log_probs_np[i], blank=blank, beam_width=mwer_beam,
          nbest=mwer_nbest, id_to_char=None)
      if not hyps:
        mwer_losses.append(torch.tensor(0.0, device=ctc_logits.device))
        continue

      hyp_token_ids = []
      for _, hyp in hyps:
        hyp_token_ids.append(list(hyp) if not isinstance(hyp, str) else [])
      hyp_texts = self._decode_ctc_token_seqs_to_texts(hyp_token_ids)

      with torch.no_grad():
        wers = [_single_utterance_wer(ref_text, hyp_text) for hyp_text in hyp_texts]
        wers_t = torch.tensor(wers, dtype=torch.float32, device=ctc_logits.device)
        baseline = wers_t.mean()
        advantage = wers_t - baseline

      valid = [(j, ids) for j, ids in enumerate(hyp_token_ids) if ids]
      if not valid:
        mwer_losses.append(torch.tensor(0.0, device=ctc_logits.device))
        continue

      valid_idx, valid_ids = zip(*valid)
      n_valid = len(valid_idx)
      lp_i = log_probs_diff[i].unsqueeze(1).expand(T, n_valid, V)
      input_lengths = torch.full(
          (n_valid,),
          T if lengths is None else int(lengths[i]),
          dtype=torch.long,
          device=ctc_logits.device)
      if enc_len is not None:
        input_lengths[:] = enc_len[i].clamp(max=T).long()

      all_targets = []
      target_lengths = []
      for ids in valid_ids:
        all_targets.extend(ids)
        target_lengths.append(len(ids))
      targets_t = torch.tensor(all_targets, dtype=torch.long, device=ctc_logits.device)
      target_lengths_t = torch.tensor(target_lengths, dtype=torch.long, device=ctc_logits.device)

      nll = F.ctc_loss(lp_i, targets_t, input_lengths, target_lengths_t,
                       blank=blank, reduction='none', zero_infinity=True)

      adv_valid = advantage[list(valid_idx)]
      if n_valid > 1:
        adv_std = adv_valid.std().clamp(min=1e-6)
        adv_valid = adv_valid / adv_std

      mwer_loss_i = (nll * adv_valid.detach()).mean()
      mwer_losses.append(mwer_loss_i)

    return torch.stack(mwer_losses)

  def _compute_word_ctc_loss(self, enc_out, word_labels):
    """Compute CTC loss for word (English text) or pseudo-IPA auxiliary task.
    
    Supports three modes:
      - pseudo_ipa_ctc: 53-class IPA vocab (same as primary CTC head)
      - word_ctc_bpe: backbone tokenizer vocab (NeMo SP 1024 or Whisper BPE)
      - char-level (default): 29-class (a-z + space + apostrophe + blank)
    
    Returns: (B,) per-sample CTC loss.
    """
    word_ctc_logits = self.word_ctc_head(enc_out)
    log_probs = F.log_softmax(word_ctc_logits.float(), dim=-1)
    B, T, V = log_probs.shape
    log_probs = log_probs.transpose(0, 1)  # (T, B, V)

    # Encoder lengths
    enc_len = getattr(self, '_last_enc_len', None)
    if enc_len is not None:
      input_lengths = enc_len.to(log_probs.device).clamp(max=T)
    else:
      input_lengths = torch.full((B,), T, dtype=torch.long, device=log_probs.device)

    # Get text labels (word text or pseudo-IPA text depending on mode)
    word_label_texts = getattr(self, '_current_word_label_texts', None)
    if word_label_texts and any(word_label_texts):
      texts = list(word_label_texts)
    elif self.tokenizer is not None:
      clean = word_labels.clone()
      clean[clean == -100] = self.tokenizer.pad_token_id or 0
      texts = self._tokenizer_batch_decode(clean.long().cpu().numpy(),
                                           skip_special_tokens=True)
    else:
      return torch.zeros(B, device=enc_out.device, requires_grad=True)

    if getattr(self, '_pseudo_ipa_ctc', False):
      # Pseudo-IPA mode: tokenize with IPA char vocab (same as primary CTC)
      # word_label_texts already contain pseudo-IPA text from preprocess
      char_seqs = []
      for text in texts:
        text = _normalize_ipa(text)  # same normalization as primary CTC
        seq = [IPA_CHAR_TO_ID[ch] for ch in text if ch in IPA_CHAR_TO_ID]
        char_seqs.append(seq)

      target_lengths = torch.tensor([len(s) for s in char_seqs],
                                     dtype=torch.long, device=log_probs.device)
      if target_lengths.sum() == 0:
        return torch.zeros(B, device=log_probs.device, requires_grad=True)
      targets_flat = torch.tensor([c for s in char_seqs for c in s],
                                   dtype=torch.long, device=log_probs.device)
    elif getattr(self, '_word_ctc_bpe', False):
      # BPE-level targets: tokenize text with backbone tokenizer
      nemo_tok = getattr(self, '_nemo_tokenizer', None)
      add_blank = bool(getattr(self, '_word_ctc_bpe_add_blank', False))
      blank_id = int(getattr(self, '_word_ctc_blank', 0))
      token_seqs = []
      for text in texts:
        text = _normalize_word_text(text)
        if nemo_tok is not None:
          ids = nemo_tok.text_to_ids(text)
        elif self.tokenizer is not None:
          ids = tokenize_text(self.tokenizer, text)
        else:
          ids = []
        ids = [int(token_id) for token_id in ids]
        if add_blank:
          seq = [token_id + 1 for token_id in ids]
        else:
          if any(token_id == blank_id for token_id in ids):
            raise ValueError(
                'word_ctc_bpe legacy blank=0 protocol received tokenizer id 0 in targets; '
                'use --word_ctc_bpe_add_blank or switch tokenizer/protocol')
          seq = ids
        token_seqs.append(seq)

      target_lengths = torch.tensor([len(s) for s in token_seqs],
                                     dtype=torch.long, device=log_probs.device)
      if target_lengths.sum() == 0:
        return torch.zeros(B, device=log_probs.device, requires_grad=True)
      targets_flat = torch.tensor([t for s in token_seqs for t in s],
                                   dtype=torch.long, device=log_probs.device)
      if torch.any(targets_flat >= V):
        raise ValueError(
            f'word_ctc_bpe target id overflow: max_target={int(targets_flat.max().item())} '
            f'>= vocab={V}, add_blank={add_blank}, blank_id={blank_id}')
    else:
      # Char-level targets
      char_seqs = []
      for text in texts:
        text = _normalize_word_text(text)
        seq = [WORD_CHAR_TO_ID[ch] for ch in text if ch in WORD_CHAR_TO_ID]
        char_seqs.append(seq)

      target_lengths = torch.tensor([len(s) for s in char_seqs],
                                     dtype=torch.long, device=log_probs.device)
      if target_lengths.sum() == 0:
        return torch.zeros(B, device=log_probs.device, requires_grad=True)
      targets_flat = torch.tensor([c for s in char_seqs for c in s],
                                   dtype=torch.long, device=log_probs.device)

    word_ctc_loss = self.word_ctc_loss_fn(log_probs, targets_flat, input_lengths, target_lengths)
    # Normalize per sample by target length
    word_ctc_loss = word_ctc_loss / target_lengths.clamp(min=1).float()
    return word_ctc_loss

  def _get_ctc_lm(self):
    """Lazily load character n-gram LM for CTC beam search."""
    if not self._ctc_lm_loaded:
      self._ctc_lm_loaded = True
      lm_path = getattr(FLAGS, 'ctc_lm_path', '')
      if lm_path and os.path.exists(lm_path):
        self._ctc_lm = CharNgramLM.load(lm_path)
        logger.info(f'CTC LM loaded from {lm_path} '
                     f'(order={self._ctc_lm.order}, vocab={len(self._ctc_lm.vocab)})')
      else:
        self._ctc_lm = None
    return self._ctc_lm

  def _ctc_decode(self, ctc_logits):
    """CTC decode with greedy or beam search + optional LM.
    Char-level mode returns list of strings.
    BPE mode returns list of token-ID lists."""
    beam_width = getattr(FLAGS, 'ctc_beam_width', 1)
    lm_weight = getattr(FLAGS, 'ctc_lm_weight', 0.0)

    if getattr(FLAGS, 'ctc_decode_fp32', False):
      log_probs = F.log_softmax(ctc_logits.float(), dim=-1)  # (B, T, V)
    else:
      log_probs = F.log_softmax(ctc_logits, dim=-1)  # (B, T, V)

    # For variable-length encoders (NeMo), mask out padded frames
    # to avoid decoding garbage beyond actual encoder output lengths.
    enc_len = getattr(self, '_last_enc_len', None)
    if enc_len is not None:
      B, T, V = log_probs.shape
      # Create mask: (B, T) True for valid positions
      lengths = enc_len.to(log_probs.device).clamp(max=T)
      range_t = torch.arange(T, device=log_probs.device).unsqueeze(0)  # (1, T)
      mask = range_t >= lengths.unsqueeze(1)  # (B, T) True = padded
      # Set padded positions to blank-dominant logits
      log_probs = log_probs.clone()
      blank_id = IPA_CTC_BLANK if self._ctc_char_level else self.ctc_blank_id
      # Build a blank-dominant row: -inf everywhere except blank=0
      blank_row = torch.full((V,), float('-inf'), device=log_probs.device)
      blank_row[blank_id] = 0.0
      log_probs[mask] = blank_row

    # Cache log_probs for ensemble extraction (detached, same device)
    self._last_ctc_log_probs = log_probs.detach()

    if self._ctc_char_level:
      lm = self._get_ctc_lm() if lm_weight > 0 else None
      return ctc_decode(
          log_probs, blank=IPA_CTC_BLANK, beam_width=beam_width,
          id_to_char=IPA_ID_TO_CHAR, lm=lm, lm_weight=lm_weight)

    # Standard BPE CTC
    return ctc_decode(
        log_probs, blank=self.ctc_blank_id, beam_width=beam_width)

  def _pad_generated(self, generated, device):
    """Pad/truncate generated tokens to FLAGS.max_new_tokens."""
    pad_id = self._pad_token_id
    max_len = FLAGS.max_new_tokens
    b, cur_len = generated.shape
    if cur_len < max_len:
      pad = torch.full((b, max_len - cur_len), pad_id,
                       dtype=generated.dtype, device=device)
      generated = torch.cat([generated, pad], dim=1)
    else:
      generated = generated[:, :max_len]
    return generated

  # ---- Decode method helpers (called from forward's generate block) ----

  def _generate_ctc(self, enc_out, ctc_logits, device):
    """CTC greedy/beam decode → padded (B, max_new_tokens) tensor."""
    if ctc_logits is None:
      # Use fused multi-layer representation if available
      fused = self._fuse_ctc_layers()
      ctc_input = fused if fused is not None else enc_out
      ctc_logits = self.ctc_head(ctc_input)
    
    # ---- Pseudo-IPA head ensemble: average logits from both CTC heads ----
    if (getattr(FLAGS, 'pseudo_ipa_ensemble', False) 
        and getattr(self, '_pseudo_ipa_ctc', False)
        and hasattr(self, 'word_ctc_head')):
      pseudo_logits = self.word_ctc_head(enc_out)
      w = getattr(FLAGS, 'pseudo_ipa_ensemble_weight', 0.5)
      ctc_logits = (1.0 - w) * ctc_logits + w * pseudo_logits
    
    decoded_seqs = self._ctc_decode(ctc_logits)
    max_len = FLAGS.max_new_tokens
    b = enc_out.shape[0]

    if self._ctc_char_level:
      # Store decoded text strings so callers can use pred_texts directly
      # (avoids lossy IPA char ID → tokenizer roundtrip in submit2.py)
      self._last_pred_texts = list(decoded_seqs)
      generated = torch.zeros((b, max_len), dtype=torch.long, device=device)
      for i, text in enumerate(decoded_seqs):
        char_ids = [IPA_CHAR_TO_ID.get(ch, 0) for ch in text]
        seq_len = min(len(char_ids), max_len)
        if seq_len > 0:
          generated[i, :seq_len] = torch.tensor(
              char_ids[:seq_len], dtype=torch.long)
    else:
      pad_id = self.ctc_blank_id if getattr(self, '_nemo_tokenizer', None) is not None else self._pad_token_id
      # Decode CTC token IDs using the SAME tokenizer that built the CTC head.
      # _init_ctc_head uses self.tokenizer first, _nemo_tokenizer as fallback.
      # Mirror that priority here: HF tokenizer (e.g. HuBERT) > NeMo tokenizer.
      nemo_tok = getattr(self, '_nemo_tokenizer', None)
      if self.tokenizer is not None:
        self._last_pred_texts = self._tokenizer_batch_decode(
            decoded_seqs, skip_special_tokens=True)
      elif nemo_tok is not None:
        self._last_pred_texts = [
            nemo_tok.ids_to_text(seq) if seq else '' for seq in decoded_seqs
        ]
      else:
        self._last_pred_texts = [''] * len(decoded_seqs)
      generated = torch.full((b, max_len), pad_id,
                             dtype=torch.long, device=device)
      for i, seq in enumerate(decoded_seqs):
        seq_len = min(len(seq), max_len)
        generated[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=torch.long)
    return generated

  def _generate_s2s(self, input_features, enc_out, attention_mask):
    """S2S autoregressive decode → padded (B, max_new_tokens) tensor."""
    generated = self._dispatch_s2s_generate(input_features, enc_out,
                                             attention_mask=attention_mask)
    dt = getattr(self, '_s2s_decoder_type', 'native')
    if dt == 'native':
      return self._pad_generated(generated, input_features.device)
    return generated  # custom decoders already return padded

  def _take_pred_texts(self):
    texts = getattr(self, '_last_pred_texts', None)
    self._last_pred_texts = None
    if texts is None:
      return None
    return list(texts)

  def _peek_decode_meta(self):
    rows = getattr(self, '_last_decode_meta', None)
    if rows is None:
      return None
    return list(rows)

  def _take_decode_meta(self):
    rows = getattr(self, '_last_decode_meta', None)
    self._last_decode_meta = None
    if rows is None:
      return None
    return list(rows)

  def _append_eval_primary_decode_meta(self):
    rows = self._take_decode_meta()
    if not rows:
      return
    if gezi.get('eval_primary_decode_meta') is None:
      gezi.set('eval_primary_decode_meta', [])
    gezi.get('eval_primary_decode_meta').extend(rows)

  def _take_aux_pred_texts(self):
    texts = getattr(self, '_last_aux_pred_texts', None)
    self._last_aux_pred_texts = None
    if texts is None:
      return None
    return list(texts)

  def _append_eval_ctc_logprobs(self):
    log_probs = getattr(self, '_last_ctc_log_probs', None)
    if log_probs is None:
      return
    if gezi.get('eval_ctc_logprobs') is None:
      gezi.set('eval_ctc_logprobs', [])
    enc_len = getattr(self, '_last_enc_len', None)
    B = log_probs.shape[0]
    for i in range(B):
      T_i = int(enc_len[i].item()) if enc_len is not None else log_probs.shape[1]
      T_i = min(T_i, log_probs.shape[1])
      gezi.get('eval_ctc_logprobs').append(log_probs[i, :T_i].cpu().numpy())

  def _append_eval_word_ctc_logprobs(self):
    log_probs = getattr(self, '_last_aux_ctc_log_probs', None)
    if log_probs is None:
      return
    if gezi.get('eval_ctc_logprobs_word') is None:
      gezi.set('eval_ctc_logprobs_word', [])
    enc_len = getattr(self, '_last_enc_len', None)
    B = log_probs.shape[0]
    for i in range(B):
      T_i = int(enc_len[i].item()) if enc_len is not None else log_probs.shape[1]
      T_i = min(T_i, log_probs.shape[1])
      gezi.get('eval_ctc_logprobs_word').append(log_probs[i, :T_i].cpu().numpy())

  def _run_infer_extra_heads(self):
    return getattr(FLAGS, 'infer_extra_heads', True)

  def _get_aux_head_type(self):
    if not hasattr(self, 'word_ctc_head'):
      return None
    if getattr(self, '_pseudo_ipa_ctc', False):
      return 'pseudo_ipa'
    if getattr(self, '_word_ctc_bpe', False):
      return 'word_ctc_bpe'
    return 'word_ctc'

  def _decode_aux_word_ctc(self, aux_logits):
    if aux_logits is None:
      return None

    beam_width = getattr(FLAGS, 'ctc_beam_width', 1)
    if getattr(FLAGS, 'ctc_decode_fp32', False):
      log_probs = F.log_softmax(aux_logits.float(), dim=-1)
    else:
      log_probs = F.log_softmax(aux_logits, dim=-1)

    enc_len = getattr(self, '_last_enc_len', None)
    if enc_len is not None:
      B, T, V = log_probs.shape
      lengths = enc_len.to(log_probs.device).clamp(max=T)
      range_t = torch.arange(T, device=log_probs.device).unsqueeze(0)
      mask = range_t >= lengths.unsqueeze(1)
      if getattr(self, '_pseudo_ipa_ctc', False):
        blank_id = IPA_CTC_BLANK
      elif getattr(self, '_word_ctc_bpe', False):
        blank_id = getattr(self, '_word_ctc_blank', 0)
      else:
        blank_id = 0
      blank_row = torch.full((V,), float('-inf'), device=log_probs.device)
      blank_row[blank_id] = 0.0
      log_probs = log_probs.clone()
      log_probs[mask] = blank_row

    self._last_aux_ctc_log_probs = log_probs.detach()

    if getattr(self, '_pseudo_ipa_ctc', False):
      texts = ctc_decode(
          log_probs,
          blank=IPA_CTC_BLANK,
          beam_width=beam_width,
          id_to_char=IPA_ID_TO_CHAR)
    elif getattr(self, '_word_ctc_bpe', False):
      blank_id = getattr(self, '_word_ctc_blank', 0)
      decoded_ids = ctc_decode(log_probs, blank=blank_id, beam_width=beam_width)
      if getattr(self, '_word_ctc_bpe_add_blank', False):
        token_seqs = [[tid - 1 for tid in seq if tid > 0] for seq in decoded_ids]
      else:
        token_seqs = [list(seq) for seq in decoded_ids]
      nemo_tok = getattr(self, '_nemo_tokenizer', None)
      if nemo_tok is not None:
        texts = [nemo_tok.ids_to_text(seq) if seq else '' for seq in token_seqs]
      elif self.tokenizer is not None:
        texts = self._tokenizer_batch_decode(token_seqs, skip_special_tokens=True)
      else:
        texts = [''] * len(token_seqs)
    else:
      texts = ctc_decode(
          log_probs,
          blank=WORD_CTC_BLANK,
          beam_width=beam_width,
          id_to_char=WORD_ID_TO_CHAR)

    self._last_aux_pred_texts = list(texts)
    return texts

  def _maybe_export_aux_word_head(self, enc_out):
    if not getattr(FLAGS, 'save_word_head_preds', True):
      return
    if not hasattr(self, 'word_ctc_head'):
      return

    aux_logits = self.word_ctc_head(enc_out)
    aux_texts = self._decode_aux_word_ctc(aux_logits)
    if aux_texts is None:
      return

    if gezi.get('eval_word_pred_texts') is None:
      gezi.set('eval_word_pred_texts', [])
    gezi.get('eval_word_pred_texts').extend(list(aux_texts))

    if gezi.get('eval_word_head_type') is None:
      gezi.set('eval_word_head_type', self._get_aux_head_type())

    if getattr(FLAGS, 'save_logprobs', False):
      # Skip word logprob accumulation for BPE mode — vocab is ~1025 vs IPA ~53,
      # resulting in ~19x larger tensors (~19 GB for 30k utterances).
      # Downstream ensemble feat_word does not use BPE logprobs anyway.
      if getattr(self, '_word_ctc_bpe', False):
        pass  # BPE word logprobs too large and unused by ensemble
      else:
        self._append_eval_word_ctc_logprobs()

  def _maybe_export_dual_head_preds(self, input_features, enc_out, attention_mask,
                                    ctc_logits=None, primary_method=None,
                                    primary_texts=None,
                                    ctc_logprobs_saved=False,
                                    primary_decode_meta=None):
    if not getattr(FLAGS, 'save_dual_head_preds', False):
      return
    if not getattr(self, '_ctc_char_level', False):
      return
    # Allow export when model has either CTC head or TDT decoder (or both).
    # TDT-only models (ctc_weight=0) have tdt_decoder but no CTC head.
    has_ctc = self.use_ctc
    has_tdt = hasattr(self, 'tdt_decoder')
    if not has_ctc and not has_tdt:
      return

    ctc_texts = list(primary_texts) if primary_method == 'ctc' and primary_texts is not None else None
    tdt_texts = list(primary_texts) if primary_method == 'tdt' and primary_texts is not None else None

    if ctc_texts is None and has_ctc:
      self._generate_ctc(enc_out, ctc_logits, input_features.device)
      ctc_texts = self._take_pred_texts()
      if not ctc_logprobs_saved:
        self._append_eval_ctc_logprobs()
        ctc_logprobs_saved = True

    if tdt_texts is None and has_tdt:
      self._tdt_reuse_generate(enc_out)
      tdt_texts = self._take_pred_texts()

    # At least one head must have produced output
    if ctc_texts is None and tdt_texts is None:
      return
    # Fill missing head with empty strings so downstream columns stay aligned
    B = enc_out.shape[0]
    if ctc_texts is None:
      ctc_texts = [''] * B
    if tdt_texts is None:
      tdt_texts = [''] * B

    if gezi.get('eval_dual_head_preds') is None:
      gezi.set('eval_dual_head_preds', [])
    rows = gezi.get('eval_dual_head_preds')
    primary_texts = list(primary_texts) if primary_texts is not None else None
    primary_decode_meta = list(primary_decode_meta) if primary_decode_meta is not None else None
    for i, ctc_text in enumerate(ctc_texts):
      tdt_text = tdt_texts[i]
      if primary_texts is not None and i < len(primary_texts):
        primary_text = primary_texts[i]
      elif primary_method == 'tdt':
        primary_text = tdt_text
      elif primary_method == 'ctc':
        primary_text = ctc_text
      else:
        primary_text = ''
      tdt_score = np.nan
      if (primary_method == 'tdt' and primary_decode_meta is not None
          and i < len(primary_decode_meta) and primary_decode_meta[i] is not None):
        row_meta = primary_decode_meta[i]
        if primary_text == tdt_text and 'pred_score' in row_meta:
          tdt_score = row_meta['pred_score']
      rows.append({
          'pred_ctc': ctc_text,
          'pred_tdt': tdt_text,
          'pred_tdt_score': tdt_score,
          'pred_primary': primary_text,
          'pred_primary_method': primary_method or '',
          'pred_heads_agree': int(ctc_text == tdt_text),
          'pred_ctc_len': len(ctc_text),
          'pred_tdt_len': len(tdt_text),
          'pred_dual_len_gap': abs(len(ctc_text) - len(tdt_text)),
      })

  def _generate_joint(self, input_features, enc_out, ctc_logits, attention_mask):
    """Joint CTC + S2S beam search decode (ESPnet-style).
    
    score = (1 - w) * log P_s2s(y|x) + w * log P_ctc(y|x)
    
    Currently supports AED decoder + CTC char-level only.
    Falls back to S2S decode if not supported.
    """
    dt = getattr(self, '_s2s_decoder_type', 'native')
    if not (self.use_ctc and self._ctc_char_level and dt == 'aed'):
      # Joint decode only implemented for CTC char + AED; fall back to S2S
      logger.warning('Joint CTC+S2S decode only supports CTC char-level + AED. '
                     'Falling back to S2S decode.')
      return self._generate_s2s(input_features, enc_out, attention_mask)

    # Get CTC log probs
    if ctc_logits is None:
      ctc_logits = self.ctc_head(enc_out)
    
    # ---- Pseudo-IPA head ensemble ----
    if (getattr(FLAGS, 'pseudo_ipa_ensemble', False) 
        and getattr(self, '_pseudo_ipa_ctc', False)
        and hasattr(self, 'word_ctc_head')):
      pseudo_logits = self.word_ctc_head(enc_out)
      w = getattr(FLAGS, 'pseudo_ipa_ensemble_weight', 0.5)
      ctc_logits = (1.0 - w) * ctc_logits + w * pseudo_logits
    
    ctc_log_probs = F.log_softmax(ctc_logits.float(), dim=-1)  # (B, T, V_ctc)

    # Mask padded frames
    enc_len = getattr(self, '_last_enc_len', None)
    if enc_len is not None:
      B, T, V = ctc_log_probs.shape
      lengths = enc_len.to(ctc_log_probs.device).clamp(max=T)
      range_t = torch.arange(T, device=ctc_log_probs.device).unsqueeze(0)
      mask = range_t >= lengths.unsqueeze(1)
      blank_row = torch.full((V,), float('-inf'), device=ctc_log_probs.device)
      blank_row[IPA_CTC_BLANK] = 0.0
      ctc_log_probs = ctc_log_probs.clone()
      ctc_log_probs[mask] = blank_row

    ctc_w = getattr(FLAGS, 'joint_ctc_decode_weight', 0.3)
    beam_size = max(getattr(FLAGS, 'num_beams', 1), 4)  # at least 4 for joint
    max_len = FLAGS.max_new_tokens
    B = enc_out.shape[0]
    device = enc_out.device

    aed = self.aed_decoder
    memory = aed.enc_proj(enc_out)  # (B, T, aed_dim)

    # Memory key padding mask
    memory_kpm = None
    if enc_len is not None:
      T_mem = memory.shape[1]
      rng = torch.arange(T_mem, device=device).unsqueeze(0)
      memory_kpm = rng >= enc_len.to(device).clamp(max=T_mem).unsqueeze(1)

    all_results = []
    for b in range(B):
      mem_b = memory[b:b+1]  # (1, T, D)
      mask_b = memory_kpm[b:b+1] if memory_kpm is not None else None
      ctc_lp_b = ctc_log_probs[b]  # (T, V_ctc)
      T_b = int(enc_len[b].item()) if enc_len is not None else ctc_lp_b.shape[0]

      # Each hypothesis: (tokens, s2s_score, ctc_prefix_prob_state)
      # For CTC prefix scoring we track (prob_blank, prob_nonblank) per hypothesis
      Hyp = namedtuple('Hyp', ['tokens', 's2s_score', 'ctc_score'])

      # Initialize with SOS
      init_hyps = [Hyp(tokens=[aed.sos_id], s2s_score=0.0, ctc_score=0.0)]

      for step in range(max_len):
        all_candidates = []
        for hyp in init_hyps:
          tok_tensor = torch.tensor([hyp.tokens], dtype=torch.long, device=device)
          L = tok_tensor.shape[1]
          positions = torch.arange(L, device=device).unsqueeze(0)
          tgt_emb = aed.embedding(tok_tensor) + aed.pos_enc(positions)
          causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=device)
          dec_out = aed.decoder(tgt=tgt_emb, memory=mem_b,
                                tgt_mask=causal_mask,
                                memory_key_padding_mask=mask_b)
          logits = aed.output_proj(dec_out[:, -1, :])  # (1, V_aed)
          s2s_lp = F.log_softmax(logits.float(), dim=-1).squeeze(0)  # (V_aed,)

          # Top-k from S2S
          topk_vals, topk_ids = s2s_lp.topk(beam_size)
          for j in range(beam_size):
            tok_id = topk_ids[j].item()
            new_s2s = hyp.s2s_score + topk_vals[j].item()

            # CTC prefix score for new token sequence
            # Simplified: use CTC log prob of the token at best alignment position
            new_tokens = hyp.tokens + [tok_id]
            ctc_score = self._ctc_prefix_score(ctc_lp_b, new_tokens[1:], T_b)

            combined = (1 - ctc_w) * new_s2s + ctc_w * ctc_score
            all_candidates.append(Hyp(tokens=new_tokens,
                                      s2s_score=new_s2s,
                                      ctc_score=ctc_score))

        # Prune to beam_size
        all_candidates.sort(key=lambda h: (1 - ctc_w) * h.s2s_score + ctc_w * h.ctc_score,
                            reverse=True)
        init_hyps = all_candidates[:beam_size]

        # Check if best hypothesis ended with EOS
        if init_hyps[0].tokens[-1] == aed.eos_id:
          break

      # Best hypothesis, remove SOS and EOS
      best = init_hyps[0].tokens[1:]  # remove SOS
      if best and best[-1] == aed.eos_id:
        best = best[:-1]
      all_results.append(best)

    # Pad to (B, max_new_tokens)
    generated = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(all_results):
      sl = min(len(seq), max_len)
      if sl > 0:
        generated[i, :sl] = torch.tensor(seq[:sl], dtype=torch.long)
    return generated

  def _ctc_prefix_score(self, ctc_log_probs, token_seq, T):
    """Simplified CTC prefix score for a token sequence.
    
    Uses a forward-algorithm-like computation:
    P_ctc(y_1..y_n | x) ≈ sum over valid alignments.
    
    For efficiency, uses a simplified version:
    score = sum of log P(y_i) at the most likely CTC positions.
    
    Args:
      ctc_log_probs: (T, V) CTC log probs for one sample
      token_seq: list of IPA char IDs (without SOS/EOS)
      T: actual encoder length
    Returns:
      float: approximate CTC prefix log probability
    """
    if not token_seq:
      return 0.0

    # Full CTC prefix scoring is O(T*U) — we use a greedy approximation:
    # Find best monotonic alignment of token_seq to CTC frames
    U = len(token_seq)
    t = 0
    score = 0.0
    blank_id = IPA_CTC_BLANK

    for u in range(U):
      tok = token_seq[u]
      if tok >= ctc_log_probs.shape[1]:
        # Token not in CTC vocab — skip (AED might use EOS which is not in CTC)
        continue
      best_t = t
      best_val = float('-inf')
      # Search forward from current position for best frame for this token
      search_end = min(T, t + (T - t) // max(U - u, 1) + 5)  # heuristic window
      for tt in range(t, min(search_end, T)):
        val = ctc_log_probs[tt, tok].item()
        if val > best_val:
          best_val = val
          best_t = tt
      score += best_val
      t = best_t + 1  # advance past this frame

    return score

  # ---- Shared forward ----

  def forward(self, inputs):
    """Pure inference: encode, decode (for loss), generate (for eval).
    
    Returns res dict with raw per-sample losses (NOT combined/reduced).
    get_loss_fn() handles weighting, reduction, and scalar logging.
    Does NOT return 'loss' key — melt eager's loss_fn_ will fall through
    to loss_fn (from get_loss_fn), which is the intended path.
    """
    input_features = inputs['input_features']
    attention_mask = inputs.get('attention_mask', None)
    labels = inputs.get('labels', None)
    # Test mode: collate_fn fills empty labels with [-100]; treat as no labels
    has_labels = labels is not None and labels.numel() > 0 and (labels != -100).any()

    # ---- NeMo label_texts: raw strings passed from dataloader ----
    # When NeMo backbone + tokenizer=None, labels tensor is all -100 placeholders
    # but label_texts contains the actual text strings for loss computation.
    label_texts = inputs.get('label_texts', None)
    if label_texts and any(label_texts):
      has_labels = True
    # Store on self so NeMo loss functions can access without signature changes
    self._current_label_texts = label_texts
    self._current_ipa_label_texts = inputs.get('ipa_label_texts', None)
    self._current_word_label_texts = inputs.get('word_label_texts', None)

    # During eval generate phase, accumulate label_texts for evaluate() to use
    # Only accumulate when do_generate=True (full eval), not during validation steps
    do_generate = not self.training and gezi.get('do_generate', False)
    if do_generate and self.tokenizer is None:
      if gezi.get('eval_label_texts') is None:
        gezi.set('eval_label_texts', [])
      if label_texts:
        gezi.get('eval_label_texts').extend(label_texts)
      else:
        # No label_texts in this batch (e.g. word-only samples with NeMo backbone)
        # — accumulate empty strings to maintain alignment with predictions
        batch_size = input_features.shape[0]
        gezi.get('eval_label_texts').extend([''] * batch_size)

    # ---- Multi-task labels ----
    ipa_labels = inputs.get('ipa_labels', None)
    word_labels = inputs.get('word_labels', None)
    ipa_mask = inputs.get('ipa_mask', None)    # (B,) 1.0 if has IPA label
    word_mask = inputs.get('word_mask', None)   # (B,) 1.0 if has word label
    ipa_weight = getattr(FLAGS, 'ipa_weight', 1.0)
    word_weight = getattr(FLAGS, 'word_weight', 0.0)
    use_multitask = (ipa_labels is not None or word_labels is not None) and (ipa_weight > 0 or word_weight > 0)

    # ---- Encoder ----
    enc_out = self._encode(input_features, attention_mask=attention_mask)
    # Cache enc_out for external extraction (e.g. aux head logprobs in save_logits)
    self._last_enc_out = enc_out.detach()

    # ---- fast_infer: skip all loss, only encode + generate ----
    if do_generate and getattr(FLAGS, 'fast_infer', False):
      B = input_features.shape[0]
      res = {}
      ctc_logprobs_saved = False
      run_infer_extra_heads = self._run_infer_extra_heads()
      with torch.no_grad():
        dm = getattr(FLAGS, 'decode_method', 'auto')
        use_ctc_decode = (dm in ('auto', 'ctc')) if self.ctc_only else (dm == 'ctc')
        use_s2s_decode = (dm == 's2s') or (dm == 'auto' and not self.ctc_only)
        use_tdt_decode = (dm == 'tdt')
        use_native_decode = (dm == 'native')
        if use_ctc_decode:
          res['pred'] = self._generate_ctc(enc_out, None, input_features.device)
          res['pred_texts'] = self._take_pred_texts()
          if (getattr(FLAGS, 'save_logprobs', False)
              or (run_infer_extra_heads and getattr(FLAGS, 'save_dual_head_preds', False))):
            self._append_eval_ctc_logprobs()
            ctc_logprobs_saved = True
        elif use_tdt_decode:
          assert hasattr(self, 'tdt_decoder'), \
            '--decode_method=tdt requires --s2s_decoder=tdt_reuse or --s2s_decoder=tdt_scratch'
          res['pred'] = self._tdt_reuse_generate(enc_out)
          res['pred_texts'] = self._take_pred_texts()
        elif use_native_decode:
          res['pred'] = self._s2s_generate(input_features, enc_out, attention_mask)
          res['pred_texts'] = self._take_pred_texts()
        elif use_s2s_decode:
          res['pred'] = self._generate_s2s(input_features, enc_out, attention_mask)
          res['pred_texts'] = self._take_pred_texts()
        else:
          res['pred'] = torch.zeros(B, FLAGS.max_new_tokens, dtype=torch.long,
                                    device=input_features.device)
        decode_meta = self._peek_decode_meta()
        if (getattr(FLAGS, 'save_logprobs', False)
            and not ctc_logprobs_saved
            and self.use_ctc
            and getattr(self, '_ctc_char_level', False)):
          self._generate_ctc(enc_out, None, input_features.device)
          self._take_pred_texts()
          self._append_eval_ctc_logprobs()
          ctc_logprobs_saved = True
        if run_infer_extra_heads:
          self._maybe_export_dual_head_preds(
              input_features, enc_out, attention_mask,
              primary_method='ctc' if use_ctc_decode else ('tdt' if use_tdt_decode else dm),
              primary_texts=res.get('pred_texts'),
              ctc_logprobs_saved=ctc_logprobs_saved,
              primary_decode_meta=decode_meta)
          self._maybe_export_aux_word_head(enc_out)
        self._append_eval_primary_decode_meta()
        if (run_infer_extra_heads
            and (getattr(self, '_aux_age', False) or getattr(self, '_aux_domain', False)
                 or getattr(self, '_aux_nchars', False) or getattr(self, '_aux_nspaces', False))):
          aux_res = self._compute_aux_losses(enc_out, inputs)
          res.update(aux_res)
          if getattr(self, '_aux_age', False) and 'aux_age_logits' in aux_res:
            if gezi.get('eval_aux_age_logits') is None:
              gezi.set('eval_aux_age_logits', [])
            gezi.get('eval_aux_age_logits').append(aux_res['aux_age_logits'].cpu())
          if getattr(self, '_aux_domain', False) and 'aux_domain_logits' in aux_res:
            if gezi.get('eval_aux_domain_logits') is None:
              gezi.set('eval_aux_domain_logits', [])
            gezi.get('eval_aux_domain_logits').append(aux_res['aux_domain_logits'].cpu())
          if getattr(self, '_aux_nchars', False) and 'aux_nchars_pred' in aux_res:
            if gezi.get('eval_aux_nchars_pred') is None:
              gezi.set('eval_aux_nchars_pred', [])
            gezi.get('eval_aux_nchars_pred').append(aux_res['aux_nchars_pred'].cpu())
          if getattr(self, '_aux_nspaces', False) and 'aux_nspaces_pred' in aux_res:
            if gezi.get('eval_aux_nspaces_pred') is None:
              gezi.set('eval_aux_nspaces_pred', [])
            gezi.get('eval_aux_nspaces_pred').append(aux_res['aux_nspaces_pred'].cpu())
      if do_generate and res.get('pred_texts'):
        if gezi.get('eval_pred_texts') is None:
          gezi.set('eval_pred_texts', [])
        gezi.get('eval_pred_texts').extend(res['pred_texts'])
      return res

    # valid 阶段只算 loss，eval 阶段才 generate/decode
    # (do_generate computed earlier for label_texts accumulation)

    B = input_features.shape[0]

    res = {}

    # ---- CTC path: raw per-sample loss ----
    ctc_logits = None
    if self.use_ctc and has_labels:
      ctc_loss, ctc_logits, ctc_target_lengths = self._compute_ctc_loss(enc_out, labels)
      res['ctc_loss'] = ctc_loss  # (B,)
      if getattr(FLAGS, 'corpus_level_loss', False):
        res['ctc_target_lengths'] = ctc_target_lengths.float()  # (B,)
      # CTC entropy regularization: retrieve per-sample entropy computed in _compute_ctc_loss
      if getattr(self, '_ctc_entropy', None) is not None:
        res['ctc_entropy'] = self._ctc_entropy
        self._ctc_entropy = None

    _mcer_w = getattr(FLAGS, 'mcer_weight', 0.0)
    _mwer_w = getattr(FLAGS, 'mwer_weight', 0.0)
    assert not (_mcer_w > 0 and _mwer_w > 0), (
        'Use either MCER or MWER in one run, not both. '
        'Set one of mcer_weight / mwer_weight to 0.')

    # ---- MCER: Minimum Character Error Rate loss ----
    if (self.use_ctc and has_labels and self.training
        and ctc_logits is not None
        and self._ctc_char_level
        and _mcer_w > 0
        and gezi.get('epoch', 0) >= getattr(FLAGS, 'mcer_start_epoch', 0)):
      mcer_loss = self._compute_mcer_loss(ctc_logits, labels)
      if mcer_loss is not None:
        res['mcer_loss'] = mcer_loss  # (B,)

    # ---- MWER: Minimum Word Error Rate loss ----
    if (self.use_ctc and has_labels and self.training
        and ctc_logits is not None
        and not self._ctc_char_level
        and _mwer_w > 0
        and gezi.get('epoch', 0) >= getattr(FLAGS, 'mwer_start_epoch', 0)):
      mwer_loss = self._compute_mwer_loss(ctc_logits, labels)
      if mwer_loss is not None:
        res['mwer_loss'] = mwer_loss  # (B,)

    # ---- InterCTC: intermediate-layer CTC losses ----
    if self.use_ctc and has_labels and getattr(self, '_inter_ctc_enabled', False):
      inter_hidden = getattr(self, '_inter_hidden_states', [])
      if inter_hidden:
        inter_ctc_loss = self._compute_inter_ctc_losses(inter_hidden, labels)
        if inter_ctc_loss is not None:
          res['inter_ctc_loss'] = inter_ctc_loss  # (B,)

    # ---- Seq2Seq path: raw per-sample losses ----
    # Skip expensive RNNT/TDT s2s forward during eval/validation.
    # The RNNT joint tensor [B,T,U,V] can exceed GPU memory on 24 GB cards
    # even at eval_batch_size=16 (tried to allocate 8-27 GiB).
    # - Full eval: actual metrics come from greedy decode generation.
    # - Step-level valid: training loss is sufficient for monitoring.
    # CTC losses are cheap and still computed for both paths.
    # AED decoder has minimal memory cost (no joint tensor), so don't skip it.
    _s2s_type = getattr(self, '_s2s_decoder_type', 'native')
    _skip_s2s = not self.training and _s2s_type not in ('aed',)
    if use_multitask:
      # word_aux_ipa is special: keep the main word S2S/TDT objective on
      # `labels`, and add IPA as an auxiliary branch via word_ctc/word_tdt.
      if (getattr(FLAGS, 'word_aux_ipa', False)
          and getattr(FLAGS, 'track', None) == 'word'
          and has_labels
          and not self.ctc_only
          and not _skip_s2s):
        s2s_loss = self._dispatch_s2s_forward(input_features, labels, enc_out,
                                              attention_mask=attention_mask)
        res['s2s_loss'] = s2s_loss

      if ipa_weight > 0 and ipa_labels is not None:
        has_active_ipa = ipa_mask is None or bool((ipa_mask > 0).any().item())
        if self.ctc_only:
          # IPA is already handled by character-level CTC above (53 chars),
          # skip redundant RNNT s2s to save ~50% joint tensor memory.
          pass
        elif not has_active_ipa:
          pass
        elif _skip_s2s:
          pass  # skip expensive RNNT forward during eval generate
        else:
          ipa_s2s = self._dispatch_s2s_forward(input_features, ipa_labels, enc_out,
                                              attention_mask=attention_mask,
                                              raw_texts=getattr(self, '_current_ipa_label_texts', None))
          res['ipa_s2s_loss'] = ipa_s2s        # (B,)
      if word_weight > 0 and word_labels is not None:
        # Optionally detach encoder to prevent word gradients from
        # conflicting with IPA CTC gradients on the shared encoder.
        enc_for_word = enc_out.detach() if getattr(FLAGS, 'word_detach_encoder', False) else enc_out
        word_raw_texts = getattr(self, '_current_word_label_texts', None)
        if getattr(FLAGS, 'word_tdt_pseudo_ipa', False):
          word_s2s = self._word_tdt_pseudo_ipa_forward(enc_for_word, word_raw_texts)
          if getattr(FLAGS, 'word_loss_normalize', True) and word_s2s.dim() > 0:
            word_target_lens = self._get_word_tdt_target_lengths(word_raw_texts).to(word_s2s.device).clamp(min=1)
            word_s2s = word_s2s / word_target_lens
          res['word_s2s_loss'] = word_s2s
        elif getattr(FLAGS, 'word_tdt_mixed', False):
          word_s2s = self._word_tdt_mixed_forward(enc_for_word, word_raw_texts)
          if getattr(FLAGS, 'word_loss_normalize', True) and word_s2s.dim() > 0:
            word_target_lens = self._get_word_tdt_mixed_target_lengths(word_raw_texts).to(word_s2s.device).clamp(min=1)
            word_s2s = word_s2s / word_target_lens
          res['word_s2s_loss'] = word_s2s
        elif getattr(FLAGS, 'word_ctc', False) and hasattr(self, 'word_ctc_head'):
          # Word/pseudo-IPA auxiliary via CTC
          # - word_ctc + pseudo_ipa_ctc: IPA vocab (53 chars), targets are pseudo-IPA
          # - word_ctc only: word char vocab (29: a-z + space + apostrophe) or BPE
          word_ctc_loss = self._compute_word_ctc_loss(enc_for_word, word_labels)
          res['word_s2s_loss'] = word_ctc_loss  # reuse same key for loss_fn compatibility
        elif not _skip_s2s:
          # Word auxiliary via S2S (RNNT) — BPE vocab
          word_s2s = self._s2s_forward(input_features, word_labels, enc_for_word,
                                        attention_mask=attention_mask)
          # Normalize per-sample RNNT loss by target length to match CTC magnitude
          if getattr(FLAGS, 'word_loss_normalize', True) and word_s2s.dim() > 0:
            word_target_lens = (word_labels != -100).sum(dim=-1).float().clamp(min=1)
            word_s2s = word_s2s / word_target_lens
          res['word_s2s_loss'] = word_s2s       # (B,)
      # Pass masks through for loss_fn
      if ipa_mask is not None:
        res['ipa_mask'] = ipa_mask
      if word_mask is not None:
        # word_only_loss: zero out word_mask for samples that have IPA labels,
        # so word auxiliary loss only trains on samples without IPA ground truth.
        if getattr(FLAGS, 'word_only_loss', False) and ipa_mask is not None:
          word_mask = word_mask * (1.0 - ipa_mask)
        if getattr(FLAGS, 'word_tdt_pseudo_ipa_only_nonipa', False) and ipa_mask is not None:
          word_mask = word_mask * (1.0 - ipa_mask)
        res['word_mask'] = word_mask
    else:
      if not self.ctc_only and has_labels and not _skip_s2s:
        s2s_loss = self._dispatch_s2s_forward(input_features, labels, enc_out,
                                              attention_mask=attention_mask)
        res['s2s_loss'] = s2s_loss            # (B,)

    # ---- Corpus-level loss: pass token counts for weighted reduction ----
    if getattr(FLAGS, 'corpus_level_loss', False):
      if use_multitask:
        if has_labels and 's2s_loss' in res:
          res['s2s_token_counts'] = (labels != -100).sum(dim=-1).float()
        if ipa_labels is not None and 'ipa_s2s_loss' in res:
          res['ipa_token_counts'] = (ipa_labels != -100).sum(dim=-1).float()
        if word_labels is not None and 'word_s2s_loss' in res:
          if getattr(FLAGS, 'word_tdt_pseudo_ipa', False):
            res['word_token_counts'] = self._get_word_tdt_target_lengths(
                getattr(self, '_current_word_label_texts', None)).to(enc_out.device)
          elif getattr(FLAGS, 'word_tdt_mixed', False):
            res['word_token_counts'] = self._get_word_tdt_mixed_target_lengths(
                getattr(self, '_current_word_label_texts', None)).to(enc_out.device)
          else:
            res['word_token_counts'] = (word_labels != -100).sum(dim=-1).float()
      elif has_labels and 's2s_loss' in res:
        res['s2s_token_counts'] = (labels != -100).sum(dim=-1).float()

    # ---- mean_volume reduction: store NeMo SentencePiece target lengths ----
    if getattr(FLAGS, 'mean_volume_loss', False) and 's2s_loss' in res:
      target_lens = getattr(self, '_last_s2s_target_lengths', None)
      if target_lens is not None:
        res['s2s_target_lengths'] = target_lens.float()

    # ---- Auxiliary metadata losses (age / domain / nchars / nspaces) ----
    _any_aux = (getattr(self, '_aux_age', False) or getattr(self, '_aux_domain', False)
                or getattr(self, '_aux_nchars', False) or getattr(self, '_aux_nspaces', False))
    if _any_aux:
      aux_res = self._compute_aux_losses(enc_out, inputs)
      res.update(aux_res)
      # Accumulate logits/preds during eval for metrics
      if do_generate:
        if getattr(self, '_aux_age', False) and 'aux_age_logits' in aux_res:
          if gezi.get('eval_aux_age_logits') is None:
            gezi.set('eval_aux_age_logits', [])
          gezi.get('eval_aux_age_logits').append(aux_res['aux_age_logits'].cpu())
        if getattr(self, '_aux_domain', False) and 'aux_domain_logits' in aux_res:
          if gezi.get('eval_aux_domain_logits') is None:
            gezi.set('eval_aux_domain_logits', [])
          gezi.get('eval_aux_domain_logits').append(aux_res['aux_domain_logits'].cpu())
        if getattr(self, '_aux_nchars', False) and 'aux_nchars_pred' in aux_res:
          if gezi.get('eval_aux_nchars_pred') is None:
            gezi.set('eval_aux_nchars_pred', [])
          gezi.get('eval_aux_nchars_pred').append(aux_res['aux_nchars_pred'].cpu())
        if getattr(self, '_aux_nspaces', False) and 'aux_nspaces_pred' in aux_res:
          if gezi.get('eval_aux_nspaces_pred') is None:
            gezi.set('eval_aux_nspaces_pred', [])
          gezi.get('eval_aux_nspaces_pred').append(aux_res['aux_nspaces_pred'].cpu())

    # ---- Pass through sample weights for loss_fn ----
    weights = inputs.get('weight', None)
    if weights is not None:
      res['weight'] = weights

    # ---- Legacy loss: compute combined scalar loss inline (ctc6 behaviour) ----
    if getattr(FLAGS, 'legacy_loss', False) and has_labels:
      ctc_l = res.get('ctc_loss', None)
      s2s_l = res.get('s2s_loss', None)
      if self.ctc_only:
        loss = ctc_l if ctc_l is not None else torch.zeros(B, device=input_features.device)
      elif self.use_ctc and ctc_l is not None and s2s_l is not None:
        loss = (1 - self.ctc_weight) * s2s_l + self.ctc_weight * ctc_l
      elif s2s_l is not None:
        loss = s2s_l
      else:
        loss = torch.zeros(B, device=input_features.device)
      # Weighted reduction (matches old forward() exactly)
      w = res.get('weight', None)
      if w is not None and loss.dim() > 0:
        w = w.to(loss.device)
        loss = (loss * w).sum() / w.sum()
      elif loss.dim() > 0:
        loss = loss.mean()
      res['loss'] = loss
    # ---- Generate predictions ----
    if do_generate:
      with torch.no_grad():
        # Determine decode method: auto / ctc / s2s / tdt / native / joint
        dm = getattr(FLAGS, 'decode_method', 'auto')
        use_ctc_decode = False
        use_s2s_decode = False
        use_tdt_decode = False
        use_native_decode = False
        use_joint_decode = False
        if dm == 'auto':
          use_ctc_decode = self.ctc_only
          use_s2s_decode = not self.ctc_only
        elif dm == 'ctc':
          use_ctc_decode = True
        elif dm == 's2s':
          use_s2s_decode = True
        elif dm == 'tdt':
          use_tdt_decode = True
        elif dm == 'native':
          use_native_decode = True
        elif dm == 'joint':
          use_joint_decode = True
        else:
          use_ctc_decode = self.ctc_only
          use_s2s_decode = not self.ctc_only

        ctc_logprobs_saved = False
        run_infer_extra_heads = self._run_infer_extra_heads()

        if use_ctc_decode:
          res['pred'] = self._generate_ctc(enc_out, ctc_logits, input_features.device)
          # CTC decode stores decoded texts directly (avoids tokenizer roundtrip)
          res['pred_texts'] = self._take_pred_texts()
          # Accumulate CTC logprobs for offline ensemble (gated by flag)
          if (getattr(FLAGS, 'save_logprobs', False)
              or (run_infer_extra_heads and getattr(FLAGS, 'save_dual_head_preds', False))):
            self._append_eval_ctc_logprobs()
            ctc_logprobs_saved = True
        elif use_tdt_decode:
          assert hasattr(self, 'tdt_decoder'), \
            '--decode_method=tdt requires --s2s_decoder=tdt_reuse or --s2s_decoder=tdt_scratch'
          res['pred'] = self._tdt_reuse_generate(enc_out)
          res['pred_texts'] = self._take_pred_texts()
        elif use_native_decode:
          res['pred'] = self._s2s_generate(input_features, enc_out, attention_mask)
          res['pred_texts'] = self._take_pred_texts()
        elif use_s2s_decode:
          res['pred'] = self._generate_s2s(input_features, enc_out, attention_mask)
          # NeMo models store decoded texts directly (avoids tokenizer roundtrip)
          res['pred_texts'] = self._take_pred_texts()
        elif use_joint_decode:
          res['pred'] = self._generate_joint(input_features, enc_out,
                                              ctc_logits, attention_mask)
          res['pred_texts'] = self._take_pred_texts()
        decode_meta = self._peek_decode_meta()
        if (getattr(FLAGS, 'save_logprobs', False)
            and not ctc_logprobs_saved
            and self.use_ctc
            and getattr(self, '_ctc_char_level', False)):
          self._generate_ctc(enc_out, ctc_logits, input_features.device)
          self._take_pred_texts()
          self._append_eval_ctc_logprobs()
          ctc_logprobs_saved = True
        if run_infer_extra_heads:
          self._maybe_export_dual_head_preds(
              input_features, enc_out, attention_mask,
              ctc_logits=ctc_logits,
              primary_method='ctc' if use_ctc_decode else ('tdt' if use_tdt_decode else dm),
              primary_texts=res.get('pred_texts'),
              ctc_logprobs_saved=ctc_logprobs_saved,
              primary_decode_meta=decode_meta)
          self._maybe_export_aux_word_head(enc_out)
        self._append_eval_primary_decode_meta()
      if res.get('pred_texts'):
        if gezi.get('eval_pred_texts') is None:
          gezi.set('eval_pred_texts', [])
        gezi.get('eval_pred_texts').extend(res['pred_texts'])
    else:
      b = input_features.shape[0]
      res['pred'] = torch.zeros(b, FLAGS.max_new_tokens, dtype=torch.long,
                                device=input_features.device)

    # ---- Step-level valid: cheap CTC greedy decode for batch metrics ----
    # Only during validation steps (not training, not full generate).
    # CTC greedy decode is essentially free (argmax of existing logits).
    if not self.training and not do_generate and self.use_ctc and ctc_logits is not None:
      with torch.no_grad():
        # ---- Pseudo-IPA head ensemble for step-level decode ----
        _decode_logits = ctc_logits
        if (getattr(FLAGS, 'pseudo_ipa_ensemble', False) 
            and getattr(self, '_pseudo_ipa_ctc', False)
            and hasattr(self, 'word_ctc_head')):
          pseudo_logits = self.word_ctc_head(enc_out)
          w = getattr(FLAGS, 'pseudo_ipa_ensemble_weight', 0.5)
          _decode_logits = (1.0 - w) * ctc_logits + w * pseudo_logits
        decoded_seqs = self._ctc_decode(_decode_logits)
        if self._ctc_char_level:
          # decoded_seqs is list of strings
          res['_ctc_pred_texts'] = decoded_seqs
        else:
          # decoded_seqs is list of token-ID lists → decode to text
          tokenizer = self.tokenizer
          nemo_tok = getattr(self, '_nemo_tokenizer', None)
          if tokenizer is not None:
            res['_ctc_pred_texts'] = self._tokenizer_batch_decode(
                decoded_seqs, skip_special_tokens=True)
          elif nemo_tok is not None:
            res['_ctc_pred_texts'] = [
                nemo_tok.ids_to_text(seq) if seq else '' for seq in decoded_seqs
            ]
          else:
            res['_ctc_pred_texts'] = [''] * len(decoded_seqs)

    # ---- Step-level valid: cheap S2S teacher-forced argmax for batch metrics ----
    if not self.training and not do_generate:
      s2s_texts = getattr(self, '_s2s_pred_texts', None)
      if s2s_texts is not None:
        res['_s2s_pred_texts'] = s2s_texts
        self._s2s_pred_texts = None  # consume

    return res

  def get_valid_fn(self):
    """Batch-level approximate CER/WER for step-level validation feedback.
    
    Called by melt eager training loop at each valid step.
    Uses CTC greedy decode (stored by forward()) to compute batch metrics.
    Returns dict of metrics that get prefixed with 'metric/' and displayed in tqdm.
    """
    def valid_fn(y_, y, x=None):
      res = {}
      ctc_pred_texts = y_.get('_ctc_pred_texts')
      s2s_pred_texts = y_.get('_s2s_pred_texts')
      if ctc_pred_texts is None and s2s_pred_texts is None:
        return res

      # Decode label token IDs → text
      labels = y
      if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
      labels = np.asarray(labels)
      if labels.ndim == 1:
        labels = labels[np.newaxis, :]

      if getattr(self, '_ctc_char_level', False):
        # NeMo native CTC or IPA char-level CTC:
        # Use label_texts directly if available (NeMo backbone, tokenizer=None)
        current_label_texts = getattr(self, '_current_label_texts', None)
        if current_label_texts and any(current_label_texts):
          target_texts = list(current_label_texts)
        else:
          # Labels are IPA char IDs
          target_texts = []
          for row in labels:
            chars = []
            for cid in row:
              cid = int(cid)
              if cid <= 0:  # blank / pad / -100
                continue
              ch = IPA_ID_TO_CHAR.get(cid)
              if ch is not None:
                chars.append(ch)
            target_texts.append(''.join(chars))
      else:
        # Labels are tokenizer IDs
        tokenizer = get_tokenizer(FLAGS.backbone)
        if tokenizer is not None:
          pad_id = tokenizer.pad_token_id or 0
          labels = np.where(labels == -100, pad_id, labels).astype(np.int64)
          target_texts = self._tokenizer_batch_decode(
              labels.tolist(), skip_special_tokens=True)
        else:
          # NeMo backbone with tokenizer=None: use label_texts
          current_label_texts = getattr(self, '_current_label_texts', None)
          if current_label_texts and any(current_label_texts):
            target_texts = list(current_label_texts)
          else:
            target_texts = [''] * len(labels)

      # Compute batch CER or WER
      def _compute_cer_wer(pred_texts, target_texts, prefix=''):
        """Compute CER/WER for a set of predictions and targets."""
        metrics = {}
        try:
          import jiwer
          score_metric = getattr(FLAGS, 'score_metric', 'ipa_cer')
          if score_metric == 'ipa_cer':
            from metric.score import normalize_ipa
            norm_preds = [normalize_ipa(t) for t in pred_texts]
            norm_refs = [normalize_ipa(t) for t in target_texts]
            pairs = [(r, p) for r, p in zip(norm_refs, norm_preds) if r.strip()]
            if pairs:
              refs, hyps = zip(*pairs)
              metrics[f'{prefix}CER'] = jiwer.cer(list(refs), list(hyps))
          elif score_metric == 'wer':
            from metric.score import EnglishTextNormalizer, english_spelling_normalizer
            normalizer = EnglishTextNormalizer(english_spelling_normalizer)
            norm_preds = [normalizer(t) for t in pred_texts]
            norm_refs = [normalizer(t) for t in target_texts]
            pairs = [(r, p) for r, p in zip(norm_refs, norm_preds) if r.strip()]
            if pairs:
              refs, hyps = zip(*pairs)
              metrics[f'{prefix}WER'] = jiwer.wer(list(refs), list(hyps))
        except Exception as e:
          logger.warning(f'[get_valid_fn] metric error ({prefix}): {e}')
        return metrics

      # CTC batch metrics
      if ctc_pred_texts is not None:
        res.update(_compute_cer_wer(ctc_pred_texts, target_texts, prefix=''))

      # S2S batch metrics (teacher-forced argmax — cheap approximation)
      if s2s_pred_texts is not None:
        res.update(_compute_cer_wer(s2s_pred_texts, target_texts, prefix='s2s_'))

      return res

    return valid_fn

  def get_loss_fn(self):
    """Return calc_loss(res, y, x) for melt eager training.
    
    Computes fine-grained loss from forward() outputs, logs scalars
    that are automatically displayed in tqdm and written to history.csv.
    """
    import lele as le

    def calc_loss(res, y, x, step=None, epoch=None, training=None):
      scalars = {}
      device = res['pred'].device
      ipa_weight = getattr(FLAGS, 'ipa_weight', 1.0)
      word_weight = getattr(FLAGS, 'word_weight', 0.0)

      # ---- Collect raw per-sample losses ----
      ctc_loss = res.get('ctc_loss', None)       # (B,) or None
      s2s_loss = res.get('s2s_loss', None)        # (B,) or None (single-task)
      ipa_s2s = res.get('ipa_s2s_loss', None)     # (B,) or None (multi-task)
      word_s2s = res.get('word_s2s_loss', None)   # (B,) or None (multi-task)
      ipa_mask = res.get('ipa_mask', None)
      word_mask = res.get('word_mask', None)
      weights = res.get('weight', None)

      # ---- Determine B ----
      for v in [s2s_loss, ipa_s2s, word_s2s, ctc_loss]:
        if v is not None and v.dim() > 0:
          B = v.shape[0]
          break
      else:
        B = res['pred'].shape[0]

      # ---- Seq2Seq loss combination ----
      if s2s_loss is not None and (ipa_s2s is not None or word_s2s is not None):
        # Main-task word loss plus auxiliary cross-label losses.
        combined_s2s = s2s_loss
        scalars['loss/s2s'] = s2s_loss.mean().item()

        aux_terms = []
        if ipa_s2s is not None and ipa_weight > 0:
          m = ipa_mask.to(device) if ipa_mask is not None else 1.0
          masked_ipa = ipa_s2s * m
          combined_s2s = combined_s2s + ipa_weight * masked_ipa
          aux_terms.append(ipa_weight * masked_ipa)
          if isinstance(m, torch.Tensor) and m.sum() > 0:
            scalars['loss/phonetic'] = (masked_ipa.sum() / m.sum()).item()
          else:
            scalars['loss/phonetic'] = ipa_s2s.mean().item()

        if word_s2s is not None and word_weight > 0:
          m = word_mask.to(device) if word_mask is not None else 1.0
          masked_word = word_s2s * m
          combined_s2s = combined_s2s + word_weight * masked_word
          aux_terms.append(word_weight * masked_word)
          if isinstance(m, torch.Tensor) and m.sum() > 0:
            scalars['loss/word'] = (masked_word.sum() / m.sum()).item()
          else:
            scalars['loss/word'] = word_s2s.mean().item()

        if aux_terms:
          scalars['loss/aux'] = torch.stack(aux_terms, dim=0).sum(dim=0).mean().item()
      elif ipa_s2s is not None or word_s2s is not None:
        # Multi-task path
        mt_loss = torch.zeros(B, device=device)
        active_weights = torch.zeros(B, device=device)

        if ipa_s2s is not None and ipa_weight > 0:
          m = ipa_mask.to(device) if ipa_mask is not None else 1.0
          masked_ipa = ipa_s2s * m
          mt_loss = mt_loss + ipa_weight * masked_ipa
          active_weights = active_weights + ipa_weight * (m if isinstance(m, torch.Tensor) else torch.ones(B, device=device))
          if isinstance(m, torch.Tensor) and m.sum() > 0:
            scalars['loss/phonetic'] = (masked_ipa.sum() / m.sum()).item()
          elif ipa_s2s is not None:
            scalars['loss/phonetic'] = ipa_s2s.mean().item()

        if word_s2s is not None and word_weight > 0:
          m = word_mask.to(device) if word_mask is not None else 1.0
          masked_word = word_s2s * m
          mt_loss = mt_loss + word_weight * masked_word
          active_weights = active_weights + word_weight * (m if isinstance(m, torch.Tensor) else torch.ones(B, device=device))
          if isinstance(m, torch.Tensor) and m.sum() > 0:
            scalars['loss/word'] = (masked_word.sum() / m.sum()).item()
          elif word_s2s is not None:
            scalars['loss/word'] = word_s2s.mean().item()

        combined_s2s = mt_loss / active_weights.clamp(min=1e-8)
        scalars['loss/aux'] = combined_s2s.mean().item()
      elif s2s_loss is not None:
        combined_s2s = s2s_loss
        scalars['loss/s2s'] = s2s_loss.mean().item()
      else:
        combined_s2s = None

      # ---- CTC loss ----
      if ctc_loss is not None:
        # Focal CTC: down-weight easy samples, focus on hard ones
        if getattr(FLAGS, 'focal_ctc', False):
          gamma = getattr(FLAGS, 'focal_ctc_gamma', 2.0)
          # Normalize per-sample CTC loss to [0, 1] range for focal weighting
          with torch.no_grad():
            ctc_max = ctc_loss.detach().max().clamp(min=1e-6)
            focal_p = (ctc_loss.detach() / ctc_max)  # higher loss → higher p
            focal_weight = focal_p ** gamma
            focal_weight = focal_weight / focal_weight.mean().clamp(min=1e-6)  # normalize
          ctc_loss = ctc_loss * focal_weight
          scalars['loss/focal_weight_mean'] = focal_weight.mean().item()
        scalars['loss/ctc'] = ctc_loss.mean().item()

        # CTC entropy regularization: subtract entropy to encourage smoother distributions
        ctc_entropy = res.get('ctc_entropy', None)
        _ent_w = getattr(FLAGS, 'ctc_entropy_reg', 0.0)
        if ctc_entropy is not None and _ent_w > 0:
          # Negative entropy → minimising loss encourages higher entropy (smoother)
          ctc_loss = ctc_loss - _ent_w * ctc_entropy
          scalars['loss/ctc_entropy'] = ctc_entropy.mean().item()
          scalars['loss/ctc_with_ent'] = ctc_loss.mean().item()

      # ---- InterCTC loss ----
      inter_ctc_loss = res.get('inter_ctc_loss', None)
      if inter_ctc_loss is not None:
        scalars['loss/inter_ctc'] = inter_ctc_loss.mean().item()

      # ---- MCER loss ----
      mcer_loss = res.get('mcer_loss', None)
      _mcer_w = getattr(FLAGS, 'mcer_weight', 0.0)
      _mwer_w = getattr(FLAGS, 'mwer_weight', 0.0)
      if mcer_loss is not None and _mcer_w > 0:
        scalars['loss/mcer'] = mcer_loss.mean().item()

      mwer_loss = res.get('mwer_loss', None)
      if mwer_loss is not None and _mwer_w > 0:
        scalars['loss/mwer'] = mwer_loss.mean().item()

      # ---- Combined per-sample loss ----
      # Base CTC loss (main + InterCTC if enabled)
      effective_ctc = ctc_loss
      if effective_ctc is not None and inter_ctc_loss is not None:
        inter_w = getattr(FLAGS, 'inter_ctc_weight', 0.3)
        effective_ctc = effective_ctc + inter_w * inter_ctc_loss

      _hybrid_mv_reduced = False  # True when hybrid mean_volume already produced scalar
      if self.ctc_only:
        loss = effective_ctc if effective_ctc is not None else torch.zeros(B, device=device)
        # Multi-task auxiliary: add word loss (CTC or S2S)
        # IPA is handled by CTC; combined_s2s here contains only word loss
        if combined_s2s is not None and word_weight > 0:
          loss = loss + word_weight * combined_s2s
      elif self.use_ctc and effective_ctc is not None and combined_s2s is not None:
        if getattr(FLAGS, 'mean_volume_loss', False):
          # Hybrid mode: CTC per-sample is per-token normalized, S2S is raw total.
          # Reduce each component to mean_volume scalar separately, then combine.
          ctc_tgt_lens = res.get('ctc_target_lengths', torch.ones(B, device=device)).to(device).float()
          s2s_tgt_lens = res.get('s2s_target_lengths', torch.ones(B, device=device)).to(device).float()
          # CTC: undo per-token normalization, then mean_volume
          ctc_mv = (effective_ctc * ctc_tgt_lens).sum() / ctc_tgt_lens.sum().clamp(min=1e-8)
          # S2S: already raw totals, direct mean_volume
          s2s_mv = combined_s2s.sum() / s2s_tgt_lens.sum().clamp(min=1e-8)
          loss = (1 - self.ctc_weight) * s2s_mv + self.ctc_weight * ctc_mv
          _hybrid_mv_reduced = True
          scalars['loss/ctc_mv'] = ctc_mv.item()
          scalars['loss/s2s_mv'] = s2s_mv.item()
        else:
          loss = (1 - self.ctc_weight) * combined_s2s + self.ctc_weight * effective_ctc
      elif combined_s2s is not None:
        loss = combined_s2s
      else:
        loss = torch.zeros(B, device=device)

      # ---- Auxiliary metadata losses (age / domain) ----
      aux_age_w = getattr(FLAGS, 'aux_age_weight', 0)
      aux_age_loss = res.get('aux_age_loss', None)
      if aux_age_loss is not None and aux_age_w > 0:
        aux_age_loss = aux_age_loss.to(device)
        age_mask = res.get('aux_age_mask', torch.ones(B, device=device)).to(device)
        n_valid = age_mask.sum().clamp(min=1)
        scalars['loss/aux_age'] = (aux_age_loss.sum() / n_valid).item()
        loss = loss + aux_age_w * aux_age_loss

      aux_domain_w = getattr(FLAGS, 'aux_domain_weight', 0)
      aux_domain_loss = res.get('aux_domain_loss', None)
      if aux_domain_loss is not None and aux_domain_w > 0:
        aux_domain_loss = aux_domain_loss.to(device)
        domain_mask = res.get('aux_domain_mask', torch.ones(B, device=device)).to(device)
        n_valid = domain_mask.sum().clamp(min=1)
        scalars['loss/aux_domain'] = (aux_domain_loss.sum() / n_valid).item()
        loss = loss + aux_domain_w * aux_domain_loss

      # ---- Length prediction aux losses (nchars / nspaces) ----
      aux_nchars_w = getattr(FLAGS, 'aux_nchars_weight', 0)
      aux_nchars_loss = res.get('aux_nchars_loss', None)
      if aux_nchars_loss is not None and aux_nchars_w > 0:
        aux_nchars_loss = aux_nchars_loss.to(device)
        nchars_mask = res.get('aux_nchars_mask', torch.ones(B, device=device)).to(device)
        n_valid = nchars_mask.sum().clamp(min=1)
        scalars['loss/aux_nchars'] = (aux_nchars_loss.sum() / n_valid).item()
        loss = loss + aux_nchars_w * aux_nchars_loss

      aux_nspaces_w = getattr(FLAGS, 'aux_nspaces_weight', 0)
      aux_nspaces_loss = res.get('aux_nspaces_loss', None)
      if aux_nspaces_loss is not None and aux_nspaces_w > 0:
        aux_nspaces_loss = aux_nspaces_loss.to(device)
        nspaces_mask = res.get('aux_nspaces_mask', torch.ones(B, device=device)).to(device)
        n_valid = nspaces_mask.sum().clamp(min=1)
        scalars['loss/aux_nspaces'] = (aux_nspaces_loss.sum() / n_valid).item()
        loss = loss + aux_nspaces_w * aux_nspaces_loss

      # ---- MCER loss (REINFORCE) ----
      if mcer_loss is not None and _mcer_w > 0:
        loss = loss + _mcer_w * mcer_loss

      # ---- MWER loss (REINFORCE) ----
      if mwer_loss is not None and _mwer_w > 0:
        loss = loss + _mwer_w * mwer_loss

      # ---- Loss Truncation: zero out top-K% highest per-sample losses ----
      trunc_ratio = getattr(FLAGS, 'loss_truncation_ratio', 0.0)
      if trunc_ratio > 0 and loss.dim() > 0 and B > 1 and training:
        k = max(1, int(B * trunc_ratio))
        _, topk_idx = loss.detach().topk(k)
        trunc_mask = torch.ones(B, device=device)
        trunc_mask[topk_idx] = 0.0
        loss = loss * trunc_mask
        scalars['loss/truncated_frac'] = 1.0 - trunc_mask.mean().item()
      else:
        trunc_mask = None

      # ---- Weighted reduction ----
      if weights is not None and loss.dim() > 0:
        weights = weights.to(loss.device)
        if trunc_mask is not None:
          weights = weights * trunc_mask
        loss = (loss * weights).sum() / weights.sum().clamp(min=1e-8)
      elif getattr(FLAGS, 'mean_volume_loss', False) and loss.dim() > 0:
        # mean_volume reduction (matches NeMo native training_step):
        # loss.sum() / target_lengths.sum()
        # For RNNT/TDT: per-sample losses are totals (not per-token means),
        # so we divide total loss by total target tokens directly.
        s2s_tgt_lens = res.get('s2s_target_lengths', None)
        if s2s_tgt_lens is not None and s2s_tgt_lens.sum() > 0:
          s2s_tgt_lens = s2s_tgt_lens.to(loss.device)
          if trunc_mask is not None:
            loss = loss * trunc_mask
            s2s_tgt_lens = s2s_tgt_lens * trunc_mask
          loss = loss.sum() / s2s_tgt_lens.sum().clamp(min=1e-8)
        else:
          loss = loss.mean()
      elif getattr(FLAGS, 'corpus_level_loss', False) and loss.dim() > 0:
        # Corpus-level reduction: weight each sample by its token count,
        # so longer samples contribute more (matches corpus-level CER/WER).
        # loss_i is per-sample mean; multiply back by token count to get
        # per-sample total, then divide by total token count across batch.
        # NOTE: for RNNT/TDT per-sample losses are already totals, so use
        # --mean_volume_loss instead for NeMo models.
        # For CTC-only: use ctc_target_lengths
        # For S2S-only or hybrid: use s2s_token_counts
        # For multi-task: combined s2s counts are used for s2s part
        s2s_counts = res.get('s2s_token_counts', None)
        ipa_counts = res.get('ipa_token_counts', None)
        word_counts = res.get('word_token_counts', None)
        ctc_counts = res.get('ctc_target_lengths', None)

        # Build corpus-level weights: pick the dominant token counts.
        # When both CTC and S2S exist: S2S counts are used for combined loss
        # because S2S typically dominates the loss mix and has same labels.
        if s2s_counts is not None:
          corpus_weights = s2s_counts
        elif ipa_counts is not None:
          corpus_weights = ipa_counts
        elif ctc_counts is not None:
          corpus_weights = ctc_counts
        else:
          corpus_weights = None

        if corpus_weights is not None and corpus_weights.sum() > 0:
          corpus_weights = corpus_weights.to(loss.device)
          if trunc_mask is not None:
            corpus_weights = corpus_weights * trunc_mask
          loss = (loss * corpus_weights).sum() / corpus_weights.sum().clamp(min=1e-8)
        else:
          # Fallback: no valid corpus weights (e.g. all word-only batch with
          # ctc_target_lengths=0). Use simple mean so auxiliary loss still flows.
          loss = loss.mean()
      elif loss.dim() > 0:
        loss = loss.mean()

      # ---- Log weights info ----
      scalars['loss/total'] = loss.item()
      if ipa_s2s is not None or word_s2s is not None:
        scalars['info/ipa_weight'] = ipa_weight
        scalars['info/word_weight'] = word_weight
        if ipa_mask is not None:
          scalars['info/ipa_ratio'] = ipa_mask.mean().item()
        if word_mask is not None:
          scalars['info/word_ratio'] = word_mask.mean().item()

      le.update_scalars(scalars, decay=getattr(FLAGS, 'loss_decay', 0), training=training)
      return loss

    return calc_loss
