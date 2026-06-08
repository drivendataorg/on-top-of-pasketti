#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   preprocess.py
#        \author   chenghuige  
#          \date   2025-02-13
#   \Description   Shared data loading, preprocessing for Pasketti ASR.
#                  Track-specific settings (train file, label column) via FLAGS.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import librosa
import soundfile as sf
from gezi.common import * 
from src.config import *


# ================= Audio processing =================

def load_audio(audio_path, sr=None):
  """Load audio file and resample to target sample rate.
  
  Uses soundfile by default (FLAGS.use_soundfile=True) which is faster
  than librosa for .wav/.flac files. Falls back to librosa if needed.
  """
  sr = sr or FLAGS.sample_rate
  try:
    if getattr(FLAGS, 'use_soundfile', True):
      audio, file_sr = sf.read(str(audio_path), dtype='float32')
      if audio.ndim > 1:
        audio = audio.mean(axis=1)
      if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    else:
      audio, _ = librosa.load(audio_path, sr=sr)
  except Exception as e:
    ic(f"Error loading audio file: {audio_path}")
    raise e
  return audio


# ================= Feature extraction =================

def is_waveform_backbone(backbone=None):
  """Check if backbone expects raw waveform input (wav2vec2/hubert/wavlm/data2vec/moonshine/nemo)."""
  backbone = backbone or FLAGS.backbone
  backbone = BACKBONES.get(backbone, backbone).lower()
  return any(k in backbone for k in ['wav2vec2', 'hubert', 'wavlm', 'data2vec',
                                      'moonshine',
                                      'parakeet', 'fastconformer', 'canary',
                                      'nemo', 'conformer'])


def _is_nemo_backbone(backbone):
  """Check if backbone is a NeMo model (parakeet/fastconformer/canary)."""
  name = backbone.lower()
  return any(k in name for k in ['parakeet', 'fastconformer', 'canary', 'nemo', 'conformer'])


def load_nemo_tokenizer(backbone_name, model_dir=''):
  import sys as _sys
  import gc
  import importlib.machinery

  _old_paths = _sys.path[:]
  _sys.path = [p for p in _sys.path
                if 'aslfr/third/NeMo' not in p and 'aslfr\\third\\NeMo' not in p]
  _stale_keys = [k for k in _sys.modules if k.startswith('nemo') and
                 'aslfr' in str(getattr(_sys.modules[k], '__file__', '') or '')]
  for k in _stale_keys:
    del _sys.modules[k]

  try:
    import polars as pl
    if pl.__spec__ is None:
      pl.__spec__ = importlib.machinery.ModuleSpec('polars', None)
  except ImportError:
    pass

  try:
    import nemo.collections.asr as nemo_asr
  except ImportError as exc:
    raise ImportError('nemo_toolkit[asr] is required to load a NeMo tokenizer backbone') from exc
  finally:
    for p in _old_paths:
      if p not in _sys.path:
        _sys.path.append(p)

  full_backbone = BACKBONES.get(backbone_name, backbone_name)
  model = None
  if model_dir:
    model_dir = os.path.abspath(model_dir)
    slim_path = os.path.join(model_dir, 'nemo_model_slim.nemo')
    full_path = os.path.join(model_dir, 'nemo_model.nemo')
    if os.path.exists(full_path):
      model = nemo_asr.models.ASRModel.restore_from(full_path, map_location='cpu')
    elif os.path.exists(slim_path):
      model = nemo_asr.models.ASRModel.restore_from(slim_path, strict=False, map_location='cpu')

  if model is None:
    model = nemo_asr.models.ASRModel.from_pretrained(full_backbone, map_location='cpu')

  tokenizer = getattr(model, 'tokenizer', None)
  assert tokenizer is not None, f'Failed to load NeMo tokenizer for {full_backbone}'
  try:
    model.cpu()
  except Exception:
    pass
  del model
  gc.collect()
  try:
    import torch
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  except Exception:
    pass
  return tokenizer


processors = {}
def get_processor(backbone=None):
  """Get and cache the audio feature extractor / processor.

  - Whisper / distil-whisper -> WhisperProcessor
  - wav2vec2 / hubert / wavlm / data2vec / moonshine -> AutoFeatureExtractor (raw-waveform normaliser)
  - NeMo (parakeet, etc.) -> AutoFeatureExtractor from wav2vec2 (just normalises waveform)
  """
  backbone = backbone or FLAGS.backbone
  backbone = BACKBONES.get(backbone, backbone)
  if backbone in processors:
    return processors[backbone]
  if _is_nemo_backbone(backbone):
    # NeMo models have their own preprocessor built-in (mel conversion);
    # we just need to normalise the raw waveform, so use Wav2Vec2FeatureExtractor.
    # NOTE: Do NOT use WhisperFeatureExtractor here — it computes mel spectrograms,
    # but NeMo expects raw waveform input and does mel conversion internally.
    #
    # nemo_native_preprocess: when True, skip per-utterance mean/variance
    # normalization (do_normalize=False) so NeMo's preprocessor sees the
    # same raw waveform as model.transcribe(). The default do_normalize=True
    # changes the mel spectrum and degrades WER by ~3%.
    from transformers import Wav2Vec2FeatureExtractor
    do_normalize = not getattr(FLAGS, 'nemo_native_preprocess', False)
    processor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=do_normalize, return_attention_mask=True
    )
    if not do_normalize:
      ic(backbone, 'nemo_native_preprocess=True: do_normalize=False (raw waveform)')
  elif is_waveform_backbone(backbone):
    from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor
    try:
      processor = AutoFeatureExtractor.from_pretrained(FLAGS.model_dir)
    except Exception:
      try:
        processor = AutoFeatureExtractor.from_pretrained(backbone)
      except Exception:
        # Offline fallback: construct Wav2Vec2FeatureExtractor with standard params.
        # All wav2vec2/hubert/wavlm/data2vec models use the same feature extractor.
        processor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0,
            do_normalize=True, return_attention_mask=True
        )
  else:
    from transformers import WhisperProcessor
    try:
      processor = WhisperProcessor.from_pretrained(FLAGS.model_dir)
    except Exception:
      processor = WhisperProcessor.from_pretrained(backbone)
  processors[backbone] = processor
  return processor


tokenizers = {}

def _normalize_apostrophes_and_space(text):
  text = str(text or '').replace('\x92', "'").replace('\u2019', "'").replace('\u2018', "'")
  import re as _re
  text = _re.sub(r'\s+', ' ', text).strip()
  return text


def is_native_ctc_char_tokenizer(tokenizer):
  if tokenizer is None:
    return False
  vocab_size = getattr(tokenizer, 'vocab_size', None)
  if vocab_size is None or vocab_size > 64:
    return False
  try:
    vocab = tokenizer.get_vocab()
  except Exception:
    return False
  if '|' not in vocab:
    return False
  return all(ch in vocab for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def normalize_text_for_tokenizer(text, tokenizer):
  text = _normalize_apostrophes_and_space(text)
  if tokenizer is None:
    return text
  if is_native_ctc_char_tokenizer(tokenizer):
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")
    text = text.upper()
    text = ''.join(ch for ch in text if ch in allowed)
    text = _normalize_apostrophes_and_space(text)
  return text


def tokenize_text(tokenizer, text):
  if tokenizer is None:
    return []
  text = normalize_text_for_tokenizer(text, tokenizer)
  if not text:
    return []
  return tokenizer(text, add_special_tokens=False).input_ids


def get_tokenizer(backbone=None):
  """Get and cache the text tokenizer.

  For Whisper backbones the tokenizer comes from WhisperProcessor.
  For waveform backbones (wav2vec2/hubert/wavlm/data2vec/moonshine) we use a separate Whisper
  tokenizer so that labels stay consistent across all encoder types.

  Always prefer the fast (Rust) tokenizer — the slow Python tokenizer
  rebuilds added_tokens_encoder on every decode call and is ~100x slower.

  Tokenizer resolution order (waveform backbones):
    1. --word_tokenizer flag (if explicitly set):
       - "nemo"/"parakeet" → return None (NeMo SP loaded separately by model)
       - any BACKBONES key or HF model ID → load that tokenizer
    2. --word_tokenizer=None (default):
       - try backbone's native tokenizer (e.g. hubert-large-ft has CTC char tok)
       - fallback to --default_word_tokenizer (default: whisper → 50257 vocab)
  """
  backbone = backbone or FLAGS.backbone
  backbone = BACKBONES.get(backbone, backbone)
  if backbone in tokenizers:
    return tokenizers[backbone]

  # ---- word_tokenizer override ----
  word_tok = getattr(FLAGS, 'word_tokenizer', None)

  # NeMo backbones: always return None (SP tokenizer loaded by model)
  if _is_nemo_backbone(backbone):
    # Unless word_tokenizer explicitly points to a non-NeMo HF tokenizer
    if word_tok and not _is_nemo_backbone(BACKBONES.get(word_tok, word_tok)):
      from transformers import AutoTokenizer
      resolved = BACKBONES.get(word_tok, word_tok)
      tokenizer = AutoTokenizer.from_pretrained(resolved, use_fast=True)
      ic(word_tok, 'word_tokenizer override for NeMo backbone', resolved, tokenizer.vocab_size)
      tokenizers[backbone] = tokenizer
      return tokenizer
    tokenizer = None
    tokenizers[backbone] = tokenizer
    return tokenizer

  from transformers import AutoTokenizer

  if is_waveform_backbone(backbone):
    if word_tok is not None:
      # Explicit --word_tokenizer set
      if word_tok in ('nemo', 'parakeet') or _is_nemo_backbone(BACKBONES.get(word_tok, word_tok)):
        # NeMo tokenizer requested for non-NeMo backbone:
        # return None; model code will load NeMo SP tokenizer via load_nemo_tokenizer()
        ic(word_tok, 'word_tokenizer=nemo for waveform backbone, tokenizer=None (loaded by model)')
        tokenizers[backbone] = None
        return None
      resolved = BACKBONES.get(word_tok, word_tok)
      try:
        tokenizer = AutoTokenizer.from_pretrained(resolved, use_fast=True)
        ic(word_tok, 'word_tokenizer loaded', resolved, tokenizer.vocab_size)
      except Exception as e:
        raise RuntimeError(
          f'--word_tokenizer={word_tok} failed to load from {resolved}: {e}'
        ) from e
      tokenizers[backbone] = tokenizer
      return tokenizer

    # word_tokenizer=None: try backbone's native tokenizer, then fallback
    try:
      tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
      ic(backbone, 'backbone native tokenizer loaded', tokenizer.vocab_size)
      tokenizers[backbone] = tokenizer
      return tokenizer
    except Exception:
      pass

    # Fallback to default_word_tokenizer
    default_tok = getattr(FLAGS, 'default_word_tokenizer', 'whisper')
    if default_tok in ('whisper', 'whisper-large-v3'):
      tok_bb = 'openai/whisper-large-v3'
    else:
      tok_bb = BACKBONES.get(default_tok, default_tok)
    try:
      tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_dir, use_fast=True)
      ic(tok_bb, 'waveform-model tokenizer loaded from model_dir (fast)')
    except Exception:
      try:
        tokenizer = AutoTokenizer.from_pretrained(tok_bb, use_fast=True)
        ic(tok_bb, 'waveform-model tokenizer loaded (fast)')
      except Exception:
        # Offline fallback: phonetic track uses IPA char CTC (53 vocab),
        # does not need the BPE tokenizer.  Return None like NeMo.
        tokenizer = None
        ic(tok_bb, 'waveform-model tokenizer unavailable (offline), returning None')
  else:
    try:
      tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    except Exception:
      # fallback: extract from processor
      processor = get_processor(backbone)
      tokenizer = processor.tokenizer
    ic(backbone, 'tokenizer loaded', type(tokenizer).__name__)
  tokenizers[backbone] = tokenizer
  return tokenizer


# ================= Data loading =================

def _find_file(root, patterns):
  """Find first existing file matching any of the patterns."""
  import glob
  for pat in patterns:
    full = f'{root}/{pat}'
    matches = glob.glob(full)
    if matches:
      return matches[0]
    if os.path.exists(full):
      return full
  return None


def _build_aux_only_row(row, source, aux_kind, aux_text):
  assert aux_kind in ('word', 'ipa'), aux_kind
  aux_text = aux_text or ''
  data = {
    'utterance_id': row['utterance_id'],
    'id': row['utterance_id'],
    'audio_path': row.get('audio_path', ''),
    'audio_duration_sec': row.get('audio_duration_sec', 0.0),
    'age_bucket': row.get('age_bucket', ''),
    'child_id': row.get('child_id', ''),
    'session_id': row.get('session_id', ''),
    'source': source,
    'label_text': '',
    'ipa_label': '',
    'word_label': '',
  }
  data[f'{aux_kind}_label'] = aux_text
  return data


def _load_cross_labels(df):
  """Join cross-track labels onto df by utterance_id.
  
  For phonetic track: adds 'word_label' from train_word_transcripts.jsonl
  For word track: adds 'ipa_label' from train_phon_transcripts.jsonl
  
  Also handles:
  - DD cross-labels (same root dir or sibling track dir)
  - EXT cross-labels (ext_root always has both JSONL files)
  - use_word_only_dd / use_word_only_ext: add phonetic-track rows with only word labels
  - use_ipa_only_dd / use_ipa_only_ext: add word-track rows with only IPA labels
  """
  root = FLAGS.root
  ext_root = FLAGS.ext_root if getattr(FLAGS, 'use_ext', False) else None
  track = getattr(FLAGS, 'track', 'phonetic')
  
  # Determine cross-track file names
  if track == 'phonetic':
    # We have phonetic_text, need orthographic_text
    cross_dd_file_name = 'train_word_transcripts.jsonl'
    cross_col = 'orthographic_text'
    df['ipa_label'] = df['label_text']
    df['word_label'] = ''
  else:
    # We have orthographic_text, need phonetic_text
    cross_dd_file_name = 'train_phon_transcripts.jsonl'
    cross_col = 'phonetic_text'
    df['ipa_label'] = ''
    df['word_label'] = df['label_text']
  
  # ---- Load DD cross-labels ----
  cross_file = getattr(FLAGS, 'cross_label_file', '') or ''
  if not cross_file:
    # Auto-detect: check same root dir, then sibling track dir
    cross_file = _find_file(root, [cross_dd_file_name])
    if not cross_file:
      # Try sibling track directory
      # e.g. root=../input/childrens-phonetic-asr -> ../input/childrens-word-asr
      sibling = root.replace('phonetic', 'word') if track == 'phonetic' else root.replace('word', 'phonetic')
      cross_file = _find_file(sibling, [cross_dd_file_name])
  
  if cross_file and os.path.exists(cross_file):
    cross_rows = []
    with open(cross_file) as f:
      for line in f:
        row = json.loads(line)
        cross_rows.append(row)
    cross_df = pd.DataFrame(cross_rows)
    # Build lookup: utterance_id -> cross label
    cross_lookup = dict(zip(cross_df['utterance_id'], cross_df[cross_col].fillna('')))
    
    if track == 'phonetic':
      df['word_label'] = df['utterance_id'].map(cross_lookup).fillna('')
    else:
      df['ipa_label'] = df['utterance_id'].map(cross_lookup).fillna('')
    
    n_matched = (df['word_label'] != '').sum() if track == 'phonetic' else (df['ipa_label'] != '').sum()
    ic('cross-labels loaded (DD)', cross_file, len(cross_df), n_matched)
    
    # ---- Add aux-only DD samples ----
    if ((getattr(FLAGS, 'use_word_only_dd', False) and track == 'phonetic')
        or (getattr(FLAGS, 'use_ipa_only_dd', False) and track == 'word')):
      existing_ids = set(df['utterance_id'])
      aux_only_rows = []
      aux_kind = 'word' if track == 'phonetic' else 'ipa'
      for _, crow in cross_df.iterrows():
        uid = crow['utterance_id']
        if uid not in existing_ids and crow.get(cross_col, ''):
          aux_only_rows.append(_build_aux_only_row(crow, 'dd', aux_kind, crow[cross_col]))
      if aux_only_rows:
        wo_df = pd.DataFrame(aux_only_rows)
        # Aux-only DD audio is in the sibling track directory.
        sibling = root.replace('phonetic', 'word') if track == 'phonetic' else root.replace('word', 'phonetic')
        wo_df['audio_file'] = wo_df['audio_path'].apply(
          lambda x: f'{sibling}/{x}' if not os.path.isabs(str(x)) else str(x)
        )
        df = pd.concat([df, wo_df], ignore_index=True)
        ic(f'{aux_kind}-only DD added', len(aux_only_rows))
  else:
    ic('cross-labels file not found, skipping', cross_file)
  
  # ---- Load EXT cross-labels ----
  # EXT dir always has both train_phon_transcripts.jsonl and train_word_transcripts.jsonl
  if ext_root:
    ext_cross_file = _find_file(ext_root, [cross_dd_file_name])
    if ext_cross_file and os.path.exists(ext_cross_file):
      ext_cross_rows = []
      with open(ext_cross_file) as f:
        for line in f:
          ext_cross_rows.append(json.loads(line))
      ext_cross_df = pd.DataFrame(ext_cross_rows)
      ext_cross_lookup = dict(zip(ext_cross_df['utterance_id'], ext_cross_df[cross_col].fillna('')))
      
      # Update existing EXT rows
      ext_mask = df['source'] == 'ext'
      if ext_mask.any():
        if track == 'phonetic':
          df.loc[ext_mask, 'word_label'] = df.loc[ext_mask, 'utterance_id'].map(ext_cross_lookup).fillna('')
        else:
          df.loc[ext_mask, 'ipa_label'] = df.loc[ext_mask, 'utterance_id'].map(ext_cross_lookup).fillna('')
        n_ext_matched = (df.loc[ext_mask, 'word_label'] != '').sum() if track == 'phonetic' else (df.loc[ext_mask, 'ipa_label'] != '').sum()
        ic('cross-labels loaded (EXT)', ext_cross_file, len(ext_cross_df), n_ext_matched)
      
      # ---- Add aux-only EXT samples ----
      if ((getattr(FLAGS, 'use_word_only_ext', False) and track == 'phonetic')
          or (getattr(FLAGS, 'use_ipa_only_ext', False) and track == 'word')):
        existing_ids = set(df['utterance_id'])
        ext_wo_rows = []
        aux_kind = 'word' if track == 'phonetic' else 'ipa'
        for _, crow in ext_cross_df.iterrows():
          uid = crow['utterance_id']
          if uid not in existing_ids and crow.get(cross_col, ''):
            ext_wo_rows.append(_build_aux_only_row(crow, 'ext', aux_kind, crow[cross_col]))
        if ext_wo_rows:
          ext_wo_df = pd.DataFrame(ext_wo_rows)
          ext_wo_df['audio_file'] = ext_wo_df['audio_path'].apply(
            lambda x: f'{ext_root}/{x}' if not os.path.isabs(str(x)) else str(x)
          )
          df = pd.concat([df, ext_wo_df], ignore_index=True)
          ic(f'{aux_kind}-only EXT added', len(ext_wo_rows))
  
  # ---- Convert word_label → pseudo-IPA when any pseudo-IPA auxiliary branch is enabled ----
  if (getattr(FLAGS, 'pseudo_ipa_ctc', False)
      or getattr(FLAGS, 'word_tdt_pseudo_ipa', False)) and track == 'phonetic':
    df = _convert_word_labels_to_pseudo_ipa(df)
  
  return df


def _convert_word_labels_to_pseudo_ipa(df):
  """Convert word_label column from orthographic text to pseudo-IPA via eng_to_ipa.
  
  Supports two modes:
    1. Pre-computed: FLAGS.pseudo_ipa_file points to a JSONL with utterance_id + phonetic_text
    2. On-the-fly: uses eng_to_ipa.convert() to convert unique word texts → IPA
  """
  from src.models.base import _normalize_ipa as normalize_ipa, VALID_IPA_CHARS
  
  pseudo_ipa_file = getattr(FLAGS, 'pseudo_ipa_file', '')
  
  # Auto-detect pre-generated pseudo-IPA file if not explicitly set
  if not pseudo_ipa_file:
    auto_paths = [
      '../input/childrens-pseudo-ipa/train_phon_transcripts.jsonl',
      os.path.join(os.path.expanduser('~/data/drivendata/childrens-pseudo-ipa'), 'train_phon_transcripts.jsonl'),
    ]
    for p in auto_paths:
      if os.path.exists(p):
        pseudo_ipa_file = p
        ic('pseudo_ipa_file auto-detected', pseudo_ipa_file)
        break
  
  word_mask = df['word_label'].fillna('').str.strip() != ''
  n_before = word_mask.sum()
  
  if pseudo_ipa_file and os.path.exists(pseudo_ipa_file):
    # Load pre-computed pseudo-IPA lookup
    pseudo_lookup = {}
    with open(pseudo_ipa_file) as f:
      for line in f:
        row = json.loads(line)
        pseudo_lookup[row['utterance_id']] = row.get('phonetic_text', '')
    
    # Map utterance_id → pseudo-IPA; keep original word_label if not in lookup
    df.loc[word_mask, 'word_label'] = df.loc[word_mask].apply(
      lambda r: pseudo_lookup.get(r['utterance_id'], r['word_label']), axis=1
    )
    
    # Fallback: overlapping samples (have word_label but NOT in pseudo-IPA file)
    # → convert their English word_label to pseudo-IPA via eng_to_ipa on-the-fly.
    # We do NOT use ground-truth ipa_label here because dual-IPA relies on the
    # two heads seeing DIFFERENT IPA targets (ground-truth vs eng_to_ipa) for
    # diversity in ensemble decoding.
    still_english = word_mask & (~df['utterance_id'].isin(pseudo_lookup))
    n_still = still_english.sum()
    if n_still > 0:
      missing_ids = df.loc[still_english, 'utterance_id'].tolist()
      try:
        from eng_to_ipa import convert as ipa_convert
      except ImportError:
        logger.warning(
          'pseudo_ipa_file is missing %d overlap samples and eng_to_ipa is not installed; '
          'clearing word_label for these rows so word auxiliary is skipped. '
          'First missing utterance_ids: %s',
          n_still,
          missing_ids[:10],
        )
        df.loc[still_english, 'word_label'] = ''
        n_mapped = (df['word_label'].fillna('').str.strip() != '').sum()
        ic('pseudo-IPA loaded from file (missing overlap skipped)',
           pseudo_ipa_file, len(pseudo_lookup), n_before, n_mapped,
           'missing_overlap', n_still)
        return df
      overlap_texts = sorted(set(df.loc[still_english, 'word_label'].unique()) - {''})
      ic('pseudo-IPA fallback: converting overlapping samples via eng_to_ipa', n_still, len(overlap_texts))
      text_to_ipa = {}
      for text in overlap_texts:
        raw = ipa_convert(text).replace('*', '')
        normed = normalize_ipa(raw)
        cleaned = ''.join(ch for ch in normed if ch in VALID_IPA_CHARS)
        text_to_ipa[text] = cleaned
      df.loc[still_english, 'word_label'] = df.loc[still_english, 'word_label'].map(
        lambda t: text_to_ipa.get(t, '')
      )
    
    n_mapped = (df['word_label'].fillna('').str.strip() != '').sum()
    ic('pseudo-IPA loaded from file', pseudo_ipa_file, len(pseudo_lookup), n_before, n_mapped,
       'on-the-fly fallback', n_still)
  else:
    # On-the-fly conversion via eng_to_ipa
    try:
      from eng_to_ipa import convert as ipa_convert
    except ImportError:
      raise ImportError(
        'eng_to_ipa is required for --pseudo_ipa_ctc without --pseudo_ipa_file. '
        'Install: pip install eng_to_ipa. '
        'Or use gen_pseudo_ipa.py to pre-generate and set --pseudo_ipa_file.')
    
    # Batch-convert unique word texts
    unique_texts = sorted(set(df.loc[word_mask, 'word_label'].unique()) - {''})
    ic('converting word_label to pseudo-IPA on-the-fly', len(unique_texts))
    
    text_to_ipa = {}
    for text in unique_texts:
      raw = ipa_convert(text).replace('*', '')
      normed = normalize_ipa(raw)
      cleaned = ''.join(ch for ch in normed if ch in VALID_IPA_CHARS)
      text_to_ipa[text] = cleaned
    
    df.loc[word_mask, 'word_label'] = df.loc[word_mask, 'word_label'].map(
      lambda t: text_to_ipa.get(t, '')
    )
    n_mapped = (df['word_label'].fillna('').str.strip() != '').sum()
    n_empty = sum(1 for v in text_to_ipa.values() if not v.strip())
    ic('pseudo-IPA converted', len(text_to_ipa), n_empty, n_before, n_mapped)
  
  return df


def _ipa_convert_worker(text):
  """Convert a single text → IPA. Top-level function for multiprocessing pickling."""
  from eng_to_ipa import convert as ipa_convert
  from src.models.base import _normalize_ipa as normalize_ipa, VALID_IPA_CHARS
  raw = ipa_convert(text).replace('*', '')
  normed = normalize_ipa(raw)
  cleaned = ''.join(ch for ch in normed if ch in VALID_IPA_CHARS)
  return text, cleaned


def _auto_generate_pseudo_ipa(output_dir, workers=None):
  """Generate pseudo-IPA from ALL word samples (DD+EXT, including overlap) if data doesn't exist.
  
  Always generates the full set. Runtime filtering by --include_overlap
  is handled in prepare() after loading.
  """
  from multiprocessing import Pool, cpu_count
  from pathlib import Path

  train_file = getattr(FLAGS, 'train_file', 'train_phon_transcripts.jsonl')
  output_jsonl = os.path.join(output_dir, train_file)
  if os.path.exists(output_jsonl):
    return  # already generated

  workers = workers or min(cpu_count(), 16)
  logger.info(f'Auto-generating pseudo-IPA at {output_dir} with {workers} workers...')

  # Data root: parent of FLAGS.root (e.g. ../input/childrens-phonetic-asr → ../input/)
  data_root = os.path.dirname(os.path.abspath(FLAGS.root))

  dd_word_jsonl = os.path.join(data_root, 'childrens-word-asr', 'train_word_transcripts.jsonl')
  ext_word_jsonl = os.path.join(data_root, 'childrens-ext-asr', 'train_word_transcripts.jsonl')
  dd_word_audio = os.path.join(data_root, 'childrens-word-asr', 'audio')
  ext_audio = os.path.join(data_root, 'childrens-ext-asr', 'audio')

  def _load_jsonl_local(path):
    rows = []
    with open(path) as f:
      for line in f:
        rows.append(json.loads(line))
    return rows

  # Load source data
  assert os.path.exists(dd_word_jsonl), f'DD word JSONL not found: {dd_word_jsonl}'
  dd_word = _load_jsonl_local(dd_word_jsonl)

  samples = []  # (row, audio_source_dir)
  # DD: all word samples (word-only + overlap)
  for r in dd_word:
    samples.append((r, dd_word_audio))
  # EXT: all word samples (word-only + overlap)
  if os.path.exists(ext_word_jsonl):
    ext_word = _load_jsonl_local(ext_word_jsonl)
    for r in ext_word:
      samples.append((r, ext_audio))

  logger.info(f'  Samples to convert: {len(samples)}')
  all_texts = sorted(set(r['orthographic_text'] for r, _ in samples))
  logger.info(f'  Unique texts: {len(all_texts)}')

  # Parallel eng_to_ipa conversion
  from tqdm import tqdm
  ipa_cache = {}
  with Pool(workers) as pool:
    for text, ipa in tqdm(pool.imap_unordered(_ipa_convert_worker, all_texts, chunksize=256),
                          total=len(all_texts), desc='eng→ipa', unit='text'):
      ipa_cache[text] = ipa

  # Create output directory and write
  output_path = Path(output_dir)
  audio_dir = output_path / 'audio'
  audio_dir.mkdir(parents=True, exist_ok=True)

  output_rows = []
  for r, audio_src_dir in samples:
    ipa_text = ipa_cache.get(r['orthographic_text'], '')
    if not ipa_text.strip():
      continue
    uid = r['utterance_id']
    src_audio = os.path.join(audio_src_dir, f'{uid}.flac')
    if not os.path.isfile(src_audio):
      continue
    link_path = audio_dir / f'{uid}.flac'
    if not link_path.exists():
      os.symlink(os.path.abspath(src_audio), link_path)
    output_rows.append({
      'utterance_id': uid,
      'child_id': r.get('child_id', ''),
      'session_id': r.get('session_id', ''),
      'audio_path': f'audio/{uid}.flac',
      'audio_duration_sec': r.get('audio_duration_sec', 0),
      'age_bucket': r.get('age_bucket', ''),
      'phonetic_text': ipa_text,
    })

  with open(output_jsonl, 'w') as f:
    for row in output_rows:
      f.write(json.dumps(row, ensure_ascii=False) + '\n')

  logger.info(f'  Generated {len(output_rows)} pseudo-IPA samples at {output_jsonl}')


def _filter_ext_eval_leakage(df):
  """Remove ext (fold=-1) samples whose child_id overlaps with DD eval fold.
  
  Prevents data leakage when ext data contains DD-origin samples
  (e.g. pseudo-IPA generated from DD word-only data with shared child_ids).
  Skipped in online mode (all data used).
  """
  if FLAGS.online:
    return df

  fold = getattr(FLAGS, 'fold', None)
  if fold is None:
    return df

  if not all(c in df.columns for c in ['source', 'child_id', 'fold']):
    return df

  dd_eval_mask = (df['source'] == 'dd') & (df['fold'] == fold)
  if dd_eval_mask.sum() == 0:
    return df

  eval_child_ids = set(df.loc[dd_eval_mask, 'child_id'].unique())
  # Only filter ext with fold=-1 (always-train); don't touch ext with proper fold assignments
  ext_leak_mask = (df['source'] == 'ext') & (df['fold'] == -1) & df['child_id'].isin(eval_child_ids)
  n_leak = ext_leak_mask.sum()
  if n_leak > 0:
    ic(f'ext leakage filter: removed {n_leak} ext samples (child_id in DD eval fold {fold})')
    df = df[~ext_leak_mask].reset_index(drop=True)

  return df


def load_df(mode='train'):
  """Load training data or test manifest.
  
  Track-specific file / column names come from FLAGS:
    FLAGS.train_file, FLAGS.label_column, FLAGS.label_column_fallback
  """
  root = FLAGS.root
  
  if mode == 'test':
    file = _find_file(root, ['utterance_metadata.jsonl', 'test.jsonl', 'test.csv'])
    assert file, f'No test manifest found in {root}'
    if file.endswith('.csv'):
      df = pd.read_csv(file)
    else:
      rows = []
      with open(file) as f:
        for line in f:
          rows.append(json.loads(line))
      df = pd.DataFrame(rows)
    df['id'] = df['utterance_id']
    ic(file, len(df))
    return df
  
  # train mode: track-specific file first, then generic fallbacks
  file = _find_file(root, [
    FLAGS.train_file,   # e.g. train_phon_transcripts.jsonl / train_word_transcripts.jsonl
    'train.jsonl',
    'train.csv',
  ])
  assert file, f'No training data found in {root}'
  
  if file.endswith('.csv'):
    df = pd.read_csv(file)
  else:
    rows = []
    with open(file) as f:
      for line in f:
        rows.append(json.loads(line))
    df = pd.DataFrame(rows)
  
  if 'id' not in df.columns:
    df['id'] = df['utterance_id']
  
  # mark source for DD vs ext tracking
  df['source'] = 'dd'
  
  # ---------- merge extended (TalkBank) data ----------
  if FLAGS.use_ext and FLAGS.ext_root:
    # Auto-generate pseudo-IPA if ext_root doesn't have the expected data yet
    ext_file_check = _find_file(FLAGS.ext_root, [FLAGS.train_file, 'train.jsonl', 'train.csv'])
    if not ext_file_check:
      _auto_generate_pseudo_ipa(FLAGS.ext_root)
    ext_file = _find_file(FLAGS.ext_root, [
      FLAGS.train_file,
      'train.jsonl',
      'train.csv',
    ])
    if ext_file:
      if ext_file.endswith('.csv'):
        ext_df = pd.read_csv(ext_file)
      else:
        ext_rows = []
        with open(ext_file) as f:
          for line in f:
            ext_rows.append(json.loads(line))
        ext_df = pd.DataFrame(ext_rows)
      if 'id' not in ext_df.columns:
        ext_df['id'] = ext_df['utterance_id']
      ext_df['source'] = 'ext'
      # Runtime filtering: exclude overlap samples if include_overlap=False
      # Only applies when ext_root is pseudo-IPA data (not the original ext/phon data)
      _is_pseudo_ipa = 'pseudo' in os.path.basename(os.path.normpath(FLAGS.ext_root)).lower()
      if not getattr(FLAGS, 'include_overlap', False) and _is_pseudo_ipa:
        data_root = os.path.dirname(os.path.abspath(FLAGS.root))
        _phon_ids = set()
        for _pf in ['childrens-phonetic-asr/train_phon_transcripts.jsonl',
                     'childrens-ext-asr/train_phon_transcripts.jsonl']:
          _pf_path = os.path.join(data_root, _pf)
          if os.path.exists(_pf_path):
            with open(_pf_path) as _f:
              for _line in _f:
                _phon_ids.add(json.loads(_line)['utterance_id'])
        n_before = len(ext_df)
        ext_df = ext_df[~ext_df['utterance_id'].isin(_phon_ids)]
        ic('include_overlap=False: filtered overlap from pseudo-ipa',
           n_before, len(ext_df), n_before - len(ext_df))
      ic(ext_file, len(ext_df))
      df = pd.concat([df, ext_df], ignore_index=True)
      ic('merged DD+ext', len(df))
  
  # target column: FLAGS.label_column → fallback → empty
  if FLAGS.label_column in df.columns:
    df['label_text'] = df[FLAGS.label_column].fillna('')
  elif FLAGS.label_column_fallback and FLAGS.label_column_fallback in df.columns:
    df['label_text'] = df[FLAGS.label_column_fallback].fillna('')
  else:
    df['label_text'] = ''
  
  # ---------- cross-track labels for multi-task learning ----------
  # When use_cross_labels=True, load the other track's labels and join by utterance_id.
  # For phonetic track: load orthographic_text from word JSONL
  # For word track: load phonetic_text from phonetic JSONL
  if getattr(FLAGS, 'use_cross_labels', False):
    df = _load_cross_labels(df)
  else:
    # Ensure columns exist even when disabled (simplifies downstream code)
    if 'ipa_label' not in df.columns:
      if FLAGS.label_column == 'phonetic_text' or (FLAGS.label_column_fallback == 'ipa_text'):
        df['ipa_label'] = df['label_text']
        df['word_label'] = ''
      else:
        df['ipa_label'] = ''
        df['word_label'] = df['label_text']
  
  ic(file, len(df), df.columns.tolist())
  return df


def _flag_present(name):
  try:
    return bool(FLAGS[name].present)
  except Exception:
    return False


def _apply_word_eval_defaults(has_ext):
  """Keep word-track eval_add_ext validation aligned with the phonetic setup."""
  if getattr(FLAGS, 'track', '') != 'word':
    return
  if not has_ext or not getattr(FLAGS, 'eval_add_ext', False):
    return

  updates = {}
  if not _flag_present('fold_group_key') and FLAGS.fold_group_key != 'child_id':
    FLAGS.fold_group_key = 'child_id'
    updates['fold_group_key'] = 'child_id'
  if not _flag_present('fold_stratify_key') and FLAGS.fold_stratify_key != 'age_bucket':
    FLAGS.fold_stratify_key = 'age_bucket'
    updates['fold_stratify_key'] = 'age_bucket'
  if not _flag_present('folds') and (FLAGS.folds is None or FLAGS.folds == 4):
    FLAGS.folds = 5
    updates['folds'] = 5
  if not _flag_present('ext_eval_group') and not getattr(FLAGS, 'ext_eval_group', False):
    FLAGS.ext_eval_group = True
    updates['ext_eval_group'] = True
  if not _flag_present('eval_ext_full') and not getattr(FLAGS, 'eval_ext_full', False):
    FLAGS.eval_ext_full = True
    updates['eval_ext_full'] = True
  if not _flag_present('eval_ext_weight') and getattr(FLAGS, 'eval_ext_weight', 1.0) != 1.0:
    FLAGS.eval_ext_weight = 1.0
    updates['eval_ext_weight'] = 1.0

  if updates:
    ic('word eval_add_ext defaults', updates)


def set_folds(df, sgkf_compat=None):
  """Assign k-folds to df (adds 'fold' column).
  Uses GroupKFold when FLAGS.fold_group_key is set (e.g. 'child_id').
  Uses StratifiedGroupKFold when both group_key and stratify_key are set.
  
  sgkf_compat: controls StratifiedGroupKFold behaviour.
    '1.6.1' — reproduce sklearn <=1.6.1 (buggy shuffle, default for backward compat).
    '1.8.0' — reproduce sklearn >=1.8.0 (fixed shuffle, correct even folds).
    None    — read from FLAGS.sgkf_compat (default).
  
  When use_ext=True and eval_ext=False:
    Only fold-split DrivenData rows; ext rows get fold=-1 (always train).
  When use_ext=True and eval_ext=True:
    Merge both datasets then apply the same fold-split logic to all rows.

  fold_align_file: JSON mapping group_key -> fold for cross-track alignment.
    Overlapping children get the specified fold; the rest are assigned normally.
  """
  if sgkf_compat is None:
    sgkf_compat = getattr(FLAGS, 'sgkf_compat', '1.6.1') or None
  if 'fold' not in df.columns:
    has_ext = 'source' in df.columns and (df['source'] == 'ext').any()
    _apply_word_eval_defaults(has_ext)
    group_key = FLAGS.fold_group_key if FLAGS.fold_group_key else None
    stratify_key = FLAGS.fold_stratify_key if FLAGS.fold_stratify_key else None
    
    # Load cross-track fold alignment mapping if specified
    fold_align_map = None
    if FLAGS.fold_align_file:
      if not os.path.exists(FLAGS.fold_align_file):
        raise FileNotFoundError(
            f'fold_align_file={FLAGS.fold_align_file} was explicitly set but does not exist')
      with open(FLAGS.fold_align_file) as _f:
        raw = json.load(_f)
      # Support both old format (flat dict) and new format (with _meta)
      if '_meta' in raw:
        meta = raw['_meta']
        fold_align_map = raw['child2fold']
        # Validate: folds and sgkf_compat must match current config
        if meta['folds'] != FLAGS.folds:
          raise ValueError(
            f"fold_align_file was generated with folds={meta['folds']} "
            f"but current FLAGS.folds={FLAGS.folds}. "
            f"Re-run: python tools/gen_fold_align.py --track={meta['track']} "
            f"--folds={FLAGS.folds} --sgkf_compat={sgkf_compat}")
        if meta.get('sgkf_compat') != sgkf_compat:
          raise ValueError(
            f"fold_align_file was generated with sgkf_compat={meta.get('sgkf_compat')} "
            f"but current sgkf_compat={sgkf_compat}. "
            f"Re-run: python tools/gen_fold_align.py --track={meta['track']} "
            f"--folds={FLAGS.folds} --sgkf_compat={sgkf_compat}")
      else:
        fold_align_map = raw  # old flat format, no validation
      ic('fold_align: loaded', FLAGS.fold_align_file, len(fold_align_map))

    cross_track_child2fold_map = None
    if getattr(FLAGS, 'cross_track_child2fold_file', ''):
      cross_track_child2fold_map = _load_child2fold_map(
          FLAGS.cross_track_child2fold_file,
          expected_label='cross_track_child2fold_file')
      ic('cross_track_child2fold: loaded',
         FLAGS.cross_track_child2fold_file,
         len(cross_track_child2fold_map),
         f'pretrain_fold={getattr(FLAGS, "pretrain_fold", -1)}')
    
    # eval_add_ext: do NOT assign folds to ext — all ext stays fold=-1.
    # Eval samples N ext; train gets all remaining ext.
    # (Previously this forced eval_ext=True, but that did CV on ext which we don't want.)
    
    if has_ext and not FLAGS.eval_ext:
      # CV only on DrivenData; ext data always in train (fold=-1)
      dd_mask = df['source'] == 'dd'
      ext_mask = df['source'] == 'ext'
      # Word-only DD rows (label_text='') are training-only auxiliaries.
      # Exclude them from fold assignment so they don't disturb
      # StratifiedGroupKFold splits (keeps folds identical to v13 baseline).
      if 'label_text' in df.columns:
        _has_label = df['label_text'].fillna('').str.strip() != ''
        dd_cv_mask = dd_mask & _has_label
      else:
        dd_cv_mask = dd_mask
      dd_df = df.loc[dd_cv_mask].copy()
      dd_df = _assign_folds_with_alignment(dd_df, group_key, stratify_key,
                                           sgkf_compat, fold_align_map)
      df['fold'] = -1  # default: ext + word-only DD → always train
      df.loc[dd_cv_mask, 'fold'] = dd_df['fold'].values
      ic('set_folds: DD-only CV', dd_cv_mask.sum(), 'word_only_dd', (dd_mask & ~dd_cv_mask).sum() if 'label_text' in df.columns else 0)
      # ext_eval_group: assign folds to ext by child_id (GroupKFold) to
      # prevent train/eval child leakage. Ext fold assignments are independent
      # of DD folds (different child populations).
      # IMPORTANT: only use base ext (with label_text) for GroupKFold so that
      # fold assignments are consistent regardless of use_word_only_ext.
      # Word-only ext inherits fold from matching child_id afterwards.
      if getattr(FLAGS, 'ext_eval_group', False):
        if 'label_text' in df.columns:
          _ext_has_label = ext_mask & (df['label_text'].fillna('').str.strip() != '')
          _ext_wo_mask = ext_mask & ~_ext_has_label
        else:
          _ext_has_label = ext_mask
          _ext_wo_mask = pd.Series(False, index=df.index)
        ext_df = df.loc[_ext_has_label].copy()
        _ext_group_key = 'child_id' if 'child_id' in ext_df.columns else group_key
        assert _ext_group_key and _ext_group_key in ext_df.columns, \
            f'ext_eval_group requires child_id column in ext data, got columns: {ext_df.columns.tolist()}'
        _ext_stratify = stratify_key if (stratify_key and stratify_key in ext_df.columns) else None
        gz.set_fold(ext_df, FLAGS.folds, group_key=_ext_group_key,
                    stratify_key=_ext_stratify, seed=FLAGS.fold_seed,
                    sgkf_compat=sgkf_compat)
        df.loc[_ext_has_label, 'fold'] = ext_df['fold'].values
        # Propagate fold to word-only ext via child_id (same child = same fold)
        if _ext_wo_mask.any() and _ext_group_key in df.columns:
          _child_fold_map = dict(zip(ext_df[_ext_group_key], ext_df['fold']))
          df.loc[_ext_wo_mask, 'fold'] = df.loc[_ext_wo_mask, _ext_group_key].map(_child_fold_map).fillna(-1).astype(int)
        ic('set_folds: ext_eval_group enabled, ext folds assigned by child_id',
           _ext_has_label.sum(), ext_df['fold'].value_counts().to_dict(),
           f'word_only_ext={_ext_wo_mask.sum()}')
      # Propagate DD fold assignments to ext samples with matching utterance_ids.
      # Handles pseudo-IPA ext containing DD-origin samples: their utterance_ids
      # match DD phon data, so they must share the same fold to avoid leakage.
      if 'utterance_id' in df.columns:
        _dd_with_fold = df.loc[dd_cv_mask, ['utterance_id', 'fold']].drop_duplicates('utterance_id')
        _dd_fold_map = dict(zip(_dd_with_fold['utterance_id'], _dd_with_fold['fold']))
        _ext_dd_match = ext_mask & df['utterance_id'].isin(_dd_fold_map)
        if _ext_dd_match.any():
          df.loc[_ext_dd_match, 'fold'] = df.loc[_ext_dd_match, 'utterance_id'].map(_dd_fold_map)
          ic('set_folds: propagated DD folds to matching ext utterance_ids', _ext_dd_match.sum())
    elif has_ext and FLAGS.eval_ext:
      # Split DD and ext independently so that DD fold assignments stay
      # identical to the --use_ext=0 case (ext does not disturb DD CV).
      dd_mask = df['source'] == 'dd'
      ext_mask = df['source'] == 'ext'
      # Exclude word-only DD from fold assignment (same as above)
      if 'label_text' in df.columns:
        _has_label = df['label_text'].fillna('').str.strip() != ''
        dd_cv_mask = dd_mask & _has_label
      else:
        dd_cv_mask = dd_mask
      dd_df = df.loc[dd_cv_mask].copy()
      ext_df = df.loc[ext_mask].copy()
      dd_df = _assign_folds_with_alignment(dd_df, group_key, stratify_key,
                                           sgkf_compat, fold_align_map)
      gz.set_fold(ext_df, FLAGS.folds, group_key=group_key,
                  stratify_key=stratify_key, seed=FLAGS.fold_seed,
                  sgkf_compat=sgkf_compat)
      df['fold'] = -1  # word-only DD → always train
      df.loc[dd_cv_mask, 'fold'] = dd_df['fold'].values
      df.loc[ext_mask, 'fold'] = ext_df['fold'].values
      ic('set_folds: DD+ext independent CV', dd_cv_mask.sum(), ext_mask.sum())
      # Propagate DD folds to ext with matching utterance_ids (same as above)
      if 'utterance_id' in df.columns:
        _dd_with_fold = df.loc[dd_cv_mask, ['utterance_id', 'fold']].drop_duplicates('utterance_id')
        _dd_fold_map = dict(zip(_dd_with_fold['utterance_id'], _dd_with_fold['fold']))
        _ext_dd_match = ext_mask & df['utterance_id'].isin(_dd_fold_map)
        if _ext_dd_match.any():
          df.loc[_ext_dd_match, 'fold'] = df.loc[_ext_dd_match, 'utterance_id'].map(_dd_fold_map)
          ic('set_folds: propagated DD folds to matching ext utterance_ids', _ext_dd_match.sum())
    else:
      # Normal fold split on all data (or no ext data)
      # Exclude word-only rows from fold assignment if present
      if 'label_text' in df.columns and 'source' in df.columns:
        _has_label = df['label_text'].fillna('').str.strip() != ''
        _wordonly = ~_has_label
        if _wordonly.any():
          cv_df = df.loc[_has_label].copy()
          cv_df = _assign_folds_with_alignment(cv_df, group_key, stratify_key,
                                               sgkf_compat, fold_align_map)
          df['fold'] = -1
          df.loc[_has_label, 'fold'] = cv_df['fold'].values
        else:
          df = _assign_folds_with_alignment(df, group_key, stratify_key,
                                            sgkf_compat, fold_align_map)
      else:
        df = _assign_folds_with_alignment(df, group_key, stratify_key,
                                          sgkf_compat, fold_align_map)
    df = _apply_cross_track_eval_guard(
        df, group_key, cross_track_child2fold_map)
  return df


def _assign_folds_with_alignment(df, group_key, stratify_key, sgkf_compat,
                                  fold_align_map=None):
  """Assign folds, then override for cross-track aligned children.
  
  When fold_align_map is provided (e.g. from phonetic track's mapping):
  1. First run normal StratifiedGroupKFold on ALL rows.
  2. Then override fold assignments for children present in fold_align_map.
  
  This ensures:
  - Overlapping children get the SAME fold as the source track → no leakage.
  - Non-overlapping children keep their normal fold assignment.
  - Fold sizes may shift slightly but remain reasonable (overlap is small
    compared to total, e.g. 103 overlap vs 2175 word-only children).
  """
  gz.set_fold(df, FLAGS.folds, group_key=group_key,
              stratify_key=stratify_key, seed=FLAGS.fold_seed,
              sgkf_compat=sgkf_compat)
  
  if fold_align_map and group_key:
    overridden = 0
    changed = 0
    for child_id, target_fold in fold_align_map.items():
      mask = df[group_key] == child_id
      if mask.any():
        old_fold = df.loc[mask, 'fold'].iloc[0]
        if old_fold != target_fold:
          changed += 1
        df.loc[mask, 'fold'] = target_fold
        overridden += 1
    ic('fold_align: overridden', overridden, 'changed', changed)
  
  return df


def _load_child2fold_map(path, expected_label='cross_track_child2fold_file'):
  if not path:
    return None
  if not os.path.exists(path):
    raise FileNotFoundError(
        f'{expected_label}={path} was explicitly set but does not exist')
  with open(path) as f:
    raw = json.load(f)
  if isinstance(raw, dict) and '_meta' in raw:
    raw = raw.get('child2fold', {})
  return {str(k): int(v) for k, v in raw.items()}


def _apply_cross_track_eval_guard(df, group_key, child2fold_map):
  """Exclude unsafe overlap children from the current eval fold only.

  This is used when the current track uses a different fold count from the
  source track (for example word 10-fold vs phonetic 5-fold). We keep the
  current track's native fold assignment, but for the *current run* we remove
  overlap children from eval unless they belonged to the matching source-track
  eval fold.
  """
  if not getattr(FLAGS, 'cross_track_eval_safe_only', False):
    return df
  if not child2fold_map or not group_key:
    return df
  if getattr(FLAGS, 'pretrain_fold', -1) < 0:
    raise ValueError(
        'cross_track_eval_safe_only=True requires --pretrain_fold>=0')
  if 'fold' not in df.columns:
    return df

  eval_fold = FLAGS.fold
  pretrain_fold = int(FLAGS.pretrain_fold)
  unique_children = df[group_key].dropna().astype(str)
  overlap_children = set(unique_children) & set(map(str, child2fold_map.keys()))
  if not overlap_children:
    ic('cross_track_eval_guard: no overlapping children found')
    return df

  unsafe_children = {
      child for child in overlap_children
      if int(child2fold_map[str(child)]) != pretrain_fold
  }
  if not unsafe_children:
    ic('cross_track_eval_guard: all overlap children safe',
       f'pretrain_fold={pretrain_fold}',
       f'overlap_children={len(overlap_children)}')
    return df

  child_series = df[group_key].astype(str)
  unsafe_eval_mask = (df['fold'] == eval_fold) & child_series.isin(unsafe_children)
  if not unsafe_eval_mask.any():
    ic('cross_track_eval_guard: no unsafe overlap children in current eval fold',
       f'pretrain_fold={pretrain_fold}',
       f'eval_fold={eval_fold}',
       f'overlap_children={len(overlap_children)}',
       f'unsafe_children={len(unsafe_children)}')
    return df

  excluded_children = set(child_series[unsafe_eval_mask])
  df.loc[unsafe_eval_mask, 'fold'] = -1
  ic('cross_track_eval_guard: excluded unsafe overlap children from eval',
     f'pretrain_fold={pretrain_fold}',
     f'eval_fold={eval_fold}',
     f'overlap_children={len(overlap_children)}',
     f'unsafe_children={len(unsafe_children)}',
     f'excluded_children={len(excluded_children)}',
     f'excluded_rows={int(unsafe_eval_mask.sum())}')
  return df


def prepare(df):
  """Preprocess DataFrame columns."""
  df = df.copy()
  if 'audio_path' in df.columns:
    # Some rows may already have audio_file set (e.g. word-only DD samples
    # added by _load_cross_labels pointing to the sibling track directory).
    # Only resolve audio_path for rows that don't already have audio_file.
    has_audio_file = 'audio_file' in df.columns
    # NaN (from concat with word-only rows) must be treated as "not set".
    # NaN is truthy so ~astype(bool) would wrongly mark NaN as "already resolved".
    needs_resolve = (df['audio_file'].isna() | (df['audio_file'] == '')) if has_audio_file else pd.Series(True, index=df.index)
    
    has_ext = 'source' in df.columns and (df['source'] == 'ext').any()
    if has_ext:
      # Resolve audio_path relative to the correct root for each source
      dd_mask = (df['source'] == 'dd') & needs_resolve
      ext_mask = (df['source'] == 'ext') & needs_resolve
      if not has_audio_file:
        df['audio_file'] = ''
      df.loc[dd_mask, 'audio_file'] = df.loc[dd_mask, 'audio_path'].apply(
        lambda x: f'{FLAGS.root}/{x}' if not os.path.isabs(str(x)) else str(x)
      )
      df.loc[ext_mask, 'audio_file'] = df.loc[ext_mask, 'audio_path'].apply(
        lambda x: f'{FLAGS.ext_root}/{x}' if not os.path.isabs(str(x)) else str(x)
      )
    else:
      root = FLAGS.root
      if not has_audio_file:
        df['audio_file'] = ''
      df.loc[needs_resolve, 'audio_file'] = df.loc[needs_resolve, 'audio_path'].apply(
        lambda x: f'{root}/{x}' if not os.path.isabs(str(x)) else str(x)
      )
  if 'audio_duration_sec' not in df.columns:
    df['audio_duration_sec'] = 0.0

  # Filter out corrupted/missing audio files
  if 'audio_file' in df.columns:
    before = len(df)
    # file must exist and be at least 10KB (empty/corrupt FLAC headers are ~8KB with no frames)
    valid_mask = df['audio_file'].apply(
      lambda p: os.path.isfile(str(p)) and os.path.getsize(str(p)) > 1024 * 10
    )
    bad = df[~valid_mask]
    n_bad = len(bad)
    if n_bad > 0:
      ic(f'Filtering {n_bad} bad/missing audio files', bad['audio_file'].tolist()[:20])
      # Safety check: if >5% of data is missing, it's likely a config/path error, not corrupt files
      max_missing_ratio = 0.05
      missing_ratio = n_bad / before
      if missing_ratio > max_missing_ratio:
        # Show per-source breakdown for diagnosis
        if 'source' in df.columns:
          bad_by_source = bad.groupby('source').size().to_dict()
          total_by_source = df.groupby('source').size().to_dict()
          ic(bad_by_source, total_by_source)
        raise RuntimeError(
          f'Too many missing/bad audio files: {n_bad}/{before} ({missing_ratio:.1%}) > {max_missing_ratio:.0%} threshold. '
          f'This likely indicates a data path or download issue, not file corruption. '
          f'Check that audio files exist at the expected paths. '
          f'First few missing: {bad["audio_file"].tolist()[:5]}'
        )
    df = df[valid_mask].reset_index(drop=True)
    ic(f'audio filter: {before} -> {len(df)}')

  # Log weak-alignment stats when flag is enabled
  wa_weight = getattr(FLAGS, 'weak_align_weight', 0)
  if wa_weight > 0 and 'audio_duration_sec' in df.columns:
    cps_thr = getattr(FLAGS, 'weak_align_cps_threshold', 3.0)
    label_col = 'phonetic_text' if 'phonetic_text' in df.columns else 'label_text'
    if label_col in df.columns:
      n_chars = df[label_col].fillna('').str.replace(' ', '', regex=False).str.len()
      dur = df['audio_duration_sec'].clip(lower=0.01)
      cps = n_chars / dur
      n_weak = (cps < cps_thr).sum()
      if 'age_bucket' in df.columns:
        weak_by_age = df[cps < cps_thr].groupby('age_bucket').size().to_dict()
      else:
        weak_by_age = {}
      ic(f'weak_align: {n_weak}/{len(df)} samples with cps<{cps_thr} will be weighted {wa_weight}',
         weak_by_age)

  return df


dfs = {}
def preprocess(mode='train'):
  """Full preprocessing pipeline with caching."""
  if mode in dfs:
    return dfs[mode]
  
  df = load_df(mode)
  
  if mode != 'test':
    df = set_folds(df)
    df = _filter_ext_eval_leakage(df)
  
  df = prepare(df)
  ic(mode, len(df), df.columns.tolist())
  
  dfs[mode] = df
  return df
