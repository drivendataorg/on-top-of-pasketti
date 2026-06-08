#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   nemo.py
#        \author   chenghuige
#          \date   2025-02-20
#   \Description   NeMo ASR models (Parakeet, FastConformer, Canary) for Pasketti.
#
#  Supports two usage modes:
#    1. CTC-only (ctc_weight=1): Use NeMo encoder + our CTC head
#       - Fine-tunable with our training pipeline
#       - Ideal for phonetic track (IPA char-level CTC)
#    2. Inference-only (ctc_weight=0): Use NeMo's built-in transcribe()
#       - No fine-tuning, just pretrained model inference
#       - Good for word track baseline / pseudo-labeling
#
#  NeMo models use .nemo format, loaded via nemo_toolkit:
#    pip install nemo_toolkit[asr]
#
#  Usage:
#    --model=nemo --backbone=parakeet-ctc-0.6b --ctc_weight=1.0
#    --model=nemo --backbone=parakeet-tdt-0.6b --ctc_weight=0  (inference only)
#    --model=nemo --backbone=fastconformer-ctc-large --ctc_weight=1.0
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
from src.config import *
from src.models.base import BaseASRModel
from src.nemo_trim_vocab_data import (
  BUILTIN_NEMO_TRIM_DEFAULT_TOPK,
  BUILTIN_NEMO_TRIM_RANKED_TOKEN_IDS,
)

import torch
import torch.nn as nn
import copy
import json


_AUG_ONCE_LOCAL_KEYS = set()


def _log_nemo_aug_once(name, info=None):
  if not getattr(FLAGS, 'aug_show_once', False):
    return
  aug_flag = name if name.startswith('aug_') else f'aug_{name}'
  payload = {'aug': name, 'aug_flag': aug_flag}
  if info:
    payload.update(info)
  ic_once(payload, key=f'aug_once:{name}')


def _should_capture_nemo_aug_once(key):
  if not getattr(FLAGS, 'aug_show_once', False):
    return False
  if key in _AUG_ONCE_LOCAL_KEYS:
    return False
  _AUG_ONCE_LOCAL_KEYS.add(key)
  return True


def _tensor_summary(x):
  if isinstance(x, torch.Tensor):
    arr = x.detach().float().cpu()
    if arr.numel() == 0:
      return {'shape': list(arr.shape), 'size': 0}
    return {
        'shape': list(arr.shape),
        'size': int(arr.numel()),
        'mean': round(float(arr.mean().item()), 6),
        'std': round(float(arr.std(unbiased=False).item()), 6),
        'min': round(float(arr.min().item()), 6),
        'max': round(float(arr.max().item()), 6),
        'zero_frac': round(float((arr == 0).float().mean().item()), 6),
    }
  return {'type': type(x).__name__}


def _load_nemo_model(backbone_name):
  """Load NeMo ASR model.

  Tries model_dir first (for fine-tuned), then downloads from HuggingFace.
  Returns (nemo_model, encoder_dim).
  """
  # Remove any stale third-party NeMo paths from sys.path to ensure we use
  # the pip-installed nemo_toolkit, not old project-local copies.
  import sys as _sys
  _old_paths = _sys.path[:]
  _sys.path = [p for p in _sys.path
                if 'aslfr/third/NeMo' not in p and 'aslfr\\third\\NeMo' not in p]
  # Also remove any already-imported old nemo submodules
  _stale_keys = [k for k in _sys.modules if k.startswith('nemo') and 
                  'aslfr' in str(getattr(_sys.modules[k], '__file__', '') or '')]
  for k in _stale_keys:
    del _sys.modules[k]
  # Workaround: polars loaded via .whl sys.path injection may have __spec__=None,
  # which breaks datasets.config (importlib.util.find_spec check). Fix it.
  try:
    import polars as _pl
    if _pl.__spec__ is None:
      import importlib.machinery
      _pl.__spec__ = importlib.machinery.ModuleSpec('polars', None)
  except ImportError:
    pass
  try:
    import nemo.collections.asr as nemo_asr
  except ImportError:
    _sys.path = _old_paths
    raise ImportError(
        'nemo_toolkit is required for NeMo models. '
        'Install with: pip install nemo_toolkit[asr]')
  finally:
    # Restore paths (other projects may need them)
    for p in _old_paths:
      if p not in _sys.path:
        _sys.path.append(p)

  # Try loading from model_dir (fine-tuned checkpoint)
  model = None
  nemo_path = os.path.join(FLAGS.model_dir, 'nemo_model.nemo')
  nemo_slim_path = os.path.join(FLAGS.model_dir, 'nemo_model_slim.nemo')
  if os.path.exists(nemo_path):
    model = nemo_asr.models.ASRModel.restore_from(nemo_path)
    logger.debug(f'NeMo model loaded from {nemo_path}')
  elif os.path.exists(nemo_slim_path):
    # Slim .nemo: architecture + tokenizer only (no weights).
    # Weights will be overridden by model.pt / best.pt via load_weights().
    model = nemo_asr.models.ASRModel.restore_from(nemo_slim_path, strict=False)
    logger.debug(f'NeMo model (architecture only) loaded from {nemo_slim_path}')
  else:
    # Download from HuggingFace
    model = nemo_asr.models.ASRModel.from_pretrained(backbone_name)
    logger.debug(f'NeMo model loaded from {backbone_name}')

  # Get encoder hidden dimension
  encoder_dim = None
  if hasattr(model, 'encoder'):
    # FastConformer / Conformer — try config first
    enc_cfg = getattr(model.encoder, 'cfg', None)
    if enc_cfg is not None:
      # OmegaConf DictConfig: use bracket access (hasattr may fail)
      try:
        encoder_dim = enc_cfg['d_model']
      except (KeyError, TypeError):
        pass
      if encoder_dim is None:
        try:
          encoder_dim = enc_cfg.d_model
        except Exception:
          pass
  # Fallback: probe from ConformerLayer's LayerNorm dimensions
  if encoder_dim is None and hasattr(model, 'encoder'):
    layers = getattr(model.encoder, 'layers', None)
    if layers is not None and len(layers) > 0:
      norm_out = getattr(layers[0], 'norm_out', None)
      if norm_out is not None and hasattr(norm_out, 'normalized_shape'):
        encoder_dim = norm_out.normalized_shape[0]
  if encoder_dim is None:
    encoder_dim = _probe_encoder_dim(model)

  logger.debug(f'NeMo model type: {type(model).__name__}, encoder_dim={encoder_dim}')
  return model, encoder_dim


def _probe_encoder_dim(model):
  """Probe encoder output dimension with a dummy input.
  
  NeMo ConformerEncoder returns (B, D, T) format (channels-first),
  so the feature dimension is shape[1], not shape[-1].
  """
  with torch.no_grad():
    # NeMo models expect (B, T) raw audio or (B, C, T) mel
    dummy = torch.randn(1, 16000 * 2).to(model.device)  # 2 seconds for safety
    try:
      # Most NeMo ASR models have preprocessor + encoder
      processed, proc_len = model.preprocessor(
          input_signal=dummy,
          length=torch.tensor([16000 * 2], device=model.device))
      enc_out, enc_len = model.encoder(audio_signal=processed, length=proc_len)
      # NeMo encoder output: (B, D, T) — D is dim 1
      # Heuristic: D (feature dim, typically 256~1024) < T (time steps)
      # but for very short audio T can be small, so prefer dim 1
      return enc_out.shape[1]
    except Exception as e:
      logger.warning(f'Failed to probe encoder dim: {e}, defaulting to 512')
      return 512


class _RemappedSentencePieceTokenizer(object):
  def __init__(self, base_tokenizer, keep_token_ids):
    keep_token_ids = [int(token_id) for token_id in keep_token_ids]
    assert keep_token_ids, 'trimmed NeMo vocab requires at least one kept token id'
    self._base = base_tokenizer
    self.keep_token_ids = tuple(keep_token_ids)
    self.vocab_size = len(self.keep_token_ids)
    self.blank_id = self.vocab_size
    self.pad_token_id = 0
    self._orig_to_new = {token_id: idx for idx, token_id in enumerate(self.keep_token_ids)}
    self._new_to_orig = {idx: token_id for idx, token_id in enumerate(self.keep_token_ids)}
    self.all_special_ids = []
    tokenizer_impl = getattr(base_tokenizer, 'tokenizer', None)
    self._tokenizer_impl = tokenizer_impl
    self._pieces = {}
    if tokenizer_impl is not None and hasattr(tokenizer_impl, 'id_to_piece'):
      for token_id in self.keep_token_ids:
        try:
          self._pieces[token_id] = tokenizer_impl.id_to_piece(int(token_id))
        except Exception:
          self._pieces[token_id] = f'<id:{token_id}>'

  @property
  def tokenizer(self):
    return self._tokenizer_impl

  def __getattr__(self, name):
    return getattr(self._base, name)

  def _map_orig_to_new(self, token_ids, text=''):
    mapped = []
    missing = []
    for token_id in token_ids:
      token_id = int(token_id)
      mapped_id = self._orig_to_new.get(token_id)
      if mapped_id is None:
        missing.append(token_id)
      else:
        mapped.append(mapped_id)
    if missing:
      missing_preview = missing[:8]
      piece_preview = [self._pieces.get(token_id, str(token_id)) for token_id in missing_preview]
      raise ValueError(
          'Trimmed NeMo vocab missing token ids for input text; '
          f'topk is too small or keep list is incompatible. text={text!r} '
          f'missing_ids={missing_preview} missing_pieces={piece_preview}')
    return mapped

  def _map_new_to_orig(self, token_ids):
    orig = []
    for token_id in token_ids:
      token_id = int(token_id)
      if token_id == self.blank_id:
        continue
      mapped = self._new_to_orig.get(token_id)
      if mapped is None:
        raise ValueError(f'Unknown compact token id {token_id} for trimmed NeMo vocab')
      orig.append(mapped)
    return orig

  def text_to_ids(self, text):
    orig_ids = self._base.text_to_ids(text)
    return self._map_orig_to_new(orig_ids, text=text)

  def encode(self, text):
    return self.text_to_ids(text)

  def ids_to_text(self, ids):
    orig_ids = self._map_new_to_orig(ids)
    return self._base.ids_to_text(orig_ids)

  def ids_to_tokens(self, ids):
    orig_ids = self._map_new_to_orig(ids)
    return [self._pieces.get(token_id, str(token_id)) for token_id in orig_ids]

  def batch_decode(self, batch_ids, skip_special_tokens=True):
    return [self.ids_to_text(ids) for ids in batch_ids]


def _load_trimmed_nemo_keep_ids(path, topk, backbone_name=None):
  token_ids = None
  if path:
    with open(path, 'r', encoding='utf-8') as stream:
      payload = json.load(stream)
    if isinstance(payload, dict):
      if 'ranked_token_ids' in payload:
        token_ids = payload['ranked_token_ids']
      elif 'top_tokens' in payload:
        token_ids = [row['token_id'] for row in payload['top_tokens']]
      elif 'token_ids' in payload:
        token_ids = payload['token_ids']
      else:
        raise ValueError(
            'nemo_trim_vocab_file JSON must contain ranked_token_ids, top_tokens, or token_ids')
    elif isinstance(payload, list):
      token_ids = payload
    else:
      raise ValueError('nemo_trim_vocab_file must be a JSON list or JSON object')
  elif backbone_name and backbone_name in BUILTIN_NEMO_TRIM_RANKED_TOKEN_IDS:
    token_ids = BUILTIN_NEMO_TRIM_RANKED_TOKEN_IDS[backbone_name]
  else:
    raise ValueError(
        f'No trimmed NeMo vocab source for backbone={backbone_name!r}; '
        'set nemo_trim_vocab_file or add a builtin ranked token list')

  token_ids = [int(token_id) for token_id in token_ids]
  if topk > 0:
    source_name = path or f'builtin:{backbone_name}'
    assert len(token_ids) >= topk, (
        f'nemo_trim_vocab_topk={topk} exceeds available token ids {len(token_ids)} in {source_name}')
    token_ids = token_ids[:topk]

  deduped = []
  seen = set()
  for token_id in token_ids:
    if token_id in seen:
      continue
    seen.add(token_id)
    deduped.append(token_id)
  assert deduped, f'No token ids loaded from {path}'
  return deduped


def _collect_loaded_trim_texts():
  try:
    from src import preprocess as preprocess_module
  except Exception:
    return []

  dfs = getattr(preprocess_module, 'dfs', None)
  if not dfs:
    return []

  texts = []
  seen = set()
  for _, df in dfs.items():
    if df is None or getattr(df, 'empty', False):
      continue
    for column in ('label_text', 'word_label'):
      if column not in df.columns:
        continue
      values = df[column].fillna('').astype(str)
      for text in values:
        text = text.strip()
        if not text or text in seen:
          continue
        seen.add(text)
        texts.append(text)
  return texts


def _expand_trimmed_nemo_keep_ids_for_loaded_texts(base_tokenizer, keep_ids, backbone_name=None):
  texts = _collect_loaded_trim_texts()
  if not texts:
    return keep_ids, 0, 0

  keep_set = set(int(token_id) for token_id in keep_ids)
  missing_counts = {}
  for text in texts:
    token_ids = base_tokenizer.text_to_ids(text)
    for token_id in token_ids:
      token_id = int(token_id)
      if token_id in keep_set:
        continue
      missing_counts[token_id] = missing_counts.get(token_id, 0) + 1

  if not missing_counts:
    return keep_ids, len(texts), 0

  rank_map = {}
  if backbone_name and backbone_name in BUILTIN_NEMO_TRIM_RANKED_TOKEN_IDS:
    rank_map = {
        int(token_id): idx
        for idx, token_id in enumerate(BUILTIN_NEMO_TRIM_RANKED_TOKEN_IDS[backbone_name])
    }
  missing_ids = sorted(
      missing_counts,
      key=lambda token_id: (rank_map.get(token_id, 10**12), -missing_counts[token_id], token_id),
  )
  expanded = list(keep_ids) + missing_ids
  return expanded, len(texts), len(missing_ids)


class Model(BaseASRModel):
  """NeMo ASR model wrapper.

  NeMo models (Parakeet, FastConformer, etc.) have their own preprocessor,
  encoder, and decoder. This wrapper:
    - Uses NeMo's preprocessor for audio feature extraction
    - Exposes the encoder output for our CTC head
    - Supports NeMo's built-in transcribe() for inference-only mode
  """

  def __init__(self, **kwargs):
    # ---- NeMo native CTC (方案B): auto-set ctc_weight before super().__init__ ----
    self._nemo_native_ctc_requested = getattr(FLAGS, 'nemo_native_ctc', False)
    self._nemo_native_ctc = False
    self._nemo_native_ctc_decoder_source = None
    self._requested_s2s_decoder = getattr(FLAGS, 's2s_decoder', 'native')
    self._tdt_scratch_fallback = False
    prefer_s2s_branch = self._requested_s2s_decoder in ('tdt_reuse', 'rnnt_reuse')
    if self._nemo_native_ctc_requested and FLAGS.ctc_weight <= 0 and not prefer_s2s_branch:
      FLAGS.ctc_weight = 1.0
      logger.debug('nemo_native_ctc: auto-set ctc_weight=1.0')

    super().__init__(**kwargs)

    backbone_name = BACKBONES.get(FLAGS.backbone, FLAGS.backbone)

    # Suppress verbose NeMo logging during model load + adapter setup in inference mode
    # (tdt_kwargs, tokenizer config, training/validation data warnings, adapter/freeze ~50 lines)
    _is_inference = getattr(FLAGS, 'mode', '') in ('test', 'eval')
    if _is_inference:
      self._suppress_nemo_verbose()
    self.nemo_model, self.encoder_dim = _load_nemo_model(backbone_name)
    self.backbone = self.nemo_model  # alias for train.py get_opt_params

    # Note: nemo_model.eval() is called here for initial loading;
    # the training loop will call model.train() which propagates to
    # all submodules including nemo_model.
    self.nemo_model.eval()

    # ---- NeMo adapter setup (matches official benchmark) ----
    self._nemo_adapter = getattr(FLAGS, 'nemo_adapter', False)
    if self._nemo_adapter:
      self._setup_nemo_adapter()

    # ---- NeMo LoRA (PEFT) setup ----
    self._nemo_lora = getattr(FLAGS, 'nemo_lora', False)
    if self._nemo_lora:
      assert not self._nemo_adapter, \
          '--nemo_lora and --nemo_adapter are mutually exclusive'
      self._setup_nemo_lora()

    # We keep NeMo's preprocessor (mel extraction) and encoder
    self.preprocessor = self.nemo_model.preprocessor
    self.encoder = self.nemo_model.encoder

    # Freeze preprocessor (always — it's just mel extraction)
    for param in self.preprocessor.parameters():
      param.requires_grad = False

    # Optionally freeze encoder (skip if adapter/LoRA mode — already handled)
    if FLAGS.freeze_encoder and not self._nemo_adapter and not self._nemo_lora:
      for param in self.encoder.parameters():
        param.requires_grad = False
      logger.debug('NeMo encoder frozen')

    # Gradient checkpointing for NeMo ConformerEncoder
    if FLAGS.gradient_checkpointing:
      self._enable_nemo_gradient_checkpointing()

    # Keep a reference to NeMo's SentencePiece tokenizer early,
    # so _init_ctc_head can use it for word_ctc_bpe vocab sizing.
    self._original_nemo_tokenizer = self.nemo_model.tokenizer
    self._nemo_tokenizer = self.nemo_model.tokenizer
    self._nemo_trim_vocab_enabled = False
    self._nemo_trim_keep_ids = None
    self._nemo_trim_compact_size = None
    self._maybe_enable_trimmed_nemo_vocab()

    # ---- NeMo native CTC mode (方案B) ----
    # Keep NeMo's pretrained CTC decoder when explicitly requested and supported.
    # This now works for both pure CTC backbones and hybrid TDT+CTC backbones.
    self._nemo_native_ctc = self._should_enable_nemo_native_ctc()
    if self._nemo_native_ctc:
      self._init_nemo_native_ctc()
      self._init_word_ctc_head(self.encoder_dim)
    else:
      # ---- CTC head (our own, for fine-tuning) ----
      self._init_ctc_head(self.encoder_dim)

    # ---- InterCTC ----
    # Determine number of Conformer layers
    nemo_layers = getattr(self.encoder, 'layers', None)
    num_encoder_layers = len(nemo_layers) if nemo_layers is not None else 18
    if not self._nemo_native_ctc:
      self._init_inter_ctc(self.encoder_dim, num_encoder_layers)

    # ---- CTC layer fusion ----
    self._init_ctc_layer_fusion(num_encoder_layers)

    # ---- Auxiliary metadata heads (age / domain) ----
    self._init_aux_heads(self.encoder_dim)

    # Encoder length cache (set in _encode, used by _s2s_forward)
    self._last_enc_len = None

    # Check if NeMo model has RNNT/TDT decoder+joint for s2s training
    self._has_rnnt_loss = (hasattr(self.nemo_model, 'decoder')
                           and hasattr(self.nemo_model, 'joint')
                           and hasattr(self.nemo_model, 'loss'))

    requested_s2s_decoder = getattr(FLAGS, 's2s_decoder', 'native')
    requires_word_tdt = (getattr(FLAGS, 'word_tdt_pseudo_ipa', False)
                         or getattr(FLAGS, 'word_tdt_mixed', False))
    can_skip_s2s_reuse = self.ctc_only and not requires_word_tdt
    if (not self._has_rnnt_loss
        and requested_s2s_decoder in ('rnnt_reuse', 'tdt_reuse')):
      if requested_s2s_decoder == 'tdt_reuse':
        if not requires_word_tdt:
          self._tdt_scratch_fallback = True
          logger.info(
              'tdt_reuse requested but backbone has no RNNT/TDT decoder; '
              'falling back to TDT scratch init that mimics parakeet-tdt head.')
        else:
          raise ValueError(
              'tdt_reuse with word_tdt_* requires a NeMo model with RNNT/TDT decoder '
              f'(e.g. parakeet-tdt-* or parakeet-tdt_ctc-*). Current backbone '
              f'{FLAGS.backbone} is pure CTC and cannot support this TDT branch.')
      elif can_skip_s2s_reuse:
        logger.info(
            'rnnt_reuse: backbone has no RNNT/TDT decoder, '
            'but this run is effectively CTC-only; skipping reuse init.')
        FLAGS.s2s_decoder = 'native'
      else:
        raise ValueError(
            f'{requested_s2s_decoder} requires a NeMo model with RNNT/TDT decoder '
            f'(e.g. parakeet-tdt-* or parakeet-tdt_ctc-*). '
            f'Current backbone {FLAGS.backbone} is pure CTC and cannot support this '
            'non-CTC branch.')

    # ---- Custom S2S decoder (Schemes 1/2/3) ----
    if not self._nemo_native_ctc:
      self._init_s2s_decoder(self.encoder_dim)

      # Scheme 1 (rnnt_reuse): extract pretrained LSTM from NeMo RNNT decoder
      if getattr(FLAGS, 's2s_decoder', 'native') == 'rnnt_reuse':
        self._init_rnnt_reuse(self.encoder_dim)

      # Scheme 4 (tdt_reuse): exact TDT replica with IPA vocab
      if getattr(FLAGS, 's2s_decoder', 'native') == 'tdt_reuse':
        if self._tdt_scratch_fallback:
          self._init_tdt_scratch(self.encoder_dim)
        else:
          self._init_tdt_reuse(self.encoder_dim)

    # Whether to use NeMo's native transcribe for generation
    # (only when not using our CTC head, i.e., ctc_weight=0)
    self._use_native_transcribe = not self.use_ctc and not self._nemo_native_ctc

    if self._use_native_transcribe:
      if self._has_rnnt_loss:
        logger.info('NeMo s2s mode: using TDT/RNNT loss for fine-tuning, '
                     'native decode for generation')
      else:
        logger.info('NeMo s2s mode: no RNNT decoder available — '
                     'inference only (use --ctc_weight=1 for CTC training)')

    # Always keep a reference to NeMo's own tokenizer (SentencePiece)
    # so _s2s_generate can use it even when HF tokenizer is None
    if not hasattr(self, '_nemo_tokenizer'):
      self._nemo_tokenizer = self.nemo_model.tokenizer
    # Store globally so eval.py decode_ids can access it
    import gezi
    gezi.set('nemo_tokenizer', self._nemo_tokenizer)

    # Decoding strategy:
    # - Training: use slow-but-safe greedy (non-batched, no CUDA graphs)
    #   to avoid CUDA error 35 on some driver/GPU combos
    # - Inference: keep NeMo default greedy_batch + CUDA graphs (~2-3x faster)
    if _is_inference:
      # Keep NeMo default: greedy_batch with GreedyBatchedTDTInfer + CUDA graphs
      logger.info('Inference mode: keeping NeMo default greedy_batch decoding (fast)')
    else:
      self._disable_nemo_cuda_graphs()

    # Restore NeMo logging after all init is done
    if _is_inference:
      self._restore_nemo_verbose()

    self._init_nemo_spec_augment()

    self._log_params()

    # For adapter/LoRA mode, set opt_params to only trainable weights.
    # This avoids optimizer maintaining state for millions of frozen params.
    if self._nemo_adapter or self._nemo_lora:
      import gezi
      opt_params = [p for p in self.parameters() if p.requires_grad]
      gezi.set('opt_params', opt_params)

  def _maybe_enable_trimmed_nemo_vocab(self):
    trim_file = getattr(FLAGS, 'nemo_trim_vocab_file', '')
    backbone = BACKBONES.get(FLAGS.backbone, FLAGS.backbone)
    topk = int(getattr(FLAGS, 'nemo_trim_vocab_topk', 0) or 0)
    if not trim_file and topk <= 0:
      return

    assert 'parakeet-tdt' in backbone or 'parakeet-rnnt' in backbone, (
        'nemo_trim_vocab_file currently only supports NeMo TDT/RNNT backbones')
    assert not getattr(FLAGS, 'constrain_ipa', False), (
        'nemo_trim_vocab_file is for word-piece NeMo tokenizer paths, not IPA-constrained runs')
    assert hasattr(self.nemo_model, 'decoder') and hasattr(self.nemo_model.decoder, 'prediction'), (
        'nemo_trim_vocab_file requires a NeMo decoder with prediction embedding')
    assert hasattr(self.nemo_model, 'joint') and hasattr(self.nemo_model.joint, 'joint_net'), (
        'nemo_trim_vocab_file requires a NeMo joint network')

    keep_ids = _load_trimmed_nemo_keep_ids(
        trim_file,
      topk=topk,
      backbone_name=backbone,
    )
    requested_compact_vocab = len(keep_ids)
    keep_ids, scanned_texts, added_for_coverage = _expand_trimmed_nemo_keep_ids_for_loaded_texts(
      self._original_nemo_tokenizer,
      keep_ids,
      backbone_name=backbone,
    )
    orig_vocab = int(getattr(self._original_nemo_tokenizer, 'vocab_size', 0) or 0)
    assert orig_vocab > 0, 'Failed to infer original NeMo tokenizer vocab size'
    assert all(0 <= token_id < orig_vocab for token_id in keep_ids), (
        f'Invalid token ids for trimmed NeMo vocab: expected [0, {orig_vocab - 1}]')

    decoder = self.nemo_model.decoder
    embed = decoder.prediction.embed
    blank_orig_id = orig_vocab
    compact_vocab = len(keep_ids)

    padding_idx = compact_vocab if getattr(decoder, 'blank_as_pad', False) else None
    new_embed = nn.Embedding(compact_vocab + 1, embed.embedding_dim, padding_idx=padding_idx)
    new_embed = new_embed.to(embed.weight.device, dtype=embed.weight.dtype)
    keep_index = torch.tensor(keep_ids + [blank_orig_id], dtype=torch.long, device=embed.weight.device)
    with torch.no_grad():
      new_embed.weight.copy_(embed.weight.index_select(0, keep_index))
    decoder.prediction.embed = new_embed
    if hasattr(decoder, 'vocab_size'):
      decoder.vocab_size = compact_vocab
    if hasattr(decoder, 'blank_idx'):
      decoder.blank_idx = compact_vocab

    joint = self.nemo_model.joint
    joint_out = joint.joint_net[-1]
    num_extra_outputs = int(getattr(joint, 'num_extra_outputs', 0) or 0)
    token_keep = keep_ids + [blank_orig_id]
    extra_rows = [orig_vocab + 1 + idx for idx in range(num_extra_outputs)]
    compact_num_classes_with_blank = len(token_keep) + num_extra_outputs
    row_index = torch.tensor(token_keep + extra_rows, dtype=torch.long, device=joint_out.weight.device)
    new_joint_out = nn.Linear(joint_out.in_features, len(token_keep) + num_extra_outputs,
                              bias=joint_out.bias is not None)
    new_joint_out = new_joint_out.to(joint_out.weight.device, dtype=joint_out.weight.dtype)
    with torch.no_grad():
      new_joint_out.weight.copy_(joint_out.weight.index_select(0, row_index))
      if joint_out.bias is not None:
        new_joint_out.bias.copy_(joint_out.bias.index_select(0, row_index))
    joint.joint_net[-1] = new_joint_out
    if hasattr(joint, '_vocab_size'):
      joint._vocab_size = compact_vocab
    if hasattr(joint, '_num_classes'):
      joint._num_classes = compact_num_classes_with_blank
    if hasattr(joint, 'blank_id'):
      try:
        blank_tensor = joint.blank_id
        if isinstance(blank_tensor, torch.Tensor):
          blank_tensor.fill_(compact_vocab)
      except Exception:
        pass

    loss_modules = []
    for loss_module in [getattr(joint, '_loss', None), getattr(self.nemo_model, 'loss', None)]:
      if loss_module is None:
        continue
      if any(loss_module is existing for existing in loss_modules):
        continue
      loss_modules.append(loss_module)
    for loss_module in loss_modules:
      if hasattr(loss_module, '_blank'):
        loss_module._blank = compact_vocab
      inner_loss = getattr(loss_module, '_loss', None)
      if inner_loss is not None and hasattr(inner_loss, 'blank'):
        inner_loss.blank = compact_vocab

    trimmed_tokenizer = _RemappedSentencePieceTokenizer(self._original_nemo_tokenizer, keep_ids)
    self.nemo_model.tokenizer = trimmed_tokenizer
    self._nemo_tokenizer = trimmed_tokenizer
    self._nemo_trim_vocab_enabled = True
    self._nemo_trim_keep_ids = tuple(keep_ids)
    self._nemo_trim_compact_size = compact_vocab

    if hasattr(self.nemo_model, 'change_decoding_strategy'):
      try:
        self._set_nemo_decode_strategy('greedy')
      except Exception as e:
        logger.warning(f'Failed to refresh NeMo decoding after trimmed vocab init: {e}')

    logger.info(
        'NeMo trimmed vocab enabled: '
        f'orig_vocab={orig_vocab} requested_compact_vocab={requested_compact_vocab} '
        f'compact_vocab={compact_vocab} keep_source={trim_file or "builtin"} topk={topk} '
        f'coverage_scan_texts={scanned_texts} coverage_added={added_for_coverage}')

  def _get_nemo_native_ctc_decoder(self):
    """Find a reusable native CTC decoder on the loaded NeMo model.

    Priority:
      1. Hybrid TDT+CTC models expose `ctc_decoder`
      2. Pure CTC models expose `decoder`
    """
    candidates = [
        ('ctc_decoder', getattr(self.nemo_model, 'ctc_decoder', None)),
        ('decoder', getattr(self.nemo_model, 'decoder', None)),
    ]
    for source, decoder in candidates:
      if decoder is None:
        continue
      vocab = getattr(decoder, 'vocabulary', None)
      if vocab is not None:
        return decoder, source, vocab
    return None, None, None

  def _should_enable_nemo_native_ctc(self):
    """Decide whether to reuse NeMo's native CTC decoder.

    Default behaviour stays unchanged:
      - only activates when `--nemo_native_ctc` is requested
      - if the backbone cannot support reuse, safely falls back to our CTC head
    """
    if not self._nemo_native_ctc_requested:
      return False
    if not self.use_ctc:
      logger.info('nemo_native_ctc requested but ctc_weight<=0 after init; fallback to project CTC path')
      return False
    if getattr(self, 'constrain_ipa', False):
      logger.info('nemo_native_ctc requested but constrain_ipa/custom IPA vocab is active; '
                  'falling back to project CTC head')
      return False

    decoder, source, vocab = self._get_nemo_native_ctc_decoder()
    if decoder is None:
      logger.info('nemo_native_ctc requested but backbone has no reusable native CTC decoder '
                  '(tdt-only / rnnt-only). Falling back to project CTC head.')
      return False

    self._nemo_native_ctc_decoder_source = source
    logger.info(f'nemo_native_ctc: reusing NeMo {source} '
                f'vocab={len(vocab)} backbone={FLAGS.backbone}')
    return True

  def _should_reuse_native_word_ctc(self):
    if not getattr(FLAGS, 'word_ctc', False):
      return False
    if not getattr(FLAGS, 'word_ctc_bpe', False):
      return False
    if getattr(FLAGS, 'pseudo_ipa_ctc', False):
      return False

    decoder, source, vocab = self._get_nemo_native_ctc_decoder()
    if decoder is None:
      return False

    num_classes = None
    if hasattr(decoder, 'num_classes_with_blank'):
      num_classes = decoder.num_classes_with_blank
    elif hasattr(decoder, 'cfg'):
      try:
        num_classes = decoder.cfg.num_classes
      except Exception:
        pass
    if num_classes is None:
      num_classes = len(vocab) + 1

    self._word_native_ctc_decoder_source = source
    self._word_native_ctc_vocab_size = len(vocab)
    self._word_native_ctc_blank = num_classes - 1
    logger.info(f'word_ctc_bpe: reusing NeMo native {source} '
                f'for auxiliary word CTC, vocab={len(vocab)} blank={self._word_native_ctc_blank}')
    return True

  def _forward_native_word_ctc_logits(self, enc_out):
    source = getattr(self, '_word_native_ctc_decoder_source', None)
    if not source:
      raise ValueError('Native auxiliary word CTC decoder is not initialized')
    decoder = getattr(self.nemo_model, source, None)
    if decoder is None:
      raise ValueError(f'Native auxiliary word CTC decoder not found: {source}')
    enc_channels = enc_out.transpose(1, 2)
    return decoder(encoder_output=enc_channels)

  def _init_nemo_native_ctc(self):
    """Initialize NeMo native CTC mode (方案B).

    Keeps NeMo's pretrained CTC decoder (linear projection + SentencePiece vocab).
    Sets up CTC loss function using NeMo's tokenizer for label encoding.
    """
    nemo_dec, source, vocab = self._get_nemo_native_ctc_decoder()
    if nemo_dec is None:
      raise ValueError(
          'nemo_native_ctc requires a reusable NeMo CTC decoder. '
          'Supported backbones include parakeet-ctc-* and parakeet-tdt_ctc-*.')

    self.nemo_ctc_decoder = nemo_dec  # keep as submodule for training
    self._nemo_native_ctc_decoder_source = source
    self._nemo_ctc_vocab = vocab
    self._nemo_ctc_vocab_size = len(vocab) + 1  # +1 for blank (index 0 or len)

    # NeMo's CTC loss: blank=len(vocab) for NeMo convention
    # Detect blank index from decoder config
    num_classes = None
    if hasattr(nemo_dec, 'num_classes_with_blank'):
      num_classes = nemo_dec.num_classes_with_blank
    elif hasattr(nemo_dec, 'cfg'):
      try:
        num_classes = nemo_dec.cfg.num_classes
      except Exception:
        pass
    if num_classes is None:
      num_classes = len(vocab) + 1
    self._nemo_ctc_blank = num_classes - 1  # blank is last index

    self.nemo_ctc_loss_fn = nn.CTCLoss(blank=self._nemo_ctc_blank,
                                       reduction='none', zero_infinity=True)

    # NeMo tokenizer for encoding labels
    self._nemo_tokenizer = self.nemo_model.tokenizer

    # Mark as char-level for step-level validation text handling in base.forward()
    # (our _ctc_decode override returns text strings, same as char_level mode)
    self._ctc_char_level = True
    logger.info(f'NeMo native CTC initialized from {source}: '
                f'blank={self._nemo_ctc_blank} vocab={len(vocab)}')

    logger.info(f'NeMo native CTC (方案B): vocab_size={len(vocab)}, '
                f'blank={self._nemo_ctc_blank}, '
                f'tokenizer={type(self._nemo_tokenizer).__name__}')

  _nemo_saved_levels = {}

  def _suppress_nemo_verbose(self):
    """Temporarily raise NeMo loggers to ERROR to suppress verbose output.
    
    Suppresses both NeMo INFO (adapter/freeze messages) and WARNING
    (training/validation data config dumps) messages.
    """
    import logging as _logging
    self._nemo_saved_levels.clear()
    for name in ['nemo', 'nemo.collections', 'nemo.collections.asr',
                  'nemo.core', 'nemo.utils', 'nemo_logging']:
      lg = _logging.getLogger(name)
      self._nemo_saved_levels[name] = lg.level
      lg.setLevel(_logging.ERROR)
    try:
      from nemo.utils import logging as nemo_log
      self._nemo_saved_levels['__nemo_log__'] = nemo_log.getEffectiveLevel()
      nemo_log.setLevel(_logging.ERROR)
    except Exception:
      pass

  def _restore_nemo_verbose(self):
    """Restore NeMo logging levels to their original values."""
    import logging as _logging
    try:
      from nemo.utils import logging as nemo_log
      if '__nemo_log__' in self._nemo_saved_levels:
        nemo_log.setLevel(self._nemo_saved_levels.pop('__nemo_log__'))
    except Exception:
      pass
    for name, level in self._nemo_saved_levels.items():
      _logging.getLogger(name).setLevel(level)
    self._nemo_saved_levels.clear()

  def _enable_fast_decode(self):
    """Restore NeMo's default fast batched TDT decoding for inference.
    
    Uses greedy_batch strategy with GreedyBatchedTDTInfer + CUDA graphs.
    ~2-3x faster than the sequential greedy strategy used during training.
    Call this after loading weights for submission/inference.
    """
    model = self.nemo_model
    if hasattr(model, 'change_decoding_strategy'):
      try:
        from omegaconf import OmegaConf
        decoding_cfg = OmegaConf.create({
            'strategy': 'greedy_batch',
            'greedy': {
                'max_symbols': 10,
            }
        })
        self._suppress_nemo_verbose()
        model.change_decoding_strategy(decoding_cfg)
        self._restore_nemo_verbose()
        logger.info('Decoding strategy restored: greedy_batch (fast)')
      except Exception as e:
        logger.warning(f'Failed to restore fast decode: {e}')

  def _set_nemo_decode_strategy(self, strategy, beam_size=None,
                                return_best_hypothesis=True,
                                score_norm=True):
    model = self.nemo_model
    assert hasattr(model, 'change_decoding_strategy'), \
        'NeMo model does not expose change_decoding_strategy'

    from omegaconf import OmegaConf
    if strategy == 'beam':
      assert beam_size is not None and beam_size > 0, 'beam strategy requires beam_size > 0'
      decoding_cfg = OmegaConf.create({
          'strategy': 'beam',
          'beam': {
              'beam_size': int(beam_size),
              'return_best_hypothesis': bool(return_best_hypothesis),
              'score_norm': bool(score_norm),
          }
      })
    elif strategy == 'greedy':
      decoding_cfg = OmegaConf.create({
          'strategy': 'greedy',
          'greedy': {
              'max_symbols': 10,
              'loop_labels': False,
          }
      })
    elif strategy == 'greedy_batch':
      decoding_cfg = OmegaConf.create({
          'strategy': 'greedy_batch',
          'greedy': {
              'max_symbols': 10,
          }
      })
    else:
      raise ValueError(f'Unsupported NeMo decode strategy: {strategy}')

    self._suppress_nemo_verbose()
    model.change_decoding_strategy(decoding_cfg)
    self._restore_nemo_verbose()

  def _extract_nemo_hypothesis_entries(self, hypothesis, limit=None):
    if hasattr(hypothesis, 'n_best_hypotheses') and hypothesis.n_best_hypotheses is not None:
      hyps = list(hypothesis.n_best_hypotheses)
    else:
      hyps = [hypothesis]
    if limit is not None:
      hyps = hyps[:limit]
    entries = []
    for hyp in hyps:
      text = getattr(hyp, 'text', None)
      if text is None:
        text = str(hyp)
      score = getattr(hyp, 'score', np.nan)
      try:
        score = float(score)
      except Exception:
        score = np.nan
      entries.append({'text': text, 'score': score})
    return entries

  def _init_nemo_spec_augment(self):
    """Initialize an opt-in NeMo-specific SpecAugment module.

    This is intentionally separate from aug_spec (Whisper mel path) so
    existing experiments keep their current behavior unless
    --aug_nemo_spec is explicitly enabled.
    """
    self.nemo_spec_augmentation = None
    if not getattr(FLAGS, 'aug_nemo_spec', False):
      return
    try:
      from nemo.collections.asr.modules import SpectrogramAugmentation
      time_ratio = float(getattr(FLAGS, 'aug_nemo_spec_time_ratio', 0.0) or 0.0)
      assert 0.0 <= time_ratio <= 1.0, \
          f'aug_nemo_spec_time_ratio must be in [0, 1], got {time_ratio}'
      time_width = time_ratio if time_ratio > 0 else getattr(FLAGS, 'aug_time_mask', 20)
      self.nemo_spec_augmentation = SpectrogramAugmentation(
          freq_masks=getattr(FLAGS, 'aug_freq_num', 2),
          time_masks=getattr(FLAGS, 'aug_time_num', 2),
          freq_width=getattr(FLAGS, 'aug_freq_mask', 15),
          time_width=time_width,
      )
      logger.info('NeMo SpecAugment enabled: '
                  f'freq_masks={getattr(FLAGS, "aug_freq_num", 2)} '
                  f'freq_width={getattr(FLAGS, "aug_freq_mask", 15)} '
                  f'time_masks={getattr(FLAGS, "aug_time_num", 2)} '
                  f'time_width={time_width}')
    except Exception as e:
      logger.warning(f'Failed to initialize NeMo SpecAugment: {e}')

  def _enable_nemo_gradient_checkpointing(self):
    """Enable gradient checkpointing for NeMo ConformerEncoder.
    
    NeMo's ConformerEncoder doesn't have built-in gradient checkpointing,
    so we monkey-patch forward_internal to wrap each layer call with
    torch.utils.checkpoint.checkpoint. This saves ~2-3x activation memory
    at the cost of ~25% slower training.
    
    Important: BatchNorm1d layers inside ConformerConvolution are unaffected
    since checkpoint with use_reentrant=False preserves BN running stats.
    """
    import torch.utils.checkpoint as cp
    encoder = self.encoder
    if not hasattr(encoder, 'layers'):
      logger.warning('NeMo encoder has no .layers attribute, skipping gradient checkpointing')
      return
    
    orig_forward_internal = encoder.forward_internal
    
    def _gc_forward_internal(audio_signal, length, **kwargs):
      """Patched forward_internal with per-layer gradient checkpointing."""
      cache_last_channel = kwargs.get('cache_last_channel', None)
      # Only apply GC during training and when no streaming cache is used
      if not encoder.training or cache_last_channel is not None:
        return orig_forward_internal(audio_signal, length, **kwargs)
      
      # Replicate the pre-encode + pos_enc + mask creation from original
      import torch
      bypass_pre_encode = kwargs.get('bypass_pre_encode', False)
      
      if not bypass_pre_encode:
        audio_signal = torch.transpose(audio_signal, 1, 2)
        if isinstance(encoder.pre_encode, nn.Linear):
          audio_signal = encoder.pre_encode(audio_signal)
        else:
          audio_signal, length = encoder.pre_encode(x=audio_signal, lengths=length)
        length = length.to(torch.int64)
      
      max_audio_length = audio_signal.size(1)
      if hasattr(encoder, 'update_max_seq_length'):
        encoder.update_max_seq_length(seq_length=max_audio_length, device=audio_signal.device)
      audio_signal, pos_emb = encoder.pos_enc(x=audio_signal, cache_len=0)
      
      att_context_size = encoder.att_context_size
      pad_mask, att_mask = encoder._create_masks(
          att_context_size=att_context_size,
          padding_length=length,
          max_audio_length=max_audio_length,
          offset=None,
          device=audio_signal.device,
      )
      
      # Run layers with gradient checkpointing
      for lth, (drop_prob, layer) in enumerate(zip(encoder.layer_drop_probs, encoder.layers)):
        original_signal = audio_signal
        
        audio_signal = cp.checkpoint(
            layer, audio_signal, att_mask, pos_emb, pad_mask,
            None, None,  # cache_last_channel, cache_last_time
            use_reentrant=False,
        )
        
        # Stochastic depth (same logic as original)
        if encoder.training and drop_prob > 0.0:
          should_drop = torch.rand(1) < drop_prob
          if should_drop:
            audio_signal = audio_signal * 0.0 + original_signal
          else:
            audio_signal = (audio_signal - original_signal) / (1.0 - drop_prob) + original_signal
        
        # Reduction subsampling if at this layer position
        if hasattr(encoder, 'reduction_position') and encoder.reduction_position == lth:
          audio_signal, length = encoder.reduction_subsampling(x=audio_signal, lengths=length)
          max_audio_length = audio_signal.size(1)
      
      # out_proj if exists
      if hasattr(encoder, 'out_proj') and encoder.out_proj is not None:
        audio_signal = encoder.out_proj(audio_signal)
      
      # Final reduction if at end
      if hasattr(encoder, 'reduction_position') and encoder.reduction_position == -1:
        if hasattr(encoder, 'reduction_subsampling'):
          audio_signal, length = encoder.reduction_subsampling(x=audio_signal, lengths=length)
      
      # Transpose back to (B, D, T) for NeMo convention
      audio_signal = torch.transpose(audio_signal, 1, 2)
      length = length.to(torch.int64)
      return audio_signal, length
    
    encoder.forward_internal = _gc_forward_internal
    logger.info(f'NeMo gradient checkpointing enabled for {len(encoder.layers)} ConformerEncoder layers')

  def _disable_nemo_cuda_graphs(self):
    """Set decoding strategy to sequential greedy for training stability.
    
    Uses OmegaConf.create() to build a fresh config (avoids 'key not in struct'
    errors from modifying the model's frozen config).
    This is slower (~2-3x) but avoids CUDA graph issues during training.
    For inference, use _enable_fast_decode() or skip this call entirely.
    """
    model = self.nemo_model
    if hasattr(model, 'change_decoding_strategy'):
      try:
        from omegaconf import OmegaConf
        decoding_cfg = OmegaConf.create({
            'strategy': 'greedy',
            'greedy': {
                'max_symbols': 10,
                'loop_labels': False,
            }
        })
        # Suppress NeMo's verbose decoding strategy dump (~100 lines)
        self._suppress_nemo_verbose()
        model.change_decoding_strategy(decoding_cfg)
        self._restore_nemo_verbose()
        logger.info('Decoding strategy set: greedy, loop_labels=False, max_symbols=10')
        return
      except Exception as e:
        logger.warning(f'change_decoding_strategy failed: {e}, trying manual disable')
    # Also disable CUDA graphs explicitly (matches baseline)
    for attr_path in [
        'decoding.decoding',
        'decoding.decoding.decoding_computer',
    ]:
      obj = model
      try:
        for part in attr_path.split('.'):
          obj = getattr(obj, part)
        for flag in ('use_cuda_graphs', 'cuda_graphs_mode'):
          if hasattr(obj, flag):
            setattr(obj, flag, False if flag == 'use_cuda_graphs' else None)
            logger.info(f'Disabled {flag} on {attr_path}')
      except AttributeError:
        pass

  def _setup_nemo_lora(self):
    """Set up LoRA (PEFT) for NeMo encoder.

    Freezes the entire NeMo model, then inserts LoRA adapters into
    specified encoder Linear layers. Unlike nemo_adapter (which adds
    a separate residual module), LoRA modifies the weight matrices
    directly: W' = W + alpha/r * B @ A.

    Key advantage: merge_and_unload() produces a standard full-rank model
    that can be used as --backbone for downstream tasks.
    """
    try:
      from peft import LoraConfig, get_peft_model
    except ImportError:
      raise ImportError(
          'peft is required for NeMo LoRA. '
          'Install with: pip install peft')

    lora_r = getattr(FLAGS, 'lora_r', 16)
    lora_alpha = getattr(FLAGS, 'lora_alpha', None) or lora_r * 2
    lora_dropout = getattr(FLAGS, 'lora_dropout', 0.05)
    target_modules_str = getattr(FLAGS, 'nemo_lora_target_modules',
                                 'linear_q,linear_k,linear_v,linear_out')
    target_modules = [m.strip() for m in target_modules_str.split(',')]

    # Freeze the entire NeMo model first
    for param in self.nemo_model.parameters():
      param.requires_grad = False

    # Apply PEFT LoRA to the encoder (not the whole ASRModel,
    # which has complex NeMo-specific structure that confuses PEFT)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias='none',
    )
    self.nemo_model.encoder = get_peft_model(self.nemo_model.encoder, lora_config)
    # Update our reference
    self.encoder = self.nemo_model.encoder

    # Log param count
    trainable = sum(p.numel() for p in self.nemo_model.encoder.parameters()
                    if p.requires_grad)
    total_encoder = sum(p.numel() for p in self.nemo_model.encoder.parameters())
    total_model = sum(p.numel() for p in self.nemo_model.parameters())
    logger.info(f'NeMo LoRA enabled on encoder: '
                f'{trainable:,} trainable / {total_encoder:,} encoder / '
                f'{total_model:,} total ({trainable/total_model*100:.2f}%)')
    logger.info(f'  r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, '
                f'target_modules={target_modules}')

  def _setup_nemo_adapter(self):
    """Set up NeMo adapter training (matches official benchmark exactly).

    Inserts LinearAdapter modules into every encoder layer of the frozen model.
    Only adapter weights are trainable (~0.26% of total params for dim=32).

    Official flow (from orthographic.ipynb):
      1. Swap encoder target class to adapter-compatible variant
      2. Add global adapter config (check_encoder/decoder/joint_adapter)
      3. model.add_adapter(name, cfg=LinearAdapter(...))
      4. model.freeze() + model.train() + model.unfreeze_enabled_adapters()
      5. Disable spec_augment (freq_masks=0, time_masks=0)
    """
    from omegaconf import OmegaConf, DictConfig, open_dict
    from nemo.core import adapter_mixins

    model = self.nemo_model
    adapter_name = getattr(FLAGS, 'adapter_name', 'asr_children')
    adapter_dim = getattr(FLAGS, 'adapter_dim', 32)
    adapter_module_name = getattr(FLAGS, 'adapter_module_name', 'encoder')

    # ---- Step 1: Swap encoder target to adapter-compatible variant ----
    # Two things must happen:
    #   a) Update config _target_ (for serialization / future instantiation)
    #   b) Swap the live encoder instance's __class__ (so add_adapter() works at runtime)
    model_cfg = model.cfg
    with open_dict(model_cfg):
      adapter_metadata = adapter_mixins.get_registered_adapter(
          model_cfg.encoder._target_)
      if adapter_metadata is not None:
        old_target = model_cfg.encoder._target_
        model_cfg.encoder._target_ = adapter_metadata.adapter_class_path
        # Critical: swap the live encoder object's class so it gains AdapterModuleMixin
        model.encoder.__class__ = adapter_metadata.adapter_class
        logger.info(f'Adapter: swapped encoder target: '
                    f'{old_target} -> {adapter_metadata.adapter_class_path}')
        logger.info(f'Adapter: encoder.__class__ is now {type(model.encoder).__name__}')
      else:
        logger.warning(f'No registered adapter for {model_cfg.encoder._target_}. '
                       f'Adapter training may not work correctly.')

    # ---- Step 2: Build adapter type config (LinearAdapter) ----
    adapter_type_cfg = OmegaConf.create({
        '_target_': 'nemo.collections.common.parts.adapter_modules.LinearAdapter',
        'in_features': self.encoder_dim,  # 1024 for parakeet
        'dim': adapter_dim,
        'activation': 'swish',
        'norm_position': 'pre',
        'dropout': 0.0,
        'adapter_strategy': {
            '_target_': 'nemo.core.classes.mixins.adapter_mixin_strategies.ResidualAddAdapterStrategy',
            'stochastic_depth': 0.0,
            'l2_lambda': 0.0,
        }
    })

    # ---- Step 3: Add global adapter config ----
    global_adapter_cfg = OmegaConf.create({
        'check_encoder_adapter': True,
        'check_decoder_adapter': True,
        'check_joint_adapter': True,
    })
    with open_dict(model.cfg):
      if 'adapters' not in model.cfg:
        model.cfg.adapters = OmegaConf.create({})
      model.cfg.adapters[model.adapter_global_cfg_key] = global_adapter_cfg
      model.update_adapter_cfg(model.cfg.adapters)

    # ---- Step 4: Add adapter with module name prefix ----
    full_adapter_name = adapter_name
    if adapter_module_name is not None and ':' not in adapter_name:
      full_adapter_name = f'{adapter_module_name}:{adapter_name}'
    self._adapter_name = full_adapter_name

    model.add_adapter(full_adapter_name, cfg=adapter_type_cfg)
    assert model.is_adapter_available(), \
        f'Adapter {full_adapter_name} not available after add_adapter()'

    # Enable only our adapter
    model.set_enabled_adapters(enabled=False)
    model.set_enabled_adapters(full_adapter_name, enabled=True)

    # ---- Step 5: Freeze all, then unfreeze adapters ----
    model.freeze()
    model = model.train()
    model.unfreeze_enabled_adapters()
    self.nemo_model = model  # reassign in case .train() returns new ref

    # ---- Step 6: Disable spec augment (official uses freq_masks=0, time_masks=0) ----
    if hasattr(model, 'spec_augmentation') and model.spec_augmentation is not None:
      try:
        with open_dict(model.cfg.spec_augment):
          model.cfg.spec_augment.freq_masks = 0
          model.cfg.spec_augment.time_masks = 0
        # Also update the actual module
        if hasattr(model.spec_augmentation, 'freq_masks'):
          model.spec_augmentation.freq_masks = 0
        if hasattr(model.spec_augmentation, 'time_masks'):
          model.spec_augmentation.time_masks = 0
        logger.info('Adapter: disabled spec_augment (freq_masks=0, time_masks=0)')
      except Exception as e:
        logger.warning(f'Failed to disable spec_augment: {e}')

    # ---- Log adapter param count ----
    adapter_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'NeMo adapter "{full_adapter_name}" enabled: '
                f'{adapter_params:,} trainable / {total_params:,} total '
                f'({adapter_params/total_params*100:.2f}%)')

  def _init_rnnt_reuse(self, encoder_dim):
    """Scheme 1: Create a CustomRNNTDecoder with pretrained LSTM from NeMo.

    Extracts the LSTM weights from NeMo's RNNT/TDT decoder prediction network
    and initializes a CustomRNNTDecoder with matching dimensions.  The LSTM
    weights are copied (warm start); embedding and joint are randomly
    initialized with IPA vocabulary.

    Raises ValueError if backbone has no RNNT decoder (e.g. parakeet-ctc).
    """
    from src.models.base import CustomRNNTDecoder, IPA_CTC_VOCAB_SIZE
    import gezi as gz

    if not self._has_rnnt_loss:
      raise ValueError(
          'rnnt_reuse requires a NeMo model with RNNT/TDT decoder '
          '(e.g. parakeet-tdt-0.6b). Use parakeet-tdt-* backbone, or '
          'switch to --s2s_decoder=aed or rnnt_custom.')

    if getattr(FLAGS, 'word_weight', 0.0) > 0:
      raise ValueError(
          'rnnt_reuse is incompatible with --word_weight>0 because it '
          'replaces the RNNT vocab. Use --s2s_decoder=aed instead, or '
          'remove --word_weight / --use_cross_labels.')

    # ---- Find LSTM in NeMo's prediction network ----
    nemo_dec = self.nemo_model.decoder
    pred_hidden = None
    lstm_state_dict = {}
    num_lstm_layers = 1

    # NeMo RNNTDecoder stores prediction in self.prediction (ModuleDict or Module)
    for name, module in nemo_dec.named_modules():
      if isinstance(module, nn.LSTM):
        pred_hidden = module.hidden_size
        num_lstm_layers = module.num_layers
        lstm_state_dict = {k: v.clone() for k, v in module.state_dict().items()}
        logger.info(f'rnnt_reuse: found LSTM "{name}" — '
                     f'hidden={pred_hidden}, layers={num_lstm_layers}')
        break

    if pred_hidden is None:
      # Fallback: try LSTMCell
      for name, module in nemo_dec.named_modules():
        if isinstance(module, nn.LSTMCell):
          pred_hidden = module.hidden_size
          logger.info(f'rnnt_reuse: found LSTMCell "{name}" — hidden={pred_hidden}')
          break

    if pred_hidden is None:
      raise ValueError(
          'rnnt_reuse: could not locate LSTM/LSTMCell in NeMo decoder. '
          'Use --s2s_decoder=rnnt_custom instead.')

    # ---- Create CustomRNNTDecoder with pretrained LSTM ----
    vocab_size = IPA_CTC_VOCAB_SIZE
    self.rnnt_decoder = CustomRNNTDecoder(
        encoder_dim=encoder_dim, vocab_size=vocab_size,
        pred_dim=pred_hidden, pred_layers=num_lstm_layers,
        joint_dim=pred_hidden)

    # Copy LSTM weights (strict=False: allow shape differences in proj layers)
    if lstm_state_dict:
      missing, unexpected = self.rnnt_decoder.pred_rnn.load_state_dict(
          lstm_state_dict, strict=False)
      if missing:
        logger.warning(f'rnnt_reuse: LSTM missing keys: {missing}')
      if unexpected:
        logger.warning(f'rnnt_reuse: LSTM unexpected keys: {unexpected}')
      logger.info(f'rnnt_reuse: copied {len(lstm_state_dict)} LSTM weight tensors')

    gz.set('s2s_ipa_chars', True)
    logger.info(f'rnnt_reuse (Scheme 1): CustomRNNTDecoder with pretrained LSTM, '
                 f'pred_dim={pred_hidden}, joint_dim={pred_hidden}, '
                 f'vocab={vocab_size}')

  def _init_tdt_scratch(self, encoder_dim):
    """Fallback for tdt_reuse on pure CTC backbones.

    Builds a CustomTDTDecoder from scratch with dimensions that mimic the
    Parakeet TDT head, while keeping the IPA character vocabulary used by the
    project. This preserves the user-facing semantics of --tdt_only: the main
    branch remains TDT even when the backbone cannot provide TDT weights to
    reuse.
    """
    from src.models.base import CustomTDTDecoder, IPA_CTC_VOCAB_SIZE
    import gezi as gz

    durations = [
        int(d) for d in getattr(FLAGS, 'tdt_durations', ['0', '1', '2', '3', '4'])
    ]
    sigma = getattr(FLAGS, 'tdt_sigma', 0.02)
    omega = getattr(FLAGS, 'tdt_omega', 0.1)

    # Mimic Parakeet TDT defaults so pure CTC backbones can still train a
    # compatible TDT-style main head when reuse is unavailable.
    pred_hidden = 640
    num_lstm_layers = 2
    joint_dim = 640
    joint_dropout = 0.2
    vocab_size = IPA_CTC_VOCAB_SIZE

    self.tdt_decoder = CustomTDTDecoder(
        encoder_dim=encoder_dim,
        vocab_size=vocab_size,
        pred_dim=pred_hidden,
        pred_layers=num_lstm_layers,
        joint_dim=joint_dim,
        durations=durations,
        sigma=sigma,
        omega=omega,
        dropout=joint_dropout,
    )

    gz.set('s2s_ipa_chars', True)
    logger.info(
        'tdt_reuse fallback: initialized CustomTDTDecoder from scratch '
        f'(mimic parakeet-tdt head), pred_dim={pred_hidden}, '
        f'joint_dim={joint_dim}, vocab={vocab_size}, durations={durations}, '
        f'sigma={sigma}, omega={omega}, dropout={joint_dropout}')

  def _init_tdt_reuse(self, encoder_dim):
    """Scheme 4: Create a CustomTDTDecoder with pretrained LSTM from NeMo TDT.

    Exact replica of the NeMo TDT architecture (prediction + joint + TDT loss
    with duration prediction), only the vocabulary changes from 1025 BPE to
    53 IPA.  LSTM weights are copied warm-start; embedding, joint output,
    and duration head are randomly initialised.

    Raises ValueError if backbone has no RNNT/TDT decoder.
    """
    from src.models.base import CustomTDTDecoder, IPA_CTC_VOCAB_SIZE
    import gezi as gz

    if not self._has_rnnt_loss:
      raise ValueError(
          'tdt_reuse requires a NeMo model with TDT decoder '
          '(e.g. parakeet-tdt-0.6b). Use parakeet-tdt-* backbone, or '
          'switch to --s2s_decoder=rnnt_custom.')

    # ---- Find LSTM in NeMo's prediction network ----
    nemo_dec = self.nemo_model.decoder
    pred_hidden = None
    lstm_state_dict = {}
    num_lstm_layers = 1
    lstm_dropout = 0.0

    for name, module in nemo_dec.named_modules():
      if isinstance(module, nn.LSTM):
        pred_hidden = module.hidden_size
        num_lstm_layers = module.num_layers
        lstm_dropout = module.dropout
        lstm_state_dict = {k: v.clone() for k, v in module.state_dict().items()}
        logger.info(f'tdt_reuse: found LSTM "{name}" \u2014 '
                     f'hidden={pred_hidden}, layers={num_lstm_layers}')
        break

    if pred_hidden is None:
      raise ValueError(
          'tdt_reuse: could not locate LSTM in NeMo decoder. '
          'Use --s2s_decoder=rnnt_custom instead.')

    # ---- Find joint dim from NeMo's joint network ----
    nemo_joint = self.nemo_model.joint
    joint_dim = pred_hidden  # fallback
    # Try to read from enc_proj
    if hasattr(nemo_joint, 'enc') and hasattr(nemo_joint.enc, 'out_features'):
      joint_dim = nemo_joint.enc.out_features
    elif hasattr(nemo_joint, 'pred') and hasattr(nemo_joint.pred, 'out_features'):
      joint_dim = nemo_joint.pred.out_features

    # ---- Find dropout from NeMo's joint_net ----
    joint_dropout = 0.2  # NeMo default
    if hasattr(nemo_joint, 'joint_net'):
      for m in nemo_joint.joint_net.modules():
        if isinstance(m, nn.Dropout):
          joint_dropout = m.p
          break

    # ---- TDT params ----
    durations = [int(d) for d in getattr(FLAGS, 'tdt_durations', ['0','1','2','3','4'])]
    sigma = getattr(FLAGS, 'tdt_sigma', 0.02)
    omega = getattr(FLAGS, 'tdt_omega', 0.1)

    # ---- Create CustomTDTDecoder with pretrained LSTM ----
    vocab_size = IPA_CTC_VOCAB_SIZE
    self.tdt_decoder = CustomTDTDecoder(
        encoder_dim=encoder_dim, vocab_size=vocab_size,
        pred_dim=pred_hidden, pred_layers=num_lstm_layers,
        joint_dim=joint_dim, durations=durations,
        sigma=sigma, omega=omega, dropout=joint_dropout)

    # Copy LSTM weights
    if lstm_state_dict:
      missing, unexpected = self.tdt_decoder.pred_rnn.load_state_dict(
          lstm_state_dict, strict=False)
      if missing:
        logger.warning(f'tdt_reuse: LSTM missing keys: {missing}')
      if unexpected:
        logger.warning(f'tdt_reuse: LSTM unexpected keys: {unexpected}')
      logger.info(f'tdt_reuse: copied {len(lstm_state_dict)} LSTM weight tensors')

    # Copy joint projection weights where dimensions match
    # enc_proj: NeMo (1024→640) → ours (1024→640) — same dims
    if hasattr(nemo_joint, 'enc'):
      src_enc = nemo_joint.enc
      dst_enc = self.tdt_decoder.enc_proj
      if src_enc.in_features == dst_enc.in_features and src_enc.out_features == dst_enc.out_features:
        dst_enc.load_state_dict(src_enc.state_dict())
        logger.info('tdt_reuse: copied enc_proj weights from NeMo joint')
    # pred_proj: NeMo (640→640) → ours (640→640) — same dims
    if hasattr(nemo_joint, 'pred'):
      src_pred = nemo_joint.pred
      dst_pred = self.tdt_decoder.pred_proj
      if src_pred.in_features == dst_pred.in_features and src_pred.out_features == dst_pred.out_features:
        dst_pred.load_state_dict(src_pred.state_dict())
        logger.info('tdt_reuse: copied pred_proj weights from NeMo joint')

    gz.set('s2s_ipa_chars', True)
    logger.info(f'tdt_reuse (Scheme 4): CustomTDTDecoder with pretrained LSTM, '
                 f'pred_dim={pred_hidden}, joint_dim={joint_dim}, '
                 f'vocab={vocab_size}, durations={durations}, '
                 f'sigma={sigma}, omega={omega}')

    if getattr(FLAGS, 'word_tdt_pseudo_ipa', False):
      if self._half_share_word_tdt_decoder():
        self.word_tdt_decoder = CustomTDTDecoder(
            encoder_dim=encoder_dim,
            vocab_size=vocab_size,
            pred_dim=pred_hidden,
            pred_layers=num_lstm_layers,
            joint_dim=joint_dim,
            durations=durations,
            sigma=sigma,
            omega=omega,
            dropout=joint_dropout,
            blank_id=self.tdt_decoder.blank_id,
            padding_idx=self.tdt_decoder.pred_embedding.padding_idx,
            pred_embedding=nn.Embedding(
                vocab_size, pred_hidden,
                padding_idx=self.tdt_decoder.pred_embedding.padding_idx),
            pred_rnn=self.tdt_decoder.pred_rnn,
            enc_proj=self.tdt_decoder.enc_proj,
            pred_proj=self.tdt_decoder.pred_proj,
        )
        logger.info('word_tdt_pseudo_ipa: created half-shared word_tdt_decoder '
                    '(shared=pred_rnn+enc_proj+pred_proj, separate=pred_embedding+joint_out)')
      elif self._share_word_tdt_decoder():
        logger.info('word_tdt_pseudo_ipa: sharing main tdt_decoder with word branch')
      else:
        self.word_tdt_decoder = copy.deepcopy(self.tdt_decoder)
        logger.info('word_tdt_pseudo_ipa: created separate word_tdt_decoder initialized from main tdt_decoder')

    if getattr(FLAGS, 'word_tdt_mixed', False):
      assert self._nemo_tokenizer is not None, \
        'word_tdt_mixed requires a NeMo tokenizer'
      bpe_vocab_size = getattr(self._nemo_tokenizer, 'vocab_size', None)
      if bpe_vocab_size is None:
        tokenizer_impl = getattr(self._nemo_tokenizer, 'tokenizer', None)
        if tokenizer_impl is not None and hasattr(tokenizer_impl, 'get_vocab_size'):
          bpe_vocab_size = int(tokenizer_impl.get_vocab_size())
      if bpe_vocab_size is None:
        raise ValueError('word_tdt_mixed: failed to infer NeMo tokenizer vocab size')
      bpe_vocab_size = int(bpe_vocab_size)
      self.word_tdt_decoder = CustomTDTDecoder(
          encoder_dim=encoder_dim,
          vocab_size=bpe_vocab_size + 1,
          pred_dim=pred_hidden,
          pred_layers=num_lstm_layers,
          joint_dim=joint_dim,
          durations=durations,
          sigma=sigma,
          omega=omega,
          dropout=joint_dropout,
          blank_id=0,
          padding_idx=None,
          pred_embedding=nn.Embedding(bpe_vocab_size + 1, pred_hidden),
          pred_rnn=self.tdt_decoder.pred_rnn,
          enc_proj=self.tdt_decoder.enc_proj,
          pred_proj=self.tdt_decoder.pred_proj,
      )
      logger.info(
          'word_tdt_mixed: created half-shared BPE word_tdt_decoder '
          f'(bpe_vocab={bpe_vocab_size}, shared=pred_rnn+enc_proj+pred_proj)')

  def _encode(self, input_features, attention_mask=None):
    """Encode audio through NeMo preprocessor + encoder.

    NeMo models expect raw waveform input.
    input_features: (B, T) raw waveform or (B, C, T) preprocessed mel
    attention_mask:  (B, T) 1=real, 0=padding (for raw waveform)

    Returns: (B, T', encoder_dim)
    """
    # Compute lengths from attention mask
    if attention_mask is not None:
      lengths = attention_mask.sum(dim=-1).long()
    else:
      B = input_features.shape[0]
      T = input_features.shape[-1]
      lengths = torch.full((B,), T, dtype=torch.long, device=input_features.device)

    # NeMo preprocessor: raw waveform -> mel features
    # input_features should be (B, T) raw audio
    if input_features.dim() == 2:
      processed, proc_len = self.preprocessor(
          input_signal=input_features, length=lengths)
    else:
      # Already preprocessed (B, C, T) mel — pass through
      processed = input_features
      proc_len = lengths

    # Opt-in SpecAugment on NeMo mel features.
    if (self.training and getattr(FLAGS, 'aug', False)
        and getattr(self, 'nemo_spec_augmentation', None) is not None):
      _debug_key = 'aug_once:nemo_spec'
      _capture = _should_capture_nemo_aug_once(_debug_key)
      _before = processed if _capture else None
      processed = self.nemo_spec_augmentation(
          input_spec=processed, length=proc_len)
      if _capture:
        _log_nemo_aug_once('nemo_spec', {
            'freq_masks': getattr(FLAGS, 'aug_freq_num', 2),
            'freq_width': getattr(FLAGS, 'aug_freq_mask', 15),
            'time_masks': getattr(FLAGS, 'aug_time_num', 2),
            'time_width': (getattr(FLAGS, 'aug_nemo_spec_time_ratio', 0.0)
                           if getattr(FLAGS, 'aug_nemo_spec_time_ratio', 0.0)
                           else getattr(FLAGS, 'aug_time_mask', 20)),
            'before': _tensor_summary(_before),
            'after': _tensor_summary(processed),
            'proc_len_min': int(proc_len.min().item()) if len(proc_len) else 0,
            'proc_len_max': int(proc_len.max().item()) if len(proc_len) else 0,
        })

    # NeMo encoder: mel -> encoder output
    # For InterCTC we need intermediate hidden states from Conformer layers
    if self._inter_ctc_enabled or self._ctc_layer_fusion_enabled:
      enc_out, enc_len = self._encode_with_inter_states(
          processed, proc_len)
      # _encode_with_inter_states returns (B, D, T) like standard NeMo encoder
      enc_out = enc_out.transpose(1, 2)
    else:
      enc_out, enc_len = self.encoder(audio_signal=processed, length=proc_len)
      # NeMo ConformerEncoder returns (B, D, T) — transpose to (B, T, D)
      enc_out = enc_out.transpose(1, 2)

    # Cache encoder lengths for _s2s_forward (RNNT loss needs them)
    self._last_enc_len = enc_len

    return enc_out

  def _encode_with_inter_states(self, processed, proc_len):
    """Encode with NeMo Conformer, capturing intermediate hidden states.
    
    Mirrors ConformerEncoder.forward_internal() flow exactly:
      1. transpose (B,C,T) -> (B,T,C)
      2. pre_encode (ConvSubsampling): (x=, lengths=)
      3. pos_enc: returns (audio_signal, pos_emb)
      4. _create_masks -> pad_mask, att_mask
      5. layer(x=, att_mask=, pos_emb=, pad_mask=)
      6. out_proj (if any)
      7. transpose (B,T,D) -> (B,D,T)
    We capture intermediate states for InterCTC at specified layers.
    """
    import torch
    encoder = self.encoder
    layers = encoder.layers
    inter_hidden = []
    fusion_hidden = []
    fusion_layers = set(getattr(self, '_ctc_fusion_layers', []))

    # ---- Step 1: transpose (B, C, T) -> (B, T, C) for pre_encode ----
    audio_signal = torch.transpose(processed, 1, 2)

    # ---- Step 2: pre_encode (ConvSubsampling) ----
    if isinstance(encoder.pre_encode, nn.Linear):
      audio_signal = encoder.pre_encode(audio_signal)
      enc_len = proc_len
    else:
      audio_signal, enc_len = encoder.pre_encode(x=audio_signal, lengths=proc_len)
    enc_len = enc_len.to(torch.int64)

    # ---- Step 3: positional encoding ----
    max_audio_length = audio_signal.size(1)
    if hasattr(encoder, 'update_max_seq_length'):
      encoder.update_max_seq_length(seq_length=max_audio_length, device=audio_signal.device)
    audio_signal, pos_emb = encoder.pos_enc(x=audio_signal, cache_len=0)

    # ---- Step 4: create masks ----
    # Use att_context_size (first mode for non-training, or random for training)
    att_context_size = encoder.att_context_size
    pad_mask, att_mask = encoder._create_masks(
        att_context_size=att_context_size,
        padding_length=enc_len,
        max_audio_length=max_audio_length,
        offset=None,
        device=audio_signal.device,
    )

    # ---- Step 5: run through Conformer layers one by one ----
    for lth, layer in enumerate(layers):
      if FLAGS.gradient_checkpointing and self.training:
        audio_signal = torch.utils.checkpoint.checkpoint(
            layer, audio_signal, att_mask, pos_emb, pad_mask,
            None, None,  # cache_last_channel, cache_last_time
            use_reentrant=False,
        )
      else:
        audio_signal = layer(
            x=audio_signal,
            att_mask=att_mask,
            pos_emb=pos_emb,
            pad_mask=pad_mask,
        )
      # Capture intermediate hidden state for InterCTC
      if lth in self._inter_ctc_layers:
        # audio_signal is (B, T, D) at this point
        inter_hidden.append(audio_signal)
      # Capture for CTC layer fusion
      if lth in fusion_layers:
        fusion_hidden.append(audio_signal)

    # ---- Step 6: out_proj (if any, e.g. dimension reduction) ----
    if hasattr(encoder, 'out_proj') and encoder.out_proj is not None:
      audio_signal = encoder.out_proj(audio_signal)

    # ---- Step 7: reduction (if at end) ----
    if hasattr(encoder, 'reduction_position') and encoder.reduction_position == -1:
      if hasattr(encoder, 'reduction_subsampling'):
        audio_signal, enc_len = encoder.reduction_subsampling(x=audio_signal, lengths=enc_len)

    self._inter_hidden_states = inter_hidden
    self._ctc_fusion_hidden_states = fusion_hidden

    # audio_signal is (B, T, D) — transpose to (B, D, T) to match NeMo encoder output format
    # Then _encode() will transpose back to (B, T, D) for downstream
    enc_out = torch.transpose(audio_signal, 1, 2)
    enc_len = enc_len.to(torch.int64)
    return enc_out, enc_len

  # ---- NeMo native CTC overrides (方案B) ----

  def _compute_nemo_native_ctc_loss(self, enc_out, labels):
    """CTC loss using NeMo's pretrained CTC decoder + SentencePiece tokenizer.

    enc_out: (B, T, D) encoder output
    labels:  (B, L) token IDs (may be all -100 when label_texts are used)
    Returns: (ctc_loss, ctc_logits, target_lengths)
      ctc_loss: (B,) per-sample CTC loss
      ctc_logits: (B, T, V) log_probs for decoding
      target_lengths: (B,) for downstream use
    """
    import torch.nn.functional as F

    # NeMo CTC decoder expects (B, D, T) channels-first
    enc_channels = enc_out.transpose(1, 2)  # (B, D, T)
    logits = self.nemo_ctc_decoder(encoder_output=enc_channels)  # (B, T', V)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    B, T, V = log_probs.shape
    log_probs_t = log_probs.transpose(0, 1)  # (T, B, V) for CTCLoss

    # Encoder lengths
    enc_len = self._last_enc_len
    if enc_len is not None:
      input_lengths = enc_len.to(log_probs.device).clamp(max=T)
    else:
      input_lengths = torch.full((B,), T, dtype=torch.long, device=log_probs.device)

    # Get label texts: prefer raw strings (no Whisper BPE roundtrip)
    label_texts = getattr(self, '_current_label_texts', None)
    if label_texts and any(label_texts):
      texts = label_texts
    elif self.tokenizer is not None:
      # Legacy fallback: decode Whisper BPE labels → text
      clean = labels.clone()
      clean[clean == -100] = self.tokenizer.pad_token_id or 0
      texts = self.tokenizer.batch_decode(clean.long().cpu().numpy(),
                                          skip_special_tokens=True)
    else:
      # No label_texts and no tokenizer — can't compute loss
      zero_loss = torch.zeros(B, device=log_probs.device, requires_grad=True)
      return zero_loss, log_probs, torch.zeros(B, dtype=torch.long, device=log_probs.device)

    # Encode text → NeMo SentencePiece token IDs directly
    nemo_seqs = []
    for text in texts:
      ids = self._nemo_tokenizer.text_to_ids(text)
      if not ids:
        ids = [0]  # avoid empty target
      nemo_seqs.append(ids)

    target_lengths = torch.tensor([len(s) for s in nemo_seqs],
                                   dtype=torch.long, device=log_probs.device)
    if target_lengths.sum() == 0:
      zero_loss = torch.zeros(B, device=log_probs.device, requires_grad=True)
      return zero_loss, log_probs, target_lengths

    targets_flat = torch.tensor([c for s in nemo_seqs for c in s],
                                 dtype=torch.long, device=log_probs.device)

    ctc_loss = self.nemo_ctc_loss_fn(log_probs_t, targets_flat,
                                     input_lengths, target_lengths)
    # Normalize per sample by target length
    ctc_loss = ctc_loss / target_lengths.clamp(min=1).float()
    # Compute per-sample entropy for entropy regularization (log_probs is already log_softmax'd)
    self._ctc_entropy = self._compute_ctc_entropy(log_probs)
    return ctc_loss, log_probs, target_lengths

  def _nemo_native_ctc_decode(self, log_probs, lengths=None):
    """Greedy CTC decode using NeMo vocabulary.

    log_probs: (B, T, V) log probabilities
    Returns: list of decoded text strings
    """
    preds = log_probs.argmax(dim=-1)  # (B, T)
    B = preds.shape[0]
    vocab = self._nemo_ctc_vocab
    blank = self._nemo_ctc_blank
    texts = []
    for i in range(B):
      T = lengths[i].item() if lengths is not None else preds.shape[1]
      seq = preds[i, :T].cpu().tolist()
      # CTC collapse: remove consecutive duplicates and blanks
      collapsed = []
      prev = -1
      for s in seq:
        if s != prev:
          if s != blank:
            collapsed.append(s)
        prev = s
      # Use NeMo tokenizer ids_to_text for proper SentencePiece detokenization
      text = self._nemo_tokenizer.ids_to_text(collapsed)
      texts.append(text)
    return texts

  def _compute_ctc_loss(self, enc_out, labels):
    """Override: route to NeMo native CTC or base class."""
    if self._nemo_native_ctc:
      return self._compute_nemo_native_ctc_loss(enc_out, labels)
    return super()._compute_ctc_loss(enc_out, labels)

  def _generate_ctc(self, enc_out, ctc_logits, device):
    """Override: NeMo native CTC decode → text → token IDs.
    Uses NeMo SP tokenizer when Whisper tokenizer is None.
    """
    if self._nemo_native_ctc:
      import torch.nn.functional as F
      enc_channels = enc_out.transpose(1, 2)
      logits = self.nemo_ctc_decoder(encoder_output=enc_channels)
      log_probs = F.log_softmax(logits.float(), dim=-1)
      lengths = self._last_enc_len
      texts = self._nemo_native_ctc_decode(log_probs, lengths)

      # Store decoded texts directly (avoids tokenizer roundtrip in eval)
      self._last_pred_texts = texts

      # Encode texts to token IDs for eval pipeline
      tokenizer = self._nemo_tokenizer if self.tokenizer is None else self.tokenizer
      B = enc_out.shape[0]
      max_len = FLAGS.max_new_tokens
      pad_id = getattr(tokenizer, 'pad_token_id', None) or 0
      generated = torch.full((B, max_len), pad_id, dtype=torch.long,
                             device=device)
      for i, text in enumerate(texts):
        ids = tokenizer.text_to_ids(text) if hasattr(tokenizer, 'text_to_ids') else tokenizer.encode(text)
        seq_len = min(len(ids), max_len)
        if seq_len > 0:
          generated[i, :seq_len] = torch.tensor(ids[:seq_len], dtype=torch.long)
      return generated
    return super()._generate_ctc(enc_out, ctc_logits, device)

  def _ctc_decode(self, ctc_logits):
    """Override: NeMo native CTC decode returns text strings (char_level mode)."""
    if self._nemo_native_ctc:
      import torch.nn.functional as F
      log_probs = F.log_softmax(ctc_logits.float(), dim=-1)
      lengths = self._last_enc_len
      return self._nemo_native_ctc_decode(log_probs, lengths)
    return super()._ctc_decode(ctc_logits)

  def _s2s_forward(self, input_features, labels, encoder_hidden_states,
                   attention_mask=None):
    """Compute NeMo TDT/RNNT loss for fine-tuning.

    Uses label_texts (raw strings) directly when available, falling back to
    Whisper BPE decode for backwards compatibility.  Encodes to NeMo
    SentencePiece tokens, then uses NeMo's decoder + joint + TDT/RNNT loss.

    encoder_hidden_states: (B, T, D) from _encode() — already transposed.
    """
    if not self._has_rnnt_loss:
      # Pure CTC NeMo model (e.g. parakeet-ctc) has no RNNT decoder;
      # user should use --ctc_weight=1.0 instead.
      return torch.tensor(0.0, device=input_features.device, requires_grad=True)

    B = encoder_hidden_states.shape[0]

    # ---- Get label texts: prefer raw strings (no Whisper BPE roundtrip) ----
    label_texts = getattr(self, '_current_label_texts', None)
    if label_texts and any(label_texts):
      texts = label_texts
    elif self.tokenizer is not None:
      # Legacy fallback: decode Whisper BPE labels → text
      clean = labels.clone()
      clean[clean == -100] = self.tokenizer.pad_token_id or 0
      texts = self.tokenizer.batch_decode(clean.long().cpu().numpy(),
                                          skip_special_tokens=True)
    else:
      return torch.tensor(0.0, device=input_features.device, requires_grad=True)

    # ---- Encode text → NeMo SentencePiece tokens directly ----
    nemo_toks = []
    nemo_lens = []
    for text in texts:
      ids = self._nemo_tokenizer.text_to_ids(text)
      if not ids:
        ids = [0]  # avoid empty target
      nemo_toks.append(ids)
      nemo_lens.append(len(ids))

    max_tlen = max(nemo_lens)
    transcript = torch.zeros(B, max_tlen, dtype=torch.long,
                             device=encoder_hidden_states.device)
    for i, ids in enumerate(nemo_toks):
      transcript[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    transcript_length = torch.tensor(nemo_lens, dtype=torch.long,
                                     device=encoder_hidden_states.device)

    # ---- NeMo decoder + joint (fused loss) ----
    enc_len = self._last_enc_len  # set by _encode()
    if enc_len is None:
      # Safety fallback: assume all frames are valid
      enc_len = torch.full((B,), encoder_hidden_states.shape[1],
                           dtype=torch.long,
                           device=encoder_hidden_states.device)

    # encoder_hidden_states is (B, T, D) from _encode(); NeMo joint/decoder
    # expect channels-first (B, D, T), so transpose back.
    enc_for_joint = encoder_hidden_states.transpose(1, 2)  # (B, D, T)

    # RNNT decoder returns 3 values: (output, output_length, states)
    dec_out, dec_len, _ = self.nemo_model.decoder(
        targets=transcript, target_length=transcript_length)
    # dec_out: (B, D_dec, U+1) — channels-first, used as-is by joint

    # Compute per-sample RNNT/TDT loss by bypassing NeMo's fused reduction.
    # NeMo's joint.forward with fuse_loss_wer=True returns batch-averaged scalar,
    # which breaks per-sample weighting & masking in multi-task training.
    # Instead, we call the joint and loss directly with reduction=None.
    enc_t = enc_for_joint.transpose(1, 2)  # (B, T, D)
    dec_t = dec_out.transpose(1, 2)        # (B, U+1, D_dec)
    
    # Trim to actual lengths for efficiency
    max_enc_len = enc_len.max()
    max_dec_len = transcript_length.max() + 1
    if enc_t.shape[1] > max_enc_len:
      enc_t = enc_t[:, :max_enc_len, :]
    if dec_t.shape[1] > max_dec_len:
      dec_t = dec_t[:, :max_dec_len, :]
    if transcript.shape[1] > transcript_length.max():
      transcript = transcript[:, :transcript_length.max()]
    
    # Compute joint logits: (B, T, U+1, V+1)
    joint_logits = self.nemo_model.joint.joint(enc_t, dec_t)
    
    # Get the loss module and compute per-sample loss
    loss_module = self.nemo_model.joint._loss
    saved_reduction = loss_module.reduction
    loss_module.reduction = None  # per-sample
    
    per_sample_loss = loss_module(
        log_probs=joint_logits,
        targets=transcript.long(),
        input_lengths=enc_len.long(),
        target_lengths=transcript_length.long(),
    )  # (B,)
    
    loss_module.reduction = saved_reduction
    
    # Store target lengths for mean_volume reduction in get_loss_fn
    self._last_s2s_target_lengths = transcript_length
    
    return per_sample_loss  # (B,)

  def _s2s_generate(self, input_features, encoder_hidden_states,
                    attention_mask=None):
    """Generate using NeMo's native decoding on tensor input.

    Only used when ctc_weight=0 (inference-only mode).
    Reuses pre-computed encoder output from _encode() when available,
    avoiding expensive double encoding (preprocessor + encoder).
    """
    self.nemo_model.eval()
    with torch.no_grad():
      B = input_features.shape[0]
      if encoder_hidden_states is not None and self._last_enc_len is not None:
        # Reuse pre-computed encoder output from _encode()
        # _encode() returns (B, T, D); NeMo decode expects (B, D, T)
        enc_out = encoder_hidden_states.transpose(1, 2)
        enc_len = self._last_enc_len
      else:
        # Fallback: compute from scratch (standalone call without prior _encode)
        if input_features.dim() == 2:
          B, T = input_features.shape
        else:
          B = input_features.shape[0]
          T = input_features.shape[-1]

        attention_mask_in = input_features.new_ones(B, T) if attention_mask is None else attention_mask
        lengths = attention_mask_in.sum(dim=-1).long()

        if input_features.dim() == 2:
          processed, proc_len = self.nemo_model.preprocessor(
              input_signal=input_features, length=lengths)
        else:
          processed = input_features
          proc_len = lengths

        enc_out, enc_len = self.nemo_model.encoder(
            audio_signal=processed, length=proc_len)

      # Decode: use _nemo_decode which handles CTC/TDT/RNNT dispatch
      texts = self._nemo_decode(enc_out, enc_len)

    # IPA conversion is now handled in eval.py for all models

    # Store decoded texts so forward() can pass them directly (avoids tokenizer roundtrip)
    self._last_pred_texts = texts

    # Encode texts back to BPE token IDs for eval pipeline
    # Use NeMo's own tokenizer (self.tokenizer may be None for NeMo backbones)
    tokenizer = self._nemo_tokenizer if self.tokenizer is None else self.tokenizer
    max_len = FLAGS.max_new_tokens
    pad_id = getattr(tokenizer, 'pad_token_id', None) or 0
    generated = torch.full((B, max_len), pad_id, dtype=torch.long,
                           device=input_features.device)
    for i, text in enumerate(texts):
      ids = tokenizer.text_to_ids(text) if hasattr(tokenizer, 'text_to_ids') else tokenizer.encode(text)
      seq_len = min(len(ids), max_len)
      if seq_len > 0:
        generated[i, :seq_len] = torch.tensor(ids[:seq_len], dtype=torch.long)

    return generated

  def _nemo_decode(self, enc_out, enc_len):
    """Decode encoder output to text using NeMo's native decoder.
    
    Handles different NeMo model types: CTC, TDT/RNNT, AED.
    Returns list of text strings.
    """
    model = self.nemo_model
    B = enc_out.shape[0]

    # --- CTC models (e.g., parakeet-ctc) ---
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'vocabulary'):
      # CTC: run decoder to get log_probs, then greedy decode
      log_probs = model.decoder(encoder_output=enc_out)
      # Greedy CTC decode
      preds = log_probs.argmax(dim=-1)  # (B, T)
      vocab = model.decoder.vocabulary
      texts = []
      for i in range(B):
        seq = preds[i, :enc_len[i]].cpu().tolist()
        # CTC collapse: remove consecutive duplicates and blanks
        collapsed = []
        prev = -1
        for s in seq:
          if s != prev and s < len(vocab):
            collapsed.append(s)
          prev = s
        # 0 is typically blank
        text = ''.join(vocab[c] for c in collapsed if c != 0 and c < len(vocab))
        texts.append(text)
      return texts

    # --- TDT / RNNT models (e.g., parakeet-tdt) ---
    if hasattr(model, 'decoding'):
      want_score = bool(getattr(FLAGS, 'save_pred_score', False))
      want_nbest = int(getattr(FLAGS, 'save_pred_nbest', 0) or 0)
      hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
          encoder_output=enc_out,
          encoded_lengths=enc_len,
          return_hypotheses=(want_score or want_nbest > 0),
      )
      if isinstance(hypotheses, tuple):
        hypotheses = hypotheses[0]

      texts = []
      if want_score or want_nbest > 0:
        greedy_entries = [self._extract_nemo_hypothesis_entries(hyp, limit=1) for hyp in hypotheses]
        beam_entries = None
        if want_nbest > 0:
          beam_size = max(int(getattr(FLAGS, 'num_beams', 1) or 1), want_nbest)
          self._set_nemo_decode_strategy(
              'beam',
              beam_size=beam_size,
              return_best_hypothesis=False,
              score_norm=True,
          )
          beam_hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
              encoder_output=enc_out,
              encoded_lengths=enc_len,
              return_hypotheses=True,
          )
          if isinstance(beam_hypotheses, tuple):
            beam_hypotheses = beam_hypotheses[0]
          beam_entries = [self._extract_nemo_hypothesis_entries(hyp, limit=want_nbest)
                          for hyp in beam_hypotheses]
          self._set_nemo_decode_strategy('greedy')

        rows = []
        for i, entries in enumerate(greedy_entries):
          primary = entries[0] if entries else {'text': '', 'score': np.nan}
          texts.append(primary['text'])
          row = {}
          if want_score:
            row['pred_score'] = primary['score']
          if want_nbest > 0 and beam_entries is not None:
            row['pred_nbest_texts'] = [item['text'] for item in beam_entries[i]]
            row['pred_nbest_scores'] = [item['score'] for item in beam_entries[i]]
          rows.append(row)
        self._last_decode_meta = rows
      else:
        self._last_decode_meta = None
        for hyp in hypotheses:
          if hasattr(hyp, 'text'):
            texts.append(hyp.text)
          elif isinstance(hyp, str):
            texts.append(hyp)
          else:
            texts.append(str(hyp))
      return texts

    # --- Fallback: use transcribe() with temp files ---
    logger.warning('NeMo model type not recognized for tensor decode, '
                   'falling back to dummy output')
    return [''] * B

  def nemo_transcribe(self, audio_paths, batch_size=8):
    """Transcribe audio files using NeMo's built-in pipeline.

    Args:
      audio_paths: list of audio file paths
      batch_size: batch size for transcription

    Returns:
      list of transcribed text strings
    """
    self.nemo_model.eval()
    with torch.no_grad():
      hypotheses = self.nemo_model.transcribe(
          [str(p) for p in audio_paths],
          batch_size=batch_size,
          verbose=False)
      if hasattr(hypotheses[0], 'text'):
        return [h.text for h in hypotheses]
      else:
        return [str(h) for h in hypotheses]

  def save_pretrained(self, path):
    """Save model to path in inference-ready format.
    
    Saves nemo_model.nemo + ctc_head.pt + model_meta.json so that
    submit.py can load directly without export_model.py.
    """
    os.makedirs(path, exist_ok=True)
    # If LoRA is active and merge_on_save is set, merge LoRA weights into
    # the base encoder so the saved .nemo is a standard full-rank model.
    if self._nemo_lora and FLAGS.nemo_lora_merge_on_save:
      logger.info('Merging LoRA weights into encoder before saving...')
      self.nemo_model.encoder = self.nemo_model.encoder.merge_and_unload()
      self.encoder = self.nemo_model.encoder
      self._nemo_lora = False  # no longer a LoRA model after merge
      logger.info('LoRA merge complete — saved model will be standard full-rank.')
    # Save NeMo model
    nemo_path = os.path.join(path, 'nemo_model.nemo')
    self.nemo_model.save_to(nemo_path)
    # Also save a slim version (architecture + tokenizer, no weights) for
    # submit2.py packing — reduces zip by ~2.4 GB since model.pt has all weights.
    try:
      import tarfile, io
      slim_path = os.path.join(path, 'nemo_model_slim.nemo')
      with tarfile.open(nemo_path, 'r') as tar_in:
        with tarfile.open(slim_path, 'w') as tar_out:
          for member in tar_in.getmembers():
            if member.name.endswith('model_weights.ckpt'):
              buf = io.BytesIO()
              torch.save({}, buf)
              buf.seek(0)
              new_member = tarfile.TarInfo(name=member.name)
              new_member.size = buf.getbuffer().nbytes
              tar_out.addfile(new_member, buf)
            else:
              f = tar_in.extractfile(member)
              tar_out.addfile(member, f)
      logger.info(f'Slim NeMo saved to {slim_path} '
                  f'({os.path.getsize(slim_path)/1024:.0f} KB vs '
                  f'{os.path.getsize(nemo_path)/1024/1024:.0f} MB full)')
    except Exception as e:
      logger.warning(f'Failed to create slim NeMo: {e}')
    # Save our CTC head
    if self.use_ctc and hasattr(self, 'ctc_head'):
      torch.save(self.ctc_head.state_dict(), f'{path}/ctc_head.pt')
    if getattr(self, '_inter_ctc_enabled', False) and hasattr(self, 'inter_ctc_heads'):
      torch.save(self.inter_ctc_heads.state_dict(), f'{path}/inter_ctc_heads.pt')
    # Save word CTC head (auxiliary)
    if hasattr(self, 'word_ctc_head'):
      torch.save(self.word_ctc_head.state_dict(), f'{path}/word_ctc_head.pt')
      logger.info(f'Word CTC head saved to {path}/word_ctc_head.pt')
    # Save custom S2S decoder (AED / RNNT)
    dt = getattr(self, '_s2s_decoder_type', 'native')
    if dt == 'aed' and hasattr(self, 'aed_decoder'):
      torch.save(self.aed_decoder.state_dict(), f'{path}/aed_decoder.pt')
      logger.info(f'AED decoder saved to {path}/aed_decoder.pt')
    elif dt in ('rnnt_custom', 'rnnt_reuse') and hasattr(self, 'rnnt_decoder'):
      torch.save(self.rnnt_decoder.state_dict(), f'{path}/rnnt_decoder.pt')
      logger.info(f'RNNT decoder saved to {path}/rnnt_decoder.pt')
    elif dt == 'tdt_reuse' and hasattr(self, 'tdt_decoder'):
      torch.save(self.tdt_decoder.state_dict(), f'{path}/tdt_decoder.pt')
      logger.info(f'TDT decoder saved to {path}/tdt_decoder.pt')
      if hasattr(self, 'word_tdt_decoder'):
        torch.save(self.word_tdt_decoder.state_dict(), f'{path}/word_tdt_decoder.pt')
        logger.info(f'Word TDT decoder saved to {path}/word_tdt_decoder.pt')
    # Save NeMo native CTC decoder state (方案B)
    if self._nemo_native_ctc and hasattr(self, 'nemo_ctc_decoder'):
      torch.save(self.nemo_ctc_decoder.state_dict(), f'{path}/nemo_ctc_decoder.pt')
      logger.info(f'NeMo native CTC decoder saved to {path}/nemo_ctc_decoder.pt')
    # Save preprocessor_config.json marker for pack_submission.sh detection
    import json as _json
    marker = {
      'model_type': 'nemo_native_ctc' if self._nemo_native_ctc else ('nemo_ctc' if (self.use_ctc and hasattr(self, 'ctc_head')) else 'nemo_s2s'),
      'backbone': getattr(FLAGS, 'backbone', ''),
      'nemo_native_ctc': self._nemo_native_ctc,
      'nemo_adapter': getattr(self, '_nemo_adapter', False),
      'adapter_name': getattr(self, '_adapter_name', ''),
      'nemo_trim_vocab_enabled': getattr(self, '_nemo_trim_vocab_enabled', False),
      'nemo_trim_vocab_size': getattr(self, '_nemo_trim_compact_size', None),
    }
    with open(os.path.join(path, 'preprocessor_config.json'), 'w') as f:
      _json.dump(marker, f, indent=2)
    if getattr(self, '_nemo_trim_vocab_enabled', False) and getattr(self, '_nemo_trim_keep_ids', None):
      with open(os.path.join(path, 'nemo_trim_vocab_ids.json'), 'w') as f:
        _json.dump(list(self._nemo_trim_keep_ids), f)
    # Save model_meta.json
    if self._nemo_native_ctc:
      meta_type = 'nemo_native_ctc'
    elif self.use_ctc and hasattr(self, 'ctc_head'):
      meta_type = 'nemo_ctc'
    else:
      meta_type = 'nemo_s2s'
    self.save_model_meta(path, meta_type, self.encoder_dim,
                         backbone=getattr(FLAGS, 'backbone', ''),
                         nemo_native_ctc=self._nemo_native_ctc,
                         nemo_adapter=getattr(self, '_nemo_adapter', False))
    logger.info(f'NeMo model & CTC head saved to {path}')
