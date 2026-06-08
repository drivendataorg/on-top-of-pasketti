#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   wav2vec2.py
#        \author   chenghuige
#          \date   2025-02-16
#   \Description   Wav2Vec2 / HuBERT / WavLM / Data2Vec Audio encoder for Pasketti ASR.
#                  Loads the appropriate encoder model based on backbone name.
#                  Input: raw 16 kHz waveform (NOT mel spectrogram).
#                  ctc_weight controls loss mix (0=s2s, 0~1=hybrid, 1=ctc).
#                  ctc_weight=1 is the primary use-case (CTC-only, no decoder).
#
#  Usage:
#    --model=wav2vec2 --backbone=wav2vec2-large --ctc_weight=1.0
#    --model=wav2vec2 --backbone=hubert-large  --ctc_weight=1.0
#    --model=wav2vec2 --backbone=wavlm-large   --ctc_weight=1.0
#    --model=wav2vec2 --backbone=data2vec-audio-large --ctc_weight=1.0
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from gezi.common import *
from src.config import *
from src.models.base import BaseASRModel, IPA_CHAR_VOCAB, IPA_CTC_VOCAB_SIZE


def _construct_empty_encoder(config_path):
  """Construct an empty encoder model from a local config.json (no weights).
  
  Like NeMo's nemo_model_slim.nemo approach: architecture only, weights
  will be loaded later via gz.load_weights(model, model.pt).
  """
  import json
  with open(config_path) as f:
    cfg = json.load(f)
  model_type = cfg.get('model_type', '')
  if model_type == 'wavlm':
    from transformers import WavLMModel, WavLMConfig
    config = WavLMConfig.from_pretrained(os.path.dirname(config_path))
    model = WavLMModel(config)
  elif model_type == 'hubert':
    from transformers import HubertModel, HubertConfig
    config = HubertConfig.from_pretrained(os.path.dirname(config_path))
    model = HubertModel(config)
  elif model_type == 'data2vec-audio':
    from transformers import Data2VecAudioModel, Data2VecAudioConfig
    config = Data2VecAudioConfig.from_pretrained(os.path.dirname(config_path))
    model = Data2VecAudioModel(config)
  else:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
    config = Wav2Vec2Config.from_pretrained(os.path.dirname(config_path))
    model = Wav2Vec2Model(config)
  logger.info(f'Constructed empty encoder from {config_path} '
              f'(type={model_type}, hidden_size={config.hidden_size})')
  return model, config.hidden_size


def _load_encoder(backbone_name):
  """Load wav2vec2 / hubert / wavlm / data2vec-audio model (encoder only, no CTC head).
  
  Supports three modes (tried in order):
    1. Local directory with config.json → construct empty model (like NeMo slim);
       weights come later via gz.load_weights(model, model.pt/best.pt).
    2. Local directory with config.json + encoder.pt → load encoder weights directly.
    3. HuggingFace repo name (e.g. 'microsoft/wavlm-large') → from_pretrained().
  """
  import torch as _torch
  if _torch.cuda.is_available():
    logger.info(f'[MEM_DIAG] before _load_encoder({backbone_name}): '
                f'peak={_torch.cuda.max_memory_allocated()/1e9:.2f}G '
                f'cur={_torch.cuda.memory_reserved()/1e9:.2f}G')
  
  import os
  if os.path.isdir(backbone_name):
    config_path = os.path.join(backbone_name, 'config.json')
    encoder_path = os.path.join(backbone_name, 'encoder.pt')
    
    if os.path.exists(config_path):
      # Mode 1 & 2: local config.json — construct empty architecture
      model, hidden_size = _construct_empty_encoder(config_path)
      
      # If encoder.pt exists, load encoder weights now (standalone export mode).
      # Otherwise weights come from model.pt/best.pt via gz.load_weights() later.
      if os.path.exists(encoder_path):
        encoder_sd = _torch.load(encoder_path, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(encoder_sd, strict=False)
        if missing:
          logger.debug(f'_load_encoder local: {len(missing)} missing keys (expected for partial load)')
        logger.info(f'Loaded encoder weights from {encoder_path}')
      else:
        logger.info(f'No encoder.pt found — weights will be loaded from model.pt/best.pt')
      return model, hidden_size
  
  # Mode 3: Standard HuggingFace loading (online or cached)
  name_lower = backbone_name.lower()
  if 'hubert' in name_lower:
    from transformers import HubertModel, HubertConfig
    config = HubertConfig.from_pretrained(backbone_name)
    model = HubertModel.from_pretrained(backbone_name)
    return model, config.hidden_size
  elif 'wavlm' in name_lower:
    from transformers import WavLMModel, WavLMConfig
    config = WavLMConfig.from_pretrained(backbone_name)
    model = WavLMModel.from_pretrained(backbone_name)
    return model, config.hidden_size
  elif 'data2vec' in name_lower:
    from transformers import Data2VecAudioModel, Data2VecAudioConfig
    config = Data2VecAudioConfig.from_pretrained(backbone_name)
    model = Data2VecAudioModel.from_pretrained(backbone_name)
    return model, config.hidden_size
  else:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
    config = Wav2Vec2Config.from_pretrained(backbone_name)
    model = Wav2Vec2Model.from_pretrained(backbone_name)
    return model, config.hidden_size


def _load_native_ctc_model(backbone_name):
  """Load a native HuggingFace ForCTC checkpoint for raw eval.

  This path is strictly eval-oriented: we reuse the pretrained encoder + lm_head
  exactly as shipped by the HF checkpoint instead of attaching a project CTC head.
  """
  from transformers import AutoConfig, AutoModelForCTC

  config = AutoConfig.from_pretrained(backbone_name)
  model = AutoModelForCTC.from_pretrained(backbone_name)

  encoder_attr_map = {
      'wav2vec2': 'wav2vec2',
      'hubert': 'hubert',
      'wavlm': 'wavlm',
      'data2vec-audio': 'data2vec_audio',
  }
  encoder_attr = encoder_attr_map.get(config.model_type)
  assert encoder_attr is not None, (
      f'raw_ctc_eval only supports wav2vec2/hubert/wavlm/data2vec-audio ForCTC checkpoints, '
      f'got model_type={config.model_type!r} from {backbone_name}')

  encoder = getattr(model, encoder_attr, None)
  assert encoder is not None, (
      f'Failed to locate native encoder attribute {encoder_attr!r} on '
      f'{type(model).__name__}')

  hidden_size = getattr(config, 'hidden_size', None)
  assert hidden_size is not None, f'Failed to infer hidden_size from {backbone_name}'

  blank_id = config.pad_token_id
  if blank_id is None:
    blank_id = 0

  return model, encoder, hidden_size, blank_id, config.model_type


# ---- Espeak IPA → Our IPA vocab mapping for pretrained CTC head transfer ----
# Known character discrepancies between espeak phonemizer vocab and our IPA vocab:
#   Our ASCII 'g' (U+0067) → espeak's 'ɡ' (U+0261, Latin small letter script G)
#   Our 'ʤ' (U+02A4, affricate ligature) → espeak's 'dʒ' (two-char token)
#   Our 'ʧ' (U+02A7, affricate ligature) → espeak's 'tʃ' (two-char token)
#   Our ' ' (space, word boundary) → espeak's '|' (pipe symbol)
#   Our 'ː' (length mark) → espeak has only combined forms (aː, iː, etc.), no standalone
_ESPEAK_CHAR_MAP = {
  'g': 'ɡ',    # ASCII g → IPA ɡ (U+0261)
  'ʤ': 'dʒ',   # affricate ligature → digraph token
  'ʧ': 'tʃ',   # affricate ligature → digraph token
  ' ': '|',     # word boundary
}


def _init_ctc_from_pretrained(ctc_head, backbone_name):
  """Initialize CTC head weights from a pretrained ForCTC model via vocab mapping.
  
  Loads Wav2Vec2ForCTC, extracts the lm_head weights (392 classes for espeak),
  maps them to our 53-class IPA vocabulary, and copies the mapped rows into
  the CTC head's projection layer.
  
  Returns True if successful, False otherwise.
  """
  import json
  import torch
  from transformers import Wav2Vec2ForCTC
  from huggingface_hub import hf_hub_download

  # 1. Load source vocabulary
  vocab_path = hf_hub_download(repo_id=backbone_name, filename='vocab.json')
  with open(vocab_path) as f:
    src_vocab = json.load(f)  # char/token → index

  # 2. Build mapping: our_idx → src_idx
  mapping = {}  # our IPA vocab index → source vocab index
  mapped_chars = []
  unmapped_chars = []

  for our_idx, char in enumerate(IPA_CHAR_VOCAB):
    if char == '<blank>':
      # blank → <pad> (index 0 in espeak vocab)
      if '<pad>' in src_vocab:
        mapping[our_idx] = src_vocab['<pad>']
        mapped_chars.append(('<blank>', '<pad>'))
      else:
        unmapped_chars.append('<blank>')
      continue

    # Try direct match
    if char in src_vocab:
      mapping[our_idx] = src_vocab[char]
      mapped_chars.append((char, char))
    # Try special mapping
    elif char in _ESPEAK_CHAR_MAP and _ESPEAK_CHAR_MAP[char] in src_vocab:
      target = _ESPEAK_CHAR_MAP[char]
      mapping[our_idx] = src_vocab[target]
      mapped_chars.append((char, target))
    else:
      unmapped_chars.append(char)

  # 3. Sanity check: need enough mappings to be useful
  min_mapped = IPA_CTC_VOCAB_SIZE * 0.8  # at least 80%
  if len(mapped_chars) < min_mapped:
    logger.error(
      f'init_ctc_from_pretrained: only {len(mapped_chars)}/{IPA_CTC_VOCAB_SIZE} '
      f'chars mapped from {backbone_name} — too few, aborting')
    return False

  # 4. Load ForCTC model to get CTC head weights
  logger.info(f'Loading Wav2Vec2ForCTC from {backbone_name} for CTC weight transfer...')
  import torch as _torch
  if _torch.cuda.is_available():
    logger.info(f'[MEM_DIAG] before Wav2Vec2ForCTC.from_pretrained: '
                f'peak={_torch.cuda.max_memory_allocated()/1e9:.2f}G '
                f'cur={_torch.cuda.memory_reserved()/1e9:.2f}G')
  model_for_ctc = Wav2Vec2ForCTC.from_pretrained(backbone_name)
  if _torch.cuda.is_available():
    logger.info(f'[MEM_DIAG] after Wav2Vec2ForCTC.from_pretrained: '
                f'peak={_torch.cuda.max_memory_allocated()/1e9:.2f}G '
                f'cur={_torch.cuda.memory_reserved()/1e9:.2f}G')
  src_weight = model_for_ctc.lm_head.weight.data  # (src_vocab_size, hidden)
  src_bias = model_for_ctc.lm_head.bias.data       # (src_vocab_size,)
  del model_for_ctc  # free memory immediately

  # 5. Copy mapped weights
  with torch.no_grad():
    for our_idx, src_idx in mapping.items():
      ctc_head.proj.weight.data[our_idx] = src_weight[src_idx]
      ctc_head.proj.bias.data[our_idx] = src_bias[src_idx]

  logger.info(
    f'CTC head initialized from pretrained {backbone_name}: '
    f'{len(mapped_chars)}/{IPA_CTC_VOCAB_SIZE} mapped, '
    f'{len(unmapped_chars)} unmapped: {unmapped_chars}')
  return True


class Model(BaseASRModel):
  """Wav2Vec2 / HuBERT / WavLM / Data2Vec Audio encoder-based ASR model.

  These models take raw 16 kHz waveforms as input (not mel spectrograms).
  The self-supervised pre-trained encoder provides strong phonetic features,
  making them especially well-suited for CTC-based IPA transcription.

  When ctc_only (default): lightweight, no decoder, parallel inference.
  When hybrid/s2s: loads Whisper decoder, projects encoder output to d_model.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    backbone_name = BACKBONES.get(FLAGS.backbone, FLAGS.backbone)
    ic(backbone_name)

    self._raw_ctc_eval = bool(getattr(FLAGS, 'raw_ctc_eval', False))
    self._native_ctc = self._raw_ctc_eval or bool(getattr(FLAGS, 'native_ctc', False))
    self._raw_ctc_logits = None
    self._raw_ctc_blank_id = None

    if self._native_ctc:
      assert FLAGS.track == 'word', 'native_ctc/raw_ctc_eval is only intended for the word track'
      assert self.ctc_only, 'native_ctc/raw_ctc_eval requires --ctc_weight=1 or another pure CTC setup'
      assert not self.constrain_ipa, 'native_ctc does not support IPA-constrained decoding'
      assert not getattr(FLAGS, 'inter_ctc', False), 'native_ctc does not support inter_ctc'
      assert not getattr(FLAGS, 'ctc_layer_fusion', []), 'native_ctc does not support ctc_layer_fusion'
      assert getattr(FLAGS, 'ctc_fusion_last_n', None) is None, 'native_ctc does not support ctc_fusion_last_n'

    # ---- Wav2Vec2 / HuBERT encoder ----
    if self._native_ctc:
      self.raw_ctc_model, self.encoder, self.encoder_dim, self._raw_ctc_blank_id, self._raw_ctc_model_type = \
        _load_native_ctc_model(backbone_name)
      self.ctc_blank_id = self._raw_ctc_blank_id
      logger.info(f'Native CTC model loaded from {backbone_name} '
                  f'(type={self._raw_ctc_model_type}, blank_id={self._raw_ctc_blank_id}, '
                  f'finetune={not self._raw_ctc_eval})')
    else:
      try:
        self.encoder, self.encoder_dim = _load_encoder(FLAGS.model_dir)
        logger.info(f'Encoder loaded from {FLAGS.model_dir}')
      except Exception:
        self.encoder, self.encoder_dim = _load_encoder(backbone_name)
        logger.info(f'Encoder loaded from {backbone_name}')

    import torch as _torch
    if _torch.cuda.is_available():
      logger.info(f'[MEM_DIAG] after _load_encoder done: '
                  f'peak={_torch.cuda.max_memory_allocated()/1e9:.2f}G '
                  f'cur={_torch.cuda.memory_reserved()/1e9:.2f}G')

    logger.info(f'Encoder type: {type(self.encoder).__name__}, '
                 f'hidden_size={self.encoder_dim}')

    # ---- Detect WavLM for fp32 encoder workaround ----
    from transformers import WavLMModel
    raw_encoder = self.encoder
    self._is_wavlm = isinstance(raw_encoder, WavLMModel)
    if self._is_wavlm:
      logger.info('WavLM detected: encoder forward will run in fp32 '
                  '(gated position bias is numerically unstable under bf16/fp16)')

    # ---- Disable built-in spec-augment time masking ----
    # HuBERT/Wav2Vec2 apply random time masking during training which crashes
    # when a short audio produces fewer hidden frames than mask_time_length
    # (default 10).  We have our own augmentation pipeline, so turn it off.
    if hasattr(self.encoder, 'config'):
      self.encoder.config.apply_spec_augment = False

    # ---- LoRA (PEFT) setup ----
    self._wav2vec2_lora = getattr(FLAGS, 'whisper_lora', False)  # reuse same flag
    if self._wav2vec2_lora and not self._native_ctc:
      self._setup_lora()

    # ---- Freeze feature extractor CNN (common practice) ----
    # Skip if LoRA mode — PEFT handles freezing
    # NOTE: apply for both native_ctc training and normal training
    if getattr(FLAGS, 'freeze_feature_extractor', True) and not self._wav2vec2_lora:
      self.encoder.feature_extractor._freeze_parameters()
      logger.info('Feature extractor CNN frozen')

    # ---- Gradient checkpointing for encoder ----
    # For native_ctc training, enable grad checkpointing to save memory
    if FLAGS.gradient_checkpointing and not self._raw_ctc_eval:
      self.encoder.gradient_checkpointing_enable()
      logger.info(f'Gradient checkpointing enabled for {type(self.encoder).__name__}')

    # ---- CTC head ----
    # native_ctc uses the pretrained lm_head from ForCTC model, skip project CTC head
    if self._native_ctc:
      # Set attributes that _init_ctc_head would normally initialize
      self._ctc_char_level = True  # native CTC char tokenizer (32 vocab)
      self.ctc_loss_fn = nn.CTCLoss(blank=self.ctc_blank_id, reduction='none',
                                    zero_infinity=True)
      logger.info(f'Native CTC head: lm_head from ForCTC, blank_id={self.ctc_blank_id}, '
                  f'vocab={self.raw_ctc_model.lm_head.out_features}')
    else:
      self._init_ctc_head(self.encoder_dim)

    # ---- Transfer pretrained CTC weights via vocab mapping ----
    if getattr(FLAGS, 'init_ctc_from_pretrained', False) and not self._native_ctc:
      if self.use_ctc and getattr(self, '_ctc_char_level', False):
        ok = _init_ctc_from_pretrained(self.ctc_head, backbone_name)
        assert ok, (
          f'init_ctc_from_pretrained failed for {backbone_name}. '
          f'This backbone may not have a compatible espeak IPA vocab.')
      else:
        logger.warning(
          'init_ctc_from_pretrained requires use_ctc=True and constrain_ipa=True, skipping')

    # ---- InterCTC ----
    num_encoder_layers = self.encoder.config.num_hidden_layers
    if not self._native_ctc:
      self._init_inter_ctc(self.encoder_dim, num_encoder_layers)

    # ---- CTC layer fusion ----
    if not self._native_ctc:
      self._init_ctc_layer_fusion(num_encoder_layers)

    # ---- Auxiliary metadata heads (age / domain) ----
    self._init_aux_heads(self.encoder_dim)

    # ---- Optional custom S2S / TDT decoders ----
    if not self._native_ctc:
      self._init_s2s_decoder(self.encoder_dim)

    # ---- Optional Whisper decoder for hybrid/s2s ----
    self.backbone = None  # Whisper decoder
    if (not self.ctc_only
        and getattr(self, '_s2s_decoder_type', 'native') == 'native'):
      from transformers import WhisperForConditionalGeneration, WhisperConfig
      whisper_bb = getattr(FLAGS, 'tokenizer_backbone', 'openai/whisper-large-v3')
      whisper_config = WhisperConfig.from_pretrained(whisper_bb)
      d_model = whisper_config.d_model

      try:
        self.backbone = WhisperForConditionalGeneration.from_pretrained(
            FLAGS.model_dir, trust_remote_code=True)
      except Exception:
        self.backbone = WhisperForConditionalGeneration.from_pretrained(
            whisper_bb, trust_remote_code=True)

      # Freeze Whisper encoder (we use wav2vec2/hubert instead)
      for param in self.backbone.model.encoder.parameters():
        param.requires_grad = False
      logger.info(f'Whisper decoder loaded from {whisper_bb} '
                   f'(encoder frozen, using {type(self.encoder).__name__})')

      # Projection: encoder_dim -> d_model
      if self.encoder_dim != d_model:
        self.enc_proj = nn.Linear(self.encoder_dim, d_model)
      else:
        self.enc_proj = nn.Identity()

      self.backbone.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
          language=FLAGS.language, task=FLAGS.task)
      self.backbone.config.suppress_tokens = []

      if FLAGS.gradient_checkpointing:
        self.backbone.gradient_checkpointing_enable()

    self._log_params()

    # For LoRA mode, set opt_params to only trainable weights.
    if self._wav2vec2_lora:
      import gezi
      opt_params = [p for p in self.parameters() if p.requires_grad]
      gezi.set('opt_params', opt_params)
      logger.info(f'LoRA: set opt_params with {len(opt_params)} param groups')

  def _setup_lora(self):
    """Set up LoRA (PEFT) for HuBERT/Wav2Vec2/WavLM encoder.

    Wraps self.encoder with PEFT LoRA adapters. Only LoRA weights are trainable.
    Target modules: q_proj, v_proj in each Transformer layer of the encoder.
    """
    try:
      from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
      raise ImportError(
          'peft is required for LoRA. '
          'Install with: pip install peft')

    lora_r = getattr(FLAGS, 'lora_r', 16)
    lora_alpha = getattr(FLAGS, 'lora_alpha', None) or lora_r * 2
    lora_dropout = getattr(FLAGS, 'lora_dropout', 0.05)
    target_modules_str = getattr(FLAGS, 'lora_target_modules', 'q_proj,v_proj')
    target_modules = [m.strip() for m in target_modules_str.split(',')]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias='none',
    )

    self.encoder = get_peft_model(self.encoder, lora_config)

    trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in self.encoder.parameters())
    logger.info(f'LoRA enabled on {type(self.encoder.model).__name__}: '
                f'{trainable:,} trainable / {total:,} total '
                f'({trainable/total*100:.2f}%)')
    logger.info(f'  r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, '
                f'target_modules={target_modules}')

  def _encode(self, input_features, attention_mask=None):
    """Encode raw waveform through wav2vec2/hubert.

    input_features: (B, T) raw waveform
    attention_mask:  (B, T) 1=real, 0=padding
    Returns: (B, T', encoder_dim) where T' < T due to CNN downsampling
    """
    if self._native_ctc:
      encoder = self.encoder
      outputs = encoder(
          input_values=input_features,
          attention_mask=attention_mask,
          output_hidden_states=True,
      )
      hidden_states = outputs.hidden_states
      if attention_mask is not None and hasattr(encoder, '_get_feat_extract_output_lengths'):
        input_lengths = attention_mask.sum(dim=-1).to(input_features.device)
        self._last_enc_len = encoder._get_feat_extract_output_lengths(input_lengths).long()
      else:
        self._last_enc_len = None
      self._raw_ctc_logits = self.raw_ctc_model.lm_head(self.raw_ctc_model.dropout(outputs.last_hidden_state))
      return outputs.last_hidden_state

    # With PEFT wrapping, self.encoder becomes PeftModel;
    # PeftModel delegates forward() correctly, but for output_hidden_states
    # we need to pass through the underlying model.
    encoder = self.encoder.model if self._wav2vec2_lora else self.encoder
    
    # WavLM's gated relative position bias uses F.multi_head_attention_forward
    # (_supports_sdpa=False) with learned GRU gating that is numerically
    # unstable under bf16/fp16 autocast, producing intermittent NaN.
    # Force fp32 for the entire encoder forward to prevent this.
    use_fp32_encode = self._is_wavlm
    
    need_hidden = self._inter_ctc_enabled or self._ctc_layer_fusion_enabled
    if use_fp32_encode:
      ctx = torch.cuda.amp.autocast(enabled=False)
      input_features = input_features.float()
    else:
      ctx = contextlib.nullcontext()
    
    with ctx:
      if need_hidden:
        outputs = encoder(
            input_values=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        all_hidden = outputs.hidden_states  # tuple of (B, T, D)
        if self._inter_ctc_enabled:
          self._inter_hidden_states = [
              all_hidden[layer_idx + 1] for layer_idx in self._inter_ctc_layers
          ]
        if self._ctc_layer_fusion_enabled:
          self._ctc_fusion_hidden_states = [
              all_hidden[layer_idx + 1] for layer_idx in self._ctc_fusion_layers
          ]
        return outputs.last_hidden_state
      outputs = encoder(
          input_values=input_features,
          attention_mask=attention_mask,
      )
      return outputs.last_hidden_state

  def _raw_ctc_tokenizer(self):
    tokenizer = getattr(self.processor, 'tokenizer', None)
    if tokenizer is None:
      tokenizer = self.tokenizer
    assert tokenizer is not None, 'raw_ctc_eval requires a HuggingFace tokenizer'
    return tokenizer

  def _get_raw_ctc_logits(self, enc_out=None):
    if self._raw_ctc_logits is not None:
      return self._raw_ctc_logits
    assert enc_out is not None, '_get_raw_ctc_logits requires enc_out when cache is empty'
    return self.raw_ctc_model.lm_head(self.raw_ctc_model.dropout(enc_out))

  def _compute_ctc_loss(self, enc_out, labels):
    if not self._native_ctc:
      return super()._compute_ctc_loss(enc_out, labels)

    ctc_logits = self._get_raw_ctc_logits(enc_out)
    log_probs = F.log_softmax(ctc_logits.float(), dim=-1)
    B, T, _ = log_probs.shape
    log_probs_t = log_probs.transpose(0, 1)

    if self._last_enc_len is not None:
      input_lengths = self._last_enc_len.to(log_probs.device).clamp(min=1, max=T)
    else:
      input_lengths = torch.full((B,), T, dtype=torch.long, device=log_probs.device)

    label_texts = getattr(self, '_current_label_texts', None)
    assert label_texts is not None, 'raw_ctc_eval requires label_texts for CTC loss'
    tokenizer = self._raw_ctc_tokenizer()
    # Normalize label texts before tokenizing: the raw orthographic text may be
    # lowercase / contain digits / punctuation that are not in the CTC vocab
    # (e.g. HuBERT vocab is uppercase A-Z only).  Without normalization every
    # non-vocab character becomes <unk> (id=3), corrupting CTC training.
    from src.preprocess import normalize_text_for_tokenizer
    label_texts = [normalize_text_for_tokenizer(t, tokenizer) for t in label_texts]
    tokenized = tokenizer(list(label_texts), add_special_tokens=False)
    token_seqs = tokenized.input_ids
    target_lengths = torch.tensor([len(seq) for seq in token_seqs],
                                  dtype=torch.long, device=log_probs.device)
    if target_lengths.sum() == 0:
      return torch.zeros(B, device=log_probs.device, requires_grad=True), ctc_logits, target_lengths
    targets_flat = torch.tensor([token for seq in token_seqs for token in seq],
                                dtype=torch.long, device=log_probs.device)

    ctc_loss = F.ctc_loss(
        log_probs_t,
        targets_flat,
        input_lengths,
        target_lengths,
        blank=self._raw_ctc_blank_id,
        reduction='none',
        zero_infinity=True,
    )
    ctc_loss = ctc_loss / target_lengths.clamp(min=1).float()
    self._ctc_entropy = self._compute_ctc_entropy(log_probs)
    return ctc_loss, ctc_logits, target_lengths

  def _generate_ctc(self, enc_out, ctc_logits, device):
    if not self._native_ctc:
      return super()._generate_ctc(enc_out, ctc_logits, device)

    ctc_logits = self._get_raw_ctc_logits(enc_out) if ctc_logits is None else ctc_logits

    if getattr(FLAGS, 'ctc_decode_fp32', False):
      log_probs = F.log_softmax(ctc_logits.float(), dim=-1)
    else:
      log_probs = F.log_softmax(ctc_logits, dim=-1)

    if self._last_enc_len is not None:
      B, T, V = log_probs.shape
      lengths = self._last_enc_len.to(log_probs.device).clamp(min=1, max=T)
      range_t = torch.arange(T, device=log_probs.device).unsqueeze(0)
      mask = range_t >= lengths.unsqueeze(1)
      blank_row = torch.full((V,), float('-inf'), device=log_probs.device)
      blank_row[self._raw_ctc_blank_id] = 0.0
      log_probs = log_probs.clone()
      log_probs[mask] = blank_row

    self._last_ctc_log_probs = log_probs.detach()

    pred_ids = log_probs.argmax(dim=-1)
    # Use tokenizer.batch_decode (not processor — FeatureExtractor has no batch_decode)
    tokenizer = self._raw_ctc_tokenizer()
    pred_texts = tokenizer.batch_decode(pred_ids.cpu(), group_tokens=True)
    pred_texts = [text.strip() for text in pred_texts]
    self._last_pred_texts = pred_texts

    token_seqs = tokenizer(pred_texts, add_special_tokens=False).input_ids
    max_len = FLAGS.max_new_tokens
    B = len(token_seqs)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self._raw_ctc_blank_id
    generated = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(token_seqs):
      seq_len = min(len(seq), max_len)
      if seq_len > 0:
        generated[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=torch.long, device=device)
    return generated

  def _s2s_forward(self, input_features, labels, encoder_hidden_states,
                   attention_mask=None):
    enc_proj = self.enc_proj(encoder_hidden_states)
    return self.backbone(
        encoder_outputs=(enc_proj,),
        labels=labels,
    ).loss

  def _s2s_generate(self, input_features, encoder_hidden_states,
                    attention_mask=None):
    enc_proj = self.enc_proj(encoder_hidden_states)
    encoder_output = type('EO', (), {'last_hidden_state': enc_proj})()
    return self.backbone.generate(
        encoder_outputs=encoder_output,
        max_new_tokens=FLAGS.max_new_tokens,
        num_beams=FLAGS.num_beams,
        length_penalty=FLAGS.length_penalty,
        language=FLAGS.language,
        task=FLAGS.task,
    )

  def save_pretrained(self, path):
    self.processor.save_pretrained(path)
    torch.save(self.encoder.state_dict(), f'{path}/encoder.pt')
    if self._native_ctc and hasattr(self, 'raw_ctc_model'):
      torch.save(self.raw_ctc_model.lm_head.state_dict(), f'{path}/lm_head.pt')
    if self.use_ctc and hasattr(self, 'ctc_head'):
      torch.save(self.ctc_head.state_dict(), f'{path}/ctc_head.pt')
    if getattr(self, '_inter_ctc_enabled', False) and hasattr(self, 'inter_ctc_heads'):
      torch.save(self.inter_ctc_heads.state_dict(), f'{path}/inter_ctc_heads.pt')
    if hasattr(self, 'word_ctc_head'):
      torch.save(self.word_ctc_head.state_dict(), f'{path}/word_ctc_head.pt')
    if hasattr(self, 'tdt_decoder'):
      torch.save(self.tdt_decoder.state_dict(), f'{path}/tdt_decoder.pt')
    if hasattr(self, 'word_tdt_decoder'):
      torch.save(self.word_tdt_decoder.state_dict(), f'{path}/word_tdt_decoder.pt')
    if self.backbone is not None:
      self.backbone.save_pretrained(path)
    if hasattr(self, 'enc_proj') and not isinstance(self.enc_proj, nn.Identity):
      torch.save(self.enc_proj.state_dict(), f'{path}/enc_proj.pt')
    encoder_dim = self.encoder.config.hidden_size if hasattr(self.encoder, 'config') else 768
    self.save_model_meta(path, 'wav2vec2_ctc', encoder_dim)
    logger.info(f'Model saved to {path}')
