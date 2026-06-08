#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   whisper.py
#        \author   chenghuige
#          \date   2025-02-13
#   \Description   Whisper encoder + decoder for Pasketti ASR.
#                  ctc_weight controls loss mix (0=s2s, 0~1=hybrid, 1=ctc).
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
from src.config import *
from src.models.base import BaseASRModel

import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration


class Model(BaseASRModel):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    backbone_name = BACKBONES.get(FLAGS.backbone, FLAGS.backbone)
    ic(backbone_name)

    try:
      self.backbone = WhisperForConditionalGeneration.from_pretrained(
          FLAGS.model_dir, trust_remote_code=True)
      logger.info(f'backbone loaded from {FLAGS.model_dir}')
    except Exception:
      self.backbone = WhisperForConditionalGeneration.from_pretrained(
          backbone_name, trust_remote_code=True)
      logger.info(f'backbone loaded from {backbone_name}')

    self.config = self.backbone.config

    # ---- Whisper LoRA (PEFT) setup ----
    self._whisper_lora = getattr(FLAGS, 'whisper_lora', False)
    if self._whisper_lora:
      self._setup_whisper_lora()

    # ---- CTC head ----
    self._init_ctc_head(self.config.d_model)

    # ---- InterCTC ----
    num_encoder_layers = self.config.encoder_layers
    self._init_inter_ctc(self.config.d_model, num_encoder_layers)

    # ---- CTC layer fusion ----
    self._init_ctc_layer_fusion(num_encoder_layers)

    # ---- Auxiliary metadata heads (age / domain) ----
    self._init_aux_heads(self.config.d_model)

    # Freeze encoder (skip if LoRA mode — already handled by PEFT)
    if FLAGS.freeze_encoder and not self._whisper_lora:
      for param in self._whisper_model.encoder.parameters():
        param.requires_grad = False
      logger.info('Encoder frozen, only decoder will be trained')

    if FLAGS.gradient_checkpointing:
      self.backbone.gradient_checkpointing_enable()

    self.backbone.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
        language=FLAGS.language, task=FLAGS.task)
    self.backbone.config.suppress_tokens = []

    # For LoRA mode, set opt_params to only trainable (LoRA + CTC head) weights.
    if self._whisper_lora:
      import gezi
      opt_params = [p for p in self.parameters() if p.requires_grad]
      gezi.set('opt_params', opt_params)
      logger.info(f'LoRA: set opt_params with {len(opt_params)} param groups')

    self._log_params()

  @property
  def _whisper_model(self):
    """Access the underlying WhisperModel (encoder+decoder), handling PEFT wrapping.

    Without PEFT: self.backbone = WhisperForConditionalGeneration
      -> self.backbone.model = WhisperModel
    With PEFT:    self.backbone = PeftModelForSeq2SeqLM
      -> self.backbone.model = WhisperForConditionalGeneration
      -> self.backbone.model.model = WhisperModel
    """
    if self._whisper_lora:
      return self.backbone.model.model
    return self.backbone.model

  def _encode(self, input_features, attention_mask=None):
    encoder = self._whisper_model.encoder
    need_hidden = self._inter_ctc_enabled or self._ctc_layer_fusion_enabled
    if need_hidden:
      outputs = encoder(
          input_features, output_hidden_states=True)
      # hidden_states is tuple of (embedding + N layers), 1-indexed for layers
      all_hidden = outputs.hidden_states  # tuple of (B, T, D), length = num_layers + 1
      if self._inter_ctc_enabled:
        self._inter_hidden_states = [
            all_hidden[layer_idx + 1] for layer_idx in self._inter_ctc_layers
        ]
      if self._ctc_layer_fusion_enabled:
        self._ctc_fusion_hidden_states = [
            all_hidden[layer_idx + 1] for layer_idx in self._ctc_fusion_layers
        ]
      return outputs.last_hidden_state
    return encoder(input_features).last_hidden_state

  def _s2s_forward(self, input_features, labels, encoder_hidden_states,
                   attention_mask=None):
    # Use backbone forward to get logits; compute per-sample loss ourselves
    # for ext_weight support. HF Whisper logits are aligned with labels
    # (model does shift_tokens_right internally), so NO additional shift needed.
    # IPA constraint is applied only during generation via IPALogitsProcessor,
    # NOT during training.
    # Reuse encoder output from _encode() to avoid redundant encoding,
    # especially important in multi-task mode (IPA + Word).
    from transformers.modeling_outputs import BaseModelOutput
    if encoder_hidden_states is not None:
      output = self.backbone(
          encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
          labels=labels,
      )
    else:
      output = self.backbone(input_features=input_features, labels=labels)
    logits = output.logits  # (B, L, V)
    B, L, V = logits.shape
    label_smoothing = getattr(FLAGS, 'label_smoothing', 0.0)
    per_token = F.cross_entropy(
        logits.view(-1, V),
        labels.view(-1),
        ignore_index=-100,
        reduction='none',
        label_smoothing=label_smoothing,
    ).view(B, L)
    # Mean over valid tokens per sample → (B,)
    mask = (labels != -100).float()
    per_sample = (per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    return per_sample

  def _s2s_generate(self, input_features, encoder_hidden_states,
                    attention_mask=None):
    from transformers.modeling_outputs import BaseModelOutput
    kwargs = dict(
        input_features=input_features,
        max_new_tokens=FLAGS.max_new_tokens,
        num_beams=FLAGS.num_beams,
        length_penalty=FLAGS.length_penalty,
        language=FLAGS.language,
        task=FLAGS.task,
    )
    if getattr(FLAGS, 'no_repeat_ngram_size', 0) > 0:
      kwargs['no_repeat_ngram_size'] = FLAGS.no_repeat_ngram_size
    # Reuse encoder output from forward() to avoid re-encoding
    if encoder_hidden_states is not None:
      kwargs['encoder_outputs'] = BaseModelOutput(
          last_hidden_state=encoder_hidden_states)
    lp = self._get_logits_processors()
    if lp:
      kwargs['logits_processor'] = lp
    return self.backbone.generate(**kwargs)

  def _setup_whisper_lora(self):
    """Set up LoRA (PEFT) for Whisper model.

    Wraps backbone with PEFT LoRA adapters, freezes base model,
    only LoRA weights are trainable. Weights are part of state_dict
    and saved/loaded via best.pt automatically.
    """
    try:
      from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
      raise ImportError(
          'peft is required for Whisper LoRA. '
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
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    self.backbone = get_peft_model(self.backbone, lora_config)

    # Log param count
    trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in self.backbone.parameters())
    logger.info(f'Whisper LoRA enabled: '
                f'{trainable:,} trainable / {total:,} total '
                f'({trainable/total*100:.2f}%)')
    logger.info(f'  r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, '
                f'target_modules={target_modules}')

  def save_pretrained(self, path):
    # For LoRA models, merge weights back before saving HF format
    # so that the saved model works without PEFT at inference time.
    if getattr(self, '_whisper_lora', False):
      try:
        merged = self.backbone.merge_and_unload()
        merged.save_pretrained(path)
        # Re-wrap with PEFT for continued training if needed
        # (merge_and_unload modifies in-place, so re-setup)
        logger.info(f'LoRA: merged weights saved to {path}')
      except Exception as e:
        logger.warning(f'LoRA merge_and_unload failed: {e}, saving with adapters')
        self.backbone.save_pretrained(path)
    else:
      self.backbone.save_pretrained(path)
    self.processor.save_pretrained(path)
    if self.use_ctc and hasattr(self, 'ctc_head'):
      torch.save(self.ctc_head.state_dict(), f'{path}/ctc_head.pt')
    if getattr(self, '_inter_ctc_enabled', False) and hasattr(self, 'inter_ctc_heads'):
      torch.save(self.inter_ctc_heads.state_dict(), f'{path}/inter_ctc_heads.pt')
    if hasattr(self, 'word_ctc_head'):
      torch.save(self.word_ctc_head.state_dict(), f'{path}/word_ctc_head.pt')
    model_type = 'whisper_ctc' if self.use_ctc else 'whisper_s2s'
    d_model = self.backbone.config.d_model if not getattr(self, '_whisper_lora', False) else self.config.d_model
    self.save_model_meta(path, model_type, d_model,
                         whisper_lora=getattr(self, '_whisper_lora', False))
    logger.info(f'Model & processor saved to {path}')
