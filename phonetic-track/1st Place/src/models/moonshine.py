#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   moonshine.py
#        \author   chenghuige
#          \date   2025-02-22
#   \Description   Moonshine encoder-decoder model for Pasketti ASR.
#                  UsefulSensors' Moonshine: ultra-lightweight ASR, raw waveform
#                  input, encoder-decoder with generate() support.
#
#                  Architecture: MoonshineForConditionalGeneration
#                  Input: raw 16 kHz waveform (NOT mel spectrogram)
#                  tiny:  27M params, hidden_size=288
#                  base:  61M params, hidden_size=416
#
#  Usage:
#    --model=moonshine --backbone=moonshine-tiny --ctc_weight=0
#    --model=moonshine --backbone=moonshine-base --ctc_weight=1.0
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
from src.config import *
from src.models.base import BaseASRModel

from transformers import MoonshineForConditionalGeneration


class Model(BaseASRModel):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    backbone_name = BACKBONES.get(FLAGS.backbone, FLAGS.backbone)
    ic(backbone_name)

    try:
      self.backbone = MoonshineForConditionalGeneration.from_pretrained(
          FLAGS.model_dir, trust_remote_code=True)
      logger.info(f'Moonshine loaded from {FLAGS.model_dir}')
    except Exception:
      self.backbone = MoonshineForConditionalGeneration.from_pretrained(
          backbone_name, trust_remote_code=True)
      logger.info(f'Moonshine loaded from {backbone_name}')

    self.config = self.backbone.config
    encoder_dim = self.config.hidden_size  # 288 (tiny) or 416 (base)

    # ---- CTC head ----
    self._init_ctc_head(encoder_dim)

    # ---- Auxiliary metadata heads (age / domain) ----
    self._init_aux_heads(encoder_dim)

    if FLAGS.freeze_encoder:
      for param in self.backbone.encoder.parameters():
        param.requires_grad = False
      logger.info('Encoder frozen, only decoder will be trained')

    if FLAGS.gradient_checkpointing:
      self.backbone.gradient_checkpointing_enable()

    self._log_params()

  def _encode(self, input_features, attention_mask=None):
    """Encode raw waveform through Moonshine encoder.
    
    input_features: (B, T) raw waveform at 16kHz
    Returns: (B, T', hidden_size)
    """
    encoder_outputs = self.backbone.encoder(
        input_features.unsqueeze(1) if input_features.ndim == 2 else input_features,
        attention_mask=attention_mask,
    )
    return encoder_outputs.last_hidden_state

  def _s2s_forward(self, input_features, labels, encoder_hidden_states,
                   attention_mask=None):
    """Seq2seq forward with Moonshine's native loss."""
    return self.backbone(
        input_values=input_features,
        attention_mask=attention_mask,
        labels=labels,
    ).loss

  def _s2s_generate(self, input_features, encoder_hidden_states,
                    attention_mask=None):
    """Generate token IDs using Moonshine's generate()."""
    from transformers.modeling_outputs import BaseModelOutput
    kwargs = dict(
        max_new_tokens=FLAGS.max_new_tokens,
        num_beams=FLAGS.num_beams,
        length_penalty=FLAGS.length_penalty,
    )
    if getattr(FLAGS, 'no_repeat_ngram_size', 0) > 0:
      kwargs['no_repeat_ngram_size'] = FLAGS.no_repeat_ngram_size
    # Reuse encoder output to avoid re-encoding
    if encoder_hidden_states is not None:
      kwargs['encoder_outputs'] = BaseModelOutput(
          last_hidden_state=encoder_hidden_states)
      kwargs['input_values'] = input_features
    else:
      kwargs['input_values'] = input_features
    if attention_mask is not None:
      kwargs['attention_mask'] = attention_mask
    lp = self._get_logits_processors()
    if lp:
      kwargs['logits_processor'] = lp
    return self.backbone.generate(**kwargs)

  def save_pretrained(self, path):
    self.backbone.save_pretrained(path)
    self.processor.save_pretrained(path)
    if self.use_ctc and hasattr(self, 'ctc_head'):
      import torch
      torch.save(self.ctc_head.state_dict(), f'{path}/ctc_head.pt')
    logger.info(f'Model & processor saved to {path}')
