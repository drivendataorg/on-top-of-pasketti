#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   squeezeformer.py
#        \author   chenghuige
#          \date   2025-02-16
#   \Description   SqueezeFormer encoder for Pasketti ASR.
#                  Uses lele.layers.Squeezeformer as the audio encoder.
#                  ctc_weight controls loss mix (0=s2s, 0~1=hybrid, 1=ctc).
#                  When ctc_only, no Whisper decoder is loaded (lightweight).
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
from src.config import *
from src.models.base import BaseASRModel

from lele.layers import Squeezeformer


class Model(BaseASRModel):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    # ---- Whisper config (for mel dim & d_model) ----
    from transformers import WhisperConfig
    backbone_name = BACKBONES.get(FLAGS.backbone, FLAGS.backbone)
    whisper_config = WhisperConfig.from_pretrained(backbone_name)
    mel_dim = whisper_config.num_mel_bins
    d_model = whisper_config.d_model

    # ---- SqueezeFormer encoder ----
    sf_dim = getattr(FLAGS, 'sf_dim', 256)
    sf_depth = getattr(FLAGS, 'sf_depth', 12)
    sf_heads = getattr(FLAGS, 'sf_heads', 4)
    sf_ff_mult = getattr(FLAGS, 'sf_ff_mult', 4)
    sf_conv_kernel = getattr(FLAGS, 'sf_conv_kernel', 31)
    self.encoder = Squeezeformer(
        in_dim=mel_dim,
        dim=sf_dim,
        depth=sf_depth,
        heads=sf_heads,
        ff_mult=sf_ff_mult,
        conv_kernel_size=sf_conv_kernel,
        attn_dropout=getattr(FLAGS, 'sf_attn_dropout', 0.1),
        ff_dropout=getattr(FLAGS, 'sf_ff_dropout', 0.1),
        conv_dropout=getattr(FLAGS, 'sf_conv_dropout', 0.1),
    )
    self.encoder_dim = sf_dim
    logger.info(f'SqueezeFormer encoder: mel_dim={mel_dim}, '
                 f'dim={sf_dim}, depth={sf_depth}, heads={sf_heads}')

    # ---- CTC head ----
    self._init_ctc_head(sf_dim)

    # ---- Auxiliary metadata heads (age / domain) ----
    self._init_aux_heads(sf_dim)

    # ---- Whisper decoder (only when seq2seq path is needed) ----
    self.backbone = None
    if not self.ctc_only:
      from transformers import WhisperForConditionalGeneration
      try:
        self.backbone = WhisperForConditionalGeneration.from_pretrained(
            FLAGS.model_dir, trust_remote_code=True)
        logger.info(f'Whisper decoder loaded from {FLAGS.model_dir}')
      except Exception:
        self.backbone = WhisperForConditionalGeneration.from_pretrained(
            backbone_name, trust_remote_code=True)
        logger.info(f'Whisper decoder loaded from {backbone_name}')

      # Freeze Whisper encoder (we use SqueezeFormer instead)
      for param in self.backbone.model.encoder.parameters():
        param.requires_grad = False
      logger.info('Whisper encoder frozen (using SqueezeFormer encoder)')

      # Projection: sf_dim -> d_model
      if sf_dim != d_model:
        self.enc_proj = nn.Linear(sf_dim, d_model)
      else:
        self.enc_proj = nn.Identity()

      self.backbone.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
          language=FLAGS.language, task=FLAGS.task)
      self.backbone.config.suppress_tokens = []

      if FLAGS.gradient_checkpointing:
        self.backbone.gradient_checkpointing_enable()

    self._log_params()

  def _encode(self, input_features, attention_mask=None):
    # input_features: (B, n_mels, T) from Whisper processor
    x = input_features.transpose(1, 2)  # (B, T, n_mels)
    return self.encoder(x)  # (B, T', sf_dim)

  def _s2s_forward(self, input_features, labels, encoder_hidden_states,
                   attention_mask=None):
    enc_proj = self.enc_proj(encoder_hidden_states)  # (B, T', d_model)
    if self.constrain_ipa:
      outputs = self.backbone(
          input_features=input_features,
          labels=labels,
          encoder_outputs=(enc_proj,),
      )
      return self._compute_s2s_loss(outputs.logits, labels)
    return self.backbone(
        input_features=input_features,
        labels=labels,
        encoder_outputs=(enc_proj,),
    ).loss

  def _s2s_generate(self, input_features, encoder_hidden_states,
                    attention_mask=None):
    enc_proj = self.enc_proj(encoder_hidden_states)
    encoder_output = type('EO', (), {'last_hidden_state': enc_proj})()
    kwargs = dict(
        encoder_outputs=encoder_output,
        max_new_tokens=FLAGS.max_new_tokens,
        num_beams=FLAGS.num_beams,
        length_penalty=FLAGS.length_penalty,
        language=FLAGS.language,
        task=FLAGS.task,
    )
    lp = self._get_logits_processors()
    if lp:
      kwargs['logits_processor'] = lp
    return self.backbone.generate(**kwargs)

  def save_pretrained(self, path):
    if self.backbone is not None:
      self.backbone.save_pretrained(path)
    self.processor.save_pretrained(path)
    torch.save(self.encoder.state_dict(), f'{path}/squeezeformer_encoder.pt')
    if self.use_ctc and hasattr(self, 'ctc_head'):
      torch.save(self.ctc_head.state_dict(), f'{path}/ctc_head.pt')
    if hasattr(self, 'enc_proj') and not isinstance(self.enc_proj, nn.Identity):
      torch.save(self.enc_proj.state_dict(), f'{path}/enc_proj.pt')
    sf_dim = getattr(FLAGS, 'sf_dim', 256)
    self.save_model_meta(path, 'squeezeformer_ctc', sf_dim,
                         sf_dim=sf_dim,
                         sf_depth=getattr(FLAGS, 'sf_depth', 12),
                         sf_heads=getattr(FLAGS, 'sf_heads', 4),
                         sf_ff_mult=getattr(FLAGS, 'sf_ff_mult', 4),
                         sf_conv_kernel=getattr(FLAGS, 'sf_conv_kernel', 31))
    logger.info(f'Model saved to {path}')
