#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2025-02-13
#   \Description   Phonetic Track (IPA CER) — imports shared config_base,
#                  overrides track-specific defaults only.
# ==============================================================================

from src.config_base import *
import src.config_base as config_base

MODEL_NAME = 'pasketti-phonetic'
RUN_VERSION = '9'


def init():
  FLAGS.track = 'phonetic'
  config_base.init(MODEL_NAME, run_version=RUN_VERSION)
  # phonetic-track data defaults
  FLAGS.root = FLAGS.root or '../input/childrens-phonetic-asr'
  FLAGS.train_file = FLAGS.train_file or 'train_phon_transcripts.jsonl'
  FLAGS.label_column = FLAGS.label_column or 'phonetic_text'
  FLAGS.label_column_fallback = FLAGS.label_column_fallback or 'ipa_text'
  FLAGS.score_metric = FLAGS.score_metric or 'ipa_cer'
  # StratifiedGroupKFold: group by child_id, stratify by age_bucket
  FLAGS.fold_group_key = FLAGS.fold_group_key or 'child_id'
  FLAGS.fold_stratify_key = FLAGS.fold_stratify_key or 'age_bucket'
  # max audio 21.38s in train, no need for 30s padding
  FLAGS.max_audio_sec = FLAGS.max_audio_sec or 22.0


def show():
  config_base.show()
  ic(FLAGS.ipa_method, FLAGS.use_cmudict_fallback, FLAGS.constrain_ipa)
