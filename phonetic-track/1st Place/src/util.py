#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2025-02-13
#   \Description   Shared utility helpers for Pasketti ASR
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from src.config import *


def get_model():
  """Instantiate the model specified by FLAGS.model."""
  import src.models
  Model = getattr(src.models, FLAGS.model).Model
  model = Model()
  return model
