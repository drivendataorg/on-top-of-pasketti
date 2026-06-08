"""``gezi.common`` kitchen-sink — re-exports the symbols every project
file expects to receive from ``from gezi.common import *``.

Keeping this module in a fixed shape avoids touching the existing
``dataset.py`` / ``models/*.py`` / ``submit.py`` source files.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# --- stdlib ---------------------------------------------------------------
import sys
import os
import io
import re
import gc
import json
import math
import time
import copy
import glob
import gzip
import shutil
import pickle
import random
import warnings
import subprocess
import collections
import itertools
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Literal, Optional
from multiprocessing import Pool, Manager, cpu_count

# --- numerical / data ----------------------------------------------------
import numpy as np
import pandas as pd  # noqa: F401 — used by submit.py / ensemble.py
import scipy           # noqa: F401
import sklearn         # noqa: F401
from sklearn.preprocessing import normalize  # noqa: F401

# --- pytorch (heavily used by models / dataset) --------------------------
try:
  import torch                     # noqa: F401
  import torch.nn as nn            # noqa: F401
  import torch.nn.functional as F  # noqa: F401
except Exception:                  # pragma: no cover
  torch = None                     # type: ignore[assignment]
  nn = None                        # type: ignore[assignment]
  F = None                         # type: ignore[assignment]

# --- progress & logging --------------------------------------------------
from tqdm.auto import tqdm  # noqa: F401

# --- absl flags (the project uses ``FLAGS`` directly) --------------------
from absl import app, flags  # noqa: F401
FLAGS = flags.FLAGS


def _define_flag(kind, name, default, help_text='', **kwargs):
  """Define a framework flag once.

  The original project inherited many generic training flags from `melt`.
  The standalone repo only keeps a tiny melt shim, so we register the small
  common surface needed by the released flagfiles here.
  """
  if name in FLAGS:
    return
  getattr(flags, f'DEFINE_{kind}')(name, default, help_text, **kwargs)


def _define_alias(alias, target):
  if alias in FLAGS or target not in FLAGS:
    return
  flags.DEFINE_alias(alias, target)


# Generic training / framework flags that upstream `melt` used to provide.
_define_flag('string', 'mn', '', 'model name')
_define_flag('string', 'mns', '', 'model name suffix')
_define_flag('string', 'model_name_suffix', '', 'model name suffix')
_define_flag('string', 'model_dir', '', 'checkpoint/output directory')
_define_flag('string', 'model_name', '', 'resolved model name')
_define_flag('string', 'run_version', '', 'run version')
_define_flag('integer', 'fold', 0, 'fold index')
_define_flag('integer', 'folds', 0, 'number of folds')
_define_flag('integer', 'num_folds', 0, 'number of folds')
_define_flag('integer', 'fold_seed', None, 'fold split random seed')
_define_flag('bool', 'online', False, 'train on all data')
_define_flag('string', 'mode', 'train', 'train/eval/test mode')
_define_flag('string', 'work_mode', '', 'runtime work mode')
_define_flag('bool', 'train_only', False, 'skip validation/test paths')
_define_flag('bool', 'smoke', False, 'smoke-test mode')

_define_flag('integer', 'batch_size', None, 'training batch size')
_define_flag('integer', 'eval_batch_size', None, 'evaluation batch size')
_define_alias('bs', 'batch_size')
_define_alias('eval_bs', 'eval_batch_size')
_define_flag('integer', 'num_workers', 0, 'dataloader workers')
_define_flag('bool', 'persistent_workers', False, 'dataloader persistent workers')
_define_flag('bool', 'pin_memory', True, 'dataloader pin_memory')

_define_flag('float', 'learning_rate', None, 'base learning rate')
_define_flag('float', 'lr', None, 'base learning rate alias')
_define_flag('float', 'head_lr', None, 'head learning rate')
_define_flag('float', 'weight_decay', 0.01, 'optimizer weight decay')
_define_flag('string', 'optimizer', '', 'optimizer name')
_define_flag('string', 'scheduler', '', 'scheduler name')
_define_flag('float', 'warmup_proportion', 0.1, 'warmup ratio')
_define_alias('warmup_ratio', 'warmup_proportion')
_define_flag('float', 'warmup_epochs', 0.0, 'warmup epochs')
_define_flag('integer', 'ep', None, 'epochs')
_define_flag('float', 'exit_epoch', 0.0, 'stop after this many epochs; supports fractional epochs')
_define_flag('string', 'exit_epoch_action', 'exit', 'compatibility flag for upstream fractional-epoch runs')
_define_flag('integer', 'acc_steps', None, 'gradient accumulation steps')
_define_flag('integer', 'grad_acc', 1, 'gradient accumulation steps')
_define_flag('bool', 'gradc', False, 'enable gradient clipping')
_define_flag('float', 'grad_clip', None, 'gradient clipping norm')
_define_flag('bool', 'clip_grads', False, 'enable gradient clipping')

_define_flag('bool', 'fp16', None, 'use fp16 mixed precision')
_define_flag('bool', 'bfloat16', False, 'use bf16 mixed precision')
_define_flag('bool', 'amp_infer', False, 'use AMP for inference')
_define_flag('bool', 'torch', False, 'use torch model')
_define_flag('bool', 'torch_only', True, 'use torch-only input pipeline')
_define_flag('bool', 'gradient_checkpointing', False, 'enable gradient checkpointing')
_define_flag('bool', 'distributed', False, 'distributed training')
_define_flag('integer', 'num_gpus', 1, 'number of GPUs')
_define_flag('string', 'device', '', 'device type')

_define_flag('bool', 'ema_train', False, 'enable EMA training')
_define_flag('float', 'ema_start_epoch', None, 'EMA start epoch')
_define_flag('float', 'ema_decay', None, 'EMA decay')
_define_flag('bool', 'save_ema_init', False, 'save EMA init checkpoint')
_define_flag('bool', 'save_fp16', False, 'save fp16 checkpoint')
_define_flag('bool', 'save_best', None, 'save best checkpoint')
_define_flag('bool', 'save_best_only', False, 'save only best checkpoint')
_define_flag('bool', 'force_save', False, 'force save outputs')
_define_flag('bool', 'load_best', False, 'load best checkpoint')
_define_flag('bool', 'restore_configs', False, 'restore flags from checkpoint')
_define_flag('bool', 'restore_pretrain_configs', False, 'restore pretrain flags')
_define_flag('string', 'pretrain', '', 'pretrained model name/path')
_define_flag('integer', 'pretrain_restart', 0, 'restart from pretrain')
_define_flag('integer', 'pretrain_online', 0, 'load online pretrain')
_define_flag('string', 'cf', '', 'config/checkpoint alias')

_define_flag('bool', 'early_stop', False, 'enable early stopping')
_define_flag('integer', 'patience', 0, 'early stopping patience')
_define_flag('string', 'metric_key', '', 'primary metric key')
_define_flag('string', 'metric2', '', 'secondary metric key')
_define_flag('bool', 'greater_is_better', False, 'maximize metric')
_define_flag('float', 'vie', None, 'valid interval epochs')
_define_flag('float', 'sie', None, 'save interval epochs')

_define_flag('bool', 'wandb', None, 'enable wandb')
_define_flag('string', 'wandb_mode', '', 'wandb mode')
_define_flag('string', 'wandb_project', '', 'wandb project')
_define_flag('bool', 'write_summary', False, 'write training summary')

_define_flag('bool', 'fast_infer', False, 'fast inference path')
_define_flag('bool', 'pymp', False, 'parallel multiprocessing')
_define_flag('bool', 'train_allnew', False, 'train all-new model')

# Tree-model flags referenced by the ensemble code.
_define_flag('string', 'tree_model', 'cb', 'tree model type')
_define_flag('integer', 'trees', 500, 'number of trees')
_define_flag('integer', 'iters', 500, 'number of iterations')
_define_flag('float', 'tree_lr', 0.05, 'tree learning rate')
_define_flag('integer', 'max_depth', 6, 'tree max depth')
_define_flag('integer', 'num_leaves', 31, 'tree leaves')
_define_flag('float', 'reg_lambda', 5.0, 'tree regularization')
_define_flag('float', 'feature_fraction', 1.0, 'tree feature fraction')
_define_flag('float', 'tree_bagging', 1.0, 'tree bagging fraction')
_define_flag('string', 'objective', '', 'tree objective')
_define_flag('integer', 'tree_seed', 42, 'tree random seed')
_define_flag('integer', 'num_tree_threads', 0, 'tree threads')
_define_flag('integer', 'tree_metric_period', 0, 'tree metric period')
_define_flag('bool', 'tree_verbose', False, 'verbose tree training')
_define_flag('integer', 'tree_verbose_eval', 0, 'tree verbose eval')
_define_flag('bool', 'tree_fit', False, 'fit tree model')
_define_flag('bool', 'tree_eval_train', False, 'evaluate train split')
_define_flag('bool', 'tree_convert', False, 'convert tree model')
_define_flag('bool', 'tree_tb', False, 'tree TensorBoard output')
_define_flag('bool', 'use_best_model', False, 'tree use_best_model')

# --- our own shim packages re-exported under the historical aliases ------
import gezi  # this same package
import gezi as gz  # noqa: F401  — gz.* alias
from gezi import (  # noqa: F401
    logger,
    ic,
    ic_once,
    ic_nth,
    ico,
    icn,
    gic,
    dic,
    icl,
    rtqdm,
    Globals,
    Globals as GLBS,
    Timer,
    tree,
)
import melt           # noqa: F401
import melt as mt     # noqa: F401
try:
  import lele
  import lele as le   # noqa: F401
except Exception:     # pragma: no cover
  lele = None         # type: ignore[assignment]
  le = None           # type: ignore[assignment]
import husky          # noqa: F401  — placeholder

# Dotted attribute for ``gezi.common.PERCENTILES`` style lookups in the
# upstream code (rarely used here, but kept for safety).
PERCENTILES = [.25, .5, .75, .9, .95, .99]
SPECIAL_CHAR = 'ʶ'
SPECIAL_EN = '。'
FAIL = '❌'
PASS = '✅'
WARNING = '⚠️'
STAR = '⭐'
FIRE = '🔥'
TABLE = '📋'
CHART = '📊'
SEARCH = '🔍'

# Some modules expect a `set` callable; restore the builtin to avoid
# accidental shadowing by ``gezi.set``.
from builtins import set  # noqa: F401, A004
