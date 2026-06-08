"""Minimal `gezi` compatibility shim for the standalone solution repo.

This package replaces the much larger internal `gezi` toolkit so that the
core ASR project files (dataset.py, models/, submit.py, ensemble.py, ...)
can be released as open source without bundling the proprietary parent
library. Only the API surface actually used by this project is implemented.

API summary (mirrors the upstream gezi names):
  - Globals dict:        ``set(key, val)`` / ``get(key, default=None)``
  - Timer:               ``Timer().elapsed_minutes()``
  - Filesystem helpers:  ``try_mkdir``, ``try_create``
  - Project / FLAGS:     ``FLAGS``, ``init_flags``, ``restore_configs``,
                        ``prepare_project``, ``save_globals``,
                        ``set_fold``, ``batch_size``, ``eval_batch_size``
  - Model I/O:           ``load_weights``, ``save_model``
  - Misc:                ``tree``, ``ic_once``, ``ic_nth``,
                        ``oof_metrics`` (no-op stub)


The module is intentionally tiny — under ~250 lines — and has no
hidden side-effects beyond installing ``loguru`` as the default logger
and ``icecream`` as ``ic``.
"""
from __future__ import annotations

import os
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch

# --------------------------------------------------------------------------
# absl FLAGS re-export (some modules do ``from gezi import FLAGS``).
# --------------------------------------------------------------------------
from absl import flags as _absl_flags
FLAGS = _absl_flags.FLAGS

# --------------------------------------------------------------------------
# Logging: prefer loguru if available, otherwise stdlib.
# --------------------------------------------------------------------------
try:
  from loguru import logger  # noqa: F401
except Exception:  # pragma: no cover
  import logging as _logging
  logger = _logging.getLogger('gezi')
  if not logger.handlers:
    _h = _logging.StreamHandler()
    _h.setFormatter(_logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(_h)
    logger.setLevel(_logging.INFO)

# Compat sub-module ``gezi.logging`` so ``from gezi import logging`` works.
import sys as _sys
import types as _types
_logging_mod = _types.ModuleType('gezi.logging')
_logging_mod.logger = logger
_sys.modules.setdefault('gezi.logging', _logging_mod)

# --------------------------------------------------------------------------
# icecream debug print — fall back to a tiny shim if not installed.
# --------------------------------------------------------------------------
try:
  from icecream import ic  # noqa: F401
except Exception:  # pragma: no cover
  class _IC:
    enabled = True
    def __call__(self, *args):
      if not self.enabled:
        return args[0] if args else None
      logger.info('ic | ' + ' '.join(repr(a) for a in args))
      return args[0] if args else None
    def disable(self):
      self.enabled = False
    def enable(self):
      self.enabled = True
  ic = _IC()


def _ic_factory():
  """Return a no-op ``ic_once``/``ic_nth`` style helper."""
  seen = {}
  def f(*args, key=None, n=0):
    k = key if key is not None else repr(args)
    seen[k] = seen.get(k, 0) + 1
    if seen[k] == 1 or (n and seen[k] % n == 0):
      return ic(*args)
    return args[0] if args else None
  return f


ic_once = _ic_factory()
ic_nth = _ic_factory()
ico = ic_once
icn = ic_nth
gic = ic
dic = ic
icl = ic


# --------------------------------------------------------------------------
# Global key-value store (originally ``gezi.Globals``).
# --------------------------------------------------------------------------
class Globals(dict):
  """Tiny global key/value bag used to share state across modules."""

  _singleton: 'Globals | None' = None

  @classmethod
  def instance(cls) -> 'Globals':
    if cls._singleton is None:
      cls._singleton = cls()
    return cls._singleton


def set(key: str, value: Any) -> Any:  # noqa: A001 (mirror upstream API)
  Globals.instance()[key] = value
  return value


def get(key: str, default: Any = None) -> Any:
  return Globals.instance().get(key, default)


# --------------------------------------------------------------------------
# Timer
# --------------------------------------------------------------------------
class Timer:
  def __init__(self):
    self.t0 = time.time()

  def elapsed(self) -> float:
    return time.time() - self.t0

  def elapsed_minutes(self) -> float:
    return self.elapsed() / 60.0


# --------------------------------------------------------------------------
# Filesystem helpers
# --------------------------------------------------------------------------
def try_mkdir(path: str | os.PathLike) -> None:
  Path(path).mkdir(parents=True, exist_ok=True)


def try_create(path: str | os.PathLike) -> None:
  p = Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  p.touch(exist_ok=True)


# --------------------------------------------------------------------------
# Tree (auto-vivifying nested dict).
# --------------------------------------------------------------------------
def tree() -> defaultdict:
  return defaultdict(tree)


# --------------------------------------------------------------------------
# FLAGS / config plumbing
# --------------------------------------------------------------------------
def init_flags(argv: Optional[list[str]] = None) -> list[str]:
  """Parse ``--flagfile`` and command-line flags into the project FLAGS.

  This shim delegates to ``absl.flags`` (the same parser the upstream code
  used) so all existing flag definitions in ``config_base.py`` and
  ``config.py`` continue to work unchanged.
  """
  from absl import flags as _flags
  argv = argv if argv is not None else _sys.argv
  remaining = _flags.FLAGS(list(argv), known_only=True)
  return list(remaining)


def restore_configs(model_dir: str | os.PathLike, flag_overrides: Optional[dict] = None) -> dict:
  """Load ``flags.json`` from a trained model directory and apply to FLAGS.

  Mirrors ``gezi.restore_configs`` — used by submit.py / ensemble.py to
  re-hydrate per-model training-time FLAGS at inference time.
  """
  from absl import flags as _flags
  flags_path = Path(model_dir) / 'flags.json'
  if not flags_path.exists():
    return {}
  data = json.loads(flags_path.read_text())
  if flag_overrides:
    data.update(flag_overrides)
  for k, v in data.items():
    if k in _flags.FLAGS:
      try:
        setattr(_flags.FLAGS, k, v)
      except Exception:
        pass
  return data


def save_globals(model_dir: str | os.PathLike) -> None:
  """Persist current FLAGS to ``<model_dir>/flags.json``."""
  from absl import flags as _flags
  out = {}
  for name in _flags.FLAGS:
    try:
      out[name] = getattr(_flags.FLAGS, name)
    except Exception:
      continue
  Path(model_dir).mkdir(parents=True, exist_ok=True)
  with open(Path(model_dir) / 'flags.json', 'w') as f:
    json.dump(out, f, default=str, indent=2)


def prepare_project(model_dir: str | os.PathLike, model_name: str = '') -> None:
  Path(model_dir).mkdir(parents=True, exist_ok=True)
  if model_name:
    (Path(model_dir) / 'MODEL_NAME').write_text(model_name)


def set_fold(df_or_fold, folds=4, group_key=None, stratify_key=None,
             force_sklearn=False, seed=1024, name='fold',
             sgkf_compat=None) -> Any:
  """Assign CV folds to a DataFrame, or store the active fold.

  Upstream ``gezi.set_fold`` supports both usages.  The training pipeline uses
  the DataFrame form during preprocessing; a few older call sites pass just an
  integer fold to update global state.
  """
  del sgkf_compat  # sklearn's splitter is sufficient for this public shim.
  if not hasattr(df_or_fold, 'columns'):
    set('fold', int(df_or_fold))
    return None

  df = df_or_fold
  assert folds, 'folds must be positive'
  seed = int(seed or 1024)

  def _group_values(frame, key):
    if key is None or key == '':
      return None
    if isinstance(key, (list, tuple)):
      return frame[list(key)].astype(str).agg('_'.join, axis=1).values
    return frame[key].values

  if stratify_key in (None, ''):
    if group_key not in (None, ''):
      groups = _group_values(df, group_key)
      if force_sklearn:
        from sklearn.model_selection import GroupKFold
        splitter = GroupKFold(n_splits=folds)
        splits = splitter.split(df, groups=groups)
        fold_values = np.zeros(len(df), dtype=int)
        for fold_idx, (_, val_idx) in enumerate(splits):
          fold_values[val_idx] = fold_idx
        df[name] = fold_values
      else:
        rng = np.random.default_rng(seed)
        unique_groups = np.asarray(pd.Series(groups).drop_duplicates().tolist())
        order = np.arange(len(unique_groups))
        rng.shuffle(order)
        chunks = np.array_split(order, folds)
        group2fold = {}
        for fold_idx, chunk in enumerate(chunks):
          for idx in chunk:
            group2fold[unique_groups[idx]] = fold_idx
        df[name] = pd.Series(groups, index=df.index).map(group2fold).astype(int)
    else:
      rng = np.random.default_rng(seed)
      order = np.arange(len(df))
      rng.shuffle(order)
      fold_values = np.zeros(len(df), dtype=int)
      for fold_idx, chunk in enumerate(np.array_split(order, folds)):
        fold_values[chunk] = fold_idx
      df[name] = fold_values
  else:
    y = df[stratify_key].astype(str).values
    fold_values = np.zeros(len(df), dtype=int)
    if group_key in (None, ''):
      from sklearn.model_selection import StratifiedKFold
      splitter = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
      splits = splitter.split(df, y)
    else:
      from sklearn.model_selection import StratifiedGroupKFold
      groups = _group_values(df, group_key)
      splitter = StratifiedGroupKFold(n_splits=folds, random_state=seed, shuffle=True)
      splits = splitter.split(df, y, groups)
    for fold_idx, (_, val_idx) in enumerate(splits):
      fold_values[val_idx] = fold_idx
    df[name] = fold_values

  return df


def init_wandb(model_name: str, wandb: bool = False) -> None:
  """Stub: the standalone release does not use Weights & Biases."""
  from absl import flags as _flags
  if getattr(_flags.FLAGS, 'wandb', None) is None:
    try:
      _flags.FLAGS.wandb = False
    except Exception:
      pass
  try:
    _flags.FLAGS.wandb_project = model_name
  except Exception:
    pass


def init_modeldir(ignores: Optional[list] = None,
                  run_version: Optional[str] = None,
                  suffix: str = '') -> None:
  """Resolve ``FLAGS.model_dir`` from FLAGS / env (simplified port).

  Layout matches the upstream convention so the same checkpoints can be
  reused by the inference pipeline:

      ../working/{offline|online}/{run_version}/{model_name}/{fold}/

  Skipped from the upstream version (not needed for reproduction):
      * automatic ``model_name_from_args`` (unique CLI suffix detection) —
        we just trust ``--mn``;
      * staged training (``--stage>1``) and ``--mns`` extra suffix;
      * ``new_structure`` toggle.
  """
  del ignores  # not used in the simplified flow
  from absl import flags as _flags
  F = _flags.FLAGS
  pre = 'online' if getattr(F, 'online', False) else 'offline'
  run_v = getattr(F, 'run_version', None) or run_version or '1'
  try:
    F.run_version = run_v
  except Exception:
    pass
  mn = getattr(F, 'mn', None) or 'model'
  if suffix:
    mn = f'{mn}{suffix}'
  fold = getattr(F, 'fold', 0) or 0
  if not getattr(F, 'model_dir', None):
    try:
      F.model_dir = f'../working/{pre}/{run_v}/{mn}/{fold}'
    except Exception:
      pass
  Path(F.model_dir).mkdir(parents=True, exist_ok=True)
  set('model_dir', F.model_dir)
  set('model_name', mn)


def dict_prefix(d: dict, prefix: str) -> dict:
  return {f'{prefix}{k}': v for k, v in d.items()}


# --------------------------------------------------------------------------
# Convenience accessors for batch_size / eval_batch_size
# (called as ``gz.batch_size()`` / ``gz.eval_batch_size()`` in dataset.py)
# --------------------------------------------------------------------------
def batch_size() -> int:
  from absl import flags as _flags
  return int(getattr(_flags.FLAGS, 'batch_size', 32) or 32)


def eval_batch_size() -> int:
  from absl import flags as _flags
  ebs = getattr(_flags.FLAGS, 'eval_batch_size', None)
  if ebs:
    return int(ebs)
  return batch_size()


# --------------------------------------------------------------------------
# Model I/O
# --------------------------------------------------------------------------
def load_weights(model: torch.nn.Module, path: str | os.PathLike, strict: bool = False,
                 map_location: str | torch.device = 'cpu') -> torch.nn.Module:
  """Load a state-dict (``.pt``) into ``model`` with relaxed key matching.

  Implements the same loose loading semantics as the upstream
  ``gezi.load_weights`` / ``lele.load_weights``: missing or extra keys
  are logged but not fatal, and weights with mismatched shapes are
  skipped instead of raising.
  """
  state = torch.load(str(path), map_location=map_location)
  if isinstance(state, dict) and 'state_dict' in state:
    state = state['state_dict']
  if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
    state = state['model']

  model_sd = model.state_dict()
  filtered: dict[str, torch.Tensor] = {}
  shape_skipped, missing = 0, 0
  for k, v in state.items():
    k2 = k[len('module.'):] if k.startswith('module.') else k
    if k2 in model_sd:
      if model_sd[k2].shape == v.shape:
        filtered[k2] = v
      else:
        shape_skipped += 1
    else:
      missing += 1
  result = model.load_state_dict(filtered, strict=False)
  logger.info(
      f'load_weights({path}): loaded={len(filtered)}, '
      f'shape_skipped={shape_skipped}, unknown_keys={missing}, '
      f'model_missing={len(result.missing_keys)}'
  )
  return model


def save_model(model: torch.nn.Module, model_dir: str | os.PathLike, fp16: bool = False) -> None:
  Path(model_dir).mkdir(parents=True, exist_ok=True)
  state = model.state_dict()
  if fp16:
    state = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}
  torch.save(state, Path(model_dir) / 'model.pt')


# --------------------------------------------------------------------------
# Stubs for rarely-used helpers — present so imports don't break.
# --------------------------------------------------------------------------
def oof_metrics(*args, **kwargs):  # noqa: D401 — stub
  """No-op: full OOF aggregation is out of scope for the public release."""
  logger.warning('gezi.oof_metrics is a no-op in the standalone release')


def kaggle_metric_utilities(*args, **kwargs):  # pragma: no cover
  raise NotImplementedError


# `from gezi import rtqdm` — alias to tqdm.auto.
try:
  from tqdm.auto import tqdm as rtqdm  # noqa: F401
except Exception:  # pragma: no cover
  rtqdm = None  # type: ignore[assignment]
