"""Minimal `melt` (a.k.a. ``mt``) compatibility shim.

The upstream ``melt`` package is a heavyweight Keras-/PyTorch-Lightning style
training framework. The standalone solution does not need any of it — we
provide a hand-written training loop in ``src.train_loop`` instead — but a
handful of light helpers (``mt.init``, ``mt.set_global``, ``mt.epoch``) are
still referenced by the project source files. They are stubbed here so the
existing imports keep working untouched.
"""
from __future__ import annotations

from typing import Any

import gezi as _gz


def init(*args, **kwargs) -> None:
  """No-op replacement for ``melt.init`` (was distributed/wandb setup)."""
  return None


def set_global(key: str, value: Any) -> Any:
  """Mirror upstream ``melt.set_global`` — alias for ``gezi.set``."""
  return _gz.set(key, value)


def get_global(key: str, default: Any = None) -> Any:
  return _gz.get(key, default)


def epoch() -> float:
  """Current training epoch as a float (set by the train loop)."""
  return float(_gz.get('epoch', 0.0))


def fit(*args, **kwargs):  # pragma: no cover
  """Not used in the standalone repo — see ``src.train_loop.train`` instead."""
  raise RuntimeError(
      'melt.fit is not available in the standalone solution; '
      'call src.train_loop.train(...) directly.'
  )
