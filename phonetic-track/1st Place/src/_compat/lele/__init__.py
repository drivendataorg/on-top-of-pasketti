"""Minimal `lele` (a.k.a. ``le``) compatibility shim.

Only the subset of helpers actually consumed by the project source files is
implemented:

  - ``get_opt_params`` / ``get_optimizer_params``: build optimizer parameter
    groups with optional split-by-backbone learning rates and weight decay.
  - ``get_sampler``: simple shuffling sampler (DDP not supported in the
    public release; we always run single-GPU training here).
  - ``BucketBatchSampler``: length-bucketed batch sampler used by
    ``dataset.py`` to keep audio durations within a batch close together.
  - ``update_scalars``: no-op (the upstream version normalised metric
    dictionaries for the framework's logger; here our train loop logs
    directly).
  - ``load_weights``: alias to ``gezi.load_weights``.
"""
from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

import gezi as _gz


# --------------------------------------------------------------------------
# Optimizer parameter grouping (ported verbatim from lele.util)
# --------------------------------------------------------------------------
def get_optimizer_params(model: torch.nn.Module,
                          backbone_lr=None,
                          base_lr=None,
                          weight_decay: bool = True,
                          weight_decay_val: float = 0.01,
                          backbone: torch.nn.Module | None = None):
  """Return a list of param-group dicts for ``torch.optim.AdamW`` etc.

  Behaviour mirrors the upstream implementation:
    * Parameters with ``ndim >= 2`` get ``weight_decay_val`` (default 0.01).
    * Bias / 1-D params get weight decay 0.
    * If both ``backbone_lr`` and ``base_lr`` are provided, backbone params
      get ``backbone_lr`` and head params get ``base_lr``.
  """
  if backbone is None:
    backbone = getattr(model, 'backbone', None)

  param_optimizer = list(model.named_parameters())
  no_decay_ids = {id(p) for p in model.parameters() if p.ndim < 2}
  backbone_ids = (
      {id(p) for p in backbone.parameters()} if backbone is not None else None
  )

  def _split_lr_group(want_decay: bool, want_backbone: bool | None):
    out = []
    for _, p in param_optimizer:
      is_decay = id(p) not in no_decay_ids
      if is_decay != want_decay:
        continue
      if want_backbone is not None and backbone_ids is not None:
        is_bb = id(p) in backbone_ids
        if is_bb != want_backbone:
          continue
      out.append(p)
    return out

  if not weight_decay:
    if backbone_lr is not None and base_lr is not None and backbone_ids is not None:
      return [
          {'params': [p for p in model.parameters() if id(p) in backbone_ids], 'lr': backbone_lr},
          {'params': [p for p in model.parameters() if id(p) not in backbone_ids], 'lr': base_lr},
      ]
    return list(model.parameters())

  if backbone_lr is None or base_lr is None or backbone_ids is None:
    return [
        {'params': _split_lr_group(True, None),  'weight_decay': weight_decay_val},
        {'params': _split_lr_group(False, None), 'weight_decay': 0.0},
    ]

  return [
      {'params': _split_lr_group(True,  True),  'weight_decay': weight_decay_val, 'lr': backbone_lr},
      {'params': _split_lr_group(True,  False), 'weight_decay': weight_decay_val, 'lr': base_lr},
      {'params': _split_lr_group(False, True),  'weight_decay': 0.0,              'lr': backbone_lr},
      {'params': _split_lr_group(False, False), 'weight_decay': 0.0,              'lr': base_lr},
  ]


get_opt_params = get_optimizer_params


# --------------------------------------------------------------------------
# Samplers
# --------------------------------------------------------------------------
class _ShuffleSampler(Sampler):
  def __init__(self, data_source, seed: int = 42):
    self.n = len(data_source)
    self.seed = seed
    self._epoch = 0

  def set_epoch(self, epoch: int) -> None:
    self._epoch = epoch

  def __iter__(self) -> Iterator[int]:
    g = torch.Generator()
    g.manual_seed(self.seed + self._epoch)
    return iter(torch.randperm(self.n, generator=g).tolist())

  def __len__(self) -> int:
    return self.n


def get_sampler(dataset, shuffle: bool = False, seed: int | None = None):
  """Return a per-epoch shuffling sampler (single-GPU only in this release)."""
  if not shuffle:
    return None
  return _ShuffleSampler(dataset, seed=seed if seed is not None else 42)


class BucketBatchSampler(Sampler[Iterable[int]]):
  """Length-bucketed batch sampler (ported from ``lele.util``).

  Items with similar lengths end up in the same batch, drastically reducing
  zero-padding for variable-length audio.
  """

  def __init__(self,
               lens: Sequence[int],
               batch_size: int,
               drop_last: bool = False,
               shuffle: bool = True,
               max_shift: float = 0.0):
    self.lens = list(lens)
    self.batch_size = batch_size
    self.drop_last = drop_last
    self.shuffle = shuffle
    self.max_shift = max_shift
    self.reset()

  def reset(self) -> None:
    lens = self.lens
    if self.max_shift > 0:
      lens = [x + np.random.uniform(-self.max_shift, self.max_shift) for x in lens]
    indices = np.argsort(lens)
    self._has_partial = len(indices) % self.batch_size > 0
    if self._has_partial:
      pad = self.batch_size - len(indices) % self.batch_size
      indices = np.append(indices, [-1] * pad)
    self.buckets = indices.reshape(-1, self.batch_size)
    if self.shuffle:
      self.permutation = np.random.permutation(self.buckets.shape[0])
    else:
      self.permutation = np.arange(self.buckets.shape[0])

  def __len__(self) -> int:
    return self.buckets.shape[0] - (int(self._has_partial) if self.drop_last else 0)

  def __iter__(self) -> Iterator[List[int]]:
    for batch in self.buckets[self.permutation]:
      batch = batch[batch >= 0]
      if self.shuffle:
        np.random.shuffle(batch)
      if len(batch) > 0:
        yield batch.tolist()
    if self.shuffle:
      self.reset()


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
def update_scalars(scalars, decay=None, training=None):  # noqa: D401
  """No-op stub. The upstream helper rewrote keys with a ``val_`` prefix
  for the framework's metric logger; the standalone train loop logs as-is."""
  return scalars


def load_weights(*args, **kwargs):
  """Alias to :func:`gezi.load_weights` — see that function for semantics."""
  return _gz.load_weights(*args, **kwargs)
