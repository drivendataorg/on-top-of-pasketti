#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from collections import Counter

import torch


class TemperatureSampler(torch.utils.data.Sampler):
  """Train-only sampler that rebalances source exposure with temperature.

  Group probability follows p_i ∝ n_i^alpha, then each sample inside a group
  is drawn uniformly. Sampling is with replacement so epoch length can stay
  unchanged while source exposure changes.
  """

  def __init__(self,
               data_source,
               group_keys,
               alpha=0.5,
               seed=42,
               num_samples=None,
               distributed=False,
               num_replicas=None,
               rank=None):
    assert len(data_source) == len(group_keys), \
      f'group_keys size mismatch: len(dataset)={len(data_source)} len(group_keys)={len(group_keys)}'
    assert alpha >= 0, f'temperature alpha must be >= 0, got {alpha}'
    self.data_source = data_source
    self.group_keys = list(group_keys)
    self.alpha = float(alpha)
    self.seed = int(seed)
    self._epoch = 0
    self._num_consumed = 0

    self.group_counts = Counter(self.group_keys)
    assert self.group_counts, 'temperature sampler requires at least one group'
    self.group_probs = self._compute_group_probs()
    self.weights = torch.tensor(
      [self.group_probs[key] / self.group_counts[key] for key in self.group_keys],
      dtype=torch.double,
    )

    epoch_size = int(num_samples) if num_samples else len(self.data_source)
    assert epoch_size > 0, f'temperature sampler requires epoch_size > 0, got {epoch_size}'

    self.distributed = bool(distributed)
    if self.distributed:
      if num_replicas is None:
        assert torch.distributed.is_available() and torch.distributed.is_initialized(), \
          'distributed=True but torch.distributed is not initialized'
        num_replicas = torch.distributed.get_world_size()
      if rank is None:
        assert torch.distributed.is_available() and torch.distributed.is_initialized(), \
          'distributed=True but torch.distributed is not initialized'
        rank = torch.distributed.get_rank()
      self.num_replicas = int(num_replicas)
      self.rank = int(rank)
      self.num_samples = int(math.ceil(epoch_size / self.num_replicas))
      self.total_size = self.num_samples * self.num_replicas
    else:
      self.num_replicas = 1
      self.rank = 0
      self.num_samples = epoch_size
      self.total_size = self.num_samples

  def _compute_group_probs(self):
    scaled = {
      key: float(count) ** self.alpha
      for key, count in self.group_counts.items()
    }
    total = sum(scaled.values())
    assert total > 0, f'invalid temperature group weights: {scaled}'
    return {key: value / total for key, value in scaled.items()}

  def set_epoch(self, epoch):
    self._epoch = int(epoch)
    self._num_consumed = 0

  def set_consumed(self, n):
    self._num_consumed = int(n)

  def _sample_indices(self):
    generator = torch.Generator()
    generator.manual_seed(self.seed + self._epoch)
    sampled = torch.multinomial(
      self.weights,
      self.total_size,
      replacement=True,
      generator=generator,
    ).tolist()
    if self.distributed:
      sampled = sampled[self.rank:self.total_size:self.num_replicas]
    if self._num_consumed > 0:
      sampled = sampled[self._num_consumed:]
      self._num_consumed = 0
    return sampled

  def __iter__(self):
    return iter(self._sample_indices())

  def __len__(self):
    return self.num_samples

  def state_dict(self):
    return {
      'epoch': self._epoch,
      'seed': self.seed,
      'alpha': self.alpha,
      'num_samples': self.num_samples,
      'distributed': self.distributed,
    }

  def load_state_dict(self, state):
    if state is None:
      return
    self._epoch = int(state.get('epoch', self._epoch))
    self.seed = int(state.get('seed', self.seed))

  def describe(self):
    return {
      'alpha': self.alpha,
      'num_samples': self.num_samples,
      'groups': {
        key: {
          'count': self.group_counts[key],
          'prob': self.group_probs[key],
        }
        for key in sorted(self.group_counts)
      },
    }


def build_temperature_group_keys(df, mode='source'):
  """Build temperature groups from the training dataframe.

  mode:
    - source: dd / ext
    - label_type: labeled / word_only (based on empty primary label_text)
    - source_label: dd:labeled, ext:word_only, ...
    - otherwise treated as a dataframe column name
  """
  assert len(df) > 0, 'temperature sampler requires non-empty training dataframe'

  if mode == 'source':
    assert 'source' in df.columns, 'temperature_sampler_group=source requires source column'
    return df['source'].fillna('unknown').astype(str).tolist()

  if mode == 'label_type':
    assert 'label_text' in df.columns, 'temperature_sampler_group=label_type requires label_text column'
    labels = df['label_text'].fillna('').astype(str).str.strip()
    return ['word_only' if not label else 'labeled' for label in labels]

  if mode == 'source_label':
    assert 'source' in df.columns, 'temperature_sampler_group=source_label requires source column'
    assert 'label_text' in df.columns, 'temperature_sampler_group=source_label requires label_text column'
    sources = df['source'].fillna('unknown').astype(str).tolist()
    labels = df['label_text'].fillna('').astype(str).str.strip().tolist()
    return [f'{source}:{"word_only" if not label else "labeled"}'
            for source, label in zip(sources, labels)]

  assert mode in df.columns, \
    f'temperature_sampler_group={mode} requires dataframe column {mode}'
  return df[mode].fillna('__nan__').astype(str).tolist()