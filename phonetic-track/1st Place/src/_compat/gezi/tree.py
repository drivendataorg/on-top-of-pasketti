from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from . import FLAGS, try_mkdir


def _group_sizes_to_ids(group_sizes):
  if group_sizes is None:
    return None
  group_ids = []
  gid = 0
  for size in group_sizes:
    size = int(size)
    if size <= 0:
      continue
    group_ids.extend([gid] * size)
    gid += 1
  return group_ids


def _catboost_task_type() -> str:
  device = str(getattr(FLAGS, 'device', '') or '').lower()
  return 'GPU' if device == 'gpu' else 'CPU'


def _default_params(task_type: str) -> dict:
  if task_type == 'ranking':
    loss_function = getattr(FLAGS, 'objective', None) or 'YetiRank'
  elif task_type == 'regression':
    loss_function = getattr(FLAGS, 'objective', None) or 'RMSE'
  else:
    loss_function = getattr(FLAGS, 'objective', None) or 'Logloss'

  params = {
      'iterations': int(getattr(FLAGS, 'iters', 0) or getattr(FLAGS, 'trees', 0) or 500),
      'learning_rate': float(getattr(FLAGS, 'tree_lr', 0.05) or 0.05),
      'random_seed': int(getattr(FLAGS, 'tree_seed', 42) or 42),
      'loss_function': loss_function,
      'task_type': _catboost_task_type(),
      'use_best_model': bool(getattr(FLAGS, 'use_best_model', True)),
      'allow_writing_files': False,
  }

  model_dir = getattr(FLAGS, 'model_dir', None)
  if model_dir:
    train_dir = Path(model_dir) / 'catboost_info'
  else:
    train_dir = Path('../working/reranker/catboost_info')
  try_mkdir(train_dir)
  params['train_dir'] = str(train_dir)

  metric_period = int(getattr(FLAGS, 'tree_metric_period', 0) or 0)
  if metric_period > 0:
    params['metric_period'] = metric_period

  num_tree_threads = int(getattr(FLAGS, 'num_tree_threads', 0) or 0)
  if num_tree_threads > 0:
    params['thread_count'] = num_tree_threads

  max_depth = int(getattr(FLAGS, 'max_depth', 0) or 0)
  if max_depth > 0:
    params['depth'] = max_depth

  num_leaves = int(getattr(FLAGS, 'num_leaves', 0) or 0)
  if num_leaves > 0:
    params['grow_policy'] = 'Lossguide'
    params['num_leaves'] = num_leaves

  reg_lambda = float(getattr(FLAGS, 'reg_lambda', 0.0) or 0.0)
  if reg_lambda > 0:
    params['l2_leaf_reg'] = reg_lambda

  tree_bagging = float(getattr(FLAGS, 'tree_bagging', 0.0) or 0.0)
  if tree_bagging > 0:
    params['bagging_temperature'] = tree_bagging

  feature_fraction = float(getattr(FLAGS, 'feature_fraction', 0.0) or 0.0)
  if feature_fraction > 0 and params['task_type'] == 'CPU':
    params['rsm'] = feature_fraction

  tree_verbose_eval = getattr(FLAGS, 'tree_verbose_eval', 0)
  if tree_verbose_eval:
    params['verbose'] = int(tree_verbose_eval)
  else:
    params['verbose'] = False

  return params


def create_model(params=None, task_type: str = 'regression'):
  tree_model = getattr(FLAGS, 'tree_model', 'cb') or 'cb'
  if tree_model != 'cb':
    raise NotImplementedError(
        f'_compat.gezi.tree currently supports only CatBoost for the standalone release; got tree_model={tree_model!r}'
    )

  from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor

  merged = _default_params(task_type)
  if params:
    merged.update(params)

  if task_type == 'ranking':
    return CatBoostRanker(**merged)
  if task_type == 'regression':
    return CatBoostRegressor(**merged)
  if task_type == 'classification':
    return CatBoostClassifier(**merged)
  raise ValueError(f'Unsupported task_type: {task_type}')


class Model:
  def __init__(self, model=None, params=None, task_type: str = 'regression', **kwargs):
    del kwargs
    self.task_type = task_type
    self.model_type = getattr(FLAGS, 'tree_model', 'cb') or 'cb'
    self.cat_features = []
    self.model = model or create_model(params=params, task_type=task_type)

  def fit(self, df, y, **kwargs):
    fit(self, df, y, **kwargs)
    return self

  def predict(self, df, **kwargs):
    if not isinstance(df, pd.DataFrame):
      df = pd.DataFrame(df)
    return self.model.predict(df, **kwargs)

  def save(self, dest):
    dest = Path(dest)
    if dest.suffix:
      dest.parent.mkdir(parents=True, exist_ok=True)
      if dest.suffix in {'.pkl', '.pickle'}:
        with open(dest, 'wb') as f:
          pickle.dump(self.model, f)
      else:
        self.model.save_model(str(dest))
      return self

    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / 'model.pkl', 'wb') as f:
      pickle.dump(self.model, f)
    return self

  def load(self, source):
    source = Path(source)
    model_pkl = source / 'model.pkl' if source.is_dir() else source
    model_json = source / 'model.json' if source.is_dir() else None
    if model_pkl.exists():
      with open(model_pkl, 'rb') as f:
        self.model = pickle.load(f)
      return True
    if model_json is not None and model_json.exists():
      self.model = create_model(task_type=self.task_type)
      self.model.load_model(str(model_json))
      return True
    return False


def fit(model,
        X_train,
        y_train,
        X_valid=None,
        y_valid=None,
        weight=None,
        cat_features=None,
        callbacks=None,
        **kwargs):
  del callbacks
  tree_model = getattr(FLAGS, 'tree_model', 'cb') or 'cb'
  if tree_model != 'cb':
    raise NotImplementedError(
        f'_compat.gezi.tree currently supports only CatBoost for the standalone release; got tree_model={tree_model!r}'
    )

  from catboost import Pool

  if not isinstance(X_train, pd.DataFrame):
    X_train = pd.DataFrame(X_train)
  if X_valid is not None and not isinstance(X_valid, pd.DataFrame):
    X_valid = pd.DataFrame(X_valid)

  model.cat_features = cat_features or []
  group_id_train = _group_sizes_to_ids(kwargs.get('group_train'))
  group_id_valid = _group_sizes_to_ids(kwargs.get('group_valid'))

  train_pool = Pool(
      X_train,
      y_train,
      weight=weight,
      group_id=group_id_train,
      cat_features=model.cat_features or None,
  )

  valid_pool = None
  if X_valid is not None:
    valid_pool = Pool(
        X_valid,
        y_valid,
        group_id=group_id_valid,
        cat_features=model.cat_features or None,
    )

  fit_kwargs = {}
  early_stop = int(getattr(FLAGS, 'early_stop', 0) or 0)
  if early_stop > 0:
    fit_kwargs['early_stopping_rounds'] = early_stop

  if valid_pool is not None:
    fit_kwargs['eval_set'] = valid_pool

  model.model.fit(train_pool, **fit_kwargs)
  return model
