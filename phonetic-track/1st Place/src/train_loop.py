"""Hand-written training loop for the standalone Pasketti Phonetic solution.

Replaces the much larger ``melt.fit`` framework used in the development
repository. Intentionally minimal — only what is required to reproduce
the published submission:

  * Mixed precision (AMP) forward / backward
  * Optional gradient accumulation
  * Optional gradient clipping
  * Linear warm-up + cosine learning-rate schedule (the schedule the
    upstream framework also defaulted to)
  * Optional EMA over model parameters with a constant decay
  * Single final validation pass after the last epoch — no per-epoch
    eval, no callbacks, no checkpoint averaging.

The loop deliberately matches the shape of one epoch in ``melt.fit``:
``len(train_dl)`` mini-batches per epoch, ``epochs`` total epochs, with
each ``optimizer.step`` advancing the LR scheduler one tick.
"""
from __future__ import annotations

import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

import gezi as gz
from gezi import logger

import lele as le


# ---------------------------------------------------------------------------
# Optimizer + scheduler
# ---------------------------------------------------------------------------
def _cosine_warmup_schedule(num_warmup: int, num_training: int):
  num_warmup = max(0, int(num_warmup))
  num_training = max(1, int(num_training))

  def lr_lambda(step: int) -> float:
    if step < num_warmup:
      return float(step) / max(1, num_warmup)
    progress = float(step - num_warmup) / max(1, num_training - num_warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

  return lr_lambda


def build_optimizer(model: torch.nn.Module, train_dl: Iterable):
  """Construct ``AdamW`` + cosine-with-warmup scheduler from FLAGS."""
  from absl import flags as _flags
  F = _flags.FLAGS

  backbone = getattr(model, 'backbone', None) or getattr(model, 'encoder', None)
  param_groups = le.get_optimizer_params(
      model,
      backbone_lr=(getattr(F, 'lr', None) or getattr(F, 'learning_rate', None)),
      base_lr=(getattr(F, 'head_lr', None)
               or getattr(F, 'lr', None)
               or getattr(F, 'learning_rate', None)),
      weight_decay=True,
      weight_decay_val=float(getattr(F, 'weight_decay', 0.01) or 0.01),
      backbone=backbone,
  )

  lr = float(getattr(F, 'lr', None) or getattr(F, 'learning_rate', None) or 1e-4)
  optimizer = AdamW(param_groups, lr=lr,
                    betas=(0.9, 0.999),
                    eps=1e-8)

  epochs = float(getattr(F, 'exit_epoch', 0.0) or getattr(F, 'ep', 1) or 1)
  steps_per_epoch = len(train_dl)
  total_steps = max(1, int(math.ceil(steps_per_epoch * epochs)))
  warmup_ratio = float(getattr(F, 'warmup_proportion', 0.1) or 0.1)
  warmup_steps = int(total_steps * warmup_ratio)
  scheduler = LambdaLR(optimizer, _cosine_warmup_schedule(warmup_steps, total_steps))

  logger.info(
      f'Optimizer ready: lr={lr:g}, epochs={epochs}, '
      f'steps_per_epoch={steps_per_epoch}, warmup_steps={warmup_steps}, '
      f'total_steps={total_steps}'
  )
  return optimizer, scheduler


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------
class _EMA:
  """Plain exponential moving average over a model's parameters.

  Replaces the upstream ``EMAv2`` callback. Decay is held constant, EMA
  starts updating after ``warmup_steps`` optimizer steps (matching the
  upstream "start from epoch 1" default with one epoch warmup).
  """

  def __init__(self, model: torch.nn.Module, decay: float, warmup_steps: int = 0):
    self.decay = decay
    self.warmup_steps = warmup_steps
    self.shadow: dict[str, torch.Tensor] = {
        name: p.detach().clone() for name, p in model.named_parameters()
        if p.requires_grad
    }
    self._step = 0

  @torch.no_grad()
  def update(self, model: torch.nn.Module) -> None:
    self._step += 1
    if self._step <= self.warmup_steps:
      # Snapshot but don't average yet.
      for name, p in model.named_parameters():
        if name in self.shadow:
          self.shadow[name].copy_(p.detach())
      return
    d = self.decay
    for name, p in model.named_parameters():
      if name in self.shadow:
        self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

  def apply_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Swap the EMA weights into ``model`` and return the original weights."""
    backup: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
      if name in self.shadow:
        backup[name] = p.detach().clone()
        p.data.copy_(self.shadow[name])
    return backup

  def restore(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
    for name, p in model.named_parameters():
      if name in backup:
        p.data.copy_(backup[name])


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------
def _to_device(batch: Any, device: torch.device) -> Any:
  if torch.is_tensor(batch):
    return batch.to(device, non_blocking=True)
  if isinstance(batch, dict):
    return {k: _to_device(v, device) for k, v in batch.items()}
  if isinstance(batch, (list, tuple)):
    return type(batch)(_to_device(v, device) for v in batch)
  return batch


def _normalize_batch(batch: Any) -> tuple[dict, Any]:
  """``(inputs, labels_or_None)``. The collate_fn already merges labels
  into ``inputs`` so we always return ``(batch, None)``."""
  if isinstance(batch, dict):
    return batch, None
  if isinstance(batch, (list, tuple)) and len(batch) == 2:
    return batch[0], batch[1]
  raise TypeError(f'Unsupported batch type: {type(batch)}')


def train(*,
          model: torch.nn.Module,
          train_dl: Iterable,
          eval_dl: Optional[Iterable],
          valid_dl: Optional[Iterable],
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          eval_fn: Optional[Callable] = None,
          model_dir: str | os.PathLike = '.',
          epochs: int = 1,
          grad_accum_steps: int = 1,
          use_amp: bool = True,
          ema_decay: Optional[float] = None,
          grad_clip: Optional[float] = None) -> None:
  """Run ``epochs`` of training, then a single final validation pass.

  After the last epoch the model checkpoint is written to
  ``<model_dir>/model.pt`` (EMA weights if EMA is enabled), the training
  FLAGS are dumped to ``<model_dir>/flags.json``, and ``metrics.csv`` is
  written next to it with the validation metrics.
  """
  from absl import flags as _flags
  F = _flags.FLAGS
  requested_device = str(getattr(F, 'device', '') or '').lower()
  cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')
  if requested_device.startswith('cuda') or requested_device == 'gpu':
    # Match the original melt/mt.fit behavior: with CUDA_VISIBLE_DEVICES set,
    # visible GPU 0 is the selected single card.  Avoid a preflight
    # torch.cuda.is_available() check because on degraded multi-GPU machines it
    # can pessimistically return false before set_device/model.to gets a chance.
    device = torch.device('cuda:0')
    try:
      torch.cuda.set_device(device)
      logger.info(
          f'Using CUDA device {device} (CUDA_VISIBLE_DEVICES={cuda_visible}, '
          f'name={torch.cuda.get_device_name(device)})')
    except Exception as e:
      raise RuntimeError(
          f'CUDA was explicitly requested with --device={requested_device}, but '
          f'PyTorch could not select cuda:0 (CUDA_VISIBLE_DEVICES={cuda_visible}). '
          f'{type(e).__name__}: {e}') from e
  elif torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    logger.info(
        f'Using CUDA device {device} (CUDA_VISIBLE_DEVICES={cuda_visible}, '
        f'name={torch.cuda.get_device_name(device)})')
  else:
    device = torch.device('cpu')
    logger.warning(
        f'CUDA is not available in this process (CUDA_VISIBLE_DEVICES={cuda_visible}); '
        'falling back to CPU. Pass --device=cuda to fail fast instead.')
  model = model.to(device)
  model_dir = Path(model_dir)
  model_dir.mkdir(parents=True, exist_ok=True)

  # AMP — bf16 by default for SSL backbones (more numerically stable than fp16),
  # otherwise fp16. Decision matches the upstream defaults.
  amp_enabled = bool(use_amp and device.type == 'cuda')
  amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
  scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype is torch.float16)

  ema: _EMA | None = None
  if ema_decay is not None and ema_decay > 0:
    warmup_steps = max(1, len(train_dl)) * 1  # one-epoch warmup, like upstream
    ema = _EMA(model, decay=float(ema_decay), warmup_steps=warmup_steps)
    gz.set('EMAv2', ema)
    logger.info(f'EMA enabled: decay={ema_decay}, warmup_steps={warmup_steps}')

  loss_fn = model.get_loss_fn()
  global_step = 0
  history = []

  steps_per_epoch = len(train_dl)
  total_train_batches = max(1, int(math.ceil(steps_per_epoch * float(epochs))))
  max_epochs = max(1, int(math.ceil(float(epochs))))

  for epoch in range(max_epochs):
    epoch_start_batch = epoch * steps_per_epoch
    epoch_target_batches = min(steps_per_epoch, total_train_batches - epoch_start_batch)
    if epoch_target_batches <= 0:
      break
    gz.set('epoch', float(epoch))
    model.train()
    pbar = tqdm(
        train_dl,
        desc=f'epoch {epoch + 1}/{epochs:g}',
        total=epoch_target_batches,
        dynamic_ncols=True,
    )
    optimizer.zero_grad(set_to_none=True)
    epoch_loss, n_seen = 0.0, 0
    t0 = time.time()

    for it, raw in enumerate(pbar):
      if it >= epoch_target_batches:
        break
      inputs, labels = _normalize_batch(raw)
      inputs = _to_device(inputs, device)

      with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        res = model(inputs)
        loss = loss_fn(res, labels, inputs,
                        step=global_step, epoch=epoch + it / max(1, len(train_dl)),
                        training=True)
        loss = loss / max(1, grad_accum_steps)

      if scaler.is_enabled():
        scaler.scale(loss).backward()
      else:
        loss.backward()

      if (it + 1) % grad_accum_steps == 0:
        if grad_clip is not None and grad_clip > 0:
          if scaler.is_enabled():
            scaler.unscale_(optimizer)
          torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        if scaler.is_enabled():
          scaler.step(optimizer)
          scaler.update()
        else:
          optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        global_step += 1
        if ema is not None:
          ema.update(model)

      with torch.no_grad():
        epoch_loss += float(loss.item()) * max(1, grad_accum_steps)
        n_seen += 1
      pbar.set_postfix(loss=f'{epoch_loss / max(1, n_seen):.4f}',
                       lr=f'{scheduler.get_last_lr()[0]:.2e}')

      gz.set('epoch', epoch + (it + 1) / max(1, steps_per_epoch))

    logger.info(
        f'Epoch {epoch + 1}/{epochs:g} done in {(time.time() - t0) / 60:.1f} min, '
        f'avg_loss={epoch_loss / max(1, n_seen):.4f}'
    )
    history.append({'epoch': epoch + 1, 'train_loss': epoch_loss / max(1, n_seen)})

  # ---- Save final checkpoint ----
  if ema is not None:
    backup = ema.apply_to(model)
    logger.info('Swapped EMA weights into model for final save / eval')
  ckpt_path = model_dir / 'model.pt'
  torch.save(model.state_dict(), ckpt_path)
  logger.info(f'Saved final checkpoint: {ckpt_path}')

  gz.save_globals(model_dir)

  # ---- Final validation pass ----
  if eval_fn is not None and (valid_dl is not None or eval_dl is not None):
    final_dl = valid_dl if valid_dl is not None else eval_dl
    logger.info('Running final validation pass ...')
    metrics = _evaluate(model, final_dl, eval_fn, device, amp_dtype, use_amp)
    if metrics:
      _write_metrics_csv(model_dir / 'metrics.csv', metrics)
      logger.info(f'Final metrics: {json.dumps(metrics, indent=2, default=str)}')

  if ema is not None:
    ema.restore(model, backup)

  # Persist tiny training history.
  with open(model_dir / 'history.json', 'w') as f:
    json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def _evaluate(model: torch.nn.Module, dl: Iterable, eval_fn: Callable,
              device: torch.device, amp_dtype: torch.dtype, use_amp: bool) -> dict:
  model.eval()
  gz.set('do_generate', True)
  all_preds, all_labels = [], []
  for raw in tqdm(dl, desc='final eval', dynamic_ncols=True):
    inputs, _ = _normalize_batch(raw)
    inputs = _to_device(inputs, device)
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
      res = model(inputs)
    if 'pred_texts' in res:
      all_preds.extend(res['pred_texts'])
    elif 'pred' in res:
      all_preds.append(res['pred'].detach().cpu())
    if 'label_texts' in inputs:
      all_labels.extend(inputs['label_texts'])
  gz.set('do_generate', False)

  # Most ``eval_fn`` implementations in the upstream repo accept
  # ``(preds, labels, **ctx)`` and return a dict. Forward what we have
  # and let the project's own eval module decide what to do with it.
  try:
    return eval_fn(all_preds, all_labels) or {}
  except TypeError:
    try:
      return eval_fn(preds=all_preds, labels=all_labels) or {}
    except Exception as exc:  # pragma: no cover
      logger.warning(f'eval_fn invocation failed: {exc}')
      return {}


def _write_metrics_csv(path: Path, metrics: dict) -> None:
  import csv
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['metric', 'value'])
    for k, v in metrics.items():
      w.writerow([k, v])
