"""Standalone training entry point for the Pasketti Phonetic ASR solution.

Usage:
    cd src
    python train.py --flagfile=flags/v16 --mn=v16.dual_bpe --tdt_only --fold=0

The script:
    1. Adds the bundled ``_compat`` shim directory to ``sys.path`` so that
       the project source files (``dataset.py``, ``models/``, ...) keep using
       ``from gezi.common import *`` exactly as in the development repo.
    2. Parses CLI / flagfile arguments via ``absl.flags`` (the project
       defines ~500 flags in ``config_base.py`` — same semantics as the
       upstream training driver).
    3. Builds the model, dataloaders and optimizer.
    4. Runs the hand-written training loop in :mod:`src.train_loop`.
    5. After the last epoch, runs a single final validation pass and
       writes ``metrics.csv`` next to the model checkpoint.

The training loop intentionally omits framework features that are not
required for *reproducing* the published submission (per-epoch eval,
DDP, callbacks, mixed precision auto-tuning, wandb, ...). What remains
is the bare minimum: optimizer + scheduler + EMA + AMP forward.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# 1. Bootstrap: prepend the bundled compatibility shims to ``sys.path``
# ----------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_COMPAT = _HERE / '_compat'
if _COMPAT.exists():
  sys.path.insert(0, str(_COMPAT))
_RUNTIME_CANDIDATES = [
    os.environ.get('PASKETTI_RUNTIME_DIR'),
    _HERE.parents[1] / 'childrens-speech-recognition-runtime',
    _HERE.parent.parent / 'childrens-speech-recognition-runtime',
]
for _runtime_dir in _RUNTIME_CANDIDATES:
  if _runtime_dir and Path(_runtime_dir).exists():
    sys.path.insert(0, str(Path(_runtime_dir).resolve()))
    break
sys.path.insert(0, str(_HERE.parent))   # so ``import src`` works
sys.path.insert(0, str(_HERE))          # so ``from src.xxx`` resolves intra-package

# Quiet down noisy third-party libs early.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_OFFLINE', '0')

# ----------------------------------------------------------------------
# 2. Imports (after sys.path is patched)
# ----------------------------------------------------------------------
from absl import app  # noqa: E402

import gezi as gz  # noqa: E402
import melt as mt  # noqa: E402

import src  # noqa: E402  — registers the package
from src import config  # noqa: E402
from src.config import FLAGS, MODEL_NAME  # noqa: E402
from src import eval as ev  # noqa: E402
from src.dataset import get_dl  # noqa: E402
from src.train_loop import build_optimizer, train  # noqa: E402


def main(_argv):
  timer = gz.Timer()
  gz.set('timer', timer)

  # config.init() is the project's own setup hook (defined in config.py /
  # config_base.py). It populates FLAGS defaults and resolves the model dir.
  config.init()
  mt.init()
  config.post_restore()
  config.show()

  # --- Dataloaders ---
  train_dl, eval_dl, valid_dl = get_dl()

  # --- Model (dynamic: which class depends on FLAGS.model) ---
  import importlib
  model_module = importlib.import_module(f'src.models.{FLAGS.model}')
  model = model_module.Model()
  gz.logger.info(f'Built model {type(model).__name__}')

  # --- Optimizer + scheduler ---
  optimizer, scheduler = build_optimizer(model, train_dl)

  # --- Train ---
  train(
      model=model,
      train_dl=train_dl,
      eval_dl=eval_dl,
      valid_dl=valid_dl,
      optimizer=optimizer,
      scheduler=scheduler,
      eval_fn=ev.evaluate,
      model_dir=FLAGS.model_dir,
      epochs=float(getattr(FLAGS, 'exit_epoch', 0.0) or FLAGS.ep),
      grad_accum_steps=getattr(FLAGS, 'acc_steps', 1) or 1,
      use_amp=bool(getattr(FLAGS, 'fp16', True) or getattr(FLAGS, 'bfloat16', False)),
      ema_decay=getattr(FLAGS, 'ema_decay', None),
      grad_clip=(getattr(FLAGS, 'grad_clip', None)
                 or (1.0 if getattr(FLAGS, 'clip_grads', False) else None)),
  )

  gz.logger.info(f'Training complete in {timer.elapsed_minutes():.1f} min')


def run():
  app.run(main)


if __name__ == '__main__':
  run()
