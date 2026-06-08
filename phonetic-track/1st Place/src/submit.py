#!/usr/bin/env python3
"""Canonical inference entry point that reuses training code.

Submission variant B: instead of a standalone self-contained inference script
(submit.py), this bundles the training src/ codebase and pikachu utilities to
reuse the exact same model forward() and decode logic used during training/eval.

Advantages:
  - New model types work automatically (no submit.py adaptation needed)
  - Inference behavior is identical to training eval (no code duplication)
  - CTC / S2S / hybrid / joint decode methods all work unchanged

How it works:
  pack-submission-prob.sh bundles:
    src/*.py, src/models/    →  the ASR training code
    pikachu/utils/           →  gezi, melt, lele, husky
    pikachu/third/           →  third-party libraries (optional)
    wheels/                  →  pre-downloaded .whl for absl-py, icecream, dill
    model/                   →  saved model weights + flags.json

  Docker runtime (submit-test.sh) unpacks into /code_execution/src/ and runs:
    uv run src/main.py
"""

import subprocess, sys, os, json, time, io, contextlib, tempfile, pickle, atexit, signal, re

# WORKAROUND: Force numba to not use buggy python-cuda bindings in DD environment
os.environ['NUMBA_CUDA_USE_NVIDIA_BINDING'] = '0'

from pathlib import Path
from loguru import logger as _log  # Use _log to avoid overwrite by gezi.common

TIMER_START = time.time()
CURRENT_STAGE = 'boot'
_progress_file = Path('/code_execution/submission/progress.txt')
_STDOUT_LOG_LIMIT = int(os.environ.get('SUBMIT_STDOUT_MAX_LINES', '260') or '260')
_STDOUT_LOG_RESERVE = int(os.environ.get('SUBMIT_STDOUT_RESERVE_LINES', '24') or '24')
_stdout_log_lines = 0
_stdout_log_suppressed = 0
_stdout_budget_announced = False
_ensemble_final_flags_logged = False
_pred_meta_disabled_logged = False

# Write detailed logs to a separate file (not capped by Docker runtime)
_log_file = Path('/code_execution/submission/submit.log')
try:
  _log.remove()
  _log_file.parent.mkdir(parents=True, exist_ok=True)
  _log.add(str(_log_file), format="{time:HH:mm:ss} | {level} | {message}", level="DEBUG")
except Exception:
  pass
# Only add WARNING+ to stdout (Docker log.txt has 500 line cap)
_log.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="WARNING")

def _ts():
  """Elapsed time since process start as MM:SS."""
  e = int(time.time() - TIMER_START)
  return f'{e // 60:02d}:{e % 60:02d}'

def _diag(msg, force=False):
  """Print concise diagnostic to stdout (visible in Docker log.txt)."""
  global _stdout_log_lines, _stdout_log_suppressed, _stdout_budget_announced
  line = f'[{_ts()}][submit] {msg}'
  try:
    _log.debug(line)
  except Exception:
    pass
  if _STDOUT_LOG_LIMIT > 0 and not force:
    budget = max(0, _STDOUT_LOG_LIMIT - _STDOUT_LOG_RESERVE)
    if _stdout_log_lines >= budget:
      _stdout_log_suppressed += 1
      if not _stdout_budget_announced:
        print(f'[{_ts()}][submit] stdout budget reached; suppressing verbose logs, see {_log_file}', flush=True)
        _stdout_log_lines += 1
        _stdout_budget_announced = True
      return
  print(line, flush=True)
  _stdout_log_lines += 1


def _flush_diag_budget():
  """Emit a one-line summary if verbose stdout logs were suppressed."""
  if _stdout_log_suppressed > 0:
    _diag(f'suppressed {_stdout_log_suppressed} stdout log line(s); full details in {_log_file}', force=True)

def _env_flag(name, default=False):
  """Parse boolean env vars consistently."""
  value = os.environ.get(name)
  if value is None:
    return default
  value = str(value).strip().lower()
  if value in ('1', 'true', 't', 'yes', 'y', 'on'):
    return True
  if value in ('0', 'false', 'f', 'no', 'n', 'off'):
    return False
  return default

def _runtime_track():
  """Infer active track from restored flags with a safe fallback."""
  try:
    track = str(getattr(FLAGS, 'track', '') or '').strip().lower()
    if track in ('word', 'phonetic'):
      return track
    score_metric = str(getattr(FLAGS, 'score_metric', '') or '').strip().lower()
    if score_metric == 'wer':
      return 'word'
    if score_metric == 'ipa_cer':
      return 'phonetic'
    label_column = str(getattr(FLAGS, 'label_column', '') or '').strip().lower()
    if label_column == 'orthographic_text':
      return 'word'
    if label_column == 'phonetic_text':
      return 'phonetic'
  except Exception:
    pass
  return ''


def _get_runtime_text_normalizer():
  track = _runtime_track()
  if track != 'word':
    return lambda text: str(text or '').strip()

  try:
    from metric.score import EnglishTextNormalizer, english_spelling_normalizer
    word_normalizer = EnglishTextNormalizer(english_spelling_normalizer)

    def _normalize(text):
      return word_normalizer(str(text or ''))

    return _normalize
  except Exception:
    return lambda text: str(text or '').strip()


def _is_word_runtime():
  return _runtime_track() == 'word'


def _show_ensemble_progress_logs(model_count, is_word_runtime):
  """Decide whether per-milestone progress logs should be printed in ensemble mode.

  Single-model progress is useful. For larger ensembles, repeated per-milestone
  logs consume the limited stdout budget quickly, so default to summary-only
  unless explicitly re-enabled.
  """
  env_value = os.environ.get('SUBMIT_ENSEMBLE_PROGRESS', '').strip().lower()
  if env_value:
    return env_value in ('1', 'true', 't', 'yes', 'y', 'on')
  if model_count <= 1:
    return True
  if is_word_runtime:
    return False
  return model_count <= 3

def _set_stage(stage, extra=None, log_stdout=True):
  """Persist the last known stage so abrupt smoke failures are easier to classify."""
  global CURRENT_STAGE
  CURRENT_STAGE = stage
  msg = f'stage={stage}'
  if extra:
    msg = f'{msg} | {extra}'
  try:
    _progress_file.parent.mkdir(parents=True, exist_ok=True)
    _progress_file.write_text(f'{time.time():.3f}\t{msg}\n')
  except Exception:
    pass
  if log_stdout:
    _diag(msg)

def _install_exit_diagnostics():
  def _on_exit():
    _diag(f'process exit hook reached at stage={CURRENT_STAGE}')

  def _on_signal(signum, _frame):
    try:
      signame = signal.Signals(signum).name
    except Exception:
      signame = str(signum)
    _diag(f'received signal {signame} at stage={CURRENT_STAGE}')

  atexit.register(_on_exit)
  for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT):
    try:
      signal.signal(_sig, _on_signal)
    except Exception:
      pass

_install_exit_diagnostics()

# ===========================================================================
#  Phase 1: Install missing deps from bundled wheels (no network in Docker)
# ===========================================================================
SRC_DIR = Path(__file__).resolve().parent          # /code_execution/src
WHEEL_DIR = SRC_DIR / 'wheels'
VENDORED_PACKAGES_TAR = SRC_DIR / 'vendor_packages.tar.gz'
VENDORED_PACKAGES_DIR = SRC_DIR / 'vendor_pkgs'


def _restore_vendored_packages():
  if VENDORED_PACKAGES_TAR.exists() and not VENDORED_PACKAGES_DIR.is_dir():
    import tarfile
    _diag('Extracting vendor_packages.tar.gz ...')
    with tarfile.open(VENDORED_PACKAGES_TAR) as tf:
      tf.extractall(SRC_DIR)
  if VENDORED_PACKAGES_DIR.is_dir() and str(VENDORED_PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(VENDORED_PACKAGES_DIR))
    _diag(f'Enabled vendored packages from {VENDORED_PACKAGES_DIR.name}')


def _bool_flag(value):
  return '1' if bool(value) else '0'

# Pre-mock icecream before anything tries to import it.
# ic() is a debug printer — not needed for inference. Redefine as print.
import types as _types
_ic_mod = _types.ModuleType('icecream')
_ic_mod.install = lambda *a, **kw: None
class _IcMock:
    """Mock icecream.ic with all methods that gezi/utils/logging.py uses."""
    def __call__(self, *a, **kw):
        if a: print(*a)
    def configureOutput(self, *a, **kw): pass
    def __getattr__(self, name): return lambda *a, **kw: None
_ic_mod.ic = _IcMock()
_ic_mod.argumentToString = lambda *a, **kw: None
sys.modules['icecream'] = _ic_mod

def _install_wheels():
  """Install bundled wheel packages if available."""
  if not WHEEL_DIR.is_dir():
    _log.info('No wheels/ directory found, skipping local wheel install')
    return
  # pkg-name → import-name mapping (most are trivial s/-/_/)
  PKG_IMPORT = {'absl-py': 'absl', 'dill': 'dill'}  # icecream is pre-mocked above
  # Conditionally add peft if the model uses Whisper LoRA.
  # Check flags.json early to decide (before full FLAGS init).
  _flags_file = WHEEL_DIR.parent / 'model' / 'flags.json'
  _needs_peft = False
  if _flags_file.exists():
    try:
      import json as _jj
      _sf = _jj.load(open(_flags_file))
      _needs_peft = _sf.get('whisper_lora', False)
    except Exception:
      pass
  if _needs_peft:
    PKG_IMPORT['peft'] = 'peft'
  # Conditionally add tree model library (catboost / lightgbm / xgboost)
  # when tree_reranker mode is active.
  _tree_meta_file = WHEEL_DIR.parent / 'tree_reranker' / 'reranker_meta.json'
  if _tree_meta_file.exists():
    try:
      import json as _jj
      _tm = _jj.load(open(_tree_meta_file))
      for _tree_lib in _tm.get('tree_models', []):
        if _tree_lib == 'cb':
          PKG_IMPORT['catboost'] = 'catboost'
        elif _tree_lib == 'lgb':
          PKG_IMPORT['lightgbm'] = 'lightgbm'
        elif _tree_lib == 'xgb':
          PKG_IMPORT['xgboost'] = 'xgboost'
    except Exception:
      pass
  pkgs = list(PKG_IMPORT.keys())
  # Filter to packages not already installed
  missing = []
  for pkg in pkgs:
    try:
      __import__(PKG_IMPORT[pkg])
    except ImportError:
      missing.append(pkg)
  if not missing:
    _log.info(f'All required packages already installed')
    return
  _log.info(f'Installing from bundled wheels: {missing}')
  # Try pip first, then uv pip (Docker runtime uses uv)
  # --no-deps: bundled wheels are self-contained; skip dependency resolution
  #            (e.g. catboost optionally depends on plotly which we don't need)
  installed = False
  for installer in [
    [sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index', '--no-deps',
     '--find-links', str(WHEEL_DIR)] + missing,
    ['uv', 'pip', 'install', '--quiet', '--no-index', '--no-deps',
     '--find-links', str(WHEEL_DIR)] + missing,
  ]:
    r = subprocess.run(installer, check=False, capture_output=True, text=True)
    if r.returncode == 0:
      installed = True
      _log.info(f'Wheel install OK via: {installer[0:3]}')
      break
    else:
      _log.warning(f'Wheel install failed ({installer[0:3]}): {r.stderr[:200] if r.stderr else "unknown"}')
  # Verify
  still_missing = []
  for pkg in missing:
    try:
      __import__(PKG_IMPORT[pkg])
    except ImportError:
      still_missing.append(pkg)
  if still_missing and VENDORED_PACKAGES_TAR.exists():
    _log.warning(f'Wheel install incomplete, trying vendored package fallback: {still_missing}')
    _restore_vendored_packages()
    still_missing = []
    for pkg in missing:
      try:
        __import__(PKG_IMPORT[pkg])
      except ImportError:
        still_missing.append(pkg)
  if still_missing:
    _log.error(f'Packages STILL missing after wheel install: {still_missing}')

_install_wheels()

# ===========================================================================
#  Phase 1b: Fix numba NVVM for Docker CUDA runtime images
#
#  Docker runtime images have libnvidia-nvvm.so (driver-shipped NVVM) but NOT
#  the classic libnvvm.so from the CUDA toolkit.  Numba's open_cudalib('nvvm')
#  searches for libnvvm.so and fails.  We create a symlink and monkey-patch
#  numba to use it.  Must happen before any NeMo TDT loss code triggers numba
#  CUDA JIT compilation.
# ===========================================================================
def _fix_numba_nvvm():
  """Make numba find NVVM + libdevice in Docker CUDA runtime images.

  Docker runtime images ship libnvidia-nvvm.so (driver NVVM) but not the
  classic libnvvm.so.  They also lack libdevice.10.bc (CUDA bitcode math
  library).  We fix both:
    1. Symlink libnvvm.so → libnvidia-nvvm.so and monkey-patch numba
    2. Copy libdevice.10.bc from wheels/ and set NUMBA_CUDA_DRIVER_LIBDEVICE
  """
  # -- Fix 1: libnvvm.so --
  _nvvm_src = Path('/usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so.4')
  if not _nvvm_src.exists():
    return  # Not in the Docker environment, or NVVM already available
  _nvvm_link = Path('/tmp/libnvvm.so')
  if not _nvvm_link.exists():
    try:
      _nvvm_link.symlink_to(_nvvm_src)
    except OSError:
      return
  import ctypes as _ct
  try:
    _nvvm_lib = _ct.CDLL(str(_nvvm_link))
  except OSError:
    return
  try:
    from numba.cuda.cudadrv import nvvm as _nvvm_mod
    _orig_open = _nvvm_mod.open_cudalib
    def _patched_open(lib, ccc=False):
      if lib == 'nvvm':
        return _nvvm_lib
      return _orig_open(lib, ccc)
    _nvvm_mod.open_cudalib = _patched_open
    _diag('Patched numba NVVM → libnvidia-nvvm.so.4')
  except Exception:
    pass

  # -- Fix 2: libdevice.10.bc --
  _libdevice_src = WHEEL_DIR / 'libdevice.10.bc'
  if _libdevice_src.exists():
    _libdevice_dst = Path('/tmp/libdevice.10.bc')
    if not _libdevice_dst.exists():
      import shutil
      shutil.copy2(str(_libdevice_src), str(_libdevice_dst))
    try:
      from numba.cuda.cudadrv import libs as _libs_mod
      _ld_fn = lambda: str(_libdevice_dst)
      _nvvm_mod.get_libdevice = _ld_fn
      _libs_mod.get_libdevice = _ld_fn
      _diag(f'Patched numba libdevice → {_libdevice_dst}')
    except Exception:
      pass

_fix_numba_nvvm()

# Exact TDT scoring in submission must use the Numba path.
# If Numba is still unusable online after _fix_numba_nvvm(), fail loudly during
# warmup / scoring rather than silently switching to a slower PyTorch fallback.

# ===========================================================================
#  Phase 2: Mock modules that may be missing in Docker (plotly, IPython, etc.)
# ===========================================================================
import types

def _ensure_mock_module(name, attrs=None):
  """Create a no-op mock module if not importable."""
  try:
    __import__(name)
  except ImportError:
    mod = types.ModuleType(name)
    # Support dotted access: plotly.express, plotly.offline
    parts = name.split('.')
    for i in range(len(parts)):
      parent = '.'.join(parts[:i+1])
      if parent not in sys.modules:
        m = types.ModuleType(parent)
        m.__path__ = []
        sys.modules[parent] = m
    if attrs:
      for attr, val in attrs.items():
        setattr(sys.modules[name], attr, val)

# Mock modules that gezi.common imports unconditionally but aren't needed
def _noop(*a, **kw): pass
# icecream: already pre-mocked in Phase 1 (before wheel install)
_ensure_mock_module('plotly')
_ensure_mock_module('plotly.express')
_ensure_mock_module('plotly.offline', {'init_notebook_mode': lambda **kw: None})
_ensure_mock_module('IPython')
_ensure_mock_module('IPython.display', {
    'display_html': lambda *a, **kw: None,
    'display': lambda *a, **kw: None,
    'HTML': lambda *a, **kw: None,
})
# polars: imported but not needed
_ensure_mock_module('polars')
# rich_dataframe: optional pretty-printing
_ensure_mock_module('rich_dataframe', {'prettify': lambda *a, **kw: None})
# pymp: optional multiprocessing
_ensure_mock_module('pymp')
# cudf: optional
_ensure_mock_module('cudf')

# ===========================================================================
#  Phase 2.5: Extract pikachu_utils.tar.gz (packed to reduce zip entries)
# ===========================================================================
_pikachu_tar = SRC_DIR / 'pikachu_utils.tar.gz'
if _pikachu_tar.exists() and not (SRC_DIR / 'pikachu_utils').is_dir() and not (SRC_DIR / 'utils').is_dir():
    import tarfile
    _diag('Extracting pikachu_utils.tar.gz ...')
    with tarfile.open(_pikachu_tar) as tf:
        tf.extractall(SRC_DIR)
    _diag(f'pikachu_utils extracted ({time.time() - TIMER_START:.1f}s)')

_pikachu_third_tar = SRC_DIR / 'pikachu_third.tar.gz'
if _pikachu_third_tar.exists() and not (SRC_DIR / 'pikachu_third').is_dir() and not (SRC_DIR / 'third').is_dir():
  import tarfile
  _diag('Extracting pikachu_third.tar.gz ...')
  with tarfile.open(_pikachu_third_tar) as tf:
    tf.extractall(SRC_DIR)
  _diag(f'pikachu_third extracted ({time.time() - TIMER_START:.1f}s)')

PIKACHU_UTILS_DIR = SRC_DIR / 'pikachu_utils'
if not PIKACHU_UTILS_DIR.is_dir():
  PIKACHU_UTILS_DIR = SRC_DIR / 'utils'

PIKACHU_THIRD_DIR = SRC_DIR / 'pikachu_third'
if not PIKACHU_THIRD_DIR.is_dir():
  PIKACHU_THIRD_DIR = SRC_DIR / 'third'

# ===========================================================================
#  Phase 3: Set up sys.path
# ===========================================================================
CODE_DIR = SRC_DIR.parent                          # /code_execution
sys.path.insert(0, str(CODE_DIR))                  # for 'from src.xxx import *'
sys.path.insert(0, str(SRC_DIR))                   # for 'from metric.score import ...' (bundled)
sys.path.insert(0, str(PIKACHU_UTILS_DIR))         # for gezi, melt, lele, husky
sys.path.insert(0, str(PIKACHU_THIRD_DIR))         # for third-party

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# ===========================================================================
#  Phase 4: Import training code & configure
# ===========================================================================
from gezi.common import *
from src import config
from src.config import *
from src.preprocess import *
from src.eval import decode_ids
from src import util

# NOTE: 'from gezi.common import *' overwrites 'logger' to gezi's stdlib logger.
# We use '_log' (loguru) and '_diag' (print) defined BEFORE the import.
_diag(f'Imports done ({time.time() - TIMER_START:.1f}s)')

# Disable icecream globally — ic() calls in lele/gezi dump verbose debug
# output (mismatch_ignores, additional_ignores sets) that wastes Docker log lines.
try:
  ic.disable()
except Exception:
  pass

# ===========================================================================
#  Phase 5: Restore FLAGS from saved model & configure for inference
# ===========================================================================
MODEL_DIR = SRC_DIR / 'model'
DATA_DIR = Path('/code_execution/data')
SUBMISSION_DIR = Path('submission')
PROBE_FLAG = SRC_DIR / 'probe_mode'
PROBE_MODE = PROBE_FLAG.exists()
PROBE_REPORT_PATH = SUBMISSION_DIR / 'probe_report.json'

# ===========================================================================
#  Ensemble support: detect multiple model dirs (model/, model_1/, model_2/, ...)
# ===========================================================================
ENSEMBLE_META_FILE = SRC_DIR / 'ensemble_meta.json'
IS_ENSEMBLE = ENSEMBLE_META_FILE.exists()
ENSEMBLE_MODEL_DIRS = []
ENSEMBLE_INFER_MODEL_DIRS = []


def _read_model_flags_for_runtime(model_dir):
  for candidate in (model_dir / 'flags.json', model_dir.parent / 'flags.json'):
    if candidate.exists():
      try:
        with open(candidate) as f:
          return json.load(f)
      except Exception:
        return {}
  return {}


def _is_wavlm_model_name(model_name, flags_data=None):
  name = str(model_name or '').strip().lower()
  backbone = str((flags_data or {}).get('backbone', '') or '').strip().lower()
  return ('wavlm' in name) or ('wavlm' in backbone)


def _is_nemo_model_name(model_name, flags_data=None):
  if _is_wavlm_model_name(model_name, flags_data=flags_data):
    return False
  name = str(model_name or '').strip().lower()
  model_kind = str((flags_data or {}).get('model', '') or '').strip().lower()
  backbone = str((flags_data or {}).get('backbone', '') or '').strip().lower()
  if model_kind == 'nemo':
    return True
  if 'nemo' in name:
    return True
  nemo_backbone_markers = ('parakeet', 'conformer', 'fastconformer', 'citrinet')
  return any(marker in backbone for marker in nemo_backbone_markers)


def _detect_wavlm_model_names(model_names, flags_by_name=None):
  detected = []
  for mn in model_names:
    flags_data = (flags_by_name or {}).get(mn, {})
    if _is_wavlm_model_name(mn, flags_data=flags_data):
      detected.append(mn)
  return detected


def _detect_nemo_model_names(model_names, flags_by_name=None):
  detected = []
  for mn in model_names:
    flags_data = (flags_by_name or {}).get(mn, {})
    if _is_nemo_model_name(mn, flags_data=flags_data):
      detected.append(mn)
  return detected


def _normalize_model_audio_filters(raw_filters, *, source_name='model_audio_filters'):
  normalized = {}
  if not raw_filters:
    return normalized
  assert isinstance(raw_filters, dict), (
      f'{source_name} must be a dict, got {type(raw_filters).__name__}')

  for raw_name, raw_rule in raw_filters.items():
    model_name = str(raw_name or '').strip()
    if not model_name:
      continue

    if isinstance(raw_rule, (int, float, str)):
      raw_rule = {'max_audio_sec': raw_rule}

    assert isinstance(raw_rule, dict), (
        f'{source_name}[{model_name!r}] must be a dict or number, '
        f'got {type(raw_rule).__name__}')

    rule = {}
    for key in ('min_audio_sec', 'max_audio_sec'):
      value = raw_rule.get(key)
      if value in (None, ''):
        continue
      try:
        rule[key] = float(value)
      except Exception as exc:
        raise AssertionError(
            f'{source_name}[{model_name!r}][{key!r}] must be numeric, got {value!r}') from exc

    if not rule:
      continue
    if 'min_audio_sec' in rule and 'max_audio_sec' in rule:
      assert rule['min_audio_sec'] <= rule['max_audio_sec'], (
          f'{source_name}[{model_name!r}] has min_audio_sec > max_audio_sec: {rule}')
    normalized[model_name] = rule

  return normalized


def _load_runtime_model_audio_filters():
  filters = {}

  try:
    filters.update(_normalize_model_audio_filters(
        _ensemble_meta.get('model_audio_filters'),
        source_name='ensemble_meta.model_audio_filters'))
  except Exception as exc:
    raise AssertionError(f'Invalid ensemble model_audio_filters: {exc}') from exc

  tree_meta_path = SRC_DIR / 'tree_reranker' / 'reranker_meta.json'
  if not tree_meta_path.exists():
    return filters

  try:
    with open(tree_meta_path) as f:
      tree_meta = json.load(f)
  except Exception as exc:
    raise AssertionError(f'Failed to load {tree_meta_path}: {exc}') from exc

  dir_names = [model_dir.name for model_dir in ENSEMBLE_MODEL_DIRS]
  meta_model_names = [str(name or '').strip() for name in tree_meta.get('model_names', [])]
  meta_to_dir = {}
  if len(meta_model_names) == len(dir_names):
    meta_to_dir = {
        meta_model_names[idx]: dir_names[idx]
        for idx in range(len(dir_names))
        if meta_model_names[idx]
    }

  raw_filters = tree_meta.get('model_audio_filters')
  normalized = {}
  if raw_filters:
    normalized.update(_normalize_model_audio_filters(
        raw_filters,
        source_name='reranker_meta.model_audio_filters'))

  def _get_tree_exp_payload():
    exp_path = tree_meta_path.with_name('reranker_experiment.json')
    if not exp_path.exists():
      return {}
    try:
      with open(exp_path) as f:
        return json.load(f)
    except Exception as exc:
      _diag(f'WARNING: Failed to load {exp_path.name}: {exc}')
      return {}

  def _parse_wavlm_max_dur(tree_exp):
    patterns = []
    command = str(tree_exp.get('command', '') or '').strip()
    if command:
      patterns.append(command)
    save_dir = str(tree_exp.get('save_dir', '') or '').strip()
    if save_dir:
      patterns.append(save_dir)

    regexes = [
        r'--ensemble_wavlm_max_dur(?:=|\s+)([0-9]+(?:\.[0-9]+)?)',
        r'wavlm_max_dur-([0-9]+(?:\.[0-9]+)?)',
    ]
    for text in patterns:
      for pattern in regexes:
        match = re.search(pattern, text)
        if match:
          try:
            value = float(match.group(1))
          except Exception:
            continue
          if value > 0:
            return value
    return None

  if not normalized:
    tree_exp = _get_tree_exp_payload()
    wavlm_max_dur = _parse_wavlm_max_dur(tree_exp)
    if wavlm_max_dur is not None:
      wavlm_model_names = tree_meta.get('wavlm_model_names') or []
      if not wavlm_model_names and meta_model_names:
        wavlm_model_names = [
            name for name in meta_model_names
            if 'wavlm' in str(name or '').lower() or 'wav2vec2' in str(name or '').lower()
        ]
      for model_name in wavlm_model_names:
        normalized[model_name] = {'max_audio_sec': wavlm_max_dur}
      if wavlm_model_names:
        _diag(
            'Derived wavlm model_audio_filters from reranker_experiment.json: ' +
            f'max_audio_sec={wavlm_max_dur} for {wavlm_model_names}')

  if not normalized:
    return filters

  unmapped = []
  for model_name, rule in normalized.items():
    target_name = model_name
    if target_name not in dir_names:
      target_name = meta_to_dir.get(model_name, '')
    if not target_name:
      unmapped.append(model_name)
      continue
    filters[target_name] = rule

  if unmapped:
    _diag('WARNING: Ignoring model_audio_filters for unknown models: ' + ', '.join(unmapped))

  return filters


def _filter_df_for_model_audio(df, model_name):
  rule = MODEL_AUDIO_FILTERS.get(model_name)
  if not rule:
    return df, None

  assert 'audio_duration_sec' in df.columns, (
      f'model_audio_filters configured for {model_name}, but test metadata lacks audio_duration_sec')

  import pandas as pd

  durations = pd.to_numeric(df['audio_duration_sec'], errors='coerce')
  mask = durations.notna()
  if 'min_audio_sec' in rule:
    mask &= durations >= rule['min_audio_sec']
  if 'max_audio_sec' in rule:
    mask &= durations <= rule['max_audio_sec']

  filtered_df = df.loc[mask].reset_index(drop=True)
  info = {
      'rule': rule,
      'total': int(len(df)),
      'kept': int(len(filtered_df)),
      'dropped': int(len(df) - len(filtered_df)),
  }
  return filtered_df, info


def _derive_model_audio_filter_from_saved_flags(model_name, saved_flags):
  if model_name in MODEL_AUDIO_FILTERS:
    return None
  if not _is_word_runtime():
    return None

  model_kind = str(saved_flags.get('model', '') or '').strip().lower()
  backbone = str(saved_flags.get('backbone', '') or '').strip().lower()
  is_wavlm_family = model_kind == 'wav2vec2' or any(
      token in backbone for token in ('wavlm', 'wav2vec2', 'hubert', 'w2v-bert'))
  if not is_wavlm_family:
    return None

  eval_truncate_audio = bool(saved_flags.get('eval_truncate_audio', False))
  if not eval_truncate_audio:
    return None

  try:
    max_audio_sec = float(saved_flags.get('max_audio_sec', 0) or 0)
  except Exception:
    return None

  # Auto-promote only obviously short-audio specialists. General 25-30s models
  # may enable eval truncation for throughput, but they should still cover the
  # full test set rather than becoming partial-coverage ensemble members.
  if max_audio_sec <= 0 or max_audio_sec > 5.0:
    return None

  return {
      'max_audio_sec': max_audio_sec,
      '_source': 'saved_flags.eval_truncate_audio',
  }


def _augment_model_subset_feature_frame(df, group_name, subset_model_names, primary_texts=None):
  if not subset_model_names:
    return df, []

  group_key = re.sub(r'[^0-9a-zA-Z]+', '_', str(group_name).strip().lower()).strip('_')
  if not group_key:
    return df, []

  score_cols = [f'ctc_score_{mn}' for mn in subset_model_names if f'ctc_score_{mn}' in df.columns]
  rank_cols = [f'beam_rank_{mn}' for mn in subset_model_names if f'beam_rank_{mn}' in df.columns]
  if not score_cols and not rank_cols and not primary_texts:
    return df, []

  df = df.copy()
  new_cols = []
  n_models = len(subset_model_names)
  candidate_len = df['candidate_text'].str.len().astype(float)
  score_prefix = f'{group_key}_ctc_score'

  if score_cols:
    df[f'{group_key}_n_models'] = float(n_models)
    df[f'{score_prefix}_mean'] = df[score_cols].mean(axis=1, skipna=True)
    df[f'{score_prefix}_std'] = df[score_cols].std(axis=1, skipna=True)
    df[f'{score_prefix}_min'] = df[score_cols].min(axis=1, skipna=True)
    df[f'{score_prefix}_max'] = df[score_cols].max(axis=1, skipna=True)
    df[f'{score_prefix}_range'] = df[f'{score_prefix}_max'] - df[f'{score_prefix}_min']
    df[f'{score_prefix}_mean_per_char'] = df[f'{score_prefix}_mean'] / candidate_len.clip(lower=1)

    grp_mean = df.groupby('uid')[f'{score_prefix}_mean']
    grp_max = df.groupby('uid')[f'{score_prefix}_max']
    mean_best = grp_mean.transform('max')
    mean_mean = grp_mean.transform('mean')
    mean_std = grp_mean.transform('std').replace(0.0, np.nan)
    max_best = grp_max.transform('max')
    max_mean = grp_max.transform('mean')
    max_std = grp_max.transform('std').replace(0.0, np.nan)

    df[f'{score_prefix}_mean_rank'] = grp_mean.rank(ascending=False, method='min', na_option='bottom')
    df[f'{score_prefix}_mean_diff_from_best'] = df[f'{score_prefix}_mean'] - mean_best
    df[f'{score_prefix}_mean_zscore'] = (df[f'{score_prefix}_mean'] - mean_mean) / mean_std
    df[f'{score_prefix}_mean_zscore'] = df[f'{score_prefix}_mean_zscore'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df[f'{score_prefix}_max_rank'] = grp_max.rank(ascending=False, method='min', na_option='bottom')
    df[f'{score_prefix}_max_diff_from_best'] = df[f'{score_prefix}_max'] - max_best
    df[f'{score_prefix}_max_zscore'] = (df[f'{score_prefix}_max'] - max_mean) / max_std
    df[f'{score_prefix}_max_zscore'] = df[f'{score_prefix}_max_zscore'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    new_cols.extend([
        f'{group_key}_n_models',
        f'{score_prefix}_mean', f'{score_prefix}_std', f'{score_prefix}_min',
        f'{score_prefix}_max', f'{score_prefix}_range', f'{score_prefix}_mean_per_char',
        f'{score_prefix}_mean_rank', f'{score_prefix}_mean_diff_from_best',
        f'{score_prefix}_mean_zscore', f'{score_prefix}_max_rank',
        f'{score_prefix}_max_diff_from_best', f'{score_prefix}_max_zscore',
    ])

  if rank_cols:
    best_vote_cols = []
    for col in rank_cols:
      vote_col = f'{col}_{group_key}_is_best'
      df[vote_col] = (df[col] == 0).astype(int)
      best_vote_cols.append(vote_col)
    df[f'{group_key}_beam_best_vote_count'] = df[best_vote_cols].sum(axis=1)
    df[f'{group_key}_beam_best_vote_frac'] = df[f'{group_key}_beam_best_vote_count'] / max(len(rank_cols), 1)
    grp_votes = df.groupby('uid')[f'{group_key}_beam_best_vote_count']
    vote_best = grp_votes.transform('max')
    vote_second = grp_votes.transform(lambda s: s.nlargest(2).iloc[-1] if len(s) >= 2 else s.max())
    df[f'{group_key}_beam_best_vote_is_top'] = (
        (df[f'{group_key}_beam_best_vote_count'] > 0) &
        np.isclose(df[f'{group_key}_beam_best_vote_count'], vote_best)
    ).astype(int)
    df[f'{group_key}_beam_best_vote_margin'] = df[f'{group_key}_beam_best_vote_count'] - vote_second
    df.loc[df[f'{group_key}_beam_best_vote_is_top'] == 0, f'{group_key}_beam_best_vote_margin'] = 0.0
    new_cols.extend([
        f'{group_key}_beam_best_vote_count', f'{group_key}_beam_best_vote_frac',
        f'{group_key}_beam_best_vote_is_top', f'{group_key}_beam_best_vote_margin',
    ])

  if primary_texts:
    normalize_text = _get_runtime_text_normalizer()
    primary_unique_map = {}
    hit_cols = []
    for uid in df['uid'].drop_duplicates().tolist():
      texts = {
          normalize_text(text)
          for text in (primary_texts.get(mn, {}).get(uid, '') for mn in subset_model_names)
          if normalize_text(text)
      }
      primary_unique_map[uid] = float(len(texts))
    df[f'{group_key}_primary_unique_count'] = df['uid'].map(primary_unique_map).fillna(0.0)
    new_cols.append(f'{group_key}_primary_unique_count')

    for mn in subset_model_names:
      uid_to_text = primary_texts.get(mn, {})
      hit_col = f'is_{group_key}_primary_pred_{mn}'
      text_map = df['uid'].map(uid_to_text)
      df[hit_col] = ((text_map.notna()) & (df['candidate_text'] == text_map)).astype(int)
      hit_cols.append(hit_col)
      new_cols.append(hit_col)

    if hit_cols:
      df[f'{group_key}_primary_hit_count'] = df[hit_cols].sum(axis=1)
      df[f'{group_key}_primary_hit_frac'] = df[f'{group_key}_primary_hit_count'] / max(len(hit_cols), 1)
      grp_hits = df.groupby('uid')[f'{group_key}_primary_hit_count']
      hit_best = grp_hits.transform('max')
      hit_second = grp_hits.transform(lambda s: s.nlargest(2).iloc[-1] if len(s) >= 2 else s.max())
      df[f'{group_key}_primary_hit_is_top'] = (
          (df[f'{group_key}_primary_hit_count'] > 0) &
          np.isclose(df[f'{group_key}_primary_hit_count'], hit_best)
      ).astype(int)
      df[f'{group_key}_primary_hit_margin'] = df[f'{group_key}_primary_hit_count'] - hit_second
      df.loc[df[f'{group_key}_primary_hit_is_top'] == 0, f'{group_key}_primary_hit_margin'] = 0.0
      new_cols.extend([
          f'{group_key}_primary_hit_count', f'{group_key}_primary_hit_frac',
          f'{group_key}_primary_hit_is_top', f'{group_key}_primary_hit_margin',
      ])

  return df, new_cols


def _augment_family_group_features(df, feat_cols, model_names, family_name, subset_model_names,
                                   primary_texts=None):
  if not subset_model_names:
    return df, feat_cols

  subset_set = set(subset_model_names)
  other_model_names = [mn for mn in model_names if mn not in subset_set]
  df, subset_cols = _augment_model_subset_feature_frame(
      df,
      group_name=family_name,
      subset_model_names=subset_model_names,
      primary_texts=primary_texts,
  )
  feat_cols = list(dict.fromkeys(list(feat_cols) + list(subset_cols)))

  if other_model_names:
    other_group_name = f'non{family_name}'
    df, other_cols = _augment_model_subset_feature_frame(
        df,
        group_name=other_group_name,
        subset_model_names=other_model_names,
    )
    gap_cols = []
    for left, right, gap_name in [
        (f'{family_name}_ctc_score_mean', f'{other_group_name}_ctc_score_mean',
         f'{family_name}_vs_{other_group_name}_ctc_score_mean_gap'),
        (f'{family_name}_ctc_score_max', f'{other_group_name}_ctc_score_max',
         f'{family_name}_vs_{other_group_name}_ctc_score_max_gap'),
        (f'{family_name}_beam_best_vote_count', f'{other_group_name}_beam_best_vote_count',
         f'{family_name}_vs_{other_group_name}_beam_best_vote_gap'),
    ]:
      if left in df.columns and right in df.columns:
        df[gap_name] = df[left] - df[right]
        gap_cols.append(gap_name)
    feat_cols = list(dict.fromkeys(list(feat_cols) + list(other_cols) + list(gap_cols)))

  return df, feat_cols


def _build_runtime_model_ctc_meta(model):
  meta = {}
  if not getattr(model, 'use_ctc', False):
    return meta

  normalize_text = _get_runtime_text_normalizer()

  blank_id = int(getattr(model, 'ctc_blank_id', 0) or 0)
  if getattr(model, '_ctc_char_level', False):
    meta['blank_id'] = blank_id
    return meta

  tokenizer = getattr(model, 'tokenizer', None)
  nemo_tokenizer = getattr(model, '_nemo_tokenizer', None)
  if tokenizer is None and nemo_tokenizer is None:
    return meta

  def _decode_ids_to_text(token_ids):
    token_ids = [int(token_id) for token_id in token_ids]
    if tokenizer is not None:
      text = model._tokenizer_batch_decode([token_ids], skip_special_tokens=True)[0]
    else:
      text = nemo_tokenizer.ids_to_text(token_ids) if token_ids else ''
    return normalize_text(text)

  def _text_to_ids(text):
    text = normalize_text(text)
    if nemo_tokenizer is not None:
      return nemo_tokenizer.text_to_ids(text)
    return tokenize_text(tokenizer, text)

  meta['blank_id'] = blank_id
  meta['decode_ids_to_text'] = _decode_ids_to_text
  meta['text_to_ids'] = _text_to_ids
  return meta


def _ensemble_infer_priority(model_dir):
  flags_data = _read_model_flags_for_runtime(model_dir)
  model_name = str(flags_data.get('model', '') or '').strip().lower()
  backbone = str(flags_data.get('backbone', '') or '').strip().lower()
  decode_method = str(flags_data.get('decode_method', '') or '').strip().lower()
  s2s_decoder = str(flags_data.get('s2s_decoder', '') or '').strip().lower()
  try:
    ctc_weight = float(flags_data.get('ctc_weight', 0) or 0)
  except Exception:
    ctc_weight = 0.0

  is_wavlm_family = any(tag in backbone for tag in ('wavlm', 'wav2vec2', 'hubert', 'w2v-bert'))
  if is_wavlm_family or model_name == 'wav2vec2':
    return 0, 'wavlm_ctc'

  is_tdt = (
      model_name == 'nemo'
      and ctc_weight < 1.0
      and (
          (decode_method == 'tdt' and s2s_decoder == 'tdt_reuse')
          or (decode_method in ('auto', '') and 'tdt' in backbone)
      )
  )
  if is_tdt:
    return 1, 'tdt'

  is_nemo_ctc = (model_name == 'nemo' and ctc_weight >= 1.0)
  if is_nemo_ctc:
    return 2, 'nemo_ctc'

  is_nemo = (model_name == 'nemo')
  if is_nemo:
    return 2, 'nemo_s2s'

  return 3, 'other'


def _order_ensemble_model_dirs(model_dirs):
  indexed = []
  for idx, model_dir in enumerate(model_dirs):
    _, group = _ensemble_infer_priority(model_dir)
    indexed.append((idx, group, model_dir))
  ordered = [item[2] for item in indexed]
  order_log = ', '.join(f'{item[2].name}:{item[1]}' for item in indexed)
  return ordered, order_log

if IS_ENSEMBLE:
  with open(ENSEMBLE_META_FILE) as _ef:
    _ensemble_meta = json.load(_ef)
  ENSEMBLE_MODEL_DIRS = [SRC_DIR / d for d in _ensemble_meta.get('model_dirs', [])]
  _diag(f'ENSEMBLE mode: {len(ENSEMBLE_MODEL_DIRS)} models: {", ".join(d.name for d in ENSEMBLE_MODEL_DIRS)}')
  ENSEMBLE_INFER_MODEL_DIRS, _ensemble_order_log = _order_ensemble_model_dirs(ENSEMBLE_MODEL_DIRS)
  _diag(f'ENSEMBLE infer order: {", ".join(d.name for d in ENSEMBLE_INFER_MODEL_DIRS)}')
  _diag(f'ENSEMBLE infer groups: {_ensemble_order_log}')
else:
  _ensemble_meta = {}
  _diag('Single model mode')
  ENSEMBLE_INFER_MODEL_DIRS = ENSEMBLE_MODEL_DIRS

MODEL_AUDIO_FILTERS = _load_runtime_model_audio_filters() if IS_ENSEMBLE else {}
if MODEL_AUDIO_FILTERS:
  _diag('Model audio filters: ' + ', '.join(
      f'{model_name}={rule}' for model_name, rule in sorted(MODEL_AUDIO_FILTERS.items())))

if PROBE_MODE:
  _diag('PROBE mode enabled: will collect extra runtime/model statistics')

def setup_flags(log_final=True):
  """Restore training FLAGS from model directory and override for inference.
  
  Follows the same proven order as Kaggle submit notebooks:
    1. gz.init_flags() + config.init()   — initialize defaults
    2. gz.restore_configs(model_dir)     — restore training flags
    3. fix_flags()                       — override for inference
  """
  # Load flags.json from model dir (for manual fallback)
  flags_file = MODEL_DIR / 'flags.json'
  if flags_file.exists():
    with open(flags_file) as f:
      saved_flags = json.load(f)
    _log.debug(f'Loaded flags.json with {len(saved_flags)} entries')
  else:
    saved_flags = {}
    _diag('WARNING: No flags.json found')

  # ---- Step 1: Initialize defaults (same as Kaggle) ----
  try:
    gz.init_flags()
  except Exception as e:
    _log.debug(f'gz.init_flags() failed: {e}')
  
  try:
    config.init()
  except Exception as e:
    _log.debug(f'config.init() failed: {e}, manually initializing')
    from src.config_base import infer_model_from_backbone, BACKBONES
    if FLAGS.backbone in BACKBONES:
      FLAGS.backbone = BACKBONES[FLAGS.backbone]
    FLAGS.model = infer_model_from_backbone(FLAGS.backbone)
  
  # ---- Step 2: Restore training flags (same as Kaggle) ----
  try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
      gz.restore_configs(str(MODEL_DIR))
  except Exception as e:
    _log.debug(f'gz.restore_configs() raised: {e}')
  
  # Manual fallback: force-set critical flags from flags.json using absl API
  # (needed if flags.pkl missing and restore_flags_from_json failed/partial)
  critical_flags = [
      'backbone', 'model', 'ctc_weight', 'constrain_ipa', 'nemo_native_ctc',
      'max_audio_sec', 'sample_rate', 'max_new_tokens', 'num_beams',
      'score_metric', 'label_column', 'train_file',
      'decode_method', 'ctc_beam_width', 'joint_ctc_decode_weight',
      's2s_decoder', 'inter_ctc', 'inter_ctc_layers',
      'ipa_weight', 'word_weight', 'use_cross_labels',
      'word_ctc', 'word_detach_encoder',
      'aux_loss', 'aux_age_weight', 'aux_age_mode', 'aux_domain_weight', 'aux_pool',
      'length_penalty',
      'nemo_adapter', 'adapter_dim', 'adapter_name', 'adapter_module_name',
      'whisper_lora', 'lora_r', 'lora_alpha', 'lora_dropout', 'lora_target_modules',
  ]
  for key in critical_flags:
    if key in saved_flags:
      try:
        if key in FLAGS:
          FLAGS[key].value = saved_flags[key]
        else:
          setattr(FLAGS, key, saved_flags[key])
      except Exception as e:
        _log.debug(f'Failed to set flag {key}={saved_flags[key]}: {e}')

  # Some older checkpoints predate word_tokenizer-related flags.
  # Reset them to registry defaults when missing so later ensemble models
  # don't inherit tokenizer settings from earlier models.
  for key in ['word_tokenizer', 'default_word_tokenizer']:
    if key not in FLAGS:
      continue
    try:
      value = saved_flags[key] if key in saved_flags else FLAGS[key].default
      FLAGS[key].value = value
    except Exception as e:
      _log.debug(f'Failed to restore optional flag {key}: {e}')
  
  # ---- Step 3: Override for inference (same as Kaggle fix_flags) ----
  FLAGS.root = str(DATA_DIR)
  FLAGS.model_dir = str(MODEL_DIR)
  FLAGS.torch = True
  FLAGS.torch_only = True
  FLAGS.restore_configs = False
  FLAGS.distributed = False
  FLAGS.train_allnew = False
  FLAGS.grad_acc = 1
  
  eval_bs = saved_flags.get('eval_batch_size', 8)
  # Scale batch_size for runtime GPU — adaptive to model type
  # Actual param-aware scaling is done in _scale_batch_size() after model load
  import torch as _torch
  if _torch.cuda.is_available():
    _gpu_mem_gb = _torch.cuda.get_device_properties(0).total_memory / (1024**3)
    _model_type = saved_flags.get('model', '')
    if any(k in _model_type for k in ('ctc', 'squeezeformer', 'wav2vec2')):
      _scale = max(1.0, _gpu_mem_gb / 6.0)
    else:
      _scale = max(1.0, _gpu_mem_gb / 24.0)
    eval_bs = min(256, max(eval_bs, int(eval_bs * _scale)))
  requested_eval_bs = eval_bs
  FLAGS.eval_batch_size = eval_bs
  FLAGS.batch_size = eval_bs
  try:
    mt.set_global('eval_batch_size', eval_bs)
  except Exception:
    pass
  
  FLAGS.mode = 'test'
  FLAGS.work_mode = 'test'
  FLAGS.pymp = False
  FLAGS.num_workers = 0
  FLAGS.persistent_workers = False
  FLAGS.pin_memory = False
  FLAGS.num_gpus = 1
  fast_infer_env = os.environ.get('FAST_INFER')
  if fast_infer_env is None:
    FLAGS.fast_infer = bool(saved_flags.get(
        'fast_infer', getattr(FLAGS, 'fast_infer', False)))
  else:
    FLAGS.fast_infer = fast_infer_env.strip().lower() not in ('0', 'false', 'no')
  infer_extra_heads_env = os.environ.get('INFER_EXTRA_HEADS')
  if infer_extra_heads_env is None:
    FLAGS.infer_extra_heads = bool(saved_flags.get(
        'infer_extra_heads', getattr(FLAGS, 'infer_extra_heads', True)))
  else:
    FLAGS.infer_extra_heads = infer_extra_heads_env.strip().lower() not in ('0', 'false', 'no')
  keep_pred_meta = os.environ.get(
      'SUBMIT_KEEP_PRED_META', '').strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')
  global _pred_meta_disabled_logged
  if not keep_pred_meta:
    if (not _pred_meta_disabled_logged and
        (bool(getattr(FLAGS, 'save_pred_score', False)) or int(getattr(FLAGS, 'save_pred_nbest', 0) or 0) > 0)):
      _diag(f'Disable pred metadata export for submission runtime: '
            f'save_pred_score={getattr(FLAGS, "save_pred_score", False)}, '
            f'save_pred_nbest={getattr(FLAGS, "save_pred_nbest", 0)}')
      _pred_meta_disabled_logged = True
    FLAGS.save_pred_score = False
    FLAGS.save_pred_nbest = 0
  
  # Final verification: print all critical flags
  if log_final:
    _diag(f'[FINAL] backbone={FLAGS.backbone}, model={FLAGS.model}, '
        f'requested_eval_bs={requested_eval_bs}, '
          f'decode_method={getattr(FLAGS, "decode_method", "auto")}, '
          f'ctc_weight={getattr(FLAGS, "ctc_weight", 0)}, '
          f'nemo_native_ctc={getattr(FLAGS, "nemo_native_ctc", False)}, '
          f'infer_extra_heads={getattr(FLAGS, "infer_extra_heads", True)}')
  if PROBE_MODE:
    _diag(f'[AUX] aux_age_weight={getattr(FLAGS, "aux_age_weight", 0)}, '
          f'aux_domain_weight={getattr(FLAGS, "aux_domain_weight", 0)}, '
          f'aux_age_mode={getattr(FLAGS, "aux_age_mode", None)}, '
          f'aux_pool={getattr(FLAGS, "aux_pool", None)}, '
          f'fast_infer={getattr(FLAGS, "fast_infer", False)}, '
          f'infer_extra_heads={getattr(FLAGS, "infer_extra_heads", True)}')
  
  # SAFETY CHECK: if ctc_weight should be > 0 (from flags.json) but got reset
  json_ctc = saved_flags.get('ctc_weight', None)
  actual_ctc = getattr(FLAGS, 'ctc_weight', 0.0)
  if json_ctc is not None and json_ctc > 0 and actual_ctc == 0:
    _diag(f'SAFETY: ctc_weight reset to 0, forcing to {json_ctc}')
    FLAGS.ctc_weight = json_ctc
  return saved_flags

# ===========================================================================
#  Phase 6: Load model & predict
# ===========================================================================

def _find_best_weights(model_dir):
  """Find the best checkpoint file in model_dir, preferring best.pt > model.pt."""
  import glob
  files = (glob.glob(f'{model_dir}/*.pt') +
           glob.glob(f'{model_dir}/*.bin') +
           glob.glob(f'{model_dir}/*.tar'))
  if not files:
    return None
  # Sort by modification time (oldest first)
  files = sorted(files, key=lambda x: os.path.getmtime(x))
  _log.debug(f'Weight files found: {[os.path.basename(f) for f in files]}')
  # Prefer best.pt > model.pt > newest other file
  for preferred in ('best.pt', 'model.pt'):
    for f in files:
      if os.path.basename(f) == preferred:
        return f
  # Otherwise newest
  return files[-1]


def _load_weights(model, weight_path, strict=False):
  """Standalone load_weights — mirrors lele.load_weights without tensorflow.
  
  Handles:
    - dill-pickled checkpoints (torch.save with pickle_module=dill)
    - EMA weights (EMAv2/EMAv3) 
    - _orig_mod. prefix stripping (torch.compile)
    - Shape-mismatch tolerance (strict=False)
  """
  import torch
  try:
    import dill
  except ImportError:
    dill = None
  
  # Load checkpoint — try dill first (training saves with dill), then plain
  if dill is not None:
    try:
      checkpoint = torch.load(weight_path, map_location='cpu', pickle_module=dill)
    except Exception:
      checkpoint = torch.load(weight_path, map_location='cpu')
  else:
    checkpoint = torch.load(weight_path, map_location='cpu')
  
  # Extract state_dict
  if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state = checkpoint['state_dict']
  elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state = checkpoint['model_state_dict']
  else:
    state = checkpoint  # raw state_dict
  
  # Prefer EMA weights if available
  _ema_sd_used = None  # track for post-load alias fixup
  if isinstance(checkpoint, dict):
    if 'EMAv2' in checkpoint and isinstance(checkpoint['EMAv2'], dict):
      ema_state = checkpoint['EMAv2'].get('ema_state_dict')
      if ema_state:
        state = ema_state
        _ema_sd_used = ema_state
        _log.debug('Using EMAv2 weights from checkpoint')
    elif 'EMAv3' in checkpoint and isinstance(checkpoint['EMAv3'], dict):
      ema_state = checkpoint['EMAv3'].get('ema_state_dict')
      if ema_state:
        state = ema_state
        _ema_sd_used = ema_state
        _log.debug('Using EMAv3 weights from checkpoint')
  
  # Strip _orig_mod. prefix (torch.compile artifacts)
  unwanted_prefix = '_orig_mod.'
  for k in list(state.keys()):
    if k.startswith(unwanted_prefix):
      state[k[len(unwanted_prefix):]] = state.pop(k)
  
  model_ = model.module if hasattr(model, 'module') else model
  mismatch_keys = set()
  
  if strict:
    model_.load_state_dict(state, strict=True)
  else:
    model_state = model_.state_dict()
    
    # Find mismatches (shape or missing keys)
    for key in model_state:
      if key not in state or state[key].shape != model_state[key].shape:
        mismatch_keys.add(key)
    extra_keys = set()
    for key in state:
      if key not in model_state:
        extra_keys.add(key)
    
    if mismatch_keys:
      _log.debug(f'{len(mismatch_keys)} weights not loaded (missing/shape mismatch)')
      _log.debug(f'Mismatch keys: {mismatch_keys}')
    if extra_keys:
      _log.debug(f'{len(extra_keys)} extra keys in checkpoint (not in model)')
      _log.debug(f'Extra keys: {extra_keys}')
    
    if not mismatch_keys and not extra_keys:
      model_.load_state_dict(state)
    else:
      new_state = model_state.copy()
      new_state.update({
        k: v for k, v in state.items()
        if k in new_state and k not in mismatch_keys
      })
      model_.load_state_dict(new_state)
  
  # Post-load EMA fixup: when EMA state dict has fewer keys than the model
  # (e.g. old checkpoints missing aliased module keys like backbone.*, encoder.*),
  # load_state_dict may overwrite EMA values with non-EMA values for aliased params.
  # Force-copy EMA values via named_parameters to ensure consistency.
  if _ema_sd_used is not None and mismatch_keys:
    fixed = 0
    for name, param in model_.named_parameters():
      if name in _ema_sd_used:
        ema_val = _ema_sd_used[name]
        if ema_val.device != param.device:
          ema_val = ema_val.to(param.device)
        param.data.copy_(ema_val)
        fixed += 1
    for name, buf in model_.named_buffers():
      if name in _ema_sd_used:
        ema_val = _ema_sd_used[name]
        if ema_val.device != buf.device:
          ema_val = ema_val.to(buf.device)
        buf.data.copy_(ema_val)
        fixed += 1
    if fixed > 0:
      _log.debug(f'EMA fixup applied to {fixed} params/buffers (aliased param consistency)')

  n_loaded = sum(1 for k in state if k in model_.state_dict() and k not in mismatch_keys)
  _log.debug(f'Loaded {n_loaded} weight tensors from {os.path.basename(weight_path)}')


def load_model(device, verbose=True):
  """Create model from training code and load saved weights.
  
  Follows the same pattern as Kaggle submit:
    model = util.get_model()
    gz.load_weights(model, model_dir, strict=False)
  """
  # Use importlib like train.py (nemo not in models/__init__.py since it's heavy)
  import importlib
  if verbose:
    _diag(f'  Building model class: {FLAGS.model} ({Path(MODEL_DIR).name})')
  model_module = importlib.import_module(f'src.models.{FLAGS.model}')
  Model = model_module.Model
  model = Model()
  
  # Load weights — try gz.load_weights first (same as Kaggle), fallback to custom
  weight_path = _find_best_weights(str(MODEL_DIR))
  if weight_path:
    if verbose:
      _diag(f'  Loading checkpoint: {Path(weight_path).name}')
    _log.debug(f'Loading weights from: {weight_path}')
    try:
      gz.load_weights(model, weight_path, strict=False)
      _log.debug('gz.load_weights() succeeded')
    except Exception as e:
      _log.debug(f'gz.load_weights() failed: {e}, using custom _load_weights')
      _load_weights(model, weight_path, strict=False)
  else:
    _log.debug(f'No .pt/.bin/.tar weight files found in {MODEL_DIR}, '
               f'using NeMo pretrained weights only')
  
  model = model.to(device).eval()
  n_params_m = sum(p.numel() for p in model.parameters()) / 1e6
  ckpt_name = Path(weight_path).name if weight_path else 'pretrained_only'
  if verbose:
    _diag(f'  Load success: checkpoint={ckpt_name}, class={model.__class__.__name__}, '
          f'params={n_params_m:.0f}M, device={device}')
    if getattr(FLAGS, 'model', '') == 'nemo':
      _diag(f'  NeMo ready: decode_method={getattr(FLAGS, "decode_method", "auto")}, '
            f's2s_decoder={getattr(FLAGS, "s2s_decoder", "native")}, '
            f'ctc_weight={getattr(FLAGS, "ctc_weight", 0)}, '
            f'nemo_native_ctc={getattr(FLAGS, "nemo_native_ctc", False)}')
  
  # Ensure fast batched decoding for NeMo TDT/RNNT models.
  # Training may have set slow sequential decoder (greedy, non-batched)
  # for stability. For inference we want greedy_batch + CUDA graphs (~2-3x faster).
  enable_fast_decode = os.environ.get('SUBMIT_ENABLE_FAST_DECODE', '1').strip().lower() not in ('0', 'false', 'no')
  if enable_fast_decode and hasattr(model, '_enable_fast_decode'):
    model._enable_fast_decode()
    if verbose and getattr(FLAGS, 'model', '') == 'nemo':
      _diag('  NeMo fast decode enabled')
  elif verbose and getattr(FLAGS, 'model', '') == 'nemo':
    _diag('  NeMo fast decode disabled')
  
  return model


def _scale_batch_size(model, verbose=True):
  """Re-scale batch_size after model is loaded, using actual param count.
  This overrides the heuristic from setup_flags() with a more accurate estimate.
  Set env FORCE_BATCH_SIZE=N to skip auto-scaling and use a fixed batch size."""
  import torch, os
  force_bs = os.environ.get('FORCE_BATCH_SIZE', '')
  if force_bs:
    bs = int(force_bs)
    FLAGS.eval_batch_size = bs
    FLAGS.batch_size = bs
    try:
      import melt as mt
      mt.set_global('eval_batch_size', bs)
    except Exception:
      pass
    if verbose:
      _diag(f'Batch size forced via FORCE_BATCH_SIZE={bs}')
    return
  if not torch.cuda.is_available():
    return
  try:
    n_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    model_mem_gb = n_params_m * 2 / 1024  # rough fp16 model footprint
    avail_gb = gpu_mem_gb - model_mem_gb - 2.0  # 2GB OS/framework overhead
    avail_gb = max(avail_gb, 2.0)
    # Factor in max audio duration: longer audio → more memory per sample
    # NeMo conformer self-attention scales O(T^2) with sequence length
    max_dur = getattr(FLAGS, 'max_audio_sec', 15) or 15
    _model_type = getattr(FLAGS, 'model', '')
    _backbone = getattr(FLAGS, 'backbone', '')
    _runtime_key = f'{_model_type} {_backbone}'.lower()
    _is_wavlm_family = any(k in _runtime_key for k in ('wavlm', 'wav2vec2', 'hubert', 'w2v-bert'))
    if any(k in _model_type for k in ('ctc', 'squeezeformer', 'wav2vec2')):
      # Encoder-only CTC: light memory footprint, ~linear in duration
      dur_factor = max(1.0, max_dur / 10.0)
      per_sample_gb = max(0.02, model_mem_gb / 100) * dur_factor
    elif 'nemo' in _model_type:
      # NeMo conformer: heavy self-attention, O(T^2) in audio length
      # Reference: 600M params, max_audio=30s, bs=8 works on 24GB
      dur_factor = (max(1.0, max_dur / 10.0)) ** 1.5
      per_sample_gb = max(0.15, model_mem_gb / 8) * dur_factor
    else:
      # Whisper/other S2S: moderate scaling
      dur_factor = max(1.0, max_dur / 15.0)
      per_sample_gb = max(0.1, model_mem_gb / 20) * dur_factor
    safe_bs = max(1, int(avail_gb / per_sample_gb))
    safe_bs = min(safe_bs, 256)
    if _is_wavlm_family:
      # WavLM / wav2vec2 front-end activations are much heavier than the
      # old generic CTC heuristic suggests. But long-audio protection is
      # already handled later by runtime duration-based splitting using the
      # actual batch max duration, so the global init batch here should not be
      # penalized again by training-time max_audio_sec.
      # Empirical target: ~8 on 24GB, ~32 on 96GB for typical submit workloads.
      wavlm_cap = max(1, int(gpu_mem_gb / 3.0))
      safe_bs = min(safe_bs, wavlm_cap)
      safe_bs = max(safe_bs, wavlm_cap)
    # NeMo TDT/RNNT autoregressive decode has diminishing returns with very
    # large batches (more padding waste, decode is sequential per token).
    # Benchmark: bs=26 → 49 utt/s vs bs=96 → 25 utt/s on A100 80GB.
    if 'nemo' in _model_type:
      _ctc_only = getattr(FLAGS, 'ctc_weight', 0) >= 1.0 or getattr(FLAGS, 'nemo_native_ctc', False)
      if _ctc_only:
        safe_bs = min(safe_bs, 128)  # CTC encoder-only batches well
      else:
        safe_bs = min(safe_bs, 32)   # TDT/RNNT decode bottleneck
    old_bs = FLAGS.eval_batch_size
    if safe_bs != old_bs:
      FLAGS.eval_batch_size = safe_bs
      FLAGS.batch_size = safe_bs
      try:
        import melt as mt
        mt.set_global('eval_batch_size', safe_bs)
      except Exception:
        pass
      if verbose:
        _diag(f'Batch size adjusted: {old_bs} -> {safe_bs} '
          f'(params={n_params_m:.0f}M, model_mem={model_mem_gb:.1f}GB, '
          f'gpu={gpu_mem_gb:.0f}GB, avail={avail_gb:.1f}GB)')
  except Exception as e:
    _log.debug(f'_scale_batch_size failed: {e}')


def _softmax_np(values):
  import numpy as np
  arr = np.asarray(values, dtype=np.float64)
  if arr.size == 0:
    return arr
  arr = arr - np.max(arr)
  exp_arr = np.exp(arr)
  denom = exp_arr.sum()
  if denom <= 0:
    return np.full_like(arr, 1.0 / max(len(arr), 1))
  return exp_arr / denom


def _ordinal_age_probs(logits):
  import numpy as np
  arr = np.asarray(logits, dtype=np.float64).reshape(-1)
  if arr.size != 3:
    return np.full(4, 0.25, dtype=np.float64)
  cum = 1.0 / (1.0 + np.exp(-arr))
  probs = np.array([
      1.0 - cum[0],
      cum[0] - cum[1],
      cum[1] - cum[2],
      cum[2],
  ], dtype=np.float64)
  probs = np.clip(probs, 0.0, 1.0)
  total = probs.sum()
  if total <= 0:
    return np.full(4, 0.25, dtype=np.float64)
  return probs / total


def _summarize_numeric(values):
  import numpy as np
  arr = np.asarray(list(values), dtype=np.float64)
  if arr.size == 0:
    return {'count': 0}
  return {
      'count': int(arr.size),
      'min': float(np.min(arr)),
      'mean': float(np.mean(arr)),
      'p50': float(np.percentile(arr, 50)),
      'p90': float(np.percentile(arr, 90)),
      'max': float(np.max(arr)),
  }


def _format_summary(summary, keys=('min', 'mean', 'p50', 'p90', 'max'), precision=2):
  if not summary or summary.get('count', 0) == 0:
    return 'n=0'
  parts = [f'n={summary["count"]}']
  for key in keys:
    if key in summary:
      parts.append(f'{key}={summary[key]:.{precision}f}')
  return ', '.join(parts)


def _prediction_summary(predictions):
  texts = list(predictions.values())
  non_empty = [text for text in texts if str(text or '').strip()]
  avg_chars = sum(len(text) for text in non_empty) / len(non_empty) if non_empty else 0.0
  max_chars = max((len(text) for text in texts), default=0)
  return {
      'count': int(len(texts)),
      'non_empty': int(len(non_empty)),
      'empty': int(len(texts) - len(non_empty)),
      'avg_chars_non_empty': float(avg_chars),
      'max_chars': int(max_chars),
  }


def _infer_runtime_profile(model, device):
  import torch
  sample_rate = getattr(FLAGS, 'sample_rate', 16000)
  runtime_key = f'{getattr(FLAGS, "model", "")} {getattr(FLAGS, "backbone", "")}'.lower()
  is_wavlm_family = any(k in runtime_key for k in ('wavlm', 'wav2vec2', 'hubert', 'w2v-bert'))
  gpu_total_gb = 0.0
  model_mem_gb = 0.0
  gpu_avail_gb = 0.0
  if torch.cuda.is_available():
    gpu_total_gb = torch.cuda.get_device_properties(device).total_memory / (1 << 30)
    model_mem_gb = sum(p.nbytes for p in model.parameters()) / (1 << 30)
    gpu_avail_gb = gpu_total_gb - model_mem_gb - 3.0
  return {
      'sample_rate': sample_rate,
      'runtime_key': runtime_key,
      'is_wavlm_family': is_wavlm_family,
      'gpu_total_gb': gpu_total_gb,
      'model_mem_gb': model_mem_gb,
      'gpu_avail_gb': gpu_avail_gb,
      'max_trunc_sec': 30.0,
  }


def _duration_safe_bs(max_dur_sec, base_bs, profile):
  safe = base_bs
  if profile['is_wavlm_family']:
    ref_dur = max(max_dur_sec, 15.0)
    wavlm_cap = max(1, int(profile['gpu_total_gb'] / 3.0)) if profile['gpu_total_gb'] > 0 else base_bs
    dur_penalty = max(1.0, (ref_dur / 15.0) ** 1.15)
    safe = min(safe, max(1, int(wavlm_cap / dur_penalty)))
  if 'nemo' in profile['runtime_key'] and max_dur_sec > 20 and profile['gpu_avail_gb'] > 0:
    k = 0.0006
    mem_per_sample = k * max_dur_sec ** 2
    safe = min(safe, max(1, int(0.85 * profile['gpu_avail_gb'] / mem_per_sample)))
  return max(1, min(base_bs, safe))


def _get_batch_max_dur(batch, sample_rate):
  import torch
  if 'input_features' in batch:
    feat = batch['input_features']
    if isinstance(feat, torch.Tensor):
      return feat.shape[-1] / sample_rate
  return 0.0


def _slice_batch(batch, start, end, device):
  import torch
  sliced = {}
  for key, value in batch.items():
    if isinstance(value, torch.Tensor):
      sliced[key] = value[start:end].to(device, non_blocking=True)
    elif isinstance(value, list):
      sliced[key] = value[start:end]
    else:
      sliced[key] = value
  return sliced


def _truncate_single_batch(single_batch, sample_rate, max_trunc_sec):
  import torch
  max_samples = int(max_trunc_sec * sample_rate)
  trunc = {}
  for key, value in single_batch.items():
    if key in ('input_features', 'attention_mask') and isinstance(value, torch.Tensor) and value.shape[-1] > max_samples:
      trunc[key] = value[..., :max_samples]
    else:
      trunc[key] = value
  return trunc


def _cuda_ok(device):
  import torch
  try:
    torch.cuda.synchronize(device)
    temp = torch.zeros(1, device=device)
    del temp
    return True
  except Exception:
    return False


def _init_probe_state(df):
  import pandas as pd

  uid_col = 'id' if 'id' in df.columns else ('utterance_id' if 'utterance_id' in df.columns else None)
  meta_by_uid = {}
  if uid_col is not None:
    for _, row in df.iterrows():
      uid = row[uid_col]
      audio_dur = None
      if 'audio_duration_sec' in row.index and pd.notna(row.get('audio_duration_sec')):
        try:
          audio_dur = float(row.get('audio_duration_sec'))
        except Exception:
          audio_dur = None
      age_bucket = ''
      if 'age_bucket' in row.index and pd.notna(row.get('age_bucket')):
        age_bucket = str(row.get('age_bucket')).strip()
      source = ''
      if 'source' in row.index and pd.notna(row.get('source')):
        source = str(row.get('source')).strip()
      meta_by_uid[uid] = {
          'audio_duration_sec': audio_dur,
          'age_bucket': age_bucket,
          'source': source,
      }

  def _count_non_empty(col):
    if col not in df.columns:
      return 0
    return int(df[col].fillna('').astype(str).str.strip().ne('').sum())

  return {
      'uid_col': uid_col,
      'n_rows': int(len(df)),
      'meta_by_uid': meta_by_uid,
      'age_mode': getattr(FLAGS, 'aux_age_mode', None) or 'classify',
      'age_probs': {},
      'domain_prob_dd': {},
      'metadata_presence': {
          'age_bucket_non_empty': _count_non_empty('age_bucket'),
          'source_non_empty': _count_non_empty('source'),
          'audio_duration_non_empty': int(df['audio_duration_sec'].notna().sum()) if 'audio_duration_sec' in df.columns else 0,
      },
  }


def _update_probe_aux_outputs(probe_state, batch, res):
  if not probe_state or 'id' not in batch:
    return
  import numpy as np

  ids = list(batch.get('id') or [])
  if not ids:
    return

  age_logits_batch = res.get('aux_age_logits', None)
  age_mode = probe_state.get('age_mode', 'classify')
  if age_logits_batch is not None:
    age_logits_np = age_logits_batch.detach().float().cpu().numpy()
    for uid, logits in zip(ids, age_logits_np):
      if age_mode == 'classify':
        probs = _softmax_np(logits)
      elif age_mode == 'ordinal':
        probs = _ordinal_age_probs(logits)
      else:
        val = float(np.asarray(logits).reshape(-1)[0])
        cls = int(np.digitize([val], bins=[0.25, 0.5, 0.75])[0])
        probs = np.zeros(4, dtype=np.float64)
        probs[min(max(cls, 0), 3)] = 1.0
      probe_state['age_probs'][uid] = [float(x) for x in probs]

  domain_logits_batch = res.get('aux_domain_logits', None)
  if domain_logits_batch is not None:
    domain_logits_np = domain_logits_batch.detach().float().cpu().numpy().reshape(-1)
    domain_probs = 1.0 / (1.0 + np.exp(-domain_logits_np))
    for uid, prob in zip(ids, domain_probs):
      probe_state['domain_prob_dd'][uid] = float(prob)


def _emit_probe_report(predictions, probe_state):
  import numpy as np

  age_labels = ['3-4', '5-7', '8-11', '12+']
  meta_by_uid = probe_state.get('meta_by_uid', {})
  uids = list(predictions.keys())

  pred_lengths = []
  pred_space_counts = []
  chars_per_sec = []
  dur_groups = [
      ('<=2s', 0.0, 2.0),
      ('2-4s', 2.0, 4.0),
      ('4-8s', 4.0, 8.0),
      ('8-12s', 8.0, 12.0),
      ('12-20s', 12.0, 20.0),
      ('20s+', 20.0, None),
  ]
  dur_group_stats = {label: {'n': 0, 'pred_lengths': [], 'space_counts': [], 'chars_per_sec': []}
                     for label, _, _ in dur_groups}

  def _find_dur_group(duration):
    for label, lower, upper in dur_groups:
      if duration is None:
        continue
      if duration >= lower and (upper is None or duration < upper):
        return label
    return None

  durations = []
  empty_preds = 0
  for uid in uids:
    text = str(predictions.get(uid, '') or '')
    pred_len = len(text)
    space_count = text.count(' ')
    pred_lengths.append(pred_len)
    pred_space_counts.append(space_count)
    if not text.strip():
      empty_preds += 1
    meta = meta_by_uid.get(uid, {})
    duration = meta.get('audio_duration_sec')
    if duration is not None:
      durations.append(duration)
      if duration > 0:
        chars_per_sec.append(pred_len / duration)
      label = _find_dur_group(duration)
      if label is not None:
        bucket = dur_group_stats[label]
        bucket['n'] += 1
        bucket['pred_lengths'].append(pred_len)
        bucket['space_counts'].append(space_count)
        if duration > 0:
          bucket['chars_per_sec'].append(pred_len / duration)

  duration_summary = _summarize_numeric(durations)
  pred_len_summary = _summarize_numeric(pred_lengths)
  space_summary = _summarize_numeric(pred_space_counts)
  cps_summary = _summarize_numeric(chars_per_sec)

  metadata_presence = probe_state.get('metadata_presence', {})
  report = {
      'probe_mode': True,
      'n_predictions': int(len(uids)),
      'empty_predictions': int(empty_preds),
      'metadata_presence': metadata_presence,
      'audio_duration_sec': duration_summary,
      'prediction_length_chars': pred_len_summary,
      'prediction_space_count': space_summary,
      'prediction_chars_per_sec': cps_summary,
      'duration_vs_prediction': {},
  }

  _diag(f'[probe] metadata: age_bucket={metadata_presence.get("age_bucket_non_empty", 0)}/{probe_state.get("n_rows", 0)}, '
        f'source={metadata_presence.get("source_non_empty", 0)}/{probe_state.get("n_rows", 0)}, '
        f'audio_duration={metadata_presence.get("audio_duration_non_empty", 0)}/{probe_state.get("n_rows", 0)}')
  _diag(f'[probe] audio_dur_sec: {_format_summary(duration_summary)}')
  _diag(f'[probe] pred_chars: empty={empty_preds}/{len(uids)}, {_format_summary(pred_len_summary, keys=("mean", "p50", "p90", "max"), precision=2)}')
  _diag(f'[probe] pred_spaces: {_format_summary(space_summary, keys=("mean", "p50", "p90", "max"), precision=2)}')
  _diag(f'[probe] pred_chars_per_sec: {_format_summary(cps_summary, keys=("mean", "p50", "p90", "max"), precision=2)}')

  for label, stats in dur_group_stats.items():
    bucket_report = {
        'n': int(stats['n']),
        'pred_chars': _summarize_numeric(stats['pred_lengths']),
        'spaces': _summarize_numeric(stats['space_counts']),
        'chars_per_sec': _summarize_numeric(stats['chars_per_sec']),
    }
    report['duration_vs_prediction'][label] = bucket_report
    _diag(f'[probe] dur_bucket {label}: n={stats["n"]}, '
          f'mean_chars={bucket_report["pred_chars"].get("mean", 0.0):.2f}, '
          f'mean_spaces={bucket_report["spaces"].get("mean", 0.0):.2f}, '
          f'mean_chars_per_sec={bucket_report["chars_per_sec"].get("mean", 0.0):.2f}')

  domain_probs = list(probe_state.get('domain_prob_dd', {}).values())
  if domain_probs:
    arr = np.asarray(domain_probs, dtype=np.float64)
    domain_summary = {
        'count': int(arr.size),
        'mean_dd_prob': float(arr.mean()),
        'mean_ext_prob': float((1.0 - arr).mean()),
        'p10_dd_prob': float(np.percentile(arr, 10)),
        'p50_dd_prob': float(np.percentile(arr, 50)),
        'p90_dd_prob': float(np.percentile(arr, 90)),
        'dd_pred_count': int((arr >= 0.5).sum()),
        'ext_pred_count': int((arr < 0.5).sum()),
        'uncertain_0.4_0.6_count': int(((arr >= 0.4) & (arr <= 0.6)).sum()),
        'bucket_counts': {},
    }
    bucket_counts = {}
    edges = [i / 10.0 for i in range(11)]
    for left, right in zip(edges[:-1], edges[1:]):
      if right < 1.0:
        mask = (arr >= left) & (arr < right)
      else:
        mask = (arr >= left) & (arr <= right)
      key = f'{left:.1f}-{right:.1f}'
      bucket_counts[key] = int(mask.sum())
    domain_summary['bucket_counts'] = bucket_counts
    report['domain_dd_prob'] = domain_summary
    _diag(f'[probe] domain_dd_prob: mean={domain_summary["mean_dd_prob"]:.3f}, '
          f'ext_mean={domain_summary["mean_ext_prob"]:.3f}, '
          f'p10={domain_summary["p10_dd_prob"]:.3f}, '
          f'p50={domain_summary["p50_dd_prob"]:.3f}, '
          f'p90={domain_summary["p90_dd_prob"]:.3f}, '
          f'uncertain_0.4_0.6={domain_summary["uncertain_0.4_0.6_count"]}/{domain_summary["count"]}')
    _diag('[probe] domain_dd_prob_buckets: ' + ', '.join(
        f'{k}={v}' for k, v in bucket_counts.items()))
  else:
    report['domain_dd_prob'] = None
    _diag('[probe] domain_dd_prob: unavailable (no aux domain outputs)')

  age_probs = list(probe_state.get('age_probs', {}).values())
  if age_probs:
    age_arr = np.asarray(age_probs, dtype=np.float64)
    mean_probs = age_arr.mean(axis=0)
    pred_idx = age_arr.argmax(axis=1)
    pred_counts = {age_labels[i]: int((pred_idx == i).sum()) for i in range(len(age_labels))}
    mean_prob_map = {age_labels[i]: float(mean_probs[i]) for i in range(len(age_labels))}
    report['age_pred'] = {
        'mode': probe_state.get('age_mode', 'classify'),
        'count': int(age_arr.shape[0]),
        'pred_counts': pred_counts,
        'mean_probs': mean_prob_map,
    }
    _diag('[probe] age_pred_counts: ' + ', '.join(f'{k}={v}' for k, v in pred_counts.items()))
    _diag('[probe] age_mean_probs: ' + ', '.join(f'{k}={v:.3f}' for k, v in mean_prob_map.items()))
  else:
    report['age_pred'] = None
    _diag('[probe] age_pred: unavailable (no aux age outputs)')

  try:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    PROBE_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=True, indent=2))
    _diag(f'[probe] report_written: {PROBE_REPORT_PATH}')
  except Exception as e:
    _diag(f'[probe] report_write_failed: {e}')


def run_inference(model, device):
  """Run inference using training code's forward() with do_generate=True."""
  import torch
  import numpy as np
  from src.dataset import get_dl
  from src.preprocess import preprocess
  
  # Adaptive batch size based on actual model param count
  _scale_batch_size(model)
  
  # Prepare test data
  df = preprocess(mode='test')
  # Sort by duration (longest-first) to minimize padding waste.
  # Without sorting, a batch mixing 30s + 0.2s audio pads everything to 30s.
  # With longest-first, similar-length audio groups together → ~2x speedup on 77K items.
  if 'audio_duration_sec' in df.columns:
    df = df.sort_values('audio_duration_sec', ascending=False).reset_index(drop=True)
  _log.info(f'Test data: {len(df)} utterances (sorted by duration, longest-first)')
  probe_state = _init_probe_state(df) if PROBE_MODE else None
  
  test_dl = get_dl(mode='test', df=df)
  _log.info(f'DataLoader ready, batch_size={gz.eval_batch_size()}')
  
  # Enable generation mode
  gz.set('do_generate', True)
  
  all_ids = []
  all_preds = []
  _quiet_word_logs = _is_word_runtime()
  
  # Progress logging: use print+flush instead of tqdm to avoid buffering
  # issues when stdout/stderr is piped through `tee` (non-TTY = full buffering,
  # tqdm output gets stuck in buffer until process exit).
  # For word track online submission keep progress sparser to stay well below
  # the Docker 500-line log cap; phonetic keeps the original detail.
  n_total = len(df)
  n_done = 0
  _infer_start = time.time()
  _MILESTONES = ([1, 5, 10, 25, 50, 75, 100] if _quiet_word_logs
                 else sorted(set([1, 2, 5] + list(range(5, 101, 5)))))
  _next_milestone_idx = 0
  _split_log_count = 0
  _split_log_suppressed = 0
  
  def _print_progress(n_done, n_total, force=False):
    nonlocal _next_milestone_idx
    pct = 100.0 * n_done / n_total if n_total > 0 else 0
    should_print = force
    if _next_milestone_idx < len(_MILESTONES) and pct >= _MILESTONES[_next_milestone_idx]:
      should_print = True
      # Skip past all milestones we've passed (in case a big batch jumps multiple %)
      while _next_milestone_idx < len(_MILESTONES) and pct >= _MILESTONES[_next_milestone_idx]:
        _next_milestone_idx += 1
    if should_print:
      elapsed = time.time() - _infer_start
      speed = n_done / elapsed if elapsed > 0 else 0
      eta = (n_total - n_done) / speed if speed > 0 else 0
      print(f'[{_ts()}][submit2] Inference: {n_done}/{n_total} ({pct:.1f}%) '
            f'[{elapsed:.0f}s elapsed, {speed:.1f} utt/s, ETA {eta:.0f}s]',
            flush=True)

  _oom_count = 0
  _MAX_TRUNC_SEC = 30.0  # last-resort truncation when bs=1 still OOMs
  _SAMPLE_RATE = getattr(FLAGS, 'sample_rate', 16000)
  _runtime_key = f'{getattr(FLAGS, "model", "")} {getattr(FLAGS, "backbone", "")}'.lower()
  _is_wavlm_family = any(k in _runtime_key for k in ('wavlm', 'wav2vec2', 'hubert', 'w2v-bert'))

  # ── Proactive duration-based batch splitting ──────────────────────────
  # Conformer self-attention is O(T²): long audio requires huge attention
  # matrices.  A CUDA OOM inside NeMo's fused kernels irrecoverably
  # corrupts the GPU context ("illegal memory access").  We MUST prevent
  # OOM by pre-computing safe batch sizes from audio duration.
  #
  # Calibration (empirical, A100 80 GB):
  #   69 s audio, bs=32 → peak ≈ 80 GiB → OOM.  Per-sample ≈ 2.5 GiB.
  #   Memory scales as O(dur²) due to attention.
  _gpu_total_gb = torch.cuda.get_device_properties(device).total_memory / (1 << 30)
  _model_mem_gb = sum(p.nbytes for p in model.parameters()) / (1 << 30)
  _gpu_avail_gb = _gpu_total_gb - _model_mem_gb - 3.0  # 3 GB headroom
  _diag(f'GPU mem: total={_gpu_total_gb:.1f}GB, model={_model_mem_gb:.1f}GB, '
        f'avail_for_act={_gpu_avail_gb:.1f}GB')

  def _duration_safe_bs(max_dur_sec, base_bs):
    """Max safe batch size for the given max audio duration (seconds).

    Quadratic model: mem_per_sample ≈ k · dur².
    k = 0.0006 (conservative; empirical 2.5/69² = 0.000525).
    """
    safe = base_bs
    if _is_wavlm_family:
      ref_dur = max(max_dur_sec, 15.0)
      wavlm_cap = max(1, int(_gpu_total_gb / 3.0))
      dur_penalty = max(1.0, (ref_dur / 15.0) ** 1.15)
      safe = min(safe, max(1, int(wavlm_cap / dur_penalty)))
    if 'nemo' in _runtime_key and max_dur_sec > 20:
      k = 0.0006  # GiB per second² per sample
      mem_per_sample = k * max_dur_sec ** 2
      safe = min(safe, max(1, int(0.85 * _gpu_avail_gb / mem_per_sample)))
    return max(1, min(base_bs, safe))

  def _get_batch_max_dur(batch):
    """Max audio duration (sec) in a batch, from input_features tensor."""
    if 'input_features' in batch:
      feat = batch['input_features']
      if isinstance(feat, torch.Tensor):
        return feat.shape[-1] / _SAMPLE_RATE
    return 0.0  # unknown → no proactive split

  def _slice_batch(batch, start, end):
    """Slice a collated batch dict along batch dimension, move to device."""
    sliced = {}
    for k, v in batch.items():
      if isinstance(v, torch.Tensor):
        sliced[k] = v[start:end].to(device, non_blocking=True)
      elif isinstance(v, list):
        sliced[k] = v[start:end]
      else:
        sliced[k] = v
    return sliced

  def _truncate_single(single_batch):
    """Truncate a bs=1 batch's audio to _MAX_TRUNC_SEC."""
    max_samples = int(_MAX_TRUNC_SEC * _SAMPLE_RATE)
    trunc = {}
    for k, v in single_batch.items():
      if k in ('input_features', 'attention_mask') and isinstance(v, torch.Tensor) and v.shape[-1] > max_samples:
        trunc[k] = v[..., :max_samples]
      else:
        trunc[k] = v
    return trunc

  def _infer_batch(sub_batch):
    """Run model on sub_batch, return decoded texts and raw forward result."""
    res = model(sub_batch)
    from src.eval import decode_predictions
    return decode_predictions(res, model=model), res

  def _cuda_ok():
    """Quick sanity check: can we still allocate on GPU?"""
    try:
      torch.cuda.synchronize()
      _t = torch.zeros(1, device=device)
      del _t
      return True
    except Exception:
      return False

  def _safe_infer_single(batch, idx):
    """Infer a single item with truncation + empty fallback."""
    nonlocal _oom_count
    single = _slice_batch(batch, idx, idx + 1)
    try:
      texts, res = _infer_batch(single)
      _update_probe_aux_outputs(probe_state, single, res)
      return texts
    except torch.cuda.OutOfMemoryError:
      del single
      torch.cuda.empty_cache()
      _oom_count += 1
      uid = batch['id'][idx] if 'id' in batch and idx < len(batch['id']) else '?'
      if not _cuda_ok():
        _diag(f'CUDA corrupted after OOM (item {idx}, id={uid}), empty prediction')
        return ['']
      _diag(f'OOM at bs=1 (item {idx}, id={uid}), truncating to {_MAX_TRUNC_SEC}s')
      trunc = _truncate_single(_slice_batch(batch, idx, idx + 1))
      try:
        texts, res = _infer_batch(trunc)
        _update_probe_aux_outputs(probe_state, trunc, res)
        return texts
      except Exception:
        torch.cuda.empty_cache()
        _diag(f'Failed even after truncation (item {idx}), empty prediction')
        return ['']

  def _infer_range_with_backoff(batch, start, end, max_dur):
    """Infer [start:end) with recursive batch halving on CUDA OOM."""
    nonlocal _oom_count
    sub_bs = end - start
    sub = _slice_batch(batch, start, end)
    try:
      sub_texts, sub_res = _infer_batch(sub)
      _update_probe_aux_outputs(probe_state, sub, sub_res)
      return sub_texts
    except torch.cuda.OutOfMemoryError:
      del sub
      torch.cuda.empty_cache()
      _oom_count += 1
      if not _cuda_ok():
        _diag(f'CUDA corrupted after OOM (bs={sub_bs}, dur={max_dur:.1f}s), empty predictions')
        return [''] * sub_bs
      if sub_bs <= 1:
        return _safe_infer_single(batch, start)
      next_bs = max(1, sub_bs // 2)
      _diag(f'OOM at sub-batch (bs={sub_bs}, dur={max_dur:.1f}s), shrinking to <= {next_bs}')
      texts = []
      for cs in range(start, end, next_bs):
        ce = min(cs + next_bs, end)
        texts.extend(_infer_range_with_backoff(batch, cs, ce, max_dur))
      return texts

  with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
    for batch in test_dl:
      bs = len(batch['id']) if 'id' in batch else (
          batch['input_features'].shape[0] if 'input_features' in batch else 1)
      max_dur = _get_batch_max_dur(batch)
      safe_bs = max(1, _duration_safe_bs(max_dur, base_bs=bs))
      if safe_bs < bs:
        if _quiet_word_logs:
          _split_log_count += 1
          if _split_log_count <= 5 or safe_bs == 1:
            _diag(f'Proactive split: bs {bs}->{safe_bs} (max_dur={max_dur:.1f}s)')
          else:
            _split_log_suppressed += 1
        else:
          _diag(f'Proactive split: bs {bs}->{safe_bs} (max_dur={max_dur:.1f}s)')

      texts = []
      for cs in range(0, bs, safe_bs):
        ce = min(cs + safe_bs, bs)
        texts.extend(_infer_range_with_backoff(batch, cs, ce, max_dur))

      # Collect
      if 'id' in batch:
        all_ids.extend(batch['id'])
      all_preds.extend(texts)
      n_done += len(texts)
      _print_progress(n_done, n_total)

  if _oom_count > 0:
    _diag(f'OOM fallback triggered {_oom_count} time(s) during inference')
  if _split_log_suppressed > 0:
    _diag(f'Proactive split: suppressed {_split_log_suppressed} additional events '
          f'(word online quiet mode)')
  
  _print_progress(n_done, n_total, force=True)
  _log.info(f'Inference done: {len(all_preds)} predictions')
  
  # Build predictions dict
  predictions = {}
  for uid, text in zip(all_ids, all_preds):
    predictions[uid] = text

  if PROBE_MODE:
    _emit_probe_report(predictions, probe_state)
  
  return predictions


def write_submission(predictions):
  """Write submission.jsonl matching the DrivenData format."""
  data_dir = DATA_DIR
  submission_format_path = data_dir / 'submission_format.jsonl'
  submission_path = SUBMISSION_DIR / 'submission.jsonl'
  
  # Detect prediction column
  pred_col = 'phonetic_text'
  with submission_format_path.open('r') as f0:
    first_item = json.loads(f0.readline())
    if 'orthographic_text' in first_item:
      pred_col = 'orthographic_text'
    elif 'phonetic_text' in first_item:
      pred_col = 'phonetic_text'
  _log.info(f'Submission pred column: {pred_col}')
  
  n_written = 0
  with submission_format_path.open('r') as fr, submission_path.open('w') as fw:
    for line in fr:
      item = json.loads(line)
      pred = predictions.get(item['utterance_id'], '')
      item[pred_col] = pred
      fw.write(json.dumps(item) + '\n')
      if n_written < 5:
        _log.info(f'  [{item["utterance_id"]}] {pred[:150]}')
      n_written += 1
  
  # Summary
  all_preds = list(predictions.values())
  non_empty = [p for p in all_preds if p.strip()]
  avg_chars = sum(len(p) for p in non_empty) / len(non_empty) if non_empty else 0
  _log.info(f'Submission: {n_written} items, {len(non_empty)} non-empty, '
              f'avg {avg_chars:.0f} chars/sample')
  return {
      'submission_path': str(submission_path),
      'pred_col': pred_col,
      'written': int(n_written),
      'non_empty': int(len(non_empty)),
      'empty': int(n_written - len(non_empty)),
      'avg_chars_non_empty': float(avg_chars),
  }


def _maybe_dump_tree_reranker_features(df, feat_cols):
  dump_flag = os.environ.get('TREE_RERANKER_DUMP_FEATS', '').strip().lower()
  if dump_flag not in ('1', 'true', 't', 'yes', 'y', 'on'):
    return None

  dump_df = df.copy()
  uid_limit = int(os.environ.get('TREE_RERANKER_DUMP_UID_LIMIT', '0') or 0)
  if uid_limit > 0 and 'uid' in dump_df.columns:
    keep_uids = dump_df['uid'].drop_duplicates().tolist()[:uid_limit]
    dump_df = dump_df[dump_df['uid'].isin(set(keep_uids))].copy()

  dump_path_env = os.environ.get('TREE_RERANKER_DUMP_PATH', '').strip()
  dump_path = Path(dump_path_env) if dump_path_env else (SUBMISSION_DIR / 'tree_reranker_online_feats.pkl')
  dump_path.parent.mkdir(parents=True, exist_ok=True)
  dump_df.to_pickle(dump_path)

  meta_path = dump_path.with_suffix(dump_path.suffix + '.meta.json') if dump_path.suffix else Path(str(dump_path) + '.meta.json')
  meta = {
      'feat_cols': list(feat_cols),
      'columns': dump_df.columns.tolist(),
      'rows': int(len(dump_df)),
      'uid_count': int(dump_df['uid'].nunique()) if 'uid' in dump_df.columns else 0,
      'uid_limit': int(uid_limit),
  }
  meta_path.write_text(json.dumps(meta, indent=2))
  _diag(f'Dumped tree reranker online features: {dump_path} ({meta["rows"]} rows, {meta["uid_count"]} uids)')
  return dump_path


def _maybe_dump_online_ctc_logprobs(all_logprobs):
  dump_flag = os.environ.get('TREE_RERANKER_DUMP_LOGPROBS', '').strip().lower()
  if dump_flag not in ('1', 'true', 't', 'yes', 'y', 'on'):
    return None

  dump_path_env = os.environ.get('TREE_RERANKER_DUMP_LOGPROBS_PATH', '').strip()
  dump_path = Path(dump_path_env) if dump_path_env else (SUBMISSION_DIR / 'tree_reranker_online_logprobs.pkl')

  uid_text = os.environ.get('TREE_RERANKER_DUMP_LOGPROBS_UIDS', '').strip()
  model_text = os.environ.get('TREE_RERANKER_DUMP_LOGPROBS_MODELS', '').strip()
  keep_uids = {x.strip() for x in uid_text.split(',') if x.strip()}
  keep_models = {x.strip() for x in model_text.split(',') if x.strip()}

  model_aliases = {}
  for model_dir in ENSEMBLE_MODEL_DIRS:
    aliases = {model_dir.name}
    flags_data = _read_model_flags_for_runtime(model_dir)
    for key in ('model_name', 'mn'):
      value = str(flags_data.get(key, '') or '').strip()
      if value:
        aliases.add(value)
    model_aliases[model_dir.name] = aliases

  filtered = {}
  for model_name, uid_map in (all_logprobs or {}).items():
    aliases = model_aliases.get(model_name, {model_name})
    if keep_models and aliases.isdisjoint(keep_models):
      continue
    if keep_uids:
      rows = {uid: value for uid, value in uid_map.items() if uid in keep_uids}
    else:
      rows = dict(uid_map)
    if rows:
      filtered[model_name] = {
          'aliases': sorted(aliases),
          'uid_map': rows,
      }

  dump_path.parent.mkdir(parents=True, exist_ok=True)
  with open(dump_path, 'wb') as out:
    pickle.dump(filtered, out)

  resolved_uid_maps = {
      model_name: payload['uid_map']
      for model_name, payload in filtered.items()
  }
  meta = {
      'models': sorted(filtered.keys()),
      'model_aliases': {model_name: payload['aliases'] for model_name, payload in filtered.items()},
      'model_count': int(len(filtered)),
      'uids': sorted({uid for uid_map in resolved_uid_maps.values() for uid in uid_map.keys()}),
      'uid_count': int(len({uid for uid_map in resolved_uid_maps.values() for uid in uid_map.keys()})),
      'model_filter': sorted(keep_models),
      'uid_filter': sorted(keep_uids),
  }
  meta_path = dump_path.with_suffix(dump_path.suffix + '.meta.json') if dump_path.suffix else Path(str(dump_path) + '.meta.json')
  meta_path.write_text(json.dumps(meta, indent=2))
  _diag(f'Dumped online ctc logprobs: {dump_path} (models={meta["model_count"]}, uids={meta["uid_count"]})')
  return dump_path


# ===========================================================================
#  MBR Ensemble Support
# ===========================================================================

# MBR metric mode: controlled by ENSEMBLE_MBR_METRIC env var,
# or 'mbr_metric' in ensemble_meta.json, default char_cer.
# Supported values: char_cer (default), word_wer, hybrid.
_MBR_METRIC_ENV = os.environ.get('ENSEMBLE_MBR_METRIC', '').strip().lower()
# Final resolution deferred to _resolve_mbr_metric() after ensemble_meta is loaded.
_MBR_METRIC = None  # resolved lazily

def _normalize_ipa_mbr(s):
  """Normalize IPA string for MBR CER computation."""
  import unicodedata, re
  s = unicodedata.normalize('NFC', s)
  s = re.sub(r'\s+', ' ', s).strip()
  return s

def _mbr_cer(ref, hyp):
  """Compute character-level edit distance ratio between ref and hyp."""
  r = _normalize_ipa_mbr(ref).strip()
  h = _normalize_ipa_mbr(hyp).strip()
  if not r:
    return 0.0 if not h else 1.0
  try:
    import editdistance
    return editdistance.eval(r, h) / len(r)
  except ImportError:
    # Pure Python fallback (Levenshtein)
    m, n = len(r), len(h)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
      prev, dp[0] = dp[0], i
      for j in range(1, n + 1):
        tmp = dp[j]
        dp[j] = prev if r[i-1] == h[j-1] else 1 + min(prev, dp[j], dp[j-1])
        prev = tmp
    return dp[n] / m

def _mbr_wer(ref, hyp):
  """Compute word-level edit distance ratio between ref and hyp."""
  r = _normalize_ipa_mbr(ref).split()
  h = _normalize_ipa_mbr(hyp).split()
  if not r:
    return 0.0 if not h else 1.0
  try:
    import editdistance
    return editdistance.eval(r, h) / len(r)
  except ImportError:
    m, n = len(r), len(h)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
      prev, dp[0] = dp[0], i
      for j in range(1, n + 1):
        tmp = dp[j]
        dp[j] = prev if r[i-1] == h[j-1] else 1 + min(prev, dp[j], dp[j-1])
        prev = tmp
    return dp[n] / m

def _mbr_hybrid(ref, hyp):
  """Average of char CER and word WER."""
  return (_mbr_cer(ref, hyp) + _mbr_wer(ref, hyp)) / 2.0

def _resolve_mbr_metric():
  """Resolve MBR metric from env var → ensemble_meta.json → default."""
  global _MBR_METRIC, _mbr_dist
  if _MBR_METRIC is not None:
    return _MBR_METRIC
  metric = _MBR_METRIC_ENV
  if not metric:
    metric = _ensemble_meta.get('mbr_metric', '') if IS_ENSEMBLE else ''
  if not metric:
    metric = 'char_cer'
  assert metric in ('char_cer', 'word_wer', 'hybrid'), \
      f'MBR metric must be char_cer, word_wer, or hybrid, got "{metric}"'
  _MBR_METRIC = metric
  if metric == 'word_wer':
    _mbr_dist = _mbr_wer
  elif metric == 'hybrid':
    _mbr_dist = _mbr_hybrid
  else:
    _mbr_dist = _mbr_cer
  return _MBR_METRIC

# Default distance function (will be updated by _resolve_mbr_metric)
_mbr_dist = _mbr_cer

def _mbr_select(candidates):
  """Select candidate with minimum expected distance to others."""
  _resolve_mbr_metric()
  if len(candidates) == 1:
    return candidates[0]
  n = len(candidates)
  avg_dist = []
  for i in range(n):
    total_dist = sum(_mbr_dist(candidates[j], candidates[i])
                     for j in range(n) if j != i)
    avg_dist.append(total_dist / (n - 1))
  best_idx = int(min(range(n), key=lambda i: avg_dist[i]))
  return candidates[best_idx]


# --- Logging suppression for ensemble (avoid flooding 500-line log limit) ---
_nemo_saved_levels = {}
_gezi_saved_level = None

def _suppress_gezi_logging():
  """Suppress gezi's stdlib logger during model loading to avoid duplicate warnings."""
  global _gezi_saved_level
  import logging
  _gezi_logger = logging.getLogger('gezi')
  _gezi_saved_level = _gezi_logger.level
  _gezi_logger.setLevel(logging.ERROR)
  # Also suppress root logger handlers that gezi may have added
  for handler in logging.root.handlers:
    if hasattr(handler, '_gezi_orig_level'):
      continue
    handler._gezi_orig_level = handler.level
    handler.setLevel(logging.ERROR)

def _restore_gezi_logging():
  """Restore gezi's stdlib logger to its original level."""
  global _gezi_saved_level
  import logging
  if _gezi_saved_level is not None:
    logging.getLogger('gezi').setLevel(_gezi_saved_level)
    _gezi_saved_level = None
  for handler in logging.root.handlers:
    if hasattr(handler, '_gezi_orig_level'):
      handler.setLevel(handler._gezi_orig_level)
      del handler._gezi_orig_level

def _suppress_nemo_logging():
  """Temporarily raise NeMo/nemo_logging to ERROR to suppress verbose model load output."""
  import logging
  _nemo_saved_levels.clear()
  for name in ['nemo', 'nemo.collections', 'nemo.collections.asr',
                'nemo.core', 'nemo.utils', 'nemo_logging']:
    lg = logging.getLogger(name)
    _nemo_saved_levels[name] = lg.level
    lg.setLevel(logging.ERROR)
  # Also suppress NeMo's custom logging module if available
  try:
    from nemo.utils import logging as nemo_log
    _nemo_saved_levels['__nemo_log__'] = nemo_log.getEffectiveLevel()
    nemo_log.setLevel(logging.ERROR)
  except Exception:
    pass

def _restore_nemo_logging():
  """Restore NeMo logging levels to their original values."""
  import logging
  try:
    from nemo.utils import logging as nemo_log
    if '__nemo_log__' in _nemo_saved_levels:
      nemo_log.setLevel(_nemo_saved_levels.pop('__nemo_log__'))
  except Exception:
    pass
  for name, level in _nemo_saved_levels.items():
    logging.getLogger(name).setLevel(level)
  _nemo_saved_levels.clear()


_NBEST_MP_STATE = {}


def _nbest_pool_init():
  try:
    import torch
    torch.set_num_threads(1)
    if hasattr(torch, 'set_num_interop_threads'):
      torch.set_num_interop_threads(1)
  except Exception:
    pass


def _get_nbest_parallel_backend():
  backend = os.environ.get('ENSEMBLE_NBEST_BACKEND', 'auto').strip().lower()
  assert backend in ('auto', 'process', 'thread', 'none'), \
      f'Invalid ENSEMBLE_NBEST_BACKEND={backend}'
  if backend == 'auto':
    return 'process'
  if backend == 'none':
    return 'serial'
  return backend


def _get_nbest_score_backend():
  backend = os.environ.get('ENSEMBLE_NBEST_SCORE_BACKEND', 'auto').strip().lower()
  assert backend in ('auto', 'process', 'thread', 'none', 'serial'), \
      f'Invalid ENSEMBLE_NBEST_SCORE_BACKEND={backend}'
  if backend == 'auto':
    return 'serial'
  if backend in ('none', 'serial'):
    return 'serial'
  return backend


def _get_nbest_num_workers(total):
  env_value = os.environ.get('ENSEMBLE_N_WORKERS', '').strip()
  if env_value:
    n_workers = int(env_value)
  else:
    n_workers = 0
  if n_workers <= 0:
    n_workers = min(os.cpu_count() or 1, 16)
  if total < 512:
    return 1
  return max(1, n_workers)


def _create_nbest_pool(n_workers, backend):
  if backend == 'process':
    import multiprocessing as mp
    ctx = mp.get_context('fork')
    return ctx.Pool(n_workers, initializer=_nbest_pool_init)
  if backend == 'thread':
    from multiprocessing.pool import ThreadPool
    _nbest_pool_init()
    return ThreadPool(n_workers)
  return None


def _use_nbest_subprocess():
  return os.environ.get('ENSEMBLE_NBEST_SUBPROCESS', '0').strip().lower() in ('1', 'true', 'yes')


def _run_nbest_rescore_subprocess(all_logprobs, all_model_preds, all_model_pred_map,
                                  skip_ctc_candidate_models, nbest, beam_width):
  helper_path = SRC_DIR / 'submit_nbest_helper.py'
  assert helper_path.exists(), f'N-best helper not found: {helper_path}'

  with tempfile.TemporaryDirectory(prefix='submit_nbest_', dir=str(SUBMISSION_DIR)) as tmpdir:
    payload_path = Path(tmpdir) / 'payload.pkl'
    result_path = Path(tmpdir) / 'result.pkl'
    payload = {
        'all_logprobs': all_logprobs,
        'all_model_preds': all_model_preds,
        'all_model_pred_map': all_model_pred_map,
        'skip_ctc_candidate_models': skip_ctc_candidate_models,
        'nbest': nbest,
        'beam_width': beam_width,
        'n_workers': _get_nbest_num_workers(len(next(iter(all_logprobs.values()))) if all_logprobs else 0),
    }
    with payload_path.open('wb') as f:
      pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    _diag('Dispatching N-best rescoring to isolated CPU subprocess')
    env = os.environ.copy()
    env.pop('CUDA_VISIBLE_DEVICES', None)
    cmd = [sys.executable, str(helper_path), '--input', str(payload_path), '--output', str(result_path)]
    subprocess.run(cmd, check=True, env=env)

    with result_path.open('rb') as f:
      result = pickle.load(f)
    _diag(f'Isolated N-best rescoring done: avg candidates {result.get("avg_candidates_raw", 0):.1f} raw, '
          f'{result.get("avg_candidates_unique", 0):.1f} unique')
    if result.get('fallback_count', 0):
      _diag(f'MBR fallback for {result["fallback_count"]} utterances without logprobs')
    return result['predictions']


def _maybe_log_nbest_progress(desc, idx, total, start_time, interval):
  if idx % interval != 0 and idx != total:
    return
  elapsed = time.time() - start_time
  rate = idx / elapsed if elapsed > 0 else 0.0
  eta = (total - idx) / rate if rate > 0 else 0.0
  print(f'[{_ts()}][submit2] {desc}: {idx}/{total} ({rate:.1f} utt/s, ETA {eta:.0f}s)', flush=True)


def _build_nbest_candidates_single_uid(uid):
  import numpy as np

  state = _NBEST_MP_STATE
  candidate_set = set()
  raw_count = 0
  for mn in state['candidate_model_names']:
    if mn in state['all_logprobs'] and not state['skip_ctc_candidate_models'].get(mn, False):
      lp = state['all_logprobs'][mn][uid].astype(np.float32)
      hyps = state['prefix_beam_search_nbest'](
          lp,
          state['blank_id'],
          state['beam_width'],
          nbest=state['nbest'],
          id_to_char=state['id_to_char'])
      raw_count += len(hyps)
      for _score, text in hyps:
        candidate_set.add(text)

    pred = str(state['all_model_pred_map'].get(uid, {}).get(mn, '') or '')
    if pred:
      raw_count += 1
      candidate_set.add(pred)

  candidates = list(candidate_set)
  return uid, candidates, raw_count, len(candidates)


def _nbest_rescore_single_uid(uid):
  import numpy as np

  state = _NBEST_MP_STATE
  candidates = state['candidate_lists'][uid]
  if not candidates:
    return uid, ''
  if len(candidates) == 1:
    return uid, candidates[0]

  all_token_ids = []
  for cand_text in candidates:
    token_ids = [state['char_to_id'][ch] for ch in cand_text if ch in state['char_to_id']]
    all_token_ids.append(token_ids)

  avg_scores = np.zeros(len(candidates), dtype=np.float64)
  for mn in state['model_names']:
    lp = state['all_logprobs'][mn][uid].astype(np.float32)
    scores = state['ctc_force_score_batch'](lp, all_token_ids, blank=state['blank_id'])
    for i, sc in enumerate(scores):
      avg_scores[i] += sc
  avg_scores /= len(state['model_names'])

  best_idx = int(np.argmax(avg_scores))
  return uid, candidates[best_idx]


def _run_nbest_rescore(all_logprobs, all_model_preds, all_model_pred_map=None,
                       skip_ctc_candidate_models=None, nbest=10, beam_width=10):
  """N-best rescoring from in-memory CTC log_probs dicts.
  
  Reuses the same algorithm as ensemble-fold0.py:nbest_rescore_ensemble():
    1. For each model, beam search to get top-N candidate hypotheses
    2. Pool all unique candidates across models
    3. CTC force-score all candidates under every model
    4. Select candidate with highest average score
  
  Args:
    all_logprobs: dict of {model_name: {uid: numpy (T_i, V) float16}}
    all_model_preds: dict of {uid: [text_model0, text_model1, ...]}
    all_model_pred_map: optional dict of {uid: {model_name: primary_text}}
    skip_ctc_candidate_models: optional dict of {model_name: bool}
  
  Falls back to MBR text selection for utterances without logprobs.
  """
  import numpy as np
  from src.ctc_decode import prefix_beam_search_nbest, ctc_force_score_batch
  from src.models.base import IPA_ID_TO_CHAR, IPA_CTC_BLANK

  blank_id = IPA_CTC_BLANK
  id_to_char = IPA_ID_TO_CHAR
  char_to_id = {ch: cid for cid, ch in id_to_char.items()}
  all_model_pred_map = all_model_pred_map or {}
  skip_ctc_candidate_models = skip_ctc_candidate_models or {}
  if _use_nbest_subprocess():
    return _run_nbest_rescore_subprocess(
        all_logprobs,
        all_model_preds,
        all_model_pred_map,
        skip_ctc_candidate_models,
        nbest,
        beam_width,
    )
  
  score_model_names = list(all_logprobs.keys())
  candidate_model_names = list(score_model_names)
  for pred_map in all_model_pred_map.values():
    for mn in pred_map.keys():
      if mn not in candidate_model_names:
        candidate_model_names.append(mn)

  _lp_counts = [f'{len(all_logprobs[mn])}' for mn in score_model_names]
  _diag(f'Logprobs: {len(score_model_names)} scoring models, utterances per model: {", ".join(_lp_counts)}')
  if len(candidate_model_names) != len(score_model_names):
    extra_candidate_models = [mn for mn in candidate_model_names if mn not in score_model_names]
    _diag('Extra candidate-only models (no CTC logprobs): ' + ', '.join(extra_candidate_models))

  # Find common UIDs that have logprobs from ALL models
  uid_sets = [set(all_logprobs[mn].keys()) for mn in score_model_names]
  common_uids = set.intersection(*uid_sets)
  _diag(f'N-best rescore: {len(common_uids)} utterances x {len(score_model_names)} scoring models, {len(candidate_model_names)} candidate models')

  predictions = {}
  uids_list = sorted(common_uids)
  total = len(uids_list)
  t0 = time.time()
  build_progress_interval = max(10000, total // 5)
  score_progress_interval = max(2000, total // 10)
  n_workers = _get_nbest_num_workers(total)
  build_backend = _get_nbest_parallel_backend()
  score_backend = _get_nbest_score_backend()
  can_parallel_build = n_workers > 1 and build_backend != 'serial'
  can_parallel_score = n_workers > 1 and score_backend != 'serial'
  build_chunksize = max(4, min(32, total // max(n_workers * 16, 1))) if can_parallel_build else 1
  score_chunksize = max(4, min(32, total // max(n_workers * 16, 1))) if can_parallel_score else 1
  if can_parallel_build or can_parallel_score:
    _diag(
        'N-best backends: '
        f'build={build_backend}, score={score_backend}, workers={n_workers}, '
        f'build_chunksize={build_chunksize}, score_chunksize={score_chunksize}')

  global _NBEST_MP_STATE
  _NBEST_MP_STATE = {
      'candidate_model_names': candidate_model_names,
      'all_logprobs': all_logprobs,
      'all_model_pred_map': all_model_pred_map,
      'skip_ctc_candidate_models': skip_ctc_candidate_models,
      'blank_id': blank_id,
      'beam_width': beam_width,
      'nbest': nbest,
      'id_to_char': id_to_char,
      'char_to_id': char_to_id,
      'prefix_beam_search_nbest': prefix_beam_search_nbest,
      'ctc_force_score_batch': ctc_force_score_batch,
  }

  candidate_lists = {}
  n_cands_total = 0
  n_unique_total = 0
  if can_parallel_build:
    build_start = time.time()
    with _create_nbest_pool(n_workers, build_backend) as pool:
      for idx, (uid, candidates, raw_count, unique_count) in enumerate(
          pool.imap_unordered(_build_nbest_candidates_single_uid, uids_list, chunksize=build_chunksize), start=1):
        candidate_lists[uid] = candidates
        n_cands_total += raw_count
        n_unique_total += unique_count
        _maybe_log_nbest_progress('Build candidates', idx, total, build_start, build_progress_interval)
  else:
    build_start = time.time()
    for idx, uid in enumerate(uids_list, start=1):
      uid, candidates, raw_count, unique_count = _build_nbest_candidates_single_uid(uid)
      candidate_lists[uid] = candidates
      n_cands_total += raw_count
      n_unique_total += unique_count
      _maybe_log_nbest_progress('Build candidates', idx, total, build_start, build_progress_interval)

  _NBEST_MP_STATE = {
    'model_names': score_model_names,
      'all_logprobs': all_logprobs,
      'candidate_lists': candidate_lists,
      'blank_id': blank_id,
      'char_to_id': char_to_id,
      'ctc_force_score_batch': ctc_force_score_batch,
  }

  if can_parallel_score:
    _diag('Build candidates done; starting exact CTC rescoring')
  else:
    _diag('Build candidates done; starting exact CTC rescoring in serial mode (online-stable path)')
  score_start = time.time()
  if can_parallel_score:
    with _create_nbest_pool(n_workers, score_backend) as pool:
      for idx, (uid, pred) in enumerate(
          pool.imap_unordered(_nbest_rescore_single_uid, uids_list, chunksize=score_chunksize), start=1):
        predictions[uid] = pred
        _maybe_log_nbest_progress('N-best rescore', idx, total, score_start, score_progress_interval)
  else:
    for idx, uid in enumerate(uids_list, start=1):
      uid, pred = _nbest_rescore_single_uid(uid)
      predictions[uid] = pred
      _maybe_log_nbest_progress('N-best rescore', idx, total, score_start, score_progress_interval)

  _NBEST_MP_STATE = {}

  elapsed = time.time() - t0
  _diag(f'N-best rescore done: {total} utterances in {elapsed:.1f}s ({total/max(elapsed,1):.1f} utt/s)')
  if total > 0:
    _diag(f'Avg candidates per utterance: {n_cands_total / total:.1f} raw, {n_unique_total / total:.1f} unique')

  # Fall back to MBR for utterances that don't have logprobs (shouldn't happen, but safety)
  n_fallback = 0
  for uid, candidates in all_model_preds.items():
    if uid not in predictions:
      if len(set(candidates)) == 1:
        predictions[uid] = candidates[0]
      else:
        predictions[uid] = _mbr_select(candidates)
      n_fallback += 1
  if n_fallback > 0:
    _diag(f'MBR fallback for {n_fallback} utterances without logprobs')

  return predictions


def _build_tdt_feature_candidate_lists(df, tdt_model_names, primary_tdt_texts,
                                       topk=8, force_keep_preds=True):
  candidate_lists = {}
  topk = max(int(topk or 0), 1)
  for uid, group in df.groupby('uid', sort=False):
    ranked = group.sort_values('ctc_score_mean', ascending=False)['candidate_text'].tolist()
    keep = []
    keep_set = set()
    for cand_text in ranked[:topk]:
      if cand_text and cand_text not in keep_set:
        keep.append(cand_text)
        keep_set.add(cand_text)
    if force_keep_preds:
      for mn in tdt_model_names:
        text = (primary_tdt_texts.get(mn, {}) or {}).get(uid, '')
        if text and text not in keep_set:
          keep.append(text)
          keep_set.add(text)
    candidate_lists[uid] = keep
  return candidate_lists


def _convert_tdt_score_arrays(candidate_lists, scores_by_uid):
  score_map = {}
  for uid, scores in scores_by_uid.items():
    candidates = candidate_lists.get(uid, [])
    if len(candidates) != len(scores):
      raise ValueError(
          f'TDT score size mismatch for {uid}: {len(scores)} vs {len(candidates)} candidates')
    score_map[uid] = {
        cand_text: float(score)
        for cand_text, score in zip(candidates, scores)
    }
  return score_map


def _augment_tdt_feature_frame(df, tdt_model_names, primary_tdt_texts,
                               tdt_score_maps=None,
                               include_light=True, include_exact=True,
                               include_score_compare=True):
  """Add TDT-related features to the candidate DataFrame.

  Ported from ensemble.py _augment_tdt_feature_frame to stay in sync.
  """
  df = df.copy()
  candidate_len = df['candidate_text'].str.len().astype(float)
  candidate_spaces = df['candidate_text'].str.count(' ').astype(float)
  new_cols = []

  hit_cols = []
  len_diff_cols = []
  space_diff_cols = []
  score_cols = []
  score_best_cols = []

  for mn in tdt_model_names:
    uid_to_text = primary_tdt_texts.get(mn, {})
    text_map = df['uid'].map(uid_to_text)
    len_map = {uid: float(len(text)) for uid, text in uid_to_text.items()}
    space_map = {uid: float(text.count(' ')) for uid, text in uid_to_text.items()}

    hit_col = f'is_tdt_pred_{mn}'
    len_diff_col = f'tdt_len_diff_{mn}'
    space_diff_col = f'tdt_spaces_diff_{mn}'

    if include_light:
      df[hit_col] = ((text_map.notna()) & (df['candidate_text'] == text_map)).astype(int)
      df[len_diff_col] = (candidate_len - df['uid'].map(len_map)).abs()
      df[space_diff_col] = (candidate_spaces - df['uid'].map(space_map)).abs()

      hit_cols.append(hit_col)
      len_diff_cols.append(len_diff_col)
      space_diff_cols.append(space_diff_col)
      new_cols.extend([hit_col, len_diff_col, space_diff_col])

    score_col = f'tdt_score_{mn}'
    if include_exact and tdt_score_maps is not None:
      rows = []
      uid_to_scores = tdt_score_maps.get(mn, {})
      for uid, cand_scores in uid_to_scores.items():
        for cand_text, score in cand_scores.items():
          rows.append({
              'uid': uid,
              'candidate_text': cand_text,
              score_col: float(score),
          })
      if rows:
        score_df = pd.DataFrame(rows)
        df = df.merge(score_df, on=['uid', 'candidate_text'], how='left')
      else:
        df[score_col] = np.nan
      per_char_col = f'tdt_score_per_char_{mn}'
      df[per_char_col] = df[score_col] / candidate_len.clip(lower=1)
      # Per-model group-relative TDT features (ported from ensemble.py)
      grp_score = df.groupby('uid')[score_col]
      rank_col = f'tdt_score_rank_{mn}'
      pct_col = f'tdt_score_pct_{mn}'
      diff_col = f'tdt_score_diff_from_best_{mn}'
      centered_col = f'tdt_score_centered_{mn}'
      zscore_col = f'tdt_score_zscore_{mn}'
      is_best_col = f'tdt_score_is_best_{mn}'
      margin_second_col = f'tdt_score_margin_to_second_{mn}'
      best_score = grp_score.transform('max')
      mean_score = grp_score.transform('mean')
      std_score = grp_score.transform('std').replace(0.0, np.nan)
      second_best_score = grp_score.transform(
          lambda s: s.nlargest(2).iloc[-1] if s.notna().any() else np.nan)
      df[rank_col] = grp_score.rank(ascending=False, method='min', na_option='bottom')
      df[pct_col] = grp_score.rank(ascending=False, pct=True, na_option='bottom')
      df[diff_col] = df[score_col] - best_score
      df[centered_col] = df[score_col] - mean_score
      df[zscore_col] = (df[score_col] - mean_score) / std_score
      df[zscore_col] = df[zscore_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
      df[is_best_col] = ((df[score_col].notna()) & np.isclose(df[score_col], best_score)).astype(int)
      df[margin_second_col] = df[score_col] - second_best_score
      df.loc[df[is_best_col] == 0, margin_second_col] = 0.0
      score_cols.append(score_col)
      score_best_cols.append(is_best_col)
      new_cols.extend([
          score_col, per_char_col, rank_col, pct_col, diff_col,
          centered_col, zscore_col, is_best_col, margin_second_col,
      ])

  if include_light and hit_cols:
    df['n_tdt_pred_hits'] = df[hit_cols].sum(axis=1)
    new_cols.append('n_tdt_pred_hits')
  if include_light and len_diff_cols:
    df['tdt_len_diff_mean'] = df[len_diff_cols].mean(axis=1, skipna=True)
    df['tdt_len_diff_min'] = df[len_diff_cols].min(axis=1, skipna=True)
    df['tdt_len_diff_max'] = df[len_diff_cols].max(axis=1, skipna=True)
    new_cols.extend(['tdt_len_diff_mean', 'tdt_len_diff_min', 'tdt_len_diff_max'])
  if include_light and space_diff_cols:
    df['tdt_spaces_diff_mean'] = df[space_diff_cols].mean(axis=1, skipna=True)
    df['tdt_spaces_diff_min'] = df[space_diff_cols].min(axis=1, skipna=True)
    df['tdt_spaces_diff_max'] = df[space_diff_cols].max(axis=1, skipna=True)
    new_cols.extend(['tdt_spaces_diff_mean', 'tdt_spaces_diff_min', 'tdt_spaces_diff_max'])
  if include_exact and score_cols:
    df['n_tdt_scored_models'] = df[score_cols].notna().sum(axis=1)
    df['tdt_score_mean'] = df[score_cols].mean(axis=1, skipna=True)
    df['tdt_score_std'] = df[score_cols].std(axis=1, skipna=True)
    df['tdt_score_min'] = df[score_cols].min(axis=1, skipna=True)
    df['tdt_score_max'] = df[score_cols].max(axis=1, skipna=True)
    df['tdt_score_range'] = df['tdt_score_max'] - df['tdt_score_min']
    df['tdt_score_mean_per_char'] = df['tdt_score_mean'] / candidate_len.clip(lower=1)
    grp_tdt = df.groupby('uid')['tdt_score_mean']
    grp_ctc = df.groupby('uid')['ctc_score_mean']
    tdt_best = grp_tdt.transform('max')
    tdt_mean = grp_tdt.transform('mean')
    tdt_std = grp_tdt.transform('std').replace(0.0, np.nan)
    tdt_second = grp_tdt.transform(
        lambda s: s.nlargest(2).iloc[-1] if s.notna().any() else np.nan)
    ctc_best = grp_ctc.transform('max')
    df['tdt_score_mean_rank'] = grp_tdt.rank(ascending=False, method='min', na_option='bottom')
    df['tdt_score_mean_pct'] = grp_tdt.rank(ascending=False, pct=True, na_option='bottom')
    df['tdt_score_diff_from_best'] = df['tdt_score_mean'] - tdt_best
    df['tdt_score_mean_centered'] = df['tdt_score_mean'] - tdt_mean
    df['tdt_score_mean_zscore'] = (df['tdt_score_mean'] - tdt_mean) / tdt_std
    df['tdt_score_mean_zscore'] = df['tdt_score_mean_zscore'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df['tdt_score_mean_is_best'] = ((df['tdt_score_mean'].notna()) & np.isclose(df['tdt_score_mean'], tdt_best)).astype(int)
    df['tdt_score_mean_margin_to_second'] = df['tdt_score_mean'] - tdt_second
    df.loc[df['tdt_score_mean_is_best'] == 0, 'tdt_score_mean_margin_to_second'] = 0.0
    df['tdt_score_best_vote_count'] = df[score_best_cols].sum(axis=1) if score_best_cols else 0.0
    df['tdt_score_best_vote_frac'] = df['tdt_score_best_vote_count'] / df['n_tdt_scored_models'].clip(lower=1)
    grp_votes = df.groupby('uid')['tdt_score_best_vote_count']
    vote_best = grp_votes.transform('max')
    df['tdt_score_best_vote_is_top'] = ((df['tdt_score_best_vote_count'] > 0) &
                                        np.isclose(df['tdt_score_best_vote_count'], vote_best)).astype(int)
    df['tdt_score_best_vote_margin'] = df['tdt_score_best_vote_count'] - grp_votes.transform(
        lambda s: s.nlargest(2).iloc[-1] if s.notna().any() else np.nan)
    df.loc[df['tdt_score_best_vote_is_top'] == 0, 'tdt_score_best_vote_margin'] = 0.0
    new_cols.extend([
        'n_tdt_scored_models', 'tdt_score_mean', 'tdt_score_std', 'tdt_score_min',
        'tdt_score_max', 'tdt_score_range', 'tdt_score_mean_per_char',
        'tdt_score_mean_rank', 'tdt_score_mean_pct',
        'tdt_score_diff_from_best', 'tdt_score_mean_centered',
        'tdt_score_mean_zscore', 'tdt_score_mean_is_best',
        'tdt_score_mean_margin_to_second', 'tdt_score_best_vote_count',
        'tdt_score_best_vote_frac', 'tdt_score_best_vote_is_top',
        'tdt_score_best_vote_margin',
    ])
    if include_score_compare:
      df['tdt_ctc_score_gap'] = df['tdt_score_mean'] - df['ctc_score_mean']
      df['tdt_score_ctc_rank_gap'] = df['tdt_score_mean_rank'] - grp_ctc.rank(
          ascending=False, method='min', na_option='bottom')
      df['tdt_score_mean_vs_ctc_best_gap'] = df['tdt_score_mean'] - ctc_best
      new_cols.extend([
          'tdt_ctc_score_gap',
          'tdt_score_ctc_rank_gap',
          'tdt_score_mean_vs_ctc_best_gap',
      ])

  return df, [c for c in new_cols if c in df.columns]


def _score_tdt_candidates_for_model_dir(model_dir, candidate_lists, verbose=True):
  import importlib
  import torch

  assert model_dir.exists(), f'Model dir not found: {model_dir}'
  best_pt = model_dir / 'model.pt'
  if not best_pt.exists():
    best_pt = model_dir / 'best.pt'
  assert best_pt.exists(), f'No model.pt or best.pt found in {model_dir}'

  import gezi as gz
  import melt as mt  # noqa: F401
  from gezi import FLAGS
  from src import config
  from src.preprocess import preprocess
  from src.dataset import Dataset as PaskettiDataset

  gz.init_flags()
  config.init()
  gz.restore_configs(str(model_dir))
  # Override model_dir to Docker path (restore_configs sets it to training path)
  FLAGS.model_dir = str(model_dir)
  ctc_only = bool(getattr(FLAGS, 'ctc_only', False))
  ctc_weight = float(getattr(FLAGS, 'ctc_weight', 1.0) or 0.0)
  s2s_decoder = str(getattr(FLAGS, 's2s_decoder', 'native') or 'native')
  has_tdt_cfg = (not ctc_only) and (ctc_weight < 1.0) and (s2s_decoder == 'tdt_reuse')
  if not has_tdt_cfg:
    if verbose:
      _diag(f'Skip online TDT scoring for {model_dir.name}: ctc_only={ctc_only}, ctc_weight={ctc_weight}, s2s_decoder={s2s_decoder}')
    return {}

  FLAGS.mode = 'test'
  FLAGS.work_mode = 'test'
  FLAGS.distributed = False
  FLAGS.num_workers = 0
  FLAGS.persistent_workers = False
  FLAGS.batch_size = 8
  FLAGS.eval_batch_size = 8
  tdt_score_method = str(getattr(FLAGS, 'tdt_score_method', 'numba') or 'numba').strip().lower()
  if tdt_score_method == 'auto':
    tdt_score_method = 'numba'
  FLAGS.tdt_score_method = tdt_score_method

  model_module = importlib.import_module(f'src.models.{FLAGS.model}')
  Model = model_module.Model
  model = Model()
  try:
    gz.load_weights(model, str(best_pt), strict=False)
  except Exception:
    _load_weights(model, str(best_pt), strict=False)
  if not hasattr(model, 'tdt_decoder'):
    if verbose:
      _diag(f'Skip online TDT scoring for {model_dir.name}: no tdt_decoder')
    return {}

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device).eval()

  df = preprocess(mode='test')
  if hasattr(df, 'columns') and 'utterance_id' in df.columns:
    keep_uids = set(candidate_lists.keys())
    before = len(df)
    df = df[df['utterance_id'].isin(keep_uids)].reset_index(drop=True)
    if verbose:
      _diag(f'Online TDT subset for {model_dir.name}: {before} -> {len(df)} utterances')

  # Sort by duration (longest-first) to minimize padding waste — same as main inference.
  if 'audio_duration_sec' in df.columns:
    df = df.sort_values('audio_duration_sec', ascending=False).reset_index(drop=True)

  ds = PaskettiDataset(df, mode='test')

  def _collate_eval_batch(batch):
    batch = [b for b in batch if b is not None]
    assert batch, 'All online TDT rescoring samples in batch are None'

    first_feat = batch[0]['input_features']
    if np.asarray(first_feat).ndim == 1:
      waveforms = [torch.tensor(b['input_features'], dtype=torch.float32) for b in batch]
      lengths = [w.shape[0] for w in waveforms]
      max_len = max(lengths)
      input_features = torch.zeros(len(batch), max_len, dtype=torch.float32)
      attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
      for i, (w, l) in enumerate(zip(waveforms, lengths)):
        input_features[i, :l] = w
        attention_mask[i, :l] = 1
    else:
      input_features = torch.stack(
          [torch.tensor(b['input_features'], dtype=torch.float32) for b in batch], dim=0)
      attention_mask = None

    out = {
        'input_features': input_features,
        'labels': torch.full((len(batch), 1), -100, dtype=torch.long),
        'id': [b.get('id', '') for b in batch],
    }
    if attention_mask is not None:
      out['attention_mask'] = attention_mask
    return out

  # Scale batch size for TDT scoring — encoder-only forward (no autoregressive decode),
  # so we can use larger batches than main inference.
  import torch as _torch
  tdt_bs_env = int(os.environ.get('TDT_SCORE_BATCH_SIZE', '0') or 0)
  tdt_chunk_env = int(os.environ.get('TDT_SCORE_CHUNK', '0') or 0)
  tdt_bs_cap = int(os.environ.get('TDT_SCORE_BATCH_CAP', '9') or 9)
  tdt_chunk_cap = int(os.environ.get('TDT_SCORE_CHUNK_CAP', '9') or 9)
  assert tdt_bs_env >= 0, f'Invalid TDT_SCORE_BATCH_SIZE={tdt_bs_env}'
  assert tdt_chunk_env >= 0, f'Invalid TDT_SCORE_CHUNK={tdt_chunk_env}'
  assert tdt_bs_cap >= 1, f'Invalid TDT_SCORE_BATCH_CAP={tdt_bs_cap}'
  assert tdt_chunk_cap >= 1, f'Invalid TDT_SCORE_CHUNK_CAP={tdt_chunk_cap}'
  if _torch.cuda.is_available():
    gpu_mem_gb = _torch.cuda.get_device_properties(0).total_memory / (1024**3)
    # The raw memory-based heuristic can overshoot safe shapes for NeMo's
    # TDTLossNumba on large GPUs. Cap it to the locally validated range.
    raw_tdt_bs = max(8, min(48, int(gpu_mem_gb / 2.5)))
    tdt_bs = min(raw_tdt_bs, tdt_bs_cap)
  else:
    gpu_mem_gb = 0.0
    raw_tdt_bs = 8
    tdt_bs = 8
  if tdt_bs_env > 0:
    tdt_bs = tdt_bs_env
  # Keep TDT exact scoring single-process by default. In Docker runtimes with
  # small /dev/shm, worker processes can crash when sharing large audio batches.
  tdt_num_workers = int(os.environ.get('TDT_SCORE_NUM_WORKERS', '0') or 0)
  assert tdt_num_workers >= 0, f'Invalid TDT_SCORE_NUM_WORKERS={tdt_num_workers}'
  tdt_pin_memory = os.environ.get('TDT_SCORE_PIN_MEMORY', '0').strip().lower() not in ('0', 'false', 'no')
  if tdt_num_workers > 0:
    try:
      _torch.multiprocessing.set_sharing_strategy('file_system')
    except Exception:
      pass
  total_candidate_count = sum(len(v) for v in candidate_lists.values())
  # TDT scoring chunk size — max candidates per score_tdt_texts call.
  raw_tdt_score_chunk = (max(8, min(48, int(gpu_mem_gb / 2.5)))
                         if _torch.cuda.is_available() else 8)
  tdt_score_chunk = min(raw_tdt_score_chunk, tdt_chunk_cap)
  if tdt_chunk_env > 0:
    tdt_score_chunk = tdt_chunk_env
  if verbose:
    _diag(
        f'TDT scoring batch_size={tdt_bs} (gpu={gpu_mem_gb:.0f}GB, '
        f'workers={tdt_num_workers}, pin_memory={tdt_pin_memory}, '
        f'raw_bs={raw_tdt_bs}, cap={tdt_bs_cap})')
    _diag(
        f'TDT exact start: model={model_dir.name}, utterances={len(candidate_lists)}, '
        f'candidates={total_candidate_count}, chunk={tdt_score_chunk}, '
        f'raw_chunk={raw_tdt_score_chunk}, chunk_cap={tdt_chunk_cap}')

  test_dl = torch.utils.data.DataLoader(
      ds,
      batch_size=tdt_bs,
      shuffle=False,
        num_workers=tdt_num_workers,
        pin_memory=tdt_pin_memory,
        persistent_workers=bool(tdt_num_workers),
      collate_fn=_collate_eval_batch,
  )
  gz.set('do_generate', False)

  _cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
  _amp_dtype = torch.bfloat16 if _cc >= (8, 0) else torch.float16
  autocast_ctx = (torch.amp.autocast('cuda', dtype=_amp_dtype)
                  if torch.cuda.is_available() else contextlib.nullcontext())

  scores_by_uid = {}
  n_batches = len(test_dl)
  progress_interval_env = os.environ.get('TDT_PROGRESS_INTERVAL')
  progress_target_logs = max(1, int(os.environ.get('TDT_PROGRESS_TARGET_LOGS', '12') or '12'))
  progress_interval = (max(1, int(progress_interval_env))
                       if progress_interval_env is not None
                       else max(1, n_batches // progress_target_logs))
  tdt_batch_debug = _env_flag('TDT_BATCH_DEBUG', False)
  last_progress_time = time.time()
  _set_stage('tdt_exact', f'model={model_dir.name}, batches={n_batches}, candidates={total_candidate_count}')
  try:
    with torch.no_grad(), autocast_ctx:
      for batch_idx, batch in enumerate(test_dl, start=1):
        batch_t0 = time.time()
        batch_ids = batch.get('id', [])
        should_log_detail = tdt_batch_debug and batch_idx <= 2
        should_log_batch = (
            batch_idx <= 3 or
            batch_idx == n_batches or
            batch_idx % progress_interval == 0)
        if verbose and (should_log_detail or batch_idx <= 2):
          _diag(f'TDT batch {batch_idx}/{n_batches} loaded: utterances={len(batch_ids)}')
        input_batch = {}
        for k, v in batch.items():
          if isinstance(v, torch.Tensor):
            input_batch[k] = v.to(device, non_blocking=True)
          else:
            input_batch[k] = v

        forward_t0 = time.time()
        model(input_batch)
        if torch.cuda.is_available():
          torch.cuda.synchronize()
        forward_dt = time.time() - forward_t0
        enc_out = getattr(model, '_last_enc_out', None)
        enc_len = getattr(model, '_last_enc_len', None)
        if enc_out is None:
          raise RuntimeError(f'{model_dir.name} forward did not populate _last_enc_out')
        if verbose and batch_idx <= 2:
          _diag(f'TDT batch {batch_idx}/{n_batches} enc_out: shape={list(enc_out.shape)}, forward={forward_dt:.2f}s')

        # Batch scoring: gather all candidates across utterances in this batch
        batch_uids = []
        batch_n_cands = []
        all_enc_slices = []
        all_enc_lens = []
        all_texts = []
        for i, uid in enumerate(batch_ids):
          candidates = candidate_lists.get(uid)
          if not candidates:
            continue
          n_cands = len(candidates)
          batch_uids.append(uid)
          batch_n_cands.append(n_cands)
          all_enc_slices.append(enc_out[i:i + 1].expand(n_cands, -1, -1))
          if enc_len is not None:
            all_enc_lens.append(enc_len[i:i + 1].expand(n_cands))
          all_texts.extend(candidates)

        if not batch_uids:
          continue

        total_cands = len(all_texts)
        cat_enc = torch.cat(all_enc_slices, dim=0)
        cat_enc_len = torch.cat(all_enc_lens, dim=0) if all_enc_lens else None
        if verbose and (should_log_detail or batch_idx <= 2):
          _diag(
              f'TDT batch {batch_idx}/{n_batches} scoring: '
              f'candidates={total_cands}, forward={forward_dt:.2f}s')

        score_t0 = time.time()
        try:
          if tdt_score_chunk > 0 and total_cands > tdt_score_chunk:
            score_parts = []
            for c_start in range(0, total_cands, tdt_score_chunk):
              c_end = min(c_start + tdt_score_chunk, total_cands)
              if verbose and should_log_detail:
                _diag(f'TDT batch {batch_idx}/{n_batches} chunk {c_start}:{c_end}/{total_cands}')
              c_scores = model.score_tdt_texts(
                  cat_enc[c_start:c_end], all_texts[c_start:c_end],
                  enc_lengths=cat_enc_len[c_start:c_end] if cat_enc_len is not None else None)
              if torch.cuda.is_available():
                torch.cuda.synchronize()
              score_parts.append(c_scores.detach().cpu().float())
            scores_flat = torch.cat(score_parts, dim=0).numpy()
          else:
            scores_flat = model.score_tdt_texts(
                cat_enc, all_texts, enc_lengths=cat_enc_len)
            if torch.cuda.is_available():
              torch.cuda.synchronize()
            scores_flat = scores_flat.detach().cpu().float().numpy()
        except Exception as e:
          _diag(
              f'TDT batch {batch_idx}/{n_batches} failed during score_tdt_texts: '
              f'candidates={total_cands}, sample_uids={batch_uids[:3]}, err={type(e).__name__}: {e}')
          raise
        score_dt = time.time() - score_t0

        offset = 0
        for uid, n_cands in zip(batch_uids, batch_n_cands):
          scores_by_uid[uid] = scores_flat[offset:offset + n_cands]
          offset += n_cands

        now = time.time()
        if verbose and (
            should_log_batch or
            now - last_progress_time >= 30):
          _diag(
              f'TDT batch {batch_idx}/{n_batches} done: '
              f'scored_uids={len(scores_by_uid)}, candidates={total_cands}, '
              f'batch_time={now - batch_t0:.2f}s, forward={forward_dt:.2f}s, '
              f'score_time={score_dt:.2f}s')
          last_progress_time = now
  except Exception as e:
    _diag(
        f'TDT exact failed for {model_dir.name}: scored_uids={len(scores_by_uid)}/{len(candidate_lists)}, '
        f'err={type(e).__name__}: {e}')
    raise

  if verbose:
    _diag(f'Online TDT scored {len(scores_by_uid)} utterances for {model_dir.name}')
  return scores_by_uid


def _run_tree_reranker(all_logprobs, all_model_preds, all_model_pred_map=None,
                       model_ctc_meta=None,
                       nbest=10, beam_width=10):
  """Tree reranker: build features from logprobs, use saved tree model to select best.

  Loads a pre-trained tree model (LGB/XGB/CB) from the packed 'tree_reranker/'
  directory. Computes features identical to offline training (ensemble-fold0.py)
  and uses the tree model to rank/predict the best candidate for each utterance.

  Args:
    all_logprobs: dict {model_name: {uid: numpy (T, V) float16}}
    all_model_preds: dict {uid: [text_model0, text_model1, ...]}
    nbest: N-best candidates per model beam search
    beam_width: beam search width

  Falls back to N-best rescore for utterances without logprobs.
  """
  import numpy as np
  from src.reranker_features import build_reranker_features

  def _augment_audio_meta_features(df):
    audio_meta_feats = {
        'audio_duration_sec',
        'has_audio_duration_sec',
        'chars_per_audio_sec',
        'words_per_audio_sec',
        'audio_minus_frame_duration_sec',
        'audio_to_frame_duration_ratio',
    }
    if not any(col in feat_cols for col in audio_meta_feats):
      return df

    meta_df = preprocess(mode='test')
    uid_col = 'utterance_id' if 'utterance_id' in meta_df.columns else ('id' if 'id' in meta_df.columns else None)
    assert uid_col is not None, 'preprocess(mode=test) output missing utterance_id/id for audio metadata features'

    meta_df = meta_df[[uid_col] + ([ 'audio_duration_sec' ] if 'audio_duration_sec' in meta_df.columns else [])].copy()
    meta_df = meta_df.drop_duplicates(uid_col)
    if 'audio_duration_sec' not in meta_df.columns:
      meta_df['audio_duration_sec'] = np.nan
    meta_df['audio_duration_sec'] = pd.to_numeric(meta_df['audio_duration_sec'], errors='coerce')
    meta_df = meta_df.rename(columns={uid_col: 'uid'})

    df = df.merge(meta_df, on='uid', how='left', suffixes=('', '_meta'))
    if 'audio_duration_sec_meta' in df.columns:
      df['audio_duration_sec'] = df['audio_duration_sec_meta']
      df = df.drop(columns=['audio_duration_sec_meta'])

    text_len = df['text_len'] if 'text_len' in df.columns else df['candidate_text'].fillna('').astype(str).str.len()
    n_spaces = df['n_spaces'] if 'n_spaces' in df.columns else df['candidate_text'].fillna('').astype(str).str.count(' ')
    frame_duration = pd.to_numeric(df['duration_sec'], errors='coerce') if 'duration_sec' in df.columns else pd.Series(np.nan, index=df.index)
    audio_duration = pd.to_numeric(df['audio_duration_sec'], errors='coerce')
    has_audio = audio_duration.notna() & np.isfinite(audio_duration) & (audio_duration > 0)

    df['has_audio_duration_sec'] = has_audio.astype(np.int32)
    df['chars_per_audio_sec'] = np.where(has_audio, text_len / np.maximum(audio_duration, 0.01), np.nan)
    df['words_per_audio_sec'] = np.where(has_audio, (n_spaces + 1) / np.maximum(audio_duration, 0.01), np.nan)
    df['audio_minus_frame_duration_sec'] = np.where(has_audio, audio_duration - frame_duration, np.nan)
    df['audio_to_frame_duration_ratio'] = np.where(
        has_audio,
        audio_duration / np.maximum(frame_duration.fillna(0.0), 0.01),
        np.nan,
    )
    return df

  def _build_tree_uid_folds(uids, n_folds, split_seed):
    from sklearn.model_selection import StratifiedGroupKFold
    from src.preprocess import preprocess

    meta_df = preprocess(mode='test')
    assert 'utterance_id' in meta_df.columns, 'utterance_id missing in preprocess(mode=test) output'
    keep_uids = set(uids)
    meta_df = meta_df[meta_df['utterance_id'].isin(keep_uids)].copy()
    meta_df = meta_df.drop_duplicates('utterance_id')
    assert len(meta_df) == len(keep_uids), (
        f'OOF tree routing uid mismatch: metadata={len(meta_df)} feature_uids={len(keep_uids)}')

    if 'child_id' not in meta_df.columns:
      meta_df['child_id'] = meta_df['utterance_id']
    meta_df['child_id'] = meta_df['child_id'].fillna('').astype(str)
    empty_child_mask = meta_df['child_id'] == ''
    if empty_child_mask.any():
      meta_df.loc[empty_child_mask, 'child_id'] = meta_df.loc[empty_child_mask, 'utterance_id'].astype(str)

    if 'source' not in meta_df.columns:
      source_series = None
      for source_col in ('audio_file', 'audio_path'):
        if source_col in meta_df.columns:
          path_series = meta_df[source_col].fillna('').astype(str).str.lower()
          inferred = np.where(path_series.str.contains('ext_audio'), 'ext', 'dd')
          source_series = pd.Series(inferred, index=meta_df.index)
          break
      if source_series is None:
        source_series = pd.Series(['dd'] * len(meta_df), index=meta_df.index)
      meta_df['source'] = source_series
    meta_df['source'] = meta_df['source'].fillna('').astype(str)
    empty_source_mask = meta_df['source'] == ''
    if empty_source_mask.any():
      inferred = pd.Series(['dd'] * len(meta_df), index=meta_df.index)
      for source_col in ('audio_file', 'audio_path'):
        if source_col in meta_df.columns:
          path_series = meta_df[source_col].fillna('').astype(str).str.lower()
          inferred = pd.Series(np.where(path_series.str.contains('ext_audio'), 'ext', 'dd'), index=meta_df.index)
          break
      meta_df.loc[empty_source_mask, 'source'] = inferred.loc[empty_source_mask]

    if 'age_bucket' not in meta_df.columns:
      meta_df['age_bucket'] = ''
    meta_df['age_bucket'] = meta_df['age_bucket'].fillna('').astype(str)
    meta_df['strat_key'] = meta_df['source'] + '_' + meta_df['age_bucket']

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
    uid_arr = meta_df['utterance_id'].values
    groups = meta_df['child_id'].values
    strat_labels = meta_df['strat_key'].values
    uid_folds = {}
    for fold_i, (_, val_idx) in enumerate(sgkf.split(uid_arr, strat_labels, groups)):
      for idx in val_idx:
        uid_folds[uid_arr[idx]] = fold_i
    assert len(uid_folds) == len(keep_uids), (
        f'Failed to assign OOF folds for all utterances: {len(uid_folds)} vs {len(keep_uids)}')
    return uid_folds

  # Load reranker metadata
  tree_dir = SRC_DIR / 'tree_reranker'
  meta_path = tree_dir / 'reranker_meta.json'
  assert meta_path.exists(), f'reranker_meta.json not found at {meta_path}'
  with open(meta_path) as f:
    reranker_meta = json.load(f)

  tree_models_list = reranker_meta['tree_models']  # e.g. ['cb']
  feat_cols = reranker_meta['feat_cols']
  feat_flags = reranker_meta.get('feat_flags', {})
  feat_tdt = bool(reranker_meta.get('feat_tdt', False))
  feat_tdt_light = bool(reranker_meta.get('feat_tdt_light', feat_tdt))
  feat_tdt_exact = bool(reranker_meta.get('feat_tdt_exact', feat_tdt))
  tdt_feat_topk = int(reranker_meta.get('tdt_feat_topk', 8) or 8)
  tdt_force_keep_preds = bool(reranker_meta.get('tdt_force_keep_preds', True))
  tdt_model_names = reranker_meta.get('tdt_model_names', [])
  feat_tdt_score_compare = bool(reranker_meta.get('feat_flags', {}).get('feat_tdt_score_compare', False))
  feat_tdt_group = bool(reranker_meta.get('feat_flags', {}).get('feat_tdt_group', False))
  feat_wavlm_group = bool(reranker_meta.get('feat_flags', {}).get('feat_wavlm_group', False))
  feat_nemo_group = bool(reranker_meta.get('feat_flags', {}).get('feat_nemo_group', False))
  feat_mbr = bool(reranker_meta.get('feat_flags', {}).get('feat_mbr', False)
                  or reranker_meta.get('feat_mbr', False))
  no_ctc_score_feats = bool(reranker_meta.get('feat_flags', {}).get('no_ctc_score_feats', False)
                            or reranker_meta.get('no_ctc_score_feats', False))
  no_lm_feats = reranker_meta.get('no_lm_feats', False)
  tree_task = reranker_meta.get('tree_task', 'ranking')
  n_folds = reranker_meta.get('n_folds', 5)
  split_seed = int(reranker_meta.get('split_seed', 42) or 42)
  meta_nbest = reranker_meta.get('nbest', nbest)
  meta_beam = reranker_meta.get('beam_width', beam_width)
  tree_infer_mode = os.environ.get('TREE_RERANKER_INFER_MODE', 'avg').strip().lower()
  if tree_infer_mode == 'mean':
    tree_infer_mode = 'avg'
  assert tree_infer_mode in ('avg', 'oof'), (
      f'Unsupported TREE_RERANKER_INFER_MODE={tree_infer_mode}; expected avg or oof')

  def _tree_reranker_requires_lm():
    return (not no_lm_feats) and any(str(col).startswith('lm_score') for col in feat_cols)

  def _load_tree_reranker_lm():
    from src.ctc_decode import load_ngram_lm

    candidate_paths = []
    for key in ('packed_lm_path', 'lm_path'):
      value = str(reranker_meta.get(key, '') or '').strip()
      if value and value != '<none>':
        candidate_paths.append(value)
    candidate_paths.extend([
        'tree_reranker/lm.json',
        'lm.json',
    ])

    checked = []
    for candidate in candidate_paths:
      path_obj = Path(candidate)
      if path_obj.is_absolute():
        resolved_candidates = [path_obj]
      else:
        resolved_candidates = [SRC_DIR / path_obj, tree_dir / path_obj]
      for resolved in resolved_candidates:
        if resolved in checked:
          continue
        checked.append(resolved)
        if resolved.exists():
          lm = load_ngram_lm(str(resolved))
          _diag(f'Loaded tree reranker LM from {resolved}')
          return lm
    return None

  _diag(f'Tree reranker: {tree_models_list}, {len(feat_cols)} features, '
        f'task={tree_task}, folds={n_folds}, nbest={meta_nbest}, beam={meta_beam}')
  _diag(f'  feat_flags: {feat_flags}, no_lm_feats={no_lm_feats}')
  _diag(f'  infer_mode: {tree_infer_mode} (split_seed={split_seed})')
  _set_stage('tree_reranker_setup', f'models={tree_models_list}, infer_mode={tree_infer_mode}')
  if feat_tdt_light or feat_tdt_exact:
    _diag(f'  TDT features: light={feat_tdt_light}, exact={feat_tdt_exact}, '
          f'models={tdt_model_names}, topk={tdt_feat_topk}, force_keep={tdt_force_keep_preds}')
  if feat_tdt_group or feat_wavlm_group or feat_nemo_group:
    _diag(f'  subgroup features: tdt_group={feat_tdt_group}, '
          f'wavlm_group={feat_wavlm_group}, nemo_group={feat_nemo_group}')

  reranker_lm = None
  if _tree_reranker_requires_lm():
    reranker_lm = _load_tree_reranker_lm()
    assert reranker_lm is not None, (
        'Tree reranker requires LM features, but no packed LM was found in the submission bundle. '
        'Re-pack the submission after ensuring the reranker LM file is included.')

  # Load tree models
  loaded_models = {}  # tm -> [model_fold0, model_fold1, ...]
  for tm in tree_models_list:
    models = []
    _loaded_folds = []
    for fi in range(n_folds):
      model_dir = tree_dir / f'tree_{tm}_fold{fi}'
      assert model_dir.exists(), f'Tree model dir not found: {model_dir}'

      # Detect model type and load natively
      model_txt = model_dir / 'model.txt'
      model_json = model_dir / 'model.json'
      model_pkl = model_dir / 'model.pkl'

      if model_txt.exists():
        # LightGBM
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(model_txt))
      elif model_json.exists() and tm == 'cb':
        # CatBoost
        from catboost import CatBoostRanker, CatBoostRegressor
        if tree_task == 'ranking':
          model = CatBoostRanker()
        else:
          model = CatBoostRegressor()
        model.load_model(str(model_json))
      elif model_json.exists() and tm == 'xgb':
        # XGBoost
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(str(model_json))
      elif model_pkl.exists():
        import pickle
        with open(model_pkl, 'rb') as pf:
          model = pickle.load(pf)
      else:
        raise FileNotFoundError(f'No model file found in {model_dir}')
      models.append(model)
      _loaded_folds.append(str(fi))
    loaded_models[tm] = models
    _diag(f'  Loaded {tm} fold models: {",".join(_loaded_folds)}')

  # Use first tree model type for prediction (multi-model: average later)
  primary_tm = tree_models_list[0]

  # Build features — reuse the model_names from reranker_meta
  meta_model_names = reranker_meta['model_names']
  assert len(ENSEMBLE_MODEL_DIRS) == len(meta_model_names), \
      (f'Model count mismatch: ensemble has {len(ENSEMBLE_MODEL_DIRS)} model dirs '
       f'but reranker_meta expects {len(meta_model_names)} model_names')

  # Build mapping between physical dir names and training model names
  dir_to_meta = {}
  model_name_map = {}  # meta_mn -> dir_name
  for i, meta_mn in enumerate(meta_model_names):
    dir_name = ENSEMBLE_MODEL_DIRS[i].name
    dir_to_meta[dir_name] = meta_mn
    model_name_map[meta_mn] = dir_name

  # Remap logprobs: only include models with actual CTC logprobs
  remapped_logprobs = {}
  for dir_name, lp_dict in all_logprobs.items():
    meta_mn = dir_to_meta.get(dir_name)
    if meta_mn and lp_dict:
      remapped_logprobs[meta_mn] = lp_dict

  score_model_names = [mn for mn in meta_model_names if mn in remapped_logprobs]
  _diag(f'Score models (CTC logprobs): {len(score_model_names)}/{len(meta_model_names)}')
  if len(score_model_names) < len(meta_model_names):
    non_score = [mn for mn in meta_model_names if mn not in score_model_names]
    _diag(f'  Candidate-only models (no CTC logprobs): {non_score}')

  # Build eval predictions (greedy text) for ALL models from all_model_pred_map
  all_model_pred_map = all_model_pred_map or {}
  remapped_eval_preds = {}
  for meta_mn in meta_model_names:
    dir_name = model_name_map[meta_mn]
    uid_to_text = {}
    for uid, pred_map in all_model_pred_map.items():
      text = pred_map.get(dir_name, '') if pred_map else ''
      if text:
        uid_to_text[uid] = text.strip()
    remapped_eval_preds[meta_mn] = uid_to_text

  remapped_model_ctc_meta = None
  if model_ctc_meta:
    remapped_model_ctc_meta = {}
    for current_name, meta in model_ctc_meta.items():
      target_name = dir_to_meta.get(current_name, current_name)
      remapped_model_ctc_meta[target_name] = meta

  model_flags_by_name = {}
  for meta_mn in meta_model_names:
    dir_name = model_name_map[meta_mn]
    model_dir = next((d for d in ENSEMBLE_MODEL_DIRS if d.name == dir_name), None)
    model_flags_by_name[meta_mn] = _read_model_flags_for_runtime(model_dir) if model_dir else {}

  uid_sets = [set(uid_map.keys()) for uid_map in remapped_logprobs.values() if uid_map]
  if remapped_eval_preds:
    uid_sets.extend(set(uid_map.keys()) for uid_map in remapped_eval_preds.values() if uid_map)
  n_uids = len(set.union(*uid_sets)) if uid_sets else 0
  _diag(f'Building features for {n_uids} utterances...')
  _set_stage('tree_reranker_build_features', f'uids={n_uids}')
  # Filter feat_flags to only those accepted by build_reranker_features
  _accepted_feat_flags = {
      'feat_text', 'feat_ipa', 'feat_ctc_stats', 'feat_audio',
      'feat_consensus', 'feat_mbr', 'feat_group_ext', 'feat_align', 'feat_logprob_proxy',
      'no_ctc_score_feats',
  }
  _filtered_flags = {k: v for k, v in feat_flags.items() if k in _accepted_feat_flags}
  df, all_feat_cols = build_reranker_features(
      remapped_logprobs, meta_model_names,
      nbest=meta_nbest, beam_width=meta_beam, verbose=True,
      lm=reranker_lm, no_lm_feats=no_lm_feats, all_eval_preds=remapped_eval_preds,
      model_ctc_meta=remapped_model_ctc_meta,
      normalize_text_fn=_get_runtime_text_normalizer(),
      **_filtered_flags)
  df = _augment_audio_meta_features(df)

  if (feat_tdt_light or feat_tdt_exact) and tdt_model_names:
    _set_stage('tree_reranker_tdt_features',
               f'exact={feat_tdt_exact}, light={feat_tdt_light}, models={tdt_model_names}')
    all_model_pred_map = all_model_pred_map or {}
    primary_tdt_texts = {}
    for meta_mn in tdt_model_names:
      actual_mn = model_name_map.get(meta_mn)
      uid_to_text = {}
      if actual_mn is not None:
        for uid, pred_map in all_model_pred_map.items():
          text = pred_map.get(actual_mn, '') if pred_map else ''
          if text:
            uid_to_text[uid] = text.strip()
      primary_tdt_texts[meta_mn] = uid_to_text

    tdt_score_maps = None
    if feat_tdt_exact:
      candidate_lists = _build_tdt_feature_candidate_lists(
          df,
          tdt_model_names,
          primary_tdt_texts,
          topk=tdt_feat_topk,
          force_keep_preds=tdt_force_keep_preds,
      )
      tdt_score_maps = {}
      for meta_mn in tdt_model_names:
        actual_mn = model_name_map.get(meta_mn)
        if actual_mn is None:
          continue
        # Find model_dir by matching dir name
        model_dir = None
        for d in ENSEMBLE_MODEL_DIRS:
          if d.name == actual_mn:
            model_dir = d
            break
        assert model_dir is not None, f'Model dir not found for {actual_mn}'
        _diag(f'TDT exact scoring begin: meta_model={meta_mn}, dir={model_dir.name}')
        scores_by_uid = _score_tdt_candidates_for_model_dir(model_dir, candidate_lists, verbose=True)
        tdt_score_maps[meta_mn] = _convert_tdt_score_arrays(candidate_lists, scores_by_uid)
        _diag(f'TDT exact scoring end: meta_model={meta_mn}, scored={len(scores_by_uid)}')
    df, _ = _augment_tdt_feature_frame(
        df,
        tdt_model_names,
        primary_tdt_texts,
        tdt_score_maps=tdt_score_maps,
        include_light=feat_tdt_light,
        include_exact=feat_tdt_exact,
        include_score_compare=feat_tdt_score_compare,
    )
    if feat_tdt_group:
      df, feat_cols = _augment_family_group_features(
          df, feat_cols, meta_model_names, 'tdt', tdt_model_names, primary_texts=primary_tdt_texts)

  wavlm_model_names = reranker_meta.get('wavlm_model_names') or []
  if feat_wavlm_group:
    if not wavlm_model_names:
      wavlm_model_names = _detect_wavlm_model_names(meta_model_names, flags_by_name=model_flags_by_name)
    if wavlm_model_names:
      df, feat_cols = _augment_family_group_features(
          df, feat_cols, meta_model_names, 'wavlm', wavlm_model_names)

  nemo_model_names = reranker_meta.get('nemo_model_names') or []
  if feat_nemo_group:
    if not nemo_model_names:
      nemo_model_names = _detect_nemo_model_names(meta_model_names, flags_by_name=model_flags_by_name)
    if nemo_model_names:
      df, feat_cols = _augment_family_group_features(
          df, feat_cols, meta_model_names, 'nemo', nemo_model_names)

  _maybe_dump_tree_reranker_features(df, feat_cols)

  # Validate features match training (feat_cols from meta)
  missing_feats = [f for f in feat_cols if f not in df.columns]
  assert not missing_feats, (
      f'Online reranker feature mismatch: {len(missing_feats)} features in '
      f'reranker_meta.json but not built online: {missing_feats}. '
      f'Retrain tree reranker with matching feature flags or fix feature building.')

  X = df[feat_cols].values
  _diag(f'Feature matrix: {X.shape}')
  _set_stage('tree_reranker_predict', f'rows={X.shape[0]}, cols={X.shape[1]}')

  def _predict_tree_scores(tm, model):
    if tm == 'lgb':
      return model.predict(X)
    if tm == 'cb':
      import pandas as _pd
      return model.predict(_pd.DataFrame(X, columns=feat_cols))
    if tm == 'xgb':
      import xgboost as xgb
      return model.predict(xgb.DMatrix(X, feature_names=feat_cols))
    raise ValueError(f'Unsupported tree model type: {tm}')

  if tree_infer_mode == 'avg':
    # Submission mode: average across all inner-fold tree models.
    all_scores = np.zeros(len(df), dtype=np.float64)
    n_models_total = 0
    for tm in tree_models_list:
      for fi, model in enumerate(loaded_models[tm]):
        all_scores += _predict_tree_scores(tm, model)
        n_models_total += 1
    all_scores /= n_models_total
  else:
    # Local validation mode: simulate offline OOF routing by sending each uid
    # only through the tree fold where it was validation data during training.
    uid_folds = _build_tree_uid_folds(df['uid'].unique(), n_folds=n_folds, split_seed=split_seed)
    df['tree_fold'] = df['uid'].map(uid_folds).astype(int)
    fold_counts = df.groupby('tree_fold')['uid'].nunique().to_dict()
    _diag(f'  OOF tree routing uid counts: {fold_counts}')

    row_fold_idx = df['tree_fold'].to_numpy(dtype=np.int64)
    row_idx = np.arange(len(df), dtype=np.int64)
    all_scores = np.zeros(len(df), dtype=np.float64)
    n_tree_types = 0
    for tm in tree_models_list:
      fold_scores = []
      for fi, model in enumerate(loaded_models[tm]):
        fold_scores.append(np.asarray(_predict_tree_scores(tm, model), dtype=np.float64))
      fold_scores = np.stack(fold_scores, axis=0)
      all_scores += fold_scores[row_fold_idx, row_idx]
      n_tree_types += 1
    all_scores /= max(n_tree_types, 1)

  df['tree_score'] = all_scores

  # Select best candidate per utterance
  predictions = {}
  for uid, group in df.groupby('uid'):
    if tree_task == 'ranking':
      best_idx = group['tree_score'].idxmax()
    else:
      # regression on CER → lower is better
      best_idx = group['tree_score'].idxmin()
    predictions[uid] = group.loc[best_idx, 'candidate_text']

  _diag(f'Tree reranker selected predictions for {len(predictions)} utterances')
  _set_stage('tree_reranker_done', f'predictions={len(predictions)}')

  # Fallback for utterances without logprobs
  n_fallback = 0
  for uid, candidates in all_model_preds.items():
    if uid not in predictions:
      if len(set(candidates)) == 1:
        predictions[uid] = candidates[0]
      else:
        predictions[uid] = _mbr_select(candidates)
      n_fallback += 1
  if n_fallback > 0:
    _diag(f'MBR fallback for {n_fallback} utterances without logprobs')

  return predictions


def run_ensemble_inference(device):
  """Run inference with multiple models, N-best rescore or MBR text ensemble.
  
  Mode is controlled by ENSEMBLE_MODE env var:
    'nbest_rescore' (default): extract CTC logprobs, run N-best rescoring
    'text': MBR text-level selection (original behavior)
    'tree_reranker': extract CTC logprobs, build features, use tree model to select best
  
  For nbest_rescore/tree_reranker: logprobs are kept in memory (no disk I/O).
  Estimated memory: ~4.5 GB per model for 77K utterances (float16).
  Falls back to MBR text if logprobs unavailable (non-CTC models).
  """
  import torch
  import numpy as np
  from src.dataset import get_dl
  from src.preprocess import preprocess

  ensemble_mode = os.environ.get('ENSEMBLE_MODE', '').strip().lower()
  if not ensemble_mode:
    # Fallback: read from ensemble_meta.json if present
    ensemble_mode = _ensemble_meta.get('ensemble_mode', 'nbest_rescore') if IS_ENSEMBLE else 'nbest_rescore'
  assert ensemble_mode in ('text', 'nbest_rescore', 'tree_reranker'), \
      f'ENSEMBLE_MODE must be "text", "nbest_rescore", or "tree_reranker", got "{ensemble_mode}"'
  collect_logprobs = (ensemble_mode in ('nbest_rescore', 'tree_reranker'))
  _diag(f'Ensemble mode: {ensemble_mode} (collect_logprobs={collect_logprobs})')

  # Setup FLAGS from the primary (first) model — all models share architecture
  global MODEL_DIR
  MODEL_DIR = ENSEMBLE_MODEL_DIRS[0]
  inference_model_dirs = ENSEMBLE_INFER_MODEL_DIRS or ENSEMBLE_MODEL_DIRS
  global _ensemble_final_flags_logged
  setup_flags(log_final=not _ensemble_final_flags_logged)
  _ensemble_final_flags_logged = True

  # Adaptive batch size after first model load
  _diag(f'FLAGS: model={FLAGS.model}, backbone={FLAGS.backbone}')

  # Prepare test data once (shared across all models)
  df = preprocess(mode='test')
  sort_by_duration = os.environ.get('SUBMIT_SORT_BY_DURATION', '1').strip().lower() not in ('0', 'false', 'no')
  # Sort by duration (longest-first) to minimize padding waste in batches.
  if sort_by_duration and 'audio_duration_sec' in df.columns:
    df = df.sort_values('audio_duration_sec', ascending=False).reset_index(drop=True)
    _diag(f'Test data: {len(df)} utterances (sorted by duration, longest-first)')
  else:
    _diag(f'Test data: {len(df)} utterances (original order)')

  # Collect predictions from each model: {uid: [pred_model0, pred_model1, ...]}
  all_model_preds = {}  # uid -> list of texts
  all_model_pred_map = {}  # uid -> {model_name: primary_text}
  # In-memory logprobs for N-best rescore: {model_name: {uid: numpy (T_i, V)}}
  all_logprobs = {}  # model_name -> {uid -> numpy float16}
  all_model_ctc_meta = {}
  skip_tdt_primary_ctc_candidates = os.environ.get(
      'ENSEMBLE_SKIP_TDT_PRIMARY_CTC_CANDIDATES', '1').strip().lower() not in ('0', 'false', 'no')
  skip_ctc_candidate_models = {}
  missing_logprobs_warned = False

  for model_idx, model_dir in enumerate(inference_model_dirs):
    MODEL_DIR = model_dir
    FLAGS.model_dir = str(model_dir)

    # Restore this model's own training flags before constructing the model.
    saved_flags = setup_flags(log_final=False)

    derived_filter = _derive_model_audio_filter_from_saved_flags(model_dir.name, saved_flags)
    if derived_filter:
      MODEL_AUDIO_FILTERS[model_dir.name] = {
          key: value for key, value in derived_filter.items() if not key.startswith('_')
      }
      _diag(
          f'Auto-derived model audio filter for {model_dir.name}: '
          f'max_audio_sec={derived_filter["max_audio_sec"]} '
          f'(source={derived_filter.get("_source", "saved_flags")})')

    model_df, filter_info = _filter_df_for_model_audio(df, model_dir.name)
    if filter_info:
      _diag(
          f'--- Model {model_idx+1}/{len(inference_model_dirs)}: {model_dir.name} duration filter '
          f'keep {filter_info["kept"]}/{filter_info["total"]} '
          f'(drop {filter_info["dropped"]}, rule={filter_info["rule"]}) ---')
      if filter_info['kept'] == 0:
        _diag(f'  Skipping {model_dir.name}: duration filter excluded all utterances')
        continue

    # If this model has its own flags.json, update relevant FLAGS
    _mn = ''
    if saved_flags:
      _mn = saved_flags.get('model_name', '') or saved_flags.get('mn', '')

    decode_method = str(saved_flags.get('decode_method', 'auto') or 'auto')
    s2s_decoder = str(saved_flags.get('s2s_decoder', 'native') or 'native')
    ctc_only = bool(saved_flags.get('ctc_only', False))
    skip_ctc_candidate_models[model_dir.name] = (
        skip_tdt_primary_ctc_candidates
        and decode_method == 'tdt'
        and s2s_decoder == 'tdt_reuse'
        and not ctc_only)

    model_parts = [
        f'--- Model {model_idx+1}/{len(inference_model_dirs)}: {model_dir.name}',
    ]
    if _mn:
      model_parts.append(f'({_mn})')
    model_parts.append(
        f'[{getattr(FLAGS, "model", "")}:decode={getattr(FLAGS, "decode_method", "auto")},'
        f'ctc={getattr(FLAGS, "ctc_weight", 0)},'
        f's2s={getattr(FLAGS, "s2s_decoder", "native")},'
        f'fast={_bool_flag(os.environ.get("SUBMIT_ENABLE_FAST_DECODE", "1").strip().lower() not in ("0", "false", "no"))}]')
    if skip_ctc_candidate_models[model_dir.name]:
      model_parts.append('[skip_ctc_candidates]')
    if filter_info:
      model_parts.append(
          f'[filter keep {filter_info["kept"]}/{filter_info["total"]}, drop {filter_info["dropped"]}]')
    _diag(' '.join(model_parts))

    # Suppress verbose NeMo + gezi logging for 2nd+ models (save log space)
    if model_idx > 0:
      _suppress_nemo_logging()
    _suppress_gezi_logging()

    # Load model
    _set_stage('ensemble_model_load_begin',
               f'index={model_idx+1}/{len(inference_model_dirs)}, dir={model_dir.name}',
               log_stdout=False)
    model = load_model(device, verbose=False)
    if collect_logprobs:
      model_ctc_meta = _build_runtime_model_ctc_meta(model)
      if model_ctc_meta:
        all_model_ctc_meta[model_dir.name] = model_ctc_meta
    _set_stage('ensemble_model_load_done',
               f'index={model_idx+1}/{len(inference_model_dirs)}, dir={model_dir.name}',
               log_stdout=False)

    # Restore logging after load
    _restore_gezi_logging()
    if model_idx > 0:
      _restore_nemo_logging()
    _runtime_diag = (model_idx == 0)
    _scale_batch_size(model, verbose=False)
    runtime_profile = _infer_runtime_profile(model, device)

    # Run inference — collect both text predictions and (optionally) CTC logprobs
    test_dl = get_dl(mode='test', df=model_df)
    gz.set('do_generate', True)

    n_done = 0
    n_total = len(model_df)
    _infer_start = time.time()
    model_name = model_dir.name
    if collect_logprobs:
      all_logprobs[model_name] = {}

    # Only show progress milestones for the first model (save log lines for 2nd+)
    _quiet_word_logs = _is_word_runtime()
    _show_progress = (model_idx == 0 and
              _show_ensemble_progress_logs(len(inference_model_dirs), _quiet_word_logs))
    _MILESTONES = [25, 50, 100] if _quiet_word_logs else [10, 25, 50, 75, 100]
    _next_ms = 0
    _split_log_count = 0
    _split_log_suppressed = 0
    _oom_count = 0

    _cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    _amp_dtype = torch.bfloat16 if _cc >= (8, 0) else torch.float16

    def _decode_sub_batch(input_batch):
      res = model(input_batch)
      from src.eval import decode_predictions
      return decode_predictions(res, model=model), res

    def _collect_results(src_batch, texts, res):
      nonlocal missing_logprobs_warned
      if 'id' in src_batch:
        for uid, text in zip(src_batch['id'], texts):
          if uid not in all_model_preds:
            all_model_preds[uid] = []
          all_model_preds[uid].append(text)
          if uid not in all_model_pred_map:
            all_model_pred_map[uid] = {}
          all_model_pred_map[uid][model_name] = text

      if collect_logprobs:
        log_probs = getattr(model, '_last_ctc_log_probs', None)
        enc_len = getattr(model, '_last_enc_len', None)
        if log_probs is not None and 'id' in src_batch:
          batch_ids = src_batch['id']
          batch_size = min(log_probs.shape[0], len(batch_ids))
          for i in range(batch_size):
            uid = batch_ids[i]
            if enc_len is not None:
              actual_len = min(int(enc_len[i].item()), log_probs.shape[1])
            else:
              actual_len = log_probs.shape[1]
            all_logprobs[model_name][uid] = log_probs[i, :actual_len].cpu().to(torch.float16).numpy()
        elif log_probs is None and n_done == 0 and not missing_logprobs_warned:
          missing_logprobs_warned = True
          _diag('  WARNING: _last_ctc_log_probs not available for this model; it will be excluded from CTC rescoring')

    def _safe_single(batch, idx):
      nonlocal _oom_count
      single = _slice_batch(batch, idx, idx + 1, device)
      try:
        texts, res = _decode_sub_batch(single)
        _collect_results(single, texts, res)
        return texts
      except torch.cuda.OutOfMemoryError:
        del single
        torch.cuda.empty_cache()
        _oom_count += 1
        uid = batch['id'][idx] if 'id' in batch and idx < len(batch['id']) else '?'
        if not _cuda_ok(device):
          _diag(f'  CUDA corrupted after OOM (model={model_name}, item={uid}), empty prediction')
          return ['']
        _diag(f'  OOM at bs=1 (model={model_name}, item={uid}), truncating to {runtime_profile["max_trunc_sec"]:.0f}s')
        trunc = _truncate_single_batch(_slice_batch(batch, idx, idx + 1, device),
                                       runtime_profile['sample_rate'],
                                       runtime_profile['max_trunc_sec'])
        try:
          texts, res = _decode_sub_batch(trunc)
          _collect_results(trunc, texts, res)
          return texts
        except Exception:
          torch.cuda.empty_cache()
          _diag(f'  Failed even after truncation (model={model_name}, item={uid}), empty prediction')
          return ['']

    def _infer_range(batch, start, end, max_dur):
      nonlocal _oom_count
      sub_bs = end - start
      sub_batch = _slice_batch(batch, start, end, device)
      try:
        texts, res = _decode_sub_batch(sub_batch)
        _collect_results(sub_batch, texts, res)
        return texts
      except torch.cuda.OutOfMemoryError:
        del sub_batch
        torch.cuda.empty_cache()
        _oom_count += 1
        if not _cuda_ok(device):
          _diag(f'  CUDA corrupted after OOM (model={model_name}, bs={sub_bs}, dur={max_dur:.1f}s), empty predictions')
          return [''] * sub_bs
        if sub_bs <= 1:
          return _safe_single(batch, start)
        next_bs = max(1, sub_bs // 2)
        _diag(f'  OOM at sub-batch (model={model_name}, bs={sub_bs}, dur={max_dur:.1f}s), shrinking to <= {next_bs}')
        texts = []
        for chunk_start in range(start, end, next_bs):
          chunk_end = min(chunk_start + next_bs, end)
          texts.extend(_infer_range(batch, chunk_start, chunk_end, max_dur))
        return texts

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=_amp_dtype):
      for batch in test_dl:
        bs = len(batch['id']) if 'id' in batch else (
            batch['input_features'].shape[0] if 'input_features' in batch else 1)
        max_dur = _get_batch_max_dur(batch, runtime_profile['sample_rate'])
        safe_bs = _duration_safe_bs(max_dur, bs, runtime_profile)
        if safe_bs < bs:
          if not _runtime_diag:
            pass
          elif _quiet_word_logs:
            _split_log_count += 1
            if _split_log_count <= 3 or safe_bs == 1:
              _diag(f'  Proactive split [{model_name}]: bs {bs}->{safe_bs} (max_dur={max_dur:.1f}s)')
            else:
              _split_log_suppressed += 1
          else:
            _diag(f'  Proactive split [{model_name}]: bs {bs}->{safe_bs} (max_dur={max_dur:.1f}s)')

        texts = []
        for chunk_start in range(0, bs, safe_bs):
          chunk_end = min(chunk_start + safe_bs, bs)
          texts.extend(_infer_range(batch, chunk_start, chunk_end, max_dur))

        n_done += len(texts)
        if _show_progress:
          pct = 100.0 * n_done / n_total if n_total > 0 else 0
          if _next_ms < len(_MILESTONES) and pct >= _MILESTONES[_next_ms]:
            elapsed = time.time() - _infer_start
            speed = n_done / elapsed if elapsed > 0 else 0
            print(f'[{_ts()}][submit2]   [{model_dir.name}] {n_done}/{n_total} ({pct:.0f}%) '
                  f'[{elapsed:.0f}s, {speed:.0f} utt/s]', flush=True)
            while _next_ms < len(_MILESTONES) and pct >= _MILESTONES[_next_ms]:
              _next_ms += 1

    elapsed = time.time() - _infer_start
    total_elapsed = time.time() - TIMER_START
    _lp_info = ''
    if collect_logprobs and model_name in all_logprobs:
      n_lp = len(all_logprobs[model_name])
      mem_bytes = sum(v.nbytes for v in all_logprobs[model_name].values())
      _lp_info = f', logprobs: {n_lp} utt ({mem_bytes/1e9:.1f}GB)'
    if _oom_count > 0:
      _diag(f'  OOM fallback triggered {_oom_count} time(s) for {model_name}')
    if _runtime_diag and _split_log_suppressed > 0:
      _diag(f'  Proactive split [{model_name}]: suppressed {_split_log_suppressed} additional events')
    done_parts = [
        f'  Done: {n_done} preds in {elapsed:.1f}s ({n_done/max(elapsed,1):.1f} utt/s){_lp_info},',
        f'total_elapsed={total_elapsed:.1f}s',
    ]
    if FLAGS.eval_batch_size:
      done_parts.append(f'eval_bs={FLAGS.eval_batch_size}')
    _diag(' '.join(done_parts))

    # Free GPU memory before loading next model
    _set_stage('ensemble_model_infer_done',
           f'index={model_idx+1}/{len(inference_model_dirs)}, dir={model_dir.name}, done={n_done}',
           log_stdout=False)
    del model
    torch.cuda.empty_cache()

  # Select ensemble method
  available_logprobs = {k: v for k, v in all_logprobs.items() if v}

  if collect_logprobs and available_logprobs:
    _maybe_dump_online_ctc_logprobs(available_logprobs)

  if collect_logprobs and ensemble_mode == 'tree_reranker' and available_logprobs:
    total_mem = sum(sum(v.nbytes for v in lp.values()) for lp in available_logprobs.values())
    _diag(f'Total logprobs memory: {total_mem/1e9:.1f} GB')

    _diag(f'Running tree reranker with {len(available_logprobs)}/{len(ENSEMBLE_MODEL_DIRS)} models having CTC logprobs...')
    predictions = _run_tree_reranker(
      all_logprobs,
      all_model_preds,
      all_model_pred_map=all_model_pred_map,
      model_ctc_meta=all_model_ctc_meta)
    # Free logprobs memory
    del all_logprobs
  elif collect_logprobs and ensemble_mode == 'nbest_rescore' and available_logprobs:
    total_mem = sum(sum(v.nbytes for v in lp.values()) for lp in available_logprobs.values())
    _diag(f'Total logprobs memory: {total_mem/1e9:.1f} GB')
    if len(available_logprobs) < len(ENSEMBLE_MODEL_DIRS):
      _diag(f'Running N-best rescore with {len(available_logprobs)}/{len(ENSEMBLE_MODEL_DIRS)} models providing CTC logprobs')
    else:
      _diag(f'Running N-best rescore with {len(available_logprobs)} models...')
    predictions = _run_nbest_rescore(
        available_logprobs,
        all_model_preds,
        all_model_pred_map=all_model_pred_map,
        skip_ctc_candidate_models=skip_ctc_candidate_models)
    del all_logprobs
  else:
    if collect_logprobs:
      _diag('Falling back to MBR text (CTC logprobs unavailable for all rescoring models)')
    _diag(f'MBR text selection across {len(ENSEMBLE_MODEL_DIRS)} models for {len(all_model_preds)} utterances (metric={_resolve_mbr_metric()})...')
    _mbr_start = time.time()
    predictions = {}
    n_unanimous = 0
    n_mbr = 0
    for uid, candidates in all_model_preds.items():
      if len(set(candidates)) == 1:
        predictions[uid] = candidates[0]
        n_unanimous += 1
      else:
        predictions[uid] = _mbr_select(candidates)
        n_mbr += 1
    _diag(f'MBR done: {n_unanimous} unanimous, {n_mbr} MBR-selected ({time.time()-_mbr_start:.1f}s)')

  return predictions


# ===========================================================================
#  Main
# ===========================================================================
def main():
  import torch
  import traceback
  
  try:
    _set_stage('main_start', f'ensemble={IS_ENSEMBLE}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _diag(f'Device: {device}, GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A"}')
    
    if IS_ENSEMBLE:
      # ---- Ensemble MBR path ----
      _diag(f'Running ENSEMBLE inference with {len(ENSEMBLE_MODEL_DIRS)} models')
      _set_stage('ensemble_inference', f'models={len(ENSEMBLE_MODEL_DIRS)}')
      predictions = run_ensemble_inference(device)
      _diag(f'Ensemble inference done: {len(predictions)} preds ({time.time() - TIMER_START:.1f}s)')
    else:
      # ---- Single model path (original) ----
      # Setup
      setup_flags()
      _diag(f'FLAGS: model={FLAGS.model}, backbone={FLAGS.backbone}, ctc_weight={getattr(FLAGS, "ctc_weight", 0)}')
      # Log model identity so online smoke logs show which model is running
      _mn = getattr(FLAGS, 'model_name', '') or getattr(FLAGS, 'mn', '')
      _model_type = ''
      _pack_dir = ''
      _train_dir = ''
      # Read model_meta.json for pack-time info (more reliable than FLAGS on Docker)
      _meta_file = MODEL_DIR / 'model_meta.json'
      if _meta_file.exists():
        try:
          import json as _json
          _meta = _json.load(open(_meta_file))
          if not _mn:
            _mn = _meta.get('model_name', '') or _meta.get('mn', '')
          _pack_dir = _meta.get('pack_model_dir', '')
          _train_dir = _meta.get('model_dir', '')
          _model_type = _meta.get('model_type', '')
        except Exception:
          pass
      # Show the most informative model dir: pack source > training flags > Docker local
      _show_dir = _pack_dir or _train_dir or getattr(FLAGS, 'model_dir', '')
      _diag(f'Model name: {_mn}')
      _diag(f'Model dir:  {_show_dir}')
      if _model_type:
        _diag(f'Model type: {_model_type}')
      
      # Load model
      _suppress_gezi_logging()
      _set_stage('single_model_load')
      model = load_model(device)
      _restore_gezi_logging()
      _diag(f'Model loaded ({time.time() - TIMER_START:.1f}s)')
      
      # Run inference
      _set_stage('single_model_inference')
      predictions = run_inference(model, device)
      _diag(f'Inference done: {len(predictions)} preds ({time.time() - TIMER_START:.1f}s)')
    
    pred_summary = _prediction_summary(predictions)

    # Show sample predictions
    _quiet_word_logs = _is_word_runtime()
    sample_keys = list(predictions.keys())[:(3 if _quiet_word_logs else 10)]
    if sample_keys:
      _sample_log = _log.debug if _quiet_word_logs else _log.info
      _sample_log('Sample predictions (first %d):', len(sample_keys))
      for uid in sample_keys:
        _sample_log(f'  [{uid}] {predictions[uid][:150]}')
    if PROBE_MODE and IS_ENSEMBLE:
      _diag('[probe] ensemble mode detected: report currently focuses on single-model probe flow')
    
    # throw_exception mode: abort after inference to avoid consuming smoke quota
    throw_flag = Path(__file__).parent / 'throw_exception'
    if throw_flag.exists():
      elapsed = time.time() - TIMER_START
      n = len(predictions)
      _log.warning('throw_exception flag detected — aborting after inference (smoke speed-test mode)')
      raise RuntimeError(
          f'[throw_exception] Inference completed successfully: {n} utterances '
          f'in {elapsed:.1f}s ({n/elapsed:.0f} utt/s). '
          f'Intentionally aborting to avoid consuming smoke quota.'
      )
    
    # Write output
    _set_stage('write_submission', f'predictions={len(predictions)}')
    submission_info = write_submission(predictions)
    _set_stage('success')
    _flush_diag_budget()
    _diag(f'Prediction summary: total={pred_summary["count"]}, non_empty={pred_summary["non_empty"]}, '
          f'empty={pred_summary["empty"]}, avg_chars_non_empty={pred_summary["avg_chars_non_empty"]:.1f}, '
          f'max_chars={pred_summary["max_chars"]}', force=True)
    _diag(f'Submission summary: path={submission_info["submission_path"]}, '
          f'column={submission_info["pred_col"]}, written={submission_info["written"]}, '
          f'non_empty={submission_info["non_empty"]}, empty={submission_info["empty"]}, '
          f'avg_chars_non_empty={submission_info["avg_chars_non_empty"]:.1f}', force=True)
    _diag(f'SUCCESS - Total time: {time.time() - TIMER_START:.1f}s', force=True)
  except Exception as e:
    _set_stage('fatal_error', f'{type(e).__name__}: {e}')
    # Ensure error is ALWAYS visible in Docker log.txt (even with 500 line cap)
    err_msg = f'FATAL ERROR: {type(e).__name__}: {e}'
    tb = traceback.format_exc()
    _flush_diag_budget()
    _diag(err_msg, force=True)
    _diag(tb[-2000:], force=True)  # Last 2000 chars of traceback
    # Also write to separate error file
    try:
      err_file = Path('/code_execution/submission/error.txt')
      err_file.write_text(f'{err_msg}\n\n{tb}')
    except Exception:
      pass
    raise


if __name__ == '__main__':
  main()
