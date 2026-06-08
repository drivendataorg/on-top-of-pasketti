#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige
#          \date   2025-02-13
#   \Description   Shared audio dataset & dataloader for Pasketti ASR
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect
import random
import numpy as np
from gezi.common import *
import traceback
import torch

from torch.utils.data import Dataset as TorchDataset
from src.config import *
from src.preprocess import *
from src.train_sampling import TemperatureSampler, build_temperature_group_keys


# ===================== Audio Augmentation =====================

_AUG_ONCE_LOCAL_KEYS = set()

def _aug_debug_context(info=None):
  if not info:
    return {}
  keep = {}
  for key in ['utterance_id', 'source', 'age_bucket', 'audio_duration_sec']:
    if key in info and info[key] not in [None, '']:
      keep[key] = info[key]
  if 'audio_duration_sec' in keep:
    keep['audio_duration_sec'] = round(float(keep['audio_duration_sec']), 3)
  return keep


def _can_log_aug_once():
  if not getattr(FLAGS, 'aug_show_once', False):
    return False
  try:
    worker = torch.utils.data.get_worker_info()
    return worker is None or worker.id == 0
  except Exception:
    return True


def _log_aug_once(name, info=None, key=None):
  if not _can_log_aug_once():
    return
  aug_flag = name if name.startswith('aug_') else f'aug_{name}'
  payload = {'aug': name, 'aug_flag': aug_flag}
  if info:
    payload.update(info)
  ic_once(payload, key=key or f'aug_once:{name}')


def _should_capture_aug_once(key):
  if not _can_log_aug_once():
    return False
  if key in _AUG_ONCE_LOCAL_KEYS:
    return False
  _AUG_ONCE_LOCAL_KEYS.add(key)
  return True


def _array_summary(x):
  arr = np.asarray(x)
  if arr.size == 0:
    return {'shape': list(arr.shape), 'size': 0}
  arr = arr.astype(np.float32, copy=False)
  return {
      'shape': list(arr.shape),
      'size': int(arr.size),
      'mean': round(float(arr.mean()), 6),
      'std': round(float(arr.std()), 6),
      'min': round(float(arr.min()), 6),
      'max': round(float(arr.max()), 6),
      'zero_frac': round(float(np.mean(arr == 0)), 6),
  }


def _before_after_summary(before, after):
  return {
      'before': _array_summary(before),
      'after': _array_summary(after),
  }


def _slice_by_ratio(seq, start_frac, keep_frac):
  n = len(seq)
  if n == 0:
    return seq
  keep = max(1, int(n * keep_frac))
  start = int(n * start_frac)
  start = max(0, min(start, n - 1))
  end = min(n, start + keep)
  return seq[start:end]


def _truncate_text_by_ratio(text, start_frac, keep_frac):
  text = (text or '').strip()
  if not text:
    return ''
  # Prefer whitespace-delimited units for word transcripts / word-separated IPA.
  units = text.split()
  if len(units) > 1:
    return ' '.join(_slice_by_ratio(units, start_frac, keep_frac))
  return ''.join(_slice_by_ratio(list(text), start_frac, keep_frac))


def _estimate_aug_mix_label_units(text):
  text = (text or '').strip()
  if not text:
    return 0
  track = getattr(FLAGS, 'track', '')
  score_metric = getattr(FLAGS, 'score_metric', '')
  if track == 'word' or score_metric == 'wer':
    return max(len(text.split()), 1)
  return max(len(text.replace(' ', '')), 1)


def _get_aug_mix_guard(partner_audio_len, partner_label_text,
                       total_audio_len, total_label_units):
  partner_label_units = _estimate_aug_mix_label_units(partner_label_text)
  mixed_label_units = total_label_units + partner_label_units
  max_label_units = int(getattr(FLAGS, 'aug_mix_max_label_units', 0) or 0)
  if (getattr(FLAGS, 'aug_mix_fit_label', False)
      and max_label_units > 0
      and mixed_label_units > max_label_units):
    return True, mixed_label_units, None
  max_cost = float(getattr(FLAGS, 'aug_mix_max_cost', 0) or 0)
  if getattr(FLAGS, 'aug_mix_fit_cost', False) and max_cost > 0:
    mixed_audio_sec = (total_audio_len + partner_audio_len) / FLAGS.sample_rate
    mixed_cost = mixed_audio_sec * max(mixed_label_units, 1)
    if mixed_cost > max_cost:
      return True, mixed_label_units, mixed_cost
  return False, mixed_label_units, None


def _aug_speed(audio, sr, debug_info=None):
  """Speed perturbation: change speed without changing pitch."""
  _debug_key = 'aug_once:speed'
  _capture = _should_capture_aug_once(_debug_key)
  _before = audio.copy() if _capture else None
  factor = random.uniform(FLAGS.aug_speed_min, FLAGS.aug_speed_max)
  if abs(factor - 1.0) < 1e-3:
    return audio
  # Resample: stretch then resample back to original sr
  import librosa
  audio = librosa.effects.time_stretch(audio, rate=factor)
  if _capture:
    _log_aug_once('speed', {
        'factor': round(float(factor), 4),
        **_before_after_summary(_before, audio),
        **_aug_debug_context(debug_info),
    }, key=_debug_key)
  return audio


def _aug_noise(audio, debug_info=None):
  """Add Gaussian noise at random SNR."""
  _debug_key = 'aug_once:noise'
  _capture = _should_capture_aug_once(_debug_key)
  _before = audio.copy() if _capture else None
  snr_db = random.uniform(FLAGS.aug_noise_snr_min, FLAGS.aug_noise_snr_max)
  rms_signal = np.sqrt(np.mean(audio ** 2)) + 1e-9
  rms_noise = rms_signal / (10 ** (snr_db / 20))
  noise = np.random.normal(0, rms_noise, audio.shape).astype(audio.dtype)
  audio = audio + noise
  if _capture:
    _log_aug_once('noise', {
        'snr_db': round(float(snr_db), 3),
        **_before_after_summary(_before, audio),
        **_aug_debug_context(debug_info),
    }, key=_debug_key)
  return audio


def _aug_volume(audio, debug_info=None):
  """Random volume/gain perturbation."""
  _debug_key = 'aug_once:volume'
  _capture = _should_capture_aug_once(_debug_key)
  _before = audio.copy() if _capture else None
  gain = random.uniform(FLAGS.aug_volume_min, FLAGS.aug_volume_max)
  audio = audio * gain
  if _capture:
    _log_aug_once('volume', {
        'gain': round(float(gain), 4),
        **_before_after_summary(_before, audio),
        **_aug_debug_context(debug_info),
    }, key=_debug_key)
  return audio


def _aug_pitch(audio, sr, debug_info=None):
  """Pitch shift by random semitones."""
  _debug_key = 'aug_once:pitch'
  _capture = _should_capture_aug_once(_debug_key)
  _before = audio.copy() if _capture else None
  import librosa
  n_steps = random.uniform(-FLAGS.aug_pitch_range, FLAGS.aug_pitch_range)
  audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
  if _capture:
    _log_aug_once('pitch', {
        'n_steps': round(float(n_steps), 4),
        **_before_after_summary(_before, audio),
        **_aug_debug_context(debug_info),
    }, key=_debug_key)
  return audio


def _aug_resample(audio, sr, debug_info=None):
  """Downsample to low sr then upsample back, losing high-frequency info.
  Simulates low-quality recording equipment. (Bengali.AI 1st: 16k→8k→16k)"""
  _debug_key = 'aug_once:resample'
  _capture = _should_capture_aug_once(_debug_key)
  _before = audio.copy() if _capture else None
  import librosa
  target_sr = FLAGS.aug_resample_sr
  if target_sr >= sr:
    return audio
  audio_low = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
  audio_back = librosa.resample(audio_low, orig_sr=target_sr, target_sr=sr)
  # match original length (resample can differ by ±1 sample)
  if len(audio_back) > len(audio):
    audio_back = audio_back[:len(audio)]
  elif len(audio_back) < len(audio):
    audio_back = np.pad(audio_back, (0, len(audio) - len(audio_back)))
  if _capture:
    _log_aug_once('resample', {
        'orig_sr': int(sr),
        'target_sr': int(target_sr),
        **_before_after_summary(_before, audio_back),
        **_aug_debug_context(debug_info),
    }, key=_debug_key)
  return audio_back


# ---- Classroom noise augmentation (RealClass) ----

_noise_files_cache = None  # module-level cache for noise file list
_noise_audio_cache = {}    # LRU-style cache for loaded noise waveforms

def _get_noise_files():
  """Get list of noise .flac files from noise_dir (cached)."""
  global _noise_files_cache
  if _noise_files_cache is None:
    import glob
    noise_dir = FLAGS.noise_dir
    if not noise_dir:
      return []
    patterns = [f'{noise_dir}/**/*.flac', f'{noise_dir}/**/*.wav', f'{noise_dir}/**/*.mp3']
    files = []
    for p in patterns:
      files.extend(glob.glob(p, recursive=True))
    if not files:
      # try flat directory
      files = glob.glob(f'{noise_dir}/*.flac')
    _noise_files_cache = sorted(files)
    if _noise_files_cache:
      ic(f'Loaded {len(_noise_files_cache)} noise files from {noise_dir}')
    else:
      ic(f'WARNING: No noise files found in {noise_dir}')
  return _noise_files_cache


def _load_noise_audio(noise_path, sr):
  """Load a noise file with caching (up to 200 files)."""
  cache_key = (noise_path, sr)
  if cache_key not in _noise_audio_cache:
    if len(_noise_audio_cache) > 200:
      # evict a random entry to bound memory
      evict_key = random.choice(list(_noise_audio_cache.keys()))
      del _noise_audio_cache[evict_key]
    import soundfile as sf
    audio, file_sr = sf.read(noise_path, dtype='float32')
    if audio.ndim > 1:
      audio = audio.mean(axis=1)  # mono
    if file_sr != sr:
      import librosa
      audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    _noise_audio_cache[cache_key] = audio
  return _noise_audio_cache[cache_key]


def _aug_classroom_noise(audio, sr, debug_info=None):
  """Mix real classroom background noise into speech at random SNR.
  
  Randomly picks a noise file, extracts a segment matching the speech length,
  and mixes at a random SNR within the configured range.
  
  This is more effective than Gaussian noise because it contains
  realistic classroom sounds (other children, furniture, ambient reverb).
  """
  _debug_key = 'aug_once:classroom_noise'
  _capture = _should_capture_aug_once(_debug_key)
  _before = audio.copy() if _capture else None
  noise_files = _get_noise_files()
  if not noise_files:
    return audio
  
  # pick a random noise file
  noise_path = random.choice(noise_files)
  noise = _load_noise_audio(noise_path, sr)
  
  speech_len = len(audio)
  noise_len = len(noise)
  
  if noise_len == 0:
    return audio
  
  # extract a segment of noise matching speech length
  if noise_len >= speech_len:
    # random crop from noise
    start = random.randint(0, noise_len - speech_len)
    noise_segment = noise[start:start + speech_len]
  else:
    # loop/tile noise to match speech length
    repeats = (speech_len // noise_len) + 1
    noise_tiled = np.tile(noise, repeats)
    start = random.randint(0, len(noise_tiled) - speech_len)
    noise_segment = noise_tiled[start:start + speech_len]
  
  # compute SNR-based mixing
  snr_db = random.uniform(FLAGS.aug_classroom_snr_min, FLAGS.aug_classroom_snr_max)
  rms_signal = np.sqrt(np.mean(audio ** 2)) + 1e-9
  rms_noise = np.sqrt(np.mean(noise_segment ** 2)) + 1e-9
  
  # scale noise to target SNR: SNR = 20*log10(rms_signal / (gain * rms_noise))
  # => gain = rms_signal / (rms_noise * 10^(snr_db/20))
  target_noise_rms = rms_signal / (10 ** (snr_db / 20))
  gain = target_noise_rms / rms_noise
  
  mixed = audio + gain * noise_segment
  
  # prevent clipping
  max_val = np.max(np.abs(mixed))
  if max_val > 1.0:
    mixed = mixed / max_val

  mixed = mixed.astype(audio.dtype)
  if _capture:
    _log_aug_once('classroom_noise', {
        'snr_db': round(float(snr_db), 3),
        'noise_file': os.path.basename(noise_path),
        **_before_after_summary(_before, mixed),
        **_aug_debug_context(debug_info),
    }, key=_debug_key)
  return mixed


def _aug_vtln_waveform(audio, sr, debug_info=None):
  """VTLN via STFT frequency-axis warping on raw waveform.
  Simulates different vocal tract lengths by warping the frequency axis.
  Works for ANY backbone (NeMo, wav2vec2, Whisper, etc.).
  
  alpha > 1: shorter vocal tract (higher formants, children-like)
  alpha < 1: longer vocal tract (lower formants, adult-like)
  
  audio: 1-D numpy float32 waveform
  sr: sample rate
  Returns: warped audio of same length
  """
  _debug_key = 'aug_once:vtln_waveform'
  _capture = _should_capture_aug_once(_debug_key)
  _before = audio.copy() if _capture else None
  import librosa
  from scipy.ndimage import zoom

  alpha = random.uniform(FLAGS.aug_vtln_alpha_min, FLAGS.aug_vtln_alpha_max)
  if abs(alpha - 1.0) < 1e-3:
    return audio

  n_fft = 512
  hop_length = 128

  # STFT
  stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
  magnitude = np.abs(stft)
  phase = np.angle(stft)
  n_freq = stft.shape[0]  # n_fft//2 + 1

  # Warp frequency axis by 1/alpha (bilinear interpolation)
  warped_mag = zoom(magnitude, (1.0 / alpha, 1.0), order=1)
  warped_phase = zoom(phase, (1.0 / alpha, 1.0), order=1)

  # Crop or zero-pad back to original frequency bins
  if warped_mag.shape[0] >= n_freq:
    warped_mag = warped_mag[:n_freq, :]
    warped_phase = warped_phase[:n_freq, :]
  else:
    pad_rows = n_freq - warped_mag.shape[0]
    warped_mag = np.concatenate(
      [warped_mag, np.zeros((pad_rows, warped_mag.shape[1]), dtype=warped_mag.dtype)], axis=0)
    warped_phase = np.concatenate(
      [warped_phase, np.zeros((pad_rows, warped_phase.shape[1]), dtype=warped_phase.dtype)], axis=0)

  # ISTFT back to waveform
  warped_stft = warped_mag * np.exp(1j * warped_phase)
  audio_warped = librosa.istft(warped_stft, hop_length=hop_length, length=len(audio))
  audio_warped = audio_warped.astype(audio.dtype)
  if _capture:
    _log_aug_once('vtln_waveform', {
        'alpha': round(float(alpha), 4),
        **_before_after_summary(_before, audio_warped),
        **_aug_debug_context(debug_info),
    }, key=_debug_key)
  return audio_warped


def _aug_vtln_mel(mel, debug_info=None):
  """VTLN on mel spectrogram: warp frequency axis via scipy.ndimage.zoom.
  Faster than waveform-level VTLN (no STFT/ISTFT round-trip).
  Only applicable for Whisper (mel already extracted).
  mel: (n_mels, T) numpy array
  Returns: warped mel of same shape
  """
  _debug_key = 'aug_once:vtln_mel'
  _capture = _should_capture_aug_once(_debug_key)
  _before = mel.copy() if _capture else None
  from scipy.ndimage import zoom

  alpha = random.uniform(FLAGS.aug_vtln_alpha_min, FLAGS.aug_vtln_alpha_max)
  if abs(alpha - 1.0) < 1e-3:
    return mel

  n_mels, T = mel.shape
  # zoom frequency axis by 1/alpha, keep time axis unchanged
  warped = zoom(mel, (1.0 / alpha, 1.0), order=1)

  if warped.shape[0] >= n_mels:
    warped = warped[:n_mels, :]
  else:
    pad = np.zeros((n_mels - warped.shape[0], T), dtype=mel.dtype)
    warped = np.concatenate([warped, pad], axis=0)
  if _capture:
    _log_aug_once('vtln_mel', {
        'alpha': round(float(alpha), 4),
        **_before_after_summary(_before, warped),
        **_aug_debug_context(debug_info),
    }, key=_debug_key)
  return warped


def _aug_specaugment(mel, debug_info=None):
  """SpecAugment: frequency and time masking on mel spectrogram.
  mel: numpy array (n_mels, T) — Whisper processor output shape."""
  _debug_key = 'aug_once:spec'
  _capture = _should_capture_aug_once(_debug_key)
  _before = mel.copy() if _capture else None
  mel = mel.copy()
  n_mels, T = mel.shape
  freq_masks = []
  time_masks = []

  # Frequency masking
  for _ in range(FLAGS.aug_freq_num):
    f = random.randint(0, min(FLAGS.aug_freq_mask, n_mels - 1))
    f0 = random.randint(0, n_mels - f)
    mel[f0:f0 + f, :] = 0.0
    freq_masks.append((int(f0), int(f)))

  # Time masking
  for _ in range(FLAGS.aug_time_num):
    t = random.randint(0, min(FLAGS.aug_time_mask, T - 1))
    t0 = random.randint(0, T - t)
    mel[:, t0:t0 + t] = 0.0
    time_masks.append((int(t0), int(t)))

  if _capture:
    _log_aug_once('spec', {
        'mel_shape': [int(n_mels), int(T)],
        'freq_masks': freq_masks,
        'time_masks': time_masks,
        **_before_after_summary(_before, mel),
        **_aug_debug_context(debug_info),
    }, key=_debug_key)

  return mel


def augment_audio(audio, sr, skip_speed=False, debug_info=None):
  """Apply waveform-level augmentations (train only).
  
  Args:
    skip_speed: If True, skip speed perturbation (already applied per-segment in aug_mix).
  """
  if FLAGS.aug_speed and not skip_speed and random.random() < FLAGS.aug_speed_prob:
    if not FLAGS.aug_speed_short_only or len(audio) / sr <= FLAGS.aug_speed_short_dur:
      audio = _aug_speed(audio, sr, debug_info=debug_info)
  if FLAGS.aug_volume and random.random() < FLAGS.aug_volume_prob:
    audio = _aug_volume(audio, debug_info=debug_info)
  # Classroom noise (RealClass) — applied before Gaussian noise so both can stack
  if getattr(FLAGS, 'aug_classroom_noise', False) and random.random() < FLAGS.aug_classroom_noise_prob:
    audio = _aug_classroom_noise(audio, sr, debug_info=debug_info)
  # Gaussian noise (skip if classroom_noise_only is set and classroom noise was applied)
  if FLAGS.aug_noise and random.random() < FLAGS.aug_noise_prob:
    if not getattr(FLAGS, 'aug_classroom_noise_only', False):
      audio = _aug_noise(audio, debug_info=debug_info)
  if FLAGS.aug_pitch and random.random() < FLAGS.aug_pitch_prob:
    audio = _aug_pitch(audio, sr, debug_info=debug_info)
  # Resample degradation (16k→8k→16k): lose high-freq info
  if getattr(FLAGS, 'aug_resample', False) and random.random() < FLAGS.aug_resample_prob:
    audio = _aug_resample(audio, sr, debug_info=debug_info)
  # VTLN: frequency-axis warping (waveform-level, works for all backbones)
  if getattr(FLAGS, 'aug_vtln', False) and random.random() < FLAGS.aug_vtln_prob:
    audio = _aug_vtln_waveform(audio, sr, debug_info=debug_info)
  return audio


def _aug_temporal_mask(mel, debug_info=None):
  """Random temporal masking: zero out a random proportion of time steps.
  Each time frame is independently masked with probability p.
  Inspired by ASLFR temporal_mask (pointwise dropout on time axis).
  mel: (n_mels, T)"""
  _debug_key = 'aug_once:temporal_mask'
  _capture = _should_capture_aug_once(_debug_key)
  _before = mel.copy() if _capture else None
  mel = mel.copy()
  n_mels, T = mel.shape
  mask_range = getattr(FLAGS, 'aug_temporal_mask_range', [])
  if mask_range and len(mask_range) >= 2:
    lo, hi = float(mask_range[0]), float(mask_range[1])
    prob = random.uniform(lo, hi)
  else:
    prob = FLAGS.aug_temporal_mask_prob
  # mask shape (T,) — same mask for all freq bins at each time step
  mask = np.random.uniform(size=T) > prob
  mel *= mask[np.newaxis, :]  # broadcast (1, T) over (n_mels, T)
  if _capture:
    _log_aug_once('temporal_mask', {
        'prob': round(float(prob), 4),
        **_before_after_summary(_before, mel),
        **_aug_debug_context(debug_info),
    }, key=_debug_key)
  return mel


def _aug_spatio_mask(mel, debug_info=None):
  """Random spatial/frequency masking: zero out a random proportion of freq channels.
  Each frequency bin is independently masked (same across all time steps).
  Inspired by ASLFR spatio_mask.
  mel: (n_mels, T)"""
  _debug_key = 'aug_once:spatio_mask'
  _capture = _should_capture_aug_once(_debug_key)
  _before = mel.copy() if _capture else None
  mel = mel.copy()
  n_mels, T = mel.shape
  prob = FLAGS.aug_spatio_mask_prob
  # mask shape (n_mels,) — same mask for all time steps
  mask = np.random.uniform(size=n_mels) > prob
  mel *= mask[:, np.newaxis]  # broadcast (n_mels, 1) over (n_mels, T)
  if _capture:
    _log_aug_once('spatio_mask', {
        'prob': round(float(prob), 4),
        **_before_after_summary(_before, mel),
        **_aug_debug_context(debug_info),
    }, key=_debug_key)
  return mel


def _aug_st_mask(mel, debug_info=None):
  """Random spatio-temporal 2D masking: each (freq, time) cell is independently
  masked (zeroed) with probability p. Equivalent to 2D dropout on the mel.
  mel: (n_mels, T)"""
  _debug_key = 'aug_once:st_mask'
  _capture = _should_capture_aug_once(_debug_key)
  _before = mel.copy() if _capture else None
  mel = mel.copy()
  n_mels, T = mel.shape
  mask_range = getattr(FLAGS, 'aug_st_mask_range', [])
  if mask_range and len(mask_range) >= 2:
    lo, hi = float(mask_range[0]), float(mask_range[1])
    prob = random.uniform(lo, hi)
  else:
    prob = getattr(FLAGS, 'aug_st_mask_prob', 0.15)
  mask = np.random.uniform(size=(n_mels, T)) > prob
  mel *= mask
  if _capture:
    _log_aug_once('st_mask', {
        'prob': round(float(prob), 4),
        **_before_after_summary(_before, mel),
        **_aug_debug_context(debug_info),
    }, key=_debug_key)
  return mel


def _aug_cutmix(mel, debug_info=None):
  """CutMix-style mel augmentation without label mixing.
  Copies a random patch from one region of the same mel and pastes it to
  another region (same patch size), preserving sequence labels.
  mel: (n_mels, T)"""
  _debug_key = 'aug_once:cutmix'
  _capture = _should_capture_aug_once(_debug_key)
  _before = mel.copy() if _capture else None
  mel = mel.copy()
  n_mels, T = mel.shape

  if n_mels < 2 or T < 2:
    return mel

  if random.random() >= getattr(FLAGS, 'aug_cutmix_prob', 0.3):
    return mel

  def _parse_ratio_range(v, default_lo, default_hi):
    try:
      if v and len(v) >= 2:
        lo, hi = float(v[0]), float(v[1])
      else:
        lo, hi = default_lo, default_hi
    except Exception:
      lo, hi = default_lo, default_hi
    lo = max(0.01, min(lo, 0.95))
    hi = max(lo, min(hi, 0.95))
    return lo, hi

  t_lo, t_hi = _parse_ratio_range(getattr(FLAGS, 'aug_cutmix_time_ratio', []), 0.05, 0.2)
  f_lo, f_hi = _parse_ratio_range(getattr(FLAGS, 'aug_cutmix_freq_ratio', []), 0.05, 0.2)
  num_ops = max(1, int(getattr(FLAGS, 'aug_cutmix_num', 1)))

  for _ in range(num_ops):
    t_ratio = random.uniform(t_lo, t_hi)
    f_ratio = random.uniform(f_lo, f_hi)

    patch_t = max(1, min(T - 1, int(T * t_ratio)))
    patch_f = max(1, min(n_mels - 1, int(n_mels * f_ratio)))

    src_t0 = random.randint(0, T - patch_t)
    src_f0 = random.randint(0, n_mels - patch_f)
    dst_t0 = random.randint(0, T - patch_t)
    dst_f0 = random.randint(0, n_mels - patch_f)

    patch = mel[src_f0:src_f0 + patch_f, src_t0:src_t0 + patch_t].copy()
    mel[dst_f0:dst_f0 + patch_f, dst_t0:dst_t0 + patch_t] = patch

  if _capture:
    _log_aug_once('cutmix', {
        'mel_shape': [int(n_mels), int(T)],
        'num_ops': int(num_ops),
        'time_ratio_range': list(getattr(FLAGS, 'aug_cutmix_time_ratio', [])),
        'freq_ratio_range': list(getattr(FLAGS, 'aug_cutmix_freq_ratio', [])),
        **_before_after_summary(_before, mel),
        **_aug_debug_context(debug_info),
    }, key=_debug_key)

  return mel


def augment_mel(mel, debug_info=None):
  """Apply spectrogram-level augmentations (Whisper, train only)."""
  # VTLN on mel (faster path for Whisper — skip if already applied at waveform level)
  if getattr(FLAGS, 'aug_vtln', False) and not is_waveform_backbone():
    if random.random() < FLAGS.aug_vtln_prob:
      mel = _aug_vtln_mel(mel, debug_info=debug_info)
  if FLAGS.aug_spec:
    mel = _aug_specaugment(mel, debug_info=debug_info)
  if getattr(FLAGS, 'aug_cutmix', False):
    mel = _aug_cutmix(mel, debug_info=debug_info)
  if getattr(FLAGS, 'aug_temporal_mask', False):
    mel = _aug_temporal_mask(mel, debug_info=debug_info)
  if getattr(FLAGS, 'aug_spatio_mask', False):
    mel = _aug_spatio_mask(mel, debug_info=debug_info)
  if getattr(FLAGS, 'aug_st_mask', False):
    mel = _aug_st_mask(mel, debug_info=debug_info)
  return mel


# ===================== Cross-sample CutMix (Priority 1) =====================

def _xcutmix_mel(mel_a, label_a, mel_b, label_b):
  """Cross-sample CutMix: splice two mel spectrograms along the time axis.
  
  Cuts sample A at time ratio `lam` and sample B at ratio `(1-lam)`,
  concatenates A-left + B-right, and concatenates labels proportionally.
  
  Args:
    mel_a:   (n_mels, T_a) numpy array
    label_a: list of token ids
    mel_b:   (n_mels, T_b) numpy array
    label_b: list of token ids
  Returns:
    (new_mel, new_labels) — spliced mel and concatenated label list
  """
  ratio_range = getattr(FLAGS, 'aug_xcutmix_ratio', ['0.3', '0.7'])
  try:
    lo, hi = float(ratio_range[0]), float(ratio_range[1])
  except Exception:
    lo, hi = 0.3, 0.7
  lam = random.uniform(lo, hi)

  n_mels, T_a = mel_a.shape
  _, T_b = mel_b.shape

  cut_a = max(1, int(T_a * lam))
  cut_b = max(1, int(T_b * (1 - lam)))

  # splice mel: first part of A + last part of B
  new_mel = np.concatenate([mel_a[:, :cut_a], mel_b[:, -cut_b:]], axis=1)

  # splice labels proportionally
  if len(label_a) > 0 and len(label_b) > 0:
    label_cut_a = max(1, int(len(label_a) * lam))
    label_cut_b = max(1, int(len(label_b) * (1 - lam)))
    new_labels = label_a[:label_cut_a] + label_b[-label_cut_b:]
  elif len(label_a) > 0:
    new_labels = label_a
  elif len(label_b) > 0:
    new_labels = label_b
  else:
    new_labels = []

  return new_mel, new_labels


# ===================== Alignment-aware SpliceMix (Priority 2) =====================

_alignment_cache = {}  # module-level cache for loaded alignment data

def _load_alignments():
  """Load pre-computed alignments from pickle file (cached)."""
  alignment_file = FLAGS.alignment_file
  if not alignment_file:
    return None
  if alignment_file not in _alignment_cache:
    import pickle
    with open(alignment_file, 'rb') as f:
      _alignment_cache[alignment_file] = pickle.load(f)
    ic(f'Loaded alignments from {alignment_file}', len(_alignment_cache[alignment_file]))
  return _alignment_cache[alignment_file]


def _splicemix_mel(mel_a, label_a, id_a, mel_b, label_b, id_b):
  """SpliceMix: alignment-aware cross-sample CutMix.
  
  Uses pre-computed CTC alignments to find phoneme boundaries, then
  splices at a boundary in sample A and a boundary in sample B to produce
  a clean concatenation with exact label correspondence.
  
  Alignment format per utterance:
    list of (phoneme_id, start_frame, end_frame)
  
  Falls back to simple xcutmix if alignment is not available for either sample.
  
  Args:
    mel_a, label_a, id_a: mel, labels, utterance id for sample A
    mel_b, label_b, id_b: mel, labels, utterance id for sample B
  Returns:
    (new_mel, new_labels)
  """
  alignments = _load_alignments()
  if alignments is None:
    return _xcutmix_mel(mel_a, label_a, mel_b, label_b)

  align_a = alignments.get(id_a)
  align_b = alignments.get(id_b)

  # fallback if alignment missing for either sample
  if not align_a or not align_b:
    return _xcutmix_mel(mel_a, label_a, mel_b, label_b)

  # align_a/b: list of (phoneme_id, start_frame, end_frame)
  n_phonemes_a = len(align_a)
  n_phonemes_b = len(align_b)

  if n_phonemes_a < 2 or n_phonemes_b < 2:
    return _xcutmix_mel(mel_a, label_a, mel_b, label_b)

  # pick a random cut point in A (after phoneme k) and in B (before phoneme j)
  k = random.randint(1, n_phonemes_a - 1)  # keep at least 1 phoneme from A
  j = random.randint(0, n_phonemes_b - 2)  # keep at least 1 phoneme from B

  # frame boundary: end of k-th phoneme in A, start of j-th phoneme in B
  cut_frame_a = align_a[k - 1][2]  # end_frame of (k-1)-th phoneme (0-indexed)
  cut_frame_b = align_b[j][1]      # start_frame of j-th phoneme

  # clamp to valid range
  _, T_a = mel_a.shape
  _, T_b = mel_b.shape
  cut_frame_a = max(1, min(cut_frame_a, T_a - 1))
  cut_frame_b = max(0, min(cut_frame_b, T_b - 1))

  # splice mel
  part_a = mel_a[:, :cut_frame_a]
  part_b = mel_b[:, cut_frame_b:]

  # optional crossfade at boundary
  if getattr(FLAGS, 'aug_splicemix_crossfade', True):
    cf_frames = min(
      getattr(FLAGS, 'aug_splicemix_crossfade_frames', 3),
      part_a.shape[1],
      part_b.shape[1],
    )
    if cf_frames > 0:
      fade_out = np.linspace(1.0, 0.0, cf_frames).astype(np.float32)
      fade_in = np.linspace(0.0, 1.0, cf_frames).astype(np.float32)
      part_a[:, -cf_frames:] *= fade_out[np.newaxis, :]
      part_b[:, :cf_frames] *= fade_in[np.newaxis, :]

  new_mel = np.concatenate([part_a, part_b], axis=1)

  # splice labels: first k phonemes from A + phonemes from j onwards from B
  # The alignment phoneme ids should correspond to the label token ids.
  # We reconstruct labels from alignment phoneme_ids for precision.
  label_part_a = [seg[0] for seg in align_a[:k]]
  label_part_b = [seg[0] for seg in align_b[j:]]
  new_labels = label_part_a + label_part_b

  return new_mel, new_labels


# ===================== EXT data loader for eval_add_ext =====================

_ext_eval_df_cache = None

def _load_ext_for_eval():
  """Load EXT data independently for eval when use_ext=False.
  
  Loads ext JSONL, assigns folds (same logic as preprocess.set_folds),
  prepares audio paths, and returns all valid ext rows.
  Result is cached so it's only loaded once per process.
  """
  global _ext_eval_df_cache
  if _ext_eval_df_cache is not None:
    return _ext_eval_df_cache
  
  import json, os
  ext_root = getattr(FLAGS, 'eval_ext_root', '') or FLAGS.ext_root
  if not ext_root or not os.path.isdir(ext_root):
    ic('eval_add_ext: ext_root not found', ext_root)
    _ext_eval_df_cache = pd.DataFrame()
    return _ext_eval_df_cache
  
  # Find the ext training file (same logic as load_df)
  train_files = [FLAGS.train_file, 'train.jsonl', 'train.csv']
  ext_file = None
  for fname in train_files:
    candidate = os.path.join(ext_root, fname)
    if os.path.exists(candidate):
      ext_file = candidate
      break
  if ext_file is None:
    ic('eval_add_ext: no ext JSONL/CSV found', ext_root)
    _ext_eval_df_cache = pd.DataFrame()
    return _ext_eval_df_cache
  
  if ext_file.endswith('.csv'):
    ext_df = pd.read_csv(ext_file)
  else:
    rows = []
    with open(ext_file) as f:
      for line in f:
        rows.append(json.loads(line))
    ext_df = pd.DataFrame(rows)
  
  if 'id' not in ext_df.columns:
    ext_df['id'] = ext_df['utterance_id']
  ext_df['source'] = 'ext'
  
  # Label column
  if FLAGS.label_column in ext_df.columns:
    ext_df['label_text'] = ext_df[FLAGS.label_column].fillna('')
  elif FLAGS.label_column_fallback and FLAGS.label_column_fallback in ext_df.columns:
    ext_df['label_text'] = ext_df[FLAGS.label_column_fallback].fillna('')
  else:
    ext_df['label_text'] = ''
  
  # Ensure cross-label columns exist
  if 'ipa_label' not in ext_df.columns:
    ext_df['ipa_label'] = ''
  if 'word_label' not in ext_df.columns:
    ext_df['word_label'] = ''
  
  # Assign folds (same as preprocess.set_folds for ext-only)
  group_key = FLAGS.fold_group_key if FLAGS.fold_group_key else None
  stratify_key = FLAGS.fold_stratify_key if FLAGS.fold_stratify_key else None
  gz.set_fold(ext_df, FLAGS.folds, group_key=group_key,
              stratify_key=stratify_key, seed=FLAGS.fold_seed,
              sgkf_compat=getattr(FLAGS, 'sgkf_compat', '1.6.1') or None)
  
  # Resolve audio paths
  if 'audio_path' in ext_df.columns:
    ext_df['audio_file'] = ext_df['audio_path'].apply(
      lambda x: f'{ext_root}/{x}' if not os.path.isabs(str(x)) else str(x)
    )
  
  # Filter bad/missing audio
  if 'audio_file' in ext_df.columns:
    valid_mask = ext_df['audio_file'].apply(
      lambda p: os.path.isfile(str(p)) and os.path.getsize(str(p)) > 1024 * 10
    )
    ext_df = ext_df[valid_mask].reset_index(drop=True)
  
  if 'audio_duration_sec' not in ext_df.columns:
    ext_df['audio_duration_sec'] = 0.0
  
  ic('_load_ext_for_eval', len(ext_df))
  _ext_eval_df_cache = ext_df
  return _ext_eval_df_cache


_VALID_MIX_STRATEGIES = {'random', 'same_session', 'same_child', 'same_source', 'cross_source', 'same_age'}

# Cache parsed strategy specs to avoid re-parsing every sample
_mix_strategy_cache = {}

def _parse_mix_strategy(strategy_str):
  """Parse strategy string into list of (strategy_name, probability).
  
  Formats:
    'cross_source'                          → [('cross_source', 1.0)]
    'cross_source,same_child'               → [('cross_source', 0.5), ('same_child', 0.5)]
    'cross_source:0.6,same_child:0.4'       → [('cross_source', 0.6), ('same_child', 0.4)]
  """
  if strategy_str in _mix_strategy_cache:
    return _mix_strategy_cache[strategy_str]
  
  parts = [p.strip() for p in strategy_str.split(',')]
  strategies = []
  for part in parts:
    if ':' in part:
      name, weight = part.split(':', 1)
      name = name.strip()
      weight = float(weight.strip())
    else:
      name = part.strip()
      weight = None  # will be filled with equal prob
    assert name in _VALID_MIX_STRATEGIES, \
      f'Invalid aug_mix strategy "{name}". Valid: {sorted(_VALID_MIX_STRATEGIES)}'
    strategies.append((name, weight))
  
  # Fill equal weights for entries without explicit weight
  has_weights = [w is not None for _, w in strategies]
  if all(has_weights):
    total = sum(w for _, w in strategies)
    assert abs(total - 1.0) < 0.01, \
      f'aug_mix_strategy weights must sum to 1.0, got {total}: {strategy_str}'
  elif not any(has_weights):
    # All equal
    w = 1.0 / len(strategies)
    strategies = [(name, w) for name, _ in strategies]
  else:
    raise ValueError(
      f'aug_mix_strategy: either all entries have weights or none. '
      f'Got mixed: {strategy_str}')
  
  result = strategies
  _mix_strategy_cache[strategy_str] = result
  return result


def _sample_mix_strategy(strategy_str):
  """Sample a strategy name from the parsed strategy spec."""
  strategies = _parse_mix_strategy(strategy_str)
  if len(strategies) == 1:
    return strategies[0][0]
  names = [s[0] for s in strategies]
  weights = [s[1] for s in strategies]
  return random.choices(names, weights=weights, k=1)[0]


class Dataset(TorchDataset):
  """ASR dataset: loads audio and tokenises labels on-the-fly."""

  def __init__(self, df, mode='eval'):
    self.mode = mode
    self.processor = get_processor(FLAGS.backbone)
    self.tokenizer = get_tokenizer(FLAGS.backbone)

    # ---- Helper: identify word-only rows (label_text='') ----
    # These are training-only auxiliary data added by use_word_only_dd/ext.
    # They must be excluded from eval and from the base train/ext split so
    # that eval/train partition is identical to v13 baseline.
    _has_label_col = 'label_text' in df.columns
    _has_source_col = 'source' in df.columns
    if _has_label_col:
      _is_word_only = df['label_text'].fillna('').str.strip() == ''
    else:
      _is_word_only = pd.Series(False, index=df.index)
    # Base df without word-only rows (same rows as v13)
    df_base = df[~_is_word_only]
    # Word-only rows to append to train later
    df_word_only = df[_is_word_only]

    if mode in ['eval', 'valid']:
      if getattr(FLAGS, 'official_data', False) and mode == 'valid':
        # official_data: valid set = same as eval set (the 1840 IDs excluded from train)
        # Avoids data leakage since train contains all data minus eval IDs.
        # The actual eval set is built below and downsampled by eval_samples;
        # valid must use the same subset to avoid overlap with train.
        # We replicate the eval construction here (eval_add_ext + eval_samples).
        _id_col = 'utterance_id' if 'utterance_id' in df_base.columns else 'id'
        if _has_source_col:
          dd_eval = df_base[(df_base.fold == FLAGS.fold) & (df_base.source == 'dd')]
        else:
          dd_eval = df_base[df_base.fold == FLAGS.fold]
        valid_df = dd_eval
        if FLAGS.use_ext and _has_source_col:
          base_ext = df_base[df_base.source == 'ext']
          n_dd = len(dd_eval)
          if len(base_ext) >= n_dd > 0:
            ext_sample = base_ext.sample(n=n_dd, random_state=42)
          elif len(base_ext) > 0:
            ext_sample = base_ext
          else:
            ext_sample = pd.DataFrame()
          if len(ext_sample) > 0:
            valid_df = pd.concat([valid_df, ext_sample], ignore_index=True)
        eval_samples = FLAGS.eval_samples if FLAGS.eval_samples else 0
        if eval_samples > 0 and eval_samples < len(valid_df):
          valid_df = valid_df.sample(eval_samples, random_state=42)
        self.df = valid_df.reset_index(drop=True)
        ic('official_data valid: same as eval IDs', len(self.df))
      elif getattr(FLAGS, 'eval_add_ext', False):
        # eval_add_ext: base eval = DD only from eval fold; ext added via sampling below
        # This supersedes eval_ext_only (contradictory semantics)
        if _has_source_col:
          dd_eval = df_base[(df_base.fold == FLAGS.fold) & (df_base.source == 'dd')]
          self.df = dd_eval.reset_index(drop=True)
        else:
          self.df = df_base[df_base.fold == FLAGS.fold].reset_index(drop=True)
        n_dd = len(self.df)
        _eval_ext_root = getattr(FLAGS, 'eval_ext_root', '') or ''
        _ext_root = getattr(FLAGS, 'ext_root', '')
        _eval_uses_separate_ext = (
          _eval_ext_root and _eval_ext_root != _ext_root
          and os.path.normpath(_eval_ext_root) != os.path.normpath(_ext_root)
        )
        if FLAGS.use_ext and _has_source_col and not _eval_uses_separate_ext:
          # ext from base df (word-only excluded) — identical pool to v13
          ext_eval = df_base[df_base.source == 'ext']
        else:
          # ext not loaded, or eval_ext_root differs → load real ext independently
          ext_eval = _load_ext_for_eval()
          # Also exclude empty-label rows from independently loaded ext
          if 'label_text' in ext_eval.columns:
            ext_eval = ext_eval[ext_eval['label_text'].fillna('').str.strip() != '']
        if len(ext_eval) > 0:
          if getattr(FLAGS, 'ext_eval_group', False) and 'fold' in ext_eval.columns:
            # ext_eval_group: use fold-based split (child_id grouped) to
            # prevent train/eval child leakage.
            ext_fold_eval = ext_eval[ext_eval.fold == FLAGS.fold]
            if getattr(FLAGS, 'eval_ext_full', False):
              # eval_ext_full: use ALL ext in this fold (no downsampling)
              ext_sample = ext_fold_eval if len(ext_fold_eval) > 0 else pd.DataFrame()
              ic('eval_add_ext (ext_eval_group, FULL)', n_dd,
                 f'ext_fold_pool={len(ext_fold_eval)}')
            elif len(ext_fold_eval) >= n_dd:
              ext_sample = ext_fold_eval.sample(n=n_dd, random_state=42)
            elif len(ext_fold_eval) > 0:
              ext_sample = ext_fold_eval
            else:
              ext_sample = pd.DataFrame()
            if not getattr(FLAGS, 'eval_ext_full', False) and len(ext_sample) > 0:
              ic('eval_add_ext (ext_eval_group)', n_dd,
                 f'ext_fold_pool={len(ext_fold_eval)}',
                 f'ext_sample={len(ext_sample)}')
          else:
            if len(ext_eval) >= n_dd:
              ext_sample = ext_eval.sample(n=n_dd, random_state=42)
            else:
              ext_sample = ext_eval  # fewer ext than DD, use all
          if len(ext_sample) > 0:
            self.df = pd.concat([self.df, ext_sample], ignore_index=True)
            _pool = len(ext_fold_eval) if (getattr(FLAGS, 'ext_eval_group', False) and 'fold' in ext_eval.columns) else len(ext_eval)
            ic('eval_add_ext', n_dd, f'ext_pool={_pool}', len(ext_sample), len(self.df))
      else:
        self.df = df_base[df_base.fold == FLAGS.fold].reset_index(drop=True)
        # eval_ext_only: evaluate only on ext data
        if getattr(FLAGS, 'eval_ext_only', False) and 'source' in self.df.columns:
          self.df = self.df[self.df.source == 'ext'].reset_index(drop=True)
    elif mode == 'train':
      if getattr(FLAGS, 'official_data', False) and not FLAGS.online:
        # official_data: ALL data minus eval utterance IDs (matches standalone official-baseline.py)
        # Replicate the FULL eval construction (eval_add_ext + eval_samples downsampling)
        # to get the exact same eval IDs, then exclude only those from training.
        _id_col = 'utterance_id' if 'utterance_id' in df_base.columns else 'id'
        # Step 1: build eval set same as eval path (eval_add_ext)
        if _has_source_col:
          dd_eval = df_base[(df_base.fold == FLAGS.fold) & (df_base.source == 'dd')]
        else:
          dd_eval = df_base[df_base.fold == FLAGS.fold]
        eval_df = dd_eval
        if FLAGS.use_ext and _has_source_col:
          base_ext = df_base[df_base.source == 'ext']
          n_dd = len(dd_eval)
          if len(base_ext) >= n_dd > 0:
            ext_sample = base_ext.sample(n=n_dd, random_state=42)
          elif len(base_ext) > 0:
            ext_sample = base_ext
          else:
            ext_sample = pd.DataFrame()
          if len(ext_sample) > 0:
            eval_df = pd.concat([eval_df, ext_sample], ignore_index=True)
        # Step 2: apply eval_samples downsampling (same as post-split sampling)
        eval_samples = FLAGS.eval_samples if FLAGS.eval_samples else 0
        if eval_samples > 0 and eval_samples < len(eval_df):
          eval_df = eval_df.sample(eval_samples, random_state=42)
        eval_ids = set(eval_df[_id_col].tolist())
        self.df = df_base[~df_base[_id_col].isin(eval_ids)].reset_index(drop=True)
        if len(df_word_only) > 0:
          self.df = pd.concat([self.df, df_word_only], ignore_index=True)
        ic('official_data train: ALL data minus eval IDs', len(eval_ids), len(self.df))
      elif getattr(FLAGS, 'eval_add_ext', False) and FLAGS.use_ext and _has_source_col and not FLAGS.online:
        # Build base train from df_base (identical to v13): DD non-eval + ext minus eval sample
        _eval_ext_root = getattr(FLAGS, 'eval_ext_root', '') or ''
        train_dd = df_base[(df_base.fold != FLAGS.fold) & (df_base.source == 'dd')]
        base_ext = df_base[df_base.source == 'ext']
        # When eval_ext_root differs from ext_root, eval ext comes from a
        # separate data source → no need to exclude anything from training ext.
        _ext_root = getattr(FLAGS, 'ext_root', '')
        _eval_uses_separate_ext = (
          _eval_ext_root and _eval_ext_root != _ext_root
          and os.path.normpath(_eval_ext_root) != os.path.normpath(_ext_root)
        )
        if _eval_uses_separate_ext:
          train_ext = base_ext
          # Pseudo-IPA / cross-label ext may share utterance_ids with the
          # separate eval ext source.  Replicate eval-set construction to
          # compute the exact eval IDs, then exclude them from training.
          _id_col = 'utterance_id' if 'utterance_id' in base_ext.columns else 'id'
          _dd_eval = df_base[(df_base.fold == FLAGS.fold) & (df_base.source == 'dd')]
          _eval_ids = set(_dd_eval[_id_col])
          _sep_ext = _load_ext_for_eval()
          if len(_sep_ext) > 0 and _id_col in _sep_ext.columns:
            if getattr(FLAGS, 'ext_eval_group', False) and 'fold' in _sep_ext.columns:
              _ext_fold = _sep_ext[_sep_ext.fold == FLAGS.fold]
              if getattr(FLAGS, 'eval_ext_full', False):
                _eval_ids.update(_ext_fold[_id_col])
              elif len(_ext_fold) >= len(_dd_eval) > 0:
                _eval_ids.update(_ext_fold.sample(n=len(_dd_eval), random_state=42)[_id_col])
              elif len(_ext_fold) > 0:
                _eval_ids.update(_ext_fold[_id_col])
            elif len(_sep_ext) >= len(_dd_eval) > 0:
              _eval_ids.update(_sep_ext.sample(n=len(_dd_eval), random_state=42)[_id_col])
            elif len(_sep_ext) > 0:
              _eval_ids.update(_sep_ext[_id_col])
          n_before_excl = len(train_ext)
          train_ext = train_ext[~train_ext[_id_col].isin(_eval_ids)]
          ic('eval_add_ext train: eval_ext_root differs, excluded eval IDs',
             len(train_dd), len(train_ext), f'n_excluded={n_before_excl - len(train_ext)}')
        elif getattr(FLAGS, 'ext_eval_group', False):
          # ext_eval_group: fold-based split — train ext = all ext NOT in eval fold.
          # ext rows have fold assigned by GroupKFold(child_id) in set_folds.
          train_ext = base_ext[base_ext.fold != FLAGS.fold]
          ic('eval_add_ext train (ext_eval_group): child-grouped',
             len(train_dd), len(train_ext),
             f'ext_eval_fold={FLAGS.fold}',
             f'ext_excluded={(base_ext.fold == FLAGS.fold).sum()}')
        else:
          # Reproduce eval ext sampling on base_ext (same pool as eval path)
          dd_eval_base = df_base[(df_base.fold == FLAGS.fold) & (df_base.source == 'dd')]
          n_dd_eval = len(dd_eval_base)
          if len(base_ext) > 0 and n_dd_eval > 0:
            if len(base_ext) >= n_dd_eval:
              eval_ext_sample = base_ext.sample(n=n_dd_eval, random_state=42)
            else:
              eval_ext_sample = base_ext
            train_ext = base_ext.drop(eval_ext_sample.index)
          else:
            train_ext = base_ext
        # Base train (identical to v13)
        self.df = pd.concat([train_dd, train_ext], ignore_index=True)
        ic('eval_add_ext train: base (v13-identical)', len(train_dd), len(train_ext), len(self.df))
        # Append word-only rows (training-only auxiliary data)
        if len(df_word_only) > 0:
          self.df = pd.concat([self.df, df_word_only], ignore_index=True)
          ic('eval_add_ext train: +word_only', len(df_word_only), 'total', len(self.df))
      elif not FLAGS.online:
        # Base train (v13-identical): exclude eval fold, exclude word-only
        self.df = df_base[df_base.fold != FLAGS.fold].reset_index(drop=True)
        # Append word-only rows (training-only auxiliary data)
        if len(df_word_only) > 0:
          n_before = len(self.df)
          self.df = pd.concat([self.df, df_word_only], ignore_index=True)
          ic('train: +word_only', len(df_word_only), 'total', len(self.df))
      else:
        # online: all DD + all ext (include word-only for training)
        self.df = df.reset_index(drop=True)
      # eval_ext_only: DD data from eval fold goes back to train
      # (skip when eval_add_ext is on — it supersedes eval_ext_only)
      if getattr(FLAGS, 'eval_ext_only', False) and not getattr(FLAGS, 'eval_add_ext', False) \
          and 'source' in self.df.columns and not FLAGS.online:
        eval_dd = df[(df.fold == FLAGS.fold) & (df.source == 'dd')].reset_index(drop=True)
        if len(eval_dd):
          self.df = pd.concat([self.df, eval_dd], ignore_index=True)
      if getattr(FLAGS, 'ext_only', False) and 'source' in self.df.columns:
        self.df = self.df[self.df.source == 'ext'].reset_index(drop=True)
      if getattr(FLAGS, 'train_ext_only', False) and 'source' in self.df.columns:
        n_before = len(self.df)
        self.df = self.df[self.df.source != 'dd'].reset_index(drop=True)
        ic('train_ext_only: filtered out DD from training', n_before, len(self.df))
    else:  # test
      self.df = df.reset_index(drop=True)
    
    samples = 0
    if mode == 'train':
      if FLAGS.samples:
        samples = FLAGS.samples
    if mode == 'eval':
      if FLAGS.eval_samples:
        samples = FLAGS.eval_samples
    if samples > 0 and samples < len(self.df):
      self.df = self.df.sample(samples, random_state=42).reset_index(drop=True)

    if mode == 'train':
      self._log_cross_label_train_stats()

    # ---- word_only_dd/ext_sample_rate: cycle-based sub-sampling of word-only rows (train only) ----
    self._wo_sampling = False
    wo_dd_rate = getattr(FLAGS, 'word_only_dd_sample_rate', 1.0)
    wo_ext_rate = getattr(FLAGS, 'word_only_ext_sample_rate', 1.0)
    if mode == 'train' and 'source' in self.df.columns and 'label_text' in self.df.columns:
      wo_mask = self.df['label_text'].fillna('').str.strip() == ''
      needs_wo_dd = wo_dd_rate < 1.0 and (wo_mask & (self.df.source == 'dd')).any()
      needs_wo_ext = wo_ext_rate < 1.0 and (wo_mask & (self.df.source == 'ext')).any()
      if needs_wo_dd or needs_wo_ext:
        self._wo_sampling = True
        self._wo_dd_rate = wo_dd_rate
        self._wo_ext_rate = wo_ext_rate
        # indices of non-word-only rows (always kept)
        self._non_wo_indices = self.df.index[~wo_mask].tolist()
        # word-only DD and EXT indices (subject to sampling)
        self._wo_dd_indices = self.df.index[wo_mask & (self.df.source == 'dd')].tolist() if needs_wo_dd else []
        self._wo_ext_indices = self.df.index[wo_mask & (self.df.source == 'ext')].tolist() if needs_wo_ext else []
        # word-only rows NOT subject to sampling (rate == 1.0)
        self._wo_kept_indices = []
        if not needs_wo_dd:
          self._wo_kept_indices += self.df.index[wo_mask & (self.df.source == 'dd')].tolist()
        if not needs_wo_ext:
          self._wo_kept_indices += self.df.index[wo_mask & (self.df.source == 'ext')].tolist()
        self._wo_epoch = 0
        self._wo_full_df = self.df.copy()  # save full df for re-sampling each epoch
        self._apply_wo_sampling()
        ic('word_only sampling enabled',
           f'dd_rate={wo_dd_rate} n={len(self._wo_dd_indices)}',
           f'ext_rate={wo_ext_rate} n={len(self._wo_ext_indices)}',
           f'non_wo={len(self._non_wo_indices)}',
           f'active={len(self.df)}')

    # ---- ext_sample_rate: cycle-based EXT sub-sampling (train only) ----
    self._ext_sampling = False
    ext_sample_rate = getattr(FLAGS, 'ext_sample_rate', 1.0)
    if mode == 'train' and ext_sample_rate < 1.0 and 'source' in self.df.columns:
      self._dd_indices = self.df.index[self.df.source != 'ext'].tolist()
      self._all_ext_indices = self.df.index[self.df.source == 'ext'].tolist()
      if self._all_ext_indices:
        self._ext_sampling = True
        self._ext_sample_rate = ext_sample_rate
        self._epoch = 0
        self._apply_ext_sampling()
        ic('ext_sample_rate enabled', ext_sample_rate,
           len(self._dd_indices), len(self._all_ext_indices),
           len(self._active_indices))

    # Sort eval/valid by duration (descending) so similar-length audio is
    # batched together → less padding → faster eval.  We sort the df itself
    # (not the sampler) so that gz.set('eval_df') stays aligned with iteration order.
    if mode in ('eval', 'valid') and getattr(FLAGS, 'sort_by_duration', False) \
        and 'audio_duration_sec' in self.df.columns:
      self.df = self.df.sort_values('audio_duration_sec', ascending=False).reset_index(drop=True)

    # Store eval_df AFTER sorting so lengths & order match in evaluate()
    if mode == 'eval':
      gz.set('eval_df', self.df)

    # ---- Cross-sample CutMix / SpliceMix / MixAug: build child_id → indices map ----
    self._child_to_indices = {}
    self._session_to_indices = {}  # session_id → [indices] for same-session mixing
    self._all_train_indices = []
    self._source_to_indices = {}  # source → [indices] for cross-source mixing
    self._labeltype_to_indices = {}  # 'phonetic'/'word_only' → [indices] for same-type mixing
    self._age_to_indices = {}  # age_bucket → [indices] for same-age mixing
    mix_enabled = (getattr(FLAGS, 'aug_xcutmix', False)
                   or getattr(FLAGS, 'aug_splicemix', False)
                   or getattr(FLAGS, 'aug_mix', False))
    if mode == 'train' and mix_enabled:
      self._all_train_indices = list(range(len(self.df)))
      if 'child_id' in self.df.columns:
        for i, cid in enumerate(self.df['child_id']):
          self._child_to_indices.setdefault(cid, []).append(i)
        ic('child_to_indices built', len(self._child_to_indices))
      if 'session_id' in self.df.columns:
        for i, sid in enumerate(self.df['session_id']):
          self._session_to_indices.setdefault(sid, []).append(i)
        ic('session_to_indices built', len(self._session_to_indices))
      if 'source' in self.df.columns:
        for i, src in enumerate(self.df['source']):
          self._source_to_indices.setdefault(src, []).append(i)
        ic('source_to_indices built', {k: len(v) for k, v in self._source_to_indices.items()})
      # Build label-type index: phonetic (has label_text) vs word_only (no label_text)
      if 'label_text' in self.df.columns:
        for i, lt in enumerate(self.df['label_text']):
          ltype = 'phonetic' if lt else 'word_only'
          self._labeltype_to_indices.setdefault(ltype, []).append(i)
        ic('labeltype_to_indices built', {k: len(v) for k, v in self._labeltype_to_indices.items()})
      if 'age_bucket' in self.df.columns:
        for i, ab in enumerate(self.df['age_bucket'].fillna('')):
          if ab:  # skip empty/unknown
            self._age_to_indices.setdefault(ab, []).append(i)
        ic('age_to_indices built', {k: len(v) for k, v in self._age_to_indices.items()})

    # ---- Duration-aware concat: pre-compute duration stats ----
    self._durations = None
    self._dur_short = 0.0
    self._dur_max_target = 0.0
    if (mode == 'train' and getattr(FLAGS, 'aug_mix_dur_aware', False)
        and getattr(FLAGS, 'aug_mix', False)):
      assert 'audio_duration_sec' in self.df.columns, \
        'aug_mix_dur_aware requires audio_duration_sec column in dataframe'
      self._durations = self.df['audio_duration_sec'].values.copy()
      positive_durs = self._durations[self._durations > 0]
      assert len(positive_durs) > 0, 'No positive durations found for aug_mix_dur_aware'
      pmin = getattr(FLAGS, 'aug_mix_target_dur_pmin', 0.1)
      pmax = getattr(FLAGS, 'aug_mix_target_dur_pmax', 0.95)
      assert 0 < pmin < pmax <= 1.0, \
        f'aug_mix_target_dur_pmin ({pmin}) must be < aug_mix_target_dur_pmax ({pmax}), both in (0, 1]'
      self._dur_short = float(np.percentile(positive_durs, pmin * 100))  # typical 1-word duration
      self._dur_max_target = float(np.percentile(positive_durs, pmax * 100))  # upper bound
      # Clamp to max_audio_sec if set
      if FLAGS.max_audio_sec and self._dur_max_target > FLAGS.max_audio_sec:
        self._dur_max_target = FLAGS.max_audio_sec
      # Pre-sort durations for data-distribution sampling
      self._dur_sample = getattr(FLAGS, 'aug_mix_dur_sample', False)
      if self._dur_sample:
        self._sorted_durs = np.sort(positive_durs)
      ic('aug_mix_dur_aware enabled',
         f'dur_short(p{pmin*100:.0f})={self._dur_short:.2f}s',
         f'max_target(p{pmax*100:.0f})={self._dur_max_target:.2f}s',
         f'dur_sample={self._dur_sample}')

  def _log_cross_label_train_stats(self):
    if not getattr(FLAGS, 'use_cross_labels', False):
      return
    if getattr(FLAGS, 'track', None) != 'word':
      return
    if 'label_text' not in self.df.columns or 'ipa_label' not in self.df.columns:
      return

    label_text = self.df['label_text'].fillna('').astype(str).str.strip()
    ipa_label = self.df['ipa_label'].fillna('').astype(str).str.strip()
    source = self.df['source'] if 'source' in self.df.columns else pd.Series('', index=self.df.index)

    overlap_mask = (label_text != '') & (ipa_label != '')
    ipa_only_dd_mask = (label_text == '') & (ipa_label != '') & (source == 'dd')
    ipa_aux_total_mask = ipa_label != ''

    ic('word cross-label train stats',
       f'overlap={int(overlap_mask.sum())}',
       f'ipa_only_dd={int(ipa_only_dd_mask.sum())}',
       f'ipa_aux_total={int(ipa_aux_total_mask.sum())}')

  def __len__(self):
    if self._ext_sampling:
      return len(self._active_indices)
    return len(self.df)

  def _apply_ext_sampling(self):
    """Select a subset of EXT indices for current epoch using cycling
    permutation for uniform coverage across epochs.
    
    Each full cycle spans ceil(n_ext / n_select) epochs and covers every
    EXT sample exactly once. Different cycles use different random
    permutations for variety.
    """
    n_ext = len(self._all_ext_indices)
    n_select = max(1, int(n_ext * self._ext_sample_rate))
    if n_select >= n_ext:
      self._active_indices = list(range(len(self.df)))
      return

    # Determine which cycle and offset within that cycle
    cycle_len = n_ext  # one full permutation
    global_offset = self._epoch * n_select
    cycle_num = global_offset // cycle_len
    offset = global_offset % cycle_len

    # Permutation seeded by cycle number for reproducibility
    _ext_seed = getattr(FLAGS, 'ext_sample_seed', 42)
    rng = np.random.RandomState(_ext_seed + cycle_num)
    perm = rng.permutation(n_ext)

    # Take n_select elements starting at offset (wrap around)
    if offset + n_select <= n_ext:
      sel = perm[offset:offset + n_select]
    else:
      sel = np.concatenate([perm[offset:], perm[:n_select - (n_ext - offset)]])

    selected_ext = [self._all_ext_indices[i] for i in sel]
    self._active_indices = sorted(self._dd_indices + selected_ext)

  @staticmethod
  def _cycle_select(all_indices, rate, epoch, seed_base=137):
    """Cycle-based sub-sampling: select *rate* fraction of *all_indices*
    for the given epoch, using a cycling permutation for uniform coverage."""
    n = len(all_indices)
    n_select = max(1, int(n * rate))
    if n_select >= n:
      return list(all_indices)
    cycle_len = n
    global_offset = epoch * n_select
    cycle_num = global_offset // cycle_len
    offset = global_offset % cycle_len
    rng = np.random.RandomState(seed_base + cycle_num)
    perm = rng.permutation(n)
    if offset + n_select <= n:
      sel = perm[offset:offset + n_select]
    else:
      sel = np.concatenate([perm[offset:], perm[:n_select - (n - offset)]])
    return [all_indices[i] for i in sel]

  def _apply_wo_sampling(self):
    """Sub-sample word-only DD/EXT rows for current epoch, then rebuild df.
    Uses the same cycling permutation strategy as ext_sample_rate."""
    keep_indices = list(self._non_wo_indices) + list(self._wo_kept_indices)
    _wo_dd_seed = getattr(FLAGS, 'word_only_dd_sample_seed', 42)
    if self._wo_dd_indices and self._wo_dd_rate < 1.0:
      keep_indices += self._cycle_select(
        self._wo_dd_indices, self._wo_dd_rate, self._wo_epoch, seed_base=_wo_dd_seed)
    else:
      keep_indices += self._wo_dd_indices
    _wo_ext_seed = getattr(FLAGS, 'word_only_ext_sample_seed', 42)
    if self._wo_ext_indices and self._wo_ext_rate < 1.0:
      keep_indices += self._cycle_select(
        self._wo_ext_indices, self._wo_ext_rate, self._wo_epoch, seed_base=_wo_ext_seed)
    else:
      keep_indices += self._wo_ext_indices
    keep_indices = sorted(set(keep_indices))
    self.df = self._wo_full_df.iloc[keep_indices].reset_index(drop=True)

  def _get_mix_partner_idx(self, current_idx, same_child=False):
    """Get a random partner index for cross-sample mixing.
    
    Args:
      current_idx: actual df index of the current sample
      same_child: if True, restrict to same child_id
    Returns:
      actual df index of partner, or None if not possible
    """
    if same_child and self._child_to_indices:
      child_id = self.df.iloc[current_idx].get('child_id', '')
      candidates = self._child_to_indices.get(child_id, [])
      candidates = [i for i in candidates if i != current_idx]
      if not candidates:
        # fallback: pick randomly from all
        candidates = [i for i in self._all_train_indices if i != current_idx]
    else:
      candidates = [i for i in self._all_train_indices if i != current_idx]
    
    if not candidates:
      return None
    return random.choice(candidates)

  def _load_raw_audio(self, actual_idx):
    """Load raw audio waveform + label_text (+ cross-labels) + metadata for a given df index.
    Returns (audio_np_array, label_text_str, ipa_label_str, word_label_str, age_bucket_str, source_str) or None on failure.
    Used by aug_mix (concat augmentation) at waveform level.
    """
    try:
      row = self.df.iloc[actual_idx]
      audio = load_audio(row['audio_file'], sr=FLAGS.sample_rate)
      label_text = ''
      if 'label_text' in row and row['label_text']:
        label_text = row['label_text']
      ipa_label = row.get('ipa_label', '') or ''
      word_label = row.get('word_label', '') or ''
      age_bucket = row.get('age_bucket', '') or ''
      source = row.get('source', '') or ''
      return audio, label_text, ipa_label, word_label, age_bucket, source
    except Exception:
      return None

  def _resolve_aug_mix_candidates(self, actual_idx, row, cur_labeltype,
                                  strategy_str='', exclude_indices=None,
                                  chosen_strategy=None):
    """Resolve a fresh aug_mix candidate pool for one partner draw."""
    exclude_indices = set(exclude_indices or ())
    exclude_indices.add(actual_idx)

    cur_source = row.get('source', 'dd') if 'source' in row.index else 'dd'
    if strategy_str:
      chosen = chosen_strategy or _sample_mix_strategy(strategy_str)
      same_session = (chosen == 'same_session')
      same_child = (chosen == 'same_child')
      same_source = (chosen == 'same_source')
      cross_source = (chosen == 'cross_source')
      same_age = (chosen == 'same_age')
    else:
      chosen = 'legacy'
      same_session = getattr(FLAGS, 'aug_mix_same_session', False)
      cross_source = getattr(FLAGS, 'aug_mix_cross_source', False)
      same_source = getattr(FLAGS, 'aug_mix_same_source', False)
      same_child = getattr(FLAGS, 'aug_mix_same_child', False)
      same_age = False

    mix_candidates = None
    if same_session and self._session_to_indices:
      session_id = row.get('session_id', '') if 'session_id' in row.index else ''
      mix_candidates = [i for i in self._session_to_indices.get(session_id, [])
                        if i not in exclude_indices]
      if not mix_candidates and self._child_to_indices:
        child_id = row.get('child_id', '') if 'child_id' in row.index else ''
        mix_candidates = [i for i in self._child_to_indices.get(child_id, [])
                          if i not in exclude_indices]
    elif same_age and self._age_to_indices:
      cur_age = row.get('age_bucket', '') if 'age_bucket' in row.index else ''
      mix_candidates = [i for i in self._age_to_indices.get(cur_age, [])
                        if i not in exclude_indices]
      if not mix_candidates and self._source_to_indices:
        mix_candidates = [i for i in self._source_to_indices.get(cur_source, [])
                          if i not in exclude_indices]
    elif same_child and self._child_to_indices:
      child_id = row.get('child_id', '') if 'child_id' in row.index else ''
      mix_candidates = [i for i in self._child_to_indices.get(child_id, [])
                        if i not in exclude_indices]
    elif same_source and self._source_to_indices:
      mix_candidates = [i for i in self._source_to_indices.get(cur_source, [])
                        if i not in exclude_indices]
    elif cross_source and self._source_to_indices:
      mix_candidates = []
      for src, idxs in self._source_to_indices.items():
        if src != cur_source:
          mix_candidates.extend(i for i in idxs if i not in exclude_indices)

    labeltype_set = set(self._labeltype_to_indices.get(cur_labeltype, []))
    if mix_candidates is not None:
      mix_candidates = [i for i in mix_candidates if i in labeltype_set]
    else:
      mix_candidates = [i for i in self._labeltype_to_indices.get(cur_labeltype, [])
                        if i not in exclude_indices]

    return chosen, mix_candidates

  def _load_raw_sample(self, actual_idx):
    """Load audio → features + labels for a given df index.
    Returns (input_features, labels, utterance_id) or None on failure.
    Used by cross-sample CutMix to load the partner sample.
    """
    try:
      row = self.df.iloc[actual_idx]
      audio = load_audio(row['audio_file'], sr=FLAGS.sample_rate)
      max_samples = int(FLAGS.max_audio_sec * FLAGS.sample_rate)
      orig_len = len(audio)
      crop_start = 0
      if len(audio) > max_samples and self.mode == 'train':
        if FLAGS.random_crop and random.random() < FLAGS.crop_prob:
          crop_start = random.randint(0, len(audio) - max_samples)
          audio = audio[crop_start:crop_start + max_samples]
        else:
          audio = audio[:max_samples]
      # waveform augmentation
      if FLAGS.aug and self.mode == 'train':
        audio = augment_audio(audio, FLAGS.sample_rate)
        if len(audio) > max_samples:
          audio = audio[:max_samples]
      # feature extraction
      if is_waveform_backbone():
        processed = self.processor(
          audio, sampling_rate=FLAGS.sample_rate, return_tensors='np'
        )
        input_features = processed.input_values[0]
      else:
        input_features = self.processor.feature_extractor(
          audio, sampling_rate=FLAGS.sample_rate, return_tensors='np'
        ).input_features[0]
      # labels
      labels = []
      if 'label_text' in row and row['label_text']:
        if self.tokenizer is not None:
          labels = tokenize_text(self.tokenizer, row['label_text'])
        else:
          labels = []  # tokenizer=None (NeMo): labels passed as text via collate
        max_label_len = getattr(FLAGS, 'max_label_tokens', 448)
        if len(labels) > max_label_len:
          labels = labels[:max_label_len]
        # Duration-based label token limiting
        max_tps = getattr(FLAGS, 'max_tokens_per_sec', 0)
        if max_tps > 0 and labels:
          dur = row.get('audio_duration_sec', 0)
          if dur > 0:
            dur_limit = max(10, int(dur * max_tps))
            if len(labels) > dur_limit:
              labels = labels[:dur_limit]
        if (FLAGS.random_crop and FLAGS.crop_label
            and self.mode == 'train' and orig_len > max_samples and labels):
          crop_frac = max_samples / orig_len
          keep_tokens = max(1, int(len(labels) * crop_frac))
          start_token = int(len(labels) * (crop_start / orig_len))
          end_token = min(len(labels), start_token + keep_tokens)
          labels = labels[start_token:end_token]
      uid = row.get('id', row.get('utterance_id', ''))
      return input_features, labels, uid
    except Exception:
      return None

  def __getitem__(self, idx):
    try:
      # Map idx through active indices when ext_sample_rate < 1
      actual_idx = self._active_indices[idx] if self._ext_sampling else idx
      row = self.df.iloc[actual_idx]
      utterance_id = row.get('id', row.get('utterance_id', ''))
      _aug_debug_info = {
          'utterance_id': utterance_id,
          'source': row.get('source', ''),
          'age_bucket': row.get('age_bucket', ''),
          'audio_duration_sec': row.get('audio_duration_sec', 0),
      }

      # load audio
      audio = load_audio(row['audio_file'], sr=FLAGS.sample_rate)

      # ---- Concat augmentation (aug_mix): concatenate random samples at waveform level ----
      label_text_override = None  # set when aug_mix modifies the label
      ipa_label_override = None   # cross-label override for aug_mix
      word_label_override = None  # cross-label override for aug_mix
      _aug_mix_age_valid = True   # all segments same age_bucket? (for aux age label)
      _aug_mix_domain_valid = True  # all segments same source? (for aux domain label)
      _aug_mix_did_speed = False  # True when per-segment speed aug was applied
      _aug_mix_max_dur = getattr(FLAGS, 'aug_mix_max_dur', 0)
      if (self.mode == 'train' and getattr(FLAGS, 'aug_mix', False)
          and random.random() < getattr(FLAGS, 'aug_mix_prob', 0.5)
          and self._all_train_indices
          and not (getattr(FLAGS, 'aug_mix_ext_only', False) and row.get('source', 'dd') == 'dd')
          and not (getattr(FLAGS, 'aug_mix_dd_only', False) and row.get('source', 'dd') == 'ext')
          and (_aug_mix_max_dur <= 0 or len(audio) / FLAGS.sample_rate <= _aug_mix_max_dur)):
        _debug_key = 'aug_once:mix'
        _capture_mix = _should_capture_aug_once(_debug_key)
        _before_mix = audio.copy() if _capture_mix else None
        cur_label = row.get('label_text', '') if 'label_text' in row.index else ''
        cur_ipa = row.get('ipa_label', '') if 'ipa_label' in row.index else ''
        cur_word = row.get('word_label', '') if 'word_label' in row.index else ''
        mix_num_max = max(1, int(getattr(FLAGS, 'aug_mix_num', 1)))
        mix_num = random.randint(1, mix_num_max) if getattr(FLAGS, 'aug_mix_random_num', False) else mix_num_max
        # --- Resolve mixing strategy ---
        # Priority: per-source (dd_strategy/ext_strategy) > global (aug_mix_strategy) > legacy bool flags
        cur_source = row.get('source', 'dd') if 'source' in row.index else 'dd'
        per_source_strategy = ''
        if cur_source == 'dd':
          per_source_strategy = getattr(FLAGS, 'aug_mix_dd_strategy', '')
        elif cur_source == 'ext':
          per_source_strategy = getattr(FLAGS, 'aug_mix_ext_strategy', '')
        strategy_str = per_source_strategy or getattr(FLAGS, 'aug_mix_strategy', '')
        # Collect all segments (original + partners), then optionally shuffle
        _limit_len = getattr(FLAGS, 'aug_mix_limit_len', False)
        _fit_len = getattr(FLAGS, 'aug_mix_fit_len', False)
        _max_mix_samples = int(FLAGS.max_audio_sec * FLAGS.sample_rate) if (_limit_len or _fit_len) else None
        cur_labeltype = 'phonetic' if cur_label else 'word_only'
        cur_age_bucket = row.get('age_bucket', '') if 'age_bucket' in row.index else ''
        cur_source_val = cur_source
        segments = [(audio, cur_label, cur_ipa, cur_word, cur_age_bucket, cur_source_val)]
        total_audio_len = len(audio)
        total_label_units = _estimate_aug_mix_label_units(cur_label)
        partner_strategies = []
        blocked_partner_indices = set()
        _dur_aware = getattr(FLAGS, 'aug_mix_dur_aware', False) and self._durations is not None
        if _dur_aware:
          # --- Duration-aware partner selection ---
          # target = uniform(cur_dur + dur_short, dur_max_target)
          # Guarantees at least 1 partner (~1-word). Longer audio → narrower range → fewer partners.
          cur_dur = len(audio) / FLAGS.sample_rate
          target_min = cur_dur + self._dur_short
          target_max = self._dur_max_target
          if target_min < target_max:
            if self._dur_sample:
              # Sample from real data distribution: pick a random training sample's duration
              # Reject samples shorter than target_min (resample up to 10 times, then fallback)
              for _ in range(10):
                sampled = self._sorted_durs[random.randint(0, len(self._sorted_durs) - 1)]
                if sampled >= target_min:
                  break
              target_dur = max(sampled, target_min)
              target_dur = min(target_dur, target_max)
            else:
              target_dur = random.uniform(target_min, target_max)
          else:
            # cur_dur already near max → just add one short partner
            target_dur = cur_dur + self._dur_short
          remaining = target_dur - cur_dur
          max_partners = mix_num  # aug_mix_num as cap
          _max_retries = max_partners * 3  # avoid infinite loop on bad luck
          n_retries = 0
          n_added = 0
          while remaining > 0 and n_added < max_partners and n_retries < _max_retries:
            chosen_strategy, mix_candidates = self._resolve_aug_mix_candidates(
              actual_idx,
              row,
              cur_labeltype,
              strategy_str=strategy_str,
              exclude_indices=blocked_partner_indices,
            )
            if not mix_candidates:
              break
            mix_cand_arr = np.array(mix_candidates)
            cand_durs = self._durations[mix_cand_arr]
            pick = random.randint(0, len(mix_cand_arr) - 1)
            if cand_durs[pick] > remaining:
              n_retries += 1
              continue  # too long, try another
            partner_idx = int(mix_cand_arr[pick])
            partner = self._load_raw_audio(partner_idx)
            if partner is None or (not partner[1] and not partner[2] and not partner[3]):
              blocked_partner_indices.add(partner_idx)
              continue
            # Guard against label-audio mismatch
            if _max_mix_samples is not None and total_audio_len + len(partner[0]) > _max_mix_samples:
              if _fit_len:
                blocked_partner_indices.add(partner_idx)
                continue
              else:
                break
            _skip_partner, mixed_label_units, _ = _get_aug_mix_guard(
              len(partner[0]), partner[1], total_audio_len, total_label_units)
            if _skip_partner:
              blocked_partner_indices.add(partner_idx)
              continue
            segments.append(partner)
            partner_strategies.append(chosen_strategy)
            blocked_partner_indices.add(partner_idx)
            total_audio_len += len(partner[0])
            total_label_units = mixed_label_units
            remaining -= cand_durs[pick]
            n_added += 1
          # Guarantee at least 1 partner: if retries exhausted, force-pick shortest
          if n_added == 0:
            chosen_strategy, mix_candidates = self._resolve_aug_mix_candidates(
              actual_idx,
              row,
              cur_labeltype,
              strategy_str=strategy_str,
              exclude_indices=blocked_partner_indices,
            )
            if mix_candidates:
              mix_cand_arr = np.array(mix_candidates)
              cand_durs = self._durations[mix_cand_arr]
              pick = int(np.argmin(cand_durs))
              partner_idx = int(mix_cand_arr[pick])
              partner = self._load_raw_audio(partner_idx)
              if partner is not None and (partner[1] or partner[2] or partner[3]):
                if _max_mix_samples is not None and total_audio_len + len(partner[0]) > _max_mix_samples:
                  blocked_partner_indices.add(partner_idx)
                else:
                  _skip_partner, mixed_label_units, _ = _get_aug_mix_guard(
                    len(partner[0]), partner[1], total_audio_len, total_label_units)
                  if _skip_partner:
                    blocked_partner_indices.add(partner_idx)
                  else:
                    segments.append(partner)
                    partner_strategies.append(chosen_strategy)
                    blocked_partner_indices.add(partner_idx)
                    total_audio_len += len(partner[0])
                    total_label_units = mixed_label_units
        else:
          # --- Original random partner selection ---
          for _ in range(mix_num):
            chosen_strategy, mix_candidates = self._resolve_aug_mix_candidates(
              actual_idx,
              row,
              cur_labeltype,
              strategy_str=strategy_str,
              exclude_indices=blocked_partner_indices,
            )
            if mix_candidates:
              partner_idx = random.choice(mix_candidates)
            else:
              break  # no compatible candidates
            if partner_idx is not None:
              partner = self._load_raw_audio(partner_idx)
              if partner is not None:
                # Skip partners with no usable labels at all
                if not partner[1] and not partner[2] and not partner[3]:
                  blocked_partner_indices.add(partner_idx)
                  continue
                # Guard against label-audio mismatch: audio gets truncated but
                # full concatenated label is kept, violating CTC constraint.
                if _max_mix_samples is not None and total_audio_len + len(partner[0]) > _max_mix_samples:
                  if _fit_len:
                    blocked_partner_indices.add(partner_idx)
                    continue  # skip this partner, try shorter ones
                  else:
                    break  # stop adding entirely (aug_mix_limit_len)
                _skip_partner, mixed_label_units, _ = _get_aug_mix_guard(
                  len(partner[0]), partner[1], total_audio_len, total_label_units)
                if _skip_partner:
                  blocked_partner_indices.add(partner_idx)
                  continue
                segments.append(partner)
                partner_strategies.append(chosen_strategy)
                blocked_partner_indices.add(partner_idx)
                total_audio_len += len(partner[0])
                total_label_units = mixed_label_units
              else:
                blocked_partner_indices.add(partner_idx)
        if getattr(FLAGS, 'aug_mix_shuffle', False) and len(segments) > 1:
          random.shuffle(segments)
        if len(segments) > 1 and _capture_mix:
          _mixed_audio = np.concatenate([s[0] for s in segments])
          _log_aug_once('mix', {
              'num_segments': int(len(segments)),
              'segment_audio_sec': [round(len(s[0]) / FLAGS.sample_rate, 3) for s in segments],
              'segment_label_units': [_estimate_aug_mix_label_units(s[1]) for s in segments],
              'total_audio_sec': round(sum(len(s[0]) for s in segments) / FLAGS.sample_rate, 3),
              'total_label_units': int(total_label_units),
              'strategies': partner_strategies[:],
              **_before_after_summary(_before_mix, _mixed_audio),
              **_aug_debug_context(_aug_debug_info),
          }, key=_debug_key)
        # Debug logging
        if getattr(FLAGS, 'aug_mix_debug', False) and len(segments) > 1:
          if not hasattr(self, '_aug_mix_debug_count'):
            self._aug_mix_debug_count = 0
          max_debug = getattr(FLAGS, 'aug_mix_debug_count', 50)
          if self._aug_mix_debug_count < max_debug:
            self._aug_mix_debug_count += 1
            orig_dur = len(segments[0][0]) / FLAGS.sample_rate
            partner_durs = [len(s[0]) / FLAGS.sample_rate for s in segments[1:]]
            total_dur = sum(len(s[0]) for s in segments) / FLAGS.sample_rate
            total_units = sum(_estimate_aug_mix_label_units(s[1]) for s in segments)
            strategy_info = strategy_str if strategy_str else 'default'
            if partner_strategies:
              strategy_info = f'{strategy_info} -> {partner_strategies}'
            dur_aware_info = f' target={target_dur:.2f}s' if _dur_aware else ''
            logger.info(
              f'[aug_mix_debug #{self._aug_mix_debug_count}] '
              f'orig={orig_dur:.2f}s +{len(partner_durs)} partners '
              f'[{", ".join(f"{d:.2f}s" for d in partner_durs)}] '
              f'-> total={total_dur:.2f}s units={total_units} | '
              f'strategy={strategy_info}{dur_aware_info} '
              f'label="{segments[0][1][:30]}..."'
            )
        # Per-segment speed perturbation (before concat, so each segment gets
        # independent speed variation and short_only judges original duration)
        if FLAGS.aug_speed and len(segments) > 1:
          new_segments = []
          for seg in segments:
            seg_audio = seg[0]
            if random.random() < FLAGS.aug_speed_prob:
              if not FLAGS.aug_speed_short_only or len(seg_audio) / FLAGS.sample_rate <= FLAGS.aug_speed_short_dur:
                seg_audio = _aug_speed(seg_audio, FLAGS.sample_rate, debug_info=_aug_debug_info)
            new_segments.append((seg_audio,) + seg[1:])
          segments = new_segments
          _aug_mix_did_speed = True
        # Concatenate all segments
        audio = np.concatenate([s[0] for s in segments])
        cur_label = ' '.join(s[1] for s in segments if s[1])
        cur_ipa = ' '.join(s[2] for s in segments if s[2])
        cur_word = ' '.join(s[3] for s in segments if s[3])
        # Guard: if any segment is missing a label type, clear that override
        # to prevent partial-coverage labels (CTC alignment error).
        # e.g. word-only + phonetic → label_text only from phonetic → clear it.
        # e.g. phonetic (has word) + phonetic (no word) → word partial → clear it.
        # word-only + word-only → label_text/ipa all empty (OK), word full coverage (OK).
        if len(segments) > 1:
          all_have_label = all(s[1] for s in segments)
          all_have_ipa = all(s[2] for s in segments)
          all_have_word = all(s[3] for s in segments)
          label_text_override = cur_label if all_have_label else ''
          ipa_label_override = cur_ipa if all_have_ipa else ''
          word_label_override = cur_word if all_have_word else ''
          # Aux age/domain: only valid when ALL segments share the same value
          seg_ages = [s[4] for s in segments]
          seg_sources = [s[5] for s in segments]
          _aug_mix_age_valid = all(a == seg_ages[0] and a for a in seg_ages)
          _aug_mix_domain_valid = all(s == seg_sources[0] and s for s in seg_sources)
        else:
          label_text_override = cur_label
          ipa_label_override = cur_ipa
          word_label_override = cur_word

      # filter_long_audio: exclude (not truncate) training clips > max_audio_sec
      # Official baseline excludes >25s clips entirely; default melt truncates.
      max_samples = int(FLAGS.max_audio_sec * FLAGS.sample_rate)
      if FLAGS.filter_long_audio and self.mode == 'train' and len(audio) > max_samples:
        return None

      # truncate / random-crop if too long (train/valid truncate; eval/test uses full audio)
      crop_start = 0  # track for optional label cropping
      orig_len = len(audio)
      _truncate_modes = {'train', 'valid'}
      if getattr(FLAGS, 'eval_truncate_audio', False):
        _truncate_modes.add('eval')
      if len(audio) > max_samples and self.mode in _truncate_modes:
        if FLAGS.random_crop and random.random() < FLAGS.crop_prob and self.mode == 'train':
          _debug_key = 'aug_once:random_crop'
          _capture_crop = _should_capture_aug_once(_debug_key)
          _before_crop = audio.copy() if _capture_crop else None
          crop_start = random.randint(0, len(audio) - max_samples)
          audio = audio[crop_start:crop_start + max_samples]
          if _capture_crop:
            _log_aug_once('random_crop', {
                'orig_audio_sec': round(orig_len / FLAGS.sample_rate, 3),
                'crop_start_sec': round(crop_start / FLAGS.sample_rate, 3),
                'crop_audio_sec': round(max_samples / FLAGS.sample_rate, 3),
                **_before_after_summary(_before_crop, audio),
                **_aug_debug_context(_aug_debug_info),
            }, key=_debug_key)
        else:
          audio = audio[:max_samples]
      crop_keep_frac = min(1.0, max_samples / max(orig_len, 1))
      crop_start_frac = crop_start / max(orig_len, 1)
      crop_label_active = (
          getattr(FLAGS, 'crop_label', False)
          and self.mode == 'train'
          and orig_len > max_samples
      )

      # waveform-level augmentation (train only)
      do_aug = FLAGS.aug and self.mode == 'train'
      if do_aug:
        audio = augment_audio(audio, FLAGS.sample_rate, skip_speed=_aug_mix_did_speed,
                              debug_info=_aug_debug_info)
        # re-truncate after speed perturbation may have changed length
        if len(audio) > max_samples:
          audio = audio[:max_samples]

      # extract features (mel for whisper, normalised waveform for wav2vec2/hubert)
      if is_waveform_backbone():
        processed = self.processor(
          audio, sampling_rate=FLAGS.sample_rate, return_tensors='np'
        )
        # Wav2Vec2/HuBERT: input_values; WhisperFeatureExtractor: input_features
        if hasattr(processed, 'input_values') and 'input_values' in processed:
          input_features = processed.input_values[0]  # (T,)
        else:
          input_features = processed.input_features[0]
      else:
        input_features = self.processor.feature_extractor(
          audio, sampling_rate=FLAGS.sample_rate, return_tensors='np'
        ).input_features[0]
        # spectrogram-level augmentation (Whisper mel, train only)
        if do_aug:
          input_features = augment_mel(input_features, debug_info=_aug_debug_info)

      # tokenise labels early (needed for cross-sample mixing)
      labels = []
      label_text = ''
      # Use overridden label from aug_mix if available
      _raw_label = label_text_override if label_text_override is not None else (
        row['label_text'] if ('label_text' in row and row['label_text']) else ''
      )
      if crop_label_active and _raw_label:
        _raw_label = _truncate_text_by_ratio(
            _raw_label, crop_start_frac, crop_keep_frac)
      if self.mode != 'test' and _raw_label:
        label_text = _raw_label
        if self.tokenizer is not None:
          labels = tokenize_text(self.tokenizer, _raw_label)
        # When tokenizer is None (e.g. NeMo backbone), labels stay empty;
        # the raw label_text string will be passed through collate_fn and
        # used directly by the model's loss function (avoids Whisper BPE roundtrip).
        max_label_len = getattr(FLAGS, 'max_label_tokens', 448)
        if len(labels) > max_label_len:
          labels = labels[:max_label_len]
        # Duration-based label token limiting
        max_tps = getattr(FLAGS, 'max_tokens_per_sec', 0)
        if max_tps > 0 and labels:
          dur = row.get('audio_duration_sec', 0)
          if dur > 0:
            dur_limit = max(10, int(dur * max_tps))
            if len(labels) > dur_limit:
              labels = labels[:dur_limit]
      
      # ---- Cross-sample CutMix / SpliceMix (train, non-waveform only) ----
      if do_aug and not is_waveform_backbone() and labels:
        applied_mix = False
        # Priority 2: SpliceMix (alignment-aware)
        if getattr(FLAGS, 'aug_splicemix', False) and random.random() < FLAGS.aug_splicemix_prob:
          same_child = getattr(FLAGS, 'aug_splicemix_same_child', False)
          partner_idx = self._get_mix_partner_idx(actual_idx, same_child=same_child)
          if partner_idx is not None:
            partner = self._load_raw_sample(partner_idx)
            if partner is not None:
              p_mel, p_labels, p_id = partner
              if p_labels and p_mel.shape[0] == input_features.shape[0]:
                input_features, labels = _splicemix_mel(
                  input_features, labels, utterance_id,
                  p_mel, p_labels, p_id
                )
                applied_mix = True
        # Priority 1: Simple cross-sample CutMix (fallback if splicemix not applied)
        if not applied_mix and getattr(FLAGS, 'aug_xcutmix', False) and random.random() < FLAGS.aug_xcutmix_prob:
          same_child = getattr(FLAGS, 'aug_xcutmix_same_child', False)
          partner_idx = self._get_mix_partner_idx(actual_idx, same_child=same_child)
          if partner_idx is not None:
            partner = self._load_raw_sample(partner_idx)
            if partner is not None:
              p_mel, p_labels, p_id = partner
              if p_labels and p_mel.shape[0] == input_features.shape[0]:
                input_features, labels = _xcutmix_mel(
                  input_features, labels, p_mel, p_labels
                )

      fe = {
        'input_features': input_features,
      }

      fe['labels'] = labels
      if label_text:
        fe['label_text'] = label_text

      # ---- Multi-task labels (IPA + Word) ----
      # When use_cross_labels is enabled, provide both ipa_labels and word_labels.
      # Samples missing one label type will have empty lists → masked in loss.
      if getattr(FLAGS, 'use_cross_labels', False) and self.mode != 'test':
        max_label_len = getattr(FLAGS, 'max_label_tokens', 448)
        use_word_aux_ipa = getattr(FLAGS, 'word_aux_ipa', False) and getattr(FLAGS, 'track', None) == 'word'
        # Use aug_mix overrides if available (concatenated cross-labels from partner)
        ipa_text = ipa_label_override if ipa_label_override is not None else (row.get('ipa_label', '') or '')
        word_text = word_label_override if word_label_override is not None else (row.get('word_label', '') or '')
        if use_word_aux_ipa:
          word_text = ipa_text
          ipa_text = ''
        if crop_label_active:
          if ipa_text:
            ipa_text = _truncate_text_by_ratio(
                ipa_text, crop_start_frac, crop_keep_frac)
          if word_text:
            word_text = _truncate_text_by_ratio(
                word_text, crop_start_frac, crop_keep_frac)
        if ipa_text and self.tokenizer is not None:
          ipa_labels = tokenize_text(self.tokenizer, ipa_text)
          if len(ipa_labels) > max_label_len:
            ipa_labels = ipa_labels[:max_label_len]
          fe['ipa_labels'] = ipa_labels
        elif ipa_text and self.tokenizer is None:
          # NeMo backbone: no HF tokenizer, but CTC loss uses raw text
          # (via _current_ipa_label_texts). Set dummy non-empty label so
          # collate produces ipa_mask=1.0.
          fe['ipa_labels'] = [0]
        else:
          fe['ipa_labels'] = []
        if word_text and self.tokenizer is not None:
          word_labels = tokenize_text(self.tokenizer, word_text)
          if len(word_labels) > max_label_len:
            word_labels = word_labels[:max_label_len]
          fe['word_labels'] = word_labels
        elif word_text and self.tokenizer is None:
          # NeMo backbone: word_ctc loss uses raw text strings
          # (via _current_word_label_texts). Set dummy non-empty label so
          # collate produces word_mask=1.0.
          fe['word_labels'] = [0]
        else:
          fe['word_labels'] = []
        # Pass raw text for cross-labels (avoids BPE roundtrip)
        if ipa_text:
          fe['ipa_label_text'] = ipa_text
        if word_text:
          fe['word_label_text'] = word_text

      fe['id'] = utterance_id

      # Per-sample loss weight: DD=1.0, EXT=ext_weight
      ext_weight = getattr(FLAGS, 'ext_weight', 1.0)
      if ext_weight != 1.0 and 'source' in row.index and row['source'] == 'ext':
        fe['weight'] = ext_weight
      else:
        fe['weight'] = 1.0

      # Down-weight weak-alignment samples (short label + long audio)
      wa_weight = getattr(FLAGS, 'weak_align_weight', 0)
      if wa_weight > 0:
        dur = row.get('audio_duration_sec', 0)
        label_text = row.get('phonetic_text', '') or row.get('label_text', '') or ''
        n_chars = len(label_text.replace(' ', ''))
        if dur > 0 and n_chars > 0:
          cps = n_chars / dur
          cps_thr = getattr(FLAGS, 'weak_align_cps_threshold', 3.0)
          if cps < cps_thr:
            fe['weight'] *= wa_weight

      # ---- Metadata labels for auxiliary losses (age / domain) ----
      # After aug_mix, age/domain labels are only valid when all mixed segments agree
      if getattr(FLAGS, 'aux_age_weight', 0) > 0 and 'age_bucket' in row.index:
        if label_text_override is not None and not _aug_mix_age_valid:
          fe['age_bucket'] = ''  # mixed ages → mask out
        else:
          fe['age_bucket'] = row.get('age_bucket', '')
      if getattr(FLAGS, 'aux_domain_weight', 0) > 0 and 'source' in row.index:
        if label_text_override is not None and not _aug_mix_domain_valid:
          fe['domain_label'] = -1  # mixed sources → mask out
        else:
          fe['domain_label'] = 1 if row.get('source', '') == 'dd' else 0

      # ---- Length prediction labels (nchars / nspaces) ----
      if getattr(FLAGS, 'aux_nchars_weight', 0) > 0 or getattr(FLAGS, 'aux_nspaces_weight', 0) > 0:
        _lt = label_text_override if label_text_override is not None else (label_text or '')
        if _lt:
          import math
          if getattr(FLAGS, 'aux_nchars_weight', 0) > 0:
            _n_ipa_chars = len(_lt.replace(' ', ''))
            fe['nchars_label'] = math.log1p(_n_ipa_chars)  # log(1 + n_chars)
          if getattr(FLAGS, 'aux_nspaces_weight', 0) > 0:
            _n_spaces = _lt.count(' ')
            fe['nspaces_label'] = math.log1p(_n_spaces)  # log(1 + n_spaces)

      return fe
    except Exception as e:
      # Attempt to capture worker id and full traceback for debugging transient IO errors
      try:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 'main'
      except Exception:
        worker_id = 'unknown'
      tb = traceback.format_exc()
      audio_file = None
      try:
        audio_file = row.get('audio_file', '')
      except Exception:
        audio_file = ''
      msg = (
        f"Dataset __getitem__ exception\n"
        f"idx: {idx}\n"
        f"worker: {worker_id}\n"
        f"audio_file: {audio_file}\n"
        f"exception: {e}\n"
        f"traceback:\n{tb}\n"
      )
      # print and persist to temp log to surface in worker stdout
      print(msg)
      try:
        with open('/tmp/dataset_read_errors.log', 'a') as f:
          f.write(msg + '\n' + ('-' * 80) + '\n')
      except Exception:
        pass
      # fallback: return a random other sample instead of crashing training
      fallback_idx = random.randint(0, len(self) - 1)
      if fallback_idx == idx:
        fallback_idx = (idx + 1) % len(self)
      return self[fallback_idx]

  def set_epoch(self, epoch):
    self.epoch = epoch
    if self._wo_sampling:
      self._wo_epoch = int(epoch)
      self._apply_wo_sampling()
      # ext_sampling indices depend on df which was rebuilt by wo_sampling,
      # so re-init ext indices if needed
      if self._ext_sampling:
        self._dd_indices = self.df.index[self.df.source != 'ext'].tolist()
        self._all_ext_indices = self.df.index[self.df.source == 'ext'].tolist()
    if self._ext_sampling:
      self._epoch = int(epoch)
      self._apply_ext_sampling()


def _check_train_eval_overlap(train_ds, eval_ds, valid_ds):
  """Assert that train and eval/valid datasets have no overlapping sample IDs.
  
  Uses 'utterance_id' or 'id' column. Raises AssertionError on overlap.
  This is a safety net against data leakage — should hold for ALL projects.
  """
  id_col = 'utterance_id' if 'utterance_id' in train_ds.df.columns else 'id'
  if id_col not in train_ds.df.columns:
    return  # no ID column to check

  train_ids = set(train_ds.df[id_col])

  for name, ds in [('eval', eval_ds), ('valid', valid_ds)]:
    if ds is None or len(ds) == 0:
      continue
    if id_col not in ds.df.columns:
      continue
    eval_ids = set(ds.df[id_col])
    overlap = train_ids & eval_ids

    if FLAGS.online:
      # Online (full-data) mode: eval MUST be a subset of train.
      # All data is used for training; eval fold is kept for monitoring only.
      not_in_train = eval_ids - train_ids
      assert not not_in_train, (
        f'Online mode: {len(not_in_train)} {name} IDs not found in train '
        f'({len(train_ids)}). Eval must be a subset of train in online mode. '
        f'Examples: {sorted(not_in_train)[:10]}'
      )
      ic(f'online: {name} is subset of train',
         len(eval_ids), len(train_ids))
    else:
      # Offline: no overlap allowed (data leakage check).
      if overlap:
        n = len(overlap)
        examples = sorted(overlap)[:10]
        ic(f'DATA LEAKAGE: {n} overlapping IDs between train and {name}', examples)
        assert not overlap, (
          f'Data leakage detected: {n} IDs overlap between train ({len(train_ids)}) '
          f'and {name} ({len(eval_ids)}). Examples: {examples}'
        )
  ic('train/eval overlap check passed',
     len(train_ids),
     len(eval_ds) if eval_ds else 0,
     len(valid_ds) if valid_ds else 0)


def get_dl(mode='train', df=None):
  """Build train / eval / valid DataLoaders."""
  assert mode in ['train', 'test']

  tokenizer = get_tokenizer(FLAGS.backbone)
  pad_token_id = (tokenizer.pad_token_id or 0) if tokenizer is not None else 0

  def _get_bucket_label_texts(df_):
    texts = pd.Series([''] * len(df_), index=df_.index, dtype=object)
    for col in ['label_text', 'word_label', 'ipa_label']:
      if col in df_.columns:
        cur = df_[col].fillna('').astype(str)
        empty_mask = texts.astype(str).str.strip() == ''
        texts.loc[empty_mask] = cur.loc[empty_mask]
    return texts.fillna('').astype(str)

  def _estimate_bucket_target_units(df_):
    texts = _get_bucket_label_texts(df_)
    track = getattr(FLAGS, 'track', '')
    score_metric = getattr(FLAGS, 'score_metric', '')
    if track == 'word' or score_metric == 'wer':
      units = texts.apply(lambda x: max(len(x.split()), 1) if x.strip() else 1)
    else:
      units = texts.apply(
          lambda x: max(len(x.replace(' ', '')), 1) if x.strip() else 1)
    return units.astype(np.float32).to_numpy()

  def _get_bucket_durations(df_):
    durations = df_['audio_duration_sec'].fillna(0).astype(np.float32).to_numpy()
    max_audio_sec = getattr(FLAGS, 'max_audio_sec', None)
    if max_audio_sec:
      durations = np.minimum(durations, float(max_audio_sec))
    return np.maximum(durations, 1e-3)

  def _get_bucket_lens(dataset):
    df_ = dataset.df
    durations = _get_bucket_durations(df_)
    key = getattr(FLAGS, 'bucket_batch_key', 'audio') or 'audio'
    if key == 'audio':
      return durations, {
          'key': key,
          'duration_mean': round(float(durations.mean()), 3) if len(durations) else 0.0,
      }
    if key == 'rnnt_cost':
      target_units = _estimate_bucket_target_units(df_)
      lens = durations * np.maximum(target_units, 1.0)
      return lens, {
          'key': key,
          'duration_mean': round(float(durations.mean()), 3) if len(durations) else 0.0,
          'target_units_mean': round(float(target_units.mean()), 3) if len(target_units) else 0.0,
          'cost_mean': round(float(lens.mean()), 3) if len(lens) else 0.0,
          'cost_p95': round(float(np.percentile(lens, 95)), 3) if len(lens) else 0.0,
      }
    raise ValueError(f'Unsupported bucket_batch_key: {key}')

  def _summarize_array(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
      return 'n=0'
    return (
        f'n={arr.size} min={arr.min():.3f} mean={arr.mean():.3f} '
        f'p50={np.percentile(arr, 50):.3f} p95={np.percentile(arr, 95):.3f} '
        f'max={arr.max():.3f}'
    )

  def _log_bucket_sampler_preview(bucket_sampler, dataset, bucket_lens, bucket_info):
    preview_batches = max(int(getattr(FLAGS, 'bucket_batch_debug_batches', 0) or 0), 0)
    if preview_batches <= 0:
      return
    if not hasattr(bucket_sampler, 'buckets') or not hasattr(bucket_sampler, 'permutation'):
      return

    df_ = dataset.df.reset_index(drop=True)
    durations = _get_bucket_durations(df_)
    key = bucket_info.get('key', getattr(FLAGS, 'bucket_batch_key', 'audio'))
    target_units = _estimate_bucket_target_units(df_) if key == 'rnnt_cost' else None

    logger.info(
        '[bucket_batch.preview] global '
        f'key={key} batches={len(bucket_sampler)} '
        f'bucket_lens=({_summarize_array(bucket_lens)}) '
        f'durations=({_summarize_array(durations)})'
        + (f' target_units=({_summarize_array(target_units)})' if target_units is not None else '')
    )

    preview_perm = np.asarray(bucket_sampler.permutation)[:preview_batches]
    for i, bucket_idx in enumerate(preview_perm):
      indices = np.asarray(bucket_sampler.buckets[bucket_idx])
      indices = indices[indices >= 0]
      if not len(indices):
        continue
      batch_bucket_lens = np.asarray(bucket_lens)[indices]
      batch_durations = durations[indices]
      msg = (
          f'[bucket_batch.preview] batch={i} bucket_id={int(bucket_idx)} '
          f'size={len(indices)} bucket_lens=({_summarize_array(batch_bucket_lens)}) '
          f'durations=({_summarize_array(batch_durations)})'
      )
      if target_units is not None:
        batch_units = target_units[indices]
        msg += f' target_units=({_summarize_array(batch_units)})'
      if 'id' in df_.columns or 'utterance_id' in df_.columns:
        id_col = 'id' if 'id' in df_.columns else 'utterance_id'
        sample_ids = df_.iloc[indices][id_col].astype(str).tolist()[:3]
        msg += f' sample_ids={sample_ids}'
      logger.info(msg)

  def collate_fn(batch):
    """Pad audio features and labels within batch."""
    import torch
    
    # Filter out None samples (e.g. from filter_long_audio)
    batch = [b for b in batch if b is not None]
    assert batch, 'All samples in batch were None (filtered). Check filter_long_audio / max_audio_sec settings.'
    
    if is_waveform_backbone():
      # Variable-length waveforms: pad & create attention mask
      waveforms = [torch.tensor(b['input_features'], dtype=torch.float32) for b in batch]
      lengths = [w.shape[0] for w in waveforms]
      max_len = max(lengths)
      input_features = torch.zeros(len(batch), max_len, dtype=torch.float32)
      attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
      for i, (w, l) in enumerate(zip(waveforms, lengths)):
        input_features[i, :l] = w
        attention_mask[i, :l] = 1
    else:
      # Fixed-size mel spectrograms (Whisper pads to 30s internally)
      input_features = torch.stack(
        [torch.tensor(b['input_features'], dtype=torch.float32) for b in batch], dim=0
      )
      attention_mask = None
    
    # Build labels tensor with shape (B, T).  Samples without primary labels
    # (e.g. word-only samples with no phonetic_text) get a [-100] placeholder
    # so that the tensor always has batch_size rows.  CTC/S2S loss naturally
    # produces 0 for all-(-100) samples.  'labels' key must always be present
    # so that get_x_y() in the training loop does not crash.
    from torch.nn.utils.rnn import pad_sequence as _pad_seq
    labels_list = []
    for b in batch:
      bl = b['labels']
      if bl:
        labels_list.append(torch.tensor(bl, dtype=torch.long))
      else:
        labels_list.append(torch.tensor([-100], dtype=torch.long))

    out = {
      'input_features': input_features,
      'labels': _pad_seq(labels_list, batch_first=True, padding_value=-100),
    }
    if attention_mask is not None:
      out['attention_mask'] = attention_mask

    ids = [b.get('id', '') for b in batch]
    if any(ids):
      out['id'] = ids

    # Pass raw label text strings for NeMo (or any backbone with tokenizer=None)
    label_texts = [b.get('label_text', '') for b in batch]
    if any(label_texts):
      out['label_texts'] = label_texts

    # Per-sample loss weight (from ext_weight flag)
    # Always include weight tensor when ext_weight != 1.0 to ensure consistent
    # keys in model output dict across all batches (melt evaluate requires this).
    weights = [b.get('weight', 1.0) for b in batch]
    ext_weight = getattr(FLAGS, 'ext_weight', 1.0)
    if ext_weight != 1.0 or any(w != 1.0 for w in weights):
      out['weight'] = torch.tensor(weights, dtype=torch.float32)

    # ---- Metadata labels for auxiliary losses (age / domain) ----
    if any('age_bucket' in b for b in batch):
      import re as _re_age
      age_mode = getattr(FLAGS, 'aux_age_mode', 'classify')
      age_labels = []
      age_mask = []
      # 4-class mapping: 3-4=0, 5-7=1, 8-11=2, 12+=3
      # Normalized to [0,1] for regress mode so MSE loss ~O(1), comparable to CE
      _AGE_REGRESS_TARGETS = {0: 0.0, 1: 0.2632, 2: 0.6316, 3: 1.0}
      # Midpoints [3.5, 6.0, 9.5, 13.0] → linear map: (x - 3.5) / (13.0 - 3.5)
      for b in batch:
        ab = b.get('age_bucket', '')
        if not ab or (isinstance(ab, float) and pd.isna(ab)):
          age_labels.append(0.0)
          age_mask.append(0.0)
          continue
        s = str(ab).strip()
        if s in ('3-4',):
          cls = 0
        elif s in ('5-7',):
          cls = 1
        elif s in ('8-11', '8-12'):
          cls = 2
        elif s in ('12+',):
          cls = 3
        else:
          m = _re_age.match(r'(\d+)', s)
          if m:
            n = int(m.group(1))
            if n <= 4: cls = 0
            elif n <= 7: cls = 1
            elif n <= 11: cls = 2
            else: cls = 3
          else:
            age_labels.append(0.0)
            age_mask.append(0.0)
            continue
        if age_mode == 'regress':
          age_labels.append(_AGE_REGRESS_TARGETS[cls])
        else:
          age_labels.append(float(cls))
        age_mask.append(1.0)
      out['age_label'] = torch.tensor(age_labels, dtype=torch.float32)
      out['age_mask'] = torch.tensor(age_mask, dtype=torch.float32)

    if any('domain_label' in b for b in batch):
      domain_labels = [b.get('domain_label', -1) for b in batch]
      domain_mask = [1.0 if dl >= 0 else 0.0 for dl in domain_labels]
      domain_labels = [max(dl, 0) for dl in domain_labels]  # replace -1 with 0 for masked
      out['domain_label'] = torch.tensor(domain_labels, dtype=torch.float32)
      out['domain_mask'] = torch.tensor(domain_mask, dtype=torch.float32)

    # ---- Length prediction labels (nchars / nspaces) ----
    if any('nchars_label' in b for b in batch):
      nchars_vals = [b.get('nchars_label', -1.0) for b in batch]
      nchars_mask = [1.0 if v >= 0 else 0.0 for v in nchars_vals]
      nchars_vals = [max(v, 0.0) for v in nchars_vals]
      out['nchars_label'] = torch.tensor(nchars_vals, dtype=torch.float32)
      out['nchars_mask'] = torch.tensor(nchars_mask, dtype=torch.float32)

    if any('nspaces_label' in b for b in batch):
      nspaces_vals = [b.get('nspaces_label', -1.0) for b in batch]
      nspaces_mask = [1.0 if v >= 0 else 0.0 for v in nspaces_vals]
      nspaces_vals = [max(v, 0.0) for v in nspaces_vals]
      out['nspaces_label'] = torch.tensor(nspaces_vals, dtype=torch.float32)
      out['nspaces_mask'] = torch.tensor(nspaces_mask, dtype=torch.float32)

    # ---- Multi-task labels (IPA + Word) ----
    if any('ipa_labels' in b for b in batch):
      from torch.nn.utils.rnn import pad_sequence
      ipa_labels_list = []
      ipa_mask = []  # 1.0 if sample has IPA label, 0.0 if not
      for b in batch:
        il = b.get('ipa_labels', [])
        if il:
          ipa_labels_list.append(torch.tensor(il, dtype=torch.long))
          ipa_mask.append(1.0)
        else:
          # Placeholder: single -100 token, will be masked in loss
          ipa_labels_list.append(torch.tensor([-100], dtype=torch.long))
          ipa_mask.append(0.0)
      out['ipa_labels'] = pad_sequence(ipa_labels_list, batch_first=True, padding_value=-100)
      out['ipa_mask'] = torch.tensor(ipa_mask, dtype=torch.float32)

    if any('word_labels' in b for b in batch):
      from torch.nn.utils.rnn import pad_sequence
      word_labels_list = []
      word_mask = []
      for b in batch:
        wl = b.get('word_labels', [])
        if wl:
          word_labels_list.append(torch.tensor(wl, dtype=torch.long))
          word_mask.append(1.0)
        else:
          word_labels_list.append(torch.tensor([-100], dtype=torch.long))
          word_mask.append(0.0)
      out['word_labels'] = pad_sequence(word_labels_list, batch_first=True, padding_value=-100)
      out['word_mask'] = torch.tensor(word_mask, dtype=torch.float32)

    # Pass raw label text strings for cross-labels (avoids BPE roundtrip)
    ipa_label_texts = [b.get('ipa_label_text', '') for b in batch]
    if any(ipa_label_texts):
      out['ipa_label_texts'] = ipa_label_texts
    word_label_texts = [b.get('word_label_text', '') for b in batch]
    if any(word_label_texts):
      out['word_label_texts'] = word_label_texts

    return out

  num_workers = int(FLAGS.num_workers or 0)
  kwargs = {
    'num_workers': num_workers,
    'pin_memory': bool(getattr(FLAGS, 'pin_memory', True)),
    'persistent_workers': bool(FLAGS.persistent_workers) and num_workers > 0,
    'collate_fn': collate_fn,
  }

  def _get_sampler(dataset, shuffle=False):
    """Simple sampler that works without lele (for Docker compat)."""
    try:
      return le.get_sampler(dataset, shuffle=shuffle)
    except (AttributeError, ImportError):
      if shuffle:
        return torch.utils.data.RandomSampler(dataset)
      return torch.utils.data.SequentialSampler(dataset)

  if df is None:
    df = preprocess(mode)

  if mode == 'test':
    ds = Dataset(df, mode='test')
    sampler = _get_sampler(ds, shuffle=False)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=gz.eval_batch_size(), sampler=sampler, **kwargs)
    return dl

  train_ds = Dataset(df, mode='train')
  use_temperature_sampler = getattr(FLAGS, 'temperature_sampler', False)
  if use_temperature_sampler:
    assert not getattr(FLAGS, 'stress_test_memory', False), \
      'temperature_sampler is incompatible with stress_test_memory'
    assert not getattr(FLAGS, 'bucket_batch', False), \
      'temperature_sampler is incompatible with bucket_batch for now'
    assert getattr(FLAGS, 'ext_sample_rate', 1.0) >= 1.0, (
      'temperature_sampler cannot be combined with ext_sample_rate < 1. '
      'Use one train exposure control at a time.')
    assert getattr(FLAGS, 'word_only_dd_sample_rate', 1.0) >= 1.0, (
      'temperature_sampler cannot be combined with word_only_dd_sample_rate < 1. '
      'Use one train exposure control at a time.')
    assert getattr(FLAGS, 'word_only_ext_sample_rate', 1.0) >= 1.0, (
      'temperature_sampler cannot be combined with word_only_ext_sample_rate < 1. '
      'Use one train exposure control at a time.')
  if getattr(FLAGS, 'stress_test_memory', False):
    # --- Memory stress test: longest samples first ---
    # Sort by duration descending so the first few batches contain the
    # longest audio (= worst-case memory). If these survive without OOM,
    # the rest of training is safe.  Also consider label length for S2S.
    durations = train_ds.df['audio_duration_sec'].tolist()
    sorted_indices = sorted(range(len(durations)), key=lambda i: -durations[i])
    class _LongestFirstSampler(torch.utils.data.Sampler):
      def __init__(self, indices):
        self.indices = indices
      def __iter__(self):
        return iter(self.indices)
      def __len__(self):
        return len(self.indices)
    sampler = _LongestFirstSampler(sorted_indices)
    logger.info(f'[stress_test_memory] Longest-first batching enabled. '
                f'Top durations: {[durations[i] for i in sorted_indices[:5]]}')
    dl = torch.utils.data.DataLoader(
        train_ds, batch_size=gz.batch_size(), sampler=sampler,
        drop_last=True, **kwargs)
  elif getattr(FLAGS, 'bucket_batch', False):
    bucket_lens, bucket_info = _get_bucket_lens(train_ds)
    bucket_sampler = le.BucketBatchSampler(
        lens=bucket_lens,
        batch_size=gz.batch_size(),
        shuffle=True,
        drop_last=True,
    )
    logger.info(
        f'[bucket_batch] key={bucket_info.pop("key", "audio")} '
        f'batch_size={gz.batch_size()} num_batches={len(bucket_sampler)} '
        + ' '.join(f'{k}={v}' for k, v in bucket_info.items()))
    _log_bucket_sampler_preview(
        bucket_sampler,
        train_ds,
        bucket_lens=bucket_lens,
        bucket_info={'key': getattr(FLAGS, 'bucket_batch_key', 'audio') or 'audio'},
    )
    dl = torch.utils.data.DataLoader(
        train_ds, batch_sampler=bucket_sampler, **kwargs)
  elif use_temperature_sampler:
    group_mode = getattr(FLAGS, 'temperature_sampler_group', 'source')
    group_keys = build_temperature_group_keys(train_ds.df, mode=group_mode)
    epoch_size = getattr(FLAGS, 'temperature_sampler_epoch_size', 0) or len(train_ds)
    sampler_seed = getattr(FLAGS, 'temperature_sampler_seed', 42)
    sampler = TemperatureSampler(
      train_ds,
      group_keys=group_keys,
      alpha=getattr(FLAGS, 'temperature_sampler_alpha', 0.5),
      seed=sampler_seed,
      num_samples=epoch_size,
      distributed=getattr(FLAGS, 'distributed', False),
    )
    sampler_desc = sampler.describe()
    logger.info(
      '[temperature_sampler] '
      f'group={group_mode} alpha={sampler_desc["alpha"]} '
      f'epoch_size={sampler_desc["num_samples"]} '
      f'groups={sampler_desc["groups"]}'
    )
    dl = torch.utils.data.DataLoader(
        train_ds, batch_size=gz.batch_size(), sampler=sampler,
        drop_last=True, **kwargs)
  else:
    sampler = _get_sampler(train_ds, shuffle=True)
    dl = torch.utils.data.DataLoader(
        train_ds, batch_size=gz.batch_size(), sampler=sampler,
        drop_last=True, **kwargs)

  if FLAGS.train_only:
    return dl, None, None

  eval_ds = Dataset(df, mode='eval')
  eval_dl = None
  if len(eval_ds):
    sampler = _get_sampler(eval_ds, shuffle=False)
    eval_dl = torch.utils.data.DataLoader(
        eval_ds, batch_size=gz.eval_batch_size(), sampler=sampler, **kwargs)

  valid_ds = Dataset(df, mode='valid')
  valid_dl = None
  if len(valid_ds):
    sampler = _get_sampler(valid_ds, shuffle=False)
    valid_dl = torch.utils.data.DataLoader(
        valid_ds, batch_size=gz.eval_batch_size(), sampler=sampler, **kwargs)

  # ---------- Sanity check: train / eval ID overlap ----------
  _check_train_eval_overlap(train_ds, eval_ds, valid_ds)

  return dl, eval_dl, valid_dl
