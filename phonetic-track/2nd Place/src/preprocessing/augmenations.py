"""
Augmentation classes for child-speech phonetic transcription with Wav2Vec2.

This module wraps `audiomentations` to bridge the adult→child domain gap, 
providing high-quality, artifact-free DSP transformations while maintaining 
a pure PyTorch Tensor input/output interface for DataLoaders.
"""

from __future__ import annotations

import torch
import numpy as np
from audiomentations import Compose, TimeStretch, PitchShift, BandStopFilter, AddGaussianNoise, AddBackgroundNoise

# ---------------------------------------------------------------------------
# WaveformAugmentor (PyTorch <-> Audiomentations Bridge)
# ---------------------------------------------------------------------------

class WaveformAugmentor:
    """
    Chains multiple waveform-level augmentations using `audiomentations`.
    Accepts PyTorch Tensors, processes them efficiently in Numpy, and 
    returns PyTorch Tensors.

    Usage:
        augmentor = WaveformAugmentor.from_config(cfg.augmentation)
        waveform_tensor = augmentor(waveform_tensor)
    """

    def __init__(self, compose_transform: Compose, sample_rate: int = 16_000):
        self.transform = compose_transform
        self.sample_rate = sample_rate

    @classmethod
    def from_config(cls, aug_cfg, sample_rate: int = 16_000) -> "WaveformAugmentor":
        """Build from an OmegaConf augmentation sub-config."""
        from omegaconf import OmegaConf

        # Handle the "dummy" / disabled case
        if aug_cfg is None or aug_cfg == "dummy":
            return cls(Compose([]), sample_rate)

        # Materialise to a plain dict so .get() works cleanly
        if not isinstance(aug_cfg, dict):
            aug_cfg = OmegaConf.to_container(aug_cfg, resolve=True)

        transforms = []

        # --- 1. Time Stretch (Replaces VTLN for vocal tract/speed adjustment) ---
        ts_cfg = aug_cfg.get("time_stretch", {})
        if ts_cfg.get("enabled", False):
            transforms.append(
                TimeStretch(
                    min_rate=ts_cfg.get("min_rate", 0.85),
                    max_rate=ts_cfg.get("max_rate", 1.15),
                    leave_length_unchanged=False,
                    p=ts_cfg.get("p", 0.5),
                )
            )

        # --- 2. Pitch Shift ---
        ps_cfg = aug_cfg.get("pitch_shift", {})
        if ps_cfg.get("enabled", False):
            transforms.append(
                PitchShift(
                    min_semitones=ps_cfg.get("min_semitones", -4.0),
                    max_semitones=ps_cfg.get("max_semitones", 2.0),
                    p=ps_cfg.get("p", 0.5),
                )
            )

        # --- 3. Frequency Mask (Waveform-level SpecAugment) ---
        bsf_cfg = aug_cfg.get("band_stop_filter", {})
        if bsf_cfg.get("enabled", False):
            transforms.append(
                BandStopFilter(
                    min_center_freq=bsf_cfg.get("min_center_freq", 200.0),
                    max_center_freq=bsf_cfg.get("max_center_freq", 4000.0),
                    min_bandwidth_fraction=bsf_cfg.get("min_bandwidth_fraction", 0.5),
                    max_bandwidth_fraction=bsf_cfg.get("max_bandwidth_fraction", 1.99),
                    p=bsf_cfg.get("p", 0.5),
                )
            )

        # --- 4. Noise Injection ---
        wnoise_cfg = aug_cfg.get("white_noise", {})
        if wnoise_cfg.get("enabled", False):
            transforms.append(
                AddGaussianNoise(
                    min_amplitude=wnoise_cfg.get("min_amplitude", 0.001),
                    max_amplitude=wnoise_cfg.get("max_amplitude", 0.015),
                    p=wnoise_cfg.get("p", 0.5),
                )
            )
        bn_cfg = aug_cfg.get("background_noise", {})
        if bn_cfg.get("enabled", False):
            transforms.append(
                AddBackgroundNoise(
                    sounds_path=bn_cfg.get("noise_dir", "data/noise/audio"),
                    min_snr_db=bn_cfg.get("snr_min", 5),
                    max_snr_db=bn_cfg.get("snr_max", 10),
                    p=bn_cfg.get("p", 0.5),
                )
            )

        return cls(Compose(transforms), sample_rate)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # If no augmentations are enabled, return the original tensor immediately
        if not self.transform.transforms:
            return waveform

        # 1. Handle PyTorch shapes and convert to 1D Numpy
        is_2d = waveform.dim() == 2
        wav_np = waveform.squeeze(0).numpy() if is_2d else waveform.numpy()

        # 2. Apply the highly optimized audiomentations pipeline
        augmented_np = self.transform(samples=wav_np, sample_rate=self.sample_rate)

        # 3. Convert back to PyTorch Tensor
        augmented_tensor = torch.from_numpy(augmented_np)

        # 4. Re-normalize to prevent clipping from DSP math
        peak = augmented_tensor.abs().max()
        if peak > 1.0:
            augmented_tensor = augmented_tensor / (peak + 1e-8)

        # 5. Restore original 2D shape [1, T] if necessary
        if is_2d:
            augmented_tensor = augmented_tensor.unsqueeze(0)

        return augmented_tensor

    def __repr__(self) -> str:
        lines = ["WaveformAugmentor(["]
        for t in self.transform.transforms:
            lines.append(f"  {t.__class__.__name__}(p={t.p}),")
        lines.append("])")
        return "\n".join(lines)