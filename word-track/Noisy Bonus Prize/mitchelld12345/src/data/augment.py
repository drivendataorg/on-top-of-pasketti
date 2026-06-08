"""Audio augmentation pipeline using audiomentations."""
from audiomentations import (
    AddBackgroundNoise,
    Compose,
    Gain,
    PitchShift,
    TimeStretch,
)

from src.paths import NOISE_DIR, TARGET_SR


def build_augment(cfg):
    transforms = []

    if cfg.get("noise", True):
        noise_cfg = cfg.get("noise_cfg", {})
        transforms.append(AddBackgroundNoise(
            sounds_path=str(NOISE_DIR),
            min_snr_db=noise_cfg.get("min_snr_db", 0),
            max_snr_db=noise_cfg.get("max_snr_db", 40),
            p=noise_cfg.get("p", 0.5),
        ))

    if cfg.get("gain", True):
        gain_cfg = cfg.get("gain_cfg", {})
        transforms.append(Gain(
            min_gain_db=gain_cfg.get("min_db", -6),
            max_gain_db=gain_cfg.get("max_db", 6),
            p=gain_cfg.get("p", 0.5),
        ))

    if cfg.get("time_stretch", True):
        ts_cfg = cfg.get("time_stretch_cfg", {})
        transforms.append(TimeStretch(
            min_rate=ts_cfg.get("min_rate", 0.9),
            max_rate=ts_cfg.get("max_rate", 1.1),
            p=ts_cfg.get("p", 0.3),
        ))

    if cfg.get("pitch_shift", False):
        ps_cfg = cfg.get("pitch_shift_cfg", {})
        transforms.append(PitchShift(
            min_semitones=ps_cfg.get("min_semitones", -2),
            max_semitones=ps_cfg.get("max_semitones", 2),
            p=ps_cfg.get("p", 0.3),
        ))

    if not transforms:
        return None

    return Compose(transforms)
