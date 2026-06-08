import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from csrc_qwen.config import AugmentConfig


def build_prefix_messages(prompt: str, audio: object) -> list[dict]:
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio}]},
    ]


def build_prefix_text(processor: object, prompt: str) -> str:
    msgs = build_prefix_messages(prompt, None)
    return processor.apply_chat_template(  # type: ignore[union-attr]
        [msgs], add_generation_prompt=True, tokenize=False,
    )[0]


class Qwen3ASRManifestDataset(Dataset):
    def __init__(self, entries: list[dict], prefix_text: str) -> None:
        self.entries = entries
        self.prefix_text = prefix_text

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        return {
            "audio": entry["audio_filepath"],
            "target": entry["text"],
            "prefix_text": self.prefix_text,
            "source": entry.get("source", "unknown"),
        }


@dataclass
class Qwen3ASRDataCollator:
    processor: Any
    sampling_rate: int = 16000
    augment_cfg: AugmentConfig | None = field(default=None)
    training: bool = True
    _noise_files: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.training and self.augment_cfg and self.augment_cfg.add_noise and self.augment_cfg.add_noise.prob:
            noise_dir = Path(self.augment_cfg.add_noise.noise_dir)
            self._noise_files = sorted(str(p) for p in noise_dir.glob("*.wav"))

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]
        sources = [f.get("source", "unknown") for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets, strict=True)]

        # 1. Speed Perturbation (waveform level)
        speed_rates = self._get_speed_rates(sources)
        audios = [self._load_audio(p, rate=r) for p, r in zip(audio_paths, speed_rates, strict=True)]

        # 1.5. Add background noise (waveform level)
        if self.training and self._noise_files:
            audios = [self._apply_add_noise(a, s) for a, s in zip(audios, sources, strict=True)]

        # 2. Processor (feature extraction + tokenization)
        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        # 3. SpecAugment (spectrogram level, full_inputs only)
        if self.training and self.augment_cfg is not None and self.augment_cfg.spec_augment is not None and self.augment_cfg.spec_augment.prob:
            self._apply_spec_augment(
                full_inputs["input_features"],
                full_inputs["feature_attention_mask"],
                sources,
            )

        # 4. Label masking
        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        full_pad_lens = (full_inputs["attention_mask"] == 0).sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, (pl, fpl) in enumerate(zip(prefix_lens, full_pad_lens)):
            labels[i, : fpl + pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs

    def _get_speed_rates(self, sources: list[str]) -> list[float]:
        if not self.training or self.augment_cfg is None or self.augment_cfg.speed_perturb is None or not self.augment_cfg.speed_perturb.prob:
            return [1.0] * len(sources)
        sp = self.augment_cfg.speed_perturb
        rates = []
        for source in sources:
            prob = sp.prob.get(source, 0.0)
            if random.random() < prob:
                rates.append(random.choice(sp.rates))
            else:
                rates.append(1.0)
        return rates

    def _apply_spec_augment(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        sources: list[str],
    ) -> None:
        sa = self.augment_cfg.spec_augment  # type: ignore[union-attr]
        n_mels = input_features.shape[1]
        for i, source in enumerate(sources):
            if random.random() >= sa.prob.get(source, 0.0):
                continue
            t_len = int(feature_attention_mask[i].sum().item())
            max_tw = max(1, int(t_len * sa.time_width))
            # Frequency masking
            for _ in range(sa.freq_masks):
                f = random.randint(0, sa.freq_width)
                f0 = random.randint(0, max(0, n_mels - f))
                input_features[i, f0 : f0 + f, :] = 0.0
            # Time masking
            for _ in range(sa.time_masks):
                t = random.randint(0, max_tw)
                t0 = random.randint(0, max(0, t_len - t))
                input_features[i, :, t0 : t0 + t] = 0.0

    def _apply_add_noise(self, wav: np.ndarray, source: str) -> np.ndarray:
        cfg = self.augment_cfg.add_noise  # type: ignore[union-attr]
        if random.random() >= cfg.prob.get(source, 0.0):
            return wav
        noise, _ = librosa.load(random.choice(self._noise_files), sr=self.sampling_rate, mono=True)
        if len(noise) < len(wav):
            noise = np.tile(noise, (len(wav) // len(noise)) + 1)
        noise = noise[: len(wav)]
        signal_power = np.mean(wav**2)
        noise_power = np.mean(noise**2)
        if noise_power == 0:
            return wav
        snr_db = random.uniform(cfg.min_snr_db, cfg.max_snr_db)
        scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
        return wav + scale * noise

    def _load_audio(self, path: str, rate: float = 1.0) -> np.ndarray:
        wav, _ = librosa.load(path, sr=self.sampling_rate, mono=True)
        if rate != 1.0:
            wav = librosa.resample(wav, orig_sr=int(self.sampling_rate * rate), target_sr=self.sampling_rate)
        return wav
