from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.get_json import _resolve_audio_path, get_data_from_json
class Wav2Vec2TextDataset(Dataset):
    """Generic waveform + text dataset for CTC with configurable text field."""

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer,
        data: List[Dict[str, Any]],
        text_key: str,
        inference: bool = False,
    ) -> None:
        self.audio_folder = Path(cfg.data.audio_folder).resolve()
        self.project_root = Path(__file__).resolve().parents[2]
        self.target_sample_rate = int(cfg.preprocessing.get("sample_rate", 16000))
        self.max_duration = float(cfg.preprocessing.get("max_duration_sec", 50.0))

        self.tokenizer = tokenizer
        self.text_key = text_key
        self.inference = inference

        self.data = [item for item in data if float(item.get("audio_duration_sec", 0) or 0) <= self.max_duration]

        if not inference:
            self.data = self._filter_invalid_ctc_samples(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __call__(self, text: str):
        return self.tokenizer(text)

    def _filter_invalid_ctc_samples(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        removed = 0
        for item in data:
            duration_sec = float(item.get("audio_duration_sec", 0) or 0)
            input_len = int(duration_sec * self.target_sample_rate)
            text = str(item.get(self.text_key, "") or "")
            label_len = len(self.tokenizer(text))
            # Wav2Vec2 feature extractor has an effective stride of ~320 samples.
            output_timesteps = input_len // 320
            if output_timesteps > label_len and label_len > 0 and input_len > 0:
                filtered.append(item)
            else:
                removed += 1
        print(
            f"Removed {removed} non-valid samples from Wav2Vec2TextDataset[{self.text_key}] "
            f"({len(data)} -> {len(filtered)})."
        )
        return filtered

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        audio_path = _resolve_audio_path(item["audio_path"], self.audio_folder, self.project_root)
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception:
            return None

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sample_rate)

        waveform = waveform / (waveform.abs().max() + 1e-8)
        waveform = waveform.squeeze(0)

        transcript_text = ""
        labels: List[int] = []
        if not self.inference:
            transcript_text = str(item.get(self.text_key, "") or "")
            labels = self.tokenizer(transcript_text)

        return {
            "waveform": waveform,
            "labels": torch.tensor(labels, dtype=torch.long),
            "transcript_text": transcript_text,
            "input_length": waveform.shape[0],
            "target_length": len(labels),
            "utterance_id": item.get("utterance_id", ""),
            "child_id": item.get("child_id", "unknown"),
            "age_bucket": item.get("age_bucket", "unknown"),
        }


class Wav2Vec2TextDataCollatorCTC:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [f for f in features if f is not None]
        if not features:
            return None

        waveforms = [f["waveform"] for f in features]
        labels = [f["labels"] for f in features]
        input_lengths = [f["input_length"] for f in features]
        target_lengths = [f["target_length"] for f in features]

        padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)

        attention_mask = torch.zeros(padded_waveforms.shape, dtype=torch.long)
        for i, length in enumerate(input_lengths):
            attention_mask[i, :length] = 1

        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.pad_token_id)

        return {
            "input_features": padded_waveforms,
            "attention_mask": attention_mask,
            "labels": padded_labels,
            "transcript_text": [f["transcript_text"] for f in features],
            "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
            "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
            "utterance_ids": [f["utterance_id"] for f in features],
            "child_ids": [f["child_id"] for f in features],
            "age_buckets": [f["age_bucket"] for f in features],
        }


def _load_word_jsonl_rows(cfg: DictConfig) -> List[Dict[str, Any]]:
    paths: List[str] = [str(cfg.data.word_train_jsonl)]
    tb_path = cfg.data.get("word_train_jsonl_talkbank", None)
    if tb_path:
        paths.append(str(tb_path))

    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()

    audio_dir = Path(cfg.data.audio_folder).resolve()
    project_root = Path(__file__).resolve().parents[2]

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                item = json.loads(raw)
                utt = item.get("utterance_id")
                if not utt or utt in seen:
                    continue
                resolved = _resolve_audio_path(item.get("audio_path", ""), audio_dir, project_root)
                if not resolved.exists():
                    continue
                seen.add(utt)
                rows.append(item)
    return rows


def _split_word_rows_by_val_children(
    rows: List[Dict[str, Any]],
    val_child_ids: set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_rows = [r for r in rows if r.get("child_id") not in val_child_ids]
    val_rows = [r for r in rows if r.get("child_id") in val_child_ids]
    return train_rows, val_rows


def prepare_mtl_datasets(
    cfg: DictConfig,
    fold: int,
    inference: bool = False,
) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
    """Prepare separate word/phonetic loaders for MTL training."""

    phon_all_data = get_data_from_json(cfg, inference=inference)

    if inference:
        phon_train, phon_val = phon_all_data, []
    else:
        phon_train, phon_val = instantiate(cfg.cv.splitter)(all_data=phon_all_data, fold=fold)

    val_child_ids = {row.get("child_id") for row in phon_val}

    word_all_rows = _load_word_jsonl_rows(cfg)
    if inference:
        word_train, word_val = word_all_rows, []
    else:
        word_train, word_val = _split_word_rows_by_val_children(word_all_rows, val_child_ids)
        if len(word_val) == 0 and len(word_train) > 0:
            # Fallback to a regular child-group split when no children overlap.
            word_train, word_val = instantiate(cfg.cv.splitter)(all_data=word_all_rows, fold=fold)

    if cfg.get("debug", False):
        phon_train = phon_train[:2000]
        phon_val = phon_val[:400]
        word_train = word_train[:2000]
        word_val = word_val[:400]

    phon_tokenizer = instantiate(cfg.tokenizer)
    word_tokenizer = instantiate(cfg.mtl.word_tokenizer)

    min_duration = float(cfg.preprocessing.min_duration_sec)
    phon_train = [x for x in phon_train if float(x.get("audio_duration_sec", 0) or 0) >= min_duration]
    word_train = [x for x in word_train if float(x.get("audio_duration_sec", 0) or 0) >= min_duration]

    phon_train_ds = Wav2Vec2TextDataset(cfg=cfg, tokenizer=phon_tokenizer, data=phon_train, text_key="phonetic_text", inference=inference)
    phon_val_ds = Wav2Vec2TextDataset(cfg=cfg, tokenizer=phon_tokenizer, data=phon_val, text_key="phonetic_text", inference=inference)

    word_train_ds = Wav2Vec2TextDataset(cfg=cfg, tokenizer=word_tokenizer, data=word_train, text_key="orthographic_text", inference=inference)
    word_val_ds = Wav2Vec2TextDataset(cfg=cfg, tokenizer=word_tokenizer, data=word_val, text_key="orthographic_text", inference=inference)

    dl_cfg = cfg.training.dataloader
    num_workers = int(dl_cfg.num_workers)
    pin_memory = bool(dl_cfg.pin_memory)
    prefetch_factor = dl_cfg.get("prefetch_factor")
    persistent_workers = num_workers > 0

    sampler_cfg = cfg.get("sampler")
    if isinstance(sampler_cfg, DictConfig) and ("train" in sampler_cfg or "val" in sampler_cfg):
        train_sampler_cfg = sampler_cfg.get("train")
        val_sampler_cfg = sampler_cfg.get("val") or train_sampler_cfg
    else:
        train_sampler_cfg = sampler_cfg
        val_sampler_cfg = sampler_cfg

    def make_loader(dataset, collator, sampler_conf):
        sampler = None
        if sampler_conf is not None:
            sampler = instantiate(sampler_conf, durations=[x["audio_duration_sec"] for x in dataset.data])
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    phon_collator = Wav2Vec2TextDataCollatorCTC(pad_token_id=phon_tokenizer.pad_token_id)
    word_collator = Wav2Vec2TextDataCollatorCTC(pad_token_id=word_tokenizer.pad_token_id)

    loaders = {
        "phon_train": make_loader(phon_train_ds, phon_collator, train_sampler_cfg),
        "phon_val": make_loader(phon_val_ds, phon_collator, val_sampler_cfg),
        "word_train": make_loader(word_train_ds, word_collator, train_sampler_cfg),
        "word_val": make_loader(word_val_ds, word_collator, val_sampler_cfg),
    }

    dataset_info = {
        "phon_tokenizer": phon_tokenizer,
        "word_tokenizer": word_tokenizer,
        "phon_vocab_size": phon_tokenizer.vocab_size,
        "word_vocab_size": word_tokenizer.vocab_size,
        "phon_blank_token_id": phon_tokenizer.blank_token_id,
        "word_blank_token_id": word_tokenizer.blank_token_id,
        "phon_pad_token_id": phon_tokenizer.pad_token_id,
        "word_pad_token_id": word_tokenizer.pad_token_id,
        "phon_train_size": len(phon_train_ds),
        "word_train_size": len(word_train_ds),
        "phon_val_size": len(phon_val_ds),
        "word_val_size": len(word_val_ds),
    }

    print(
        "MTL datasets prepared | "
        f"phon_train={len(phon_train_ds)} word_train={len(word_train_ds)} "
        f"phon_val={len(phon_val_ds)} word_val={len(word_val_ds)}"
    )

    return loaders, dataset_info
