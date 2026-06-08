import json
import os
import random
import torch
import torchaudio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Dict, Any, Tuple
from omegaconf import DictConfig, OmegaConf
from transformers import WhisperFeatureExtractor
from hydra.utils import instantiate

from src.preprocessing.phoneme_tokenizer import PhonemeTokenizer
from src.preprocessing.get_json import get_data_from_json
from src.preprocessing.augmenations import WaveformAugmentor
from src.utils.hf_local import resolve_hf_load_path


def is_valid_ctc_sample(input_len: int, label_len: int, downsample: int = 320) -> bool:
    output_timesteps = input_len // downsample
    return output_timesteps > label_len and label_len > 0 and input_len > 0


def _filter_invalid_ctc_samples(
    data: List[Dict[str, Any]],
    tokenizer: PhonemeTokenizer,
    sample_rate: int,
    dataset_name: str,
    downsample: int = 320,
) -> List[Dict[str, Any]]:
    total = len(data)
    if total == 0:
        print(f"[CTC] Removed 0 non-valid samples from {dataset_name} (0 -> 0).")
        return data

    valid_items: List[Dict[str, Any]] = []
    removed = 0
    for item in data:
        duration_sec = item.get("audio_duration_sec") or 0
        if duration_sec < 0:
            duration_sec = 0
        input_len = int(duration_sec * sample_rate)
        phonetic_text = item.get("phonetic_text") or ""
        label_len = len(tokenizer(phonetic_text))
        if is_valid_ctc_sample(input_len, label_len, downsample=downsample):
            valid_items.append(item)
        else:
            removed += 1

    print(
        f"Removed {removed} non-valid samples from {dataset_name} "
        f"({total} -> {len(valid_items)})."
    )
    return valid_items

class DurationBatchSampler(Sampler):
    """
    Groups samples so that the TOTAL duration of audio in each batch
    does not exceed `max_batch_seconds`.
    """

    def __init__(
        self,
        durations: List[float],
        max_batch_seconds: float,
        bucket_size: int = 100,  # Size of the chunk for local shuffling
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.durations = durations
        self.max_batch_seconds = max_batch_seconds
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # We keep the original indices and durations handy
        self.sorted_indices = np.argsort(self.durations)[::-1].tolist()

        # Initialize batches once in case __len__ is called before __iter__
        self.batches = self._form_batches()

    def _form_batches(self) -> List[List[int]]:
        batches = []

        # Iterate through the sorted data in chunks (buckets)
        # to allow for local shuffling of similar-length samples
        for bucket_start in range(0, len(self.sorted_indices), self.bucket_size):
            bucket = self.sorted_indices[bucket_start : bucket_start + self.bucket_size]

            if self.shuffle:
                random.shuffle(bucket)

            # Greedily pack samples into batches based on seconds
            current_batch = []
            current_batch_sec = 0.0

            for idx in bucket:
                sample_sec = self.durations[idx]

                # If adding this sample exceeds the limit, save the current batch and start a new one
                if current_batch_sec + sample_sec > self.max_batch_seconds and len(current_batch) > 0:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_sec = 0.0

                current_batch.append(idx)
                current_batch_sec += sample_sec

            # Add leftover samples from the bucket
            if current_batch:
                batches.append(current_batch)

        if self.drop_last and len(batches) > 0:
            batches.pop()

        return batches

    def __iter__(self):
        # We re-form the batches every epoch!
        # Because we locally shuffle the buckets, reforming batches dynamically
        # ensures the exact groupings are slightly different every epoch.
        if self.shuffle:
            self.batches = self._form_batches()

        # Shuffle the order in which the batches are yielded
        batch_order = list(range(len(self.batches)))
        if self.shuffle:
            random.shuffle(batch_order)

        for idx in batch_order:
            yield self.batches[idx]

    def __len__(self):
        return len(self.batches)


class BucketBatchSampler(Sampler):
    """
    Groups samples by audio duration so each batch has similar-length sequences,
    minimizing wasted padding. Buckets are shuffled each epoch for randomness.

    Algorithm:
      1. Sort all indices by audio_duration_sec
      2. Chunk sorted indices into buckets of size (batch_size * bucket_multiplier)
      3. Within each bucket, create batches of batch_size
      4. Shuffle the order of batches each epoch (but samples within a batch
         stay similar in length)
    """

    def __init__(
        self,
        durations: List[float],
        batch_size: int,
        bucket_multiplier: int = 10,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Sort indices by duration (longest first)
        sorted_indices = np.argsort(durations)[::-1]

        # Create buckets: groups of (batch_size * bucket_multiplier) similar-length samples
        bucket_size = batch_size * bucket_multiplier
        self.batches: List[List[int]] = []

        for bucket_start in range(0, len(sorted_indices), bucket_size):
            bucket = sorted_indices[bucket_start : bucket_start + bucket_size].tolist()

            # Shuffle within each bucket for minor randomness
            if self.shuffle:
                random.shuffle(bucket)

            # Split bucket into batches
            for i in range(0, len(bucket), batch_size):
                batch = bucket[i : i + batch_size]
                if len(batch) == batch_size or not self.drop_last:
                    self.batches.append(batch)

    def __iter__(self):
        # Shuffle batch order each epoch
        batch_order = list(range(len(self.batches)))
        if self.shuffle:
            random.shuffle(batch_order)
        for idx in batch_order:
            yield self.batches[idx]

    def __len__(self):
        return len(self.batches)





class WhisperDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: PhonemeTokenizer,
        data: List[Dict] = None, # <,------ new
        inference: bool = False,
        apply_augmentation: bool = True,
    ):
        self.cfg = cfg
        self.audio_folder = Path(cfg.data.audio_folder).resolve()
        self.project_root = Path(__file__).resolve().parents[2]
        self.target_sample_rate = cfg.preprocessing.get("sample_rate", 16000)
        self.max_duration = cfg.preprocessing.get("max_duration_sec", 30.0)
        
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        
        raw_items = data

        # Filter out anything longer than max_duration
        self.data = []
        for item in raw_items:
            if item.get("audio_duration_sec", 0) <= self.max_duration:
                self.data.append(item)

        if not inference:
            self.data = _filter_invalid_ctc_samples(
                data=self.data,
                tokenizer=self.tokenizer,
                sample_rate=self.target_sample_rate,
                dataset_name="WhisperDataset",
            )

        self.inference = inference
        self.apply_augmentation = apply_augmentation and not inference
        if self.apply_augmentation:
            self.augmentor = WaveformAugmentor.from_config(
                cfg.get("augmentation", None),
                sample_rate=self.target_sample_rate,
            )
            print(f"Waveform augmentations: {self.augmentor}")
        else:
            self.augmentor = WaveformAugmentor.from_config(None, sample_rate=self.target_sample_rate)
                        
    def __len__(self):
        return len(self.data)

    def _resolve_audio_path(self, item: Dict[str, Any]) -> Path:
        rel_path = Path(item["audio_path"])
        candidates = [
            self.audio_folder / rel_path.name,
            self.audio_folder / rel_path,
            self.project_root / "data" / rel_path,
            self.project_root / rel_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # --- 1. Audio Processing ---
        audio_path = self._resolve_audio_path(item)
        waveform, sr = torchaudio.load(audio_path)
        
        # make all mono or stereo
        if self.cfg.preprocessing.sound_dimension == "mono" and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # resample to target_sample_rate
        if sr != self.target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sample_rate)

        waveform = waveform / (waveform.abs().max() + 1e-8)

        if self.apply_augmentation:
            waveform = self.augmentor(waveform)
    
        features = self.feature_extractor(
            waveform.squeeze().numpy(), 
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        input_features = features.input_features.squeeze(0)
        
        # --- 2. Text Processing ---
        if self.inference:
            labels = []  # No labels during inference
            phonetic_text = ""
        else:
            phonetic_text = item["phonetic_text"]
            labels = self.tokenizer(phonetic_text) 
        
        # --- 3. Lengths for CTC Loss ---
        input_length = input_features.shape[-1]
        target_length = len(labels)
        
        return {
            "input_features": input_features,
            "labels": torch.tensor(labels, dtype=torch.long),
            "phonetic_text": phonetic_text,
            "input_length": input_length,
            "target_length": target_length,
            "utterance_id": item["utterance_id"],
            "child_id": item.get("child_id", "unknown"),
            "age_bucket": item.get("age_bucket", "unknown"),
        }


class DataCollatorCTCWithPadding:
    def __init__(self, feature_extractor: WhisperFeatureExtractor, pad_token_id: int):
        self.feature_extractor = feature_extractor
        self.pad_token_id = pad_token_id
        self.age_bucket_map = {
            "3-4": 0,
            "5-7": 1,
            "8-11": 2,
            "12+": 3,
            "unknown": -100
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [feature["input_features"] for feature in features]
        labels = [feature["labels"] for feature in features]
        input_lengths = [feature["input_length"] for feature in features]
        target_lengths = [feature["target_length"] for feature in features]
        phonetic_texts = [f["phonetic_text"] for f in features]
        age_labels = [self.age_bucket_map.get(f.get("age_bucket", "unknown"), -100) for f in features]
        age_buckets = [f.get("age_bucket", "unknown") for f in features]

        # --- Pad Audio Features (Mel Spectrograms) ---
        # input_features are [80, L]. pad_sequence expects [L, *], so we transpose, pad, and transpose back.
        features_transposed = [f.transpose(0, 1) for f in input_features]
        
        input_features_padded = torch.nn.utils.rnn.pad_sequence(
            features_transposed, 
            batch_first=True, 
            padding_value=0.0  # Whisper padding value for log-Mel is 0.0
        ).transpose(1, 2)      # Result shape: [Batch, 80, max_L_in_batch]

        # --- Pad Labels (Phoneme IDs) ---
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=self.pad_token_id
        )

        return {
            "input_features": input_features_padded,
            "labels": labels_padded,
            "phonetic_text": phonetic_texts,
            "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
            "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
            "utterance_ids": [feature["utterance_id"] for feature in features],
            "child_ids": [feature["child_id"] for feature in features],
            "age_buckets": age_buckets,
            "age_labels": torch.tensor(age_labels, dtype=torch.long),
        }

class Wav2Vec2Dataset(Dataset):
    """
    PyTorch Dataset for Wav2Vec2-based phonetic transcription.

    Returns raw waveforms (resampled to 16 kHz, mono, amplitude-normalised).
    Padding is handled at the collation stage.

    Waveform-level augmentations (TimeStretch, PitchShift, BandStopFilter) are applied here in
    ``__getitem__`` during training only.  Feature-level augmentations
    (SpecAugment) are applied in the model forward pass.
    """

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: "PhonemeTokenizer",
        data: List[Dict] = None,
        inference: bool = False,
        apply_augmentation: bool = True,
    ):
        self.audio_folder = Path(cfg.data.audio_folder).resolve()
        self.project_root = Path(__file__).resolve().parents[2]
        self.target_sample_rate = cfg.preprocessing.get("sample_rate", 16000)
        self.max_duration = cfg.preprocessing.get("max_duration_sec", 30.0)
        self.tokenizer = tokenizer
        self.data = [
            item for item in data
            if item.get("audio_duration_sec", 0) <= self.max_duration
        ]
        if not inference:
            self.data = _filter_invalid_ctc_samples(
                data=self.data,
                tokenizer=self.tokenizer,
                sample_rate=self.target_sample_rate,
                dataset_name="Wav2Vec2Dataset",
            )
        self.inference = inference
        self.apply_augmentation = apply_augmentation and not inference

        # Waveform-level augmentations — training only when enabled
        if self.apply_augmentation:
            self.augmentor = WaveformAugmentor.from_config(
                cfg.get("augmentation", None)
            )
            print(f"Waveform augmentations: {self.augmentor}")
        else:
            self.augmentor = WaveformAugmentor.from_config(None)

        # Load preprocessed waveforms into RAM if a cache directory is configured
        cache_dir_str = cfg.data.get("audio_processed_folder", None)
        cache_path = Path(cache_dir_str).resolve() if cache_dir_str else None
        if cache_path and (cache_path / "manifest.json").exists():
            print(f"Loading {len(self.data)} preprocessed waveforms into RAM from {cache_path} ...")
            self._waveform_cache = self._load_cache(cache_path)
            hit_rate = len(self._waveform_cache) / len(self.data) * 100 if self.data else 0
            print(f"  Cached {len(self._waveform_cache)}/{len(self.data)} waveforms ({hit_rate:.1f}% hit rate).")
        else:
            self._waveform_cache = None
            if cache_dir_str:
                print(f"[WARNING] audio_processed_folder '{cache_dir_str}' not found or missing manifest — falling back to on-disk loading.")

    def _load_cache(self, cache_path: Path) -> Dict[str, np.ndarray]:
        """Load all waveforms for this dataset split into a dict in RAM using threads."""
        def _load_one(uid: str):
            p = cache_path / f"{uid}.npy"
            return uid, np.load(p) if p.exists() else None

        uids = [item["utterance_id"] for item in self.data]
        n_workers = min(os.cpu_count() or 4, 16)
        cache: Dict[str, np.ndarray] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for uid, arr in executor.map(_load_one, uids):
                if arr is not None:
                    cache[uid] = arr
        return cache

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_audio_path(self, item: Dict[str, Any]) -> Path:
        rel_path = Path(item["audio_path"])
        candidates = [
            self.audio_folder / rel_path.name,
            self.audio_folder / rel_path,
            self.project_root / "data" / rel_path,
            self.project_root / rel_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        uid = item["utterance_id"]

        if self._waveform_cache is not None and uid in self._waveform_cache:
            # Zero-copy: torch.from_numpy shares memory with the numpy array
            waveform = torch.from_numpy(self._waveform_cache[uid])
        else:
            audio_path = self._resolve_audio_path(item)
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.target_sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.target_sample_rate)
            waveform = waveform / (waveform.abs().max() + 1e-8)
            waveform = waveform.squeeze(0)  # [T]

        # Apply waveform-level augmentations (training only)
        if self.apply_augmentation:
            waveform = self.augmentor(waveform)

        if self.inference:
            labels = []  # No labels during inference
            phonetic_text = ""  # No phonetic text during inference
        else:   
            phonetic_text = item["phonetic_text"]
            labels = self.tokenizer(phonetic_text)

        return {
            "waveform": waveform,
            "labels": torch.tensor(labels, dtype=torch.long),
            "phonetic_text": phonetic_text,
            "input_length": waveform.shape[0],
            "target_length": len(labels),
            "utterance_id": item["utterance_id"],
            "child_id": item.get("child_id", "unknown"),
            "age_bucket": item.get("age_bucket", "unknown")
        }


class Wav2Vec2DataCollatorCTC:
    """Pads raw waveforms and creates attention masks for Wav2Vec2 CTC training."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
        self.age_bucket_map = {
            "3-4": 0,
            "5-7": 1,
            "8-11": 2,
            "12+": 3,
            "unknown": -100
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        waveforms = [f["waveform"] for f in features]
        labels = [f["labels"] for f in features]
        input_lengths = [f["input_length"] for f in features]
        target_lengths = [f["target_length"] for f in features]
        phonetic_texts = [f["phonetic_text"] for f in features]
        age_labels = [self.age_bucket_map.get(f.get("age_bucket", "unknown"), -100) for f in features]
        age_buckets = [f.get("age_bucket", "unknown") for f in features]

        # Pad waveforms to max length in batch
        padded_waveforms = torch.nn.utils.rnn.pad_sequence(
            waveforms, batch_first=True, padding_value=0.0
        )

        # Attention mask: 1 = real sample, 0 = padding
        attention_mask = torch.zeros(padded_waveforms.shape, dtype=torch.long)
        for i, length in enumerate(input_lengths):
            attention_mask[i, :length] = 1

        # Pad labels
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.pad_token_id
        )

        return {
            "input_features": padded_waveforms,
            "attention_mask": attention_mask,
            "labels": padded_labels,
            "phonetic_text": phonetic_texts,
            "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
            "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
            "utterance_ids": [f["utterance_id"] for f in features],
            "child_ids": [f.get("child_id", "unknown") for f in features],
            "age_buckets": age_buckets,
            "age_labels": torch.tensor(age_labels, dtype=torch.long),
        }

def prepare_dl_dataset(
    cfg: DictConfig,
    fold: int = 0,
    inference: bool = False,
    data_override: List[Dict[str, Any]] | None = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Orchestrates data loading, splitting, tokenization, and DataLoader creation.
    """
    import json
    from pathlib import Path
    
    # 1. Load all raw data from disk
    all_data = data_override if data_override is not None else get_data_from_json(cfg, inference=inference)
                
    # 2. Perform the GroupKFold split (ensuring no child_id overlap)
    print(f"Splitting data for fold {fold+1}...")
   
    if inference:
        train_data = all_data
        val_data = []
    elif cfg.data.get("train_on_all_data", False):
        print("train_on_all_data=True: using ALL data for training, no validation split.")
        train_data = all_data
        val_data = []
    else:
        train_data, val_data = instantiate(
            cfg.cv.splitter,
            # all_data=all_data,
            # fold=fold
        )(all_data=all_data, fold=fold)

        if cfg.data.get("remove_bad_data", False):
            bad_data_path = cfg.data.get("removed_utterances_jsonl")
            if bad_data_path and Path(bad_data_path).exists():
                print(f"Removing bad data based on {bad_data_path}")
                removed_ids = set()
                with open(bad_data_path) as f:
                    for line in f:
                        if line.strip():
                            removed_ids.add(json.loads(line)["utterance_id"])
                
                original_len = len(train_data)
                train_data = [item for item in train_data if item["utterance_id"] not in removed_ids]
                print(f"Filtered {original_len - len(train_data)} bad utterances from train_data.")

        # 

    # Debug mode: limit training data to first 2000 samples
    if cfg.get("debug", False):
        max_debug_samples = 2000
        train_data = train_data[:max_debug_samples]
        val_data = val_data[:max_debug_samples // 5]  # smaller val set for faster debugging
        print(f"[DEBUG MODE] Limiting training data to {len(train_data)} samples.")

    # 3. Setup Tokenizer  <-- does not per se need to be built every fold, but oh well
    print("Initializing Phoneme Tokenizer")
    tokenizer = instantiate(cfg.tokenizer)
    
    # 4-6. Setup datasets and collator (dispatched by model backend)
    model_target = cfg.model.get("_target_", "")

    if "Wav2Vec2" in model_target:
        train_data = [ 
                item for item in train_data  
                if item.get("audio_duration_sec", 1.0) >= cfg.preprocessing.min_duration_sec
            ]
        print("Initializing Wav2Vec2 datasets...")
        train_dataset = Wav2Vec2Dataset(
            cfg=cfg,
            tokenizer=tokenizer,
            data=train_data,
            inference=inference,
            apply_augmentation=not inference,
        )
        val_dataset = Wav2Vec2Dataset(
            cfg=cfg,
            tokenizer=tokenizer,
            data=val_data,
            inference=inference,
            apply_augmentation=False,
        )
        collator = Wav2Vec2DataCollatorCTC(pad_token_id=tokenizer.pad_token_id)
        feature_extractor = None
    else:
        model_id = cfg.model.get("whisper_model_id", "openai/whisper-tiny")
        load_path, local_files_only = resolve_hf_load_path(model_id, inference=inference)
        if inference:
            if load_path != model_id:
                print(f"Loading Whisper feature extractor in INFERENCE mode from local path: {load_path}")
            else:
                print(
                    f"Local Whisper feature extractor path not found for '{model_id}'; "
                    "trying local Hugging Face cache only."
                )
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            load_path,
            local_files_only=local_files_only,
        )
        print("Initializing Whisper datasets...")
        train_dataset = WhisperDataset(
            cfg=cfg, feature_extractor=feature_extractor,
            tokenizer=tokenizer, data=train_data, inference=inference, apply_augmentation=not inference
        )
        val_dataset = WhisperDataset(
            cfg=cfg, feature_extractor=feature_extractor,
            tokenizer=tokenizer, data=val_data, inference=inference, apply_augmentation=False
        )
        collator = DataCollatorCTCWithPadding(feature_extractor, tokenizer.pad_token_id)

    # DataLoader settings
    dl_cfg = cfg.training.dataloader
    num_workers = dl_cfg.num_workers
    pin_memory = dl_cfg.pin_memory
    prefetch_factor = dl_cfg.get("prefetch_factor")
    persistent_workers = num_workers > 0

    sampler_cfg = cfg.get("sampler")

    # Resolve sampler configs
    train_sampler_cfg = None
    val_sampler_cfg = None

    if sampler_cfg:
        if isinstance(sampler_cfg, DictConfig) and ("train" in sampler_cfg or "val" in sampler_cfg):
            train_sampler_cfg = sampler_cfg.get("train")
            val_sampler_cfg = sampler_cfg.get("val") or train_sampler_cfg
        else:
            train_sampler_cfg = val_sampler_cfg = sampler_cfg

    # Instantiate samplers
    train_sampler = (
        instantiate(train_sampler_cfg, durations=[x["audio_duration_sec"] for x in train_dataset.data])
        if train_sampler_cfg else None
    )

    val_sampler = (
        instantiate(val_sampler_cfg, durations=[x["audio_duration_sec"] for x in val_dataset.data])
        if val_sampler_cfg and len(val_dataset.data) > 0 else None
    )

    def make_loader(dataset, sampler):
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    train_loader = make_loader(train_dataset, train_sampler)
    val_loader = make_loader(val_dataset, val_sampler)
    
    # 7. Package configuration info needed for model/loss initialization
    dataset_info = {
        "vocab_size": tokenizer.vocab_size,
        "blank_token_id": tokenizer.blank_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "tokenizer": tokenizer,
        "feature_extractor": feature_extractor,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
    }

    print("Dataset preparation complete!")
    return train_loader, val_loader, dataset_info

if __name__ == "__main__":
    import json
    from omegaconf import OmegaConf

    print("--- Starting Dataset Debugging ---")
    
    # 1. Load Config and Paths
    cfg = OmegaConf.load("configs/default.yaml")
    jsonl_paths = [cfg.data.train_jsonl, cfg.data.train_jsonl_talkbank]

    print("Loading JSONL data into memory...")
    all_data = get_data_from_json(cfg)


    # 2. Initialize Dependencies
    print("Loading feature extractor and building tokenizer...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    
    tokenizer = instantiate(cfg.tokenizer)
    # FIX: Pass the loaded list of dictionaries!
    tokenizer.build_vocab(all_data)
    print(f"Vocab size built: {tokenizer.vocab_size}")
    
    # 3. Initialize Dataset
    print("Initializing dataset...")
    dataset = WhisperDataset(
        cfg=cfg, 
        feature_extractor=feature_extractor, 
        tokenizer=tokenizer,
        data=all_data  # FIX: Pass the pre-loaded data here too
    )
    print(f"Total utterances in dataset: {len(dataset)}")
    
    print("\n--- Testing get item for single item ---")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Utterance ID: {sample['utterance_id']}")
        print(f"Input features shape (Mel Spec): {sample['input_features'].shape}")
        print(f"Labels shape (Phoneme IDs): {sample['labels'].shape}")
        print(f"Raw input length (frames): {sample['input_length']}")
        print(f"Raw target length (phonemes): {sample['target_length']}")
    
    # 4. Initialize DataLoader & Collator to test Batching
    print("\n--- Testing Batch Collation with BucketBatchSampler ---")
    collator = DataCollatorCTCWithPadding(feature_extractor, tokenizer.pad_token_id)

    batch_size = 4
    durations = [item["audio_duration_sec"] for item in dataset.data]
    sampler = BucketBatchSampler(
        durations=durations,
        batch_size=batch_size,
        bucket_multiplier=10,
        shuffle=True,
        drop_last=False,
    )
    print(f"BucketBatchSampler created: {len(sampler)} batches of ~{batch_size}")

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
    )
    
    # Fetch one batch
    for batch in dataloader:
        print("Batch Keys:", list(batch.keys()))
        print(f"Padded Input features shape: {batch['input_features'].shape}")
        print(f"Padded Labels shape: {batch['labels'].shape}")
        print(f"Batch input lengths: {batch['input_lengths'].tolist()}")
        print(f"Batch target lengths: {batch['target_lengths'].tolist()}")
        print(f"Utterance IDs: {batch['utterance_ids']}")

        # Show padding efficiency
        lengths = batch["input_lengths"].float()
        max_len = batch["input_features"].shape[-1]
        efficiency = lengths.mean() / max_len * 100
        print(f"\n  Padding efficiency: {efficiency:.1f}% (avg_len/max_len in batch)")
        break # Just test the first batch
        
    print("\n--- Debugging Complete ---")
