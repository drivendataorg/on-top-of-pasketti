"""Dataset and data loading for word track fine-tuning."""
import json

import librosa
import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data.augment import build_augment
from src.paths import RAW_AUDIO_DIR, TARGET_SR, TRAIN_TRANSCRIPTS, VAL_TRANSCRIPTS


MAX_DURATION = 30
MAX_CHARS_PER_SEC = 20

BAD_UTTERANCE_IDS = {
    "U_b8a4e8220e65219b",
}


def _is_valid_entry(e, text_field="orthographic_text"):
    if e.get("utterance_id", "") in BAD_UTTERANCE_IDS:
        return False
    dur = e["audio_duration_sec"]
    if dur >= MAX_DURATION:
        return False
    if dur > 0 and len(e[text_field]) / dur > MAX_CHARS_PER_SEC:
        return False
    return True


class ChildASRDataset(Dataset):

    def __init__(self, entries, tokenizer, audio_base_dir=None, train=True, augment_cfg=None):
        if train:
            self.entries = [e for e in entries if _is_valid_entry(e)]
        else:
            self.entries = [e for e in entries if e.get("utterance_id", "") not in BAD_UTTERANCE_IDS]
        self.tokenizer = tokenizer
        self.audio_base_dir = audio_base_dir
        self.augment = build_augment(augment_cfg) if train and augment_cfg else None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        if self.audio_base_dir is not None:
            audio_path = self.audio_base_dir / entry["audio_path"]
        else:
            audio_path = RAW_AUDIO_DIR.parent / entry["audio_path"]
        try:
            audio, _ = librosa.load(audio_path, sr=TARGET_SR, dtype="float32", mono=True)
            if self.augment is not None:
                audio = self.augment(samples=audio, sample_rate=TARGET_SR)
            signal = torch.from_numpy(audio)
            transcript = self.tokenizer.text_to_ids(entry["orthographic_text"])
            transcript = torch.tensor(transcript, dtype=torch.long)
            return signal, transcript, entry["orthographic_text"]
        except Exception as e:
            uid = entry.get("utterance_id", audio_path.stem)
            logger.warning(f"Skipping {uid}: {e}")
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    signals, transcripts, raw_texts = zip(*batch)
    signal_lens = torch.tensor([s.shape[0] for s in signals], dtype=torch.long)
    transcript_lens = torch.tensor([t.shape[0] for t in transcripts], dtype=torch.long)
    signals_padded = pad_sequence(signals, batch_first=True, padding_value=0.0)
    transcripts_padded = pad_sequence(transcripts, batch_first=True, padding_value=0)
    return signals_padded, signal_lens, transcripts_padded, transcript_lens, list(raw_texts)


def load_and_split(val_ratio=None, seed=None):
    with open(TRAIN_TRANSCRIPTS) as f:
        train = [json.loads(line) for line in f]
    with open(VAL_TRANSCRIPTS) as f:
        val = [json.loads(line) for line in f]
    return train, val
