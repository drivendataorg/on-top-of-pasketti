from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
AUDIO_DIR = PROCESSED_DIR / "audio"
MODELS_DIR = ROOT_DIR / "models"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"

TRAIN_TRANSCRIPTS = RAW_DIR / "train_word_transcripts_nosmoketest.jsonl"
VAL_TRANSCRIPTS = RAW_DIR / "val_word_smoketest.jsonl"
ALL_TRANSCRIPTS = RAW_DIR / "train_word_transcripts.jsonl"
SUBMISSION_FORMAT_A = RAW_DIR / "submission_format_aqPHQ8m.jsonl"
SUBMISSION_FORMAT_B = RAW_DIR / "submission_format_z2HCh3r.jsonl"

RAW_AUDIO_DIR = RAW_DIR / "audio"
NOISE_DIR = RAW_DIR / "noise"

TARGET_SR = 16000


if __name__ == "__main__":
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, AUDIO_DIR, MODELS_DIR, SUBMISSIONS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")
