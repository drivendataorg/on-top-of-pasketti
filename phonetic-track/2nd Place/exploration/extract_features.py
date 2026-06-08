import json
import hashlib
import librosa
import soundfile as sf
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
from unicodedata import normalize
import concurrent.futures
import io
import os
import eng_to_ipa as ipa 

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPLORATION_DIR = PROJECT_ROOT / "exploration"

PHON_JSONL_FILES = [
    DATA_DIR / "train_phon_transcripts.jsonl",
    DATA_DIR / "train_phon_transcripts_talkbank.jsonl",
]

WORD_JSONL_FILES = [
    DATA_DIR / "train_word_transcripts.jsonl",
    DATA_DIR / "train_word_transcripts_talkbank.jsonl",
]

OUTPUT_PARQUET = EXPLORATION_DIR / "engineered_audio_features.parquet"



# added some stuff from main could be fomarted better
VALID_IPA_CHARS = {
    " ",
    "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w", "x", "z",
    "e", "i", "o", "u",
    "ɑ", "æ", "ɐ", "ɔ", "ə", "ɚ", "ɛ", "ɪ", "ʊ", "ʌ",
    "ç", "ð", "ŋ", "ɟ", "ɫ", "ɬ", "ɹ", "ɾ", "ʁ", "ʃ", "ʒ", "ʔ", "ʝ", "θ", "χ",
    "ʧ", "ʤ", "ː",
}


def normalize_ipa_for_submission(text: str) -> str:
    text = normalize("NFC", text)
    text = text.replace("ɝ", "ɚ")
    text = text.replace("tʃ", "ʧ").replace("dʒ", "ʤ").replace("r","ɹ").replace("a","ɑ")
    text = " ".join(text.split())
    return "".join(ch for ch in text if ch in VALID_IPA_CHARS)


def process_utterance(row_data):
    """Processes a single utterance. Designed to be run in parallel."""
    audio_file_path = DATA_DIR / row_data["audio_path"]
    
    # 1. Define the baseline schema with None (Null) values
    features = {
        'is_stereo': None,
        'native_sample_rate': None,
        'bit_depth': None,
        'calculated_md5': None,
        'clipping_amount': None,
        'zcr_mean': None,
        'centroid_mean': None,
        'f0_mean': None,
        'error': None
    }

    # Check if file exists; if not, return the Null schema + the error message
    if not audio_file_path.exists():
        features['error'] = "File not found"
        return {**row_data, **features}

    try:
        # Read file into memory
        with open(audio_file_path, "rb") as f:
            file_bytes = f.read()
            
        features['calculated_md5'] = hashlib.md5(file_bytes).hexdigest()

        # Audio Metadata
        with sf.SoundFile(io.BytesIO(file_bytes)) as sf_file:
            features['is_stereo'] = sf_file.channels > 1
            features['native_sample_rate'] = sf_file.samplerate
            features['bit_depth'] = sf_file.subtype
            y = sf_file.read(dtype='float32')

        if features['is_stereo']:
            y = np.mean(y, axis=1)

        # DSP Features
        features['clipping_amount'] = int(np.sum(np.abs(y) >= 0.999))
        features['zcr_mean'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        features['centroid_mean'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=features['native_sample_rate'])))
        
        f0 = librosa.yin(y, fmin=130, fmax=1046, sr=features['native_sample_rate'])
        features['f0_mean'] = float(np.nanmean(f0))

        return {**row_data, **features}

    except Exception as e:
        # If the file exists but is corrupted, return the Null schema + the error message
        features['error'] = str(e)
        return {**row_data, **features}

def add_words(df: pl.DataFrame, word_files: list) -> pl.DataFrame:
    """Combines all word transcripts, joins them, and rapidly translates to proper IPA."""
    
    # 1. Read and combine all word files into one single DataFrame first
    word_dfs = [pl.read_ndjson(f) for f in word_files]
    df_combined_words = pl.concat(word_dfs)
    
    # Ensure there are no duplicate utterance_ids in the word transcripts to avoid exploding the join
    df_combined_words = df_combined_words.unique(subset=["utterance_id"])

    # 2. Perform a single left join
    df = df.join(
        df_combined_words.select(['utterance_id', 'orthographic_text']), 
        on='utterance_id', 
        how='left'
    )

    # 3. Fast translation logic
    # Get unique texts (ignoring nulls)
    unique_texts = df.filter(pl.col("orthographic_text").is_not_null()).select("orthographic_text").unique()
    
    unique_map = {
    text: normalize_ipa_for_submission(ipa.convert(text))
    for text in tqdm(unique_texts["orthographic_text"], desc="Translating to IPA")
}

    # 4. Map the dictionary to a new 'proper_phones' column
    df = df.with_columns(
        proper_phones = pl.col("orthographic_text").replace(unique_map)
    )

    return df

def main():
    all_rows = []
    
    for jsonl_file in PHON_JSONL_FILES:
        print(f"Loading metadata from: {jsonl_file.name}")
        lines = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                # Parse the JSON and append the source file name
                row_data = json.loads(line)
                row_data["source_file"] = jsonl_file.name
                lines.append(row_data)
            
        print(f"Starting parallel processing for {len(lines)} files...")
        
        max_workers = os.cpu_count() or 4
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # map over the lines, wrapping with tqdm for the progress bar
            results = list(tqdm(
                executor.map(process_utterance, lines), 
                total=len(lines), 
                desc="Extracting Features"
            ))
            all_rows.extend(results)

    print("\nConverting to Polars DataFrame...")
    # infer_schema_length=None forces Polars to check all rows for the correct data types
    df = pl.DataFrame(all_rows, infer_schema_length=None)
    
    print("\nAdding text and IPA translations...")
    df = add_words(df, WORD_JSONL_FILES)

    print(f"\nSaving Parquet to: {OUTPUT_PARQUET}")
    df.write_parquet(OUTPUT_PARQUET)
    
    print("\n--- Summary ---")
    print(f"Total Utterances Processed: {df.height}")
    errors = df.filter(pl.col("error").is_not_null()).height
    print(f"Files with extraction errors: {errors}")

if __name__ == "__main__":
    main()