"""Generate synthetic child speech for SLP diagnostic words using Qwen3-TTS voice cloning.

144 speakers (evenly sampled by age), each generates all 236 words = 33,984 total.
Uses the direct generate_voice_clone API from the reference implementation.
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
from loguru import logger
from qwen_tts import Qwen3TTSModel

from src.data.utils import load_jsonl
from src.paths import RAW_AUDIO_DIR, TRAIN_TRANSCRIPTS

OUTPUT_DIR = Path("data/tts_words")
SPEAKERS_PER_AGE = 34
SEED = 42
BATCH_SIZE = 16

SLP_DIAGNOSTIC_WORDS = [
    "alligator", "animal", "apple", "baby", "bag", "ball", "balloon", "banana", "basket",
    "bat", "bath", "bathtub", "bear", "bed", "bee", "bell", "bicycle", "bird", "blue",
    "boat", "book", "boy", "bridge", "brown", "brush", "bug", "bus", "butter", "butterfly",
    "button", "cage", "cake", "camera", "candle", "car", "carrot", "cat", "catch", "chair",
    "cheese", "chest", "chicken", "Christmas", "church", "clown", "coat", "cookie", "cow",
    "crab", "crayon", "cup", "deer", "dinosaur", "dish", "dog", "doll", "dolphin", "door",
    "dragon", "dress", "drum", "duck", "eagle", "ear", "egg", "elephant", "face", "fan",
    "farm", "feather", "finger", "fire", "firefighter", "fish", "fishing", "five", "flag",
    "flower", "flowers", "fork", "fox", "frog", "fruit", "game", "gate", "gem", "giraffe",
    "girl", "glasses", "glove", "go", "goat", "grass", "green", "guitar", "gum", "gun",
    "hair", "hamburger", "hammer", "hand", "hat", "head", "heart", "helicopter", "hippopotamus",
    "horse", "hospital", "house", "ice", "jam", "jar", "jeep", "jet", "juice", "jumping",
    "kangaroo", "key", "king", "kitchen", "kite", "kitten", "knife", "ladder", "lamp", "leaf",
    "leg", "lemon", "light", "lion", "lip", "lock", "magic", "man", "map", "matches", "milk",
    "mirror", "monkey", "moon", "mother", "mouse", "mushroom", "nail", "name", "nest", "nine",
    "nose", "nut", "ocean", "orange", "oven", "owl", "page", "paint", "pajamas", "pan",
    "paper", "park", "peach", "pen", "pencils", "pig", "pizza", "plane", "plate", "puzzle",
    "quack", "queen", "rabbit", "rain", "ring", "road", "robot", "rock", "rope", "rug",
    "run", "sandwich", "Santa Claus", "scissors", "seal", "sheep", "ship", "shirt", "shoe",
    "shovel", "six", "sleeping", "slide", "snake", "snow", "soap", "sock", "soldier",
    "spaghetti", "spider", "spoon", "squirrel", "star", "stove", "sun", "swing", "table",
    "teacher", "teeth", "telephone", "ten", "this", "three", "thumb", "tiger", "toe",
    "tongue", "tooth", "toothbrush", "train", "tree", "truck", "umbrella", "vacuum", "van",
    "vase", "vine", "wagon", "watch", "watches", "water", "wave", "web", "wheel", "window",
    "yellow", "zebra", "zipper", "zoo",
]


def find_eligible_speakers(entries, min_clips=3):
    by_speaker = defaultdict(list)
    for e in entries:
        if 2.5 <= e["audio_duration_sec"] <= 4.0 and len(e["orthographic_text"].split()) >= 3:
            by_speaker[e["child_id"]].append(e)

    by_age = defaultdict(list)
    for cid, clips in by_speaker.items():
        if len(clips) < min_clips:
            continue
        best = sorted(clips, key=lambda e: abs(e["audio_duration_sec"] - 3.0))[0]
        age = best.get("age_bucket", "unknown")
        by_age[age].append({
            "child_id": cid,
            "age_bucket": age,
            "ref_entry": best,
        })

    rng = random.Random(SEED)
    selected = []
    for age in ["3-4", "5-7", "8-11"]:
        pool = by_age[age]
        rng.shuffle(pool)
        n = min(SPEAKERS_PER_AGE, len(pool))
        selected.extend(pool[:n])
        logger.info(f"  {age}: {n}/{len(pool)} speakers")
    selected.extend(by_age["unknown"])
    logger.info(f"  unknown: {len(by_age['unknown'])} speakers")
    return selected


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir = OUTPUT_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)

    entries = load_jsonl(TRAIN_TRANSCRIPTS)
    words = SLP_DIAGNOSTIC_WORDS
    speakers = find_eligible_speakers(entries)
    total = len(speakers) * len(words)
    logger.info(f"{len(speakers)} speakers x {len(words)} words = {total} total")

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    manifest_path = OUTPUT_DIR / "tts_words_manifest.jsonl"
    done_keys = set()
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                done_keys.add((entry["child_id"], entry["orthographic_text"]))
    if done_keys:
        logger.info(f"Resuming: {len(done_keys)} already done")

    manifest_file = open(manifest_path, "a", encoding="utf-8", buffering=1)
    t0 = time.time()
    total_generated = len(done_keys)

    for si, speaker in enumerate(speakers):
        remaining_words = [w for w in words if (speaker["child_id"], w) not in done_keys]
        if not remaining_words:
            continue

        ref = speaker["ref_entry"]
        ref_audio_path = str(RAW_AUDIO_DIR / f"{ref['utterance_id']}.flac")
        ref_text = ref["orthographic_text"]
        speaker_t0 = time.time()

        for i in range(0, len(remaining_words), BATCH_SIZE):
            batch = remaining_words[i : i + BATCH_SIZE]
            try:
                with torch.inference_mode():
                    wav_list, sr = model.generate_voice_clone(
                        text=batch,
                        language=["English"] * len(batch),
                        ref_audio=ref_audio_path,
                        ref_text=ref_text,
                        max_new_tokens=60,
                    )

                for word, wav in zip(batch, wav_list):
                    safe_word = word.lower().replace(" ", "_").replace("'", "")
                    fname = f"{speaker['child_id']}_{safe_word}.wav"
                    fpath = audio_dir / fname
                    sf.write(str(fpath), wav, sr)
                    manifest_file.write(json.dumps({
                        "audio_path": str(Path("tts_words/audio") / fname),
                        "orthographic_text": word,
                        "child_id": speaker["child_id"],
                        "age_bucket": speaker["age_bucket"],
                        "audio_duration_sec": round(len(wav) / sr, 3),
                        "source": "tts",
                    }, ensure_ascii=False) + "\n")
                    total_generated += 1

            except Exception as e:
                logger.error(f"Batch failure for {speaker['child_id']} at word {batch[0]}: {e}")

        speaker_elapsed = time.time() - speaker_t0
        total_elapsed = time.time() - t0
        new_generated = total_generated - len(done_keys)
        rate = new_generated / total_elapsed if total_elapsed > 0 else 0
        remaining = total - total_generated
        eta_min = remaining / rate / 60 if rate > 0 else 0
        logger.info(f"Speaker {si+1}/{len(speakers)} done in {speaker_elapsed:.0f}s | "
                    f"{total_generated}/{total} ({rate:.1f} wps) | ETA {eta_min:.0f}min")

    manifest_file.close()
    logger.info(f"Done. {total_generated} total samples.")


if __name__ == "__main__":
    main()
