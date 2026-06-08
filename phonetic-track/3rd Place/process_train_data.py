import json
import os
import random

# ── Step 1: Transform all 4 files (rename audio_path + fix phon/drivendata prefix) ──

files = [
    ("train_phon_transcripts_drivendata.jsonl", True),   # rename key + fix path prefix
    ("train_phon_transcripts_talkbank.jsonl",   False),  # rename key only
    ("train_word_transcripts_drivendata.jsonl", False),  # rename key only
    ("train_word_transcripts_talkbank.jsonl",   False),  # rename key only
]

transformed = {}  # filename -> list of transformed records

for filename, fix_phon_path in files:
    if not os.path.exists(filename):
        print(f"SKIP (not found): {filename}")
        transformed[filename] = []
        continue

    records = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            # 1. Rename audio_path -> audio_filepath
            if "audio_path" in record:
                record["audio_filepath"] = record.pop("audio_path")

            # 2. Fix path prefix for phon+drivendata file only
            if fix_phon_path and "audio_filepath" in record:
                record["audio_filepath"] = record["audio_filepath"].replace(
                    "audio_drivendata/", "audio_drivendata_phon/", 1
                )

            records.append(record)

    transformed[filename] = records
    print(f"Transformed: {filename} ({len(records):,} records)")

# ── Step 2: Collect phon and word records by utterance_id ────────────────────

phon_records = {}
for fname in ["train_phon_transcripts_drivendata.jsonl", "train_phon_transcripts_talkbank.jsonl"]:
    for r in transformed.get(fname, []):
        phon_records[r["utterance_id"]] = r

word_records = {}
for fname in ["train_word_transcripts_drivendata.jsonl", "train_word_transcripts_talkbank.jsonl"]:
    for r in transformed.get(fname, []):
        word_records[r["utterance_id"]] = r

# ── Step 3: Split into train (both labels) and unlabel (word only) ────────────

train_rows   = []
unlabel_rows = []

for uid, word_rec in word_records.items():
    if uid in phon_records:
        merged = dict(word_rec)
        merged["phonetic_text"] = phon_records[uid].get("phonetic_text", "")
        train_rows.append(merged)
    else:
        unlabel_rows.append(word_rec)

# ── Step 4: Write outputs (keep only required fields) ────────────────────────

with open("phon_train_all_filtered.json", "w", encoding="utf-8") as f:
    for r in train_rows:
        out = {
            "audio_filepath": r.get("audio_filepath", ""),
            "duration":       r.get("audio_duration_sec", ""),
            "text":           r.get("phonetic_text", ""),
            "word":           r.get("orthographic_text", ""),
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

with open("nolabel_phon.json", "w", encoding="utf-8") as f:
    for r in unlabel_rows:
        out = {
            "audio_filepath": r.get("audio_filepath", ""),
            "duration":       r.get("audio_duration_sec", ""),
            "text":           "",
            "word":           r.get("orthographic_text", ""),
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"\nphon_train_all_filtered.json -> {len(train_rows):,} records (phon + word)")
print(f"nolabel_phon.json            -> {len(unlabel_rows):,} records (word only)")

# ── Step 5: Sample random 1/10 of train_rows -> val_f0_phon_filtered.jsonl ─────────────────────

sample_size = max(1, len(train_rows) // 10)
f0_rows = random.sample(train_rows, sample_size)

with open("val_f0_phon_filtered.json", "w", encoding="utf-8") as f:
    for r in f0_rows:
        out = {
            "audio_filepath": r.get("audio_filepath", ""),
            "duration":       r.get("audio_duration_sec", ""),
            "text":           r.get("phonetic_text", ""),
            "word":           r.get("orthographic_text", ""),
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"val_f0_phon_filtered.json                      -> {len(f0_rows):,} records (random 1/10 of train)")


