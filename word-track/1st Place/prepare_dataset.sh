#!/bin/bash

# === make validation smoke ====
uv run csrc_prep/src/csrc_prep/make_smoke.py \
input/csrc-smoke/submission_format_z2HCh3r.jsonl \
input/raw-csrc/train_word_transcripts.jsonl \
input/raw-talkbank/train_word_transcripts.jsonl

# === prepare competition hosted data ===
# flac to mp3
uv run csrc_prep/src/csrc_prep/flac_to_mp3.py input/raw-csrc/audio_part_0.zip input/raw-csrc/audio_part_1.zip input/raw-csrc/audio_part_2.zip input/raw-csrc/audio.tar

# get audio duration (audio_part.tar: audio/XXX.mp3)
uv run csrc_prep/src/csrc_prep/get_audio_duration.py input/raw-csrc

# convert train_word_transcripts.jsonl to csv
uv run csrc_prep/src/csrc_prep/convert_train_jsonl_to_csv.py input/raw-csrc/

# forced align
## create manifest
tar -xf input/raw-csrc/audio.tar -C input/raw-csrc/
uv run csrc_prep/src/csrc_prep/forced_align/manifest.py input/raw-csrc/

## align
uv run csrc_prep/src/csrc_prep/forced_align/align.py input/raw-csrc/ input/parakeet-ctc-1.1b/parakeet-ctc-1.1b.nemo NeMo/

## split by alignment
uv run csrc_prep/src/csrc_prep/forced_align/split_by_alignment.py input/raw-csrc/

# convert result_json to train_word_transcripts.csv
uv run csrc_prep/src/csrc_prep/forced_align/final_result_jsonl_to_csv.py input/raw-csrc/

# concat original train, forced aligned csv
uv run csrc_prep/src/csrc_prep/merge_train_csv.py input/raw-csrc/ input/csrc-processed-input

# ==== prepare talkbank hosted data ===
# flac to mp3
uv run csrc_prep/src/csrc_prep/flac_to_mp3.py input/raw-talkbank/audio.zip input/raw-talkbank/audio.tar

# get audio duration (audio_part.tar: audio/XXX.mp3)
uv run csrc_prep/src/csrc_prep/get_audio_duration.py input/raw-talkbank

# convert train_word_transcripts.jsonl to csv
uv run csrc_prep/src/csrc_prep/convert_train_jsonl_to_csv.py input/raw-talkbank/

# forced align
## create manifest
tar -xf input/raw-talkbank/audio.tar -C input/raw-talkbank/
uv run csrc_prep/src/csrc_prep/forced_align/manifest.py input/raw-talkbank/

## align
uv run csrc_prep/src/csrc_prep/forced_align/align.py input/raw-talkbank/ input/parakeet-ctc-1.1b/parakeet-ctc-1.1b.nemo NeMo/

## split by alignment
uv run csrc_prep/src/csrc_prep/forced_align/split_by_alignment.py input/raw-talkbank/

# convert result_json to train_word_transcripts.csv
uv run csrc_prep/src/csrc_prep/forced_align/final_result_jsonl_to_csv.py input/raw-talkbank/

# === concat original train, forced aligned csv ===
uv run csrc_prep/src/csrc_prep/merge_train_csv.py input/raw-talkbank/ input/csrc-processed-talkbank