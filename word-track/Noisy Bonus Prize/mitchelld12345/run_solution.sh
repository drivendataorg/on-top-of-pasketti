#!/bin/bash
set -eou pipefail

export PYTHONPATH=.
ENV="micromamba run -n child_asr"
TRAIN_CONFIG="config/training/word/qwen_distill_augment.yaml"
CHECKPOINT_DIR="models/qwen_word"

DATA_DIR="data"
RAW_DIR="$DATA_DIR/raw"
AUDIO_DIR="$RAW_DIR/audio"
NOISE_DIR="$RAW_DIR/noise"

echo "=== Step 0: Extract and prepare data ==="

mkdir -p "$AUDIO_DIR" "$NOISE_DIR"

# Extract DrivenData audio
for zip in "$DATA_DIR"/drivendata_data/audio_part_*.zip; do
    echo "Extracting $zip ..."
    unzip -o -q "$zip" -d "$RAW_DIR"
done

# Extract DrivenData noise
for zip in "$DATA_DIR"/drivendata_data/noise_part_*.zip; do
    echo "Extracting $zip ..."
    unzip -o -q "$zip" -d "$DATA_DIR/tmp_noise"
done
mv "$DATA_DIR"/tmp_noise/audio/* "$NOISE_DIR/"
rm -rf "$DATA_DIR/tmp_noise"

# Extract TalkBank audio
echo "Extracting $DATA_DIR/talkbank_data/talkbank_audio.zip ..."
unzip -o -q "$DATA_DIR/talkbank_data/talkbank_audio.zip" -d "$RAW_DIR"

# Merge transcripts: cat dd + tb -> combined, then split off smoketest for val
cp "$DATA_DIR/drivendata_data/val_word_smoketest.jsonl" "$RAW_DIR/val_word_smoketest.jsonl"
cat "$DATA_DIR/drivendata_data/train_word_transcripts_dd.jsonl" \
    "$DATA_DIR/talkbank_data/train_word_transcripts_tb.jsonl" \
    > "$RAW_DIR/train_word_transcripts.jsonl"

# Create nosmoketest split (remove val IDs from combined transcripts)
$ENV python -c "
import json
val_ids = set()
with open('$RAW_DIR/val_word_smoketest.jsonl') as f:
    for line in f:
        val_ids.add(json.loads(line)['utterance_id'])
with open('$RAW_DIR/train_word_transcripts.jsonl') as f:
    entries = [json.loads(line) for line in f]
with open('$RAW_DIR/train_word_transcripts_nosmoketest.jsonl', 'w') as f:
    for e in entries:
        if e['utterance_id'] not in val_ids:
            f.write(json.dumps(e) + '\n')
print(f'Total: {len(entries)}, Val: {len(val_ids)}, Train: {len(entries) - len(val_ids)}')
"

# Copy submission format
cp "$DATA_DIR/drivendata_data/submission_format_aqPHQ8m.jsonl" "$RAW_DIR/"

echo "Audio files: $(ls "$AUDIO_DIR"/*.flac 2>/dev/null | wc -l)"
echo "Noise files: $(ls "$NOISE_DIR"/*.flac 2>/dev/null | wc -l)"

# Set up smoketest evaluation data for Docker runtime
SMOKE_DIR="runtime/data-smoke/word"
SMOKE_AUDIO="$SMOKE_DIR/audio"
mkdir -p "$SMOKE_AUDIO" "$SMOKE_DIR/output"

$ENV python -c "
import json
with open('$RAW_DIR/val_word_smoketest.jsonl') as f:
    entries = [json.loads(line) for line in f]
with open('$SMOKE_DIR/submission_format.jsonl', 'w') as sf, \
     open('$SMOKE_DIR/ground_truth.jsonl', 'w') as gt, \
     open('$SMOKE_DIR/utterance_metadata.jsonl', 'w') as um:
    for e in entries:
        sf.write(json.dumps({'utterance_id': e['utterance_id'], 'orthographic_text': ''}) + '\n')
        gt.write(json.dumps({'utterance_id': e['utterance_id'], 'orthographic_text': e['orthographic_text'], 'audio_duration_sec': e['audio_duration_sec']}) + '\n')
        um.write(json.dumps({'utterance_id': e['utterance_id'], 'audio_duration_sec': e['audio_duration_sec'], 'age_bucket': e['age_bucket']}) + '\n')
print(f'Smoketest: {len(entries)} utterances')
"

# Symlink audio files into smoketest directory
AUDIO_ABS=$(cd "$AUDIO_DIR" && pwd)
for uid in $(python3 -c "import json; [print(json.loads(l)['utterance_id']) for l in open('$SMOKE_DIR/submission_format.jsonl')]"); do
    ln -sf "$AUDIO_ABS/$uid.flac" "$SMOKE_AUDIO/$uid.flac" 2>/dev/null || true
done
echo "Smoketest audio symlinks: $(ls "$SMOKE_AUDIO"/*.flac 2>/dev/null | wc -l)"

echo ""
echo "=== Step 1: Generate TTS augmentation data ==="
echo "Generating ~34k synthetic single-word utterances via Qwen3-TTS voice cloning."
echo "This requires a GPU and takes ~2 hours."
$ENV python src/data/generate_tts_words.py

echo ""
echo "=== Step 2: Train Qwen3-ASR-1.7B ==="
echo "Fine-tuning with KL distillation on ~376k utterances (5 epochs)."
echo "This takes ~17 hours on a single GPU."
$ENV python src/models/word/finetune.py \
    --config "$TRAIN_CONFIG" \
    distillation.kl_alpha=0.5 \
    training.epochs=5 \
    training.validate_first=false

echo ""
echo "=== Step 3: Find best checkpoint ==="
BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/*.ckpt | head -1)
echo "Best checkpoint: $BEST_CKPT"

echo ""
echo "=== Step 4: Package submission ==="
echo "Extracting fine-tuned weights into submission directory."
bash package_submission.sh qwen "$BEST_CKPT" --full-model

echo ""
echo "=== Step 5: Run inference (Docker) ==="
echo "Testing submission against the 9k smoketest set."
cd runtime && KIDSASR_DATA_DIR="$(pwd)/data-smoke/word" just track=word run

echo ""
echo "=== Done ==="
