# Solution — Pasketti: Children's Speech Recognition (Phonetic Track)

**Username**: chenghuige  
**Final Score**: Public LB **0.2539** / Private LB **0.2559** (IPA CER ↓)

> This document is the methodological write-up copied from the original
> experiment repository. For the standalone release in this folder, use
> [README.md](../README.md) and [Makefile](../Makefile) as the authoritative
> replication entry points. Some path names below intentionally refer to the
> original training workspace to preserve historical context.

---

## Summary

An **11-model heterogeneous ensemble** with a **CatBoost tree reranker**, combining two complementary backbone families:

- **NeMo Parakeet-TDT-0.6B** (7 models: 3 TDT + 4 CTC) — excels on long audio & DrivenData data
- **WavLM-Large** (4 CTC models) — excels on short audio & TalkBank (EXT) data
- **Dual-head training**: IPA phonetic CTC/TDT head + Word BPE CTC auxiliary head (weight 0.3), leveraging the larger word-track dataset to strengthen the shared encoder
- **Concat Mix augmentation**: most impactful augmentation — concatenate up to 8 audio clips with their labels
- **CatBoost LambdaRank reranker**: 5-fold CV tree model that reranks N-best candidates using CTC scores, edit distances, and audio features (~200 features)

Pipeline: each model generates N-best candidates → all models CTC-rescore all candidates → CatBoost selects the best candidate per utterance.

---

## Original Development Repository Structure

```
pasketti-phonetic/
├── src/
│   ├── README-solution.md     ← You are here
│   ├── main.py                # Training entry point → delegates to shared train.run()
│   ├── config.py              # Track='phonetic', IPA CER metric, fold config
│   ├── eval_score.py          # Hierarchical evaluation (by source, age, duration)
│   ├── ensemble.py            # Full ensemble pipeline (MBR, N-best, tree reranker)
│   ├── ensemble_feats.py      # Reranker feature engineering
│   ├── export_model.py        # Export training checkpoint → inference format
│   ├── pack_submission.sh     # Pack single-model submission.zip
│   ├── pack_ensemble.sh       # Pack multi-model ensemble submission.zip
│   ├── models.txt             # Active model list for final ensemble (11 models)
│   ├── flags/                 # Flagfile configs (v13→v17 inheritance chain)
│   │   ├── base, v16, v17, ...
│   └── online-logs/           # Competition submission logs (1.txt-16.txt, score.txt)
│
├── ../pasketti/src/           # Shared code (both phonetic & word tracks)
│   ├── train.py               # Training loop (delegates to melt mt.fit)
│   ├── config_base.py         # All shared FLAGS definitions (~500+ flags)
│   ├── dataset.py             # Data loading + augmentation (concat mix, SpecAugment, noise)
│   ├── eval.py                # Evaluation logic (IPA CER / WER)
│   ├── submit.py              # Inference entry point (Docker runtime on competition platform)
│   ├── ctc_decode.py          # CTC beam search decoding
│   ├── reranker_features.py   # Tree reranker feature extraction
│   └── models/
│       ├── nemo.py            # NeMo Parakeet TDT/CTC model wrapper
│       ├── wav2vec2.py        # WavLM / HuBERT / Wav2Vec2 CTC wrapper
│       ├── whisper.py         # Whisper CTC/S2S wrapper
│       └── base.py            # Model base class
│
├── ../input/childrens-phonetic-asr/   # Competition data (not included)
│
└── ../working/
    ├── offline/9/<model_name>/0/      # Fold 0 training outputs
    │   ├── model.pt, flags.json       # Checkpoint + config
    │   ├── log.html, metrics.csv      # Training log + metrics
    │   └── eval.csv                   # Validation predictions
    └── online/9/<model_name>/         # Full-train (all data) outputs
```

**Compatibility utilities** (bundled in this repository under `src/_compat/`):

| Library | Path | Description |
|---|---|---|
| `gezi` (gz) | `src/_compat/gezi` | Minimal public shim for Timer, logging, FLAGS/config restore, globals, fold setup, and checkpoint loading |
| `melt` (mt) | `src/_compat/melt` | Minimal public shim for `mt.init`, `mt.epoch`, and global state helpers; training uses `src/train_loop.py` instead of `mt.fit` |
| `lele` | `src/_compat/lele` | Minimal public shim for optimizer parameter groups, samplers, bucket batching, and checkpoint loading |

---

## Setup

### 1. System Requirements

- **OS**: Ubuntu 20.04+ (tested on Ubuntu 22.04)
- **Python**: 3.10+ (Miniconda recommended)
- **GPU**: NVIDIA GPU with ≥24GB VRAM for training; A100 80GB for full ensemble inference
- **CUDA**: 12.x
- **Disk**: ~50GB for model weights (11 models)

### 2. Create Environment

```bash
conda create -n torch python=3.10 -y
conda activate torch
```

### 3. Install Dependencies

```bash
# Core PyTorch
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# NeMo toolkit (for Parakeet-TDT backbone)
pip install nemo_toolkit[asr]==2.1.0

# HuggingFace (for WavLM backbone)
pip install transformers==4.47.0 safetensors

# Audio processing
pip install librosa soundfile jiwer editdistance

# Tree reranker
pip install catboost lightgbm

# Training utilities
pip install absl-py icecream pandas numpy scipy tqdm scikit-learn peft sentencepiece
```

### 4. Set Up Paths

```bash
# Clone repository
git clone https://github.com/chenghuige/pikachu.git
cd pikachu

# Set environment variables
export PYTHONPATH=$PWD/utils:$PWD/third:$PYTHONPATH
export HF_HOME=~/.cache/huggingface
```

### 5. Data Setup

Download competition data from [DrivenData](https://www.drivendata.org/competitions/pasketti/) and place in:

```
pikachu/projects/drivendata/input/childrens-phonetic-asr/
├── train_phon_transcripts.jsonl
└── audio/
    └── <utterance_id>.flac

pikachu/projects/drivendata/input/childrens-ext-asr/   # optional TalkBank / EXT data
├── train_phon_transcripts.jsonl
└── audio/
    └── <utterance_id>.flac
```

The training code reads the official `.flac` files directly; no `.wav`
conversion step is required. If the data lives elsewhere, either symlink these
directories or pass `DATA_DIR` and `EXT_DATA_DIR` to the Makefile targets.

### 6. Download Pretrained Model Weights

Download the 11 trained model weights and tree reranker from the public
Hugging Face model repository:

```bash
HF_REPO_ID=huigecheng/pasketti-phonetic-weights bash scripts/download_weights.sh
```

Place model weights in:

```
pikachu/projects/drivendata/pasketti-phonetic-solution/working/online/17/
├── v17.backbone-wavlm-large.ep3.5.leval/        # WavLM model 1
├── v16.backbone-wavlm-large.dual_bpe.mix4.eval/ # WavLM model 2
├── v16.backbone-wavlm-large.dual_bpe.mix4.mix_csss.ep4.5.eval/  # WavLM model 3
├── v16.backbone-wavlm-large.dual_bpe.eval/      # WavLM model 4
├── v16.dual_bpe.tdt_only.eval/                   # NeMo TDT model 1
├── v16.dual_bpe.mix_csss.tdt_only.eval/          # NeMo TDT model 2
├── v16.dual_bpe.mix2.mix_csss.tdt_only.eval/     # NeMo TDT model 3
├── v16.dual_bpe.wo_scale-2.eval/                 # NeMo CTC model 1
├── v16.aux_loss.dual_bpe.eval/                   # NeMo CTC model 2
├── v16.dual_bpe.mix4.eval/                       # NeMo CTC model 3
└── v16.dual_bpe.mix2.eval/                       # NeMo CTC model 4
```

Each model directory contains: `model.pt` (weights), `flags.json` (config), `nemo_model_slim.nemo` (NeMo encoder, for NeMo models only).

Total size: ~15 GB (11 models × ~1.2 GB each + reranker).

---

## Hardware

| Resource | Training | Inference (competition) |
|---|---|---|
| **GPU** | RTX 4090/5090 (NeMo), RTX Pro 6000 (WavLM) | NVIDIA A100 80GB |
| **Memory** | 128 GB RAM | 80 GB GPU + system RAM |
| **OS** | Ubuntu 22.04 | Docker (CUDA 12.6) |

| Step | Duration |
|---|---|
| NeMo model training (per model) | ~1-2 hours (5 epochs, 4090) |
| WavLM model training (per model) | ~15-25 hours (3-5 epochs, Pro 6000) |
| Total training (11 models) | ~100-150 hours |
| Ensemble reranker training | ~30 min |
| **Submission inference (11 models + reranker)** | **~30-56 min on A100** |

---

## Run Training

### Single Model Training (NeMo Parakeet-TDT)

```bash
cd pikachu/projects/drivendata/pasketti-phonetic-solution

# Train fold 0 with the public standalone entry point
make train-fold0 GPU=1

# Equivalent explicit command
cd src
PYTHONPATH=_compat:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 python train.py \
    --flagfile=flags/v17 \
    --mn=v17.fold0 \
    --fold=0 \
    --root=../../input/childrens-phonetic-asr \
    --ext_root=../../input/childrens-ext-asr \
    --eval_ext_root=../../input/childrens-ext-asr

# Full train (all data, no validation holdout) for final submission
make train-online GPU=1
```

### Single Model Training (WavLM-Large)

```bash
# WavLM models are training-expensive (~5h/epoch on Pro 6000)
cd pikachu/projects/drivendata/pasketti-phonetic-solution/src
PYTHONPATH=_compat:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 python train.py --flagfile=flags/v16 \
    --backbone=microsoft/wavlm-large --model=wav2vec2 \
    --mn=v16.backbone-wavlm-large.dual_bpe --ep=5
```

### Flagfile Inheritance Chain

```
flags/base → v13 → v13-ema → v14-ema → v14-ema5-shuffle → v15 → v16 → v17
```

Key flags introduced at each stage:

| Flag | Version | Description |
|---|---|---|
| `--corpus_level_loss` | v8/v9 | Corpus-level CTC loss computation |
| `--dual_bpe` | v13+ | Dual-head with word BPE CTC auxiliary loss (weight 0.3) |
| `--tdt_only` | v16 | TDT-only decoding (skip CTC head at inference) |
| `--aug_mix=N` | v14+ | Concat mix augmentation (N clips per sample) |
| `--mix_csss` | v14+ | DD/EXT equal-probability mix strategy |
| `--ema --ema_decay=0.999` | v13+ | EMA model averaging, start epoch 1 |
| `--cnoise` | v16 | Classroom noise augmentation |

### Train Tree Reranker (after all models are trained)

In the standalone release, the recommended reproduction path is to use the
helper scripts from `src/` rather than calling the historical internal entry
point directly:

```bash
cd pikachu/projects/drivendata/pasketti-phonetic-solution/src

# Step 1: build offline fold-0 eval artifacts for all 11 models
bash reproduce_offline_fold0.sh

# Step 2: train the CatBoost reranker from those offline artifacts
bash reproduce_tree_reranker.sh
```

A healthy reranker run should print the selected offline artifact root, confirm
that all 11 model eval directories were found, build the feature table, run
5-fold CatBoost CV, and finally copy release artifacts into `src/tree_reranker/`.
Typical milestones look like:

```text
Using offline artifact root: ../../pasketti-phonetic/working/offline/9
Found 11 model eval dirs; 8 have ctc_logprobs.pt.
Built 1068582 candidate rows for 30645 utterances
Dataset: 1068582 rows, 212 features
--- Tree Reranker FullAvg (cb, 5-fold models) Results ---
Overall CER: 0.26086
Copied tree reranker artifacts to tree_reranker
```

Some warnings are expected during this step and are not fatal, for example
missing optional `aux_meta_preds.pt`, missing optional `model.pt`, or missing
`ctc_logprobs.pt` for some TDT-only models. The hard requirements are that each
model has an `eval.csv`, and that at least one model provides `ctc_logprobs.pt`.

For a full line-by-line successful run log, see
[`TREE_RERANKER_SUCCESS_LOG.md`](TREE_RERANKER_SUCCESS_LOG.md).

---

## Run Inference

### Method 1: Competition Submission (Docker)

This is how the competition platform runs inference:

```bash
cd pikachu/projects/drivendata/pasketti-phonetic-solution

# Step 1: Pack the 11-model ensemble + tree reranker into submission.zip
make pack

# Step 2: Upload submission.zip to DrivenData
python dd_submit.py submission.zip
```

Inside the Docker container, `submit.py` runs the pipeline:

1. Loads each of the 11 models sequentially (to fit in GPU memory)
2. Each model generates N-best candidates (beam_width=10) + CTC log-probabilities
3. Cross-model CTC rescoring: all models score all candidates
4. CatBoost reranker selects the best candidate per utterance
5. Outputs `submission.csv` with IPA transcriptions

### Method 2: Local Inference on New Data

```bash
cd pikachu/projects/drivendata/pasketti-phonetic-solution/src

# Run full ensemble inference on any audio data in competition format
python ensemble.py --mode=tree_reranker \
    --models_file=models.txt \
    --data_dir=../../input/childrens-phonetic-asr/ \
    --output=submission.csv
```

### Model Weights Access

All 11 trained model weights + tree reranker are available at
<https://huggingface.co/huigecheng/pasketti-phonetic-weights>.

---

## II. Basic Information for Winner Announcement

- **Name**: ChengHuige
- **Hometown**: Beijing, China
- **Social handle / URL**: https://github.com/chenghuige
- **Picture**: GitHub avatar (https://github.com/chenghuige.png) — also available on request

---

## III. Write-up: Model Documentation

### 1. Who are you?

I am **ChengHuige**, a software engineer based in Beijing, China.

I have participated in many Kaggle competitions and am a **Kaggle Grandmaster**, with a peak ranking of **#7** worldwide. I have also competed in and won numerous domestic competitions in China, including **1st place in the Tencent WBDC 2021** (We-Chat Big Data Challenge).

For this challenge, all source code was authored with the assistance of GPT-4 and Claude (Copilot coding agents); I focused on experiment design, strategy, ablation analysis, and iteration direction.

### 2. Motivation

Children's ASR is a challenging and impactful problem — child speech is highly variable (age, pronunciation development, L1 transfer). The phonetic track (IPA transcription) adds extra difficulty: models must output fine-grained phonetic sequences, not just words. I was drawn to the unique combination of ASR + phonetics + low-resource child speech.

### 3. High-Level Approach

**11-model heterogeneous ensemble + CatBoost tree reranker**:

| Component | Details |
|---|---|
| **NeMo Parakeet-TDT-0.6B** (3 TDT + 4 CTC) | Main backbone. TDT converges fast (~5 epochs), excels on long audio & DD data |
| **WavLM-Large** (4 CTC models) | Excels on short audio & EXT (TalkBank) data. 5h/epoch on Pro 6000 |
| **Dual-head training** | IPA CTC/TDT head + Word BPE CTC head (weight 0.3) — leverages larger word-track data |
| **Concat Mix augmentation** | Concatenate up to 8 audio clips + labels; most impactful single augmentation |
| **CatBoost LambdaRank reranker** | 5-fold CV, ~200 features, selects best N-best candidate per utterance |

```
Raw Audio → 11 Models (N-best) → CTC Rescoring → CatBoost Reranker → Final IPA
```

**Key insight**: WavLM excels on short/EXT audio, NeMo TDT excels on long/DD audio — this complementarity is the foundation of the ensemble.

### 4. Visualizations

**Single model CV (fold 0, IPA CER ↓)**:

| Model | Overall | DD | EXT |
|---|---|---|---|
| v17.wavlm-large.ep3.5 | **0.2900** | 0.3445 | **0.2355** |
| v16.wavlm-large.dual_bpe.mix4 | 0.2923 | 0.3477 | 0.2368 |
| v16.dual_bpe.mix2.mix_csss.tdt_only | 0.2913 | **0.3416** | 0.2410 |
| v16.dual_bpe.tdt_only | 0.2931 | 0.3447 | 0.2415 |
| v16.dual_bpe.mix4 | 0.2928 | 0.3393 | 0.2464 |
| v16.aux_loss.dual_bpe | 0.2952 | 0.3418 | 0.2486 |

**Ensemble CV (fold 0)**:

| Method | CER |
|---|---|
| Best single model | 0.2900 |
| Baseline (best-of-models per utt) | 0.2724 |
| Full-avg MBR | 0.2672 |
| **CatBoost Reranker (final)** | **0.2628** |
| Oracle (upper bound) | 0.1685 |

### 5. Three Most Impactful Code Segments

#### 5.1 Concat Mix Augmentation (`dataset.py`)

```python
# Concatenate multiple audio clips + labels for data augmentation
# Most impactful single augmentation technique
def _concat_mix(self, batch_items, max_mix=8):
    """Randomly concat up to max_mix audio segments with their IPA labels.
    Strategies:
    - 'csss': select from DD/EXT with equal probability (DD-friendly)
    - 'random': uniform random from DD+EXT combined pool
    - fit_cost: limit total cost = audio_sec * label_units <= max_cost(120)
    """
    mixed_audio = []
    mixed_labels = []
    for item in selected_items:
        mixed_audio.append(item['audio'])
        mixed_labels.append(item['label'])
    return torch.cat(mixed_audio), ' '.join(mixed_labels)
```

**Impact**: Child speech utterances are very short (1-3 words). Concatenating clips creates longer, more diverse training examples. No-mix → mix8 improved single-model CER by ~0.01 absolute.

#### 5.2 Dual-Head Training (`models/nemo.py`)

```python
# Two decoder heads sharing one encoder
# Primary: IPA phonetic CTC/TDT (vocab=53)
# Secondary: Word BPE CTC (weight=0.3) — leverages larger word-track data
enc_out = self.encoder(audio)
phon_loss = self.phonetic_head(enc_out, phonetic_labels)
word_loss = self.word_bpe_head(enc_out, word_labels)
total_loss = phon_loss + 0.3 * word_loss
```

**Impact**: Word track has ~5x more training data. Shared encoder benefits from this larger dataset. Improved local CV by ~0.003 absolute.

#### 5.3 CatBoost Tree Reranker (`ensemble.py` + `ensemble_feats.py`)

```python
# Build feature matrix: CTC scores (per model), edit distances, audio features
features = build_reranker_features(candidates_df, ctc_scores, model_names)
# CatBoost LambdaRank selects best candidate per utterance
catboost_model = CatBoostRanker(
    iterations=1000, learning_rate=0.05, depth=6,
    loss_function='YetiRankPairwise',
)
catboost_model.fit(X_train, y_train, group_id=group_train)
```

**Impact**: Learns WavLM scores matter more for short/EXT audio, NeMo TDT for long/DD audio, edit distance consensus is a strong signal. Improved ensemble CER from ~0.272 → ~0.263.

### 6. Machine Specs & Time

Training was performed across a mix of a local workstation and rented cloud GPUs:

| Resource | Local workstation | Rented (NeMo training) | Rented (WavLM training) |
|---|---|---|---|
| **CPU** | Intel(R) Xeon(R) Platinum 8336C @ 2.30 GHz | (cloud, comparable Xeon class) | (cloud, comparable Xeon class) |
| **GPU** | NVIDIA RTX 4090 (24 GB) | NVIDIA RTX 5090 (32 GB) | NVIDIA RTX PRO 6000 (96 GB) |
| **System RAM** | 1024 GB | 62 GB | 120 GB |
| **OS** | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

**Train duration** (per single model):
- NeMo Parakeet-TDT-0.6B: **~1-2 hours** (5 epochs, RTX 4090/5090)
- WavLM-Large: **~15-25 hours** (3-5 epochs, RTX PRO 6000)
- Total wall-clock for the 11-model ensemble: **~100-150 GPU hours**
- CatBoost reranker (5-fold): **~30 min** (CPU)

**Inference duration**:
- Final 11-model ensemble + CatBoost reranker on the competition runtime (NVIDIA A100 80 GB): **~30-56 minutes** for the full test set.

### 7. Caveats & Known Issues

1. **Inference time**: 11-model ensemble takes ~56 min on A100, tight for competition time limit. Inference order matters — run TDT models first (slow but don't need CTC logprobs), then CTC models (fast with NeMo fast decode ~580 utt/s).
2. **EXT CV bias**: Early local CV for TalkBank data was overly optimistic. Fixed by ensuring proper `child_id` grouping in `StratifiedGroupKFold`. Corrected CV aligned well with LB (offset ≈ -0.008).
3. **WavLM memory**: WavLM-Large requires significant GPU memory. Used gradient accumulation on Pro 6000.
4. **bfloat16**: Inference uses bfloat16 with careful handling of CTC log-probability underflow.
5. **NeMo version**: Requires `nemo_toolkit>=2.0` for TDT decoding support.

### 8. Tools for Data Preparation / EDA

- **jiwer**: Official CER/WER computation
- **NeMo toolkit**: Model training, TDT/CTC decoding
- **Transformers (HuggingFace)**: WavLM backbone
- **CatBoost / LightGBM**: Tree reranker
- **librosa / soundfile**: Audio loading and processing
- **ICE cream (ic)**: Debug logging
- **tqdm**: Progress tracking

### 9. Performance Evaluation Beyond Competition Metric

Hierarchical breakdowns by multiple dimensions:

| Dimension | Slices |
|---|---|
| **Data source** | DD (DrivenData) vs EXT (TalkBank) |
| **Age group** | 3-4 years vs 5+ years |
| **Audio duration** | Short (<3s) / Medium (3-10s) / Long (>10s) |
| **Word count** | 1 word / 2-5 words / 6+ words |
| **Combined** | DD/3-4, DD/5+, EXT/3-4, EXT/5+ |

This breakdown revealed the key insight: WavLM excels on short/EXT, NeMo TDT excels on long/DD → drove the heterogeneous ensemble design.

### 10. Things Tried That Didn't Make Final

| Approach | Result |
|---|---|
| Whisper-small backbone | Vastly inferior to NeMo (0.56 vs 0.30 CER) |
| Parakeet-TDT-1.1B | Marginal gain over 0.6B, too slow for ensemble |
| Focal CTC loss | No clear improvement |
| Pseudo-label training | Mixed results |
| N-gram LM shallow fusion | Minimal improvement for IPA sequences |
| MCER (Minimum CER) loss | Experimented but not in final |
| LoRA fine-tuning | Full fine-tuning worked better |
| Logits/Prob-level ensemble | Inferior to N-best + tree reranker |
| Pure MBR ensemble | Good but tree reranker consistently better |
| Dual char head (vs BPE) | Inferior to dual BPE head |

### 11. Future Improvements

1. **More WavLM variants** with different augmentations — WavLM showed the most complementary performance
2. **IPA-specific language model** for N-best rescoring
3. **Cross-utterance context** — children's speech has contextual cues from surrounding utterances
4. **Self-training / pseudo-labels** on unlabeled TalkBank audio
5. **Larger pre-trained models** (NeMo Parakeet-1.1B or Canary-1B)

### 12. Simplifications for Faster Inference

| Simplification | Speed Gain | Accuracy Cost |
|---|---|---|
| 3 models (1 WavLM + 1 TDT + 1 CTC) | ~4x faster | ~0.01-0.02 CER |
| Remove tree reranker, use N-best rescore | ~10% faster | ~0.005 CER |
| Reduce beam_width 10→5 | ~30% faster | ~0.002 CER |
| CTC-only decoding (skip TDT) | ~20% faster | ~0.003 CER |

A minimal 3-model pipeline with N-best rescore would achieve ~0.27 CER in ~15 min on A100.

---

## Submission History

| Ver | Date | Public | Private | Key Change |
|---|---|---|---|---|
| 1 | 02-16 | 0.5468 | 0.5464 | Whisper CTC baseline (d_model=384) |
| 2 | 02-16 | 0.5620 | 0.5620 | Whisper CTC, full train |
| 3 | 02-16 | 0.5621 | 0.5626 | Whisper CTC, lr=1e-4 |
| 4 | 02-24 | 0.3009 | 0.3017 | **NeMo Parakeet-TDT-0.6B** + EXT data |
| 5 | 02-26 | 0.2941 | 0.2936 | Head lr(1e-4), corpus_level_loss |
| 6 | 03-01 | 0.2884 | 0.2893 | New submit framework, ctc_weight=1.0 |
| 7 | 03-03 | 0.2886 | 0.2893 | Pretrain word→IPA (no gain) |
| 8 | 03-06 | 0.2830 | 0.2835 | **First ensemble**: 4-model MBR |
| 9 | 03-08 | 0.2781 | 0.2778 | 6-model N-best rescore |
| 10 | 03-11 | 0.2754 | 0.2761 | 10-model N-best rescore |
| 11 | 03-16 | 0.2716 | 0.2733 | 5-model optimized pipeline (1 TDT + 4 CTC) |
| 12 | 03-18 | 0.2687 | 0.2705 | **First tree reranker**: 8 models + CatBoost |
| 13 | 03-19 | 0.2605 | 0.2620 | **WavLM-Large added**: 9 heterogeneous models |
| 14 | 03-23 | 0.2582 | 0.2598 | 10 models (3 WavLM + 7 NeMo) |
| 15 | 03-25 | 0.2551 | 0.2569 | 10 models, full train (offline→online) |
| 16 | 04-06 | **0.2539** | **0.2559** | **Final**: 11 models (4 WavLM + 7 NeMo) + tree reranker |

### Evolution Phases

| Phase | Submissions | Core Innovation | Relative Improvement |
|---|---|---|---|
| **Whisper Baseline** | v1→v3 | Whisper CTC small model | Baseline (0.547) |
| **NeMo Backbone** | v3→v4 | Switch to Parakeet-TDT-0.6B | **-46.4%** (0.562→0.301) |
| **Single Model Tuning** | v4→v7 | Head lr, corpus loss, EXT data | -4.1% (0.301→0.289) |
| **Ensemble** | v7→v11 | MBR → N-best rescore | -6.0% (0.289→0.272) |
| **Tree Reranker** | v11→v12 | CatBoost LambdaRank | -1.1% (0.272→0.269) |
| **Heterogeneous** | v12→v16 | Add WavLM-Large models | -5.6% (0.269→0.254) |
| **Overall** | v1→v16 | Full pipeline | **-53.6%** (0.547→0.254) |

---

*Solution by Chenghuige. Code authored with GPT-4 and Claude assistance.*
