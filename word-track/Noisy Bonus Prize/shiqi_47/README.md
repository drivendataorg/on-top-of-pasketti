# Solution — On Top of Pasketti: Children’s Speech Recognition Challenge (Word Track) https://www.drivendata.org/competitions/308/childrens-word-asr/page/973/

Username: shiqi_47

License:  Apache 2.0

## Summary

Full-parameter fine-tuning of Qwen3-ASR-1.7B on clean, non-augmented data: competition training set (~86k utterances) + TalkBank children's speech corpus (~255k utterances). No data augmentation of any kind.

Pipeline:
1. Convert competition and TalkBank transcripts to Qwen3-ASR training format
2. Merge, shuffle, split 97/3 into train/eval (328,830 train / 10,231 eval)
3. Fine-tune Qwen3-ASR-1.7B for 2 epochs, linear LR decay, best checkpoint at step 16,000
4. Inference via vLLM with greedy decoding

Private leaderboard: Clean WER 0.2115 / Noisy WER 0.4919

# Setup

1. Python 3.10+ (Ubuntu 22.04 system Python or conda)

2. System dependencies:
```bash
sudo apt update && sudo apt install -y ffmpeg libsndfile1 build-essential cmake
```

3. Python packages:
```bash
pip install -r requirements.txt
```
4. Pre-trained base model: `Qwen3-ASR/Qwen3-ASR-1.7B` (from Hugging Face or local)
Hugging Face : https://huggingface.co/Qwen/Qwen3-ASR-1.7B


5. Same data layout as above.

## Repository Structure

```
word_track_solution/
├── README.md
├── LICENSE.txt
├── requirements.txt
├── setup.py
├── run_train.sh               ← Training entry point (shell)
├── run_inference.sh            ← Inference entry point (shell)
├── data/
│   ├── audio/                                        ← Competition audio files (.flac)
│   ├── train_word_transcripts.jsonl                   ← Competition transcripts
│   ├── talkbank_audio/audio/                          ← TalkBank audio files
│   ├── TalkBank_corpus_train_word_transcripts.jsonl   ← TalkBank transcripts
│   ├── utterance_metadata.jsonl    ← Sample utterance metadata
│   ├── submission_format.jsonl     ← Sample submission template
│   └── pure/
│       ├── train.jsonl             ← Training split (328,830 samples)
│       ├── eval.jsonl              ← Eval split (10,231 samples)
│       └── test.jsonl              ← Local test split (2,000 samples)
├── model/                          ← Trained model weights + configs
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── generation_config.json
│   ├── preprocessor_config.json
│   └── ...
├── reports/
│   └── model_documentation.md      ← Model write-up (competition Q&A)
└── src/
    ├── prepare_train_data.py       ← Convert competition transcripts
    ├── prepare_talkbank.py         ← Convert TalkBank transcripts
    ├── prepare_pure.py             ← Merge + shuffle + 97/3 split
    ├── run_train.py                ← Training (full-parameter fine-tuning)
    ├── run_inference.py            ← Inference via vLLM
    └── eval_wer.py                 ← WER evaluation
```

# Hardware

- CPU: Intel i9-10850K (10C/20T, 3.6–5.2 GHz)
- RAM: 64 GB DDR4
- GPU: NVIDIA RTX 3090 24 GB
- OS: Ubuntu 22.04, Linux x86_64

Training time: ~20 hours (16,000 steps to best checkpoint, single GPU; manually stopped after eval plateau)
Inference time: ~15 minutes for ~10k utterances (vLLM, batch_size=64)

# Run Training

## Data Preparation

### Step 1: Convert competition transcripts
```bash
python src/prepare_train_data.py \
    --data_dir data \
    --transcript_file data/train_word_transcripts.jsonl \
    --output_file data/train.jsonl
```
Output: `data/train.jsonl` (~86k samples)

### Step 2: Convert TalkBank transcripts
```bash
python src/prepare_talkbank.py \
    --data_dir data/talkbank_audio \
    --transcript_file data/TalkBank_corpus_train_word_transcripts.jsonl \
    --output_file data/talkbank_train.jsonl
```
Output: `data/talkbank_train.jsonl` (~255k samples)

### Step 3: Merge and split
```bash
python src/prepare_pure.py
```

Output (seed=42, 97/3 split):
- `data/pure/train.jsonl` — 328,830 samples (25.2% competition, 74.8% TalkBank)
- `data/pure/eval.jsonl` — 10,231 samples

Note: `data/pure/test.jsonl` (2,000 samples) was manually split from the training set prior to the 97/3 split, used for local WER evaluation. It has no overlap with train or eval.

## Training

```
bash run_train.sh
```
or

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python -u src/train.py \
    --model_path Qwen3-ASR/Qwen3-ASR-1.7B \
    --train_file data/pure/train.jsonl \
    --eval_file data/pure/eval.jsonl \
    --output_dir checkpoints \
    --batch_size 3 --eval_batch_size 1 --grad_acc 12 \
    --lr 1e-5 --epochs 2 --warmup_ratio 0.02 \
    --save_steps 2000 --log_steps 50 --save_total_limit 5 \
    --num_workers 8 --gradient_checkpointing 1 \
    --precache_workers 16 --max_length 4096
```

| Parameter | Value |
|-----------|-------|
| Effective batch size | 36 (3 × 12 grad_acc) |
| Learning rate | 1e-5 (linear decay) |
| Warmup | 2% of total steps |
| Precision | bf16 |
| FlashAttention 2 | Enabled |
| Gradient checkpointing | Enabled |
| Total steps | 18,270 (planned); stopped at 16,000 |

Best checkpoint: step 16,000 (eval_loss = 0.1936)

## Model Weights

Trained weights: 
https://drive.google.com/file/d/1DkcD-z5xiaS6aOaH3ziOTZFzYeB3uxXv/view?usp=sharing

Then extract weights to workdir "model/":

Model size: ~3.8 GB (model.safetensors)

# Run Inference
```
bash run_inference.sh
```
or

```bash
python src/run_inference.py
```

By default, predictions are saved to `submission/submission.jsonl`.

## Evaluate WER

```bash
python src/eval_wer.py --model_path model/ --eval_file data/utterance_metadata.jsonl
```
