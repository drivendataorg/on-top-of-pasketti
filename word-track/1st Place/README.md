# DRIVENDATA Children’s Speech Recognition Challenge - Word Track - 1st place solution

## Environment

- python: 3.12
- uv: latest
- GPU: Ampere generation or later (tested on RTX 5090)

## 1. Prepare Dataset

```bash
uv sync --package drivendata-csrc --package csrc --package csrc_prep --extra nemo
git clone https://github.com/NVIDIA-NeMo/NeMo.git && git -C NeMo checkout e66e26b
uv run hf download nvidia/parakeet-ctc-1.1b --local-dir input/parakeet-ctc-1.1b
```

### Before Preprocess

```text
.input
|-- csrc-smoke
|   `-- submission_format_z2HCh3r.jsonl
|-- raw-csrc
|   |-- audio_part_0.zip
|   |-- audio_part_1.zip
|   |-- audio_part_2.zip
|   `-- train_word_transcripts.jsonl
`-- raw-talkbank
    |-- audio_flac.zip
    `-- train_word_transcripts.jsonl
```

### After Preprocess

```bash
./prepare_dataset.sh
```

```text
.input
|-- csrc-processed-input
|   |-- audio
|   `-- train.csv
|-- csrc-processed-talkbank
|   |-- audio
|   `-- train.csv
|-- csrc-smoke
|   |-- audio
|   |-- smoke.csv
|   `-- val_manifest.json
```

## 2. Parakeet Part

### 2.1. setup

```bash
uv sync --package drivendata-csrc --package csrc --package csrc_parakeet --extra nemo
```

### 2.3. fine tuning

```bash
uv run csrc_parakeet/src/csrc_parakeet/manifest.py val input/csrc-smoke/smoke.csv input/csrc-smoke/audio input/csrc-smoke/val_manifest.json
uv run csrc_parakeet/src/csrc_parakeet/manifest.py train csrc_parakeet/configs/parakeet_exp013.yaml
uv run csrc_parakeet/src/csrc_parakeet/finetune.py csrc_parakeet/configs/parakeet_exp013.yaml
```

### 2.4. predict all samples

```bash
uv run csrc_parakeet/src/csrc_parakeet/predict_val.py \
  output/parakeet_exp013/best_score_path.nemo \
  output/parakeet_exp013/train_manifest.json \
  output/parakeet_exp013

# rename
mv output/parakeet_exp013/val_pred.csv output/parakeet_exp013/train_pred.csv
```

### 2.5. merge pred

```bash
uv run csrc_parakeet/src/csrc_parakeet/merge_pred.py \
  --pred-csv output/_parakeet_exp013/train_pred.csv \
  --input-train-csv input/csrc-processed-input/train.csv \
  --talkbank-train-csv input/csrc-processed-talkbank/train.csv \
  --output-input-csv input/csrc-processed-input/train_pred_parakeet_exp013.csv \
  --output-talkbank-csv input/csrc-processed-talkbank/train_pred_parakeet_exp013.csv
```

## 3. Qwen-3-ASR-1.7B Part

### 3.1. setup

```bash
uv sync --package drivendata-csrc --package csrc --package csrc_qwen --extra vllm --extra flash-attn
```

### 3.2. fine tuning

```bash
uv run csrc_qwen/src/csrc_qwen/manifest.py val input/csrc-smoke/smoke.csv input/csrc-smoke/audio input/csrc-smoke/val_manifest.json
uv run csrc_qwen/src/csrc_qwen/manifest.py train csrc_qwen/configs/qwen_exp022.yaml
uv run csrc_qwen/src/csrc_qwen/manifest.py train csrc_qwen/configs/qwen_exp023.yaml
uv run csrc_qwen/src/csrc_qwen/manifest.py train csrc_qwen/configs/qwen_exp025.yaml
uv run csrc_qwen/src/csrc_qwen/manifest.py train csrc_qwen/configs/qwen_exp026.yaml
```

#### merge adapter

```bash
uv run csrc_qwen/src/csrc_qwen/merge_adapter.py qwen/Qwen3-ASR-1.7B path/to/checkpoint path/to/output_dir
```

#### predict val

```bash
uv run csrc_qwen/src/csrc_qwen/predict_val.py input/csrc-smoke/val_manifest.json path/to/merged_path path/to/merged_path
```

### 3.3. average adapters

- Get validation WER for all checkpoints and select approximately 3 appropriate checkpoints

```bash
uv run csrc_qwen/src/csrc_qwen/average_adapters.py path/to/checkpoint_1 path/to/checkpoint_2 path/to/checkpoint_3 -o path/to/output
```

### 3.4. average models

- Average the merged models (adapters already averaged and merged)

```bash
uv run csrc_qwen/src/csrc_qwen/average_models.py path/to/merged_model_1 path/to/merged_model_2 path/to/merged_model_3 -o path/to/output
```

## Inference (generate submission.zip)

clone runtime repository

```bash
git clone https://github.com/drivendataorg/childrens-speech-recognition-runtime.git
```

```text
.runtime
├── configs
│   └── qwen_config.yaml
├── pack_submission.sh
└── src
    ├── main.py
    ├── model
```

qwen_config.yaml

```yaml
model:
  type: "qwen"
  name_or_path: "model_dir_name"
  batch_size: 64
  language: "English"
  max_new_tokens: 300
  max_inference_batch_size: 32
```

### pack

```bash
./runtime/pack_submission.sh runtime/configs/qwen_config.yaml ./childrens-speech-recognition-runtime/submission
```

## Score

| Experiment                   | model                | method                   | Smoke WER | Public WER | Noisy WER |
| ---------------------------- | -------------------- | ------------------------ | --------- | ---------- | --------- |
| no finetuning                | parakeet-tdt-0.6b-v3 | -                        | -         | 0.3202     | 0.5680    |
| parakeet_exp007              | parakeet-tdt-0.6b-v2 | full finetuning (epoch1) | 0.2175    | 0.2404     | 0.5956    |
| parakeet_exp008              | parakeet-tdt-0.6b-v2 | adapter (epoch1)         | 0.2013    | 0.2350     | 0.6122    |
| parakeet_exp012              | parakeet-tdt-0.6b-v3 | adapter (epoch3)         | 0.1955    | 0.2347     | 0.5609    |
| parakeet_exp013              | parakeet-tdt-0.6b-v2 | adapter (epoch3)         | 0.1900    | NA         | NA        |
| qwen_exp022                  | Qwen3-ASR-1.7B       | LoRA (epoch3)            | 0.1632    | 0.1977     | 0.4973    |
| qwen_exp023 (checkpoint avg) | Qwen3-ASR-1.7B       | LoRA (epoch3)            | 0.1631    | NA         | NA        |
| model_avg (022+023)          | -                    | -                        | 0.1593    | 0.1914     | 0.4879    |
| qwen_exp025 (checkpoint avg) | Qwen3-ASR-1.7B       | LoRA (epoch3)            | 0.1618    | NA         | NA        |
| qwen_exp026 (checkpoint avg) | Qwen3-ASR-1.7B       | LoRA (epoch3)            | 0.1578    | NA         | NA        |
| model_avg (022+023+025+026)  | -                    | -                        | NA        | 0.1885     | 0.4842    |

## Reproducibility

- Forced alignment was originally run on the main branch, but the latest commit broke this functionality. We now checkout a specific commit hash. This may affect forced alignment results.
- We found that differences in data preparation steps can cause slight deviations from the original submission results.
- Since Qwen3-ASR is trained on data filtered by Parakeet predictions, errors may propagate.
- Qwen inference results may vary across different environments. Specifically, we observed a WER difference of approximately 0.0020 between local and submission environments.
