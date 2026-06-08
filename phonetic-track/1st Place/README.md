# Pasketti Phonetic ASR — DrivenData prize-winner solution

> Reproduction code for the **Pasketti Phonetic** track of the
> DrivenData [Pasketti Speech Recognition Challenge](https://www.drivendata.org/competitions/),
> public LB **0.2539**, private LB **0.2559**.

[GitHub code release](https://github.com/chenghuige/pasketti-phonetic-solution) | [Hugging Face weights](https://huggingface.co/huigecheng/pasketti-phonetic-weights) | [Release notes](docs/RELEASE.md)

## Submission information

The full Section III write-up (12 questions, machine specs, charts, code highlights, etc.) lives in [`docs/SOLUTION.md`](docs/SOLUTION.md).

This repository is intentionally minimal: it bundles the exact training
and inference code used to produce the leaderboard score, packaged so it
can be run end-to-end without any of the author's internal libraries. A
small compatibility layer under `src/_compat/` provides just enough of
the `gezi` / `melt` / `lele` interface that the project files expect, so
the model code itself is unchanged from the development repository.

The deeper write-up of the modeling choices is in
[`docs/SOLUTION.md`](docs/SOLUTION.md). The word-track solution is **not**
included.

## Release status

This public release is split into two artifacts:

| Artifact | Contents | Link |
| --- | --- | --- |
| GitHub repository | Training code, inference code, packaging scripts, notebook demo, compatibility shims | [chenghuige/pasketti-phonetic-solution](https://github.com/chenghuige/pasketti-phonetic-solution) |
| Hugging Face model repo | Final 11-model online checkpoints plus 5-fold CatBoost reranker artifacts | [huigecheng/pasketti-phonetic-weights](https://huggingface.co/huigecheng/pasketti-phonetic-weights) |

If you want the exact published inference path, you do not need to retrain
the ensemble from scratch. Download the released weights, stage the
competition data under `../input/childrens-phonetic-asr/`, and build the submission
bundle directly.

---

## 1. Solution at a glance

| Component                  | Choice                                                           |
| -------------------------- | ---------------------------------------------------------------- |
| Acoustic backbones         | NeMo Parakeet-TDT-0.6B (TDT + CTC), WavLM-Large (CTC)            |
| Output units               | IPA phoneme set (dual-head IPA + word-BPE during training)       |
| Augmentation               | concat-mix (up to 8 clips), light classroom noise overlay     |
| Decoder                    | Beam-search CTC + TDT, top-10 N-best per model                   |
| Model averaging            | EMA (decay 0.999) saved as the final checkpoint                  |
| Ensemble                   | 11 models → cross-model CTC log-prob rescore → CatBoost LambdaRank |
| Final reranker             | CatBoost, 5-fold, ≈200 features                                  |

The final ensemble model list is in [`src/models.txt`](src/models.txt).

---

## 2. Repository layout

```
pasketti-phonetic-solution/
├── Makefile               # one-command targets (setup / train / ensemble / pack)
├── Dockerfile             # mirrors the DrivenData runtime (for local end-to-end tests)
├── requirements.txt
├── docs/SOLUTION.md       # detailed methodology
├── notebooks/
│   └── 02_run_inference.ipynb   # all-in-one single-model demo
├── scripts/
│   ├── pack_submission.sh       # build submission.zip
│   ├── _resolve_models.py       # resolve names in models.txt to dirs
│   └── sync_core_from_pikachu.sh# (maintainer-only) re-sync core files
├── src/
│   ├── train.py           # standalone training entry (was main.py upstream)
│   ├── train_loop.py      # hand-written AMP / EMA / cosine-LR loop
│   ├── config.py / config_base.py   # absl flag definitions
│   ├── dataset.py         # data + collate + bucket sampler
│   ├── eval.py            # IPA CER metric (matches official scorer)
│   ├── ctc_decode.py      # beam search
│   ├── submit.py          # Docker-runtime entry (renamed to main.py at pack time)
│   ├── ensemble.py        # cross-model rescore + CatBoost reranker
│   ├── tree_reranker/     # saved CatBoost reranker artifacts for final inference
│   ├── models/            # base.py + nemo.py + wav2vec2.py
│   ├── flags/             # versioned flag files (base, v8 … v17)
│   ├── models.txt         # names of the 11 models in the final ensemble
│   └── _compat/           # tiny gezi / melt / lele / husky shims
└── working/               # populated by training (model checkpoints, logs, metrics)
```

`src/_compat/` is the only piece of "infrastructure" code in this
repository — it implements about ~400 lines of helpers (a `Globals`
singleton, EMA-aware checkpoint loader, length-bucketed sampler, etc.)
so that the project files can keep using their original imports
(`from gezi.common import *`, `import lele as le`, `melt.init`).

> **Why `absl.flags` instead of a plain `class FLAGS`?** The training
> code defines roughly 500 flags split across `config_base.py` and
> `config.py` with default-overrides per `flags/v*` file. Switching to a
> hand-rolled config object would have meant rewriting every flag file
> as well. Using absl keeps the surface identical to the development
> setup while remaining a single ~30-line dependency.

---

## 3. Reproducing the result

### 3.0 Fast path: reproduce the released inference bundle

For most users, this is the intended path:

```bash
make setup
make data
HF_REPO_ID=huigecheng/pasketti-phonetic-weights bash scripts/download_weights.sh
make pack
```

This downloads the released online checkpoints into `working/online/17/`
and the CatBoost reranker artifacts into `src/tree_reranker/`, then builds
`submission.zip`.

### 3.1 Hardware

There are two practical usage modes:

* **Released inference bundle / `make pack` path:** no retraining is required. You mainly need enough disk to store the competition data plus the published checkpoints downloaded from Hugging Face. `make smoke` is CPU-only.
* **Full training / reproduction from scratch:** use a high-VRAM GPU. In the original runs, most training was done on a single 5090 32 GB, while some WavLM-Large-related runs used a single RTX PRO 6000 96 GB-class GPU.
* **Online inference for the final release pipeline:** the original online inference / release-time bundle generation was run on a single A100 80 GB GPU.

For the full training path, plan for:

* ~80 GB disk for raw audio + intermediate features.
* CUDA 12.1 / 12.4 with cuDNN.

### 3.2 Environment

```bash
make setup          # pip install -r requirements.txt
make data           # prints the expected .flac/jsonl layout under ../input/
make smoke          # import-only sanity check, no GPU required
```

The training code reads the official `.flac` files directly through
`soundfile` / `librosa`; no `.wav` conversion step is used. The most
reliable setup is to stage every dataset under one shared `input/` parent
with the exact directory and file names below.

Expected training-data layout:

```text
../input/
├── childrens-phonetic-asr/                 # official phonetic-track data
│   ├── train_phon_transcripts.jsonl
│   ├── audio.html                          # optional, if present in the download
│   └── audio/
│       └── <utterance_id>.flac
├── childrens-ext-asr/                      # official EXT/TalkBank data used here
│   ├── train_phon_transcripts.jsonl
│   ├── train_word_transcripts.jsonl
│   ├── audio.html                          # optional, if present in the download
│   └── audio/
│       └── <utterance_id>.flac
├── childrens-word-asr/                     # official word-track labels for cross-label training
│   ├── train_word_transcripts.jsonl
│   └── audio/
│       └── <utterance_id>.flac
├── childrens-classnoise-asr/               # classroom-noise augmentation clips
│   └── audio/
│       └── <noise_id>.flac
└── fold_align_phonetic.json                # optional fold-alignment helper, if available
```

Only `childrens-phonetic-asr/` is strictly required for a minimal smoke
training run. The released 11-model recipe and tree-reranker reproduction
use the EXT data and cross-label/auxiliary resources as shown above,
especially `childrens-ext-asr/train_word_transcripts.jsonl` and
`childrens-word-asr/train_word_transcripts.jsonl`.

If your files live elsewhere, either symlink those directories or pass
`DATA_DIR=/path/to/childrens-phonetic-asr EXT_DATA_DIR=/path/to/childrens-ext-asr`
to the `make` targets. On my local machine, the same datasets are staged in
the sibling development checkout and are auto-detected by the reproduction
scripts and `make data`:

```text
../pasketti-phonetic/input/
├── childrens-classnoise-asr/
├── childrens-ext-asr/
├── childrens-phonetic-asr/
├── childrens-word-asr/
├── childrens-pseudo-ipa/
├── childrens-pseudo-ipa-dd/
├── childrens-pseudo-ipa2/
└── fold_align_phonetic.json
```

Example staging commands for a fresh checkout:

```bash
mkdir -p ../input
ln -s /path/to/childrens-phonetic-asr   ../input/childrens-phonetic-asr
ln -s /path/to/childrens-ext-asr        ../input/childrens-ext-asr
ln -s /path/to/childrens-word-asr       ../input/childrens-word-asr
ln -s /path/to/childrens-classnoise-asr ../input/childrens-classnoise-asr
# Optional:
ln -s /path/to/fold_align_phonetic.json ../input/fold_align_phonetic.json
```

For local inference in the DrivenData runtime, keep the runtime's normal
structure with `submission_format.jsonl` and `audio/<utterance_id>.flac`
under `/code_execution/data`.

### 3.3 Train a single model (fold 0)

```bash
make train-fold0 GPU=1            # -> working/offline/17/v17.fold0/0/
```

Each `flags/v*` file is incremental: `v17` chains all the way back to
`base` via `--flagfile`. The full ensemble retrains the same recipe with
different backbones and fixed epoch counts; see `src/models.txt` for the
exact final 11-model list.

Equivalent explicit command. Set `DATA_PARENT` to the parent directory that contains
`childrens-phonetic-asr/` and `childrens-ext-asr/`. If you staged data as
`../input/` from the repository root, then from `src/` this is `../../input`.
On my local machine it is `../../pasketti-phonetic/input`.

```bash
cd src
DATA_PARENT=${DATA_PARENT:-../../pasketti-phonetic/input}  # use ../../input for a fresh ../input staging layout
PYTHONPATH=_compat:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python train.py \
  --flagfile=flags/v17 \
  --mn=v17.fold0 \
  --fold=0 \
  --root=$DATA_PARENT/childrens-phonetic-asr \
  --ext_root=$DATA_PARENT/childrens-ext-asr \
  --eval_ext_root=$DATA_PARENT/childrens-ext-asr
```

### 3.4 Full 11-model reproduction scripts

The repository includes helper scripts under `src/` that reproduce the
published 11 acoustic models and the second-stage tree reranker. They use
the model names in `src/models.txt` and automatically add the eval/export
flags needed by the reranker (`--eval_ext_full`, `--save_logprobs`,
`--save_dual_head_preds`, `--save_pred_score`).

First check that the expected data paths are visible:

```bash
make data
```

Then run the scripts from `src/`:

```bash
cd src

# 1) Offline fold-0 models used to train the tree reranker.
#    Outputs: working/offline/9/<model_name>/0/{eval.csv,ctc_logprobs.pt,dual_head_preds.pt,...}
bash reproduce_offline_fold0.sh

# 2) Second-stage CatBoost reranker trained from the offline fold-0 artifacts.
#    Outputs: working/offline/9/ensemble.feat_nemo_group.feat_tdt_group.feat_wavlm_group.0407/0/
#    and copies release artifacts to src/tree_reranker/ by default.
bash reproduce_tree_reranker.sh

# 3) Final online/full-data acoustic models for submission packaging.
#    Outputs: working/online/9/<model_name>/0/
bash reproduce_online.sh
```

Useful environment variables:

```bash
GPU=1 bash reproduce_offline_fold0.sh
FORCE=1 bash reproduce_offline_fold0.sh       # rerun even if model.pt exists
DRY_RUN=1 bash reproduce_offline_fold0.sh     # print commands only

ROOT=/path/to/childrens-phonetic-asr \
EXT_ROOT=/path/to/childrens-ext-asr \
bash reproduce_offline_fold0.sh

EXTRA_ARGS="--bs=1 --eval_bs=1 --num_workers=0" \
bash reproduce_offline_fold0.sh               # smoke/debug run

COPY_TO_RELEASE=0 bash reproduce_tree_reranker.sh

# Use existing offline artifacts from a sibling development checkout:
RUN_ROOT=../../pasketti-phonetic/working/offline/9 bash reproduce_tree_reranker.sh
```

`reproduce_tree_reranker.sh` auto-detects offline fold-0 artifacts in
`../working/offline/9`, `../../pasketti-phonetic/working/offline/9`, and
`../../pasketti/working/offline/9`; set `RUN_ROOT` explicitly if your artifacts
live elsewhere.

`reproduce_online.sh` and `reproduce_offline_fold0.sh` auto-detect data in
`../input/`, `../../input/`, `../../pasketti-phonetic/input/`, and
`../../pasketti/input/` unless `ROOT` / `EXT_ROOT` are provided explicitly.

Note that the Hugging Face release contains the final online/full-data ASR
checkpoints plus the already-trained `src/tree_reranker/` artifacts used by
`make pack`. It does **not** contain the large offline fold-0 eval artifacts
needed to retrain the tree reranker from scratch. To run
`reproduce_tree_reranker.sh`, first generate those artifacts with
`reproduce_offline_fold0.sh`.

### 3.4.1 What a successful `reproduce_tree_reranker.sh` run looks like

A healthy run usually starts with an auto-detected artifact root such as:

```text
Using offline artifact root: ../../pasketti-phonetic/working/offline/9
Found 11 model eval dirs; 8 have ctc_logprobs.pt.
+ PYTHONPATH=_compat:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python ensemble.py --ensemble_working_dir=../../pasketti-phonetic/working/offline/9 --feat_nemo_group --feat_tdt_group --feat_wavlm_group --mns=.0407
```

Then `ensemble.py` should report that it loaded all 11 model directories,
built the reranker feature table, and started 5-fold CatBoost training. Key
milestones from a successful reproduction look like:

```text
Loaded 11 models from /.../src/models.txt
Built 1068582 candidate rows for 30645 utterances
Dataset: 1068582 rows, 212 features
Parallel tree CV enabled: jobs=5, total_cores=128, per_job_tree_threads=25
--- Tree Reranker (cb, 5-fold) Results ---
Overall CER: 0.26307
--- Tree Reranker FullAvg (cb, 5-fold models) Results ---
Overall CER: 0.26086
Copied tree reranker artifacts to tree_reranker
```

The script writes the trained reranker under
`$RUN_ROOT/ensemble.feat_nemo_group.feat_tdt_group.feat_wavlm_group.0407/0/`
and, unless `COPY_TO_RELEASE=0`, also copies the release-time files into
`src/tree_reranker/` for `make pack`.

For a full line-by-line reference from a successful run, see
[`docs/TREE_RERANKER_SUCCESS_LOG.md`](docs/TREE_RERANKER_SUCCESS_LOG.md).

Some warnings in the log are expected and do **not** mean the run failed:

- missing optional `aux_meta_preds.pt`
- missing optional `model.pt`
- missing optional `ctc_logprobs.pt` for some TDT-only models
- CatBoost message `Pairwise losses don't support object weights.`
- CUDA factory registration messages like `Unable to register cuDNN factory`

The hard requirements are simpler: every model must have `eval.csv`, and at
least one model must have `ctc_logprobs.pt`. If the script prints
`Run first: bash reproduce_offline_fold0.sh`, then the required offline fold-0
artifacts were not found at the selected `RUN_ROOT`.

### 3.5 Build the ensemble + reranker

After all 11 models in `src/models.txt` are trained and `src/tree_reranker/`
contains the saved CatBoost artifacts:

```bash
make pack                         # bundles submission.zip from src/models.txt
```

The tree reranker code is already included in the repository:

* `src/ensemble.py` trains the CatBoost reranker and writes the saved tree artifacts.
* `src/reranker_features.py` builds the online/offline feature frame.
* `src/submit.py` loads the packed tree model(s) at inference time.

For the final leaderboard submission, the saved reranker artifacts must be
available under `src/tree_reranker/` before `make pack` is run.

`make pack` copies `submit.py` to `main.py` (the runtime entry expected
by the DrivenData container), tarballs `src/_compat/` as
`pikachu_utils.tar.gz`, and zips everything together with the model
weight directories. If `src/tree_reranker/` exists, it is copied into the
submission bundle as well. The runtime extracts the tar onto `sys.path`
automatically — no edits to `submit.py` are needed.

---

## 4. Published weights and reranker artifacts

The final released checkpoints and reranker artifacts are public at:

* [huigecheng/pasketti-phonetic-weights](https://huggingface.co/huigecheng/pasketti-phonetic-weights)

The supported download flow is:

```bash
python -m pip install -r requirements.txt
HF_REPO_ID=huigecheng/pasketti-phonetic-weights bash scripts/download_weights.sh
```

To additionally download the optional offline fold-0 artifacts used to
retrain the tree reranker without re-running the 11 acoustic models, use:

```bash
DOWNLOAD_OFFLINE=1 HF_REPO_ID=huigecheng/pasketti-phonetic-weights bash scripts/download_weights.sh
```

After the default download, the repo should contain:

```
working/online/17/<model_name>/model.pt
working/online/17/<model_name>/flags.json
working/online/17/<model_name>/nemo_model_slim.nemo   # NeMo backbones only

src/tree_reranker/reranker_meta.json
src/tree_reranker/reranker_features.txt
src/tree_reranker/reranker_experiment.json
src/tree_reranker/tree_cb_fold0/model.pkl
src/tree_reranker/tree_cb_fold1/model.pkl
src/tree_reranker/tree_cb_fold2/model.pkl
src/tree_reranker/tree_cb_fold3/model.pkl
src/tree_reranker/tree_cb_fold4/model.pkl
```

where `<model_name>` matches an entry in `src/models.txt`.

The current Hugging Face repo layout is:

```
online/17/<model_name>/model.pt
online/17/<model_name>/flags.json
online/17/<model_name>/nemo_model_slim.nemo
tree_reranker/reranker_meta.json
tree_reranker/reranker_features.txt
tree_reranker/reranker_experiment.json
tree_reranker/tree_cb_fold0/model.pkl
tree_reranker/tree_cb_fold1/model.pkl
tree_reranker/tree_cb_fold2/model.pkl
tree_reranker/tree_cb_fold3/model.pkl
tree_reranker/tree_cb_fold4/model.pkl

# optional, only when DOWNLOAD_OFFLINE=1 was used:
offline/9/<model_name>/0/eval.csv
offline/9/<model_name>/0/ctc_logprobs.pt
offline/9/<model_name>/0/dual_head_preds.pt
offline/9/<model_name>/0/flags.json
```

To assemble the official DrivenData runtime bundle from the public release:

```bash
HF_REPO_ID=huigecheng/pasketti-phonetic-weights bash scripts/download_weights.sh
make pack
```

Maintainers can re-stage and re-upload the exact final 11-model bundle from
the original training workspace to Hugging Face with:

```bash
HF_REPO_ID=huigecheng/pasketti-phonetic-weights UPLOAD_NOW=1 bash scripts/upload_hf_weights.sh
```

To also stage and upload the optional offline fold-0 reranker-training
artifacts, maintainers can run:

```bash
INCLUDE_OFFLINE_ARTIFACTS=1 HF_REPO_ID=huigecheng/pasketti-phonetic-weights UPLOAD_NOW=1 bash scripts/upload_hf_weights.sh
```

`INCLUDE_OFFLINE_MODEL_PT=1` can also copy offline `model.pt` files, but
those checkpoints are much larger and are not needed by the default
`reproduce_tree_reranker.sh` path.

The GitHub repository intentionally does not commit the large ASR
checkpoints, so the Hugging Face model repo is the authoritative source for
released weights.

## 5. Release contents

This public release includes:

* standalone training and inference code with no dependency on the author's
  internal monorepo;
* the exact final 11-model ensemble list in `src/models.txt`;
* packaging scripts for the DrivenData submission container;
* a helper script to download the released checkpoints and reranker artifacts;
* a notebook demo for running inference locally.

This public release does not include:

* the separate word-track solution;
* the original private training monorepo;
* raw competition data.

---

## 6. License

* Project source code: MIT (see `LICENSE`).
* Pre-trained NeMo Parakeet-TDT-0.6B and WavLM-Large weights: see their
  upstream license terms (CC-BY-NC for Parakeet, MIT for WavLM).
