# =============================================================================
# Pasketti Phonetic ASR — DrivenData prize-winner reproduction
# =============================================================================
# All paths are repo-relative. Targets assume CUDA is available.
# Override DATA_DIR / EXT_DATA_DIR if your competition data lives elsewhere.
# By default, prefer this repo's ../input layout, but also auto-detect Gezi's
# local sibling checkout: ../pasketti-phonetic/input/...
# =============================================================================
PYTHON   ?= python
GPU      ?= 0
DATA_PARENT ?= $(or $(firstword $(wildcard ../input ../pasketti-phonetic/input ../pasketti/input)),../input)
DATA_DIR ?= $(DATA_PARENT)/childrens-phonetic-asr
EXT_DATA_DIR ?= $(DATA_PARENT)/childrens-ext-asr
WORD_DATA_DIR ?= $(DATA_PARENT)/childrens-word-asr
NOISE_DATA_DIR ?= $(DATA_PARENT)/childrens-classnoise-asr
FOLD_ALIGN_FILE ?= $(DATA_PARENT)/fold_align_phonetic.json
DATA_ROOT := $(abspath $(DATA_DIR))
EXT_DATA_ROOT := $(abspath $(EXT_DATA_DIR))
WORD_DATA_ROOT := $(abspath $(WORD_DATA_DIR))
NOISE_DATA_ROOT := $(abspath $(NOISE_DATA_DIR))
FOLD_ALIGN_ROOT := $(abspath $(FOLD_ALIGN_FILE))
WORKING  ?= ./working
RUN      ?= 17

.PHONY: help setup data smoke train-fold0 train-online ensemble pack notebook clean

help:
	@echo "Targets:"
	@echo "  setup        pip install -r requirements.txt"
	@echo "  data         show instructions for staging competition data"
	@echo "  smoke        import-only sanity check (no GPU)"
	@echo "  train-fold0  CUDA_VISIBLE_DEVICES=$(GPU) train v17 on fold 0"
	@echo "  train-online CUDA_VISIBLE_DEVICES=$(GPU) train v17 on full data"
	@echo "  ensemble     run cross-model rescore + CatBoost reranker"
	@echo "  pack         build submission.zip from src/models.txt"
	@echo "  notebook     launch JupyterLab"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	@echo "Data parent: $(abspath $(DATA_PARENT))"
	@echo ""
	@echo "Expected training-data layout (exact directory and file names):"
	@echo "  $(DATA_ROOT)/train_phon_transcripts.jsonl"
	@echo "  $(DATA_ROOT)/audio/<utterance_id>.flac"
	@echo ""
	@echo "Official EXT / TalkBank data used by this solution:"
	@echo "  $(EXT_DATA_ROOT)/train_phon_transcripts.jsonl"
	@echo "  $(EXT_DATA_ROOT)/train_word_transcripts.jsonl"
	@echo "  $(EXT_DATA_ROOT)/audio/<utterance_id>.flac"
	@echo ""
	@echo "Word-track cross-label data used by dual-BPE / auxiliary training:"
	@echo "  $(WORD_DATA_ROOT)/train_word_transcripts.jsonl"
	@echo "  $(WORD_DATA_ROOT)/audio/<utterance_id>.flac"
	@echo ""
	@echo "Classroom-noise augmentation data:"
	@echo "  $(NOISE_DATA_ROOT)/audio/<noise_id>.flac"
	@echo ""
	@echo "Optional fold-alignment helper:"
	@echo "  $(FOLD_ALIGN_ROOT)"
	@echo ""
	@echo "To stage data in a fresh checkout:"
	@echo "  mkdir -p ../input"
	@echo "  ln -s /path/to/childrens-phonetic-asr   ../input/childrens-phonetic-asr"
	@echo "  ln -s /path/to/childrens-ext-asr        ../input/childrens-ext-asr"
	@echo "  ln -s /path/to/childrens-word-asr       ../input/childrens-word-asr"
	@echo "  ln -s /path/to/childrens-classnoise-asr ../input/childrens-classnoise-asr"
	@echo ""
	@echo "For local inference / runtime tests, use DrivenData's runtime layout:"
	@echo "  /code_execution/data/submission_format.jsonl"
	@echo "  /code_execution/data/audio/<utterance_id>.flac"

smoke:
	@PYTHONPATH=src/_compat:$$PYTHONPATH $(PYTHON) -c \
	  "from gezi.common import *; from src import config; print('imports OK')"

train-fold0:
	cd src && PYTHONPATH=_compat:$$PYTHONPATH \
	  CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) train.py \
	    --flagfile=flags/v17 --mn=v17.fold0 --fold=0 \
	    --root=$(DATA_ROOT) --ext_root=$(EXT_DATA_ROOT) --eval_ext_root=$(EXT_DATA_ROOT)

train-online:
	cd src && PYTHONPATH=_compat:$$PYTHONPATH \
	  CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) train.py \
	    --flagfile=flags/v17 --mn=v17.online --online \
	    --root=$(DATA_ROOT) --ext_root=$(EXT_DATA_ROOT) --eval_ext_root=$(EXT_DATA_ROOT)

ensemble:
	cd src && PYTHONPATH=_compat:$$PYTHONPATH \
	  CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) ensemble.py \
	    --models_file=models.txt

pack:
	bash scripts/pack_submission.sh ensemble src/models.txt submission.zip

notebook:
	$(PYTHON) -m jupyterlab notebooks/

clean:
	rm -rf $(WORKING)/tmp __pycache__ src/__pycache__ src/_compat/*/__pycache__
