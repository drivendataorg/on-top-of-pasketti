#!/usr/bin/env python3
"""Canonical ensemble entry point.

Supports text-level MBR, saved-logprob ensembling, N-best rescoring,
and tree reranker training/inference across one or more folds.

Usage (from pasketti-phonetic/src):
  # Text-level MBR ensemble (no GPU needed, uses existing best_eval.csv):
    python3 ensemble.py --ensemble_mode=text

  # Sweep all subsets of top-K models (text MBR):
    python3 ensemble.py --ensemble_mode=sweep --ensemble_top_k=7

  # Save CTC log_probs for one or more models (needs GPU):
    CUDA_VISIBLE_DEVICES=1 python3 ensemble.py --ensemble_mode=save_logits --ensemble_models=v13.ep-15,v13

  # Logits-level ensemble from saved log_probs (average in log-space = geometric mean of probs):
    python3 ensemble.py --ensemble_mode=logits_saved --ensemble_models=v13.ep-15,v13,v13.focal_ctc.focal_ctc_gamma-1.0

  # Prob-level ensemble from saved log_probs (average in prob-space = arithmetic mean of probs):
    python3 ensemble.py --ensemble_mode=prob_saved --ensemble_models=v13.ep-15,v13,v13.focal_ctc.focal_ctc_gamma-1.0

  # Sweep logits/prob ensemble (tries all combinations):
    python3 ensemble.py --ensemble_mode=sweep_logits --ensemble_top_k=5

  # One-shot: save logits + ensemble (GPU needed):
    CUDA_VISIBLE_DEVICES=1 python3 ensemble.py --ensemble_mode=logits --ensemble_models=v13.ep-15,v13

    # Explicit tree ranking alias (same as tree_reranker + --ensemble_tree_task=ranking):
        python3 ensemble.py --ensemble_mode=tree_ranker --ensemble_models=v13.ep-15,v13

    # Explicit tree regression alias (same as tree_reranker + --ensemble_tree_task=regression):
        python3 ensemble.py --ensemble_mode=tree_regression --ensemble_models=v13.ep-15,v13

    # Tree ranking with CER-gap relevance labels:
        python3 ensemble.py --ensemble_mode=tree_ranker --ensemble_relevance_strategy=gap \
            --ensemble_models=v13.ep-15,v13

Algorithm summary:
  Text-level MBR: For each utterance, pick the candidate prediction that has
    minimum average CER to all other candidates (consensus / median hypothesis).
  Logits-level:   Average CTC log_probs from multiple models, then greedy decode.
                   Equivalent to geometric mean of probability distributions.
  Prob-level:     Convert log_probs to probs, average, then greedy decode.
                   Equivalent to arithmetic mean of probability distributions.
                   Generally more robust than logits averaging.
"""

import sys, os, json, itertools, time, contextlib, math, subprocess, traceback, multiprocessing, re, ast
from pathlib import Path
from collections import defaultdict

from absl import app
from absl import flags as absl_flags

# ---- Path setup ----
_SCRIPT_DIR = Path(os.path.abspath(__file__)).parent   # tests/
_PROJ_DIR = _SCRIPT_DIR.parent                          # pasketti-phonetic/
_SHARED_DIR = _PROJ_DIR.parent / 'pasketti'             # shared code
_REPO_DIR = _PROJ_DIR.parent.parent.parent              # pikachu/
_RUNTIME_DIR = _PROJ_DIR.parent / 'childrens-speech-recognition-runtime'

sys.path.insert(0, str(_PROJ_DIR))
sys.path.insert(0, str(_SHARED_DIR))
sys.path.insert(0, str(_REPO_DIR / 'utils'))
sys.path.insert(0, str(_REPO_DIR / 'third'))
if _RUNTIME_DIR.exists():
    sys.path.insert(0, str(_RUNTIME_DIR))

import numpy as np
import pandas as pd
from scipy.special import logsumexp  # for stable log(mean(exp(log_probs)))
import editdistance

from ensemble_feats import build_reranker_dataset as build_reranker_dataset_impl

# ---- CTC beam search (for ensemble decode) ----
try:
    from src.ctc_decode import _prefix_beam_search_np, prefix_beam_search_nbest, ctc_force_score, ctc_force_score_batch, CharNgramLM, load_ngram_lm
    _HAS_BEAM_SEARCH = True
except ImportError:
    _HAS_BEAM_SEARCH = False

# ---- Metric ----
try:
    from metric.score import score_ipa_cer, normalize_ipa
except ImportError:
    import jiwer
    import unicodedata
    def normalize_ipa(s):
        import re
        s = unicodedata.normalize('NFC', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
        import ast
    def score_ipa_cer(actual, predicted):
        refs = [normalize_ipa(t) for t in actual]
        preds = [normalize_ipa(t) for t in predicted]
        return jiwer.cer(refs, preds)

# ---- IPA char vocab (same as base.py) ----
try:
    from src.models.base import IPA_ID_TO_CHAR, IPA_CTC_BLANK
except ImportError:
    IPA_CTC_BLANK = 0
    IPA_ID_TO_CHAR = None


# ===========================================================================
#  Working directory layout
# ===========================================================================
from src.config import RUN_VERSION
WORKING_BASE = _PROJ_DIR / 'working' / 'offline' / RUN_VERSION
FOLD = 0

FLAGS = absl_flags.FLAGS

absl_flags.DEFINE_enum('ensemble_mode', 'tree_reranker',
                       ['text', 'rover', 'sweep', 'save_logits', 'logits', 'logits_saved',
                        'prob_saved', 'max_saved', 'sweep_logits', 'nbest_rescore',
                        'nbest_rescore2',
                        'tree_reranker', 'tree_ranker', 'tree_regression',
                        'tree', 'nbest', 'nbest2'],
                       'Ensemble mode')
absl_flags.DEFINE_alias('emode', 'ensemble_mode')
absl_flags.DEFINE_string('ensemble_models', None,
                         'Comma-separated model names. Empty = load models.txt from cwd, '
                         'then fall back to auto top-K selection.')
absl_flags.DEFINE_alias('emodels', 'ensemble_models')
absl_flags.DEFINE_integer('ensemble_top_k', 5, 'Number of top models for auto-selection.')
absl_flags.DEFINE_integer('ensemble_max_sweep_k', None, 'Max combination size for sweep modes.')
absl_flags.DEFINE_string('ensemble_working_dir', None, 'Override working directory.')
absl_flags.DEFINE_string('ensemble_folds', '0',
                         'Fold selection: "0", "0,1", or "all" for OOF-style aggregation.')
absl_flags.DEFINE_alias('efolds', 'ensemble_folds')
absl_flags.DEFINE_integer('ensemble_parallel_folds', 0,
                          'For tree_reranker only: run up to N single-fold jobs in parallel subprocesses. '
                          '0 disables parallel fold execution unless --parallel_folds/--pfolds is enabled.')
absl_flags.DEFINE_integer('ensemble_parallel_cores', 0,
                          'Total CPU cores budget for parallel tree_reranker fold jobs. '
                          '0 = auto-detect from os.cpu_count().')
absl_flags.DEFINE_alias('pcores', 'ensemble_parallel_cores')
absl_flags.DEFINE_integer('ensemble_parallel_fold_workers', 0,
                          'Per-fold feature-building worker count when using parallel_folds. '
                          '0 = auto from per-fold CPU budget.')
absl_flags.DEFINE_alias('pfold_workers', 'ensemble_parallel_fold_workers')
absl_flags.DEFINE_integer('ensemble_parallel_tree_threads', 0,
                          'Per-fold tree training thread count when using parallel_folds. '
                          '0 = auto from per-fold CPU budget.')
absl_flags.DEFINE_alias('ptree_threads', 'ensemble_parallel_tree_threads')
absl_flags.DEFINE_integer('ensemble_n_efolds', 1,
                          'Number of folds to merge into one ensemble run. 1 keeps per-fold runs; '
                          'for example, --efolds=0,1 --n_efolds=2 runs one merged 0+1 evaluation.')
absl_flags.DEFINE_alias('n_efolds', 'ensemble_n_efolds')
absl_flags.DEFINE_integer('ensemble_beam_width', 10, 'Beam width for CTC decode and rescoring.')
absl_flags.DEFINE_float('ensemble_temperature', 1.0, 'Sharpening temperature after averaging logprobs.')
absl_flags.DEFINE_integer('ensemble_nbest', 10, 'Number of N-best candidates per model.')
absl_flags.DEFINE_float('ensemble_ctc_score_weight', 1.0,
                        'Weight for exact CTC rescoring in nbest_rescore / nbest_rescore2.')
absl_flags.DEFINE_float('ensemble_tdt_score_weight', 2.0,
                        'Weight for exact TDT rescoring in nbest_rescore2. Ignored by nbest_rescore.')
absl_flags.DEFINE_alias('etdt_weight', 'ensemble_tdt_score_weight')
absl_flags.DEFINE_integer('ensemble_ctc_prune_topk', 20,
                          'For nbest_rescore2: keep top-K candidates by aggregated exact CTC score, '
                          'then union with mandatory TDT model outputs before TDT rescoring.')
absl_flags.DEFINE_alias('ectc_prune_topk', 'ensemble_ctc_prune_topk')
absl_flags.DEFINE_enum('ensemble_tdt_score_method', 'numba',
                       ['numba', 'numba_cpu', 'exact', 'forced_align'],
                       'TDT scoring method for ensemble. "numba" = NeMo TDTLossNumba via '
                       'Numba CUDA JIT (fastest, default), "exact" = pure PyTorch forward '
                       'algorithm (no Numba), "numba_cpu" = custom njit Numba CPU, "forced_align" = greedy alignment '
                       'approximation (~3x faster than exact). '
                       'Overrides model-level tdt_score_method.')
absl_flags.DEFINE_alias('etdt_score', 'ensemble_tdt_score_method')
absl_flags.DEFINE_bool('ensemble_tdt_force_keep_preds', True,
                       'Force-keep raw TDT model predictions (eval pred_tdt/pred_primary) '
                       'inside bounded TDT candidate pools. Used by nbest_rescore2 and '
                       'feat_tdt exact-scoring.')
absl_flags.DEFINE_alias('etdt_force_keep', 'ensemble_tdt_force_keep_preds')
absl_flags.DEFINE_alias('tdt_keep', 'ensemble_tdt_force_keep_preds')
absl_flags.DEFINE_bool('ensemble_skip_tdt_primary_ctc_candidates', False,
                       'If True, models whose primary decode_method is tdt still contribute '
                       'their eval/dual-head TDT outputs, but their CTC beam/pred_ctc candidates '
                       'are not added to the pooled candidate set.')
absl_flags.DEFINE_alias('eskip_tdt_ctc', 'ensemble_skip_tdt_primary_ctc_candidates')
absl_flags.DEFINE_string('ensemble_lm_path', '', 'Path to primary n-gram LM JSON file for shallow fusion and reranker LM features.')
absl_flags.DEFINE_string('ensemble_word_lm_path', '',
                         'Optional secondary word-level n-gram LM JSON file for additional reranker LM features.')
absl_flags.DEFINE_alias('word_lm_path', 'ensemble_word_lm_path')
absl_flags.DEFINE_float('ensemble_lm_weight', 0.3, 'LM weight for shallow fusion.')
absl_flags.DEFINE_string('ensemble_tree_model', 'cb',
                         'Tree model(s): lgb, xgb, cb, or comma-separated ensemble.')
absl_flags.DEFINE_alias('etm', 'ensemble_tree_model')
absl_flags.DEFINE_integer('ensemble_cv_folds', 5, 'Number of CV folds for tree reranker.')
absl_flags.DEFINE_integer('ensemble_tree_iters', 500, 'Number of boosting iterations.')
absl_flags.DEFINE_alias('tree_iters', 'ensemble_tree_iters')
absl_flags.DEFINE_float('ensemble_tree_lr', 0.05, 'Learning rate for tree model.')
absl_flags.DEFINE_integer('ensemble_tree_depth', 4, 'Max tree depth.')
absl_flags.DEFINE_alias('etree_depth', 'ensemble_tree_depth')
absl_flags.DEFINE_alias('tree_depth', 'ensemble_tree_depth')
absl_flags.DEFINE_integer('ensemble_tree_leaves', 31, 'Max number of leaves.')
absl_flags.DEFINE_alias('tree_leaves', 'ensemble_tree_leaves')
absl_flags.DEFINE_float('ensemble_tree_bagging', 0.8, 'Subsample ratio.')
absl_flags.DEFINE_float('ensemble_tree_feat_frac', 0.6, 'Feature fraction / colsample ratio.')
absl_flags.DEFINE_alias('tree_feat_frac', 'ensemble_tree_feat_frac')
absl_flags.DEFINE_float('ensemble_tree_reg_lambda', 5.0, 'L2 regularization.')
absl_flags.DEFINE_alias('tree_reg_lambda', 'ensemble_tree_reg_lambda')
absl_flags.DEFINE_integer('ensemble_tree_early_stop', 50, 'Early stopping rounds.')
absl_flags.DEFINE_alias('tree_early_stop', 'ensemble_tree_early_stop')
absl_flags.DEFINE_string('ensemble_tree_device', 'cpu', 'Tree training device: cpu or gpu.')
absl_flags.DEFINE_alias('etdevice', 'ensemble_tree_device')
absl_flags.DEFINE_alias('tree_device', 'ensemble_tree_device')
absl_flags.DEFINE_float('ensemble_tree_dd_weight', 1.0,
                        'Training/eval mix weight for dd source in tree reranker.')
absl_flags.DEFINE_alias('tree_dd_weight', 'ensemble_tree_dd_weight')
absl_flags.DEFINE_float('ensemble_tree_ext_weight', 0.5,
                        'Training/eval mix weight for ext source in tree reranker.')
absl_flags.DEFINE_alias('tree_ext_weight', 'ensemble_tree_ext_weight')
absl_flags.DEFINE_enum('ensemble_tree_task', 'ranking', ['ranking', 'regression'],
                       'Tree task type.')
absl_flags.DEFINE_string('ensemble_tree_obj', '', 'Optional explicit tree objective override.')
absl_flags.DEFINE_integer('ensemble_relevance_levels', 6,
                          'Relevance levels for ranking labels. 0 = binary.')
absl_flags.DEFINE_enum('ensemble_relevance_strategy', 'gap',
                       ['rank', 'gap', 'binary_best'],
                       'How to convert target CER into ranking labels. '
                       'rank=ordinal rank bins, gap=best/worst-normalized CER gap bins, '
                       'binary_best=only best candidate(s) are positive.')
absl_flags.DEFINE_alias('rel_strategy', 'ensemble_relevance_strategy')
absl_flags.DEFINE_bool('ensemble_no_lm_feats', True, 'Exclude LM-related features from tree reranker.')
absl_flags.DEFINE_bool('ensemble_lm_feats', False,
                       'Enable LM-related reranker features. When True, overrides '
                       'ensemble_no_lm_feats and keeps lm_score features.')
absl_flags.DEFINE_alias('lm_feats', 'ensemble_lm_feats')
absl_flags.DEFINE_bool('ensemble_lm_per_word_only', False,
                       'Keep only lm_score_per_word among LM-related reranker features. '
                       'Useful for focused LM ablations on the phonetic track.')
absl_flags.DEFINE_alias('lm_per_word_only', 'ensemble_lm_per_word_only')
absl_flags.DEFINE_bool('ensemble_feat_text', False, 'Enable text structure features.')
absl_flags.DEFINE_alias('feat_text', 'ensemble_feat_text')
absl_flags.DEFINE_bool('ensemble_feat_ipa', False, 'Enable IPA phonetic features.')
absl_flags.DEFINE_alias('feat_ipa', 'ensemble_feat_ipa')
absl_flags.DEFINE_bool('ensemble_feat_ctc_stats', False, 'Enable higher-order CTC stat features.')
absl_flags.DEFINE_alias('feat_ctc_stats', 'ensemble_feat_ctc_stats')
absl_flags.DEFINE_bool('ensemble_no_ctc_score_feats', False,
                       'Exclude CTC-score-derived reranker features while keeping CTC-based '
                       'candidate generation intact. Kept False by default to avoid changing '
                       'prior experiments.')
absl_flags.DEFINE_alias('no_ctc_score_feats', 'ensemble_no_ctc_score_feats')
absl_flags.DEFINE_bool('ensemble_feat_audio', True, 'Enable audio/speaking-rate features.')
absl_flags.DEFINE_alias('feat_audio', 'ensemble_feat_audio')
absl_flags.DEFINE_bool('ensemble_feat_consensus', False,
                       'Enable lightweight consensus-style candidate features such as '
                       'mean_pairwise_edit_dist and consensus_score. Does not include '
                       'the heavier MBR group features.')
absl_flags.DEFINE_alias('feat_consensus', 'ensemble_feat_consensus')
absl_flags.DEFINE_bool('ensemble_feat_mbr', False,
                       'Enable heavier MBR-style reranker features such as is_mbr_selected '
                       'and edit_dist_to_mbr. Kept False by default because it adds '
                       'expensive per-utterance pairwise edit-distance work.')
absl_flags.DEFINE_alias('feat_mbr', 'ensemble_feat_mbr')
absl_flags.DEFINE_alias('mbr', 'ensemble_feat_mbr')
absl_flags.DEFINE_bool('ensemble_feat_group_ext', False, 'Enable extended group-relative features.')
absl_flags.DEFINE_alias('feat_group_ext', 'ensemble_feat_group_ext')
absl_flags.DEFINE_bool('ensemble_feat_align', False, 'Enable CTC alignment features.')
absl_flags.DEFINE_alias('feat_align', 'ensemble_feat_align')
absl_flags.DEFINE_bool('ensemble_feat_logprob_proxy', False, 'Enable entropy/blank-proxy features.')
absl_flags.DEFINE_alias('feat_logprob_proxy', 'ensemble_feat_logprob_proxy')
absl_flags.DEFINE_bool('ensemble_feat_tdt', False,
                       'Compatibility shortcut for enabling both feat_tdt_light and '
                       'feat_tdt_exact.')
absl_flags.DEFINE_alias('feat_tdt', 'ensemble_feat_tdt')
absl_flags.DEFINE_bool('ensemble_feat_tdt_light', True,
                       'Enable lightweight TDT-derived reranker features such as '
                       'TDT-text agreement, length-diff, and space-diff features.')
absl_flags.DEFINE_alias('feat_tdt_light', 'ensemble_feat_tdt_light')
absl_flags.DEFINE_alias('tdt_light', 'ensemble_feat_tdt_light')
absl_flags.DEFINE_bool('ensemble_feat_tdt_primary_score', False,
                       'Enable lightweight TDT greedy/primary pred_score features from eval '
                       'outputs. Kept False by default to avoid changing prior experiments.')
absl_flags.DEFINE_alias('feat_tdt_primary_score', 'ensemble_feat_tdt_primary_score')
absl_flags.DEFINE_alias('feat_tdt_pscore', 'ensemble_feat_tdt_primary_score')
absl_flags.DEFINE_alias('tdt_primary_score', 'ensemble_feat_tdt_primary_score')
absl_flags.DEFINE_bool('ensemble_feat_tdt_nbest_score', False,
                       'Enable lightweight TDT eval N-best score features from '
                       'pred_nbest_texts/pred_nbest_scores in eval outputs. Kept False by '
                       'default to avoid changing prior experiments.')
absl_flags.DEFINE_alias('feat_tdt_nbest_score', 'ensemble_feat_tdt_nbest_score')
absl_flags.DEFINE_alias('feat_tdt_nscore', 'ensemble_feat_tdt_nbest_score')
absl_flags.DEFINE_alias('tdt_nbest_score', 'ensemble_feat_tdt_nbest_score')
absl_flags.DEFINE_integer('ensemble_tdt_eval_nbest', 0,
                       'Offline candidate expansion only: add up to top-K exported '
                       'pred_nbest_texts per model from eval.csv. 0 disables. '
                       'Kept 0 by default to avoid changing prior experiments.')
absl_flags.DEFINE_alias('etdt_nbest', 'ensemble_tdt_eval_nbest')
absl_flags.DEFINE_alias('tdt_nbest', 'ensemble_tdt_eval_nbest')
absl_flags.DEFINE_bool('ensemble_feat_tdt_exact', False,
                       'Enable exact TDT-score reranker features on a pruned top-K '
                       'candidate pool.')
absl_flags.DEFINE_alias('feat_tdt_exact', 'ensemble_feat_tdt_exact')
absl_flags.DEFINE_alias('tdt_exact', 'ensemble_feat_tdt_exact')
absl_flags.DEFINE_bool('ensemble_feat_tdtctc_compare', False,
                       'Enable direct TDT-vs-CTC prediction comparison features such as '
                       'length, space-count, and edit-distance gaps from dual-head outputs.')
absl_flags.DEFINE_alias('feat_tdtctc_compare', 'ensemble_feat_tdtctc_compare')
absl_flags.DEFINE_bool('ensemble_feat_dual', False,
                       'Enable dual-head prediction features (is_dual_ctc/tdt/primary, '
                       'dual_heads_agree, dual_len_gap, n_dual_*_hits) and add dual-head '
                       'texts as reranker candidates. Requires dual_head_preds.pt per model.')
absl_flags.DEFINE_alias('feat_dual', 'ensemble_feat_dual')
absl_flags.DEFINE_bool('ensemble_feat_tdt_score_compare', True,
                       'Enable direct TDT-vs-CTC score comparison features such as mean-score '
                       'gaps and rank gaps. Kept True by default to preserve current feat_tdt behavior.')
absl_flags.DEFINE_alias('feat_tdt_score_compare', 'ensemble_feat_tdt_score_compare')
absl_flags.DEFINE_bool('ensemble_feat_tdt_group', False,
                       'Enable lightweight TDT-subgroup aggregation features based on existing '
                       'CTC scores, beam ranks, and TDT primary/greedy text consensus only. '
                       'Does not require exact TDT scoring.')
absl_flags.DEFINE_alias('feat_tdt_group', 'ensemble_feat_tdt_group')
absl_flags.DEFINE_alias('tdt_group', 'ensemble_feat_tdt_group')
absl_flags.DEFINE_bool('ensemble_feat_wavlm_group', False,
                       'Enable lightweight WavLM-vs-non-WavLM subgroup aggregation features '
                       'based on existing CTC scores and beam ranks only.')
absl_flags.DEFINE_alias('feat_wavlm_group', 'ensemble_feat_wavlm_group')
absl_flags.DEFINE_alias('wavlm_group', 'ensemble_feat_wavlm_group')
absl_flags.DEFINE_bool('ensemble_feat_nemo_group', False,
                       'Enable lightweight NeMo-vs-non-NeMo subgroup aggregation features '
                       'based on existing CTC scores and beam ranks only. Kept False by '
                       'default to avoid changing prior experiments.')
absl_flags.DEFINE_alias('feat_nemo_group', 'ensemble_feat_nemo_group')
absl_flags.DEFINE_alias('nemo_group', 'ensemble_feat_nemo_group')
absl_flags.DEFINE_bool('ensemble_feat_group_edit_dist', False,
                       'When enabled, add per-group (TDT/CTC/WavLM/NeMo) edit-distance and '
                       'MBR-like features: mean/min/max edit_dist_to_best aggregation, '
                       'within-group edit-distance ranking, and cross-group gap features. '
                       'Also adds word-level variants when word_edit_dist columns exist.')
absl_flags.DEFINE_alias('feat_group_edit_dist', 'ensemble_feat_group_edit_dist')
absl_flags.DEFINE_float('ensemble_wavlm_max_dur', 0,
                        'Max audio duration (sec) for WavLM model CTC scoring in ensemble. '
                        '0=no limit. When > 0, WavLM models only compute CTC logprob scores for '
                        'utterances with audio_duration_sec <= this value; longer utterances get '
                        'NaN for all WavLM-derived features.')
absl_flags.DEFINE_alias('wavlm_max_dur', 'ensemble_wavlm_max_dur')
absl_flags.DEFINE_integer('ensemble_tdt_feat_topk', 4,
                          'For feat_tdt_exact: exact-score at most the top-K CTC candidates per '
                          'utterance, plus forced TDT primary predictions when enabled.')
absl_flags.DEFINE_alias('tdt_topk', 'ensemble_tdt_feat_topk')
absl_flags.DEFINE_bool('ensemble_cache_tdt_exact_scores', True,
                       'Offline only: cache feat_tdt_exact candidate scores in plaintext JSONL and '
                       'incrementally reuse cache hits before rescoring misses.')
absl_flags.DEFINE_alias('cache_tdt_exact', 'ensemble_cache_tdt_exact_scores')
absl_flags.DEFINE_alias('tdt_exact_cache', 'ensemble_cache_tdt_exact_scores')
absl_flags.DEFINE_integer('ensemble_tdt_score_chunk', 16,
                         'Max candidates per TDT scoring call. Batches candidates across utterances '
                         'for fewer kernel launches. 0 = no limit (process all at once).')
absl_flags.DEFINE_alias('tdt_score_chunk', 'ensemble_tdt_score_chunk')
absl_flags.DEFINE_integer('ensemble_ctc_prune_threads', 0,
                         'Number of threads for parallel CTC prune scoring. '
                         '0 = serial (default, safe). >0 = use ThreadPoolExecutor with N threads. '
                         'torch.ctc_loss releases GIL so threads can overlap C++ execution. '
                         'Recommended: 4~8 on multi-core machines.')
absl_flags.DEFINE_alias('ctc_prune_threads', 'ensemble_ctc_prune_threads')
absl_flags.DEFINE_bool('ensemble_feat_word', False, 'Enable word-head-derived features.')
absl_flags.DEFINE_alias('feat_word', 'ensemble_feat_word')
absl_flags.DEFINE_alias('ensemble_feat_aux', 'ensemble_feat_word')
absl_flags.DEFINE_alias('feat_aux', 'ensemble_feat_word')
absl_flags.DEFINE_bool('ensemble_feat_aux_meta', False, 'Enable age/domain auxiliary meta features.')
absl_flags.DEFINE_alias('feat_aux_meta', 'ensemble_feat_aux_meta')
absl_flags.DEFINE_bool('ensemble_feat_word_label', False,
                       'Enable offline-only word label features for local reranker / pseudo labels.')
absl_flags.DEFINE_alias('feat_word_label', 'ensemble_feat_word_label')
absl_flags.DEFINE_string('ensemble_word_label_file', '',
                         'Optional JSONL/CSV file providing utterance_id -> word label mapping.')
absl_flags.DEFINE_string('ensemble_word_label_col', '',
                         'Optional explicit word label column name for ensemble_word_label_file.')
absl_flags.DEFINE_bool('ensemble_feat_all', False, 'Enable all standard online-safe feature groups.')
absl_flags.DEFINE_alias('feat_all', 'ensemble_feat_all')
absl_flags.DEFINE_bool('ensemble_show_feats', False,
                       'Print the final tree reranker feature manifest during training.')
absl_flags.DEFINE_alias('show_feats', 'ensemble_show_feats')
absl_flags.DEFINE_alias('eshow_feats', 'ensemble_show_feats')
absl_flags.DEFINE_bool('ensemble_dump_feats', False,
                       'Dump offline tree reranker feature frame for online/offline comparison.')
absl_flags.DEFINE_alias('dump_feats', 'ensemble_dump_feats')
absl_flags.DEFINE_integer('ensemble_dump_feats_limit', 0,
                          'If >0, dump only the first N utterance_ids from the reranker feature frame. '
                          '0 dumps all utterances.')
absl_flags.DEFINE_alias('dump_feats_limit', 'ensemble_dump_feats_limit')
absl_flags.DEFINE_string('ensemble_dump_feats_uids_path', '',
                         'Optional file containing the exact utterance ids to keep in the dumped '
                         'reranker feature frame. Supports jsonl/csv/txt/pkl. When set, this '
                         'filter is applied before dump_feats_limit.')
absl_flags.DEFINE_alias('dump_feats_uids_path', 'ensemble_dump_feats_uids_path')
absl_flags.DEFINE_string('ensemble_drop_feats', '', 'Comma-separated regex patterns for feature ablations.')
absl_flags.DEFINE_bool('ensemble_cache_dataset', False, 'Cache reranker dataset for repeated experiments.')
absl_flags.DEFINE_bool('ensemble_clear_cache', False,
                       'Clear reranker dataset cache and TDT exact-score cache before running, then continue.')
absl_flags.DEFINE_alias('eclear', 'ensemble_clear_cache')
absl_flags.DEFINE_alias('clear_cache', 'ensemble_clear_cache')
absl_flags.DEFINE_integer('ensemble_n_seeds', 1, 'Number of tree seeds to train and average.')
absl_flags.DEFINE_string('ensemble_save_dir', '', 'Directory to save reranker artifacts.')
absl_flags.DEFINE_string('ensemble_exp_name', '',
                         'Optional experiment name for tree reranker runs. When set and '
                         'ensemble_save_dir is empty, artifacts are saved under a dedicated '
                         'experiment directory instead of the default ensemble/<fold>.')
absl_flags.DEFINE_alias('eexp', 'ensemble_exp_name')
absl_flags.DEFINE_string('ensemble_exp_notes', '',
                         'Optional free-form notes recorded in tree reranker experiment logs.')
absl_flags.DEFINE_alias('eexp_notes', 'ensemble_exp_notes')
absl_flags.DEFINE_integer('ensemble_n_workers', 0, 'Parallel workers for feature building. 0 = auto.')
absl_flags.DEFINE_alias('n_eworkers', 'ensemble_n_workers')
absl_flags.DEFINE_enum('ensemble_progress', 'auto', ['auto', 'tqdm', 'log', 'none'],
                       'Progress display policy. auto=tqdm on local TTY, log on non-TTY/online.')
absl_flags.DEFINE_alias('eprogress', 'ensemble_progress')
absl_flags.DEFINE_string('ensemble_suffix', '',
                         'Manual suffix appended to auto-generated ensemble model name. '
                         'Example: -esuffix=.v2 → ensemble.feat_dual.v2. '
                         'When empty, ensemble falls back to melt --mns / --model_name_suffix '
                         'for backward compatibility.')
absl_flags.DEFINE_alias('esuffix', 'ensemble_suffix')
absl_flags.DEFINE_bool('ensemble_allow_eval_set_mismatch', False,
                       'Allow ensemble models to use different eval UID sets. '
                       'Default False raises an error because mixed eval sets usually '
                       'mean train/eval directories were mixed by mistake '
                       '(for example model vs model.eval).')
absl_flags.DEFINE_alias('allow_eval_mismatch', 'ensemble_allow_eval_set_mismatch')

# ---------------------------------------------------------------------------
# Auto model-name from CLI flags (following melt convention)
# ---------------------------------------------------------------------------
# Flags that should NEVER contribute to the auto-generated model name,
# even when passed with double-dash (``--``).
_ENSEMBLE_NAME_IGNORE = frozenset({
    # --- naming / meta ---
    'flagfile',
    'eval', 'eval_name',
    'ensemble_suffix', 'esuffix',
    'ensemble_exp_name', 'eexp',
    'ensemble_exp_notes', 'eexp_notes',
    'ensemble_save_dir',
    'mn', 'model_name', 'mns', 'model_name_suffix',
    # --- execution mode / model selection ---
    'ensemble_mode', 'emode',
    'ensemble_models', 'emodels',
    'ensemble_folds', 'efolds',
    'ensemble_top_k', 'etop_k',
    'ensemble_max_sweep_k',
    'ensemble_working_dir',
    'ensemble_n_efolds', 'n_efolds',
    # --- parallelism / display / caching ---
    'ensemble_parallel_folds', 'pfolds', 'parallel_folds',
    'ensemble_parallel_cores', 'pcores',
    'ensemble_parallel_fold_workers', 'pfold_workers',
    'ensemble_parallel_tree_threads', 'ptree_threads',
    'ensemble_n_workers', 'n_eworkers',
    'ensemble_progress', 'eprogress',
    'ensemble_show_feats', 'show_feats', 'eshow_feats',
    'ensemble_clear_cache', 'eclear', 'clear_cache',
    'ensemble_cache_dataset',
    'ensemble_tree_device', 'etdevice',
    'ensemble_cache_tdt_exact_scores', 'cache_tdt_exact', 'tdt_exact_cache',
    'ensemble_ctc_prune_threads', 'ctc_prune_threads',
    'ensemble_tdt_score_chunk', 'tdt_score_chunk',
})


def _build_ensemble_model_name(args=None, base='ensemble', sep='.'):
    """Build an auto-generated model name from double-dash CLI flags.

    Follows melt convention:
      ``--flag_name``           → ``.flag_name``
      ``--flag_name=value``     → ``.flag_name-value``
      ``-flag_name``            → silent (not added to name)

    The ``ensemble_`` prefix is stripped for brevity.  Flags listed in
    ``_ENSEMBLE_NAME_IGNORE`` are always excluded. ``--esuffix=.abc``
    is appended at the end. When ``esuffix`` is empty, only an explicitly
    provided CLI ``--mns`` / ``--model_name_suffix`` is honored. Restored
    configs must not silently change ensemble artifact naming.
    """
    if args is None:
        args = sys.argv[1:]
    parts = []
    for arg in args:
        if not arg.startswith('--'):
            continue
        body = arg[2:]
        if '=' in body:
            name, value = body.split('=', 1)
        else:
            name = body
            value = None
        if name in _ENSEMBLE_NAME_IGNORE:
            continue
        # Strip ensemble_ prefix for brevity
        display = name[len('ensemble_'):] if name.startswith('ensemble_') else name
        if value is None:
            # boolean flag --flag or --noflag
            parts.append(display)
        elif value.lower() in ('true', '1'):
            parts.append(display)
        elif value.lower() in ('false', '0'):
            parts.append(f'no{display}')
        else:
            parts.append(f'{display}-{value}')
    parts.sort()
    model_name = base
    for p in parts:
        model_name += sep + p
    suffix = getattr(FLAGS, 'ensemble_suffix', '')
    if not suffix and _cli_has_any_flag('mns', 'model_name_suffix', args=args):
        suffix = getattr(FLAGS, 'mns', '') or getattr(FLAGS, 'model_name_suffix', '')
    if suffix:
        model_name += suffix
    return model_name


def _cli_has_any_flag(*names, args=None):
    if args is None:
        args = sys.argv[1:]
    wanted = set(names)
    for arg in args:
        if not arg.startswith('--'):
            continue
        name = arg[2:].split('=', 1)[0]
        if name in wanted:
            return True
    return False


def _resolve_cli_override(primary_value, primary_names, fallback_candidates):
    """Resolve a value with explicit CLI precedence.

    Priority:
      1. Explicit ensemble-specific flag (or alias)
      2. Explicit generic melt/common tree flag(s)
      3. Existing ensemble default/value
    """
    if _cli_has_any_flag(*primary_names):
        return primary_value
    for value, names in fallback_candidates:
        if _cli_has_any_flag(*names):
            return value
    return primary_value


def _get_tree_cli_params():
    """Return tree reranker params with ensemble flags taking precedence.

    This keeps current ``ensemble_tree_*`` behavior unchanged while allowing
    explicit reuse of melt/common tree flags such as ``--tree_model``,
    ``--iters``, ``--tree_lr``, ``--max_depth``, ``--num_leaves``,
    ``--tree_bagging``, ``--feature_fraction`` and ``--reg_lambda``.
    """
    return {
        'tree_model': _resolve_cli_override(
            FLAGS.ensemble_tree_model,
            ('ensemble_tree_model', 'etm'),
            ((FLAGS.tree_model, ('tree_model', 'tmodel', 'tm')),),
        ),
        'tree_iters': _resolve_cli_override(
            FLAGS.ensemble_tree_iters,
            ('ensemble_tree_iters', 'tree_iters'),
            ((FLAGS.iters, ('iters',)), (FLAGS.trees, ('trees',))),
        ),
        'tree_lr': _resolve_cli_override(
            FLAGS.ensemble_tree_lr,
            ('ensemble_tree_lr',),
            ((FLAGS.tree_lr, ('tree_lr',)),),
        ),
        'tree_depth': _resolve_cli_override(
            FLAGS.ensemble_tree_depth,
            ('ensemble_tree_depth', 'etree_depth', 'tree_depth'),
            ((FLAGS.max_depth, ('max_depth',)),),
        ),
        'tree_leaves': _resolve_cli_override(
            FLAGS.ensemble_tree_leaves,
            ('ensemble_tree_leaves', 'tree_leaves'),
            ((FLAGS.num_leaves, ('num_leaves', 'max_leaves')),),
        ),
        'tree_bagging': _resolve_cli_override(
            FLAGS.ensemble_tree_bagging,
            ('ensemble_tree_bagging',),
            ((FLAGS.tree_bagging, ('tree_bagging',)),),
        ),
        'tree_feat_frac': _resolve_cli_override(
            FLAGS.ensemble_tree_feat_frac,
            ('ensemble_tree_feat_frac', 'tree_feat_frac'),
            ((FLAGS.feature_fraction, ('feature_fraction', 'feature_frac', 'feat_frac')),),
        ),
        'tree_reg_lambda': _resolve_cli_override(
            FLAGS.ensemble_tree_reg_lambda,
            ('ensemble_tree_reg_lambda', 'tree_reg_lambda'),
            ((FLAGS.reg_lambda, ('reg_lambda',)),),
        ),
        'tree_early_stop': FLAGS.ensemble_tree_early_stop,
        'tree_dd_weight': FLAGS.ensemble_tree_dd_weight,
        'tree_ext_weight': FLAGS.ensemble_tree_ext_weight,
        'tree_device': FLAGS.ensemble_tree_device,
        'tree_task': FLAGS.ensemble_tree_task,
        'tree_obj': _resolve_cli_override(
            FLAGS.ensemble_tree_obj,
            ('ensemble_tree_obj',),
            ((FLAGS.objective, ('objective', 'obj')),),
        ),
    }


def _should_use_tqdm(verbose=True, progress_mode=None):
    if not verbose:
        return False
    progress_mode = str(progress_mode or getattr(FLAGS, 'ensemble_progress', 'auto')).lower()
    if progress_mode == 'tqdm':
        return True
    if progress_mode in ('log', 'none'):
        return False
    if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS') or os.environ.get('BUILD_BUILDID'):
        return False
    return bool(sys.stdout.isatty() and sys.stderr.isatty())


def _iter_with_progress(iterable, total, desc, verbose=True, progress_mode=None):
    if not verbose or str(progress_mode or getattr(FLAGS, 'ensemble_progress', 'auto')).lower() == 'none':
        for item in iterable:
            yield item
        return

    if _should_use_tqdm(verbose=verbose, progress_mode=progress_mode):
        from tqdm import tqdm
        yield from tqdm(iterable, total=total, desc=desc, dynamic_ncols=True,
                        mininterval=0.5, leave=True,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        return

    milestones = [1, 2, 5] + list(range(10, 101, 10))
    thresholds = []
    for pct in milestones:
        if total <= 0:
            break
        thresholds.append(max(1, int(round(total * pct / 100.0))))
    thresholds = sorted(set(thresholds + ([total] if total > 0 else [])))
    next_idx = 0
    start_time = time.time()
    for index, item in enumerate(iterable, start=1):
        yield item
        if next_idx >= len(thresholds) or index < thresholds[next_idx]:
            continue
        while next_idx < len(thresholds) and index >= thresholds[next_idx]:
            next_idx += 1
        elapsed = time.time() - start_time
        rate = index / elapsed if elapsed > 0 else 0.0
        eta = (total - index) / rate if rate > 0 else 0.0
        pct = (100.0 * index / total) if total > 0 else 0.0
        print(f'{desc}: {index}/{total} ({pct:.0f}%, {rate:.1f} it/s, ETA {eta:.0f}s)', flush=True)


def _set_active_fold(fold):
    global FOLD
    FOLD = fold


def _detect_common_folds(model_names):
    common = None
    for mn in model_names:
        model_root = WORKING_BASE / mn
        if not model_root.exists():
            continue
        folds = set()
        for child in model_root.iterdir():
            if child.is_dir() and child.name.isdigit():
                folds.add(int(child.name))
        common = folds if common is None else (common & folds)
    return sorted(common or [])


def _resolve_folds(folds_value, model_names):
    folds_value = str(folds_value or '0').strip().lower()
    if folds_value == 'all':
        folds = _detect_common_folds(model_names)
        assert folds, f'No common folds found for models: {model_names}'
        return folds
    folds = []
    for item in folds_value.split(','):
        item = item.strip()
        if item:
            folds.append(int(item))
    assert folds, f'Invalid ensemble_folds={folds_value}'
    return folds


def _merge_fold_payloads(payloads, label='OOF Ensemble', verbose=True):
    merged_preds = {}
    merged_gold = {}
    merged_meta = {}
    for payload in payloads:
        merged_preds.update(payload['predictions'])
        merged_gold.update(payload['gold'])
        merged_meta.update(payload['meta'])
    return _evaluate(merged_preds, merged_gold, merged_meta, label=label, verbose=verbose)


def _chunk_folds(folds, chunk_size):
    assert chunk_size >= 1, f'ensemble_n_efolds must be >= 1, got {chunk_size}'
    return [folds[i:i + chunk_size] for i in range(0, len(folds), chunk_size)]


def _format_fold_group(folds):
    return ','.join(str(fold) for fold in folds)


def _strip_cli_flags(args, flag_names):
    cleaned = []
    skip_next = False
    flag_names = set(flag_names)
    for idx, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if not arg.startswith('--'):
            cleaned.append(arg)
            continue
        name = arg[2:].split('=', 1)[0]
        if name in flag_names:
            if '=' not in arg and idx + 1 < len(args) and not args[idx + 1].startswith('--'):
                skip_next = True
            continue
        cleaned.append(arg)
    return cleaned


def _get_tree_fold_save_dir(fold, exp_name='', save_dir=None, model_name='ensemble'):
    if save_dir:
        base = Path(save_dir)
        return base / f'fold{fold}'
    if exp_name:
        return WORKING_BASE / 'ensemble-experiments' / exp_name / str(fold)
    return WORKING_BASE / model_name / str(fold)


def _load_tree_fold_payload(save_dir):
    save_dir = Path(save_dir)
    eval_path = save_dir / 'eval.csv'
    assert eval_path.exists(), f'Missing eval.csv for fold payload: {eval_path}'
    df = pd.read_csv(eval_path)
    preds = dict(zip(df['utterance_id'], df['pred'].fillna('')))
    gold = dict(zip(df['utterance_id'], df['label'].fillna('')))
    meta = {}
    for _, row in df.iterrows():
        uid = row['utterance_id']
        meta[uid] = {
            'child_id': row.get('child_id', ''),
            'session_id': row.get('session_id', ''),
            'audio_path': row.get('audio_path', ''),
            'audio_duration_sec': row.get('audio_duration_sec', ''),
            'age_bucket': row.get('age_bucket', ''),
            'source': row.get('source', ''),
        }
    metrics_path = save_dir / 'metrics.csv'
    metrics = None
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
    return {
        'predictions': preds,
        'gold': gold,
        'meta': meta,
        'metrics': metrics,
        'save_dir': str(save_dir),
    }


def _print_parallel_tree_fold_summary(payloads):
    weighted_n = 0
    weighted_baseline = 0.0
    weighted_oracle = 0.0
    have_baseline = True
    have_oracle = True
    for payload in payloads:
        metrics = payload.get('metrics') or {}
        n = int(metrics.get('n_samples', 0) or 0)
        if n <= 0:
            continue
        weighted_n += n
        if 'baseline_cer' in metrics:
            weighted_baseline += float(metrics['baseline_cer']) * n
        else:
            have_baseline = False
        if 'oracle_cer' in metrics:
            weighted_oracle += float(metrics['oracle_cer']) * n
        else:
            have_oracle = False
    if weighted_n > 0:
        print('\n--- Parallel Fold Baselines (weighted by n_samples) ---')
        if have_baseline:
            print(f'Avg CTC Score (baseline): {weighted_baseline / weighted_n:.5f}')
        if have_oracle:
            print(f'Oracle (best candidate): {weighted_oracle / weighted_n:.5f}')


def _get_parallel_fold_job_count(folds):
    explicit_jobs = int(getattr(FLAGS, 'ensemble_parallel_folds', 0) or 0)
    if getattr(FLAGS, 'parallel_folds', False):
        return len(folds)
    return explicit_jobs


def _run_parallel_tree_folds(mode, model_names, folds):
    assert mode == 'tree_reranker', 'parallel_folds currently supports tree_reranker only'
    assert len(folds) > 1, 'parallel_folds needs multiple folds'

    requested_parallel = _get_parallel_fold_job_count(folds)
    max_parallel = min(len(folds), requested_parallel)
    assert max_parallel >= 2, (
        'parallel fold execution needs at least 2 jobs; '
        f'got requested_parallel={requested_parallel}'
    )

    total_cores = int(getattr(FLAGS, 'ensemble_parallel_cores', 0) or 0) or (os.cpu_count() or max_parallel)
    total_cores = max(total_cores, max_parallel)
    cores_per_job = max(1, total_cores // max_parallel)

    explicit_workers = int(getattr(FLAGS, 'ensemble_parallel_fold_workers', 0) or 0)
    explicit_tree_threads = int(getattr(FLAGS, 'ensemble_parallel_tree_threads', 0) or 0)
    per_job_workers = explicit_workers or max(1, min(8, cores_per_job // 3))
    per_job_tree_threads = explicit_tree_threads or max(1, cores_per_job - per_job_workers)

    child_args = _strip_cli_flags(sys.argv[1:], {
        'ensemble_folds', 'efolds',
        'ensemble_parallel_folds',
        'parallel_folds', 'pfolds',
        'ensemble_parallel_cores', 'pcores',
        'ensemble_parallel_fold_workers', 'pfold_workers',
        'ensemble_parallel_tree_threads', 'ptree_threads',
        'ensemble_n_efolds', 'n_efolds',
        'ensemble_n_workers', 'n_eworkers',
        'ensemble_save_dir',
        'ensemble_clear_cache', 'clear',
        'num_tree_threads', 'ntt',
    })

    print('\n=== Parallel tree_reranker fold runner ===')
    print(f'folds: {folds}')
    print(f'parallel jobs: {max_parallel}')
    print(f'total cores budget: {total_cores}')
    print(f'per-fold workers: {per_job_workers}')
    print(f'per-fold tree threads: {per_job_tree_threads}')

    pending = list(folds)
    running = []
    completed = []
    env_base = os.environ.copy()
    env_base['ENSEMBLE_PARALLEL_CHILD'] = '1'
    env_base['OMP_NUM_THREADS'] = str(per_job_tree_threads)
    env_base['MKL_NUM_THREADS'] = str(per_job_tree_threads)
    env_base['OPENBLAS_NUM_THREADS'] = str(per_job_tree_threads)
    env_base['NUMEXPR_NUM_THREADS'] = str(per_job_tree_threads)

    def _start_fold(fold):
        fold_save_dir = _get_tree_fold_save_dir(fold, exp_name=FLAGS.ensemble_exp_name,
                                                save_dir=FLAGS.ensemble_save_dir,
                                                model_name=_build_ensemble_model_name())
        fold_save_dir.mkdir(parents=True, exist_ok=True)
        log_path = fold_save_dir / 'parallel_fold.log'
        cmd = [sys.executable, str(Path(__file__).resolve())]
        cmd.extend(child_args)
        cmd.extend([
            f'--efolds={fold}',
            '--n_efolds=1',
            '--parallel_folds=false',
            '--ensemble_parallel_folds=0',
            f'--n_eworkers={per_job_workers}',
            f'--ntt={per_job_tree_threads}',
        ])
        if FLAGS.ensemble_save_dir:
            cmd.append(f'--ensemble_save_dir={fold_save_dir}')
        log_handle = open(log_path, 'w')
        proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env_base)
        print(f'  Started fold {fold}: pid={proc.pid} log={log_path}')
        return {
            'fold': fold,
            'proc': proc,
            'log_handle': log_handle,
            'log_path': log_path,
            'save_dir': fold_save_dir,
        }

    while pending or running:
        while pending and len(running) < max_parallel:
            running.append(_start_fold(pending.pop(0)))
        time.sleep(1)
        still_running = []
        for job in running:
            ret = job['proc'].poll()
            if ret is None:
                still_running.append(job)
                continue
            job['log_handle'].close()
            if ret != 0:
                tail = ''
                try:
                    tail = subprocess.run(['tail', '-n', '80', str(job['log_path'])], capture_output=True, text=True, check=False).stdout
                except Exception:
                    pass
                raise RuntimeError(f'Fold {job["fold"]} failed with exit code {ret}. Log: {job["log_path"]}\n{tail}')
            print(f'  Finished fold {job["fold"]}: log={job["log_path"]}')
            completed.append(job)
        running = still_running

    payloads = [_load_tree_fold_payload(job['save_dir']) for job in sorted(completed, key=lambda x: x['fold'])]
    merged = _merge_fold_payloads(payloads, label=f'OOF {mode} ({len(folds)} folds, parallel)', verbose=True)
    _print_parallel_tree_fold_summary(payloads)
    return merged


def get_model_dir(model_name):
    return WORKING_BASE / model_name / str(FOLD)


def get_eval_csv(model_dir):
    """Return best_eval.csv if it exists, otherwise fall back to eval.csv."""
    best = model_dir / 'best_eval.csv'
    if best.exists():
        return best
    return model_dir / 'eval.csv'


def _build_eval_uid_set_mismatch_message(eval_uid_infos):
    if not eval_uid_infos:
        return 'Ensemble eval-set mismatch detected, but no eval UID info was collected.'

    uid_sets = [info['uids'] for info in eval_uid_infos.values()]
    common_uids = set.intersection(*uid_sets) if uid_sets else set()
    union_uids = set.union(*uid_sets) if uid_sets else set()

    lines = [
        'Ensemble model eval-set mismatch detected. Selected models do not share the same eval UID set.',
        'This usually means train/eval directories were mixed, for example model vs model.eval.',
        f'Common utterances: {len(common_uids)} / union {len(union_uids)}',
        'Per-model eval coverage:',
    ]
    for model_name, info in eval_uid_infos.items():
        source_counts = info.get('source_counts') or {}
        source_desc = ', '.join(f'{key or "<empty>"}={value}' for key, value in sorted(source_counts.items()))
        if not source_desc:
            source_desc = 'n/a'
        only_here = len(info['uids'] - common_uids)
        missing_from_model = len(common_uids - info['uids'])
        lines.append(
            f'  - {model_name}: rows={info["n_rows"]}, uids={info["n_uids"]}, '
            f'only_here={only_here}, missing_common={missing_from_model}, sources=[{source_desc}]'
        )

    suspicious_pairs = []
    names = list(eval_uid_infos.keys())
    name_set = set(names)
    for name in names:
        if name.endswith('.eval'):
            base = name[:-5]
            if base in name_set:
                suspicious_pairs.append((base, name))
        else:
            eval_name = f'{name}.eval'
            if eval_name in name_set:
                suspicious_pairs.append((name, eval_name))
    if suspicious_pairs:
        rendered = ', '.join(f'{left} vs {right}' for left, right in sorted(set(suspicious_pairs)))
        lines.append(f'Suspicious train/eval name pairs: {rendered}')

    lines.append('If this mismatch is intentional, rerun with --allow_eval_mismatch.')
    return '\n'.join(lines)


def _validate_eval_uid_sets(eval_uid_infos, allow_mismatch=False, verbose=True):
    if not eval_uid_infos:
        return set()
    uid_sets = [info['uids'] for info in eval_uid_infos.values()]
    if all(uid_set == uid_sets[0] for uid_set in uid_sets[1:]):
        return uid_sets[0]

    message = _build_eval_uid_set_mismatch_message(eval_uid_infos)
    if allow_mismatch:
        if verbose:
            print(f'WARNING: {message}')
        return set.intersection(*uid_sets)
    raise ValueError(message)


def _get_model_dir_for_fold(model_name, fold):
    return WORKING_BASE / model_name / str(fold)


_MODEL_FLAGS_CACHE = {}


def _load_saved_flags(model_dir):
    model_dir = Path(model_dir)
    cache_key = str(model_dir)
    if cache_key in _MODEL_FLAGS_CACHE:
        return _MODEL_FLAGS_CACHE[cache_key]
    flags_path = model_dir / 'flags.json'
    if not flags_path.exists():
        _MODEL_FLAGS_CACHE[cache_key] = {}
        return _MODEL_FLAGS_CACHE[cache_key]
    try:
        with open(flags_path) as f:
            flags_data = json.load(f)
    except Exception:
        flags_data = {}
    _MODEL_FLAGS_CACHE[cache_key] = flags_data
    return flags_data


def _should_skip_tdt_primary_ctc_candidates(model_name):
    if not bool(getattr(FLAGS, 'ensemble_skip_tdt_primary_ctc_candidates', True)):
        return False
    saved_flags = _load_saved_flags(get_model_dir(model_name))
    decode_method = str(saved_flags.get('decode_method', 'auto') or 'auto')
    s2s_decoder = str(saved_flags.get('s2s_decoder', 'native') or 'native')
    ctc_only = bool(saved_flags.get('ctc_only', False))
    return (decode_method == 'tdt') and (s2s_decoder == 'tdt_reuse') and (not ctc_only)


def _read_metrics_row(model_name, fold):
    metrics_csv = _get_model_dir_for_fold(model_name, fold) / 'metrics.csv'
    if not metrics_csv.exists():
        return None
    try:
        mdf = pd.read_csv(metrics_csv)
        if mdf.empty:
            return None
        if 'score' in mdf.columns:
            score_series = pd.to_numeric(mdf['score'], errors='coerce')
            valid = score_series.notna()
            if valid.any():
                best_idx = score_series[valid].idxmin()
                return mdf.loc[best_idx].to_dict()
        return mdf.iloc[-1].to_dict()
    except Exception:
        return None


def _load_model_predictions(model_name, folds):
    preds = {}
    gold = {}
    meta = {}
    for fold in folds:
        eval_csv = get_eval_csv(_get_model_dir_for_fold(model_name, fold))
        if not eval_csv.exists():
            continue
        df = pd.read_csv(eval_csv)
        for _, row in df.iterrows():
            uid = row['utterance_id']
            pred = str(row['pred']) if pd.notna(row['pred']) else ''
            label = str(row['label']) if pd.notna(row['label']) else ''
            preds[uid] = pred
            if uid not in gold:
                gold[uid] = label
                meta[uid] = {
                    'source': row.get('source', ''),
                    'age_bucket': row.get('age_bucket', ''),
                }
    return preds, gold, meta


def _get_model_score_summary(model_name, folds=None):
    folds = list(folds) if folds is not None else [FOLD]
    if len(folds) == 1:
        row = _read_metrics_row(model_name, folds[0])
        if row is not None:
            return {
                'overall_cer': float(row.get('score', float('nan'))),
                'source_results': {
                    'dd': float(row.get('score/dd', float('nan'))),
                    'ext': float(row.get('score/ext', float('nan'))),
                },
            }

    preds, gold, meta = _load_model_predictions(model_name, folds)
    if not preds:
        return None
    return _evaluate(preds, gold, meta, label=model_name, verbose=False)


def _print_individual_model_scores(model_names, folds=None):
    print(f'\n--- Individual Model Scores ---')
    for mn in model_names:
        sr = _get_model_score_summary(mn, folds=folds)
        if sr is None:
            continue
        dd_cer = sr['source_results'].get('dd', float('nan'))
        ext_cer = sr['source_results'].get('ext', float('nan'))
        print(f'  {mn}: CER={sr["overall_cer"]:.5f} (dd={dd_cer:.5f}, ext={ext_cer:.5f})')


def _normalize_ensemble_cli_argv(argv):
    normalized = list(argv)
    corrected = False
    mode_flags = {'--ensemble_mode', '--emode'}
    for i, arg in enumerate(normalized):
        if arg.startswith('--ensemble_mode=') or arg.startswith('--emode='):
            flag, value = arg.split('=', 1)
            if value == 'tree_reanker':
                normalized[i] = f'{flag}=tree_reranker'
                corrected = True
        elif arg in mode_flags and i + 1 < len(normalized) and normalized[i + 1] == 'tree_reanker':
            normalized[i + 1] = 'tree_reranker'
            corrected = True
    if corrected:
        print('NOTE: normalized ensemble mode alias tree_reanker -> tree_reranker', file=sys.stderr)
    return normalized


def _resolve_tree_mode_and_task(mode, tree_task):
    # Simple aliases (no task override)
    simple_aliases = {
        'tree': 'tree_reranker',
        'nbest': 'nbest_rescore',
        'nbest2': 'nbest_rescore2',
    }
    if mode in simple_aliases:
        return simple_aliases[mode], tree_task

    # Tree task-override aliases
    alias_to_task = {
        'tree_ranker': 'ranking',
        'tree_regression': 'regression',
    }
    resolved_task = alias_to_task.get(mode)
    if resolved_task is None:
        return mode, tree_task
    if tree_task != resolved_task:
        print(
            f'NOTE: ensemble_mode={mode} overrides ensemble_tree_task={tree_task} -> {resolved_task}',
            file=sys.stderr,
        )
    return 'tree_reranker', resolved_task


_NBEST_MP_BUILD = {}


def _normalize_candidate_text(text):
    if text is None:
        return ''
    if isinstance(text, float) and np.isnan(text):
        return ''
    return normalize_ipa(str(text)).strip()


def _parse_serialized_text_list(value):
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            items = json.loads(text)
        except Exception:
            try:
                items = ast.literal_eval(text)
            except Exception:
                return []
    if not isinstance(items, (list, tuple)):
        return []
    return [_normalize_candidate_text(item) for item in items if _normalize_candidate_text(item)]


def _parse_serialized_float_list(value):
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            items = json.loads(text)
        except Exception:
            try:
                items = ast.literal_eval(text)
            except Exception:
                return []
    if not isinstance(items, (list, tuple)):
        return []
    scores = []
    for item in items:
        try:
            score = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(score):
            scores.append(score)
    return scores


def _append_candidate_text(candidate_list, candidate_set, text):
    text = _normalize_candidate_text(text)
    if not text or text in candidate_set:
        return False
    candidate_set.add(text)
    candidate_list.append(text)
    return True


_MODEL_CTC_META_FACTORY = None
_CTC_SCORE_ADJUST_FN = None


def _get_model_ctc_meta(model_names, all_logprobs):
    if callable(_MODEL_CTC_META_FACTORY):
        try:
            meta = _MODEL_CTC_META_FACTORY(model_names, all_logprobs)
            return meta or {}
        except Exception as e:
            print(f'WARNING: model CTC meta factory failed: {e}')
            return {}
    return {}


def _get_model_blank_id(model_name, log_probs, blank_id, model_ctc_meta=None):
    meta = (model_ctc_meta or {}).get(model_name) or {}
    if meta.get('blank_id') is not None:
        return int(meta['blank_id'])
    if meta.get('blank_last', False):
        return int(log_probs.shape[-1] - 1)
    return int(blank_id)


def _encode_candidate_text_for_model(cand_text, model_name, char_to_id=None, model_ctc_meta=None):
    meta = (model_ctc_meta or {}).get(model_name) or {}
    text_to_ids = meta.get('text_to_ids')
    if callable(text_to_ids):
        try:
            return [int(x) for x in (text_to_ids(cand_text) or [])]
        except Exception:
            return []
    if char_to_id is not None:
        return [char_to_id[ch] for ch in cand_text if ch in char_to_id]
    return []


def _get_candidate_token_ids_for_model(cand_text, model_name, *, char_to_id=None,
                                       model_ctc_meta=None, preserved_token_ids=None):
    if preserved_token_ids is not None and cand_text in preserved_token_ids:
        try:
            return [int(x) for x in preserved_token_ids[cand_text]]
        except Exception:
            pass
    return _encode_candidate_text_for_model(
        cand_text, model_name, char_to_id=char_to_id, model_ctc_meta=model_ctc_meta)


def _decode_nbest_hyps_for_model(log_probs, model_name, blank_id, beam_width, nbest,
                                 id_to_char=None, lm=None, lm_weight=0.0,
                                 prefix_beam_search_nbest_fn=None,
                                 model_ctc_meta=None,
                                 return_token_ids=False):
    if prefix_beam_search_nbest_fn is None:
        prefix_beam_search_nbest_fn = prefix_beam_search_nbest
    meta = (model_ctc_meta or {}).get(model_name) or {}
    decode_ids_to_text = meta.get('decode_ids_to_text')
    model_blank_id = _get_model_blank_id(model_name, log_probs, blank_id, model_ctc_meta)
    if callable(decode_ids_to_text):
        raw_hyps = prefix_beam_search_nbest_fn(
            log_probs, model_blank_id, beam_width, nbest=nbest, id_to_char=None)
        hyps = []
        for score, token_ids in raw_hyps:
            try:
                text = decode_ids_to_text(token_ids)
            except Exception:
                text = ''
            text = _normalize_candidate_text(text)
            if text:
                if return_token_ids:
                    hyps.append((score, text, [int(x) for x in token_ids]))
                else:
                    hyps.append((score, text))
        return hyps
    return prefix_beam_search_nbest_fn(
        log_probs, model_blank_id, beam_width, nbest=nbest, id_to_char=id_to_char,
        lm=lm, lm_weight=lm_weight)


def _adjust_ctc_scores(scores, candidates, token_lists, model_name=None, model_ctc_meta=None):
    if callable(_CTC_SCORE_ADJUST_FN):
        try:
            adjusted = _CTC_SCORE_ADJUST_FN(
                np.asarray(scores, dtype=np.float64),
                candidates=candidates,
                token_lists=token_lists,
                model_name=model_name,
                model_ctc_meta=model_ctc_meta or {},
            )
            adjusted = np.asarray(adjusted, dtype=np.float64)
            if adjusted.shape == np.asarray(scores).shape:
                return adjusted
        except Exception as e:
            print(f'WARNING: ctc score adjust failed for {model_name}: {e}')
    return np.asarray(scores, dtype=np.float64)


def _build_nbest_candidates_single_uid(uid):
    state = _NBEST_MP_BUILD
    model_names = state['model_names']
    all_logprobs = state['all_logprobs']
    all_eval_preds = state['all_eval_preds']
    all_eval_nbest_texts = state.get('all_eval_nbest_texts', {})
    all_dual_head_preds = state['all_dual_head_preds']
    skip_ctc_candidate_models = state.get('skip_ctc_candidate_models', {})
    blank_id = state['blank_id']
    beam_width = state['beam_width']
    nbest = state['nbest']
    id_to_char = state['id_to_char']
    lm = state['lm']
    lm_weight = state['lm_weight']
    use_lm = state['use_lm']
    tdt_eval_nbest = int(state.get('tdt_eval_nbest', 0) or 0)
    prefix_beam_search_nbest_fn = state['prefix_beam_search_nbest']
    model_ctc_meta = state.get('model_ctc_meta', {})

    candidate_set = set()
    candidate_list = []
    candidate_token_ids_by_model = {}
    raw_count = 0
    for mn in model_names:
        skip_ctc_candidates = bool(skip_ctc_candidate_models.get(mn, False))
        if (not skip_ctc_candidates) and mn in all_logprobs and uid in all_logprobs[mn]:
            lp = all_logprobs[mn][uid].astype(np.float32)
            hyps = _decode_nbest_hyps_for_model(
                lp, mn, blank_id, beam_width, nbest,
                id_to_char=id_to_char, lm=lm, lm_weight=lm_weight,
                prefix_beam_search_nbest_fn=prefix_beam_search_nbest_fn,
                model_ctc_meta=model_ctc_meta,
                return_token_ids=True)
            raw_count += len(hyps)
            for hyp in hyps:
                if len(hyp) == 3:
                    _score, text, token_ids = hyp
                else:
                    _score, text = hyp
                    token_ids = None
                _append_candidate_text(candidate_list, candidate_set, text)
                if token_ids is not None:
                    model_token_ids = candidate_token_ids_by_model.setdefault(mn, {})
                    model_token_ids.setdefault(text, [int(x) for x in token_ids])

        pred = all_eval_preds.get(mn, {}).get(uid, '')
        if pred:
            raw_count += 1
            _append_candidate_text(candidate_list, candidate_set, pred)

        if tdt_eval_nbest > 0:
            for text in all_eval_nbest_texts.get(mn, {}).get(uid, [])[:tdt_eval_nbest]:
                if text:
                    raw_count += 1
                    _append_candidate_text(candidate_list, candidate_set, text)

        dual_pred = all_dual_head_preds.get(mn, {}).get(uid, {})
        dual_cols = ('pred_tdt', 'pred_primary') if skip_ctc_candidates else ('pred_ctc', 'pred_tdt', 'pred_primary')
        for col in dual_cols:
            if dual_pred.get(col):
                raw_count += 1
                _append_candidate_text(candidate_list, candidate_set, dual_pred[col])

    candidates = candidate_list
    return uid, candidates, raw_count, len(candidates), candidate_token_ids_by_model


def _nbest_rescore_single_uid(uid):
    state = _NBEST_MP_BUILD
    all_logprobs = state['all_logprobs']
    all_tdt_scores = state['all_tdt_scores']
    blank_id = state['blank_id']
    char_to_id = state['char_to_id']
    use_lm = state['use_lm']
    lm = state['lm']
    lm_weight = state['lm_weight']
    ctc_force_score_batch_fn = state['ctc_force_score_batch']
    ctc_score_weight = state['ctc_score_weight']
    tdt_score_weight = state['tdt_score_weight']
    candidates = state['candidate_lists'][uid]
    candidate_token_ids_by_model = state.get('candidate_token_ids_by_model', {}).get(uid, {})
    model_ctc_meta = state.get('model_ctc_meta', {})

    if len(candidates) == 0:
        return uid, ''
    if len(candidates) == 1:
        return uid, candidates[0]

    avg_scores = np.zeros(len(candidates), dtype=np.float64)
    avg_weights = np.zeros(len(candidates), dtype=np.float64)

    if ctc_score_weight > 0:
        for mn, uid_to_logprobs in all_logprobs.items():
            if uid not in uid_to_logprobs:
                continue
            lp = uid_to_logprobs[uid].astype(np.float32)
            model_blank_id = _get_model_blank_id(mn, lp, blank_id, model_ctc_meta)
            preserved_token_ids = candidate_token_ids_by_model.get(mn, {})
            all_token_ids = [
                _get_candidate_token_ids_for_model(
                    cand_text, mn,
                    char_to_id=char_to_id,
                    model_ctc_meta=model_ctc_meta,
                    preserved_token_ids=preserved_token_ids)
                for cand_text in candidates
            ]
            scores = ctc_force_score_batch_fn(lp, all_token_ids, blank=model_blank_id)
            scores = _adjust_ctc_scores(
                scores,
                candidates=candidates,
                token_lists=all_token_ids,
                model_name=mn,
                model_ctc_meta=model_ctc_meta,
            )
            for i, sc in enumerate(scores):
                avg_scores[i] += ctc_score_weight * sc
                avg_weights[i] += ctc_score_weight

    if tdt_score_weight > 0:
        for mn, uid_to_scores in all_tdt_scores.items():
            if uid not in uid_to_scores:
                continue
            scores = uid_to_scores[uid]
            assert len(scores) == len(candidates), (
                f'TDT score size mismatch for {mn}/{uid}: '
                f'{len(scores)} vs {len(candidates)}')
            for i, sc in enumerate(scores):
                avg_scores[i] += tdt_score_weight * float(sc)
                avg_weights[i] += tdt_score_weight

    valid_mask = avg_weights > 0
    if np.any(valid_mask):
        avg_scores[valid_mask] /= avg_weights[valid_mask]
    else:
        return uid, candidates[0]

    if use_lm:
        for i, cand_text in enumerate(candidates):
            lm_score = 0.0
            ctx = ''
            for ch in cand_text:
                lm_score += lm.score(ctx, ch)
                ctx += ch
            lm_score += lm.score(ctx, '$')
            avg_scores[i] += lm_weight * lm_score

    best_idx = int(np.argmax(avg_scores))
    return uid, candidates[best_idx]


def _compute_ctc_avg_scores(candidates, all_logprobs, uid, blank_id, char_to_id,
                            ctc_score_weight=1.0, candidate_token_ids_by_model=None,
                            model_ctc_meta=None):
    if not candidates:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    avg_scores = np.zeros(len(candidates), dtype=np.float64)
    avg_weights = np.zeros(len(candidates), dtype=np.float64)

    if ctc_score_weight > 0:
        for _mn, uid_to_logprobs in all_logprobs.items():
            if uid not in uid_to_logprobs:
                continue
            lp = uid_to_logprobs[uid].astype(np.float32)
            model_blank_id = _get_model_blank_id(_mn, lp, blank_id, model_ctc_meta)
            preserved_token_ids = (candidate_token_ids_by_model or {}).get(_mn, {})
            all_token_ids = [
                _get_candidate_token_ids_for_model(
                    cand_text, _mn,
                    char_to_id=char_to_id,
                    model_ctc_meta=model_ctc_meta,
                    preserved_token_ids=preserved_token_ids)
                for cand_text in candidates
            ]
            scores = ctc_force_score_batch(lp, all_token_ids, blank=model_blank_id)
            for i, sc in enumerate(scores):
                avg_scores[i] += ctc_score_weight * float(sc)
                avg_weights[i] += ctc_score_weight

    valid_mask = avg_weights > 0
    if np.any(valid_mask):
        avg_scores[valid_mask] /= avg_weights[valid_mask]
    return avg_scores, avg_weights


def _ctc_prune_one_uid(uid, candidate_lists, all_logprobs, blank_id, char_to_id,
                       ctc_score_weight, prune_topk, force_keep_tdt_preds,
                       tdt_model_names, all_eval_preds, all_dual_head_preds,
                       all_eval_nbest_texts=None, tdt_eval_nbest=0,
                       candidate_token_ids_by_model=None,
                       model_ctc_meta=None):
    """Process one uid for CTC prune. Thread-safe (all shared state is read-only)."""
    candidates = candidate_lists.get(uid, [])
    force_keep = (_build_force_keep_tdt_candidates(
        uid, tdt_model_names, all_eval_preds, all_dual_head_preds,
        all_eval_nbest_texts=all_eval_nbest_texts,
        tdt_eval_nbest=tdt_eval_nbest)
                  if force_keep_tdt_preds else [])
    ctc_scores, ctc_weights = _compute_ctc_avg_scores(
        candidates, all_logprobs, uid, blank_id, char_to_id,
        ctc_score_weight=ctc_score_weight,
        candidate_token_ids_by_model=(candidate_token_ids_by_model or {}).get(uid, {}),
        model_ctc_meta=model_ctc_meta)
    ctc_score_map = {}
    if len(candidates) and len(ctc_scores):
        for cand_text, sc, wt in zip(candidates, ctc_scores, ctc_weights):
            if wt > 0:
                ctc_score_map[cand_text] = float(sc)

    if len(candidates) and len(ctc_scores):
        ranked = [cand for cand, _ in sorted(
            zip(candidates, ctc_scores.tolist()), key=lambda x: x[1], reverse=True)]
    else:
        ranked = list(candidates)

    next_candidates = []
    next_set = set()
    for cand in force_keep:
        _append_candidate_text(next_candidates, next_set, cand)
    for cand in ranked:
        if len(next_candidates) >= prune_topk and cand not in next_set:
            continue
        _append_candidate_text(next_candidates, next_set, cand)
        if len(next_candidates) >= prune_topk and all(fc in next_set for fc in force_keep):
            break

    return uid, force_keep, ctc_score_map, next_candidates


def _build_force_keep_tdt_candidates(uid, tdt_model_names, all_eval_preds, all_dual_head_preds,
                                     all_eval_nbest_texts=None, tdt_eval_nbest=0):
    keep_set = set()
    keep_list = []
    for mn in tdt_model_names:
        pred = all_eval_preds.get(mn, {}).get(uid, '')
        _append_candidate_text(keep_list, keep_set, pred)

        if tdt_eval_nbest > 0 and all_eval_nbest_texts is not None:
            for text in all_eval_nbest_texts.get(mn, {}).get(uid, [])[:int(tdt_eval_nbest)]:
                _append_candidate_text(keep_list, keep_set, text)

        dual_pred = all_dual_head_preds.get(mn, {}).get(uid, {})
        for col in ('pred_tdt', 'pred_primary'):
            if dual_pred.get(col):
                _append_candidate_text(keep_list, keep_set, dual_pred[col])
    return keep_list


def _nbest_rescore2_single_uid(uid):
    state = _NBEST_MP_BUILD
    candidates = state['candidate_lists'][uid]
    ctc_score_map = state['ctc_score_maps'].get(uid, {})
    all_tdt_scores = state['all_tdt_scores']
    tdt_score_weight = state['tdt_score_weight']
    ctc_score_weight = state['ctc_score_weight']
    use_lm = state['use_lm']
    lm = state['lm']
    lm_weight = state['lm_weight']

    if len(candidates) == 0:
        return uid, ''
    if len(candidates) == 1:
        return uid, candidates[0]

    avg_scores = np.zeros(len(candidates), dtype=np.float64)
    avg_weights = np.zeros(len(candidates), dtype=np.float64)

    if ctc_score_weight > 0:
        for i, cand_text in enumerate(candidates):
            if cand_text in ctc_score_map:
                avg_scores[i] += ctc_score_weight * float(ctc_score_map[cand_text])
                avg_weights[i] += ctc_score_weight

    if tdt_score_weight > 0:
        for _mn, uid_to_scores in all_tdt_scores.items():
            if uid not in uid_to_scores:
                continue
            scores = uid_to_scores[uid]
            assert len(scores) == len(candidates), (
                f'TDT score size mismatch for {_mn}/{uid}: '
                f'{len(scores)} vs {len(candidates)}')
            for i, sc in enumerate(scores):
                avg_scores[i] += tdt_score_weight * float(sc)
                avg_weights[i] += tdt_score_weight

    valid_mask = avg_weights > 0
    if np.any(valid_mask):
        avg_scores[valid_mask] /= avg_weights[valid_mask]
    else:
        return uid, candidates[0]

    if use_lm:
        for i, cand_text in enumerate(candidates):
            lm_score = 0.0
            ctx = ''
            for ch in cand_text:
                lm_score += lm.score(ctx, ch)
                ctx += ch
            lm_score += lm.score(ctx, '$')
            avg_scores[i] += lm_weight * lm_score

    best_idx = int(np.argmax(avg_scores))
    return uid, candidates[best_idx]


def _score_tdt_candidates_for_model(model_name, candidate_lists, verbose=True, use_cache=True):
    import importlib
    import torch

    model_dir = get_model_dir(model_name)
    assert model_dir.exists(), f'Model dir not found: {model_dir}'
    best_pt = model_dir / 'model.pt'
    if not best_pt.exists():
        best_pt = model_dir / 'best.pt'
    assert best_pt.exists(), f'No model.pt or best.pt found in {model_dir}'

    import gezi as gz
    import melt as mt  # noqa: F401
    from gezi import FLAGS
    from src import config
    from src.preprocess import preprocess
    from src.dataset import Dataset as PaskettiDataset

    gz.init_flags()
    config.init()
    gz.restore_configs(str(model_dir))
    ctc_only = bool(getattr(FLAGS, 'ctc_only', False))
    ctc_weight = float(getattr(FLAGS, 'ctc_weight', 1.0) or 0.0)
    s2s_decoder = str(getattr(FLAGS, 's2s_decoder', 'native') or 'native')
    has_tdt_cfg = (not ctc_only) and (ctc_weight < 1.0) and (s2s_decoder == 'tdt_reuse')
    if not has_tdt_cfg:
        if verbose:
            reason = f'ctc_only={ctc_only}, ctc_weight={ctc_weight}, s2s_decoder={s2s_decoder}'
            print(f'  Skip TDT scoring for {model_name}: {reason}')
        return {}

    FLAGS.mode = 'eval'
    FLAGS.work_mode = 'eval'
    FLAGS.distributed = False
    FLAGS.num_workers = 0
    FLAGS.persistent_workers = False
    FLAGS.batch_size = 8
    FLAGS.eval_batch_size = 8

    # Override tdt_score_method from ensemble-level flag (saved model flags may lack it)
    _etsm = getattr(FLAGS, 'ensemble_tdt_score_method', 'exact')
    FLAGS.tdt_score_method = _etsm

    cache_enabled = bool(use_cache)
    cache_paths = _get_tdt_exact_cache_paths(model_name) if cache_enabled else []
    cache_scores = _load_tdt_exact_score_cache(cache_paths, verbose=verbose) if cache_enabled else {}

    total_pairs = sum(len(cands) for cands in candidate_lists.values())
    cache_hit_pairs = 0
    cache_hit_uids = 0
    missing_candidate_lists = {}
    for uid, candidates in candidate_lists.items():
        cached = cache_scores.get(uid, {})
        missing = [cand for cand in candidates if cand not in cached]
        if not missing:
            cache_hit_uids += 1
        cache_hit_pairs += len(candidates) - len(missing)
        if missing:
            missing_candidate_lists[uid] = missing
    if verbose and cache_enabled:
        coverage = (cache_hit_pairs / max(total_pairs, 1)) if total_pairs else 0.0
        print(f'  TDT exact cache for {model_name}: '
              f'hit_pairs={cache_hit_pairs}/{total_pairs} ({coverage:.1%}), '
              f'full_hit_uids={cache_hit_uids}/{len(candidate_lists)}')
    if not missing_candidate_lists:
        if verbose:
            print(f'  TDT exact cache satisfied all candidates for {model_name}')
        return {
            uid: np.asarray([cache_scores[uid][cand] for cand in candidate_lists[uid]], dtype=np.float32)
            for uid in candidate_lists
        }

    model_module = importlib.import_module(f'src.models.{FLAGS.model}')
    Model = model_module.Model
    try:
        model = Model()
    except Exception as exc:
        if verbose:
            print(f'  Skip TDT scoring for {model_name}: model init failed ({exc})')
        return {}

    if not hasattr(model, 'tdt_decoder'):
        if verbose:
            print(f'  Skip TDT scoring for {model_name}: no tdt_decoder')
        return {}

    gz.load_weights(model, str(best_pt), strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    df = preprocess(mode='train')
    if hasattr(df, 'columns') and 'utterance_id' in df.columns:
        keep_uids = set(missing_candidate_lists.keys())
        before = len(df)
        df = df[df['utterance_id'].isin(keep_uids)].reset_index(drop=True)
        if verbose:
            print(f'  TDT eval subset for {model_name}: {before} -> {len(df)} utterances')
    ds = PaskettiDataset(df, mode='eval')

    def _collate_eval_batch(batch):
        batch = [b for b in batch if b is not None]
        assert batch, 'All TDT rescoring samples in batch are None'

        first_feat = batch[0]['input_features']
        if np.asarray(first_feat).ndim == 1:
            waveforms = [torch.tensor(b['input_features'], dtype=torch.float32) for b in batch]
            lengths = [w.shape[0] for w in waveforms]
            max_len = max(lengths)
            input_features = torch.zeros(len(batch), max_len, dtype=torch.float32)
            attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
            for i, (w, l) in enumerate(zip(waveforms, lengths)):
                input_features[i, :l] = w
                attention_mask[i, :l] = 1
        else:
            input_features = torch.stack(
                [torch.tensor(b['input_features'], dtype=torch.float32) for b in batch], dim=0)
            attention_mask = None

        out = {
            'input_features': input_features,
            'labels': torch.full((len(batch), 1), -100, dtype=torch.long),
            'id': [b.get('id', '') for b in batch],
        }
        if attention_mask is not None:
            out['attention_mask'] = attention_mask
        return out

    test_dl = torch.utils.data.DataLoader(
        ds,
        batch_size=int(getattr(FLAGS, 'eval_batch_size', 8) or 8),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=_collate_eval_batch,
    )
    gz.set('do_generate', False)

    _cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    _amp_dtype = torch.bfloat16 if _cc >= (8, 0) else torch.float16
    autocast_ctx = (torch.amp.autocast('cuda', dtype=_amp_dtype)
                    if torch.cuda.is_available() else contextlib.nullcontext())

    scores_by_uid = {}
    n_scored = 0
    infer_total = len(test_dl)
    with torch.no_grad(), autocast_ctx:
        for batch in _iter_with_progress(
                test_dl,
                total=infer_total,
                desc=f'  TDT infer {model_name}',
                verbose=verbose):
            input_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    input_batch[k] = v.to(device, non_blocking=True)
                else:
                    input_batch[k] = v

            model(input_batch)
            enc_out = getattr(model, '_last_enc_out', None)
            enc_len = getattr(model, '_last_enc_len', None)
            if enc_out is None:
                raise RuntimeError(f'{model_name} forward did not populate _last_enc_out')

            # Batch scoring: gather all candidates across utterances in this batch
            batch_ids = batch.get('id', [])
            batch_uids = []       # uid for each scored utterance
            batch_n_cands = []    # number of candidates per utterance
            all_enc_slices = []   # expanded enc_out slices
            all_enc_lens = []     # expanded enc_lengths
            all_texts = []        # flattened candidate texts
            for i, uid in enumerate(batch_ids):
                candidates = missing_candidate_lists.get(uid)
                if not candidates:
                    continue
                n_cands = len(candidates)
                batch_uids.append(uid)
                batch_n_cands.append(n_cands)
                all_enc_slices.append(enc_out[i:i + 1].expand(n_cands, -1, -1))
                if enc_len is not None:
                    all_enc_lens.append(enc_len[i:i + 1].expand(n_cands))
                all_texts.extend(candidates)

            if not batch_uids:
                continue

            # Batched scoring with optional chunking to limit GPU memory
            total_cands = len(all_texts)
            cat_enc = torch.cat(all_enc_slices, dim=0)  # (total_cands, T, D)
            cat_enc_len = torch.cat(all_enc_lens, dim=0) if all_enc_lens else None

            chunk_size = int(getattr(FLAGS, 'ensemble_tdt_score_chunk', 64) or 0)
            if chunk_size > 0 and total_cands > chunk_size:
                score_parts = []
                for c_start in range(0, total_cands, chunk_size):
                    c_end = min(c_start + chunk_size, total_cands)
                    c_enc = cat_enc[c_start:c_end]
                    c_enc_len = cat_enc_len[c_start:c_end] if cat_enc_len is not None else None
                    c_texts = all_texts[c_start:c_end]
                    c_scores = model.score_tdt_texts(c_enc, c_texts, enc_lengths=c_enc_len)
                    score_parts.append(c_scores.detach().cpu().float())
                scores_flat = torch.cat(score_parts, dim=0).numpy()
            else:
                scores_flat = model.score_tdt_texts(
                    cat_enc, all_texts, enc_lengths=cat_enc_len)
                scores_flat = scores_flat.detach().cpu().float().numpy()

            # Split scores back per utterance
            offset = 0
            for uid, n_cands in zip(batch_uids, batch_n_cands):
                scores_by_uid[uid] = scores_flat[offset:offset + n_cands]
                offset += n_cands
                n_scored += 1

    if cache_enabled and scores_by_uid:
        cache_rows = []
        for uid, scores in scores_by_uid.items():
            candidates = missing_candidate_lists.get(uid, [])
            if len(candidates) != len(scores):
                raise ValueError(
                    f'TDT exact cache append size mismatch for {uid}: {len(candidates)} vs {len(scores)}')
            uid_cache = cache_scores.setdefault(uid, {})
            for cand_text, score in zip(candidates, scores):
                score = float(score)
                uid_cache[cand_text] = score
                cache_rows.append({
                    'uid': uid,
                    'candidate_text': cand_text,
                    'score': score,
                    'model_name': model_name,
                })
        _append_tdt_exact_score_cache(cache_paths, cache_rows, verbose=verbose)

    merged_scores = {}
    for uid, candidates in candidate_lists.items():
        uid_cache = cache_scores.get(uid, {})
        missing = [cand for cand in candidates if cand not in uid_cache]
        if missing:
            raise KeyError(f'Missing TDT exact cache entries for {model_name} uid={uid}: {missing[:3]}')
        merged_scores[uid] = np.asarray([uid_cache[cand] for cand in candidates], dtype=np.float32)

    if verbose:
        print(f'  TDT scored {n_scored} utterances for {model_name} '
              f'(cache_miss_uids={len(missing_candidate_lists)})')
    return merged_scores


def _get_tdt_exact_cache_paths(model_name, fold=None):
    fold = FOLD if fold is None else fold
    path = WORKING_BASE / model_name / str(fold) / 'tdt_exact_scores.jsonl'
    return [path]


def _clear_reranker_caches(model_names, folds, verbose=True):
    removed = []

    dataset_cache_path = WORKING_BASE / 'ensemble' / 'cache' / 'dataset.pkl'
    if dataset_cache_path.exists():
        dataset_cache_path.unlink()
        removed.append(dataset_cache_path)

    exact_cache_paths = []
    seen = set()
    for fold in folds:
        for model_name in model_names:
            for path in _get_tdt_exact_cache_paths(model_name, fold=fold):
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                exact_cache_paths.append(path)

    for path in exact_cache_paths:
        if path.exists():
            path.unlink()
            removed.append(path)

    if verbose:
        if removed:
            print('\n=== Cleared reranker caches ===')
            for path in removed:
                print(f'  removed: {path}')
        else:
            print('\n=== Cleared reranker caches ===')
            print('  nothing to remove')
    return removed


def _load_tdt_exact_score_cache(cache_paths, verbose=True):
    cache = defaultdict(dict)
    loaded_rows = 0
    used_paths = []
    for path in cache_paths:
        if not path.exists():
            continue
        used_paths.append(path)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                uid = row.get('uid') or row.get('utterance_id')
                cand_text = _normalize_candidate_text(row.get('candidate_text') or row.get('text'))
                score = row.get('score', row.get('tdt_score'))
                if (not uid) or (not cand_text) or score is None or pd.isna(score):
                    continue
                cache[uid][cand_text] = float(score)
                loaded_rows += 1
    if verbose and used_paths:
        print(f'  Loaded TDT exact cache rows={loaded_rows} from {len(used_paths)} path(s): '
              f'{[str(x) for x in used_paths]}')
    return dict(cache)


def _append_tdt_exact_score_cache(cache_paths, cache_rows, verbose=True):
    if not cache_rows:
        return
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in cache_rows)
    written_paths = []
    for path in cache_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a') as f:
            f.write(payload)
        written_paths.append(path)
    if verbose:
        print(f'  Appended {len(cache_rows)} TDT exact cache rows to '
              f'{[str(x) for x in written_paths]}')


def _detect_tdt_model_names(model_names, verbose=True):
    import gezi as gz
    from gezi import FLAGS as _FLAGS
    from src import config as _shared_config

    tdt_model_names = []
    for mn in model_names:
        model_dir = get_model_dir(mn)
        if not model_dir.exists():
            continue
        gz.init_flags()
        _shared_config.init()
        gz.restore_configs(str(model_dir))
        ctc_only = bool(getattr(_FLAGS, 'ctc_only', False))
        ctc_weight_cur = float(getattr(_FLAGS, 'ctc_weight', 1.0) or 0.0)
        s2s_decoder = str(getattr(_FLAGS, 's2s_decoder', 'native') or 'native')
        if (not ctc_only) and (ctc_weight_cur < 1.0) and (s2s_decoder == 'tdt_reuse'):
            tdt_model_names.append(mn)
        elif verbose:
            print(f'  TDT disabled for {mn}: ctc_only={ctc_only}, '
                  f'ctc_weight={ctc_weight_cur}, s2s_decoder={s2s_decoder}')
    return tdt_model_names


def _load_tdt_primary_texts(model_names, tdt_model_names, verbose=True):
    import torch

    all_eval_preds = {}
    all_eval_scores = {}
    all_eval_nbest_score_maps = {}
    for mn in model_names:
        eval_csv = get_eval_csv(get_model_dir(mn))
        eval_df = pd.read_csv(eval_csv)
        pred_col = 'pred' if 'pred' in eval_df.columns else 'text'
        all_eval_preds[mn] = {
            str(uid): _normalize_candidate_text(text)
            for uid, text in zip(eval_df['utterance_id'], eval_df[pred_col].fillna(''))
        }
        if 'pred_score' in eval_df.columns:
            all_eval_scores[mn] = {
                str(uid): float(score)
                for uid, score in zip(eval_df['utterance_id'], eval_df['pred_score'])
                if pd.notna(score)
            }
        else:
            all_eval_scores[mn] = {}

        uid_to_nbest_score_map = {}
        if ('pred_nbest_texts' in eval_df.columns) and ('pred_nbest_scores' in eval_df.columns):
            for uid, texts, scores in zip(
                    eval_df['utterance_id'],
                    eval_df['pred_nbest_texts'],
                    eval_df['pred_nbest_scores']):
                parsed_texts = _parse_serialized_text_list(texts)
                parsed_scores = _parse_serialized_float_list(scores)
                if not parsed_texts or not parsed_scores:
                    continue
                cand_scores = {}
                for cand_text, cand_score in zip(parsed_texts, parsed_scores):
                    if not cand_text:
                        continue
                    prev_score = cand_scores.get(cand_text)
                    if (prev_score is None) or (cand_score > prev_score):
                        cand_scores[cand_text] = float(cand_score)
                if cand_scores:
                    uid_to_nbest_score_map[str(uid)] = cand_scores
        all_eval_nbest_score_maps[mn] = uid_to_nbest_score_map

    all_dual_head_preds = {}
    for mn in model_names:
        dual_path = get_model_dir(mn) / 'dual_head_preds.pt'
        if dual_path.exists():
            dual_data = torch.load(str(dual_path), map_location='cpu', weights_only=False)
            all_dual_head_preds[mn] = dual_data.get('preds', {})

    primary_tdt_texts = {}
    primary_tdt_scores = {}
    tdt_nbest_score_maps = {}
    for mn in tdt_model_names:
        uid_to_text = {}
        uid_to_score = {}
        eval_preds = all_eval_preds.get(mn, {})
        eval_scores = all_eval_scores.get(mn, {})
        dual_preds = all_dual_head_preds.get(mn, {})
        all_uids = set(eval_preds.keys()) | set(dual_preds.keys())
        for uid in all_uids:
            dual_info = dual_preds.get(uid, {}) or {}
            eval_text = _normalize_candidate_text(eval_preds.get(uid, ''))
            text = _normalize_candidate_text(
                dual_info.get('pred_tdt') or dual_info.get('pred_primary') or eval_text
            )
            if text:
                uid_to_text[uid] = text
                score = eval_scores.get(uid)
                if (score is not None) and (text == eval_text):
                    uid_to_score[uid] = float(score)
        primary_tdt_texts[mn] = uid_to_text
        primary_tdt_scores[mn] = uid_to_score
        tdt_nbest_score_maps[mn] = all_eval_nbest_score_maps.get(mn, {})
        if verbose:
            n_text = len(uid_to_text)
            n_score = len(uid_to_score)
            n_nbest_score = len(tdt_nbest_score_maps[mn])
            print(f'    {mn}: TDT primary texts={n_text}, scores={n_score}, '
                  f'nbest_score_uids={n_nbest_score}')

    if verbose and tdt_model_names:
        print(f'  TDT feature source models: {tdt_model_names}')
    return primary_tdt_texts, primary_tdt_scores, tdt_nbest_score_maps


def _build_tdt_feature_candidate_lists(df, tdt_model_names, primary_tdt_texts,
                                       topk=8, force_keep_preds=True):
    candidate_lists = {}
    topk = max(int(topk or 0), 1)
    for uid, group in df.groupby('uid', sort=False):
        ranked = group.sort_values('ctc_score_mean', ascending=False)['candidate_text'].tolist()
        keep = []
        keep_set = set()
        for cand_text in ranked[:topk]:
            _append_candidate_text(keep, keep_set, cand_text)
        if force_keep_preds:
            for mn in tdt_model_names:
                text = primary_tdt_texts.get(mn, {}).get(uid, '')
                _append_candidate_text(keep, keep_set, text)
        candidate_lists[uid] = keep
    return candidate_lists


def _is_wavlm_model_name(model_name, flags_data=None):
    name = str(model_name or '').strip().lower()
    backbone = str((flags_data or {}).get('backbone', '') or '').strip().lower()
    return ('wavlm' in name) or ('wavlm' in backbone)


def _is_nemo_model_name(model_name, flags_data=None):
    if _is_wavlm_model_name(model_name, flags_data=flags_data):
        return False
    name = str(model_name or '').strip().lower()
    model_kind = str((flags_data or {}).get('model', '') or '').strip().lower()
    backbone = str((flags_data or {}).get('backbone', '') or '').strip().lower()
    if model_kind == 'nemo':
        return True
    if 'nemo' in name:
        return True
    nemo_backbone_markers = ('parakeet', 'conformer', 'fastconformer', 'citrinet')
    return any(marker in backbone for marker in nemo_backbone_markers)


def _detect_wavlm_model_names(model_names, flags_by_name=None):
    detected = []
    for mn in model_names:
        flags_data = (flags_by_name or {}).get(mn)
        if flags_data is None:
            flags_data = _load_saved_flags(get_model_dir(mn))
        if _is_wavlm_model_name(mn, flags_data=flags_data):
            detected.append(mn)
    return detected


def _detect_nemo_model_names(model_names, flags_by_name=None):
    detected = []
    for mn in model_names:
        flags_data = (flags_by_name or {}).get(mn)
        if flags_data is None:
            flags_data = _load_saved_flags(get_model_dir(mn))
        if _is_nemo_model_name(mn, flags_data=flags_data):
            detected.append(mn)
    return detected


def _augment_family_group_features(df, feat_cols, model_names, family_name, subset_model_names,
                                   verbose=False, feat_edit_dist=False):
    if not subset_model_names:
        return df, feat_cols, []

    subset_set = set(subset_model_names)
    other_model_names = [mn for mn in model_names if mn not in subset_set]
    df, subset_cols = _augment_model_subset_feature_frame(
        df,
        group_name=family_name,
        subset_model_names=subset_model_names,
        feat_edit_dist=feat_edit_dist,
    )
    feat_cols = list(dict.fromkeys(list(feat_cols) + list(subset_cols)))
    added_cols = list(subset_cols)
    if verbose:
        print(f'  {family_name} subgroup feature columns added: {len(subset_cols)}')

    if other_model_names:
        other_group_name = f'non{family_name}'
        df, other_cols = _augment_model_subset_feature_frame(
            df,
            group_name=other_group_name,
            subset_model_names=other_model_names,
            feat_edit_dist=feat_edit_dist,
        )
        gap_cols = []
        for left, right, gap_name in [
            (f'{family_name}_ctc_score_mean', f'{other_group_name}_ctc_score_mean',
             f'{family_name}_vs_{other_group_name}_ctc_score_mean_gap'),
            (f'{family_name}_ctc_score_max', f'{other_group_name}_ctc_score_max',
             f'{family_name}_vs_{other_group_name}_ctc_score_max_gap'),
            (f'{family_name}_beam_best_vote_count', f'{other_group_name}_beam_best_vote_count',
             f'{family_name}_vs_{other_group_name}_beam_best_vote_gap'),
        ]:
            if left in df.columns and right in df.columns:
                df[gap_name] = df[left] - df[right]
                gap_cols.append(gap_name)
        # Cross-group edit distance gap features
        if feat_edit_dist:
            for ed_type in ['edit_dist_to_best_mean', 'word_edit_dist_to_best_mean']:
                left_col = f'{family_name}_{ed_type}'
                right_col = f'{other_group_name}_{ed_type}'
                gap_name = f'{family_name}_vs_{other_group_name}_{ed_type}_gap'
                if left_col in df.columns and right_col in df.columns:
                    df[gap_name] = df[left_col] - df[right_col]
                    gap_cols.append(gap_name)
        feat_cols = list(dict.fromkeys(list(feat_cols) + list(other_cols) + list(gap_cols)))
        added_cols.extend(list(other_cols) + list(gap_cols))
        if verbose:
            print(f'  {other_group_name} subgroup feature columns added: {len(other_cols) + len(gap_cols)}')

    return df, feat_cols, added_cols


def _augment_model_subset_feature_frame(df, group_name, subset_model_names,
                                        primary_texts=None,
                                        feat_edit_dist=False):
    if not subset_model_names:
        return df, []

    group_key = re.sub(r'[^0-9a-zA-Z]+', '_', str(group_name).strip().lower()).strip('_')
    if not group_key:
        return df, []

    score_cols = [f'ctc_score_{mn}' for mn in subset_model_names if f'ctc_score_{mn}' in df.columns]
    rank_cols = [f'beam_rank_{mn}' for mn in subset_model_names if f'beam_rank_{mn}' in df.columns]
    if not score_cols and not rank_cols and not primary_texts:
        return df, []

    df = df.copy()
    new_cols = []
    n_models = len(subset_model_names)
    candidate_len = df['candidate_text'].str.len().astype(float)
    score_prefix = f'{group_key}_ctc_score'

    if score_cols:
        df[f'{group_key}_n_models'] = float(n_models)
        df[f'{score_prefix}_mean'] = df[score_cols].mean(axis=1, skipna=True)
        df[f'{score_prefix}_std'] = df[score_cols].std(axis=1, skipna=True)
        df[f'{score_prefix}_min'] = df[score_cols].min(axis=1, skipna=True)
        df[f'{score_prefix}_max'] = df[score_cols].max(axis=1, skipna=True)
        df[f'{score_prefix}_range'] = df[f'{score_prefix}_max'] - df[f'{score_prefix}_min']
        df[f'{score_prefix}_mean_per_char'] = df[f'{score_prefix}_mean'] / candidate_len.clip(lower=1)

        grp_mean = df.groupby('uid')[f'{score_prefix}_mean']
        grp_max = df.groupby('uid')[f'{score_prefix}_max']
        mean_best = grp_mean.transform('max')
        mean_mean = grp_mean.transform('mean')
        mean_std = grp_mean.transform('std').replace(0.0, np.nan)
        max_best = grp_max.transform('max')
        max_mean = grp_max.transform('mean')
        max_std = grp_max.transform('std').replace(0.0, np.nan)

        df[f'{score_prefix}_mean_rank'] = grp_mean.rank(ascending=False, method='min', na_option='bottom')
        df[f'{score_prefix}_mean_diff_from_best'] = df[f'{score_prefix}_mean'] - mean_best
        df[f'{score_prefix}_mean_zscore'] = (df[f'{score_prefix}_mean'] - mean_mean) / mean_std
        df[f'{score_prefix}_mean_zscore'] = df[f'{score_prefix}_mean_zscore'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df[f'{score_prefix}_max_rank'] = grp_max.rank(ascending=False, method='min', na_option='bottom')
        df[f'{score_prefix}_max_diff_from_best'] = df[f'{score_prefix}_max'] - max_best
        df[f'{score_prefix}_max_zscore'] = (df[f'{score_prefix}_max'] - max_mean) / max_std
        df[f'{score_prefix}_max_zscore'] = df[f'{score_prefix}_max_zscore'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        new_cols.extend([
            f'{group_key}_n_models',
            f'{score_prefix}_mean', f'{score_prefix}_std', f'{score_prefix}_min',
            f'{score_prefix}_max', f'{score_prefix}_range', f'{score_prefix}_mean_per_char',
            f'{score_prefix}_mean_rank', f'{score_prefix}_mean_diff_from_best',
            f'{score_prefix}_mean_zscore', f'{score_prefix}_max_rank',
            f'{score_prefix}_max_diff_from_best', f'{score_prefix}_max_zscore',
        ])

    if rank_cols:
        best_vote_cols = []
        for col in rank_cols:
            vote_col = f'{col}_{group_key}_is_best'
            df[vote_col] = (df[col] == 0).astype(int)
            best_vote_cols.append(vote_col)
        df[f'{group_key}_beam_best_vote_count'] = df[best_vote_cols].sum(axis=1)
        df[f'{group_key}_beam_best_vote_frac'] = df[f'{group_key}_beam_best_vote_count'] / max(len(rank_cols), 1)
        grp_votes = df.groupby('uid')[f'{group_key}_beam_best_vote_count']
        vote_best = grp_votes.transform('max')
        vote_second = grp_votes.transform(lambda s: s.nlargest(2).iloc[-1] if len(s) >= 2 else s.max())
        df[f'{group_key}_beam_best_vote_is_top'] = (
            (df[f'{group_key}_beam_best_vote_count'] > 0) &
            np.isclose(df[f'{group_key}_beam_best_vote_count'], vote_best)
        ).astype(int)
        df[f'{group_key}_beam_best_vote_margin'] = df[f'{group_key}_beam_best_vote_count'] - vote_second
        df.loc[df[f'{group_key}_beam_best_vote_is_top'] == 0, f'{group_key}_beam_best_vote_margin'] = 0.0
        new_cols.extend([
            f'{group_key}_beam_best_vote_count', f'{group_key}_beam_best_vote_frac',
            f'{group_key}_beam_best_vote_is_top', f'{group_key}_beam_best_vote_margin',
        ])

    if primary_texts:
        hit_cols = []
        primary_unique_map = {}
        for uid in df['uid'].drop_duplicates().tolist():
            texts = {
                _normalize_candidate_text(text)
                for text in (primary_texts.get(mn, {}).get(uid, '') for mn in subset_model_names)
                if _normalize_candidate_text(text)
            }
            primary_unique_map[uid] = float(len(texts))
        df[f'{group_key}_primary_unique_count'] = df['uid'].map(primary_unique_map).fillna(0.0)
        new_cols.append(f'{group_key}_primary_unique_count')

        for mn in subset_model_names:
            uid_to_text = primary_texts.get(mn, {})
            hit_col = f'is_{group_key}_primary_pred_{mn}'
            text_map = df['uid'].map(uid_to_text)
            df[hit_col] = ((text_map.notna()) & (df['candidate_text'] == text_map)).astype(int)
            hit_cols.append(hit_col)
            new_cols.append(hit_col)

        if hit_cols:
            df[f'{group_key}_primary_hit_count'] = df[hit_cols].sum(axis=1)
            df[f'{group_key}_primary_hit_frac'] = df[f'{group_key}_primary_hit_count'] / max(len(hit_cols), 1)
            grp_hits = df.groupby('uid')[f'{group_key}_primary_hit_count']
            hit_best = grp_hits.transform('max')
            hit_second = grp_hits.transform(lambda s: s.nlargest(2).iloc[-1] if len(s) >= 2 else s.max())
            df[f'{group_key}_primary_hit_is_top'] = (
                (df[f'{group_key}_primary_hit_count'] > 0) &
                np.isclose(df[f'{group_key}_primary_hit_count'], hit_best)
            ).astype(int)
            df[f'{group_key}_primary_hit_margin'] = df[f'{group_key}_primary_hit_count'] - hit_second
            df.loc[df[f'{group_key}_primary_hit_is_top'] == 0, f'{group_key}_primary_hit_margin'] = 0.0
            new_cols.extend([
                f'{group_key}_primary_hit_count', f'{group_key}_primary_hit_frac',
                f'{group_key}_primary_hit_is_top', f'{group_key}_primary_hit_margin',
            ])

    # -- Per-group edit-distance & MBR-like features --
    if feat_edit_dist:
        # Character-level edit_dist_to_best aggregation
        ed_cols = [f'edit_dist_to_best_{mn}' for mn in subset_model_names
                   if f'edit_dist_to_best_{mn}' in df.columns]
        if ed_cols:
            ed_prefix = f'{group_key}_edit_dist_to_best'
            df[f'{ed_prefix}_mean'] = df[ed_cols].mean(axis=1, skipna=True)
            df[f'{ed_prefix}_min'] = df[ed_cols].min(axis=1, skipna=True)
            df[f'{ed_prefix}_max'] = df[ed_cols].max(axis=1, skipna=True)
            df[f'{ed_prefix}_std'] = df[ed_cols].std(axis=1, skipna=True)
            df[f'{ed_prefix}_range'] = df[f'{ed_prefix}_max'] - df[f'{ed_prefix}_min']
            new_cols.extend([
                f'{ed_prefix}_mean', f'{ed_prefix}_min', f'{ed_prefix}_max',
                f'{ed_prefix}_std', f'{ed_prefix}_range',
            ])
            # MBR-like ranking: rank candidates by mean edit distance within group
            grp_ed_mean = df.groupby('uid')[f'{ed_prefix}_mean']
            ed_best = grp_ed_mean.transform('min')
            ed_mean_val = grp_ed_mean.transform('mean')
            ed_std_val = grp_ed_mean.transform('std').replace(0.0, np.nan)
            df[f'{group_key}_edit_dist_rank'] = grp_ed_mean.rank(method='min', na_option='bottom')
            df[f'{group_key}_is_edit_dist_best'] = np.isclose(
                df[f'{ed_prefix}_mean'], ed_best).astype(int)
            df[f'{group_key}_edit_dist_diff_from_best'] = df[f'{ed_prefix}_mean'] - ed_best
            df[f'{group_key}_edit_dist_zscore'] = (
                (df[f'{ed_prefix}_mean'] - ed_mean_val) / ed_std_val
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            new_cols.extend([
                f'{group_key}_edit_dist_rank', f'{group_key}_is_edit_dist_best',
                f'{group_key}_edit_dist_diff_from_best', f'{group_key}_edit_dist_zscore',
            ])

        # Word-level edit_dist_to_best aggregation (word track only)
        wed_cols = [f'word_edit_dist_to_best_{mn}' for mn in subset_model_names
                    if f'word_edit_dist_to_best_{mn}' in df.columns]
        if wed_cols:
            wed_prefix = f'{group_key}_word_edit_dist_to_best'
            df[f'{wed_prefix}_mean'] = df[wed_cols].mean(axis=1, skipna=True)
            df[f'{wed_prefix}_min'] = df[wed_cols].min(axis=1, skipna=True)
            df[f'{wed_prefix}_max'] = df[wed_cols].max(axis=1, skipna=True)
            df[f'{wed_prefix}_std'] = df[wed_cols].std(axis=1, skipna=True)
            df[f'{wed_prefix}_range'] = df[f'{wed_prefix}_max'] - df[f'{wed_prefix}_min']
            new_cols.extend([
                f'{wed_prefix}_mean', f'{wed_prefix}_min', f'{wed_prefix}_max',
                f'{wed_prefix}_std', f'{wed_prefix}_range',
            ])
            grp_wed_mean = df.groupby('uid')[f'{wed_prefix}_mean']
            wed_best = grp_wed_mean.transform('min')
            wed_mean_val = grp_wed_mean.transform('mean')
            wed_std_val = grp_wed_mean.transform('std').replace(0.0, np.nan)
            df[f'{group_key}_word_edit_dist_rank'] = grp_wed_mean.rank(method='min', na_option='bottom')
            df[f'{group_key}_is_word_edit_dist_best'] = np.isclose(
                df[f'{wed_prefix}_mean'], wed_best).astype(int)
            df[f'{group_key}_word_edit_dist_diff_from_best'] = df[f'{wed_prefix}_mean'] - wed_best
            df[f'{group_key}_word_edit_dist_zscore'] = (
                (df[f'{wed_prefix}_mean'] - wed_mean_val) / wed_std_val
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            new_cols.extend([
                f'{group_key}_word_edit_dist_rank', f'{group_key}_is_word_edit_dist_best',
                f'{group_key}_word_edit_dist_diff_from_best', f'{group_key}_word_edit_dist_zscore',
            ])

    return df, [c for c in new_cols if c in df.columns]


def _convert_tdt_score_arrays(candidate_lists, scores_by_uid):
    score_map = {}
    for uid, scores in scores_by_uid.items():
        candidates = candidate_lists.get(uid, [])
        if len(candidates) != len(scores):
            raise ValueError(
                f'TDT score size mismatch for {uid}: {len(scores)} vs {len(candidates)} candidates')
        score_map[uid] = {
            cand_text: float(score)
            for cand_text, score in zip(candidates, scores)
        }
    return score_map


def _augment_tdt_feature_frame(df, tdt_model_names, primary_tdt_texts,
                               primary_tdt_scores=None,
                               tdt_nbest_score_maps=None,
                               tdt_score_maps=None,
                               include_light=True, include_primary_score=False,
                               include_nbest_score=False,
                               include_exact=True,
                               include_score_compare=True):
    df = df.copy()
    candidate_len = df['candidate_text'].str.len().astype(float)
    candidate_spaces = df['candidate_text'].str.count(' ').astype(float)
    new_cols = []

    hit_cols = []
    len_diff_cols = []
    space_diff_cols = []
    primary_score_cols = []
    primary_score_best_cols = []
    nbest_score_cols = []
    nbest_score_best_cols = []
    score_cols = []
    score_best_cols = []

    for mn in tdt_model_names:
        uid_to_text = primary_tdt_texts.get(mn, {})
        text_map = df['uid'].map(uid_to_text)
        len_map = {uid: float(len(text)) for uid, text in uid_to_text.items()}
        space_map = {uid: float(text.count(' ')) for uid, text in uid_to_text.items()}

        hit_col = f'is_tdt_pred_{mn}'
        len_diff_col = f'tdt_len_diff_{mn}'
        space_diff_col = f'tdt_spaces_diff_{mn}'

        if include_light:
            df[hit_col] = ((text_map.notna()) & (df['candidate_text'] == text_map)).astype(int)
            df[len_diff_col] = (candidate_len - df['uid'].map(len_map)).abs()
            df[space_diff_col] = (candidate_spaces - df['uid'].map(space_map)).abs()

            hit_cols.append(hit_col)
            len_diff_cols.append(len_diff_col)
            space_diff_cols.append(space_diff_col)
            new_cols.extend([hit_col, len_diff_col, space_diff_col])

            if include_primary_score:
                score_map = (primary_tdt_scores or {}).get(mn, {})
                primary_score_col = f'tdt_primary_score_{mn}'
                primary_per_char_col = f'tdt_primary_score_per_char_{mn}'
                primary_rank_col = f'tdt_primary_score_rank_{mn}'
                primary_pct_col = f'tdt_primary_score_pct_{mn}'
                primary_diff_col = f'tdt_primary_score_diff_from_best_{mn}'
                primary_centered_col = f'tdt_primary_score_centered_{mn}'
                primary_zscore_col = f'tdt_primary_score_zscore_{mn}'
                primary_is_best_col = f'tdt_primary_score_is_best_{mn}'
                primary_margin_second_col = f'tdt_primary_score_margin_to_second_{mn}'
                if score_map:
                    matched_scores = df['uid'].map(score_map)
                    df[primary_score_col] = matched_scores.where(df[hit_col] == 1, np.nan)
                else:
                    df[primary_score_col] = np.nan
                df[primary_per_char_col] = df[primary_score_col] / candidate_len.clip(lower=1)
                grp_primary_score = df.groupby('uid')[primary_score_col]
                primary_best_score = grp_primary_score.transform('max')
                primary_mean_score = grp_primary_score.transform('mean')
                primary_std_score = grp_primary_score.transform('std').replace(0.0, np.nan)
                primary_second_best_score = grp_primary_score.transform(
                    lambda s: s.nlargest(2).iloc[-1] if s.notna().any() else np.nan)
                df[primary_rank_col] = grp_primary_score.rank(
                    ascending=False, method='min', na_option='bottom')
                df[primary_pct_col] = grp_primary_score.rank(
                    ascending=False, pct=True, na_option='bottom')
                df[primary_diff_col] = df[primary_score_col] - primary_best_score
                df[primary_centered_col] = df[primary_score_col] - primary_mean_score
                df[primary_zscore_col] = (df[primary_score_col] - primary_mean_score) / primary_std_score
                df[primary_zscore_col] = df[primary_zscore_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                df[primary_is_best_col] = (
                    (df[primary_score_col].notna()) & np.isclose(df[primary_score_col], primary_best_score)
                ).astype(int)
                df[primary_margin_second_col] = df[primary_score_col] - primary_second_best_score
                df.loc[df[primary_is_best_col] == 0, primary_margin_second_col] = 0.0
                primary_score_cols.append(primary_score_col)
                primary_score_best_cols.append(primary_is_best_col)
                new_cols.extend([
                    primary_score_col, primary_per_char_col, primary_rank_col, primary_pct_col,
                    primary_diff_col, primary_centered_col, primary_zscore_col,
                    primary_is_best_col, primary_margin_second_col,
                ])

        score_col = f'tdt_score_{mn}'
        if include_nbest_score:
            score_rows = []
            uid_to_nbest_scores = (tdt_nbest_score_maps or {}).get(mn, {})
            nbest_score_col = f'tdt_nbest_score_{mn}'
            nbest_per_char_col = f'tdt_nbest_score_per_char_{mn}'
            nbest_rank_col = f'tdt_nbest_score_rank_{mn}'
            nbest_pct_col = f'tdt_nbest_score_pct_{mn}'
            nbest_diff_col = f'tdt_nbest_score_diff_from_best_{mn}'
            nbest_centered_col = f'tdt_nbest_score_centered_{mn}'
            nbest_zscore_col = f'tdt_nbest_score_zscore_{mn}'
            nbest_is_best_col = f'tdt_nbest_score_is_best_{mn}'
            nbest_margin_second_col = f'tdt_nbest_score_margin_to_second_{mn}'
            for uid, cand_scores in uid_to_nbest_scores.items():
                for cand_text, cand_score in cand_scores.items():
                    score_rows.append({
                        'uid': uid,
                        'candidate_text': cand_text,
                        nbest_score_col: float(cand_score),
                    })
            if score_rows:
                score_df = pd.DataFrame(score_rows)
                df = df.merge(score_df, on=['uid', 'candidate_text'], how='left')
            else:
                df[nbest_score_col] = np.nan
            df[nbest_per_char_col] = df[nbest_score_col] / candidate_len.clip(lower=1)
            grp_nbest_score = df.groupby('uid')[nbest_score_col]
            nbest_best_score = grp_nbest_score.transform('max')
            nbest_mean_score = grp_nbest_score.transform('mean')
            nbest_std_score = grp_nbest_score.transform('std').replace(0.0, np.nan)
            nbest_second_best_score = grp_nbest_score.transform(
                lambda s: s.nlargest(2).iloc[-1] if s.notna().any() else np.nan)
            df[nbest_rank_col] = grp_nbest_score.rank(
                ascending=False, method='min', na_option='bottom')
            df[nbest_pct_col] = grp_nbest_score.rank(
                ascending=False, pct=True, na_option='bottom')
            df[nbest_diff_col] = df[nbest_score_col] - nbest_best_score
            df[nbest_centered_col] = df[nbest_score_col] - nbest_mean_score
            df[nbest_zscore_col] = (df[nbest_score_col] - nbest_mean_score) / nbest_std_score
            df[nbest_zscore_col] = df[nbest_zscore_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            df[nbest_is_best_col] = (
                (df[nbest_score_col].notna()) & np.isclose(df[nbest_score_col], nbest_best_score)
            ).astype(int)
            df[nbest_margin_second_col] = df[nbest_score_col] - nbest_second_best_score
            df.loc[df[nbest_is_best_col] == 0, nbest_margin_second_col] = 0.0
            nbest_score_cols.append(nbest_score_col)
            nbest_score_best_cols.append(nbest_is_best_col)
            new_cols.extend([
                nbest_score_col, nbest_per_char_col, nbest_rank_col, nbest_pct_col,
                nbest_diff_col, nbest_centered_col, nbest_zscore_col,
                nbest_is_best_col, nbest_margin_second_col,
            ])

        score_col = f'tdt_score_{mn}'
        if include_exact and tdt_score_maps is not None:
            rows = []
            uid_to_scores = tdt_score_maps.get(mn, {})
            for uid, cand_scores in uid_to_scores.items():
                for cand_text, score in cand_scores.items():
                    rows.append({
                        'uid': uid,
                        'candidate_text': cand_text,
                        score_col: float(score),
                    })
            if rows:
                score_df = pd.DataFrame(rows)
                df = df.merge(score_df, on=['uid', 'candidate_text'], how='left')
            else:
                df[score_col] = np.nan
            per_char_col = f'tdt_score_per_char_{mn}'
            df[per_char_col] = df[score_col] / candidate_len.clip(lower=1)
            grp_score = df.groupby('uid')[score_col]
            rank_col = f'tdt_score_rank_{mn}'
            pct_col = f'tdt_score_pct_{mn}'
            diff_col = f'tdt_score_diff_from_best_{mn}'
            centered_col = f'tdt_score_centered_{mn}'
            zscore_col = f'tdt_score_zscore_{mn}'
            is_best_col = f'tdt_score_is_best_{mn}'
            margin_second_col = f'tdt_score_margin_to_second_{mn}'
            best_score = grp_score.transform('max')
            mean_score = grp_score.transform('mean')
            std_score = grp_score.transform('std').replace(0.0, np.nan)
            second_best_score = grp_score.transform(
                lambda s: s.nlargest(2).iloc[-1] if s.notna().any() else np.nan)
            df[rank_col] = grp_score.rank(ascending=False, method='min', na_option='bottom')
            df[pct_col] = grp_score.rank(ascending=False, pct=True, na_option='bottom')
            df[diff_col] = df[score_col] - best_score
            df[centered_col] = df[score_col] - mean_score
            df[zscore_col] = (df[score_col] - mean_score) / std_score
            df[zscore_col] = df[zscore_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            df[is_best_col] = ((df[score_col].notna()) & np.isclose(df[score_col], best_score)).astype(int)
            df[margin_second_col] = df[score_col] - second_best_score
            df.loc[df[is_best_col] == 0, margin_second_col] = 0.0
            score_cols.append(score_col)
            score_best_cols.append(is_best_col)
            new_cols.extend([
                score_col, per_char_col, rank_col, pct_col, diff_col,
                centered_col, zscore_col, is_best_col, margin_second_col,
            ])

    if include_light and hit_cols:
        df['n_tdt_pred_hits'] = df[hit_cols].sum(axis=1)
        new_cols.append('n_tdt_pred_hits')
    if include_light and len_diff_cols:
        df['tdt_len_diff_mean'] = df[len_diff_cols].mean(axis=1, skipna=True)
        df['tdt_len_diff_min'] = df[len_diff_cols].min(axis=1, skipna=True)
        df['tdt_len_diff_max'] = df[len_diff_cols].max(axis=1, skipna=True)
        new_cols.extend(['tdt_len_diff_mean', 'tdt_len_diff_min', 'tdt_len_diff_max'])
    if include_light and space_diff_cols:
        df['tdt_spaces_diff_mean'] = df[space_diff_cols].mean(axis=1, skipna=True)
        df['tdt_spaces_diff_min'] = df[space_diff_cols].min(axis=1, skipna=True)
        df['tdt_spaces_diff_max'] = df[space_diff_cols].max(axis=1, skipna=True)
        new_cols.extend(['tdt_spaces_diff_mean', 'tdt_spaces_diff_min', 'tdt_spaces_diff_max'])
    if include_light and include_primary_score and primary_score_cols:
        df['n_tdt_primary_scored_models'] = df[primary_score_cols].notna().sum(axis=1)
        df['tdt_primary_score_mean'] = df[primary_score_cols].mean(axis=1, skipna=True)
        df['tdt_primary_score_std'] = df[primary_score_cols].std(axis=1, skipna=True)
        df['tdt_primary_score_min'] = df[primary_score_cols].min(axis=1, skipna=True)
        df['tdt_primary_score_max'] = df[primary_score_cols].max(axis=1, skipna=True)
        df['tdt_primary_score_range'] = df['tdt_primary_score_max'] - df['tdt_primary_score_min']
        df['tdt_primary_score_mean_per_char'] = df['tdt_primary_score_mean'] / candidate_len.clip(lower=1)
        grp_primary_tdt = df.groupby('uid')['tdt_primary_score_mean']
        grp_ctc = df.groupby('uid')['ctc_score_mean']
        primary_tdt_best = grp_primary_tdt.transform('max')
        primary_tdt_mean = grp_primary_tdt.transform('mean')
        primary_tdt_std = grp_primary_tdt.transform('std').replace(0.0, np.nan)
        primary_tdt_second = grp_primary_tdt.transform(
            lambda s: s.nlargest(2).iloc[-1] if s.notna().any() else np.nan)
        ctc_best = grp_ctc.transform('max')
        df['tdt_primary_score_mean_rank'] = grp_primary_tdt.rank(
            ascending=False, method='min', na_option='bottom')
        df['tdt_primary_score_mean_pct'] = grp_primary_tdt.rank(
            ascending=False, pct=True, na_option='bottom')
        df['tdt_primary_score_diff_from_best'] = df['tdt_primary_score_mean'] - primary_tdt_best
        df['tdt_primary_score_mean_centered'] = df['tdt_primary_score_mean'] - primary_tdt_mean
        df['tdt_primary_score_mean_zscore'] = (
            df['tdt_primary_score_mean'] - primary_tdt_mean) / primary_tdt_std
        df['tdt_primary_score_mean_zscore'] = df['tdt_primary_score_mean_zscore'].replace(
            [np.inf, -np.inf], np.nan).fillna(0.0)
        df['tdt_primary_score_mean_is_best'] = (
            (df['tdt_primary_score_mean'].notna()) &
            np.isclose(df['tdt_primary_score_mean'], primary_tdt_best)
        ).astype(int)
        df['tdt_primary_score_mean_margin_to_second'] = (
            df['tdt_primary_score_mean'] - primary_tdt_second)
        df.loc[
            df['tdt_primary_score_mean_is_best'] == 0,
            'tdt_primary_score_mean_margin_to_second'
        ] = 0.0
        df['tdt_primary_score_best_vote_count'] = (
            df[primary_score_best_cols].sum(axis=1) if primary_score_best_cols else 0.0)
        df['tdt_primary_score_best_vote_frac'] = (
            df['tdt_primary_score_best_vote_count'] /
            df['n_tdt_primary_scored_models'].clip(lower=1)
        )
        grp_primary_votes = df.groupby('uid')['tdt_primary_score_best_vote_count']
        vote_best = grp_primary_votes.transform('max')
        df['tdt_primary_score_best_vote_is_top'] = (
            (df['tdt_primary_score_best_vote_count'] > 0) &
            np.isclose(df['tdt_primary_score_best_vote_count'], vote_best)
        ).astype(int)
        df['tdt_primary_score_best_vote_margin'] = (
            df['tdt_primary_score_best_vote_count'] - grp_primary_votes.transform(
                lambda s: s.nlargest(2).iloc[-1] if s.notna().any() else np.nan)
        )
        df.loc[
            df['tdt_primary_score_best_vote_is_top'] == 0,
            'tdt_primary_score_best_vote_margin'
        ] = 0.0
        new_cols.extend([
            'n_tdt_primary_scored_models', 'tdt_primary_score_mean', 'tdt_primary_score_std',
            'tdt_primary_score_min', 'tdt_primary_score_max', 'tdt_primary_score_range',
            'tdt_primary_score_mean_per_char', 'tdt_primary_score_mean_rank',
            'tdt_primary_score_mean_pct', 'tdt_primary_score_diff_from_best',
            'tdt_primary_score_mean_centered', 'tdt_primary_score_mean_zscore',
            'tdt_primary_score_mean_is_best', 'tdt_primary_score_mean_margin_to_second',
            'tdt_primary_score_best_vote_count', 'tdt_primary_score_best_vote_frac',
            'tdt_primary_score_best_vote_is_top', 'tdt_primary_score_best_vote_margin',
        ])
        if include_score_compare:
            df['tdt_primary_ctc_score_gap'] = df['tdt_primary_score_mean'] - df['ctc_score_mean']
            df['tdt_primary_score_ctc_rank_gap'] = df['tdt_primary_score_mean_rank'] - grp_ctc.rank(
                ascending=False, method='min', na_option='bottom')
            df['tdt_primary_score_mean_vs_ctc_best_gap'] = df['tdt_primary_score_mean'] - ctc_best
            new_cols.extend([
                'tdt_primary_ctc_score_gap',
                'tdt_primary_score_ctc_rank_gap',
                'tdt_primary_score_mean_vs_ctc_best_gap',
            ])
    if include_light and include_nbest_score and nbest_score_cols:
        df['n_tdt_nbest_scored_models'] = df[nbest_score_cols].notna().sum(axis=1)
        df['tdt_nbest_score_mean'] = df[nbest_score_cols].mean(axis=1, skipna=True)
        df['tdt_nbest_score_std'] = df[nbest_score_cols].std(axis=1, skipna=True)
        df['tdt_nbest_score_min'] = df[nbest_score_cols].min(axis=1, skipna=True)
        df['tdt_nbest_score_max'] = df[nbest_score_cols].max(axis=1, skipna=True)
        df['tdt_nbest_score_range'] = df['tdt_nbest_score_max'] - df['tdt_nbest_score_min']
        df['tdt_nbest_score_mean_per_char'] = df['tdt_nbest_score_mean'] / candidate_len.clip(lower=1)
        grp_nbest_tdt = df.groupby('uid')['tdt_nbest_score_mean']
        grp_ctc = df.groupby('uid')['ctc_score_mean']
        nbest_tdt_best = grp_nbest_tdt.transform('max')
        nbest_tdt_mean = grp_nbest_tdt.transform('mean')
        nbest_tdt_std = grp_nbest_tdt.transform('std').replace(0.0, np.nan)
        nbest_tdt_second = grp_nbest_tdt.transform(
            lambda s: s.nlargest(2).iloc[-1] if s.notna().any() else np.nan)
        ctc_best = grp_ctc.transform('max')
        df['tdt_nbest_score_mean_rank'] = grp_nbest_tdt.rank(
            ascending=False, method='min', na_option='bottom')
        df['tdt_nbest_score_mean_pct'] = grp_nbest_tdt.rank(
            ascending=False, pct=True, na_option='bottom')
        df['tdt_nbest_score_diff_from_best'] = df['tdt_nbest_score_mean'] - nbest_tdt_best
        df['tdt_nbest_score_mean_centered'] = df['tdt_nbest_score_mean'] - nbest_tdt_mean
        df['tdt_nbest_score_mean_zscore'] = (
            df['tdt_nbest_score_mean'] - nbest_tdt_mean) / nbest_tdt_std
        df['tdt_nbest_score_mean_zscore'] = df['tdt_nbest_score_mean_zscore'].replace(
            [np.inf, -np.inf], np.nan).fillna(0.0)
        df['tdt_nbest_score_mean_is_best'] = (
            (df['tdt_nbest_score_mean'].notna()) &
            np.isclose(df['tdt_nbest_score_mean'], nbest_tdt_best)
        ).astype(int)
        df['tdt_nbest_score_mean_margin_to_second'] = (
            df['tdt_nbest_score_mean'] - nbest_tdt_second)
        df.loc[
            df['tdt_nbest_score_mean_is_best'] == 0,
            'tdt_nbest_score_mean_margin_to_second'
        ] = 0.0
        df['tdt_nbest_score_best_vote_count'] = (
            df[nbest_score_best_cols].sum(axis=1) if nbest_score_best_cols else 0.0)
        df['tdt_nbest_score_best_vote_frac'] = (
            df['tdt_nbest_score_best_vote_count'] /
            df['n_tdt_nbest_scored_models'].clip(lower=1)
        )
        grp_nbest_votes = df.groupby('uid')['tdt_nbest_score_best_vote_count']
        nbest_vote_best = grp_nbest_votes.transform('max')
        df['tdt_nbest_score_best_vote_is_top'] = (
            (df['tdt_nbest_score_best_vote_count'] > 0) &
            np.isclose(df['tdt_nbest_score_best_vote_count'], nbest_vote_best)
        ).astype(int)
        df['tdt_nbest_score_best_vote_margin'] = (
            df['tdt_nbest_score_best_vote_count'] - grp_nbest_votes.transform(
                lambda s: s.nlargest(2).iloc[-1] if s.notna().any() else np.nan)
        )
        df.loc[
            df['tdt_nbest_score_best_vote_is_top'] == 0,
            'tdt_nbest_score_best_vote_margin'
        ] = 0.0
        new_cols.extend([
            'n_tdt_nbest_scored_models', 'tdt_nbest_score_mean', 'tdt_nbest_score_std',
            'tdt_nbest_score_min', 'tdt_nbest_score_max', 'tdt_nbest_score_range',
            'tdt_nbest_score_mean_per_char', 'tdt_nbest_score_mean_rank',
            'tdt_nbest_score_mean_pct', 'tdt_nbest_score_diff_from_best',
            'tdt_nbest_score_mean_centered', 'tdt_nbest_score_mean_zscore',
            'tdt_nbest_score_mean_is_best', 'tdt_nbest_score_mean_margin_to_second',
            'tdt_nbest_score_best_vote_count', 'tdt_nbest_score_best_vote_frac',
            'tdt_nbest_score_best_vote_is_top', 'tdt_nbest_score_best_vote_margin',
        ])
        if include_score_compare:
            df['tdt_nbest_ctc_score_gap'] = df['tdt_nbest_score_mean'] - df['ctc_score_mean']
            df['tdt_nbest_score_ctc_rank_gap'] = df['tdt_nbest_score_mean_rank'] - grp_ctc.rank(
                ascending=False, method='min', na_option='bottom')
            df['tdt_nbest_score_mean_vs_ctc_best_gap'] = df['tdt_nbest_score_mean'] - ctc_best
            new_cols.extend([
                'tdt_nbest_ctc_score_gap',
                'tdt_nbest_score_ctc_rank_gap',
                'tdt_nbest_score_mean_vs_ctc_best_gap',
            ])
    if include_exact and score_cols:
        df['n_tdt_scored_models'] = df[score_cols].notna().sum(axis=1)
        df['tdt_score_mean'] = df[score_cols].mean(axis=1, skipna=True)
        df['tdt_score_std'] = df[score_cols].std(axis=1, skipna=True)
        df['tdt_score_min'] = df[score_cols].min(axis=1, skipna=True)
        df['tdt_score_max'] = df[score_cols].max(axis=1, skipna=True)
        df['tdt_score_range'] = df['tdt_score_max'] - df['tdt_score_min']
        df['tdt_score_mean_per_char'] = df['tdt_score_mean'] / candidate_len.clip(lower=1)
        grp_tdt = df.groupby('uid')['tdt_score_mean']
        grp_ctc = df.groupby('uid')['ctc_score_mean']
        tdt_best = grp_tdt.transform('max')
        tdt_mean = grp_tdt.transform('mean')
        tdt_std = grp_tdt.transform('std').replace(0.0, np.nan)
        tdt_second = grp_tdt.transform(
            lambda s: s.nlargest(2).iloc[-1] if s.notna().any() else np.nan)
        ctc_best = grp_ctc.transform('max')
        df['tdt_score_mean_rank'] = grp_tdt.rank(ascending=False, method='min', na_option='bottom')
        df['tdt_score_mean_pct'] = grp_tdt.rank(ascending=False, pct=True, na_option='bottom')
        df['tdt_score_diff_from_best'] = df['tdt_score_mean'] - tdt_best
        df['tdt_score_mean_centered'] = df['tdt_score_mean'] - tdt_mean
        df['tdt_score_mean_zscore'] = (df['tdt_score_mean'] - tdt_mean) / tdt_std
        df['tdt_score_mean_zscore'] = df['tdt_score_mean_zscore'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df['tdt_score_mean_is_best'] = ((df['tdt_score_mean'].notna()) & np.isclose(df['tdt_score_mean'], tdt_best)).astype(int)
        df['tdt_score_mean_margin_to_second'] = df['tdt_score_mean'] - tdt_second
        df.loc[df['tdt_score_mean_is_best'] == 0, 'tdt_score_mean_margin_to_second'] = 0.0
        df['tdt_score_best_vote_count'] = df[score_best_cols].sum(axis=1) if score_best_cols else 0.0
        df['tdt_score_best_vote_frac'] = df['tdt_score_best_vote_count'] / df['n_tdt_scored_models'].clip(lower=1)
        grp_votes = df.groupby('uid')['tdt_score_best_vote_count']
        vote_best = grp_votes.transform('max')
        df['tdt_score_best_vote_is_top'] = ((df['tdt_score_best_vote_count'] > 0) &
                                            np.isclose(df['tdt_score_best_vote_count'], vote_best)).astype(int)
        df['tdt_score_best_vote_margin'] = df['tdt_score_best_vote_count'] - grp_votes.transform(
            lambda s: s.nlargest(2).iloc[-1] if s.notna().any() else np.nan)
        df.loc[df['tdt_score_best_vote_is_top'] == 0, 'tdt_score_best_vote_margin'] = 0.0
        new_cols.extend([
            'n_tdt_scored_models', 'tdt_score_mean', 'tdt_score_std', 'tdt_score_min',
            'tdt_score_max', 'tdt_score_range', 'tdt_score_mean_per_char',
            'tdt_score_mean_rank', 'tdt_score_mean_pct',
            'tdt_score_diff_from_best', 'tdt_score_mean_centered',
            'tdt_score_mean_zscore', 'tdt_score_mean_is_best',
            'tdt_score_mean_margin_to_second', 'tdt_score_best_vote_count',
            'tdt_score_best_vote_frac', 'tdt_score_best_vote_is_top',
            'tdt_score_best_vote_margin',
        ])
        if include_score_compare:
            df['tdt_ctc_score_gap'] = df['tdt_score_mean'] - df['ctc_score_mean']
            df['tdt_score_ctc_rank_gap'] = df['tdt_score_mean_rank'] - grp_ctc.rank(
                ascending=False, method='min', na_option='bottom')
            df['tdt_score_mean_vs_ctc_best_gap'] = df['tdt_score_mean'] - ctc_best
            new_cols.extend([
                'tdt_ctc_score_gap',
                'tdt_score_ctc_rank_gap',
                'tdt_score_mean_vs_ctc_best_gap',
            ])

    return df, [c for c in new_cols if c in df.columns]


def _nbest_pool_init():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    try:
        import torch
        torch.set_num_threads(1)
        if hasattr(torch, 'set_num_interop_threads'):
            torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
    except Exception:
        pass


def list_available_models(require_logprobs=False):
    """List models that have best_eval.csv with non-empty predictions."""
    models = []
    for d in sorted(WORKING_BASE.iterdir()):
        if not d.is_dir():
            continue
        fold_dir = d / str(FOLD)
        eval_csv = get_eval_csv(fold_dir)
        if not eval_csv.exists():
            continue
        if require_logprobs and not (fold_dir / 'ctc_logprobs.pt').exists():
            continue
        try:
            df = pd.read_csv(eval_csv)
            n_preds = df.pred.notna().sum()
            if n_preds > 100:
                metrics_csv = fold_dir / 'metrics.csv'
                score = None
                if metrics_csv.exists():
                    mdf = pd.read_csv(metrics_csv)
                    if 'score' in mdf.columns:
                        score = mdf['score'].min()
                models.append({
                    'name': d.name,
                    'n_preds': n_preds,
                    'score': score,
                    'has_logprobs': (fold_dir / 'ctc_logprobs.pt').exists(),
                })
        except Exception:
            pass
    return models


def auto_select_models(top_k=5, require_logprobs=False):
    """Auto-select top-K models by CER on dd source, excluding .eval/dual variants."""
    available = list_available_models(require_logprobs=require_logprobs)
    # Exclude known bad model families and .eval variants
    skip_patterns = ['.eval', '-dual', '-adapter']
    available = [m for m in available 
                 if not any(pat in m['name'] for pat in skip_patterns)]
    
    # Compute actual CER from best_eval.csv for ranking
    for m in available:
        try:
            eval_csv = get_eval_csv(get_model_dir(m['name']))
            df = pd.read_csv(eval_csv)
            df['pred'] = df['pred'].fillna('').astype(str)
            df['label'] = df['label'].fillna('').astype(str)
            if 'source' in df.columns:
                df = df[df['source'] == 'dd']
            df = df[df['label'].str.strip() != '']
            m['dd_cer'] = score_ipa_cer(df['label'].tolist(), df['pred'].tolist())
        except Exception:
            m['dd_cer'] = 999.0
    
    available = [m for m in available if m['dd_cer'] < 1.0]
    available.sort(key=lambda x: x['dd_cer'])
    model_names = [m['name'] for m in available[:top_k]]
    print(f'Auto-selected top-{top_k} models (by dd CER):')
    for m in available[:top_k]:
        lp_str = ' [logprobs]' if m.get('has_logprobs') else ''
        print(f'  {m["name"]}: dd_cer={m["dd_cer"]:.5f}{lp_str}')
    return model_names


def _parse_model_spec(spec, source='model spec'):
    token = str(spec).strip()
    assert token, f'Empty {source}'
    if ':' not in token:
        return token, None
    model_name, max_dur_str = token.rsplit(':', 1)
    model_name = model_name.strip()
    max_dur_str = max_dur_str.strip()
    assert model_name, f'Invalid {source}: missing model name in {spec!r}'
    assert max_dur_str, f'Invalid {source}: missing max_dur in {spec!r}'
    try:
        max_dur = float(max_dur_str)
    except ValueError as exc:
        raise AssertionError(f'Invalid {source}: max_dur must be float in {spec!r}') from exc
    assert max_dur > 0, f'Invalid {source}: max_dur must be > 0 in {spec!r}'
    return model_name, max_dur


def _parse_model_specs(specs, source='model specs'):
    model_names = []
    model_max_dur = {}
    for idx, raw_spec in enumerate(specs, start=1):
        model_name, max_dur = _parse_model_spec(raw_spec, source=f'{source}[{idx}]')
        model_names.append(model_name)
        if max_dur is not None:
            model_max_dur[model_name] = float(max_dur)
    return model_names, model_max_dur


def _load_models_from_file(file_path='models.txt', require_logprobs=False, verbose=True):
    path = Path(file_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return [], {}

    model_names = []
    model_max_dur = {}
    skipped = []
    for line_idx, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        model_spec = line.split('#', 1)[0].strip()
        if not model_spec:
            continue
        model_name, max_dur = _parse_model_spec(model_spec, source=f'{path}:{line_idx}')
        model_dir = get_model_dir(model_name)
        eval_csv = get_eval_csv(model_dir)
        has_logprobs = (model_dir / 'ctc_logprobs.pt').exists()
        if not eval_csv.exists():
            skipped.append(f'{model_name} (missing eval.csv)')
            continue
        if require_logprobs and not has_logprobs:
            skipped.append(f'{model_name} (missing ctc_logprobs.pt)')
            continue
        model_names.append(model_name)
        if max_dur is not None:
            model_max_dur[model_name] = float(max_dur)

    if verbose and model_names:
        print(f'Loaded {len(model_names)} models from {path}:')
        for model_name in model_names:
            max_dur = model_max_dur.get(model_name)
            if max_dur is None:
                print(f'  {model_name}')
            else:
                print(f'  {model_name} (max_dur={max_dur:.1f}s)')
    if verbose and skipped:
        print(f'Skipped {len(skipped)} entries from {path}:')
        for item in skipped:
            print(f'  {item}')
    return model_names, model_max_dur


def _mode_requires_ctc_logprobs_for_all_models(mode):
    return mode in ('logits_saved', 'prob_saved', 'max_saved', 'sweep_logits')


# ===========================================================================
#  Text-level MBR Ensemble
# ===========================================================================

def load_text_predictions(model_names):
    """Load best_eval.csv predictions from multiple models.
    Returns dict: utterance_id -> {model_name: pred_text}
    """
    all_preds = {}
    gold = {}
    meta = {}
    
    for mn in model_names:
        eval_csv = get_eval_csv(get_model_dir(mn))
        if not eval_csv.exists():
            print(f'WARNING: {eval_csv} not found, skipping {mn}')
            continue
        df = pd.read_csv(eval_csv)
        for _, row in df.iterrows():
            uid = row['utterance_id']
            pred = str(row['pred']) if pd.notna(row['pred']) else ''
            label = str(row['label']) if pd.notna(row['label']) else ''
            
            if uid not in all_preds:
                all_preds[uid] = {}
            all_preds[uid][mn] = pred
            
            if label and uid not in gold:
                gold[uid] = label
                meta[uid] = {
                    'source': row.get('source', ''),
                    'age_bucket': row.get('age_bucket', ''),
                }
    
    return all_preds, gold, meta


def _safe_cer(ref, hyp):
    """Compute CER using fast editdistance, handling empty strings gracefully."""
    r = normalize_ipa(ref).strip()
    h = normalize_ipa(hyp).strip()
    if not r:
        return 0.0 if not h else 1.0
    import editdistance
    return editdistance.eval(r, h) / len(r)


def mbr_select(candidates):
    """MBR: select candidate with minimum average CER to all others."""
    if len(candidates) == 1:
        return candidates[0], 0
    
    n = len(candidates)
    avg_cer = []
    for i in range(n):
        total_cer = sum(_safe_cer(candidates[j], candidates[i])
                        for j in range(n) if j != i)
        avg_cer.append(total_cer / (n - 1))
    
    best_idx = int(np.argmin(avg_cer))
    return candidates[best_idx], best_idx


_ROVER_GAP = '<eps>'


def _rover_backbone(columns):
    """Return current best symbol for each confusion-network column."""
    backbone = []
    for col in columns:
        symbols = [(ch, cnt) for ch, cnt in col.items() if ch != _ROVER_GAP]
        backbone.append(max(symbols, key=lambda x: (x[1], x[0]))[0] if symbols else _ROVER_GAP)
    return backbone


def _align_sequences(seq1, seq2):
    """Levenshtein alignment between 2 symbol sequences."""
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = 'del'
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = 'ins'

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            candidates = [
                (dp[i - 1][j - 1] + sub_cost, 'match'),
                (dp[i - 1][j] + 1, 'del'),
                (dp[i][j - 1] + 1, 'ins'),
            ]
            dp[i][j], bt[i][j] = min(candidates, key=lambda x: x[0])

    aligned = []
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == 'match':
            aligned.append((seq1[i - 1], seq2[j - 1]))
            i -= 1
            j -= 1
        elif op == 'del':
            aligned.append((seq1[i - 1], _ROVER_GAP))
            i -= 1
        else:
            aligned.append((_ROVER_GAP, seq2[j - 1]))
            j -= 1

    aligned.reverse()
    return aligned


def rover_fuse(candidates):
    """Simple char-level ROVER via progressive confusion-network alignment."""
    if not candidates:
        return ''
    if len(candidates) == 1:
        return candidates[0]

    columns = [{ch: 1} for ch in candidates[0]]
    n_seq = 1

    for cand in candidates[1:]:
        cand_chars = list(cand)
        backbone = _rover_backbone(columns)
        aligned = _align_sequences(backbone, cand_chars)
        new_columns = []
        col_idx = 0
        for ref_ch, hyp_ch in aligned:
            if ref_ch == _ROVER_GAP:
                new_col = {_ROVER_GAP: n_seq, hyp_ch: 1}
                new_columns.append(new_col)
            else:
                col = dict(columns[col_idx])
                col[hyp_ch] = col.get(hyp_ch, 0) + 1
                new_columns.append(col)
                col_idx += 1
        columns = new_columns
        n_seq += 1

    output = []
    for col in columns:
        symbols = [(ch, cnt) for ch, cnt in col.items() if ch != _ROVER_GAP]
        if not symbols:
            continue
        best_ch, best_cnt = max(symbols, key=lambda x: (x[1], x[0]))
        gap_cnt = col.get(_ROVER_GAP, 0)
        if best_cnt > gap_cnt:
            output.append(best_ch)

    return ''.join(output)


def _evaluate(ensemble_preds, gold, meta, label='Ensemble', verbose=True):
    """Evaluate predictions against gold labels. Returns dict with results.
    
    The 'overall_cer' uses the same weighted macro-average as eval.py:
      score = (score_dd + eval_ext_weight * score_ext) / (1 + eval_ext_weight)
    This avoids the micro-average being dominated by ext (which has ~12x more samples).
    """
    from gezi import FLAGS
    uids_with_labels = [uid for uid in gold if normalize_ipa(gold[uid]).strip()]
    targets = [gold[uid] for uid in uids_with_labels]
    preds = [ensemble_preds.get(uid, '') for uid in uids_with_labels]
    
    raw_cer = score_ipa_cer(targets, preds)
    
    source_results = {}
    for src in ['dd', 'ext']:
        src_uids = [uid for uid in uids_with_labels
                    if meta.get(uid, {}).get('source', '') == src]
        if src_uids:
            src_targets = [gold[uid] for uid in src_uids]
            src_preds = [ensemble_preds.get(uid, '') for uid in src_uids]
            src_cer = score_ipa_cer(src_targets, src_preds)
            source_results[src] = src_cer
    
    # Weighted macro-average matching eval.py: (dd + w*ext) / (1+w)
    if 'dd' in source_results and 'ext' in source_results:
        #w = getattr(FLAGS, 'eval_ext_weight', 1.0)
        w = 1.0
        # w = FLAGS.eval_ext_weight
        # w = 2
        overall_cer = (source_results['dd'] + w * source_results['ext']) / (1.0 + w)
    else:
        overall_cer = raw_cer
    
    if verbose:
        print(f'\n--- {label} Results ---')
        print(f'Overall CER: {overall_cer:.5f}  (n={len(targets)}, raw={raw_cer:.5f})')
        for src in ['dd', 'ext']:
            if src in source_results:
                src_n = len([uid for uid in uids_with_labels
                            if meta.get(uid, {}).get('source', '') == src])
                print(f'  score/{src}: {source_results[src]:.5f}  (n={src_n})')
    
    return {
        'overall_cer': overall_cer,
        'raw_cer': raw_cer,
        'source_results': source_results,
        'n_samples': len(targets),
    }


def text_ensemble(model_names, verbose=True, return_details=False):
    """Text-level MBR ensemble from existing best_eval.csv predictions."""
    all_preds, gold, meta = load_text_predictions(model_names)
    
    if verbose:
        print(f'\nText-level MBR ensemble with {len(model_names)} models:')
        for mn in model_names:
            print(f'  - {mn}')
        print(f'Total utterances: {len(all_preds)}')
        print(f'Utterances with gold labels: {len(gold)}')
    
    ensemble_preds = {}
    for uid in gold:
        candidates = [all_preds[uid][mn] for mn in model_names
                      if mn in all_preds.get(uid, {})]
        if not candidates:
            ensemble_preds[uid] = ''
        elif len(candidates) == 1:
            ensemble_preds[uid] = candidates[0]
        else:
            best_text, _ = mbr_select(candidates)
            ensemble_preds[uid] = best_text
    
    result = _evaluate(ensemble_preds, gold, meta, label='MBR Ensemble', verbose=verbose)
    
    # Individual model scores for comparison
    if verbose:
        _print_individual_model_scores(model_names)
    
    if return_details:
        return {
            'result': result,
            'predictions': ensemble_preds,
            'gold': gold,
            'meta': meta,
        }

    return result


def rover_ensemble(model_names, verbose=True, return_details=False):
    """Character-level ROVER from existing best_eval.csv predictions."""
    all_preds, gold, meta = load_text_predictions(model_names)

    if verbose:
        print(f'\nChar-level ROVER ensemble with {len(model_names)} models:')
        for mn in model_names:
            print(f'  - {mn}')
        print(f'Total utterances: {len(all_preds)}')
        print(f'Utterances with gold labels: {len(gold)}')

    ensemble_preds = {}
    for uid in gold:
        candidates = [all_preds[uid][mn] for mn in model_names
                      if mn in all_preds.get(uid, {})]
        ensemble_preds[uid] = rover_fuse(candidates)

    result = _evaluate(ensemble_preds, gold, meta, label='ROVER Ensemble', verbose=verbose)

    if verbose:
        _print_individual_model_scores(model_names)

    if return_details:
        return {
            'result': result,
            'predictions': ensemble_preds,
            'gold': gold,
            'meta': meta,
        }

    return result


# ===========================================================================
#  Logits/Prob-level Ensemble
# ===========================================================================

def save_logits_for_model(model_name):
    """Load model, run forward pass on eval data, save CTC log_probs per utterance.
    
    Uses model._last_ctc_log_probs cached by _ctc_decode() during forward().
    """
    import torch
    
    model_dir = get_model_dir(model_name)
    assert model_dir.exists(), f'Model dir not found: {model_dir}'
    best_pt = model_dir / 'model.pt'
    if not best_pt.exists():
        best_pt = model_dir / 'best.pt'
    assert best_pt.exists(), f'No model.pt or best.pt found in {model_dir}'
    
    # Import training infrastructure
    import gezi as gz
    from gezi import FLAGS
    import melt as mt
    from src import config
    from src.preprocess import preprocess

    uid_filter_env = os.environ.get('SAVE_LOGITS_UIDS', '').strip()
    uid_filter = None
    if uid_filter_env:
        uid_filter = {
            uid.strip() for uid in uid_filter_env.split(',') if uid.strip()
        }
    output_suffix = os.environ.get('SAVE_LOGITS_SUFFIX', '').strip()
    if output_suffix and not output_suffix.startswith('.'):
        output_suffix = f'.{output_suffix}'
    
    # Restore FLAGS from model
    gz.init_flags()
    config.init()
    gz.restore_configs(str(model_dir))
    FLAGS.mode = 'eval'
    FLAGS.work_mode = 'eval'
    FLAGS.distributed = False
    FLAGS.num_workers = 0
    FLAGS.persistent_workers = False
    FLAGS.batch_size = 8
    FLAGS.eval_batch_size = 8
    
    from src.dataset import Dataset, get_dl
    import importlib
    model_module = importlib.import_module(f'src.models.{FLAGS.model}')
    Model = model_module.Model
    model = Model()
    
    # Load weights
    gz.load_weights(model, str(best_pt), strict=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    
    def _pad_labels(samples):
        from torch.nn.utils.rnn import pad_sequence
        label_tensors = []
        for sample in samples:
            labels = sample.get('labels', [])
            if labels:
                label_tensors.append(torch.tensor(labels, dtype=torch.long))
            else:
                label_tensors.append(torch.tensor([-100], dtype=torch.long))
        return pad_sequence(label_tensors, batch_first=True, padding_value=-100)

    def _collate_manual(samples):
        first_input = samples[0]['input_features']
        first_tensor = torch.tensor(first_input, dtype=torch.float32)
        if first_tensor.ndim == 1:
            lengths = [len(sample['input_features']) for sample in samples]
            max_len = max(lengths)
            input_features = torch.zeros(len(samples), max_len, dtype=torch.float32)
            attention_mask = torch.zeros(len(samples), max_len, dtype=torch.long)
            for idx, sample in enumerate(samples):
                waveform = torch.tensor(sample['input_features'], dtype=torch.float32)
                cur_len = waveform.shape[0]
                input_features[idx, :cur_len] = waveform
                attention_mask[idx, :cur_len] = 1
        else:
            input_features = torch.stack(
                [torch.tensor(sample['input_features'], dtype=torch.float32) for sample in samples],
                dim=0,
            )
            attention_mask = None

        batch = {
            'input_features': input_features,
            'labels': _pad_labels(samples),
        }
        if attention_mask is not None:
            batch['attention_mask'] = attention_mask

        ids = [sample.get('id', '') for sample in samples]
        if any(ids):
            batch['id'] = ids

        label_texts = [sample.get('label_text', '') for sample in samples]
        if any(label_texts):
            batch['label_texts'] = label_texts

        weights = [sample.get('weight', 1.0) for sample in samples]
        if any(weight != 1.0 for weight in weights):
            batch['weight'] = torch.tensor(weights, dtype=torch.float32)

        return batch

    # Prepare eval-fold rows instead of full train split, but build them with the
    # inference/test dataloader path since get_dl only supports train/test modes.
    # tree reranker / saved ctc_logprobs are consumed against eval.csv rows.
    df = preprocess(mode='eval')
    if uid_filter is not None and 'id' in df.columns:
        before = len(df)
        df = df[df['id'].isin(uid_filter)].copy()
        print(f'Filtered eval rows by SAVE_LOGITS_UIDS: {before} -> {len(df)}')
    use_manual_batches = uid_filter is not None
    if use_manual_batches:
        test_ds = Dataset(df, mode='test')
        batch_size = int(getattr(FLAGS, 'eval_batch_size', 8) or 8)
        test_dl = []
        for start in range(0, len(test_ds), batch_size):
            samples = [test_ds[idx] for idx in range(start, min(start + batch_size, len(test_ds)))]
            samples = [sample for sample in samples if sample is not None]
            if samples:
                test_dl.append(_collate_manual(samples))
    else:
        test_dl = get_dl(mode='test', df=df)
    
    gz.set('do_generate', True)
    
    logprobs_dict = {}  # utterance_id -> numpy (T_i, V)
    word_logprobs_dict = {}  # utterance_id -> numpy (T_i, V_word) for word head
    aux_meta_preds = {}  # utterance_id -> aux scalar/vector predictions
    
    # Detect word head and aux-loss metadata heads
    has_aux_head = hasattr(model, 'word_ctc_head')
    has_aux_age = hasattr(model, 'aux_age_head')
    has_aux_domain = hasattr(model, 'aux_domain_head')
    has_aux_nchars = hasattr(model, 'aux_nchars_head')
    has_aux_nspaces = hasattr(model, 'aux_nspaces_head')
    aux_age_mode = getattr(model, '_aux_age_mode', None)
    if has_aux_age:
        print(f'  Aux age head detected: mode={aux_age_mode}')
    if has_aux_domain:
        print(f'  Aux domain head detected: DD=1, EXT=0')
    if has_aux_nchars:
        print('  Aux nchars head detected')
    if has_aux_nspaces:
        print('  Aux nspaces head detected')
    word_head_type = None  # 'pseudo_ipa', 'word_ctc_bpe', 'word_ctc'
    if has_aux_head:
        if getattr(model, '_pseudo_ipa_ctc', False):
            word_head_type = 'pseudo_ipa'
        elif getattr(model, '_word_ctc_bpe', False):
            word_head_type = 'word_ctc_bpe'
        else:
            word_head_type = 'word_ctc'
        print(f'  Word head detected: {word_head_type}')
    
    print(f'Saving logits for {model_name} ...')
    
    _cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    _amp_dtype = torch.bfloat16 if _cc >= (8, 0) else torch.float16
    
    import torch.nn.functional as F
    infer_total = len(test_dl)
    
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=_amp_dtype):
        for batch in _iter_with_progress(
                test_dl,
                total=infer_total,
                desc=f'  Infer {model_name}',
                verbose=True):
            input_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    input_batch[k] = v.to(device, non_blocking=True)
                else:
                    input_batch[k] = v
            
            # Forward pass — this triggers _ctc_decode which caches _last_ctc_log_probs
            res = model(input_batch)
            
            # Extract cached log_probs from _ctc_decode
            log_probs = getattr(model, '_last_ctc_log_probs', None)
            enc_len = getattr(model, '_last_enc_len', None)
            
            if log_probs is None:
                print(f'WARNING: _last_ctc_log_probs not set, model may not use CTC decode')
                continue
            
            # Compute word-head logprobs from cached encoder output
            word_log_probs = None
            if has_aux_head:
                enc_out = getattr(model, '_last_enc_out', None)
                if enc_out is not None:
                    word_logits = model.word_ctc_head(enc_out)
                    word_log_probs = F.log_softmax(word_logits.float(), dim=-1)
            
            # Extract age/domain predictions from forward result
            age_logits_batch = res.get('aux_age_logits', None)  # (B, K) or None
            domain_logits_batch = res.get('aux_domain_logits', None)  # (B,) or None
            nchars_pred_batch = res.get('aux_nchars_pred', None)  # (B,) or None
            nspaces_pred_batch = res.get('aux_nspaces_pred', None)  # (B,) or None
            
            batch_ids = batch.get('id', [])
            B = log_probs.shape[0]
            
            for i in range(B):
                uid = batch_ids[i] if i < len(batch_ids) else f'sample_{i}'
                if enc_len is not None:
                    actual_len = min(int(enc_len[i].item()), log_probs.shape[1])
                else:
                    actual_len = log_probs.shape[1]
                # Store only valid frames (no padding), float16 to save space
                logprobs_dict[uid] = log_probs[i, :actual_len].cpu().to(torch.float16).numpy()
                if word_log_probs is not None:
                    word_logprobs_dict[uid] = word_log_probs[i, :actual_len].cpu().to(torch.float16).numpy()
                # Store age/domain predictions
                meta_pred = {}
                if age_logits_batch is not None:
                    meta_pred['age_logits'] = age_logits_batch[i].cpu().float().numpy()
                if domain_logits_batch is not None:
                    meta_pred['domain_logit'] = float(domain_logits_batch[i].cpu())
                if nchars_pred_batch is not None:
                    meta_pred['nchars_pred'] = float(nchars_pred_batch[i].cpu())
                if nspaces_pred_batch is not None:
                    meta_pred['nspaces_pred'] = float(nspaces_pred_batch[i].cpu())
                if meta_pred:
                    aux_meta_preds[uid] = meta_pred
    
    # Save primary CTC logprobs
    output_path = model_dir / f'ctc_logprobs{output_suffix}.pt'
    torch.save(logprobs_dict, str(output_path))
    n = len(logprobs_dict)
    shapes = [v.shape for v in list(logprobs_dict.values())[:3]]
    size_mb = os.path.getsize(output_path) / 1e6
    print(f'Saved {n} utterance logprobs to {output_path} ({size_mb:.1f} MB)')
    print(f'Sample shapes: {shapes}')
    
    # Save word-head logprobs
    if word_logprobs_dict:
        word_output_path = model_dir / f'ctc_logprobs_word{output_suffix}.pt'
        word_meta = {'head_type': word_head_type}
        torch.save({'logprobs': word_logprobs_dict, 'meta': word_meta}, str(word_output_path))
        word_shapes = [v.shape for v in list(word_logprobs_dict.values())[:3]]
        word_size_mb = os.path.getsize(word_output_path) / 1e6
        print(f'Saved {len(word_logprobs_dict)} word ({word_head_type}) logprobs '
              f'to {word_output_path} ({word_size_mb:.1f} MB)')
        print(f'Word sample shapes: {word_shapes}')
    
    # Save age/domain meta predictions
    if aux_meta_preds:
        meta_pred_path = model_dir / f'aux_meta_preds{output_suffix}.pt'
        meta_info = {
            'age_mode': aux_age_mode,
            'has_age': has_aux_age,
            'has_domain': has_aux_domain,
            'has_nchars': has_aux_nchars,
            'has_nspaces': has_aux_nspaces,
        }
        torch.save({'preds': aux_meta_preds, 'meta': meta_info}, str(meta_pred_path))
        n_age = sum(1 for v in aux_meta_preds.values() if 'age_logits' in v)
        n_dom = sum(1 for v in aux_meta_preds.values() if 'domain_logit' in v)
        n_nchars = sum(1 for v in aux_meta_preds.values() if 'nchars_pred' in v)
        n_nspaces = sum(1 for v in aux_meta_preds.values() if 'nspaces_pred' in v)
        meta_size_mb = os.path.getsize(meta_pred_path) / 1e6
        print(f'Saved aux meta preds: {n_age} age, {n_dom} domain, '
              f'{n_nchars} nchars, {n_nspaces} nspaces '
              f'to {meta_pred_path} ({meta_size_mb:.1f} MB)')
    
    return logprobs_dict


def _greedy_ctc_decode_numpy(log_probs, blank_id=0, id_to_char=None):
    """Greedy CTC decode from numpy log_probs (T, V) -> text string."""
    token_ids = np.argmax(log_probs, axis=-1)  # (T,)
    
    # Collapse repeats + remove blank
    decoded = []
    prev = -1
    for t_id in token_ids:
        if t_id != prev:
            if t_id != blank_id:
                decoded.append(int(t_id))
            prev = t_id
    
    if id_to_char is not None:
        return ''.join(id_to_char.get(c, '') for c in decoded)
    return str(decoded)


def _beam_ctc_decode_numpy(log_probs, blank_id=0, id_to_char=None, beam_width=10):
    """Beam search CTC decode from numpy log_probs (T, V) -> text string.
    
    Critical for ensemble: averaged logprobs have smoothed peaks due to
    temporal misalignment between CTC models. Greedy decode loses characters
    because blank dominates at every frame. Beam search recovers them by
    considering alternative paths through the smoothed distribution.
    """
    if not _HAS_BEAM_SEARCH:
        # Fallback to greedy
        return _greedy_ctc_decode_numpy(log_probs, blank_id, id_to_char)
    
    lp = log_probs.astype(np.float32) if log_probs.dtype != np.float32 else log_probs
    ids = _prefix_beam_search_np(lp, blank_id, beam_width, id_to_char=id_to_char)
    
    if id_to_char is not None:
        return ''.join(id_to_char.get(c, '') for c in ids)
    return str(ids)


def logprob_ensemble(model_names, mode='logits', beam_width=10, temperature=1.0, verbose=True,
                     return_details=False):
    """Ensemble from saved CTC log_probs.
    
    mode='logits': average in log-prob space (geometric mean of probs).
    mode='prob':   average in prob space (arithmetic mean), then log.
    mode='max':    element-wise max of log-probs across models (keeps peaks).
    beam_width: beam search width for decode (default 10, 1=greedy).
    temperature: sharpening temperature applied AFTER averaging (default 1.0).
                Values < 1.0 amplify character peaks that survive averaging.
                E.g. 0.1 amplifies 10x. Useful for CTC ensemble.
    """
    import torch
    
    mode_labels = {'logits': 'Logits', 'prob': 'Prob', 'max': 'Max'}
    label = mode_labels.get(mode, mode)
    decode_label = f'beam={beam_width}' if beam_width > 1 else 'greedy'
    if temperature != 1.0:
        decode_label += f',temp={temperature}'
    if verbose:
        print(f'\n{label}-level ensemble ({decode_label}) from saved logprobs:')
        for mn in model_names:
            print(f'  - {mn}')
    
    # Load all saved logprobs
    all_logprobs = {}
    for mn in model_names:
        lp_path = get_model_dir(mn) / 'ctc_logprobs.pt'
        if not lp_path.exists():
            print(f'ERROR: {lp_path} not found. Run --mode=save_logits first.')
            return None
        all_logprobs[mn] = torch.load(str(lp_path), map_location='cpu', weights_only=False)
        if verbose:
            print(f'  Loaded {len(all_logprobs[mn])} utterances from {mn}')
    
    # Get utterance intersection
    uid_sets = [set(all_logprobs[mn].keys()) for mn in model_names]
    common_uids = set.intersection(*uid_sets)
    if verbose:
        print(f'Common utterances: {len(common_uids)}')
    
    blank_id = IPA_CTC_BLANK if IPA_CTC_BLANK is not None else 0
    id_to_char = IPA_ID_TO_CHAR
    
    # Diagnostic: check IPA vocab import
    if verbose:
        print(f'  IPA_CTC_BLANK={IPA_CTC_BLANK}, id_to_char has {len(id_to_char) if id_to_char else 0} chars')
        if id_to_char is None:
            print('  WARNING: IPA_ID_TO_CHAR is None! Decode will return token ID strings.')
    
    # Gold labels from first model's eval csv
    eval_csv = get_eval_csv(get_model_dir(model_names[0]))
    gold_df = pd.read_csv(eval_csv)
    gold = dict(zip(gold_df['utterance_id'], gold_df['label'].fillna('')))
    meta = {}
    for _, row in gold_df.iterrows():
        meta[row['utterance_id']] = {
            'source': row.get('source', ''),
            'age_bucket': row.get('age_bucket', ''),
        }
    
    # ---- Diagnostic: shape stats & logprob validation ----
    if verbose:
        for mn in model_names:
            shapes = [all_logprobs[mn][uid].shape for uid in list(common_uids)[:200]]
            t_lens = [s[0] for s in shapes]
            v_sizes = set(s[1] for s in shapes)
            sample_uid = list(common_uids)[0]
            lp_sample = all_logprobs[mn][sample_uid].astype(np.float32)
            lse = logsumexp(lp_sample, axis=-1)
            print(f'  {mn}: T range [{min(t_lens)},{max(t_lens)}], '
                  f'V={v_sizes}, logsumexp(frame) mean={lse.mean():.4f} '
                  f'std={lse.std():.4f} (expect ~0 for valid log_probs)')
        
        # Check time dimension alignment
        n_mismatch = 0
        sample_mismatches = []
        for uid in common_uids:
            ts = [all_logprobs[mn][uid].shape[0] for mn in model_names]
            if len(set(ts)) > 1:
                n_mismatch += 1
                if len(sample_mismatches) < 3:
                    sample_mismatches.append((uid, ts))
        if n_mismatch > 0:
            print(f'  WARNING: {n_mismatch}/{len(common_uids)} utterances have mismatched T!')
            for uid, ts in sample_mismatches:
                print(f'    {uid}: {dict(zip(model_names, ts))}')
        else:
            print(f'  Time dims: all {len(common_uids)} utterances aligned')
    
    # ---- Diagnostic: single-model greedy decode CER ----
    if verbose:
        print(f'\n--- Individual Model Greedy Decode from Saved Logprobs ---')
        for mn in model_names:
            single_preds = {}
            for uid in common_uids:
                lp = all_logprobs[mn][uid].astype(np.float32)
                single_preds[uid] = _greedy_ctc_decode_numpy(
                    lp, blank_id=blank_id, id_to_char=id_to_char)
            sr = _evaluate(single_preds, gold, meta,
                           label=f'{mn} (saved logprobs)', verbose=False)
            dd_cer = sr['source_results'].get('dd', float('nan'))
            ext_cer = sr['source_results'].get('ext', float('nan'))
            print(f'  {mn}: CER={sr["overall_cer"]:.5f} (dd={dd_cer:.5f}, ext={ext_cer:.5f})')
            # Show sample prediction vs gold
            sample_uid = list(common_uids)[0]
            print(f'    sample gold: {gold.get(sample_uid, "")[:80]}')
            print(f'    sample pred: {single_preds.get(sample_uid, "")[:80]}')
    
    # ---- Ensemble ----
    ensemble_preds = {}
    _diag_done = False
    
    for uid in common_uids:
        lps = [all_logprobs[mn][uid].astype(np.float32) for mn in model_names]
        min_t = min(lp.shape[0] for lp in lps)
        
        if mode == 'logits':
            # Average in log-space (geometric mean)
            avg_lp = np.zeros_like(lps[0][:min_t])
            for lp in lps:
                avg_lp += lp[:min_t]
            avg_lp /= len(lps)
        elif mode == 'max':
            # Element-wise max of log-probs (preserves character peaks)
            stacked = np.stack([lp[:min_t] for lp in lps], axis=0)  # (N, T, V)
            avg_lp = np.max(stacked, axis=0)  # (T, V)
            # Re-normalize to valid log-probs
            avg_lp = avg_lp - logsumexp(avg_lp, axis=-1, keepdims=True)
        else:
            # Average in prob-space (arithmetic mean)
            # log(mean(exp(log_p))) = logsumexp(log_ps) - log(N)
            stacked = np.stack([lp[:min_t] for lp in lps], axis=0)  # (N, T, V)
            avg_lp = logsumexp(stacked, axis=0) - np.log(len(lps))  # (T, V)
        
        # Temperature sharpening: divide logits by temp, re-normalize
        if temperature != 1.0 and temperature > 0:
            avg_lp = avg_lp / temperature
            avg_lp = avg_lp - logsumexp(avg_lp, axis=-1, keepdims=True)
        
        # Use beam search for ensemble (greedy fails due to temporal misalignment)
        if beam_width > 1:
            text = _beam_ctc_decode_numpy(avg_lp, blank_id=blank_id, 
                                          id_to_char=id_to_char, beam_width=beam_width)
        else:
            text = _greedy_ctc_decode_numpy(avg_lp, blank_id=blank_id, id_to_char=id_to_char)
        ensemble_preds[uid] = text
        
        # ---- Per-sample diagnostic for first utterance ----
        if verbose and not _diag_done:
            _diag_done = True
            print(f'\n--- Per-frame Diagnostic for uid={uid} (T={min_t}) ---')
            # Individual model argmax
            for i, mn in enumerate(model_names):
                ids_i = np.argmax(lps[i][:min_t], axis=-1)
                text_i = _greedy_ctc_decode_numpy(lps[i][:min_t], blank_id=blank_id, id_to_char=id_to_char)
                n_blank = (ids_i == blank_id).sum()
                print(f'  {mn}: {n_blank}/{min_t} blank frames, greedy="{text_i[:60]}"')
            # Ensemble greedy vs beam
            ens_ids = np.argmax(avg_lp, axis=-1)
            n_blank_ens = (ens_ids == blank_id).sum()
            greedy_text = _greedy_ctc_decode_numpy(avg_lp, blank_id=blank_id, id_to_char=id_to_char)
            beam_text = _beam_ctc_decode_numpy(avg_lp, blank_id=blank_id, 
                                               id_to_char=id_to_char, beam_width=beam_width) if beam_width > 1 else greedy_text
            print(f'  ensemble greedy: {n_blank_ens}/{min_t} blank frames, decoded="{greedy_text[:60]}"')
            print(f'  ensemble beam={beam_width}: decoded="{beam_text[:60]}"')
            print(f'  gold: "{gold.get(uid, "")[:60]}"')
            # Timing mismatch stats
            ids_all = [np.argmax(lp[:min_t], axis=-1) for lp in lps]
            ens_ids_all = np.argmax(avg_lp, axis=-1)
            agree = sum(1 for t in range(min_t) if all(ids_all[m][t] == ids_all[0][t] for m in range(len(model_names))))
            changed = sum(1 for t in range(min_t) if ens_ids_all[t] != ids_all[0][t])
            print(f'  Models agree on {agree}/{min_t} frames, ensemble differs from model0 on {changed}/{min_t} frames')
    
    result = _evaluate(ensemble_preds, gold, meta,
                       label=f'{label} Ensemble ({decode_label})', verbose=verbose)
    
    # ---- Also try greedy for comparison if using beam ----
    if verbose and beam_width > 1:
        print(f'\n--- Greedy decode comparison (same averaged logprobs, temp={temperature}) ---')
        greedy_preds = {}
        for uid in common_uids:
            lps = [all_logprobs[mn][uid].astype(np.float32) for mn in model_names]
            min_t = min(lp.shape[0] for lp in lps)
            if mode == 'logits':
                avg_lp = np.zeros_like(lps[0][:min_t])
                for lp in lps:
                    avg_lp += lp[:min_t]
                avg_lp /= len(lps)
            elif mode == 'max':
                stacked = np.stack([lp[:min_t] for lp in lps], axis=0)
                avg_lp = np.max(stacked, axis=0)
                avg_lp = avg_lp - logsumexp(avg_lp, axis=-1, keepdims=True)
            else:
                stacked = np.stack([lp[:min_t] for lp in lps], axis=0)
                avg_lp = logsumexp(stacked, axis=0) - np.log(len(lps))
            if temperature != 1.0 and temperature > 0:
                avg_lp = avg_lp / temperature
                avg_lp = avg_lp - logsumexp(avg_lp, axis=-1, keepdims=True)
            greedy_preds[uid] = _greedy_ctc_decode_numpy(avg_lp, blank_id=blank_id, id_to_char=id_to_char)
        greedy_result = _evaluate(greedy_preds, gold, meta,
                                  label=f'{label} Ensemble (greedy,temp={temperature})', verbose=True)
    
    # ---- Also try prob-level ensemble for comparison (only when mode=logits) ----
    if verbose and mode == 'logits':
        print(f'\n--- Prob-level ensemble ({decode_label}) for comparison ---')
        prob_preds = {}
        for uid in common_uids:
            lps = [all_logprobs[mn][uid].astype(np.float32) for mn in model_names]
            min_t = min(lp.shape[0] for lp in lps)
            stacked = np.stack([lp[:min_t] for lp in lps], axis=0)
            avg_lp_prob = logsumexp(stacked, axis=0) - np.log(len(lps))
            if temperature != 1.0 and temperature > 0:
                avg_lp_prob = avg_lp_prob / temperature
                avg_lp_prob = avg_lp_prob - logsumexp(avg_lp_prob, axis=-1, keepdims=True)
            if beam_width > 1:
                prob_preds[uid] = _beam_ctc_decode_numpy(avg_lp_prob, blank_id=blank_id,
                                                          id_to_char=id_to_char, beam_width=beam_width)
            else:
                prob_preds[uid] = _greedy_ctc_decode_numpy(avg_lp_prob, blank_id=blank_id, id_to_char=id_to_char)
        prob_result = _evaluate(prob_preds, gold, meta,
                                label=f'Prob Ensemble ({decode_label})', verbose=True)
    
    if return_details:
        return {
            'result': result,
            'predictions': ensemble_preds,
            'gold': gold,
            'meta': meta,
        }

    return result


# ===========================================================================
#  Tree Reranker Ensemble
# ===========================================================================

def _char_cer(ref, hyp):
    """Character-level edit distance ratio (CER for a single pair)."""
    ref = normalize_ipa(ref).strip()
    hyp = normalize_ipa(hyp).strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    import editdistance
    return editdistance.eval(ref, hyp) / len(ref)


def _ctc_force_align(log_probs, token_ids, blank=0):
    """Viterbi CTC force alignment: find best frame-to-token assignment.

    Returns per-token info: (token_confidence, duration_in_frames) for each token,
    plus the blank_frame_ratio over the whole sequence.

    Args:
        log_probs: (T, V) numpy float32 log-probabilities.
        token_ids: list[int], label sequence (no blanks).
        blank: blank token id.

    Returns:
        dict with keys:
            'token_confidences': list[float] - per-token average log-prob at aligned frames
            'token_durations': list[int] - frames assigned to each token
            'blank_frame_ratio': float - fraction of frames assigned to blank
            'frame_assignments': list[int] - per-frame label (blank or token index)
    """
    T, V = log_probs.shape
    L = len(token_ids)

    if L == 0:
        return {
            'token_confidences': [],
            'token_durations': [],
            'blank_frame_ratio': 1.0,
            'frame_assignments': [blank] * T,
        }

    # Build CTC label sequence: b t0 b t1 b ... tL-1 b  (length 2L+1)
    S = 2 * L + 1
    labels = [blank] * S
    for i, tid in enumerate(token_ids):
        labels[2 * i + 1] = tid

    # Viterbi DP: dp[t][s] = best log-prob ending at time t, state s
    NEG_INF = -1e30
    dp = np.full((T, S), NEG_INF, dtype=np.float64)
    bt = np.full((T, S), -1, dtype=np.int32)  # backtrack

    # Init t=0
    dp[0, 0] = log_probs[0, labels[0]]
    if S > 1:
        dp[0, 1] = log_probs[0, labels[1]]

    for t in range(1, T):
        for s in range(S):
            # Must be reachable: s <= 2*t+1 and s >= S - 2*(T-t)
            lbl = labels[s]
            emit = float(log_probs[t, lbl])

            # Possible previous states
            best_prev = dp[t - 1, s]
            best_s = s
            if s >= 1 and dp[t - 1, s - 1] > best_prev:
                best_prev = dp[t - 1, s - 1]
                best_s = s - 1
            if s >= 2 and labels[s] != blank and labels[s] != labels[s - 2]:
                if dp[t - 1, s - 2] > best_prev:
                    best_prev = dp[t - 1, s - 2]
                    best_s = s - 2

            if best_prev > NEG_INF:
                dp[t, s] = best_prev + emit
                bt[t, s] = best_s

    # Backtrack: find best final state (must be last blank or last token)
    if dp[T - 1, S - 1] >= dp[T - 1, S - 2]:
        s = S - 1
    else:
        s = S - 2

    path = [s]
    for t in range(T - 1, 0, -1):
        s = bt[t, s]
        path.append(s)
    path.reverse()

    # Extract per-token stats
    frame_assignments = [labels[s] for s in path]
    n_blank = sum(1 for f in frame_assignments if f == blank)
    blank_ratio = n_blank / T

    # Group consecutive frames per token
    token_conf = [[] for _ in range(L)]
    token_dur = [0] * L
    for t, s in enumerate(path):
        if s % 2 == 1:  # token state
            tok_idx = s // 2
            token_conf[tok_idx].append(float(log_probs[t, labels[s]]))
            token_dur[tok_idx] += 1

    # Average confidence per token
    avg_conf = []
    for i in range(L):
        if token_conf[i]:
            avg_conf.append(float(np.mean(token_conf[i])))
        else:
            avg_conf.append(NEG_INF)

    return {
        'token_confidences': avg_conf,
        'token_durations': token_dur,
        'blank_frame_ratio': blank_ratio,
        'frame_assignments': frame_assignments,
    }


# ---- Multiprocessing support for parallel feature building ----
_MP_BUILD = {}  # Shared state for workers (set before fork, accessed via COW)

def _build_rows_for_uid(uid):
    """Build all feature rows for one utterance. Used by both sequential and parallel paths."""
    S = _MP_BUILD
    model_names = S['model_names']
    all_logprobs = S['all_logprobs']
    blank_id = S['blank_id']
    beam_width = S['beam_width']
    nbest = S['nbest']
    id_to_char = S['id_to_char']
    char_to_id = S['char_to_id']
    lm = S['lm']
    no_lm_feats = S['no_lm_feats']
    gold = S['gold']
    meta = S['meta']
    feat_text = S['feat_text']
    feat_ipa = S['feat_ipa']
    feat_ctc_stats = S['feat_ctc_stats']
    feat_audio = S['feat_audio']
    feat_consensus = S['feat_consensus']
    feat_group_ext = S['feat_group_ext']
    feat_align = S['feat_align']
    feat_logprob_proxy = S['feat_logprob_proxy']
    feat_word = S['feat_word']
    feat_aux_meta = S['feat_aux_meta']
    all_word_logprobs = S['all_word_logprobs']
    word_head_types = S['word_head_types']
    all_aux_meta = S['all_aux_meta']
    aux_meta_info = S['aux_meta_info']
    _word_id_to_char = S['_word_id_to_char']
    _word_blank_id = S['_word_blank_id']
    _ipa_convert = S['_ipa_convert']

    # Step 1: Collect N-best from each model, with per-model ranks
    per_model_hyps = {}
    candidate_set = set()
    for mn in model_names:
        lp = all_logprobs[mn][uid].astype(np.float32)
        hyps = prefix_beam_search_nbest(
            lp, blank_id, beam_width, nbest=nbest, id_to_char=id_to_char)
        per_model_hyps[mn] = hyps
        for _score, text in hyps:
            candidate_set.add(text)

    candidates = list(candidate_set)
    if not candidates:
        return []

    # Step 2: CTC scores for all candidates under all models (batched)
    all_token_ids = []
    for cand_text in candidates:
        if char_to_id:
            token_ids = [char_to_id[ch] for ch in cand_text if ch in char_to_id]
        else:
            token_ids = []
        all_token_ids.append(token_ids)

    ctc_scores = {}
    n_frames_per_model = {}
    for mn in model_names:
        lp = all_logprobs[mn][uid].astype(np.float32)
        n_frames_per_model[mn] = lp.shape[0]
        ctc_scores[mn] = ctc_force_score_batch(lp, all_token_ids, blank=blank_id)

    # CTC force alignment + logprob proxy need ref model's logprobs
    if feat_align or feat_logprob_proxy:
        ref_mn = model_names[0]
        ref_lp = all_logprobs[ref_mn][uid].astype(np.float32)

    if feat_align:
        align_results = []
        for ci_tmp, tids in enumerate(all_token_ids):
            align_results.append(_ctc_force_align(ref_lp, tids, blank=blank_id))

    # Per-model beam rank for each candidate
    beam_ranks = {}
    best_per_model = {}
    for mn in model_names:
        hyps = per_model_hyps[mn]
        ranks = {}
        for rank, (_s, text) in enumerate(hyps):
            ranks[text] = rank
        beam_ranks[mn] = ranks
        best_per_model[mn] = hyps[0][1] if hyps else ''

    # LM scores
    if not no_lm_feats:
        lm_scores = []
        if lm is not None:
            for cand_text in candidates:
                lm_scores.append(_score_lm_candidate(lm, cand_text))
        else:
            lm_scores = [0.0] * len(candidates)

    # Auxiliary head CTC scores for pseudo_ipa models
    word_ctc_scores = {}
    word_greedy_texts = {}
    if feat_word:
        for mn in all_word_logprobs:
            if uid not in all_word_logprobs[mn]:
                continue
            ht = word_head_types[mn]
            if ht == 'pseudo_ipa':
                word_lp = all_word_logprobs[mn][uid].astype(np.float32)
                word_ctc_scores[mn] = ctc_force_score_batch(
                    word_lp, all_token_ids, blank=blank_id)
            elif ht in ('word_ctc', 'word_ctc_bpe'):
                word_lp = all_word_logprobs[mn][uid].astype(np.float32)
                if ht == 'word_ctc' and _word_id_to_char is not None:
                    word_greedy_texts[mn] = _greedy_ctc_decode_numpy(
                        word_lp, blank_id=_word_blank_id, id_to_char=_word_id_to_char)

    # Audio frame count
    n_frames = max(n_frames_per_model.values())

    # ---- Logprob proxy features (utterance-level, computed once per uid) ----
    if feat_logprob_proxy:
        _lp_probs = np.exp(ref_lp)
        _frame_entropy = -np.sum(_lp_probs * ref_lp, axis=1)
        _utt_entropy_mean = float(np.mean(_frame_entropy))
        _utt_entropy_std = float(np.std(_frame_entropy))
        _utt_entropy_max = float(np.max(_frame_entropy))
        _utt_blank_prob_mean = float(np.mean(_lp_probs[:, blank_id]))
        _utt_top1_prob_mean = float(np.mean(np.max(_lp_probs, axis=1)))
        del _lp_probs

    gold_text = gold.get(uid, '')
    source = meta.get(uid, {}).get('source', '')
    child_id = meta.get(uid, {}).get('child_id', '')
    age_bucket = meta.get(uid, {}).get('age_bucket', '')

    # Build feature row for each candidate
    import editdistance
    rows = []
    for ci, cand_text in enumerate(candidates):
        row = {
            'uid': uid,
            'candidate_text': cand_text,
            'source': source,
            'child_id': child_id,
            'age_bucket': age_bucket,
        }

        # Per-model CTC scores
        scores_arr = []
        for mn in model_names:
            sc = ctc_scores[mn][ci]
            row[f'ctc_score_{mn}'] = sc
            scores_arr.append(sc)

        scores_arr = np.array(scores_arr)
        row['ctc_score_mean'] = float(np.mean(scores_arr))
        row['ctc_score_std'] = float(np.std(scores_arr))
        row['ctc_score_min'] = float(np.min(scores_arr))
        row['ctc_score_max'] = float(np.max(scores_arr))
        row['ctc_score_range'] = float(np.max(scores_arr) - np.min(scores_arr))

        # Text length features
        text_len = len(cand_text)
        row['text_len'] = text_len
        row['n_frames'] = n_frames
        row['char_per_frame'] = text_len / max(n_frames, 1)
        # Utterance-level audio duration (original, not CTC-derived)
        _audio_dur = meta.get(uid, {}).get('audio_duration_sec', 0)
        row['audio_duration_sec'] = float(_audio_dur) if _audio_dur != '' else 0.0

        # Beam rank features
        n_models_has = 0
        for mn in model_names:
            rank = beam_ranks[mn].get(cand_text, -1)
            row[f'beam_rank_{mn}'] = rank
            if rank >= 0:
                n_models_has += 1
        row['n_models_has'] = n_models_has

        # Edit distance to each model's 1-best
        for mn in model_names:
            best_text = best_per_model[mn]
            row[f'edit_dist_to_best_{mn}'] = editdistance.eval(cand_text, best_text)

        # LM score
        if not no_lm_feats:
            row['lm_score'] = lm_scores[ci]
            row['lm_score_per_char'] = lm_scores[ci] / max(text_len, 1)
            row['lm_score_per_word'] = lm_scores[ci] / max(n_spaces + 1, 1)

        # ---- Text analysis features ----
        n_spaces = cand_text.count(' ')
        if feat_text:
            row['n_spaces'] = n_spaces
            row['n_words'] = n_spaces + 1
            words = cand_text.split(' ') if cand_text else ['']
            word_lens = [len(w) for w in words]
            row['avg_word_len'] = text_len / max(n_spaces + 1, 1)
            row['max_word_len'] = max(word_lens) if word_lens else 0
            row['min_word_len'] = min(word_lens) if word_lens else 0
            row['n_unique_chars'] = len(set(cand_text))
            if text_len > 0:
                from collections import Counter as _Counter
                _ch_counts = _Counter(cand_text)
                _probs = np.array(list(_ch_counts.values()), dtype=np.float64) / text_len
                row['char_entropy'] = float(-np.sum(_probs * np.log(_probs + 1e-12)))
            else:
                row['char_entropy'] = 0.0
            max_repeat = 1
            cur_repeat = 1
            for k in range(1, text_len):
                if cand_text[k] == cand_text[k - 1]:
                    cur_repeat += 1
                    if cur_repeat > max_repeat:
                        max_repeat = cur_repeat
                else:
                    cur_repeat = 1
            row['max_char_repeat'] = max_repeat if text_len > 0 else 0
            row['unique_char_ratio'] = len(set(cand_text)) / max(text_len, 1)
            n_repeats = sum(1 for k in range(1, text_len) if cand_text[k] == cand_text[k - 1])
            row['repeat_char_ratio'] = n_repeats / max(text_len - 1, 1)

        # ---- IPA phonetic features ----
        if feat_ipa:
            _IPA_VOWELS = set('eiouɑæɐɔəɚɛɪʊʌ')
            _IPA_CONSONANTS = set('bcdfghjklmnprstvwxzçðŋɟɫɬɹɾʁʃʒʔʝθχʧʤ')
            n_vowels = sum(1 for ch in cand_text if ch in _IPA_VOWELS)
            n_consonants = sum(1 for ch in cand_text if ch in _IPA_CONSONANTS)
            row['n_vowels'] = n_vowels
            row['n_consonants'] = n_consonants
            row['vowel_ratio'] = n_vowels / max(text_len, 1)
            row['consonant_ratio'] = n_consonants / max(text_len, 1)
            row['vc_ratio'] = n_vowels / max(n_consonants, 1)
            row['n_length_marks'] = cand_text.count('ː')

        # ---- CTC distribution features ----
        if feat_ctc_stats:
            row['ctc_score_median'] = float(np.median(scores_arr))
            abs_mean = abs(float(np.mean(scores_arr)))
            row['ctc_score_cv'] = float(np.std(scores_arr)) / max(abs_mean, 1e-8)
            if len(scores_arr) >= 3:
                from scipy.stats import skew as _skew, kurtosis as _kurtosis
                row['ctc_score_skew'] = float(_skew(scores_arr))
                row['ctc_score_kurtosis'] = float(_kurtosis(scores_arr))
            else:
                row['ctc_score_skew'] = 0.0
                row['ctc_score_kurtosis'] = 0.0
            q75, q25 = np.percentile(scores_arr, [75, 25])
            row['ctc_score_iqr'] = float(q75 - q25)

        # ---- CTC alignment features ----
        if feat_align:
            aln = align_results[ci]
            row['blank_frame_ratio'] = aln['blank_frame_ratio']
            tok_conf = aln['token_confidences']
            tok_dur = aln['token_durations']
            if tok_conf:
                row['avg_frame_confidence'] = float(np.mean(tok_conf))
                row['min_phoneme_confidence'] = float(np.min(tok_conf))
                row['max_phoneme_confidence'] = float(np.max(tok_conf))
                row['std_phoneme_confidence'] = float(np.std(tok_conf))
            else:
                row['avg_frame_confidence'] = -30.0
                row['min_phoneme_confidence'] = -30.0
                row['max_phoneme_confidence'] = -30.0
                row['std_phoneme_confidence'] = 0.0
            if tok_dur:
                dur_arr = np.array(tok_dur, dtype=np.float64)
                row['phoneme_dur_mean'] = float(np.mean(dur_arr))
                row['phoneme_dur_std'] = float(np.std(dur_arr))
                row['phoneme_dur_min'] = float(np.min(dur_arr))
                row['phoneme_dur_max'] = float(np.max(dur_arr))
                dur_z = (dur_arr - np.mean(dur_arr)) / max(float(np.std(dur_arr)), 1e-8)
                row['phoneme_dur_zscore_max'] = float(np.max(np.abs(dur_z)))
                row['single_frame_phoneme_ratio'] = float(np.sum(dur_arr <= 1)) / max(len(dur_arr), 1)
            else:
                row['phoneme_dur_mean'] = 0.0
                row['phoneme_dur_std'] = 0.0
                row['phoneme_dur_min'] = 0.0
                row['phoneme_dur_max'] = 0.0
                row['phoneme_dur_zscore_max'] = 0.0
                row['single_frame_phoneme_ratio'] = 0.0

        # ---- Logprob proxy features ----
        if feat_logprob_proxy:
            row['entropy_mean'] = _utt_entropy_mean
            row['entropy_std'] = _utt_entropy_std
            row['entropy_max'] = _utt_entropy_max
            row['blank_prob_mean'] = _utt_blank_prob_mean
            row['top1_prob_mean'] = _utt_top1_prob_mean
            row['model_ctc_std'] = float(np.std(scores_arr))
            _ranks_for_ent = []
            for mn in model_names:
                mn_scores = ctc_scores[mn]
                _r = sorted(range(len(mn_scores)), key=lambda x: mn_scores[x], reverse=True)
                _ranks_for_ent.append(_r.index(ci))
            _ranks_arr = np.array(_ranks_for_ent, dtype=np.float64)
            row['model_rank_std'] = float(np.std(_ranks_arr))
            _model_best_idx = int(np.argmax(scores_arr))
            for mi, mn in enumerate(model_names):
                row[f'is_best_model_{mn}'] = 1 if mi == _model_best_idx else 0

        # ---- Audio / speaking rate features ----
        if feat_audio:
            duration_sec = n_frames * 0.04
            row['duration_sec'] = duration_sec
            row['chars_per_sec'] = text_len / max(duration_sec, 0.01)
            row['words_per_sec'] = (n_spaces + 1) / max(duration_sec, 0.01)

        # ---- Auxiliary head features ----
        if feat_word:
            import editdistance as _ed_aux
            word_ipa_scores = []
            for mn in word_ctc_scores:
                sc = word_ctc_scores[mn][ci]
                row[f'word_ctc_score_{mn}'] = sc
                word_ipa_scores.append(sc)
                row[f'word_score_diff_{mn}'] = ctc_scores[mn][ci] - sc
            if word_ipa_scores:
                row['word_ctc_score_mean'] = float(np.mean(word_ipa_scores))
                row['word_ctc_score_std'] = float(np.std(word_ipa_scores))
                row['word_primary_diff_mean'] = row['ctc_score_mean'] - row['word_ctc_score_mean']

            for mn in word_greedy_texts:
                word_text = word_greedy_texts[mn]
                row[f'word_edit_dist_raw_{mn}'] = _ed_aux.eval(cand_text, word_text)
                row[f'word_edit_dist_norm_{mn}'] = _ed_aux.eval(cand_text, word_text) / max(text_len, 1)
                if _ipa_convert is not None and word_text.strip():
                    pseudo_ipa = _ipa_convert(word_text)
                    pseudo_ipa = pseudo_ipa.replace('*', '')
                    row[f'word_edit_dist_ipa_{mn}'] = _ed_aux.eval(cand_text, pseudo_ipa)
                    row[f'word_edit_dist_ipa_norm_{mn}'] = _ed_aux.eval(cand_text, pseudo_ipa) / max(text_len, 1)

        # ---- Age/domain meta prediction features (utterance-level) ----
        if feat_aux_meta and all_aux_meta:
            age_scores = []
            domain_probs = []
            ordinal_age_probs = {
                '5plus': [],
                '8plus': [],
                '12plus': [],
            }
            class_age_probs = {}
            for mn in all_aux_meta:
                if uid not in all_aux_meta[mn]:
                    continue
                pred = all_aux_meta[mn][uid]
                mi = aux_meta_info[mn]
                if mi['has_age'] and 'age_logits' in pred:
                    age_logits = pred['age_logits']
                    if mi['age_mode'] == 'ordinal':
                        age_probs = 1.0 / (1.0 + np.exp(-age_logits))
                        row[f'aux_age_prob_5plus_{mn}'] = float(age_probs[0])
                        row[f'aux_age_prob_8plus_{mn}'] = float(age_probs[1])
                        row[f'aux_age_prob_12plus_{mn}'] = float(age_probs[2])
                        ordinal_age_probs['5plus'].append(float(age_probs[0]))
                        ordinal_age_probs['8plus'].append(float(age_probs[1]))
                        ordinal_age_probs['12plus'].append(float(age_probs[2]))
                        age_score = float(np.mean(age_probs))
                    elif mi['age_mode'] == 'classify':
                        exp_l = np.exp(age_logits - np.max(age_logits))
                        probs = exp_l / exp_l.sum()
                        age_score = float(np.dot(probs, np.arange(len(probs))))
                        for k in range(len(probs)):
                            row[f'aux_age_class{k}_prob_{mn}'] = float(probs[k])
                            class_age_probs.setdefault(k, []).append(float(probs[k]))
                    else:
                        age_score = float(age_logits)
                    row[f'aux_age_score_{mn}'] = age_score
                    age_scores.append(age_score)
                if mi['has_domain'] and 'domain_logit' in pred:
                    domain_logit = pred['domain_logit']
                    domain_prob = 1.0 / (1.0 + np.exp(-domain_logit))
                    row[f'aux_domain_prob_dd_{mn}'] = float(domain_prob)
                    domain_probs.append(float(domain_prob))
            if age_scores:
                row['aux_age_score_mean'] = float(np.mean(age_scores))
            for suffix, values in ordinal_age_probs.items():
                if values:
                    row[f'aux_age_prob_{suffix}_mean'] = float(np.mean(values))
            for k, values in sorted(class_age_probs.items()):
                if values:
                    row[f'aux_age_class{k}_prob_mean'] = float(np.mean(values))
            if domain_probs:
                row['aux_domain_prob_dd_mean'] = float(np.mean(domain_probs))

        # ---- Beam rank aggregate features ----
        valid_ranks = [beam_ranks[mn].get(cand_text, -1) for mn in model_names]
        valid_ranks_pos = [r for r in valid_ranks if r >= 0]
        row['beam_rank_mean'] = np.mean(valid_ranks_pos) if valid_ranks_pos else nbest
        row['beam_rank_min'] = min(valid_ranks_pos) if valid_ranks_pos else nbest
        row['beam_rank_max'] = max(valid_ranks_pos) if valid_ranks_pos else nbest
        if feat_consensus:
            row['n_models_in_top3'] = sum(1 for r in valid_ranks if 0 <= r < 3)

        # ---- Pairwise edit distance / consensus ----
        if feat_consensus:
            pairwise_sum = 0
            n_pairs = 0
            for oi, other_text in enumerate(candidates):
                if oi != ci:
                    pairwise_sum += editdistance.eval(cand_text, other_text)
                    n_pairs += 1
            row['mean_pairwise_edit_dist'] = pairwise_sum / max(n_pairs, 1)
            mean_cer_to_others = 0.0
            if n_pairs > 0 and text_len > 0:
                mean_cer_to_others = pairwise_sum / (n_pairs * text_len)
            row['consensus_score'] = 1.0 - min(mean_cer_to_others, 1.0)
            row['n_exact_best'] = sum(1 for mn in model_names if best_per_model[mn] == cand_text)

        # Target: CER of this candidate vs gold
        row['target_cer'] = _char_cer(gold_text, cand_text)

        rows.append(row)

    return rows


def build_reranker_dataset(model_names, nbest=10, beam_width=10,
                           lm=None, verbose=True,
                           feat_text=False, feat_ipa=False,
                           feat_ctc_stats=False, feat_audio=False,
                           feat_consensus=False, feat_group_ext=False,
                           feat_align=False, feat_logprob_proxy=False,
                           feat_word=False, feat_aux=None, feat_aux_meta=False,
                           no_lm_feats=False, n_workers=0):
    """Build a DataFrame of (uid, candidate, features, cer_target) for tree reranker.

    Args:
        feat_text..feat_logprob_proxy: feature group flags (False=skip)
        feat_word: if True, load word-head CTC logprobs and add features
        feat_aux_meta: if True, load age/domain predictions as utterance-level features
        no_lm_feats: if True, skip LM score features entirely

    Returns:
        df: pandas DataFrame with columns:
            uid, candidate_text, target_cer, source, + feature columns
        feat_cols: list of feature column names
    """
    import torch

    feat_word = feat_word or bool(feat_aux)

    assert _HAS_BEAM_SEARCH, 'CTC beam search not available'

    # Load all saved logprobs
    all_logprobs = {}
    for mn in model_names:
        lp_path = get_model_dir(mn) / 'ctc_logprobs.pt'
        assert lp_path.exists(), f'{lp_path} not found'
        all_logprobs[mn] = torch.load(str(lp_path), map_location='cpu', weights_only=False)
        if verbose:
            print(f'  Loaded {len(all_logprobs[mn])} utterances from {mn}')

    # Load word-head logprobs (pseudo_ipa / word_ctc / word_ctc_bpe)
    all_word_logprobs = {}  # mn -> {uid: numpy(T, V_word)}
    word_head_types = {}    # mn -> 'pseudo_ipa' | 'word_ctc' | 'word_ctc_bpe'
    if feat_word:
        for mn in model_names:
            word_path = get_model_dir(mn) / 'ctc_logprobs_word.pt'
            aux_path = get_model_dir(mn) / 'ctc_logprobs_aux.pt'
            load_path = word_path if word_path.exists() else aux_path
            if load_path.exists():
                word_data = torch.load(str(load_path), map_location='cpu', weights_only=False)
                all_word_logprobs[mn] = word_data['logprobs']
                word_head_types[mn] = word_data['meta']['head_type']
                if verbose:
                    print(f'  Loaded word ({word_data["meta"]["head_type"]}) logprobs for {mn}')
        assert all_word_logprobs, (
            'feat_word=True but no ctc_logprobs_word.pt found for any model. '
            'Either set --feat_word=False or train models with --save_word_head_preds --save_logprobs.')

    # Load age/domain meta predictions
    all_aux_meta = {}   # mn -> {uid: {'age_logits': np, 'domain_logit': float}}
    aux_meta_info = {}  # mn -> {'age_mode': str, 'has_age': bool, 'has_domain': bool}
    if feat_aux_meta:
        for mn in model_names:
            mp = get_model_dir(mn) / 'aux_meta_preds.pt'
            if mp.exists():
                data = torch.load(str(mp), map_location='cpu', weights_only=False)
                all_aux_meta[mn] = data['preds']
                aux_meta_info[mn] = data['meta']
                if verbose:
                    mi = data['meta']
                    print(f'  Loaded aux meta preds for {mn}: '
                          f'age={mi["has_age"]}(mode={mi["age_mode"]}), domain={mi["has_domain"]}')
        if verbose and not all_aux_meta:
            print(f'  WARNING: feat_aux_meta=True but no aux_meta_preds.pt found for any model')

    # For word_ctc aux heads: prepare eng_to_ipa converter and vocab
    _word_id_to_char = None
    _word_blank_id = 0
    _ipa_convert = None
    if feat_word and any(ht in ('word_ctc', 'word_ctc_bpe') for ht in word_head_types.values()):
        from src.models.base import WORD_ID_TO_CHAR, WORD_CTC_BLANK
        _word_id_to_char = WORD_ID_TO_CHAR
        _word_blank_id = WORD_CTC_BLANK
        try:
            from eng_to_ipa import convert as _ipa_convert
        except ImportError:
            print('  WARNING: eng_to_ipa not installed, word_ctc aux features will be limited')
            _ipa_convert = None

    uid_sets = [set(all_logprobs[mn].keys()) for mn in model_names]
    common_uids = set.intersection(*uid_sets)

    blank_id = IPA_CTC_BLANK if IPA_CTC_BLANK is not None else 0
    id_to_char = IPA_ID_TO_CHAR
    char_to_id = {ch: cid for cid, ch in id_to_char.items()} if id_to_char else None

    # Gold labels
    eval_csv = get_eval_csv(get_model_dir(model_names[0]))
    gold_df = pd.read_csv(eval_csv)
    gold = dict(zip(gold_df['utterance_id'], gold_df['label'].fillna('')))
    meta = {}
    for _, row in gold_df.iterrows():
        meta[row['utterance_id']] = {
            'source': row.get('source', ''),
            'age_bucket': row.get('age_bucket', ''),
            'child_id': row.get('child_id', ''),
        }

    uids_list = sorted(common_uids)
    total = len(uids_list)
    t0 = time.time()

    # ---- Set up shared state for _build_rows_for_uid ----
    global _MP_BUILD
    _MP_BUILD = {
        'model_names': model_names,
        'all_logprobs': all_logprobs,
        'blank_id': blank_id,
        'beam_width': beam_width,
        'nbest': nbest,
        'id_to_char': id_to_char,
        'char_to_id': char_to_id,
        'lm': lm,
        'no_lm_feats': no_lm_feats,
        'gold': gold,
        'meta': meta,
        'feat_text': feat_text,
        'feat_ipa': feat_ipa,
        'feat_ctc_stats': feat_ctc_stats,
        'feat_audio': feat_audio,
        'feat_consensus': feat_consensus,
        'feat_group_ext': feat_group_ext,
        'feat_align': feat_align,
        'feat_logprob_proxy': feat_logprob_proxy,
        'feat_word': feat_word,
        'feat_aux_meta': feat_aux_meta,
        'all_word_logprobs': all_word_logprobs,
        'word_head_types': word_head_types,
        'all_aux_meta': all_aux_meta,
        'aux_meta_info': aux_meta_info,
        '_word_id_to_char': _word_id_to_char,
        '_word_blank_id': _word_blank_id,
        '_ipa_convert': _ipa_convert,
    }

    # Auto-detect n_workers
    if n_workers <= 0:
        import os as _os
        n_workers = min(_os.cpu_count() or 1, 16)

    rows = []

    if n_workers > 1:
        # ---- Parallel path: multiprocessing with fork (COW shared memory) ----
        import multiprocessing as _mp
        ctx = _mp.get_context('fork')

        def _pool_init():
            """Disable torch internal threading in forked workers to prevent deadlocks."""
            import torch
            torch.set_num_threads(1)
            import os
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'

        if verbose:
            print(f'  Using {n_workers} workers for parallel feature building')
        chunksize = max(1, total // (n_workers * 4))
        with ctx.Pool(n_workers, initializer=_pool_init) as pool:
            for uid_rows in _iter_with_progress(
                    pool.imap_unordered(_build_rows_for_uid, uids_list, chunksize=chunksize),
                    total=total, desc='  Building features', verbose=verbose):
                rows.extend(uid_rows)
    else:
        # ---- Sequential path ----
        for uid in _iter_with_progress(uids_list, total=total, desc='  Building features', verbose=verbose):
            uid_rows = _build_rows_for_uid(uid)
            rows.extend(uid_rows)

    _MP_BUILD = {}  # Release references

    if verbose:
        elapsed = time.time() - t0
        print(f'\r  Built {len(rows)} candidate rows for {total} utterances in {elapsed:.1f}s')

    df = pd.DataFrame(rows)

    # ---- Group-relative features (computed per uid) ----
    # Per-model CTC: rank + per_char (always), zscore (group_ext)
    for mn in model_names:
        col = f'ctc_score_{mn}'
        df[f'{col}_rank'] = df.groupby('uid')[col].rank(ascending=False, method='min')
        df[f'{col}_per_char'] = df[col] / df['text_len'].clip(lower=1)
        if feat_group_ext:
            grp_mn = df.groupby('uid')[col]
            df[f'{col}_zscore'] = (df[col] - grp_mn.transform('mean')) / grp_mn.transform('std').clip(lower=1e-8)

    # CTC mean score group features (always: rank, zscore, diff_from_best)
    df['ctc_score_mean_rank'] = df.groupby('uid')['ctc_score_mean'].rank(ascending=False, method='min')
    grp = df.groupby('uid')['ctc_score_mean']
    df['ctc_score_mean_zscore'] = (df['ctc_score_mean'] - grp.transform('mean')) / grp.transform('std').clip(lower=1e-8)
    df['ctc_score_diff_from_best'] = df['ctc_score_mean'] - grp.transform('max')

    if feat_group_ext:
        df['ctc_score_mean_pct'] = df.groupby('uid')['ctc_score_mean'].rank(pct=True)
        df['ctc_score_diff_from_min'] = df['ctc_score_mean'] - grp.transform('min')
        df['ctc_score_diff_from_group_mean'] = df['ctc_score_mean'] - grp.transform('mean')
        df['ctc_score_diff_from_group_median'] = df['ctc_score_mean'] - grp.transform('median')
        _grp_min = grp.transform('min')
        _grp_range = (grp.transform('max') - _grp_min).clip(lower=1e-8)
        df['ctc_score_minmax_norm'] = (df['ctc_score_mean'] - _grp_min) / _grp_range

    df['ctc_score_mean_per_char'] = df['ctc_score_mean'] / df['text_len'].clip(lower=1)

    if feat_group_ext:
        for mn in model_names:
            col = f'ctc_score_{mn}'
            grp_mn = df.groupby('uid')[col]
            df[f'{col}_diff_from_median'] = df[col] - grp_mn.transform('median')

    # ---- Word-head group-relative features ----
    if feat_word and 'word_ctc_score_mean' in df.columns:
        grp_word = df.groupby('uid')['word_ctc_score_mean']
        df['word_ctc_score_mean_rank'] = grp_word.rank(ascending=False, method='min')
        df['word_ctc_score_diff_from_best'] = df['word_ctc_score_mean'] - grp_word.transform('max')
        # Per-model word-head score rank
        for mn in word_head_types:
            col = f'word_ctc_score_{mn}'
            if col in df.columns:
                df[f'{col}_rank'] = df.groupby('uid')[col].rank(ascending=False, method='min')

    for mn in model_names:
        df[f'is_best_{mn}'] = (df[f'beam_rank_{mn}'] == 0).astype(int)
    df['n_models_is_best'] = sum(df[f'is_best_{mn}'] for mn in model_names)

    # Text length group features
    grp_len = df.groupby('uid')['text_len']
    df['text_len_diff_from_median'] = df['text_len'] - grp_len.transform('median')
    if feat_text:
        df['text_len_rank'] = grp_len.rank(method='min')
        df['text_len_zscore'] = (df['text_len'] - grp_len.transform('mean')) / grp_len.transform('std').clip(lower=1e-8)

    df['n_candidates'] = df.groupby('uid')['uid'].transform('count')

    edit_cols = [f'edit_dist_to_best_{mn}' for mn in model_names]
    df['mean_edit_dist_to_best'] = df[edit_cols].mean(axis=1)
    df['min_edit_dist_to_best'] = df[edit_cols].min(axis=1)
    if feat_group_ext:
        df['max_edit_dist_to_best'] = df[edit_cols].max(axis=1)

    # LM score group features
    if not no_lm_feats and feat_group_ext:
        grp_lm = df.groupby('uid')['lm_score']
        df['lm_score_rank'] = grp_lm.rank(ascending=False, method='min')
        df['lm_score_zscore'] = (df['lm_score'] - grp_lm.transform('mean')) / grp_lm.transform('std').clip(lower=1e-8)
        df['lm_score_diff_from_best'] = df['lm_score'] - grp_lm.transform('max')
        df['lm_score_pct'] = grp_lm.rank(pct=True)

    if feat_consensus:
        grp_ped = df.groupby('uid')['mean_pairwise_edit_dist']
        df['mean_pairwise_edit_dist_rank'] = grp_ped.rank(method='min')

    if feat_text and 'n_spaces' in df.columns:
        grp_sp = df.groupby('uid')['n_spaces']
        df['n_spaces_diff_from_median'] = df['n_spaces'] - grp_sp.transform('median')

    # ---- MBR-related features ----
    if feat_consensus:
        import editdistance

        def _mbr_features(group):
            texts = group['candidate_text'].values
            n = len(texts)
            if n <= 1:
                group['is_mbr_selected'] = 1
                group['edit_dist_to_mbr'] = 0
                return group
            avg_cer = np.zeros(n)
            for i in range(n):
                total = 0
                for j in range(n):
                    if i != j:
                        ri = normalize_ipa(texts[i]).strip()
                        rj = normalize_ipa(texts[j]).strip()
                        if ri:
                            total += editdistance.eval(ri, rj) / len(ri)
                avg_cer[i] = total / (n - 1)
            mbr_idx = int(np.argmin(avg_cer))
            mbr_text = texts[mbr_idx]
            group['is_mbr_selected'] = (group['candidate_text'] == mbr_text).astype(int)
            group['edit_dist_to_mbr'] = group['candidate_text'].apply(
                lambda t: editdistance.eval(t, mbr_text))
            return group

        df = df.groupby('uid', group_keys=False).apply(_mbr_features)

    # Define feature columns (all numeric columns except identifiers and target)
    exclude = {'uid', 'candidate_text', 'source', 'target_cer', 'child_id', 'age_bucket'}
    feat_cols = [c for c in df.columns if c not in exclude]

    return df, feat_cols, gold, meta


def _set_tree_flags(tree_model, tree_iters, tree_lr, seed, tree_depth,
                    tree_leaves, tree_bagging, tree_feat_frac, tree_reg_lambda,
                    tree_early_stop, tree_device, tree_task, n_folds, verbose,
                    tree_obj=''):
    """Set gz.tree FLAGS for a specific tree_model."""
    from gezi import FLAGS
    import gezi as gz
    FLAGS.tree_model = tree_model
    FLAGS.iters = tree_iters
    FLAGS.tree_lr = tree_lr
    FLAGS.tree_seed = seed
    FLAGS.max_depth = tree_depth
    FLAGS.tree_bagging = tree_bagging
    FLAGS.feature_fraction = tree_feat_frac
    FLAGS.reg_lambda = tree_reg_lambda

    # Determine objective first (needed for num_leaves decision)
    if tree_obj:
        obj = tree_obj
    elif tree_task == 'regression':
        if tree_model == 'lgb':
            obj = 'regression'
        elif tree_model == 'xgb':
            obj = 'reg:squarederror'
        elif tree_model == 'cb':
            obj = 'RMSE'
        else:
            obj = 'regression'
    elif tree_model == 'lgb':
        obj = 'lambdarank'
    elif tree_model == 'xgb':
        obj = 'rank:ndcg'
    elif tree_model == 'cb':
        obj = 'YetiRank'
    else:
        obj = ''
    FLAGS.objective = obj

    # CatBoost pairwise losses require symmetric trees (no Lossguide),
    # so skip num_leaves for those objectives
    _cb_pairwise = {'PairLogit', 'PairLogitPairwise', 'PairAccuracy'}
    if tree_model == 'cb' and obj in _cb_pairwise:
        FLAGS.num_leaves = 0  # use default symmetric tree
    else:
        FLAGS.num_leaves = tree_leaves
    FLAGS.tree_verbose = -1
    FLAGS.tree_verbose_eval = 20 if verbose else 0
    FLAGS.tree_metric_period = 20
    FLAGS.tree_fit = (tree_model == 'lgb')
    FLAGS.tree_tb = False
    FLAGS.tree_eval_train = False
    FLAGS.tree_convert = True
    FLAGS.use_best_model = True
    FLAGS.early_stop = tree_early_stop
    FLAGS.device = tree_device
    FLAGS.model_dir = str(_PROJ_DIR / 'working' / 'reranker')
    FLAGS.model_name = 'reranker'
    FLAGS.num_folds = n_folds
    gz.try_mkdir(FLAGS.model_dir)


def _train_single_tree_fold_worker(output_queue, tree_model, fold_i,
                                   X_train, y_train, X_valid, y_valid,
                                   train_weight,
                                   train_uids, valid_uids,
                                   n_folds, tree_task,
                                   tree_iters, tree_lr, seed, tree_depth,
                                   tree_leaves, tree_bagging, tree_feat_frac,
                                   tree_reg_lambda, tree_early_stop,
                                   tree_device, tree_obj,
                                   tree_threads, verbose,
                                   model_save_dir=None):
    try:
        from gezi import FLAGS
        import gezi.tree as gt

        _set_tree_flags(tree_model, tree_iters, tree_lr, seed, tree_depth,
                        tree_leaves, tree_bagging, tree_feat_frac, tree_reg_lambda,
                        tree_early_stop, tree_device, tree_task, n_folds, verbose=False,
                        tree_obj=tree_obj)
        FLAGS.fold = fold_i
        if tree_threads:
            FLAGS.num_tree_threads = tree_threads

        os.environ['OMP_NUM_THREADS'] = str(tree_threads)
        os.environ['MKL_NUM_THREADS'] = str(tree_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(tree_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(tree_threads)

        if tree_task == 'ranking':
            group_train = _build_ranking_group_sizes(train_uids, split_name='train')
            group_valid = _build_ranking_group_sizes(valid_uids, split_name='valid')
            model = gt.Model(params={}, task_type='ranking')
            gt.fit(model, X_train, y_train,
                   weight=train_weight,
                   X_valid=X_valid, y_valid=y_valid,
                   group_train=group_train, group_valid=group_valid)
        else:
            model = gt.Model(params={}, task_type='regression')
            gt.fit(model, X_train, y_train,
                   weight=train_weight,
                   X_valid=X_valid, y_valid=y_valid)

        preds = model.predict(X_valid)
        feature_importance = _extract_tree_feature_importance(model, tree_model, X_train.columns.tolist())
        if model_save_dir:
            model_save_dir = Path(model_save_dir)
            model_save_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(model_save_dir))

        output_queue.put({
            'fold': fold_i,
            'preds': preds,
            'feature_importance': feature_importance,
            'model_save_dir': str(model_save_dir) if model_save_dir else '',
        })
    except Exception as e:
        output_queue.put({
            'fold': fold_i,
            'error': str(e),
            'traceback': traceback.format_exc(),
        })


def _should_parallelize_tree_cv(n_folds):
    """Inner tree CV folds are parallel by default (n_folds > 1).
    Disabled only inside parallel child subprocesses."""
    if n_folds <= 1:
        return False
    if os.environ.get('ENSEMBLE_PARALLEL_CHILD') == '1':
        return False
    return True


def _build_tree_sample_weights(df, dd_weight=1.0, ext_weight=1.0):
    dd_weight = float(dd_weight)
    ext_weight = float(ext_weight)
    assert dd_weight > 0, f'tree_dd_weight must be > 0, got {dd_weight}'
    assert ext_weight > 0, f'tree_ext_weight must be > 0, got {ext_weight}'
    source = df['source'].fillna('').astype(str)
    weights = np.ones(len(df), dtype=np.float32)
    weights[source == 'dd'] = dd_weight
    weights[source == 'ext'] = ext_weight
    return weights


def _build_ranking_group_sizes(uids, split_name='train'):
    uids = np.asarray(uids)
    if len(uids) == 0:
        return np.zeros(0, dtype=np.int32)

    group_sizes = []
    seen = set()
    start = 0
    while start < len(uids):
        uid = uids[start]
        if uid in seen:
            raise ValueError(
                f'Ranking {split_name} data must keep each uid in one contiguous block; '
                f'found repeated uid={uid!r}'
            )
        seen.add(uid)

        end = start + 1
        while end < len(uids) and uids[end] == uid:
            end += 1
        group_sizes.append(end - start)
        start = end

    return np.asarray(group_sizes, dtype=np.int32)


def _compute_source_mix_score(source_results, dd_weight=1.0, ext_weight=1.0):
    dd_weight = float(dd_weight)
    ext_weight = float(ext_weight)
    values = []
    weights = []
    if 'dd' in source_results and not pd.isna(source_results['dd']):
        values.append(float(source_results['dd']))
        weights.append(dd_weight)
    if 'ext' in source_results and not pd.isna(source_results['ext']):
        values.append(float(source_results['ext']))
        weights.append(ext_weight)
    if not values:
        return np.nan
    total_weight = float(sum(weights))
    if total_weight <= 0:
        return np.nan
    return float(sum(v * w for v, w in zip(values, weights)) / total_weight)


def _extract_tree_feature_importance(model, tree_model, feat_cols_clean):
    model_obj = getattr(model, 'model', model)
    if tree_model == 'lgb':
        if hasattr(model_obj, 'feature_importances_'):
            imp = np.asarray(model_obj.feature_importances_, dtype=np.float64)
        elif hasattr(model_obj, 'feature_importance'):
            imp = np.asarray(model_obj.feature_importance(importance_type='gain'), dtype=np.float64)
        elif hasattr(model_obj, 'booster_'):
            imp = np.asarray(model_obj.booster_.feature_importance(importance_type='gain'), dtype=np.float64)
        else:
            imp = np.zeros(len(feat_cols_clean), dtype=np.float64)
    elif tree_model == 'xgb':
        booster = model_obj.get_booster() if hasattr(model_obj, 'get_booster') else model_obj
        imp_dict = booster.get_score(importance_type='gain') if hasattr(booster, 'get_score') else {}
        imp = []
        for idx, feat in enumerate(feat_cols_clean):
            value = imp_dict.get(feat, imp_dict.get(f'f{idx}', 0.0))
            imp.append(float(value))
        imp = np.asarray(imp, dtype=np.float64)
    elif tree_model == 'cb':
        imp = np.asarray(model_obj.get_feature_importance(type='PredictionValuesChange'), dtype=np.float64)
    else:
        imp = np.zeros(len(feat_cols_clean), dtype=np.float64)

    if len(imp) != len(feat_cols_clean):
        raise ValueError(f'Feature importance size mismatch: {len(imp)} vs {len(feat_cols_clean)}')
    return [
        {'feature': feat, 'importance': float(value)}
        for feat, value in zip(feat_cols_clean, imp)
    ]


def _summarize_tree_feature_importances(fold_importances):
    fold_importances = [x for x in fold_importances if x]
    if not fold_importances:
        return None

    by_feature = defaultdict(list)
    for fold_idx, rows in enumerate(fold_importances):
        for row in rows:
            by_feature[row['feature']].append(float(row['importance']))

    summary_rows = []
    n_folds = len(fold_importances)
    for feature, values in by_feature.items():
        arr = np.asarray(values, dtype=np.float64)
        summary_rows.append({
            'feature': feature,
            'importance_mean': float(arr.mean()),
            'importance_std': float(arr.std()),
            'importance_max': float(arr.max()),
            'nonzero_folds': int(np.sum(arr != 0)),
            'fold_coverage': float(np.sum(arr != 0) / max(n_folds, 1)),
        })

    summary_rows.sort(key=lambda x: (-x['importance_mean'], x['feature']))
    return {
        'folds': fold_importances,
        'summary': summary_rows,
    }


def _format_tree_feature_importance_text(tree_model, importance_payload, top_k=30):
    if not importance_payload or not importance_payload.get('summary'):
        return ''
    lines = [f'\n--- Feature Importance ({tree_model}) ---']
    for row in importance_payload['summary'][:top_k]:
        lines.append(
            f"{row['feature']}\tmean={row['importance_mean']:.6f}\tstd={row['importance_std']:.6f}"
            f"\tnonzero_folds={row['nonzero_folds']}"
        )
    return '\n'.join(lines) + '\n'


def _train_single_tree(tree_model, df, feat_cols_clean, y_all, n_folds, tree_task,
                       tree_iters, tree_lr, seed, tree_depth, tree_leaves,
                       tree_bagging, tree_feat_frac, tree_reg_lambda,
                       tree_early_stop, tree_device, verbose, tree_obj='',
                       save_dir=None, save_models=False):
    """Train a single tree model type across all folds, return OOF predictions and models."""
    from gezi import FLAGS
    import gezi.tree as gt

    _set_tree_flags(tree_model, tree_iters, tree_lr, seed, tree_depth,
                    tree_leaves, tree_bagging, tree_feat_frac, tree_reg_lambda,
                    tree_early_stop, tree_device, tree_task, n_folds, verbose,
                    tree_obj=tree_obj)

    X_all = pd.DataFrame(df[feat_cols_clean].values, columns=feat_cols_clean)
    train_weights_all = _build_tree_sample_weights(
        df,
        dd_weight=getattr(FLAGS, 'ensemble_tree_dd_weight', 1.0),
        ext_weight=getattr(FLAGS, 'ensemble_tree_ext_weight', 1.0),
    )
    oof_preds = np.full(len(df), np.nan)
    fold_models = [None] * n_folds  # trained models or saved model dirs per fold
    fold_importances = [None] * n_folds

    if _should_parallelize_tree_cv(n_folds):
        max_parallel = n_folds
        total_cores = int(getattr(FLAGS, 'ensemble_parallel_cores', 0) or 0) or (os.cpu_count() or max_parallel)
        total_cores = max(total_cores, max_parallel)
        explicit_tree_threads = int(getattr(FLAGS, 'ensemble_parallel_tree_threads', 0) or 0)
        per_job_tree_threads = explicit_tree_threads or max(1, total_cores // max_parallel)
        if verbose:
            print(f'\n  Parallel tree CV enabled: jobs={max_parallel}, total_cores={total_cores}, per_job_tree_threads={per_job_tree_threads}')

        ctx = multiprocessing.get_context('fork')
        queue = ctx.Queue()
        pending = list(range(n_folds))
        running = []
        save_dir = Path(save_dir) if save_dir else None

        def _start_fold(fold_i):
            train_mask = (df['fold'] != fold_i).values
            valid_mask = (df['fold'] == fold_i).values
            X_train = X_all.loc[train_mask].reset_index(drop=True)
            y_train = y_all[train_mask]
            X_valid = X_all.loc[valid_mask].reset_index(drop=True)
            y_valid = y_all[valid_mask]
            train_weight = train_weights_all[train_mask]
            train_uids = df.loc[train_mask, 'uid'].values
            valid_uids = df.loc[valid_mask, 'uid'].values
            model_save_dir = None
            if save_models and save_dir is not None:
                model_save_dir = save_dir / f'tree_{tree_model}_fold{fold_i}'
            proc = ctx.Process(
                target=_train_single_tree_fold_worker,
                args=(queue, tree_model, fold_i,
                      X_train, y_train, X_valid, y_valid,
                        train_weight,
                      train_uids, valid_uids,
                      n_folds, tree_task,
                      tree_iters, tree_lr, seed, tree_depth,
                      tree_leaves, tree_bagging, tree_feat_frac,
                      tree_reg_lambda, tree_early_stop,
                      tree_device, tree_obj,
                      per_job_tree_threads, verbose,
                      str(model_save_dir) if model_save_dir else None)
            )
            proc.start()
            if verbose:
                print(f'\n  Fold {fold_i}: train={len(X_train)}, valid={len(X_valid)}, pid={proc.pid}')
            running.append({
                'fold': fold_i,
                'proc': proc,
                'valid_mask': valid_mask,
                'model_save_dir': model_save_dir,
            })

        while pending or running:
            while pending and len(running) < max_parallel:
                _start_fold(pending.pop(0))
            result = queue.get()
            fold_i = result['fold']
            job = next(item for item in running if item['fold'] == fold_i)
            job['proc'].join()
            running = [item for item in running if item['fold'] != fold_i]
            if result.get('error'):
                raise RuntimeError(
                    f'Parallel tree CV fold {fold_i} failed: {result["error"]}\n{result.get("traceback", "")}'
                )
            oof_preds[job['valid_mask']] = result['preds']
            fold_models[fold_i] = Path(result['model_save_dir']) if result.get('model_save_dir') else None
            fold_importances[fold_i] = result.get('feature_importance')

        importance_payload = _summarize_tree_feature_importances(fold_importances)
        if verbose and importance_payload:
            print(_format_tree_feature_importance_text(tree_model, importance_payload, top_k=25), end='')
        return oof_preds, fold_models, importance_payload

    for fold_i in range(n_folds):
        train_mask = (df['fold'] != fold_i).values
        valid_mask = (df['fold'] == fold_i).values

        X_train = X_all.loc[train_mask].reset_index(drop=True)
        y_train = y_all[train_mask]
        X_valid = X_all.loc[valid_mask].reset_index(drop=True)
        y_valid = y_all[valid_mask]
        train_weight = train_weights_all[train_mask]

        if verbose:
            print(f'\n  Fold {fold_i}: train={len(X_train)}, valid={len(X_valid)}')

        FLAGS.fold = fold_i

        if tree_task == 'ranking':
            train_uids = df.loc[train_mask, 'uid'].values
            valid_uids = df.loc[valid_mask, 'uid'].values
            group_train = _build_ranking_group_sizes(train_uids, split_name='train')
            group_valid = _build_ranking_group_sizes(valid_uids, split_name='valid')

            model = gt.Model(params={}, task_type='ranking')
            gt.fit(model, X_train, y_train,
                     weight=train_weight,
                   X_valid=X_valid, y_valid=y_valid,
                   group_train=group_train, group_valid=group_valid)
        else:
            model = gt.Model(params={}, task_type='regression')
            gt.fit(model, X_train, y_train,
                     weight=train_weight,
                   X_valid=X_valid, y_valid=y_valid)

        oof_preds[valid_mask] = model.predict(X_valid)
        if save_models and save_dir is not None:
            model_save_dir = Path(save_dir) / f'tree_{tree_model}_fold{fold_i}'
            model_save_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(model_save_dir))
            fold_models[fold_i] = model_save_dir
        else:
            fold_models[fold_i] = model

        fold_importances[fold_i] = _extract_tree_feature_importance(model, tree_model, feat_cols_clean)

    importance_payload = _summarize_tree_feature_importances(fold_importances)
    if verbose and importance_payload:
        print(_format_tree_feature_importance_text(tree_model, importance_payload, top_k=25), end='')

    return oof_preds, fold_models, importance_payload


def _classify_reranker_feature(feature_name):
    if (feature_name.startswith('is_dual_') or feature_name.startswith('dual_heads_agree_') or
            feature_name.startswith('dual_len_gap_') or feature_name.startswith('n_dual_')):
        return 'dual'

    if (feature_name.startswith('tdt_') or feature_name.startswith('is_tdt_') or
            feature_name.startswith('n_tdt_') or feature_name.startswith('tdtctc_')):
        return 'tdt'

    if feature_name.startswith('word_label_') or feature_name == 'has_word_label':
        return 'word_label'

    if (feature_name.startswith('aux_age_') or feature_name.startswith('aux_domain_') or
            feature_name.startswith('aux_nchars_') or feature_name.startswith('aux_nspaces_')):
        return 'aux_meta'

    if (feature_name.startswith('word_') or feature_name == 'word_primary_diff_mean' or
            feature_name.startswith('aux_') or feature_name == 'aux_primary_diff_mean'):
        return 'word'

    if (feature_name in {'blank_frame_ratio', 'avg_frame_confidence', 'min_phoneme_confidence',
                         'max_phoneme_confidence', 'std_phoneme_confidence', 'phoneme_dur_mean',
                         'phoneme_dur_std', 'phoneme_dur_min', 'phoneme_dur_max',
                         'phoneme_dur_zscore_max', 'single_frame_phoneme_ratio'}):
        return 'align'

    if (feature_name.startswith('entropy_') or feature_name in {'blank_prob_mean', 'top1_prob_mean',
                                                                'model_ctc_std', 'model_rank_std'} or
            feature_name.startswith('is_best_model_')):
        return 'logprob_proxy'

    if feature_name in {'duration_sec', 'chars_per_sec', 'words_per_sec'}:
        return 'audio'

    if feature_name in {'n_vowels', 'n_consonants', 'vowel_ratio', 'consonant_ratio',
                        'vc_ratio', 'n_length_marks'}:
        return 'ipa'

    if feature_name in {'n_spaces', 'n_words', 'avg_word_len', 'max_word_len', 'min_word_len',
                        'n_unique_chars', 'char_entropy', 'max_char_repeat', 'unique_char_ratio',
                        'repeat_char_ratio', 'text_len_rank', 'text_len_zscore',
                        'n_spaces_diff_from_median'}:
        return 'text'

    if feature_name in {'ctc_score_median', 'ctc_score_cv', 'ctc_score_skew',
                        'ctc_score_kurtosis', 'ctc_score_iqr'}:
        return 'ctc_stats'

    if feature_name in {'n_models_in_top3', 'mean_pairwise_edit_dist', 'consensus_score',
                        'n_exact_best', 'mean_pairwise_edit_dist_rank', 'is_mbr_selected',
                        'edit_dist_to_mbr'}:
        return 'consensus'

    if feature_name.startswith('lm_score') or '_lm_score' in feature_name:
        if feature_name in {'lm_score', 'lm_score_per_char', 'lm_score_per_word'}:
            return 'lm'
        if feature_name.endswith('_lm_score') or feature_name.endswith('_lm_score_per_char') or feature_name.endswith('_lm_score_per_word'):
            return 'lm'
        return 'group_ext'

    if feature_name in {'ctc_score_mean_pct', 'ctc_score_diff_from_min',
                        'ctc_score_diff_from_group_mean', 'ctc_score_diff_from_group_median',
                        'ctc_score_minmax_norm', 'max_edit_dist_to_best'}:
        return 'group_ext'

    if feature_name.startswith('ctc_score_') and feature_name.endswith('_diff_from_median'):
        return 'group_ext'

    if (feature_name.startswith('ctc_score_') and feature_name.endswith('_zscore') and
            feature_name != 'ctc_score_mean_zscore'):
        return 'group_ext'

    return 'core'


def _build_reranker_feature_manifest(model_names, feat_cols_clean, feature_flags,
                                     no_lm_feats=False, drop_feats='', tree_task='ranking',
                                     tree_model='cb', n_folds=5, nbest=10, beam_width=10,
                                     extra_flags=None):
    from collections import OrderedDict

    group_labels = OrderedDict([
        ('core', 'core / always-on'),
        ('lm', 'lm'),
        ('text', 'feat_text'),
        ('ipa', 'feat_ipa'),
        ('ctc_stats', 'feat_ctc_stats'),
        ('audio', 'feat_audio'),
        ('consensus', 'feat_consensus'),
        ('group_ext', 'feat_group_ext'),
        ('align', 'feat_align'),
        ('dual', 'feat_dual'),
        ('tdt', 'feat_tdt'),
        ('logprob_proxy', 'feat_logprob_proxy'),
        ('word', 'feat_word'),
        ('aux_meta', 'feat_aux_meta'),
        ('word_label', 'feat_word_label'),
    ])
    grouped = {key: [] for key in group_labels}
    unknown = []
    for feat in feat_cols_clean:
        group_key = _classify_reranker_feature(feat)
        if group_key in grouped:
            grouped[group_key].append(feat)
        else:
            unknown.append(feat)

    lines = []
    lines.append('Reranker Feature Manifest')
    lines.append('=========================')
    lines.append(f'tree_model: {tree_model}')
    lines.append(f'tree_task: {tree_task}')
    lines.append(f'n_folds: {n_folds}')
    lines.append(f'nbest: {nbest}')
    lines.append(f'beam_width: {beam_width}')
    lines.append(f'n_models: {len(model_names)}')
    lines.append(f'models: {", ".join(model_names)}')
    lines.append(f'n_features: {len(feat_cols_clean)}')
    lines.append('')
    lines.append('Effective Flags')
    lines.append('---------------')
    flag_order = [
        'feat_text', 'feat_ipa', 'feat_ctc_stats', 'no_ctc_score_feats', 'feat_audio', 'feat_consensus',
        'feat_mbr',
        'feat_group_ext', 'feat_align', 'feat_dual', 'feat_tdt', 'feat_tdt_light', 'feat_tdt_exact',
        'feat_tdt_primary_score', 'feat_tdt_nbest_score',
        'feat_tdtctc_compare', 'feat_tdt_score_compare',
        'feat_logprob_proxy', 'feat_word',
        'feat_aux_meta', 'feat_word_label',
    ]
    for key in flag_order:
        lines.append(f'{key}: {bool(feature_flags.get(key, False))}')
    if extra_flags:
        for key, value in extra_flags.items():
            lines.append(f'{key}: {value}')
    lines.append(f'no_lm_feats: {bool(no_lm_feats)}')
    lines.append(f'drop_feats: {drop_feats if drop_feats else "<none>"}')
    lines.append('')
    lines.append('Final Features')
    lines.append('--------------')
    for idx, feat in enumerate(feat_cols_clean, 1):
        lines.append(f'{idx:03d}. {feat}')
    lines.append('')
    lines.append('Features By Group')
    lines.append('-----------------')
    for key, label in group_labels.items():
        feats = grouped[key]
        lines.append(f'[{label}] ({len(feats)})')
        if feats:
            lines.extend(feats)
        else:
            lines.append('<none>')
        lines.append('')
    if unknown:
        lines.append('[unknown]')
        lines.extend(unknown)
        lines.append('')
    return '\n'.join(lines).rstrip() + '\n'


def _append_tree_experiment_log(log_dir, record):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'experiment_log.csv'
    row_df = pd.DataFrame([record])
    if log_path.exists():
        try:
            prev_df = pd.read_csv(log_path)
            row_df = pd.concat([prev_df, row_df], ignore_index=True)
        except Exception:
            pass
    row_df.to_csv(log_path, index=False)
    return log_path


def _dump_reranker_feature_frame(save_dir, df, feat_cols, dump_name,
                                 verbose=True, uid_limit=0, uid_filter_path=''):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dump_df = df.copy()
    uid_filter_path = str(uid_filter_path or '').strip()
    if uid_filter_path and 'uid' in dump_df.columns:
        filter_path = Path(uid_filter_path)
        assert filter_path.exists(), f'dump_feats_uids_path not found: {filter_path}'
        suffix = filter_path.suffix.lower()
        uid_values = None
        if suffix in ('.pkl', '.pickle'):
            uid_source = pd.read_pickle(filter_path)
        elif suffix == '.csv':
            uid_source = pd.read_csv(filter_path)
        elif suffix == '.jsonl':
            uid_source = pd.read_json(filter_path, lines=True)
        else:
            uid_source = None

        if uid_source is not None:
            if isinstance(uid_source, pd.Series):
                uid_values = uid_source.astype(str).tolist()
            elif isinstance(uid_source, pd.DataFrame):
                for col in ('uid', 'utterance_id', 'id'):
                    if col in uid_source.columns:
                        uid_values = uid_source[col].astype(str).tolist()
                        break
            else:
                uid_values = list(uid_source)
        if uid_values is None:
            uid_values = [line.strip() for line in filter_path.read_text().splitlines() if line.strip()]

        keep_uid_set = set(uid_values)
        dump_df = dump_df[dump_df['uid'].astype(str).isin(keep_uid_set)].copy()
    if uid_limit and uid_limit > 0 and 'uid' in dump_df.columns:
        keep_uids = dump_df['uid'].drop_duplicates().tolist()[:uid_limit]
        dump_df = dump_df[dump_df['uid'].isin(set(keep_uids))].copy()

    dump_path = save_dir / f'{dump_name}.pkl'
    meta_path = save_dir / f'{dump_name}.meta.json'
    dump_df.to_pickle(dump_path)
    meta = {
        'feat_cols': list(feat_cols),
        'columns': dump_df.columns.tolist(),
        'rows': int(len(dump_df)),
        'uid_count': int(dump_df['uid'].nunique()) if 'uid' in dump_df.columns else 0,
        'uid_limit': int(uid_limit or 0),
        'uid_filter_path': uid_filter_path,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f'  Saved {dump_name}.pkl: {dump_path} ({meta["rows"]} rows, {meta["uid_count"]} uids)')
        print(f'  Saved {dump_name}.meta.json: {meta_path}')
    return dump_path, meta_path


def tree_reranker_ensemble(model_names, nbest=10, beam_width=10,
                           lm=None, aux_lms=None, tree_model='cb', n_folds=5, seed=42,
                           tree_iters=500, tree_lr=0.05, tree_depth=3,
                           tree_leaves=31, tree_bagging=0.8, tree_feat_frac=0.6,
                           tree_reg_lambda=5.0, tree_early_stop=50,
                           tree_dd_weight=1.0, tree_ext_weight=1.0,
                           tree_device='cpu', tree_task='ranking',
                           tree_obj='', relevance_levels=4,
                           relevance_strategy='rank',
                           no_lm_feats=False,
                           lm_per_word_only=False,
                           feat_text=False, feat_ipa=False,
                           feat_ctc_stats=False, no_ctc_score_feats=False,
                           feat_audio=False,
                           feat_consensus=False, feat_mbr=False,
                           feat_group_ext=False,
                           feat_align=False,
                           feat_tdt=False,
                           feat_tdt_light=False,
                           feat_tdt_primary_score=False,
                           feat_tdt_nbest_score=False,
                           feat_tdt_exact=False,
                           feat_tdt_group=False,
                           feat_tdtctc_compare=False,
                           feat_tdt_score_compare=True,
                           feat_wavlm_group=False,
                           feat_nemo_group=False,
                           feat_group_edit_dist=False,
                           feat_dual=False,
                           feat_logprob_proxy=False,
                           feat_word=False,
                           feat_aux=None,
                           feat_aux_meta=False,
                           feat_word_label=False,
                           word_label_file='',
                           word_label_col='',
                           tdt_eval_nbest=0,
                           tdt_feat_topk=8,
                           tdt_force_keep_preds=True,
                           cache_tdt_exact_scores=True,
                           drop_feats='',
                           cache_dataset=False,
                           n_seeds=1,
                           n_workers=0,
                           model_max_dur=None,
                           save_dir=None,
                           exp_name='',
                           exp_notes='',
                           dump_feats=False,
                           dump_feats_limit=0,
                           dump_feats_uids_path='',
                           verbose=True,
                           return_details=False):
    """Train a tree reranker on N-best candidates with K-fold CV using gz.tree.

    Supports two task modes:
      - 'ranking': LambdaRank - train ranker with relevance labels, select highest score
      - 'regression': Predict CER directly, select lowest predicted CER

    tree_model can be a single model ('lgb') or comma-separated ('lgb,xgb,cb')
    for multi-tree ensemble (averaged OOF predictions).
    """
    import melt as mt  # registers absl FLAGS
    import gezi as gz
    from gezi import FLAGS
    import gezi.tree as gt

    # ---- Parse FLAGS if not yet parsed ----
    if not FLAGS.is_parsed():
        FLAGS(sys.argv[:1], known_only=True)

    # Parse tree_model list
    tree_models = [tm.strip() for tm in tree_model.split(',')]
    for tm in tree_models:
        assert tm in ('lgb', 'xgb', 'cb'), f'Unknown tree model: {tm}'
    is_multi = len(tree_models) > 1
    tm_label = '+'.join(tree_models) if is_multi else tree_models[0]

    if verbose:
        print(f'\n=== Tree Reranker Ensemble (tm={tm_label}, folds={n_folds}) ===')
        print(f'Models: {", ".join(model_names)}')
        print(f'Tree source weights: dd={tree_dd_weight}, ext={tree_ext_weight}')

    feat_word = feat_word or bool(feat_aux)
    feat_tdt_light = bool(feat_tdt_light or feat_tdt)
    feat_tdt_primary_score = bool(feat_tdt_primary_score)
    feat_tdt_nbest_score = bool(feat_tdt_nbest_score)
    feat_tdt_exact = bool(feat_tdt_exact or feat_tdt)
    use_tdt_feats = (feat_tdt_light or feat_tdt_primary_score or feat_tdt_nbest_score or
                     feat_tdt_exact or feat_tdt_group)

    # Step 1: Build dataset (with optional caching for repeated experiments)
    _cache_dir = WORKING_BASE / 'ensemble' / 'cache'
    _cache_path = _cache_dir / 'dataset.pkl'

    # Build per-model max audio duration filter
    _wavlm_max_dur = float(getattr(FLAGS, 'ensemble_wavlm_max_dur', 0) or 0)
    _model_max_dur = {
        str(mn): float(max_dur)
        for mn, max_dur in (model_max_dur or {}).items()
        if mn in set(model_names) and max_dur is not None and float(max_dur) > 0
    }
    if _wavlm_max_dur > 0:
        _wavlm_mns = _detect_wavlm_model_names(model_names)
        if _wavlm_mns:
            for mn in _wavlm_mns:
                _model_max_dur.setdefault(mn, _wavlm_max_dur)
    if verbose and _model_max_dur:
        summary = ', '.join(
            f'{mn}<={_model_max_dur[mn]:.1f}s'
            for mn in model_names if mn in _model_max_dur
        )
        print(f'  Model max duration filters: {summary}')
    if not _model_max_dur:
        _model_max_dur = None

    if cache_dataset and _cache_path.exists():
        import pickle
        with open(_cache_path, 'rb') as _f:
            df, feat_cols, gold, meta = pickle.load(_f)
        if verbose:
            print(f'  Loaded cached dataset: {_cache_path} ({len(df)} rows, {len(feat_cols)} features)')
    else:
        os.environ['ENSEMBLE_PROGRESS'] = FLAGS.ensemble_progress
        df, feat_cols, gold, meta = build_reranker_dataset_impl(
            model_names, get_model_dir=get_model_dir, get_eval_csv=get_eval_csv,
            prefix_beam_search_nbest=prefix_beam_search_nbest,
            ctc_force_score_batch=ctc_force_score_batch,
            normalize_ipa=normalize_ipa, id_to_char=IPA_ID_TO_CHAR,
            blank_id=IPA_CTC_BLANK if IPA_CTC_BLANK is not None else 0,
            nbest=nbest, beam_width=beam_width, lm=lm, aux_lms=aux_lms, verbose=verbose,
            feat_text=feat_text, feat_ipa=feat_ipa,
            feat_ctc_stats=feat_ctc_stats, feat_audio=feat_audio,
            feat_consensus=feat_consensus, feat_mbr=feat_mbr,
            feat_group_ext=feat_group_ext,
            feat_align=feat_align, feat_logprob_proxy=feat_logprob_proxy,
            feat_tdtctc_compare=feat_tdtctc_compare,
            feat_dual=feat_dual,
            feat_word=feat_word, feat_aux_meta=feat_aux_meta,
            feat_word_label=feat_word_label,
            word_label_file=word_label_file,
            word_label_col=word_label_col,
            tdt_eval_nbest=tdt_eval_nbest,
            no_lm_feats=no_lm_feats, n_workers=n_workers,
            allow_eval_set_mismatch=bool(getattr(FLAGS, 'ensemble_allow_eval_set_mismatch', False)),
            model_max_dur=_model_max_dur)
        if cache_dataset:
            import pickle
            _cache_dir.mkdir(parents=True, exist_ok=True)
            with open(_cache_path, 'wb') as _f:
                pickle.dump((df, feat_cols, gold, meta), _f)
            if verbose:
                print(f'  Cached dataset to {_cache_path}')

    tdt_model_names = []
    if use_tdt_feats:
        tdt_model_names = _detect_tdt_model_names(model_names, verbose=verbose)
        if tdt_model_names:
            primary_tdt_texts, primary_tdt_scores, tdt_nbest_score_maps = _load_tdt_primary_texts(
                model_names, tdt_model_names, verbose=verbose)
            if verbose:
                print(f'  TDT feature groups: light={feat_tdt_light}, '
                      f'primary_score={feat_tdt_primary_score}, '
                      f'nbest_score={feat_tdt_nbest_score}, exact={feat_tdt_exact}')
            tdt_score_maps = None
            if feat_tdt_exact:
                candidate_lists = _build_tdt_feature_candidate_lists(
                    df,
                    tdt_model_names,
                    primary_tdt_texts,
                    topk=tdt_feat_topk,
                    force_keep_preds=tdt_force_keep_preds,
                )
                if verbose:
                    n_tdt_candidates = sum(len(v) for v in candidate_lists.values())
                    avg_tdt_candidates = n_tdt_candidates / max(len(candidate_lists), 1)
                    print(f'  TDT exact scoring: {len(tdt_model_names)} model(s), '
                          f'topk={tdt_feat_topk}, force_keep={tdt_force_keep_preds}, '
                          f'avg_candidates={avg_tdt_candidates:.1f}, '
                          f'cache={cache_tdt_exact_scores}')
                tdt_score_maps = {}
                for mn in tdt_model_names:
                    scores_by_uid = _score_tdt_candidates_for_model(
                        mn,
                        candidate_lists,
                        verbose=verbose,
                        use_cache=cache_tdt_exact_scores,
                    )
                    tdt_score_maps[mn] = _convert_tdt_score_arrays(candidate_lists, scores_by_uid)
            use_tdt_score_compare = bool(feat_tdt_score_compare and 'ctc_score_mean' in df.columns)
            if feat_tdt_score_compare and not use_tdt_score_compare and verbose:
                print('  TDT-vs-CTC compare features skipped: no ctc_score_mean available')
            df, tdt_feat_cols = _augment_tdt_feature_frame(
                df,
                tdt_model_names,
                primary_tdt_texts,
                primary_tdt_scores=primary_tdt_scores,
                tdt_nbest_score_maps=tdt_nbest_score_maps,
                tdt_score_maps=tdt_score_maps,
                include_light=feat_tdt_light,
                include_primary_score=feat_tdt_primary_score,
                include_nbest_score=feat_tdt_nbest_score,
                include_exact=feat_tdt_exact,
                include_score_compare=use_tdt_score_compare,
            )
            feat_cols = list(dict.fromkeys(list(feat_cols) + list(tdt_feat_cols)))
            if verbose:
                print(f'  TDT feature columns added: {len(tdt_feat_cols)} '
                      f'(light={feat_tdt_light}, exact={feat_tdt_exact})')
            if feat_tdt_group:
                df, tdt_group_cols = _augment_model_subset_feature_frame(
                    df,
                    group_name='tdt',
                    subset_model_names=tdt_model_names,
                    primary_texts=primary_tdt_texts,
                    feat_edit_dist=feat_group_edit_dist,
                )
                feat_cols = list(dict.fromkeys(list(feat_cols) + list(tdt_group_cols)))
                if verbose:
                    print(f'  TDT subgroup feature columns added: {len(tdt_group_cols)}')
        elif verbose:
            print('  TDT features requested but no TDT-capable models detected; skip TDT features')

    model_flags_by_name = {mn: _load_saved_flags(get_model_dir(mn)) for mn in model_names}

    wavlm_model_names = []
    if feat_wavlm_group:
        wavlm_model_names = _detect_wavlm_model_names(model_names, flags_by_name=model_flags_by_name)
        if wavlm_model_names:
            df, feat_cols, _ = _augment_family_group_features(
                df, feat_cols, model_names, 'wavlm', wavlm_model_names,
                verbose=verbose, feat_edit_dist=feat_group_edit_dist)

    nemo_model_names = []
    if feat_nemo_group:
        nemo_model_names = _detect_nemo_model_names(model_names, flags_by_name=model_flags_by_name)
        if nemo_model_names:
            df, feat_cols, _ = _augment_family_group_features(
                df, feat_cols, model_names, 'nemo', nemo_model_names,
                verbose=verbose, feat_edit_dist=feat_group_edit_dist)

    if verbose:
        print(f'Dataset: {len(df)} rows, {len(feat_cols)} features')
        print(f'Features: {feat_cols}')

    # Step 2: Assign folds to utterances — StratifiedGroupKFold by child_id + source_age
    from sklearn.model_selection import StratifiedGroupKFold

    # Build uid-level table for fold assignment
    uid_info = df.groupby('uid').first()[['child_id', 'source', 'age_bucket']].reset_index()
    uid_info['strat_key'] = uid_info['source'] + '_' + uid_info['age_bucket']
    groups = uid_info['child_id'].values
    strat_labels = uid_info['strat_key'].values
    uids_arr = uid_info['uid'].values

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    uid_folds = {}
    for fold_i, (_, val_idx) in enumerate(sgkf.split(uids_arr, strat_labels, groups)):
        for idx in val_idx:
            uid_folds[uids_arr[idx]] = fold_i
    df['fold'] = df['uid'].map(uid_folds)

    if verbose:
        for f in range(n_folds):
            fold_mask = df['fold'] == f
            n = fold_mask.sum()
            n_uid = (df.loc[fold_mask, 'uid']).nunique()
            n_child = (df.loc[fold_mask, 'child_id']).nunique()
            print(f'  Fold {f}: {n_uid} uids, {n_child} children, {n} candidates')

    # Step 3: Prepare labels (features already filtered by flags in build_reranker_dataset)
    feat_cols_clean = [c for c in feat_cols if c != 'fold']

    def _is_ctc_score_feature(feature_name):
        feature_name = str(feature_name)
        if 'ctc_score' in feature_name:
            return True
        if feature_name.startswith('word_score_diff_'):
            return True
        if feature_name == 'word_primary_diff_mean':
            return True
        if feature_name.endswith('_score_ctc_rank_gap'):
            return True
        if feature_name.endswith('_vs_ctc_best_gap'):
            return True
        return False

    # Ablation: drop features matching regex patterns
    if drop_feats:
        import re as _re
        patterns = [p.strip() for p in drop_feats.split(',')]
        before_n = len(feat_cols_clean)
        dropped = [c for c in feat_cols_clean if any(_re.search(p, c) for p in patterns)]
        feat_cols_clean = [c for c in feat_cols_clean if c not in set(dropped)]
        if verbose:
            print(f'  Dropped {before_n - len(feat_cols_clean)} features matching {patterns}: {before_n} -> {len(feat_cols_clean)}')
            if dropped:
                print(f'    Dropped: {dropped}')

    if lm_per_word_only:
        before_n = len(feat_cols_clean)
        lm_drop_cols = [
            c for c in feat_cols_clean
            if ((c.startswith('lm_score') or '_lm_score' in c) and c != 'lm_score_per_word')
        ]
        feat_cols_clean = [c for c in feat_cols_clean if c not in set(lm_drop_cols)]
        if verbose:
            print(f'  Applied lm_per_word_only: {before_n} -> {len(feat_cols_clean)}')
            if lm_drop_cols:
                print(f'    Dropped LM features: {lm_drop_cols}')

    if no_ctc_score_feats:
        before_n = len(feat_cols_clean)
        ctc_drop_cols = [c for c in feat_cols_clean if _is_ctc_score_feature(c)]
        feat_cols_clean = [c for c in feat_cols_clean if c not in set(ctc_drop_cols)]
        if verbose:
            print(f'  Applied no_ctc_score_feats: {before_n} -> {len(feat_cols_clean)}')
            if ctc_drop_cols:
                print(f'    Dropped CTC score features: {ctc_drop_cols}')

    if verbose:
        _enabled = []
        for name, flag in [('text', feat_text), ('ipa', feat_ipa),
                           ('ctc_stats', feat_ctc_stats), ('no_ctc_score_feats', no_ctc_score_feats),
                           ('audio', feat_audio),
                           ('consensus', feat_consensus), ('mbr', feat_mbr),
                           ('group_ext', feat_group_ext),
                           ('align', feat_align), ('dual', feat_dual),
                           ('tdt_light', feat_tdt_light),
                           ('tdt_primary_score', feat_tdt_primary_score),
                           ('tdt_nbest_score', feat_tdt_nbest_score),
                           ('tdt_exact', feat_tdt_exact),
                           ('tdt_group', feat_tdt_group),
                           ('tdtctc_compare', feat_tdtctc_compare),
                           ('tdt_score_compare', feat_tdt_score_compare),
                           ('wavlm_group', feat_wavlm_group),
                           ('nemo_group', feat_nemo_group),
                           ('logprob_proxy', feat_logprob_proxy),
                           ('word', feat_word), ('aux_meta', feat_aux_meta)]:
            _enabled.append(f'{name}={"ON" if flag else "off"}')
        print(f'  Feature groups: {", ".join(_enabled)}, no_lm_feats={no_lm_feats}, lm_per_word_only={lm_per_word_only}')
        print(f'  Final features: {len(feat_cols_clean)}')

    feature_flags = {
        'feat_text': feat_text,
        'feat_ipa': feat_ipa,
        'feat_ctc_stats': feat_ctc_stats,
        'no_ctc_score_feats': no_ctc_score_feats,
        'feat_audio': feat_audio,
        'feat_consensus': feat_consensus,
        'feat_mbr': feat_mbr,
        'feat_group_ext': feat_group_ext,
        'feat_align': feat_align,
        'feat_dual': feat_dual,
        'feat_logprob_proxy': feat_logprob_proxy,
        'feat_word': feat_word,
        'feat_aux_meta': feat_aux_meta,
        'feat_word_label': feat_word_label,
        'feat_tdtctc_compare': feat_tdtctc_compare,
        'feat_tdt_primary_score': feat_tdt_primary_score,
        'feat_tdt_nbest_score': feat_tdt_nbest_score,
        'feat_tdt_score_compare': feat_tdt_score_compare,
        'feat_tdt_group': feat_tdt_group,
        'feat_wavlm_group': feat_wavlm_group,
        'feat_nemo_group': feat_nemo_group,
    }
    manifest_feature_flags = dict(feature_flags)
    manifest_feature_flags['feat_tdt'] = use_tdt_feats
    manifest_feature_flags['feat_tdt_light'] = feat_tdt_light
    manifest_feature_flags['feat_tdt_exact'] = feat_tdt_exact
    feature_manifest = _build_reranker_feature_manifest(
        model_names=model_names,
        feat_cols_clean=feat_cols_clean,
        feature_flags=manifest_feature_flags,
        no_lm_feats=no_lm_feats,
        drop_feats=drop_feats,
        tree_task=tree_task,
        tree_model=tm_label,
        n_folds=n_folds,
        nbest=nbest,
        beam_width=beam_width,
        extra_flags={
            'ensemble_feat_all': bool(getattr(FLAGS, 'ensemble_feat_all', False)),
            'ensemble_show_feats': bool(getattr(FLAGS, 'ensemble_show_feats', False)),
            'ensemble_lm_feats': bool(getattr(FLAGS, 'ensemble_lm_feats', False)),
            'ensemble_lm_per_word_only': bool(lm_per_word_only),
            'ensemble_no_ctc_score_feats': bool(no_ctc_score_feats),
            'ensemble_feat_mbr': bool(feat_mbr),
            'ensemble_lm_path': FLAGS.ensemble_lm_path if FLAGS.ensemble_lm_path else '<none>',
            'ensemble_word_lm_path': FLAGS.ensemble_word_lm_path if FLAGS.ensemble_word_lm_path else '<none>',
            'ensemble_feat_tdt': bool(use_tdt_feats),
            'ensemble_feat_tdt_light': bool(feat_tdt_light),
            'ensemble_feat_tdt_primary_score': bool(feat_tdt_primary_score),
            'ensemble_feat_tdt_nbest_score': bool(feat_tdt_nbest_score),
            'ensemble_feat_tdt_exact': bool(feat_tdt_exact),
            'ensemble_feat_tdt_group': bool(feat_tdt_group),
            'ensemble_feat_tdtctc_compare': bool(feat_tdtctc_compare),
            'ensemble_feat_tdt_score_compare': bool(feat_tdt_score_compare),
            'ensemble_feat_wavlm_group': bool(feat_wavlm_group),
            'ensemble_feat_nemo_group': bool(feat_nemo_group),
            'ensemble_feat_dual': bool(feat_dual),
            'ensemble_tdt_feat_topk': int(tdt_feat_topk),
            'ensemble_tdt_force_keep_preds': bool(tdt_force_keep_preds),
            'ensemble_cache_tdt_exact_scores': bool(cache_tdt_exact_scores),
            'tdt_model_names': ','.join(tdt_model_names) if tdt_model_names else '<none>',
            'wavlm_model_names': ','.join(wavlm_model_names) if wavlm_model_names else '<none>',
            'nemo_model_names': ','.join(nemo_model_names) if nemo_model_names else '<none>',
            'ensemble_tree_dd_weight': float(tree_dd_weight),
            'ensemble_tree_ext_weight': float(tree_ext_weight),
            'ensemble_relevance_strategy': relevance_strategy,
            'word_label_file': word_label_file if word_label_file else '<none>',
            'word_label_col': word_label_col if word_label_col else '<none>',
        },
    )
    if getattr(FLAGS, 'ensemble_show_feats', False):
        print('\n--- Reranker Features ---')
        print(feature_manifest, end='')

    if tree_task == 'ranking':
        # --- Ranking mode: convert CER to relevance labels ---
        from scipy.stats import rankdata

        def cer_to_relevance(cer_arr, n_levels=4, strategy='rank'):
            """Convert CER values to relevance labels within a group.
            Lower CER -> higher relevance. Uses rank-based mapping."""
            cer_arr = np.asarray(cer_arr, dtype=np.float32)
            n = len(cer_arr)
            if n == 1:
                return np.array([1 if n_levels == 0 else max(int(n_levels) - 1, 0)], dtype=np.int32)
            if n_levels == 0 or strategy == 'binary_best':
                best = np.min(cer_arr)
                return np.isclose(cer_arr, best).astype(np.int32)

            n_lvl = max(2, min(n, int(n_levels)))
            if strategy == 'gap':
                best = float(np.min(cer_arr))
                worst = float(np.max(cer_arr))
                if worst <= best + 1e-8:
                    return np.full(n, n_lvl - 1, dtype=np.int32)
                normalized = 1.0 - (cer_arr - best) / (worst - best)
                relevance = np.round(normalized * (n_lvl - 1)).astype(np.int32)
                relevance = np.clip(relevance, 0, n_lvl - 1)
                return relevance

            ranks = rankdata(cer_arr, method='average')
            relevance = np.round((1 - (ranks - 1) / (n - 1)) * (n_lvl - 1)).astype(np.int32)
            return relevance

        df['relevance'] = 0
        for uid, group in df.groupby('uid'):
            cer_arr = group['target_cer'].values
            rel = cer_to_relevance(
                cer_arr,
                n_levels=relevance_levels,
                strategy=relevance_strategy,
            )
            df.loc[group.index, 'relevance'] = rel

        if verbose:
            print(f'Relevance levels: {relevance_levels} (0=binary, N=graded)')
            print(f'Relevance strategy: {relevance_strategy}')
            print(f'  Unique relevance values: {sorted(df["relevance"].unique())}')

        # Sort by uid within each fold for correct group alignment
        df = df.sort_values(['fold', 'uid']).reset_index(drop=True)
        y_all = df['relevance'].values
    else:
        y_all = df['target_cer'].values

    if save_dir is None:
        if exp_name:
            save_dir = WORKING_BASE / 'ensemble-experiments' / exp_name / str(FOLD)
        else:
            auto_name = _build_ensemble_model_name()
            save_dir = WORKING_BASE / auto_name / str(FOLD)
    else:
        save_dir = Path(save_dir)

    if dump_feats:
        _dump_reranker_feature_frame(
            save_dir,
            df,
            feat_cols_clean,
            dump_name='reranker_feats.offline',
            verbose=verbose,
            uid_limit=dump_feats_limit,
            uid_filter_path=dump_feats_uids_path,
        )

    # Step 4: Train tree model(s) — multi-seed for stable evaluation
    # Fold assignment stays fixed; only tree training seed varies.
    seeds = [seed + i for i in range(n_seeds)]
    all_seed_oof = []  # per-seed OOF predictions
    saved_fold_models = None  # save first seed's models for inference
    saved_feature_importances = None

    for s_i, s in enumerate(seeds):
        if n_seeds > 1 and verbose:
            print(f'\n--- Seed {s} ({s_i+1}/{n_seeds}) ---')

        all_oof = {}
        all_fold_models = {}
        all_importances = {}
        for tm_i, tm in enumerate(tree_models):
            if verbose and is_multi:
                print(f'\n--- Training tree model: {tm} ({tm_i+1}/{len(tree_models)}) ---')
            oof, fold_models, importance_payload = _train_single_tree(
                tm, df, feat_cols_clean, y_all, n_folds, tree_task,
                tree_iters, tree_lr, s, tree_depth, tree_leaves,
                tree_bagging, tree_feat_frac, tree_reg_lambda,
                tree_early_stop, tree_device, verbose=(verbose and n_seeds == 1),
                tree_obj=tree_obj,
                save_dir=save_dir,
                save_models=(s_i == 0))
            all_oof[tm] = oof
            all_fold_models[tm] = fold_models
            all_importances[tm] = importance_payload

        if is_multi:
            oof_stack = np.stack(list(all_oof.values()), axis=0)
            seed_oof = oof_stack.mean(axis=0)
        else:
            seed_oof = all_oof[tree_models[0]]

        all_seed_oof.append(seed_oof)
        if s_i == 0:
            saved_fold_models = all_fold_models
            saved_feature_importances = all_importances

        # Per-seed evaluation
        if n_seeds > 1 and verbose:
            df['pred_score'] = seed_oof
            _preds = {}
            if tree_task == 'ranking':
                for uid, group in df.groupby('uid'):
                    _preds[uid] = group.loc[group['pred_score'].idxmax()]['candidate_text']
            else:
                for uid, group in df.groupby('uid'):
                    _preds[uid] = group.loc[group['pred_score'].idxmin()]['candidate_text']
            _evaluate(_preds, gold, meta,
                      label=f'Seed {s} ({tm_label}, {n_folds}-fold)', verbose=verbose)

    # Average across seeds
    if n_seeds > 1:
        oof_preds = np.mean(all_seed_oof, axis=0)
        if verbose:
            # Report per-uid CER std across seeds to quantify instability
            seed_cers = []
            for soof in all_seed_oof:
                df['pred_score'] = soof
                _p = {}
                if tree_task == 'ranking':
                    for uid, group in df.groupby('uid'):
                        _p[uid] = group.loc[group['pred_score'].idxmax()]['candidate_text']
                else:
                    for uid, group in df.groupby('uid'):
                        _p[uid] = group.loc[group['pred_score'].idxmin()]['candidate_text']
                r = _evaluate(_p, gold, meta, label='', verbose=False)
                seed_cers.append(r['overall_cer'])
            print(f'\n--- Seed ensemble ({n_seeds} seeds) ---')
            print(f'  Per-seed CERs: {["{:.5f}".format(c) for c in seed_cers]}')
            print(f'  Mean: {np.mean(seed_cers):.5f}  Std: {np.std(seed_cers):.5f}')
    else:
        oof_preds = all_seed_oof[0]

    # Use first seed's models for saving
    all_fold_models = saved_fold_models
    all_feature_importances = saved_feature_importances or {}

    if is_multi and verbose:
        print(f'\n--- Averaged predictions from {len(tree_models)} tree models ---')
        for tm, oof in {tm: all_oof[tm] for tm in tree_models}.items():
            df['pred_score'] = oof
            preds_tm = {}
            if tree_task == 'ranking':
                for uid, group in df.groupby('uid'):
                    preds_tm[uid] = group.loc[group['pred_score'].idxmax()]['candidate_text']
            else:
                for uid, group in df.groupby('uid'):
                    preds_tm[uid] = group.loc[group['pred_score'].idxmin()]['candidate_text']
            _evaluate(preds_tm, gold, meta,
                      label=f'Tree Reranker ({tm}, {n_folds}-fold)', verbose=verbose)

    df['pred_score'] = oof_preds

    # Step 5: For each uid, select best candidate
    ensemble_preds = {}
    if tree_task == 'ranking':
        for uid, group in df.groupby('uid'):
            best_row = group.loc[group['pred_score'].idxmax()]
            ensemble_preds[uid] = best_row['candidate_text']
    else:
        for uid, group in df.groupby('uid'):
            best_row = group.loc[group['pred_score'].idxmin()]
            ensemble_preds[uid] = best_row['candidate_text']

    # Also build baseline (avg CTC score) predictions for comparison
    has_ctc_score_baseline = 'ctc_score_mean' in df.columns
    avg_score_preds = {}
    if has_ctc_score_baseline:
        for uid, group in df.groupby('uid'):
            best_row = group.loc[group['ctc_score_mean'].idxmax()]
            avg_score_preds[uid] = best_row['candidate_text']

    def _load_tree_model_for_eval(tree_model, model_ref):
        if not isinstance(model_ref, Path):
            return model_ref
        model_dir = Path(model_ref)
        model_txt = model_dir / 'model.txt'
        model_json = model_dir / 'model.json'
        model_pkl = model_dir / 'model.pkl'
        if model_txt.exists():
            import lightgbm as lgb
            return lgb.Booster(model_file=str(model_txt))
        if model_json.exists() and tree_model == 'cb':
            from catboost import CatBoostRanker, CatBoostRegressor
            model = CatBoostRanker() if tree_task == 'ranking' else CatBoostRegressor()
            model.load_model(str(model_json))
            return model
        if model_json.exists() and tree_model == 'xgb':
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(str(model_json))
            return model
        if model_pkl.exists():
            import pickle
            with open(model_pkl, 'rb') as pf:
                return pickle.load(pf)
        raise FileNotFoundError(f'No model file found in {model_dir}')

    def _predict_tree_scores_for_eval(tree_model, model, X_df, feat_names):
        if tree_model == 'lgb':
            return model.predict(X_df)
        if tree_model == 'cb':
            return model.predict(X_df)
        if tree_model == 'xgb':
            import xgboost as xgb
            return model.predict(xgb.DMatrix(X_df.values, feature_names=feat_names))
        raise ValueError(f'Unsupported tree model type: {tree_model}')

    def _select_best_idx(score_values):
        score_values = np.asarray(score_values)
        return int(np.argmax(score_values) if tree_task == 'ranking' else np.argmin(score_values))

    def _build_eval_rows(pred_map):
        rows = []
        for uid in sorted(pred_map.keys()):
            m = meta.get(uid, {})
            rows.append({
                'utterance_id': uid,
                'pred': pred_map[uid],
                'label': gold.get(uid, ''),
                'child_id': m.get('child_id', ''),
                'session_id': m.get('session_id', ''),
                'audio_path': m.get('audio_path', ''),
                'audio_duration_sec': m.get('audio_duration_sec', ''),
                'age_bucket': m.get('age_bucket', ''),
                'source': m.get('source', ''),
            })
        return rows

    # Control experiment: average all inner-fold tree models on the full fold0
    # candidate table. This intentionally leaks validation fold information and
    # is only used as a comparison target against the OOF-routed score above.
    X_eval = pd.DataFrame(df[feat_cols_clean].values, columns=feat_cols_clean)
    all_eval_score_arrays = []
    all_eval_score_names = []
    for tm, fold_models in all_fold_models.items():
        for fi, fm in enumerate(fold_models):
            model = _load_tree_model_for_eval(tm, fm)
            scores = np.asarray(
                _predict_tree_scores_for_eval(tm, model, X_eval, feat_cols_clean),
                dtype=np.float64,
            )
            all_eval_score_arrays.append(scores)
            all_eval_score_names.append(f'{tm}_fold{fi}')
    assert all_eval_score_arrays, 'No saved tree models available for full-fold evaluation'
    all_eval_score_arrays = np.stack(all_eval_score_arrays, axis=0)
    fullavg_scores = all_eval_score_arrays.mean(axis=0)
    df['pred_score_fullavg'] = fullavg_scores

    fullavg_preds = {}
    if tree_task == 'ranking':
        for uid, group in df.groupby('uid'):
            best_row = group.loc[group['pred_score_fullavg'].idxmax()]
            fullavg_preds[uid] = best_row['candidate_text']
    else:
        for uid, group in df.groupby('uid'):
            best_row = group.loc[group['pred_score_fullavg'].idxmin()]
            fullavg_preds[uid] = best_row['candidate_text']

    vote_preds = {}
    borda_preds = {}
    strategy_rows = []
    for uid, group in df.groupby('uid', sort=True):
        row_idx = group.index.to_numpy(dtype=np.int64)
        score_mat = all_eval_score_arrays[:, row_idx]
        n_models_here, n_candidates = score_mat.shape

        per_model_best_local_idx = np.array([_select_best_idx(score_mat[i]) for i in range(n_models_here)], dtype=np.int64)
        vote_counts = np.bincount(per_model_best_local_idx, minlength=n_candidates)
        if tree_task == 'ranking':
            rank_order = np.argsort(-score_mat, axis=1)
        else:
            rank_order = np.argsort(score_mat, axis=1)
        rank_points = np.zeros(n_candidates, dtype=np.float64)
        rank_sum = np.zeros(n_candidates, dtype=np.float64)
        for model_i in range(n_models_here):
            order = rank_order[model_i]
            rank_sum[order] += np.arange(n_candidates, dtype=np.float64)
            rank_points[order] += (n_candidates - 1 - np.arange(n_candidates, dtype=np.float64))

        mean_scores = fullavg_scores[row_idx]
        if tree_task == 'ranking':
            vote_choice_local = int(np.lexsort((-mean_scores, -rank_points, -vote_counts))[-1])
            borda_choice_local = int(np.lexsort((-mean_scores, -rank_points))[-1])
        else:
            vote_choice_local = int(np.lexsort((mean_scores, rank_sum, -vote_counts))[0])
            borda_choice_local = int(np.lexsort((mean_scores, rank_sum))[0])

        vote_best_row = group.iloc[vote_choice_local]
        borda_best_row = group.iloc[borda_choice_local]
        vote_preds[uid] = vote_best_row['candidate_text']
        borda_preds[uid] = borda_best_row['candidate_text']

        oracle_local_idx = int(group['target_cer'].values.argmin())
        sorted_vote = np.sort(vote_counts)
        vote_margin = int(sorted_vote[-1] - sorted_vote[-2]) if len(sorted_vote) >= 2 else int(sorted_vote[-1])
        strategy_rows.append({
            'uid': uid,
            'label': gold.get(uid, ''),
            'oof_pred': ensemble_preds.get(uid, ''),
            'fullavg_pred': fullavg_preds.get(uid, ''),
            'vote_pred': vote_preds.get(uid, ''),
            'borda_pred': borda_preds.get(uid, ''),
            'oracle_pred': group.iloc[oracle_local_idx]['candidate_text'],
            'oof_cer': float(_char_cer(gold.get(uid, ''), ensemble_preds.get(uid, ''))),
            'fullavg_cer': float(_char_cer(gold.get(uid, ''), fullavg_preds.get(uid, ''))),
            'vote_cer': float(_char_cer(gold.get(uid, ''), vote_preds.get(uid, ''))),
            'borda_cer': float(_char_cer(gold.get(uid, ''), borda_preds.get(uid, ''))),
            'oracle_cer': float(group.iloc[oracle_local_idx]['target_cer']),
            'n_candidates': int(n_candidates),
            'n_unique_fold_winners': int(len(set(per_model_best_local_idx.tolist()))),
            'vote_winner_count': int(vote_counts[vote_choice_local]),
            'vote_margin': vote_margin,
            'oof_eq_fullavg': int(ensemble_preds.get(uid, '') == fullavg_preds.get(uid, '')),
            'oof_eq_vote': int(ensemble_preds.get(uid, '') == vote_preds.get(uid, '')),
            'oof_eq_borda': int(ensemble_preds.get(uid, '') == borda_preds.get(uid, '')),
            'fullavg_eq_vote': int(fullavg_preds.get(uid, '') == vote_preds.get(uid, '')),
            'fullavg_eq_borda': int(fullavg_preds.get(uid, '') == borda_preds.get(uid, '')),
            'vote_eq_borda': int(vote_preds.get(uid, '') == borda_preds.get(uid, '')),
            'source': meta.get(uid, {}).get('source', ''),
            'age_bucket': meta.get(uid, {}).get('age_bucket', ''),
            'audio_duration_sec': meta.get(uid, {}).get('audio_duration_sec', np.nan),
        })

    # Step 6: Evaluate
    result_tree = _evaluate(ensemble_preds, gold, meta,
                            label=f'Tree Reranker ({tm_label}, {n_folds}-fold)', verbose=verbose)
    result_tree_target_mix = _compute_source_mix_score(
        result_tree.get('source_results', {}),
        dd_weight=tree_dd_weight,
        ext_weight=tree_ext_weight,
    )
    result_avg = None
    result_avg_target_mix = np.nan
    if has_ctc_score_baseline:
        result_avg = _evaluate(avg_score_preds, gold, meta,
                               label='Avg CTC Score (baseline)', verbose=verbose)
        result_avg_target_mix = _compute_source_mix_score(
            result_avg.get('source_results', {}),
            dd_weight=tree_dd_weight,
            ext_weight=tree_ext_weight,
        )
    elif verbose:
        print('\n--- Avg CTC Score (baseline) Results ---')
        print('Skipped: no model with ctc_logprobs.pt, so no CTC-score baseline is available.')
    result_fullavg = _evaluate(
        fullavg_preds,
        gold,
        meta,
        label=f'Tree Reranker FullAvg ({tm_label}, {n_folds}-fold models)',
        verbose=verbose,
    )
    result_fullavg_target_mix = _compute_source_mix_score(
        result_fullavg.get('source_results', {}),
        dd_weight=tree_dd_weight,
        ext_weight=tree_ext_weight,
    )
    result_vote = _evaluate(
        vote_preds,
        gold,
        meta,
        label=f'Tree Reranker Vote ({tm_label}, {n_folds}-fold models)',
        verbose=verbose,
    )
    result_vote_target_mix = _compute_source_mix_score(
        result_vote.get('source_results', {}),
        dd_weight=tree_dd_weight,
        ext_weight=tree_ext_weight,
    )
    result_borda = _evaluate(
        borda_preds,
        gold,
        meta,
        label=f'Tree Reranker Borda ({tm_label}, {n_folds}-fold models)',
        verbose=verbose,
    )
    result_borda_target_mix = _compute_source_mix_score(
        result_borda.get('source_results', {}),
        dd_weight=tree_dd_weight,
        ext_weight=tree_ext_weight,
    )

    oracle_preds = {}
    for uid, group in df.groupby('uid'):
        best_row = group.loc[group['target_cer'].idxmin()]
        oracle_preds[uid] = best_row['candidate_text']
    result_oracle = _evaluate(oracle_preds, gold, meta,
                              label='Oracle (best candidate)', verbose=verbose)
    result_oracle_target_mix = _compute_source_mix_score(
        result_oracle.get('source_results', {}),
        dd_weight=tree_dd_weight,
        ext_weight=tree_ext_weight,
    )

    # ---- Step 7: Save tree models, eval.csv, metrics.csv ----
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save tree models (one per tree_model type, using first fold for single-fold inference)
    import json as _json
    for tm, fold_models in all_fold_models.items():
        for fi, fm in enumerate(fold_models):
            model_save_dir = save_dir / f'tree_{tm}_fold{fi}'
            if isinstance(fm, Path):
                assert fm == model_save_dir, f'Unexpected saved model dir: {fm} != {model_save_dir}'
            else:
                model_save_dir.mkdir(parents=True, exist_ok=True)
                fm.save(str(model_save_dir))
            if verbose and fi == 0:
                saved_files = list(model_save_dir.iterdir())
                print(f'  Saved {tm} fold {fi}: {[f.name for f in saved_files]}')

    # Save eval.csv (per-utterance predictions, same format as single models)
    eval_df = pd.DataFrame(_build_eval_rows(ensemble_preds))
    eval_csv_path = save_dir / 'eval.csv'
    eval_df.to_csv(eval_csv_path, index=False)
    if verbose:
        print(f'  Saved eval.csv: {eval_csv_path} ({len(eval_df)} rows)')

    eval_fullavg_df = pd.DataFrame(_build_eval_rows(fullavg_preds))
    eval_fullavg_csv_path = save_dir / 'eval_fullavg.csv'
    eval_fullavg_df.to_csv(eval_fullavg_csv_path, index=False)
    if verbose:
        print(f'  Saved eval_fullavg.csv: {eval_fullavg_csv_path} ({len(eval_fullavg_df)} rows)')

    eval_vote_df = pd.DataFrame(_build_eval_rows(vote_preds))
    eval_vote_csv_path = save_dir / 'eval_vote.csv'
    eval_vote_df.to_csv(eval_vote_csv_path, index=False)
    if verbose:
        print(f'  Saved eval_vote.csv: {eval_vote_csv_path} ({len(eval_vote_df)} rows)')

    eval_borda_df = pd.DataFrame(_build_eval_rows(borda_preds))
    eval_borda_csv_path = save_dir / 'eval_borda.csv'
    eval_borda_df.to_csv(eval_borda_csv_path, index=False)
    if verbose:
        print(f'  Saved eval_borda.csv: {eval_borda_csv_path} ({len(eval_borda_df)} rows)')

    strategy_analysis_df = pd.DataFrame(strategy_rows)
    strategy_analysis_csv_path = save_dir / 'strategy_case_analysis.csv'
    strategy_analysis_df.to_csv(strategy_analysis_csv_path, index=False)
    if verbose:
        print(f'  Saved strategy_case_analysis.csv: {strategy_analysis_csv_path} ({len(strategy_analysis_df)} rows)')

    # Save metrics.csv (score breakdown, similar to single models)
    metrics_row = {
        'score': result_tree['overall_cer'],
        'n_samples': result_tree['n_samples'],
        'score/target_mix': result_tree_target_mix,
    }
    for src, cer in result_tree.get('source_results', {}).items():
        metrics_row[f'score/{src}'] = cer
    metrics_row['baseline_cer'] = result_avg['overall_cer'] if result_avg is not None else np.nan
    metrics_row['baseline_cer/target_mix'] = result_avg_target_mix
    metrics_row['fullavg_cer'] = result_fullavg['overall_cer']
    metrics_row['fullavg_cer/target_mix'] = result_fullavg_target_mix
    metrics_row['vote_cer'] = result_vote['overall_cer']
    metrics_row['vote_cer/target_mix'] = result_vote_target_mix
    metrics_row['borda_cer'] = result_borda['overall_cer']
    metrics_row['borda_cer/target_mix'] = result_borda_target_mix
    metrics_row['oracle_cer'] = result_oracle['overall_cer']
    metrics_row['oracle_cer/target_mix'] = result_oracle_target_mix
    metrics_row['tree_dd_weight'] = float(tree_dd_weight)
    metrics_row['tree_ext_weight'] = float(tree_ext_weight)
    metrics_row['tree_model'] = tm_label
    metrics_row['n_features'] = len(feat_cols_clean)
    metrics_row['n_folds'] = n_folds
    metrics_df = pd.DataFrame([metrics_row])
    metrics_csv_path = save_dir / 'metrics.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    if verbose:
        print(f'  Saved metrics.csv: {metrics_csv_path}')

    importance_rows = []
    importance_text = ''
    for tm in tree_models:
        payload = all_feature_importances.get(tm)
        if not payload or not payload.get('summary'):
            continue
        importance_text += _format_tree_feature_importance_text(tm, payload, top_k=30)
        for rank, row in enumerate(payload['summary'], start=1):
            importance_rows.append({
                'tree_model': tm,
                'rank': rank,
                **row,
            })

    feat_manifest_path = save_dir / 'reranker_features.txt'
    feat_manifest_path.write_text(feature_manifest + importance_text)
    if verbose:
        print(f'  Saved reranker_features.txt: {feat_manifest_path}')

    importance_csv_path = save_dir / 'reranker_feature_importance.csv'
    if importance_rows:
        pd.DataFrame(importance_rows).to_csv(importance_csv_path, index=False)
        if verbose:
            print(f'  Saved reranker_feature_importance.csv: {importance_csv_path}')

    # Save reranker metadata (for inference: model_names, features, config)
    reranker_meta = {
        'model_names': model_names,
        'model_max_dur': _model_max_dur or {},
        'model_audio_filters': {
            mn: {'max_audio_sec': float(max_dur)}
            for mn, max_dur in (_model_max_dur or {}).items()
        },
        'tree_models': tree_models,
        'feat_cols': feat_cols_clean,
        'tree_task': tree_task,
        'tree_obj': tree_obj,
        'relevance_levels': relevance_levels,
        'relevance_strategy': relevance_strategy,
        'nbest': nbest,
        'beam_width': beam_width,
        'n_folds': n_folds,
        'no_lm_feats': no_lm_feats,
        'drop_feats': drop_feats,
        'feat_flags': feature_flags,
        'no_ctc_score_feats': bool(no_ctc_score_feats),
        'feat_mbr': bool(feat_mbr),
        'feat_tdt': bool(use_tdt_feats),
        'feat_tdt_light': bool(feat_tdt_light),
        'feat_tdt_exact': bool(feat_tdt_exact),
        'feat_tdt_group': bool(feat_tdt_group),
        'feat_wavlm_group': bool(feat_wavlm_group),
        'feat_nemo_group': bool(feat_nemo_group),
        'tdt_eval_nbest': int(tdt_eval_nbest or 0),
        'tdt_feat_topk': int(tdt_feat_topk),
        'tdt_force_keep_preds': bool(tdt_force_keep_preds),
        'cache_tdt_exact_scores': bool(cache_tdt_exact_scores),
        'tdt_model_names': tdt_model_names,
        'wavlm_model_names': wavlm_model_names,
        'nemo_model_names': nemo_model_names,
        'tree_dd_weight': float(tree_dd_weight),
        'tree_ext_weight': float(tree_ext_weight),
        'lm_path': FLAGS.ensemble_lm_path or '',
        'word_lm_path': FLAGS.ensemble_word_lm_path or '',
        'lm_per_word_only': bool(lm_per_word_only),
        'split_seed': int(seed),
    }
    meta_path = save_dir / 'reranker_meta.json'
    with open(meta_path, 'w') as f:
        _json.dump(reranker_meta, f, indent=2)
    if verbose:
        print(f'  Saved reranker_meta.json: {meta_path}')

    exp_record = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'exp_name': exp_name or '',
        'fold': FOLD,
        'save_dir': str(save_dir),
        'score': result_tree['overall_cer'],
        'score/dd': result_tree.get('source_results', {}).get('dd', np.nan),
        'score/ext': result_tree.get('source_results', {}).get('ext', np.nan),
        'score/target_mix': result_tree_target_mix,
        'baseline_cer': result_avg['overall_cer'] if result_avg is not None else np.nan,
        'baseline_cer/target_mix': result_avg_target_mix,
        'oracle_cer': result_oracle['overall_cer'],
        'oracle_cer/target_mix': result_oracle_target_mix,
        'gain_vs_baseline': ((result_avg['overall_cer'] - result_tree['overall_cer']) if result_avg is not None else np.nan),
        'gain_vs_baseline_target_mix': (result_avg_target_mix - result_tree_target_mix if result_avg is not None else np.nan),
        'tree_dd_weight': float(tree_dd_weight),
        'tree_ext_weight': float(tree_ext_weight),
        'tree_model': tm_label,
        'tree_task': tree_task,
        'n_features': len(feat_cols_clean),
        'n_folds': n_folds,
        'nbest': nbest,
        'beam_width': beam_width,
        'relevance_strategy': relevance_strategy,
        'no_lm_feats': no_lm_feats,
        'drop_feats': drop_feats,
        'cache_tdt_exact_scores': bool(cache_tdt_exact_scores),
        'model_names': '|'.join(model_names),
        'feat_flags_json': _json.dumps(feature_flags, sort_keys=True),
        'command': ' '.join(sys.argv),
        'notes': exp_notes or '',
    }
    exp_json_path = save_dir / 'reranker_experiment.json'
    with open(exp_json_path, 'w') as f:
        _json.dump(exp_record, f, indent=2)
    log_path = _append_tree_experiment_log(WORKING_BASE / 'ensemble-experiments', exp_record)
    if verbose:
        print(f'  Saved reranker_experiment.json: {exp_json_path}')
        print(f'  Updated experiment_log.csv: {log_path}')

    if return_details:
        return {
            'result': result_tree,
            'predictions': ensemble_preds,
            'gold': gold,
            'meta': meta,
        }

    return result_tree


# ===========================================================================
#  N-best Rescore Ensemble
# ===========================================================================

def nbest_rescore_ensemble(model_names, nbest=10, beam_width=10, lm=None, lm_weight=0.0,
                           verbose=True, return_details=False, folds=None):
    """Pure CTC exact rescoring over a pooled candidate set.

    Algorithm:
      1. For each model, collect candidate hypotheses from any available source:
         CTC beam N-best, eval.csv primary prediction, and dual-head sidecars.
      2. Pool all unique candidates across models.
      3. For each candidate, compute exact CTC forward scores under every model
         that has saved CTC logprobs.
      4. Select the candidate with the highest weighted average CTC score.

    TDT-capable models can still contribute candidates through eval.csv or
    dual-head sidecars, but this mode does not run TDT second-pass rescoring.
    Use nbest_rescore2 for CTC prune + exact TDT rescoring.
    """
    import torch

    ctc_score_weight = float(getattr(FLAGS, 'ensemble_ctc_score_weight', 1.0) or 0.0)
    tdt_score_weight = 0.0
    assert ctc_score_weight > 0, 'nbest_rescore requires ensemble_ctc_score_weight > 0'

    use_lm = lm is not None and lm_weight > 0
    if verbose:
        lm_label = f', lm_weight={lm_weight}' if use_lm else ''
        print(f'\nN-best Rescore ensemble (nbest={nbest}, beam={beam_width}{lm_label}, '
              f'ctc_w={ctc_score_weight}, pure_ctc_rescore=True) '
              f'with {len(model_names)} models:')
        for mn in model_names:
            print(f'  - {mn}')

    all_logprobs = {}
    for mn in model_names:
        lp_path = get_model_dir(mn) / 'ctc_logprobs.pt'
        if lp_path.exists():
            if not _HAS_BEAM_SEARCH:
                raise RuntimeError('CTC beam search not available (import failed)')
            all_logprobs[mn] = torch.load(str(lp_path), map_location='cpu', weights_only=False)
            if verbose:
                print(f'  Loaded {len(all_logprobs[mn])} CTC logprob utterances from {mn}')
        elif verbose:
            print(f'  No CTC logprobs for {mn}, will use eval/dual candidates only')

    model_ctc_meta = _get_model_ctc_meta(model_names, all_logprobs)

    all_eval_preds = {}
    all_eval_nbest_texts = {}
    uid_sets = []
    eval_uid_infos = {}
    gold = {}
    meta = {}
    tdt_eval_nbest = int(getattr(FLAGS, 'ensemble_tdt_eval_nbest', 0) or 0)
    for mi, mn in enumerate(model_names):
        eval_csv = get_eval_csv(get_model_dir(mn))
        assert eval_csv.exists(), f'{eval_csv} not found'
        df = pd.read_csv(eval_csv)
        eval_uid_infos[mn] = {
            'uids': set(df['utterance_id'].astype(str)),
            'n_rows': int(len(df)),
            'n_uids': int(df['utterance_id'].nunique()),
            'source_counts': df['source'].fillna('').astype(str).value_counts().to_dict() if 'source' in df.columns else {},
        }
        preds = {}
        nbest_texts = {}
        for _, row in df.iterrows():
            uid = row['utterance_id']
            preds[uid] = _normalize_candidate_text(row.get('pred', ''))
            if tdt_eval_nbest > 0 and 'pred_nbest_texts' in df.columns:
                parsed = _parse_serialized_text_list(row.get('pred_nbest_texts'))
                if parsed:
                    nbest_texts[uid] = parsed[:tdt_eval_nbest]
            if mi == 0:
                gold[uid] = str(row['label']) if pd.notna(row['label']) else ''
                meta[uid] = {
                    'source': row.get('source', ''),
                    'age_bucket': row.get('age_bucket', ''),
                }
        all_eval_preds[mn] = preds
        all_eval_nbest_texts[mn] = nbest_texts
        uid_sets.append(set(preds.keys()))

    all_dual_head_preds = {}
    for mn in model_names:
        dual_path = get_model_dir(mn) / 'dual_head_preds.pt'
        if dual_path.exists():
            dual_data = torch.load(str(dual_path), map_location='cpu', weights_only=False)
            all_dual_head_preds[mn] = dual_data.get('preds', {})
            if verbose:
                print(f'  Loaded {len(all_dual_head_preds[mn])} dual-head entries from {mn}')

    common_uids = _validate_eval_uid_sets(
        eval_uid_infos,
        allow_mismatch=bool(getattr(FLAGS, 'ensemble_allow_eval_set_mismatch', False)),
        verbose=verbose,
    )
    if verbose:
        print(f'Common utterances: {len(common_uids)}')

    blank_id = IPA_CTC_BLANK if IPA_CTC_BLANK is not None else 0
    id_to_char = IPA_ID_TO_CHAR
    char_to_id = None
    if id_to_char is not None:
        char_to_id = {ch: cid for cid, ch in id_to_char.items()}

    ensemble_preds = {}
    n_cands_total = 0
    n_unique_total = 0

    uids_list = sorted(common_uids)
    total = len(uids_list)
    t0 = time.time()

    n_workers = max(0, int(getattr(FLAGS, 'ensemble_n_workers', 0) or 0))
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 16)
    can_parallel = n_workers > 1 and total >= 512
    progress_desc = '  N-best rescoring'
    build_desc = '  Build candidates'
    skip_ctc_candidate_models = {mn: _should_skip_tdt_primary_ctc_candidates(mn) for mn in model_names}

    if verbose:
        skipped = [mn for mn, skip in skip_ctc_candidate_models.items() if skip]
        if skipped:
            print('  Skip CTC candidates for TDT-primary models:')
            for mn in skipped:
                print(f'    - {mn}')
        if tdt_eval_nbest > 0:
            print(f'  Add eval TDT nbest candidates: topk={tdt_eval_nbest}')

    global _NBEST_MP_BUILD
    if can_parallel:
        _NBEST_MP_BUILD = {
            'model_names': model_names,
            'all_logprobs': all_logprobs,
            'all_eval_preds': all_eval_preds,
            'all_eval_nbest_texts': all_eval_nbest_texts,
            'all_dual_head_preds': all_dual_head_preds,
            'skip_ctc_candidate_models': skip_ctc_candidate_models,
            'blank_id': blank_id,
            'beam_width': beam_width,
            'nbest': nbest,
            'id_to_char': id_to_char,
            'char_to_id': char_to_id,
            'lm': lm,
            'lm_weight': lm_weight,
            'use_lm': use_lm,
            'tdt_eval_nbest': tdt_eval_nbest,
            'prefix_beam_search_nbest': prefix_beam_search_nbest,
            'ctc_force_score_batch': ctc_force_score_batch,
            'model_ctc_meta': model_ctc_meta,
        }

        import multiprocessing as mp
        ctx = mp.get_context('fork')
        chunksize = max(4, min(32, total // max(n_workers * 16, 1)))
        if verbose:
            print(f'  Parallel nbest_rescore: workers={n_workers}, chunksize={chunksize}')

        candidate_lists = {}
        candidate_token_ids_by_uid = {}
        with ctx.Pool(n_workers, initializer=_nbest_pool_init) as pool:
            for uid, candidates, raw_count, unique_count, candidate_token_ids in _iter_with_progress(
                    pool.imap_unordered(_build_nbest_candidates_single_uid, uids_list, chunksize=chunksize),
                    total=total, desc=build_desc, verbose=verbose):
                candidate_lists[uid] = candidates
                candidate_token_ids_by_uid[uid] = candidate_token_ids
                n_cands_total += raw_count
                n_unique_total += unique_count
    else:
        candidate_lists = {}
        candidate_token_ids_by_uid = {}
        _NBEST_MP_BUILD = {
            'model_names': model_names,
            'all_logprobs': all_logprobs,
            'all_eval_preds': all_eval_preds,
            'all_eval_nbest_texts': all_eval_nbest_texts,
            'all_dual_head_preds': all_dual_head_preds,
            'skip_ctc_candidate_models': skip_ctc_candidate_models,
            'blank_id': blank_id,
            'beam_width': beam_width,
            'nbest': nbest,
            'id_to_char': id_to_char,
            'char_to_id': char_to_id,
            'lm': lm,
            'lm_weight': lm_weight,
            'use_lm': use_lm,
            'tdt_eval_nbest': tdt_eval_nbest,
            'prefix_beam_search_nbest': prefix_beam_search_nbest,
            'ctc_force_score_batch': ctc_force_score_batch,
            'model_ctc_meta': model_ctc_meta,
        }
        for uid in _iter_with_progress(uids_list, total=total, desc=build_desc, verbose=verbose):
            uid, candidates, raw_count, unique_count, candidate_token_ids = _build_nbest_candidates_single_uid(uid)
            candidate_lists[uid] = candidates
            candidate_token_ids_by_uid[uid] = candidate_token_ids
            n_cands_total += raw_count
            n_unique_total += unique_count

    _NBEST_MP_BUILD = {
        'all_logprobs': all_logprobs,
        'all_tdt_scores': {},
        'blank_id': blank_id,
        'char_to_id': char_to_id,
        'lm': lm,
        'lm_weight': lm_weight,
        'use_lm': use_lm,
        'ctc_force_score_batch': ctc_force_score_batch,
        'ctc_score_weight': ctc_score_weight,
        'tdt_score_weight': tdt_score_weight,
        'candidate_lists': candidate_lists,
        'candidate_token_ids_by_model': candidate_token_ids_by_uid,
        'model_ctc_meta': model_ctc_meta,
    }

    if can_parallel:
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        chunksize = max(4, min(32, total // max(n_workers * 16, 1)))
        with ctx.Pool(n_workers, initializer=_nbest_pool_init) as pool:
            for uid, pred in _iter_with_progress(
                    pool.imap_unordered(_nbest_rescore_single_uid, uids_list, chunksize=chunksize),
                    total=total, desc=progress_desc, verbose=verbose):
                ensemble_preds[uid] = pred
    else:
        for uid in _iter_with_progress(uids_list, total=total, desc=progress_desc, verbose=verbose):
            uid, pred = _nbest_rescore_single_uid(uid)
            ensemble_preds[uid] = pred

    _NBEST_MP_BUILD = {}

    if verbose:
        elapsed = time.time() - t0
        print(f'  Processed {total}/{total} in {elapsed:.1f}s ({total/elapsed:.1f} utt/s)')
        avg_cands = n_cands_total / max(len(common_uids), 1)
        avg_unique = n_unique_total / max(len(common_uids), 1)
        print(f'Avg candidates per utterance: {avg_cands:.1f} raw, {avg_unique:.1f} unique')
        if not all_logprobs:
            print('  NOTE: no selected model has ctc_logprobs.pt, so nbest_rescore '
                  'did not do exact CTC rescoring; results come directly from existing '
                  'eval/best_eval predictions and any dual-head candidate sidecars.')

    result = _evaluate(ensemble_preds, gold, meta,
                       label=f'N-best Rescore (n={nbest}, beam={beam_width})', verbose=verbose)

    if verbose:
        _print_individual_model_scores(model_names, folds=folds)

    if return_details:
        return {
            'result': result,
            'predictions': ensemble_preds,
            'gold': gold,
            'meta': meta,
        }

    return result


def nbest_rescore2_ensemble(model_names, nbest=10, beam_width=10, lm=None, lm_weight=0.0,
                            verbose=True, return_details=False, folds=None):
    """Two-stage n-best rescoring.

    Stage 1: build a unified candidate pool from all sources and compute exact
    CTC scores once. Keep top-K by aggregated CTC score, while forcing raw TDT
    model outputs to survive into stage 2.

    Stage 2: score the reduced candidate pool with all TDT-capable models and
    fuse cached CTC scores + TDT scores for the final choice.
    """
    import torch

    ctc_score_weight = float(getattr(FLAGS, 'ensemble_ctc_score_weight', 1.0) or 0.0)
    tdt_score_weight = float(getattr(FLAGS, 'ensemble_tdt_score_weight', 1.0) or 0.0)
    prune_topk = int(getattr(FLAGS, 'ensemble_ctc_prune_topk', 20) or 20)
    force_keep_tdt_preds = bool(getattr(FLAGS, 'ensemble_tdt_force_keep_preds', True))
    assert ctc_score_weight > 0 or tdt_score_weight > 0, 'At least one of CTC/TDT score weights must be > 0'
    assert prune_topk > 0, f'ensemble_ctc_prune_topk must be > 0, got {prune_topk}'

    use_lm = lm is not None and lm_weight > 0
    if verbose:
        lm_label = f', lm_weight={lm_weight}' if use_lm else ''
        print(f'\nN-best Rescore2 ensemble (nbest={nbest}, beam={beam_width}{lm_label}, '
              f'ctc_topk={prune_topk}, force_keep_tdt={force_keep_tdt_preds}, '
              f'ctc_w={ctc_score_weight}, tdt_w={tdt_score_weight}) '
              f'with {len(model_names)} models:')
        for mn in model_names:
            print(f'  - {mn}')

    all_logprobs = {}
    for mn in model_names:
        lp_path = get_model_dir(mn) / 'ctc_logprobs.pt'
        if lp_path.exists():
            if not _HAS_BEAM_SEARCH:
                raise RuntimeError('CTC beam search not available (import failed)')
            all_logprobs[mn] = torch.load(str(lp_path), map_location='cpu', weights_only=False)
            if verbose:
                print(f'  Loaded {len(all_logprobs[mn])} CTC logprob utterances from {mn}')
        elif verbose:
            print(f'  No CTC logprobs for {mn}, will use eval/dual candidates only')

    model_ctc_meta = _get_model_ctc_meta(model_names, all_logprobs)

    all_eval_preds = {}
    all_eval_nbest_texts = {}
    uid_sets = []
    eval_uid_infos = {}
    gold = {}
    meta = {}
    tdt_eval_nbest = int(getattr(FLAGS, 'ensemble_tdt_eval_nbest', 0) or 0)
    for mi, mn in enumerate(model_names):
        eval_csv = get_eval_csv(get_model_dir(mn))
        assert eval_csv.exists(), f'{eval_csv} not found'
        df = pd.read_csv(eval_csv)
        eval_uid_infos[mn] = {
            'uids': set(df['utterance_id'].astype(str)),
            'n_rows': int(len(df)),
            'n_uids': int(df['utterance_id'].nunique()),
            'source_counts': df['source'].fillna('').astype(str).value_counts().to_dict() if 'source' in df.columns else {},
        }
        preds = {}
        nbest_texts = {}
        for _, row in df.iterrows():
            uid = row['utterance_id']
            preds[uid] = _normalize_candidate_text(row.get('pred', ''))
            if tdt_eval_nbest > 0 and 'pred_nbest_texts' in df.columns:
                parsed = _parse_serialized_text_list(row.get('pred_nbest_texts'))
                if parsed:
                    nbest_texts[uid] = parsed[:tdt_eval_nbest]
            if mi == 0:
                gold[uid] = str(row['label']) if pd.notna(row['label']) else ''
                meta[uid] = {
                    'source': row.get('source', ''),
                    'age_bucket': row.get('age_bucket', ''),
                }
        all_eval_preds[mn] = preds
        all_eval_nbest_texts[mn] = nbest_texts
        uid_sets.append(set(preds.keys()))

    all_dual_head_preds = {}
    for mn in model_names:
        dual_path = get_model_dir(mn) / 'dual_head_preds.pt'
        if dual_path.exists():
            dual_data = torch.load(str(dual_path), map_location='cpu', weights_only=False)
            all_dual_head_preds[mn] = dual_data.get('preds', {})
            if verbose:
                print(f'  Loaded {len(all_dual_head_preds[mn])} dual-head entries from {mn}')

    common_uids = _validate_eval_uid_sets(
        eval_uid_infos,
        allow_mismatch=bool(getattr(FLAGS, 'ensemble_allow_eval_set_mismatch', False)),
        verbose=verbose,
    )
    if verbose:
        print(f'Common utterances: {len(common_uids)}')

    blank_id = IPA_CTC_BLANK if IPA_CTC_BLANK is not None else 0
    id_to_char = IPA_ID_TO_CHAR
    char_to_id = {ch: cid for cid, ch in id_to_char.items()} if id_to_char is not None else None

    ensemble_preds = {}
    n_cands_total = 0
    n_unique_total = 0
    n_pruned_total = 0

    uids_list = sorted(common_uids)
    total = len(uids_list)
    t0 = time.time()

    n_workers = max(0, int(getattr(FLAGS, 'ensemble_n_workers', 0) or 0))
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 16)
    can_parallel = n_workers > 1 and total >= 512
    build_desc = '  Build candidates'
    progress_desc = '  N-best rescoring2'
    skip_ctc_candidate_models = {mn: _should_skip_tdt_primary_ctc_candidates(mn) for mn in model_names}

    if verbose:
        skipped = [mn for mn, skip in skip_ctc_candidate_models.items() if skip]
        if skipped:
            print('  Skip CTC candidates for TDT-primary models:')
            for mn in skipped:
                print(f'    - {mn}')
        if tdt_eval_nbest > 0:
            print(f'  Add eval TDT nbest candidates: topk={tdt_eval_nbest}')

    if can_parallel:
        global _NBEST_MP_BUILD
        _NBEST_MP_BUILD = {
            'model_names': model_names,
            'all_logprobs': all_logprobs,
            'all_eval_preds': all_eval_preds,
            'all_eval_nbest_texts': all_eval_nbest_texts,
            'all_dual_head_preds': all_dual_head_preds,
            'skip_ctc_candidate_models': skip_ctc_candidate_models,
            'blank_id': blank_id,
            'beam_width': beam_width,
            'nbest': nbest,
            'id_to_char': id_to_char,
            'char_to_id': char_to_id,
            'lm': lm,
            'lm_weight': lm_weight,
            'use_lm': use_lm,
            'tdt_eval_nbest': tdt_eval_nbest,
            'prefix_beam_search_nbest': prefix_beam_search_nbest,
            'ctc_force_score_batch': ctc_force_score_batch,
        }
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        chunksize = max(4, min(32, total // max(n_workers * 16, 1)))
        if verbose:
            print(f'  Parallel nbest_rescore2: workers={n_workers}, chunksize={chunksize}')
        candidate_lists = {}
        candidate_token_ids_by_uid = {}
        with ctx.Pool(n_workers, initializer=_nbest_pool_init) as pool:
            for uid, candidates, raw_count, unique_count, candidate_token_ids in _iter_with_progress(
                    pool.imap_unordered(_build_nbest_candidates_single_uid, uids_list, chunksize=chunksize),
                    total=total, desc=build_desc, verbose=verbose):
                candidate_lists[uid] = candidates
                candidate_token_ids_by_uid[uid] = candidate_token_ids
                n_cands_total += raw_count
                n_unique_total += unique_count
    else:
        candidate_lists = {}
        candidate_token_ids_by_uid = {}
        _NBEST_MP_BUILD = {
            'model_names': model_names,
            'all_logprobs': all_logprobs,
            'all_eval_preds': all_eval_preds,
            'all_eval_nbest_texts': all_eval_nbest_texts,
            'all_dual_head_preds': all_dual_head_preds,
            'skip_ctc_candidate_models': skip_ctc_candidate_models,
            'blank_id': blank_id,
            'beam_width': beam_width,
            'nbest': nbest,
            'id_to_char': id_to_char,
            'char_to_id': char_to_id,
            'lm': lm,
            'lm_weight': lm_weight,
            'use_lm': use_lm,
            'tdt_eval_nbest': tdt_eval_nbest,
            'prefix_beam_search_nbest': prefix_beam_search_nbest,
            'ctc_force_score_batch': ctc_force_score_batch,
        }
        for uid in _iter_with_progress(uids_list, total=total, desc=build_desc, verbose=verbose):
            uid, candidates, raw_count, unique_count, candidate_token_ids = _build_nbest_candidates_single_uid(uid)
            candidate_lists[uid] = candidates
            candidate_token_ids_by_uid[uid] = candidate_token_ids
            n_cands_total += raw_count
            n_unique_total += unique_count

    tdt_model_names = []
    if tdt_score_weight > 0:
        import gezi as gz
        from gezi import FLAGS as _FLAGS
        from src import config as _shared_config
        for mn in model_names:
            model_dir = get_model_dir(mn)
            if not model_dir.exists():
                continue
            gz.init_flags()
            _shared_config.init()
            gz.restore_configs(str(model_dir))
            ctc_only = bool(getattr(_FLAGS, 'ctc_only', False))
            ctc_weight_cur = float(getattr(_FLAGS, 'ctc_weight', 1.0) or 0.0)
            s2s_decoder = str(getattr(_FLAGS, 's2s_decoder', 'native') or 'native')
            if (not ctc_only) and (ctc_weight_cur < 1.0) and (s2s_decoder == 'tdt_reuse'):
                tdt_model_names.append(mn)
            elif verbose:
                print(f'  TDT disabled for {mn}: ctc_only={ctc_only}, '
                      f'ctc_weight={ctc_weight_cur}, s2s_decoder={s2s_decoder}')

    force_keep_tdt_candidates = {}
    ctc_score_maps = {}
    pruned_candidate_lists = {}
    if verbose:
        print('  Stage1: exact CTC prune over pooled candidates...')
    ctc_prune_threads = int(getattr(FLAGS, 'ensemble_ctc_prune_threads', 0) or 0)
    if ctc_prune_threads > 0:
        import torch as _torch
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from functools import partial
        _worker = partial(_ctc_prune_one_uid,
                          candidate_lists=candidate_lists, all_logprobs=all_logprobs,
                          blank_id=blank_id, char_to_id=char_to_id,
                          ctc_score_weight=ctc_score_weight, prune_topk=prune_topk,
                          force_keep_tdt_preds=force_keep_tdt_preds,
                          tdt_model_names=tdt_model_names,
                          all_eval_preds=all_eval_preds, all_dual_head_preds=all_dual_head_preds,
                          all_eval_nbest_texts=all_eval_nbest_texts,
                          tdt_eval_nbest=tdt_eval_nbest,
                          candidate_token_ids_by_model=candidate_token_ids_by_uid,
                          model_ctc_meta=model_ctc_meta)
        orig_torch_threads = _torch.get_num_threads()
        _torch.set_num_threads(max(1, orig_torch_threads // ctc_prune_threads))
        if verbose:
            print(f'  Parallel CTC prune: {ctc_prune_threads} threads '
                  f'(torch intra-op threads: {orig_torch_threads} -> {max(1, orig_torch_threads // ctc_prune_threads)})')
        try:
            with ThreadPoolExecutor(max_workers=ctc_prune_threads) as executor:
                futures = {executor.submit(_worker, uid): uid for uid in uids_list}
                for future in _iter_with_progress(
                        as_completed(futures), total=total, desc='  CTC prune', verbose=verbose):
                    uid, force_keep, ctc_score_map, pruned = future.result()
                    force_keep_tdt_candidates[uid] = force_keep
                    ctc_score_maps[uid] = ctc_score_map
                    pruned_candidate_lists[uid] = pruned
                    n_pruned_total += len(pruned)
        finally:
            _torch.set_num_threads(orig_torch_threads)
    else:
        for uid in _iter_with_progress(uids_list, total=total, desc='  CTC prune', verbose=verbose):
            uid, force_keep, ctc_score_map, pruned = _ctc_prune_one_uid(
                uid, candidate_lists=candidate_lists, all_logprobs=all_logprobs,
                blank_id=blank_id, char_to_id=char_to_id,
                ctc_score_weight=ctc_score_weight, prune_topk=prune_topk,
                force_keep_tdt_preds=force_keep_tdt_preds,
                tdt_model_names=tdt_model_names,
                all_eval_preds=all_eval_preds, all_dual_head_preds=all_dual_head_preds,
                all_eval_nbest_texts=all_eval_nbest_texts,
                tdt_eval_nbest=tdt_eval_nbest,
                candidate_token_ids_by_model=candidate_token_ids_by_uid,
                model_ctc_meta=model_ctc_meta)
            force_keep_tdt_candidates[uid] = force_keep
            ctc_score_maps[uid] = ctc_score_map
            pruned_candidate_lists[uid] = pruned
            n_pruned_total += len(pruned)

    all_tdt_scores = {}
    if tdt_score_weight > 0:
        if verbose and tdt_model_names:
            print('  Stage2: TDT rescoring on pruned candidates...')
        if verbose and not tdt_model_names:
            print('  No TDT-capable models found; nbest_rescore2 will use CTC prune only')
        for mn in tdt_model_names:
            model_scores = _score_tdt_candidates_for_model(mn, pruned_candidate_lists, verbose=verbose)
            if model_scores:
                all_tdt_scores[mn] = model_scores

    _NBEST_MP_BUILD = {
        'candidate_lists': pruned_candidate_lists,
        'ctc_score_maps': ctc_score_maps,
        'all_tdt_scores': all_tdt_scores,
        'ctc_score_weight': ctc_score_weight,
        'tdt_score_weight': tdt_score_weight,
        'lm': lm,
        'lm_weight': lm_weight,
        'use_lm': use_lm,
    }

    if can_parallel:
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        chunksize = max(4, min(32, total // max(n_workers * 16, 1)))
        with ctx.Pool(n_workers, initializer=_nbest_pool_init) as pool:
            for uid, pred in _iter_with_progress(
                    pool.imap_unordered(_nbest_rescore2_single_uid, uids_list, chunksize=chunksize),
                    total=total, desc=progress_desc, verbose=verbose):
                ensemble_preds[uid] = pred
    else:
        for uid in _iter_with_progress(uids_list, total=total, desc=progress_desc, verbose=verbose):
            uid, pred = _nbest_rescore2_single_uid(uid)
            ensemble_preds[uid] = pred

    _NBEST_MP_BUILD = {}

    if verbose:
        elapsed = time.time() - t0
        print(f'  Processed {total}/{total} in {elapsed:.1f}s ({total/elapsed:.1f} utt/s)')
        avg_cands = n_cands_total / max(len(common_uids), 1)
        avg_unique = n_unique_total / max(len(common_uids), 1)
        avg_pruned = n_pruned_total / max(len(common_uids), 1)
        print(f'Avg candidates per utterance: {avg_cands:.1f} raw, {avg_unique:.1f} unique, {avg_pruned:.1f} after CTC prune')

    result = _evaluate(ensemble_preds, gold, meta,
                       label=f'N-best Rescore2 (n={nbest}, beam={beam_width}, topk={prune_topk})',
                       verbose=verbose)

    if verbose:
        _print_individual_model_scores(model_names, folds=folds)

    if return_details:
        return {
            'result': result,
            'predictions': ensemble_preds,
            'gold': gold,
            'meta': meta,
        }

    return result


# ===========================================================================
#  Sweep
# ===========================================================================

def sweep_text_ensemble(model_names, max_k=None):
    """Try all combinations of models and report MBR ensemble CER."""
    if max_k is None:
        max_k = len(model_names)
    
    results = []
    total_combos = sum(len(list(itertools.combinations(model_names, k)))
                       for k in range(1, max_k + 1))
    print(f'\nSweeping {total_combos} combinations of {len(model_names)} models...')
    t0 = time.time()
    
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(model_names, k):
            r = text_ensemble(list(combo), verbose=False)
            results.append({
                'models': ' + '.join(combo),
                'n_models': k,
                'cer': r['overall_cer'],
                'dd': r['source_results'].get('dd', None),
                'ext': r['source_results'].get('ext', None),
            })
    
    elapsed = time.time() - t0
    results.sort(key=lambda x: x['cer'])
    
    print(f'\n{"="*130}')
    print(f'  Text-level MBR Sweep Results ({elapsed:.1f}s)')
    print(f'{"="*130}')
    print(f'{"Rank":>4} | {"N":>2} | {"CER":>8} | {"DD":>8} | {"EXT":>8} | Models')
    print(f'{"-"*4}-+-{"-"*2}-+-{"-"*8}-+-{"-"*8}-+-{"-"*8}-+{"-"*70}')
    
    for i, r in enumerate(results[:30]):
        dd_str = f'{r["dd"]:.5f}' if r['dd'] is not None else '   N/A'
        ext_str = f'{r["ext"]:.5f}' if r['ext'] is not None else '   N/A'
        print(f'{i+1:>4} | {r["n_models"]:>2} | {r["cer"]:.5f} | {dd_str} | {ext_str} | {r["models"]}')
    
    return results


def sweep_logprob_ensemble(model_names, max_k=None, beam_width=10, temperature=1.0):
    """Sweep logits + prob ensemble combinations."""
    if max_k is None:
        max_k = len(model_names)
    
    results = []
    total_combos = sum(len(list(itertools.combinations(model_names, k)))
                       for k in range(2, max_k + 1))
    print(f'\nSweeping {total_combos} combinations x 2 modes (logits + prob)...')
    t0 = time.time()
    
    # Single models
    for mn in model_names:
        r = logprob_ensemble([mn], mode='logits', beam_width=beam_width, temperature=temperature, verbose=False)
        if r:
            results.append({
                'models': mn,
                'n_models': 1,
                'mode': 'single',
                'cer': r['overall_cer'],
                'dd': r['source_results'].get('dd', None),
                'ext': r['source_results'].get('ext', None),
            })
    
    # Combinations x 2 modes
    for k in range(2, max_k + 1):
        for combo in itertools.combinations(model_names, k):
            for ens_mode in ['logits', 'prob']:
                r = logprob_ensemble(list(combo), mode=ens_mode, beam_width=beam_width, temperature=temperature, verbose=False)
                if r:
                    results.append({
                        'models': ' + '.join(combo),
                        'n_models': k,
                        'mode': ens_mode,
                        'cer': r['overall_cer'],
                        'dd': r['source_results'].get('dd', None),
                        'ext': r['source_results'].get('ext', None),
                    })
    
    elapsed = time.time() - t0
    results.sort(key=lambda x: x['cer'])
    
    print(f'\n{"="*140}')
    print(f'  Logits/Prob Ensemble Sweep Results ({elapsed:.1f}s)')
    print(f'{"="*140}')
    print(f'{"Rank":>4} | {"N":>2} | {"Mode":>6} | {"CER":>8} | {"DD":>8} | {"EXT":>8} | Models')
    print(f'{"-"*4}-+-{"-"*2}-+-{"-"*6}-+-{"-"*8}-+-{"-"*8}-+-{"-"*8}-+{"-"*70}')
    
    for i, r in enumerate(results[:30]):
        dd_str = f'{r["dd"]:.5f}' if r['dd'] is not None else '   N/A'
        ext_str = f'{r["ext"]:.5f}' if r['ext'] is not None else '   N/A'
        print(f'{i+1:>4} | {r["n_models"]:>2} | {r["mode"]:>6} | {r["cer"]:.5f} | {dd_str} | {ext_str} | {r["models"]}')
    
    return results


# ===========================================================================
#  Main
# ===========================================================================

def _load_lm_from_flags():
    aux_lms = {}
    if FLAGS.ensemble_lm_path:
        lm = load_ngram_lm(FLAGS.ensemble_lm_path)
        print(f'Loaded LM from {FLAGS.ensemble_lm_path} (order={lm.order})')
    else:
        lm = None
    if FLAGS.ensemble_word_lm_path:
        word_lm = load_ngram_lm(FLAGS.ensemble_word_lm_path)
        aux_lms['word'] = word_lm
        print(f'Loaded word LM from {FLAGS.ensemble_word_lm_path} (order={word_lm.order})')
    return lm, aux_lms


def _score_lm_candidate(lm, cand_text):
    if lm is None:
        return 0.0
    if hasattr(lm, 'score_text'):
        return float(lm.score_text(cand_text))
    lm_sc = 0.0
    ctx = ''
    for ch in cand_text:
        lm_sc += lm.score(ctx, ch)
        ctx += ch
    lm_sc += lm.score(ctx, '$')
    return float(lm_sc)


def _run_single_mode(mode, model_names, lm=None, aux_lms=None, return_details=False,
                     save_dir=None, verbose=True, folds=None, model_max_dur=None):
    bw = FLAGS.ensemble_beam_width
    temp = FLAGS.ensemble_temperature
    lm_per_word_only = bool(getattr(FLAGS, 'ensemble_lm_per_word_only', False))
    no_lm_feats = FLAGS.ensemble_no_lm_feats and not getattr(FLAGS, 'ensemble_lm_feats', False) and not lm_per_word_only

    if mode == 'text':
        return text_ensemble(model_names, verbose=verbose, return_details=return_details)
    if mode == 'rover':
        return rover_ensemble(model_names, verbose=verbose, return_details=return_details)
    if mode == 'sweep':
        return sweep_text_ensemble(model_names, max_k=FLAGS.ensemble_max_sweep_k)
    if mode == 'save_logits':
        for mn in model_names:
            save_logits_for_model(mn)
        return None
    if mode == 'logits_saved':
        return logprob_ensemble(model_names, mode='logits', beam_width=bw, temperature=temp,
                                verbose=verbose, return_details=return_details)
    if mode == 'prob_saved':
        return logprob_ensemble(model_names, mode='prob', beam_width=bw, temperature=temp,
                                verbose=verbose, return_details=return_details)
    if mode == 'max_saved':
        return logprob_ensemble(model_names, mode='max', beam_width=bw, temperature=temp,
                                verbose=verbose, return_details=return_details)
    if mode == 'logits':
        for mn in model_names:
            lp_path = get_model_dir(mn) / 'ctc_logprobs.pt'
            if not lp_path.exists():
                save_logits_for_model(mn)
        logprob_ensemble(model_names, mode='logits', beam_width=bw, temperature=temp)
        return logprob_ensemble(model_names, mode='prob', beam_width=bw, temperature=temp,
                                return_details=return_details)
    if mode == 'sweep_logits':
        return sweep_logprob_ensemble(model_names, max_k=FLAGS.ensemble_max_sweep_k, beam_width=bw,
                                      temperature=temp)
    if mode == 'nbest_rescore':
        return nbest_rescore_ensemble(model_names, nbest=FLAGS.ensemble_nbest, beam_width=bw,
                                      lm=lm, lm_weight=FLAGS.ensemble_lm_weight,
                                      verbose=verbose, return_details=return_details, folds=folds)
    if mode == 'nbest_rescore2':
        return nbest_rescore2_ensemble(model_names, nbest=FLAGS.ensemble_nbest, beam_width=bw,
                                       lm=lm, lm_weight=FLAGS.ensemble_lm_weight,
                                       verbose=verbose, return_details=return_details, folds=folds)
    if mode == 'tree_reranker':
        feat_all = FLAGS.ensemble_feat_all
        tree_params = _get_tree_cli_params()
        return tree_reranker_ensemble(
            model_names,
            nbest=FLAGS.ensemble_nbest,
            beam_width=bw,
            lm=lm,
            aux_lms=aux_lms,
            tree_model=tree_params['tree_model'],
            n_folds=FLAGS.ensemble_cv_folds,
            tree_iters=tree_params['tree_iters'],
            tree_lr=tree_params['tree_lr'],
            tree_depth=tree_params['tree_depth'],
            tree_leaves=tree_params['tree_leaves'],
            tree_bagging=tree_params['tree_bagging'],
            tree_feat_frac=tree_params['tree_feat_frac'],
            tree_reg_lambda=tree_params['tree_reg_lambda'],
            tree_early_stop=tree_params['tree_early_stop'],
            tree_dd_weight=tree_params['tree_dd_weight'],
            tree_ext_weight=tree_params['tree_ext_weight'],
            tree_device=tree_params['tree_device'],
            tree_task=tree_params['tree_task'],
            tree_obj=tree_params['tree_obj'],
            relevance_levels=FLAGS.ensemble_relevance_levels,
            relevance_strategy=FLAGS.ensemble_relevance_strategy,
            no_lm_feats=no_lm_feats,
            lm_per_word_only=lm_per_word_only,
            feat_text=FLAGS.ensemble_feat_text or feat_all,
            feat_ipa=FLAGS.ensemble_feat_ipa or feat_all,
            feat_ctc_stats=FLAGS.ensemble_feat_ctc_stats or feat_all,
            no_ctc_score_feats=FLAGS.ensemble_no_ctc_score_feats,
            feat_audio=FLAGS.ensemble_feat_audio or feat_all,
            feat_consensus=FLAGS.ensemble_feat_consensus or feat_all,
            feat_mbr=FLAGS.ensemble_feat_mbr,
            feat_group_ext=FLAGS.ensemble_feat_group_ext or feat_all,
            feat_align=FLAGS.ensemble_feat_align or feat_all,
            feat_tdt=FLAGS.ensemble_feat_tdt,
            feat_tdt_light=FLAGS.ensemble_feat_tdt_light,
            feat_tdt_primary_score=FLAGS.ensemble_feat_tdt_primary_score or feat_all,
            feat_tdt_nbest_score=FLAGS.ensemble_feat_tdt_nbest_score,
            feat_tdt_exact=FLAGS.ensemble_feat_tdt_exact,
            feat_tdt_group=FLAGS.ensemble_feat_tdt_group,
            feat_tdtctc_compare=FLAGS.ensemble_feat_tdtctc_compare or feat_all,
            feat_tdt_score_compare=FLAGS.ensemble_feat_tdt_score_compare or feat_all,
            feat_wavlm_group=FLAGS.ensemble_feat_wavlm_group,
            feat_nemo_group=FLAGS.ensemble_feat_nemo_group,
            feat_group_edit_dist=FLAGS.ensemble_feat_group_edit_dist,
            feat_dual=FLAGS.ensemble_feat_dual,
            feat_logprob_proxy=FLAGS.ensemble_feat_logprob_proxy or feat_all,
            feat_word=FLAGS.ensemble_feat_word or feat_all,
            feat_aux_meta=FLAGS.ensemble_feat_aux_meta or feat_all,
            feat_word_label=FLAGS.ensemble_feat_word_label,
            word_label_file=FLAGS.ensemble_word_label_file,
            word_label_col=FLAGS.ensemble_word_label_col,
            tdt_eval_nbest=FLAGS.ensemble_tdt_eval_nbest,
            tdt_feat_topk=FLAGS.ensemble_tdt_feat_topk,
            tdt_force_keep_preds=FLAGS.ensemble_tdt_force_keep_preds,
            cache_tdt_exact_scores=FLAGS.ensemble_cache_tdt_exact_scores,
            drop_feats=FLAGS.ensemble_drop_feats,
            cache_dataset=FLAGS.ensemble_cache_dataset,
            n_seeds=FLAGS.ensemble_n_seeds,
            n_workers=FLAGS.ensemble_n_workers,
            model_max_dur=model_max_dur,
            save_dir=save_dir,
            exp_name=FLAGS.ensemble_exp_name,
            exp_notes=FLAGS.ensemble_exp_notes,
            dump_feats=FLAGS.ensemble_dump_feats,
            dump_feats_limit=FLAGS.ensemble_dump_feats_limit,
            dump_feats_uids_path=FLAGS.ensemble_dump_feats_uids_path,
            verbose=verbose,
            return_details=return_details,
        )
    raise ValueError(f'Unknown ensemble_mode={mode}')


def main(argv):
    del argv
    global WORKING_BASE

    if FLAGS.ensemble_working_dir:
        WORKING_BASE = Path(FLAGS.ensemble_working_dir)

    preview_folds = str(FLAGS.ensemble_folds or '0').strip().lower()
    preview_fold = 0 if preview_folds == 'all' else int(preview_folds.split(',')[0].strip())
    _set_active_fold(preview_fold)

    original_mode = FLAGS.ensemble_mode
    mode, resolved_tree_task = _resolve_tree_mode_and_task(
        original_mode,
        FLAGS.ensemble_tree_task,
    )
    FLAGS.ensemble_tree_task = resolved_tree_task
    model_max_dur = {}
    if FLAGS.ensemble_models:
        model_specs = [m.strip() for m in FLAGS.ensemble_models.split(',') if m.strip()]
        model_names, model_max_dur = _parse_model_specs(model_specs, source='--ensemble_models')
    else:
        need_logprobs = _mode_requires_ctc_logprobs_for_all_models(mode)
        model_names, model_max_dur = _load_models_from_file(require_logprobs=need_logprobs, verbose=True)
        if not model_names:
            model_names = auto_select_models(FLAGS.ensemble_top_k, require_logprobs=need_logprobs)
            model_max_dur = {}

    assert model_names, 'No ensemble models resolved. Set --emodels or provide a valid models.txt.'

    # Auto-generate experiment name from CLI flags (melt convention).
    ensemble_model_name = _build_ensemble_model_name()
    if ensemble_model_name != 'ensemble':
        print(f'Ensemble model name: {ensemble_model_name}')

    folds = _resolve_folds(FLAGS.ensemble_folds, model_names)
    if FLAGS.ensemble_clear_cache:
        _clear_reranker_caches(model_names, folds, verbose=True)

    # Auto-enable parallel CV folds for 'tree' shortcut unless explicitly disabled.
    # This only affects outer eval-fold parallelism; inner tree CV is always parallel.
    if original_mode == 'tree':
        _pfolds_explicitly_set = any(
            '--ensemble_parallel_folds' in a or '--pfolds' in a or '--parallel_folds' in a
            for a in sys.argv[1:]
        )
        if not _pfolds_explicitly_set and FLAGS.ensemble_parallel_folds == 0:
            # Use max(len(folds), n_cv_folds) so inner CV also parallelizes
            n_cv = int(getattr(FLAGS, 'ensemble_cv_folds', 5) or 5)
            FLAGS.ensemble_parallel_folds = max(len(folds), n_cv)

    if (mode == 'tree_reranker' and len(folds) > 1 and
            _get_parallel_fold_job_count(folds) > 1 and
            os.environ.get('ENSEMBLE_PARALLEL_CHILD') != '1'):
        _run_parallel_tree_folds(mode, model_names, folds)
        return

    fold_groups = _chunk_folds(folds, FLAGS.ensemble_n_efolds)
    if mode in ('nbest_rescore', 'nbest_rescore2', 'tree_reranker'):
        lm, aux_lms = _load_lm_from_flags()
    else:
        lm, aux_lms = None, None

    aggregate_modes = {'text', 'rover', 'logits_saved', 'prob_saved', 'max_saved', 'nbest_rescore', 'nbest_rescore2', 'tree_reranker'}
    if len(folds) == 1 or mode not in aggregate_modes:
        for fold in folds:
            _set_active_fold(fold)
            print(f'\n=== Running fold {fold} mode={mode} ===')
            fold_save_dir = None
            if mode == 'tree_reranker' and FLAGS.ensemble_save_dir and len(folds) > 1:
                fold_save_dir = str(Path(FLAGS.ensemble_save_dir) / f'fold{fold}')
            _run_single_mode(mode, model_names, lm=lm, aux_lms=aux_lms,
                             save_dir=fold_save_dir, folds=[fold], model_max_dur=model_max_dur)
        return

    group_payloads = []
    for fold_group in fold_groups:
        group_label = _format_fold_group(fold_group)
        if len(fold_group) == 1:
            fold = fold_group[0]
            _set_active_fold(fold)
            print(f'\n=== Running fold {fold} mode={mode} ===')
            fold_save_dir = None
            if mode == 'tree_reranker' and FLAGS.ensemble_save_dir:
                fold_save_dir = str(Path(FLAGS.ensemble_save_dir) / f'fold{fold}')
            payload = _run_single_mode(mode, model_names, lm=lm, aux_lms=aux_lms, return_details=True,
                                       save_dir=fold_save_dir, folds=fold_group,
                                       model_max_dur=model_max_dur)
        else:
            print(f'\n=== Running folds {group_label} mode={mode} ===')
            payloads = []
            for fold in fold_group:
                _set_active_fold(fold)
                print(f'\n--- Fold {fold} / group {group_label} ---')
                fold_save_dir = None
                if mode == 'tree_reranker' and FLAGS.ensemble_save_dir:
                    fold_save_dir = str(Path(FLAGS.ensemble_save_dir) / f'fold{fold}')
                payloads.append(_run_single_mode(mode, model_names, lm=lm, aux_lms=aux_lms, return_details=True,
                                                 save_dir=fold_save_dir, verbose=True, folds=[fold],
                                                 model_max_dur=model_max_dur))
            merged = _merge_fold_payloads(payloads, label=f'OOF {mode} ({group_label})', verbose=True)
            if mode in {'text', 'rover', 'nbest_rescore', 'nbest_rescore2'}:
                _print_individual_model_scores(model_names, folds=fold_group)
            payload = {
                'result': merged,
                'predictions': {uid: pred for item in payloads for uid, pred in item['predictions'].items()},
                'gold': {uid: label for item in payloads for uid, label in item['gold'].items()},
                'meta': {uid: row for item in payloads for uid, row in item['meta'].items()},
            }
        group_payloads.append(payload)

    if len(group_payloads) > 1:
        _merge_fold_payloads(group_payloads, label=f'OOF {mode} ({len(folds)} folds)', verbose=True)


if __name__ == '__main__':
    sys.argv = _normalize_ensemble_cli_argv(sys.argv)
    app.run(main)
