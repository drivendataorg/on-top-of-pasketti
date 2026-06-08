# Tree reranker successful reproduction log

This file records a full successful `bash reproduce_tree_reranker.sh` run from the standalone release, using offline fold-0 artifacts from a sibling development checkout.

It is intended as a line-by-line reference for reproducibility checks.

```text
Using offline artifact root: ../../pasketti-phonetic/working/offline/9
WARN: missing optional ../../pasketti-phonetic/working/offline/9/v17.backbone-wavlm-large.ep3.5.leval/0/aux_meta_preds.pt
WARN: missing optional ../../pasketti-phonetic/working/offline/9/v16.backbone-wavlm-large.dual_bpe.mix4.eval/0/aux_meta_preds.pt
WARN: missing optional ../../pasketti-phonetic/working/offline/9/v16.backbone-wavlm-large.dual_bpe.mix4.mix_csss.ep4.5.eval/0/aux_meta_preds.pt
WARN: missing optional ../../pasketti-phonetic/working/offline/9/v16.backbone-wavlm-large.dual_bpe.eval/0/aux_meta_preds.pt
WARN: missing optional ../../pasketti-phonetic/working/offline/9/v16.backbone-wavlm-large.dual_bpe.eval/0/model.pt
WARN: missing optional score artifact ../../pasketti-phonetic/working/offline/9/v16.dual_bpe.tdt_only.eval/0/ctc_logprobs.pt
WARN: missing optional ../../pasketti-phonetic/working/offline/9/v16.dual_bpe.tdt_only.eval/0/aux_meta_preds.pt
WARN: missing optional score artifact ../../pasketti-phonetic/working/offline/9/v16.dual_bpe.mix_csss.tdt_only.eval/0/ctc_logprobs.pt
WARN: missing optional ../../pasketti-phonetic/working/offline/9/v16.dual_bpe.mix_csss.tdt_only.eval/0/aux_meta_preds.pt
WARN: missing optional score artifact ../../pasketti-phonetic/working/offline/9/v16.dual_bpe.mix2.mix_csss.tdt_only.eval/0/ctc_logprobs.pt
WARN: missing optional ../../pasketti-phonetic/working/offline/9/v16.dual_bpe.mix2.mix_csss.tdt_only.eval/0/aux_meta_preds.pt
WARN: missing optional ../../pasketti-phonetic/working/offline/9/v16.dual_bpe.wo_scale-2.eval/0/aux_meta_preds.pt
WARN: missing optional ../../pasketti-phonetic/working/offline/9/v16.dual_bpe.wo_scale-2.eval/0/model.pt
WARN: missing optional ../../pasketti-phonetic/working/offline/9/v16.dual_bpe.mix4.eval/0/aux_meta_preds.pt
WARN: missing optional ../../pasketti-phonetic/working/offline/9/v16.dual_bpe.mix2.eval/0/aux_meta_preds.pt
Found 11 model eval dirs; 8 have ctc_logprobs.pt.
+ PYTHONPATH=_compat:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python ensemble.py --ensemble_working_dir=../../pasketti-phonetic/working/offline/9 --feat_nemo_group --feat_tdt_group --feat_wavlm_group --mns=.0407
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1779341415.246998 1497423 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1779341415.251381 1497423 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
Loaded 11 models from /home/gezi/pikachu/projects/drivendata/pasketti-phonetic-solution/src/models.txt:
  v17.backbone-wavlm-large.ep3.5.leval
  v16.backbone-wavlm-large.dual_bpe.mix4.eval
  v16.backbone-wavlm-large.dual_bpe.mix4.mix_csss.ep4.5.eval
  v16.backbone-wavlm-large.dual_bpe.eval
  v16.dual_bpe.tdt_only.eval
  v16.dual_bpe.mix_csss.tdt_only.eval
  v16.dual_bpe.mix2.mix_csss.tdt_only.eval
  v16.dual_bpe.wo_scale-2.eval
  v16.aux_loss.dual_bpe.eval
  v16.dual_bpe.mix4.eval
  v16.dual_bpe.mix2.eval
Ensemble model name: ensemble.feat_nemo_group.feat_tdt_group.feat_wavlm_group.0407

=== Running fold 0 mode=tree_reranker ===

=== Tree Reranker Ensemble (tm=cb, folds=5) ===
Models: v17.backbone-wavlm-large.ep3.5.leval, v16.backbone-wavlm-large.dual_bpe.mix4.eval, v16.backbone-wavlm-large.dual_bpe.mix4.mix_csss.ep4.5.eval, v16.backbone-wavlm-large.dual_bpe.eval, v16.dual_bpe.tdt_only.eval, v16.dual_bpe.mix_csss.tdt_only.eval, v16.dual_bpe.mix2.mix_csss.tdt_only.eval, v16.dual_bpe.wo_scale-2.eval, v16.aux_loss.dual_bpe.eval, v16.dual_bpe.mix4.eval, v16.dual_bpe.mix2.eval
Tree source weights: dd=1.0, ext=0.5
  Loaded 30645 utterances from v17.backbone-wavlm-large.ep3.5.leval
  Loaded 30645 utterances from v16.backbone-wavlm-large.dual_bpe.mix4.eval
  Loaded 30645 utterances from v16.backbone-wavlm-large.dual_bpe.mix4.mix_csss.ep4.5.eval
  Loaded 30645 utterances from v16.backbone-wavlm-large.dual_bpe.eval
  No ctc_logprobs for v16.dual_bpe.tdt_only.eval; using primary/dual predictions as reranker candidates only
  No ctc_logprobs for v16.dual_bpe.mix_csss.tdt_only.eval; using primary/dual predictions as reranker candidates only
  No ctc_logprobs for v16.dual_bpe.mix2.mix_csss.tdt_only.eval; using primary/dual predictions as reranker candidates only
  Loaded 30645 utterances from v16.dual_bpe.wo_scale-2.eval
  Loaded 30645 utterances from v16.aux_loss.dual_bpe.eval
  Loaded 30645 utterances from v16.dual_bpe.mix4.eval
  Loaded 30645 utterances from v16.dual_bpe.mix2.eval
  Using 16 workers for parallel feature building
  Building features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30645/30645 [02:29<00:00, 205.53it/s]
  Built 1068582 candidate rows for 30645 utterances in 150.0s
I0521 13:33:26.848756 126229165422400 config_base.py:1344] word_ctc_bpe_add_blank auto-set to False (phonetic non-NeMo/non-wav2vec2 fallback defaults to no-extra-blank protocol)
[05/21/26 13:33:26] util.py:3931 in restore_flags()- FLAGS.kidx: None
  TDT disabled for v17.backbone-wavlm-large.ep3.5.leval: ctc_only=False, ctc_weight=1.0, s2s_decoder=native
[05/21/26 13:33:28] util.py:3931 in restore_flags()- FLAGS.kidx: None
  TDT disabled for v16.backbone-wavlm-large.dual_bpe.mix4.eval: ctc_only=False, ctc_weight=1.0, s2s_decoder=native
I0521 13:33:28.605925 126229165422400 config_base.py:1344] word_ctc_bpe_add_blank auto-set to False (phonetic wav2vec2/hubert/wavlm legacy auxiliary BPE checkpoints used no-extra-blank protocol)
[05/21/26 13:33:28] util.py:3931 in restore_flags()- FLAGS.kidx: None
  TDT disabled for v16.backbone-wavlm-large.dual_bpe.mix4.mix_csss.ep4.5.eval: ctc_only=False, ctc_weight=1.0, s2s_decoder=native
I0521 13:33:28.723579 126229165422400 config_base.py:1344] word_ctc_bpe_add_blank auto-set to False (phonetic wav2vec2/hubert/wavlm legacy auxiliary BPE checkpoints used no-extra-blank protocol)
[05/21/26 13:33:28] util.py:3931 in restore_flags()- FLAGS.kidx: None
  TDT disabled for v16.backbone-wavlm-large.dual_bpe.eval: ctc_only=False, ctc_weight=1.0, s2s_decoder=native
I0521 13:33:28.841136 126229165422400 config_base.py:1344] word_ctc_bpe_add_blank auto-set to False (phonetic wav2vec2/hubert/wavlm legacy auxiliary BPE checkpoints used no-extra-blank protocol)
[05/21/26 13:33:28] util.py:3931 in restore_flags()- FLAGS.kidx: None
I0521 13:33:28.958667 126229165422400 config_base.py:1344] word_ctc_bpe_add_blank auto-set to True (phonetic NeMo legacy auxiliary BPE checkpoints used vocab+1 blank protocol)
[05/21/26 13:33:29] util.py:3931 in restore_flags()- FLAGS.kidx: None
I0521 13:33:29.075858 126229165422400 config_base.py:1344] word_ctc_bpe_add_blank auto-set to True (phonetic NeMo legacy auxiliary BPE checkpoints used vocab+1 blank protocol)
[05/21/26 13:33:29] util.py:3931 in restore_flags()- FLAGS.kidx: None
I0521 13:33:29.193045 126229165422400 config_base.py:1344] word_ctc_bpe_add_blank auto-set to True (phonetic NeMo legacy auxiliary BPE checkpoints used vocab+1 blank protocol)
[05/21/26 13:33:29] util.py:3931 in restore_flags()- FLAGS.kidx: None
  TDT disabled for v16.dual_bpe.wo_scale-2.eval: ctc_only=False, ctc_weight=1.0, s2s_decoder=native
I0521 13:33:29.310584 126229165422400 config_base.py:1344] word_ctc_bpe_add_blank auto-set to True (phonetic NeMo legacy auxiliary BPE checkpoints used vocab+1 blank protocol)
[05/21/26 13:33:29] util.py:3931 in restore_flags()- FLAGS.kidx: None
  TDT disabled for v16.aux_loss.dual_bpe.eval: ctc_only=False, ctc_weight=1.0, s2s_decoder=native
I0521 13:33:29.427818 126229165422400 config_base.py:1344] word_ctc_bpe_add_blank auto-set to True (phonetic NeMo legacy auxiliary BPE checkpoints used vocab+1 blank protocol)
[05/21/26 13:33:29] util.py:3931 in restore_flags()- FLAGS.kidx: None
  TDT disabled for v16.dual_bpe.mix4.eval: ctc_only=False, ctc_weight=1.0, s2s_decoder=native
I0521 13:33:29.544043 126229165422400 config_base.py:1344] word_ctc_bpe_add_blank auto-set to True (phonetic NeMo legacy auxiliary BPE checkpoints used vocab+1 blank protocol)
[05/21/26 13:33:29] util.py:3931 in restore_flags()- FLAGS.kidx: None
  TDT disabled for v16.dual_bpe.mix2.eval: ctc_only=False, ctc_weight=1.0, s2s_decoder=native
    v16.dual_bpe.tdt_only.eval: TDT primary texts=30643, scores=30643, nbest_score_uids=0
    v16.dual_bpe.mix_csss.tdt_only.eval: TDT primary texts=30641, scores=30641, nbest_score_uids=0
    v16.dual_bpe.mix2.mix_csss.tdt_only.eval: TDT primary texts=30645, scores=30645, nbest_score_uids=0
  TDT feature source models: ['v16.dual_bpe.tdt_only.eval', 'v16.dual_bpe.mix_csss.tdt_only.eval', 'v16.dual_bpe.mix2.mix_csss.tdt_only.eval']
  TDT feature groups: light=True, primary_score=False, nbest_score=False, exact=False
  TDT feature columns added: 16 (light=True, exact=False)
  TDT subgroup feature columns added: 25
  wavlm subgroup feature columns added: 17
  nonwavlm subgroup feature columns added: 20
  nemo subgroup feature columns added: 17
  nonnemo subgroup feature columns added: 20
Dataset: 1068582 rows, 212 features
Features: ['ctc_score_mean', 'ctc_score_std', 'ctc_score_min', 'ctc_score_max', 'ctc_score_range', 'n_score_models', 'ctc_score_v17.backbone-wavlm-large.ep3.5.leval', 'ctc_score_v16.backbone-wavlm-large.dual_bpe.mix4.eval', 'ctc_score_v16.backbone-wavlm-large.dual_bpe.mix4.mix_csss.ep4.5.eval', 'ctc_score_v16.backbone-wavlm-large.dual_bpe.eval', 'ctc_score_v16.dual_bpe.tdt_only.eval', 'ctc_score_v16.dual_bpe.mix_csss.tdt_only.eval', 'ctc_score_v16.dual_bpe.mix2.mix_csss.tdt_only.eval', 'ctc_score_v16.dual_bpe.wo_scale-2.eval', 'ctc_score_v16.aux_loss.dual_bpe.eval', 'ctc_score_v16.dual_bpe.mix4.eval', 'ctc_score_v16.dual_bpe.mix2.eval', 'text_len', 'n_frames', 'char_per_frame', 'beam_rank_v17.backbone-wavlm-large.ep3.5.leval', 'beam_rank_v16.backbone-wavlm-large.dual_bpe.mix4.eval', 'beam_rank_v16.backbone-wavlm-large.dual_bpe.mix4.mix_csss.ep4.5.eval', 'beam_rank_v16.backbone-wavlm-large.dual_bpe.eval', 'beam_rank_v16.dual_bpe.tdt_only.eval', 'beam_rank_v16.dual_bpe.mix_csss.tdt_only.eval', 'beam_rank_v16.dual_bpe.mix2.mix_csss.tdt_only.eval', 'beam_rank_v16.dual_bpe.wo_scale-2.eval', 'beam_rank_v16.aux_loss.dual_bpe.eval', 'beam_rank_v16.dual_bpe.mix4.eval', 'beam_rank_v16.dual_bpe.mix2.eval', 'n_models_has', 'edit_dist_to_best_v17.backbone-wavlm-large.ep3.5.leval', 'edit_dist_to_best_v16.backbone-wavlm-large.dual_bpe.mix4.eval', 'edit_dist_to_best_v16.backbone-wavlm-large.dual_bpe.mix4.mix_csss.ep4.5.eval', 'edit_dist_to_best_v16.backbone-wavlm-large.dual_bpe.eval', 'edit_dist_to_best_v16.dual_bpe.tdt_only.eval', 'edit_dist_to_best_v16.dual_bpe.mix_csss.tdt_only.eval', 'edit_dist_to_best_v16.dual_bpe.mix2.mix_csss.tdt_only.eval', 'edit_dist_to_best_v16.dual_bpe.wo_scale-2.eval', 'edit_dist_to_best_v16.aux_loss.dual_bpe.eval', 'edit_dist_to_best_v16.dual_bpe.mix4.eval', 'edit_dist_to_best_v16.dual_bpe.mix2.eval', 'duration_sec', 'chars_per_sec', 'words_per_sec', 'audio_duration_sec', 'has_audio_duration_sec', 'chars_per_audio_sec', 'words_per_audio_sec', 'audio_minus_frame_duration_sec', 'audio_to_frame_duration_ratio', 'beam_rank_mean', 'beam_rank_min', 'beam_rank_max', 'ctc_score_v17.backbone-wavlm-large.ep3.5.leval_rank', 'ctc_score_v17.backbone-wavlm-large.ep3.5.leval_per_char', 'ctc_score_v16.backbone-wavlm-large.dual_bpe.mix4.eval_rank', 'ctc_score_v16.backbone-wavlm-large.dual_bpe.mix4.eval_per_char', 'ctc_score_v16.backbone-wavlm-large.dual_bpe.mix4.mix_csss.ep4.5.eval_rank', 'ctc_score_v16.backbone-wavlm-large.dual_bpe.mix4.mix_csss.ep4.5.eval_per_char', 'ctc_score_v16.backbone-wavlm-large.dual_bpe.eval_rank', 'ctc_score_v16.backbone-wavlm-large.dual_bpe.eval_per_char', 'ctc_score_v16.dual_bpe.tdt_only.eval_rank', 'ctc_score_v16.dual_bpe.tdt_only.eval_per_char', 'ctc_score_v16.dual_bpe.mix_csss.tdt_only.eval_rank', 'ctc_score_v16.dual_bpe.mix_csss.tdt_only.eval_per_char', 'ctc_score_v16.dual_bpe.mix2.mix_csss.tdt_only.eval_rank', 'ctc_score_v16.dual_bpe.mix2.mix_csss.tdt_only.eval_per_char', 'ctc_score_v16.dual_bpe.wo_scale-2.eval_rank', 'ctc_score_v16.dual_bpe.wo_scale-2.eval_per_char', 'ctc_score_v16.aux_loss.dual_bpe.eval_rank', 'ctc_score_v16.aux_loss.dual_bpe.eval_per_char', 'ctc_score_v16.dual_bpe.mix4.eval_rank', 'ctc_score_v16.dual_bpe.mix4.eval_per_char', 'ctc_score_v16.dual_bpe.mix2.eval_rank', 'ctc_score_v16.dual_bpe.mix2.eval_per_char', 'ctc_score_mean_rank', 'ctc_score_mean_zscore', 'ctc_score_diff_from_best', 'ctc_score_mean_per_char', 'is_best_v17.backbone-wavlm-large.ep3.5.leval', 'is_best_v16.backbone-wavlm-large.dual_bpe.mix4.eval', 'is_best_v16.backbone-wavlm-large.dual_bpe.mix4.mix_csss.ep4.5.eval', 'is_best_v16.backbone-wavlm-large.dual_bpe.eval', 'is_best_v16.dual_bpe.tdt_only.eval', 'is_best_v16.dual_bpe.mix_csss.tdt_only.eval', 'is_best_v16.dual_bpe.mix2.mix_csss.tdt_only.eval', 'is_best_v16.dual_bpe.wo_scale-2.eval', 'is_best_v16.aux_loss.dual_bpe.eval', 'is_best_v16.dual_bpe.mix4.eval', 'is_best_v16.dual_bpe.mix2.eval', 'n_models_is_best', 'text_len_diff_from_median', 'n_candidates', 'mean_edit_dist_to_best', 'min_edit_dist_to_best', 'is_tdt_pred_v16.dual_bpe.tdt_only.eval', 'tdt_len_diff_v16.dual_bpe.tdt_only.eval', 'tdt_spaces_diff_v16.dual_bpe.tdt_only.eval', 'is_tdt_pred_v16.dual_bpe.mix_csss.tdt_only.eval', 'tdt_len_diff_v16.dual_bpe.mix_csss.tdt_only.eval', 'tdt_spaces_diff_v16.dual_bpe.mix_csss.tdt_only.eval', 'is_tdt_pred_v16.dual_bpe.mix2.mix_csss.tdt_only.eval', 'tdt_len_diff_v16.dual_bpe.mix2.mix_csss.tdt_only.eval', 'tdt_spaces_diff_v16.dual_bpe.mix2.mix_csss.tdt_only.eval', 'n_tdt_pred_hits', 'tdt_len_diff_mean', 'tdt_len_diff_min', 'tdt_len_diff_max', 'tdt_spaces_diff_mean', 'tdt_spaces_diff_min', 'tdt_spaces_diff_max', 'tdt_n_models', 'tdt_ctc_score_mean', 'tdt_ctc_score_std', 'tdt_ctc_score_min', 'tdt_ctc_score_max', 'tdt_ctc_score_range', 'tdt_ctc_score_mean_per_char', 'tdt_ctc_score_mean_rank', 'tdt_ctc_score_mean_diff_from_best', 'tdt_ctc_score_mean_zscore', 'tdt_ctc_score_max_rank', 'tdt_ctc_score_max_diff_from_best', 'tdt_ctc_score_max_zscore', 'tdt_beam_best_vote_count', 'tdt_beam_best_vote_frac', 'tdt_beam_best_vote_is_top', 'tdt_beam_best_vote_margin', 'tdt_primary_unique_count', 'is_tdt_primary_pred_v16.dual_bpe.tdt_only.eval', 'is_tdt_primary_pred_v16.dual_bpe.mix_csss.tdt_only.eval', 'is_tdt_primary_pred_v16.dual_bpe.mix2.mix_csss.tdt_only.eval', 'tdt_primary_hit_count', 'tdt_primary_hit_frac', 'tdt_primary_hit_is_top', 'tdt_primary_hit_margin', 'wavlm_n_models', 'wavlm_ctc_score_mean', 'wavlm_ctc_score_std', 'wavlm_ctc_score_min', 'wavlm_ctc_score_max', 'wavlm_ctc_score_range', 'wavlm_ctc_score_mean_per_char', 'wavlm_ctc_score_mean_rank', 'wavlm_ctc_score_mean_diff_from_best', 'wavlm_ctc_score_mean_zscore', 'wavlm_ctc_score_max_rank', 'wavlm_ctc_score_max_diff_from_best', 'wavlm_ctc_score_max_zscore', 'wavlm_beam_best_vote_count', 'wavlm_beam_best_vote_frac', 'wavlm_beam_best_vote_is_top', 'wavlm_beam_best_vote_margin', 'nonwavlm_n_models', 'nonwavlm_ctc_score_mean', 'nonwavlm_ctc_score_std', 'nonwavlm_ctc_score_min', 'nonwavlm_ctc_score_max', 'nonwavlm_ctc_score_range', 'nonwavlm_ctc_score_mean_per_char', 'nonwavlm_ctc_score_mean_rank', 'nonwavlm_ctc_score_mean_diff_from_best', 'nonwavlm_ctc_score_mean_zscore', 'nonwavlm_ctc_score_max_rank', 'nonwavlm_ctc_score_max_diff_from_best', 'nonwavlm_ctc_score_max_zscore', 'nonwavlm_beam_best_vote_count', 'nonwavlm_beam_best_vote_frac', 'nonwavlm_beam_best_vote_is_top', 'nonwavlm_beam_best_vote_margin', 'wavlm_vs_nonwavlm_ctc_score_mean_gap', 'wavlm_vs_nonwavlm_ctc_score_max_gap', 'wavlm_vs_nonwavlm_beam_best_vote_gap', 'nemo_n_models', 'nemo_ctc_score_mean', 'nemo_ctc_score_std', 'nemo_ctc_score_min', 'nemo_ctc_score_max', 'nemo_ctc_score_range', 'nemo_ctc_score_mean_per_char', 'nemo_ctc_score_mean_rank', 'nemo_ctc_score_mean_diff_from_best', 'nemo_ctc_score_mean_zscore', 'nemo_ctc_score_max_rank', 'nemo_ctc_score_max_diff_from_best', 'nemo_ctc_score_max_zscore', 'nemo_beam_best_vote_count', 'nemo_beam_best_vote_frac', 'nemo_beam_best_vote_is_top', 'nemo_beam_best_vote_margin', 'nonnemo_n_models', 'nonnemo_ctc_score_mean', 'nonnemo_ctc_score_std', 'nonnemo_ctc_score_min', 'nonnemo_ctc_score_max', 'nonnemo_ctc_score_range', 'nonnemo_ctc_score_mean_per_char', 'nonnemo_ctc_score_mean_rank', 'nonnemo_ctc_score_mean_diff_from_best', 'nonnemo_ctc_score_mean_zscore', 'nonnemo_ctc_score_max_rank', 'nonnemo_ctc_score_max_diff_from_best', 'nonnemo_ctc_score_max_zscore', 'nonnemo_beam_best_vote_count', 'nonnemo_beam_best_vote_frac', 'nonnemo_beam_best_vote_is_top', 'nonnemo_beam_best_vote_margin', 'nemo_vs_nonnemo_ctc_score_mean_gap', 'nemo_vs_nonnemo_ctc_score_max_gap', 'nemo_vs_nonnemo_beam_best_vote_gap']
  Fold 0: 8787 uids, 45 children, 277796 candidates
  Fold 1: 4553 uids, 41 children, 176830 candidates
  Fold 2: 6851 uids, 38 children, 211887 candidates
  Fold 3: 5604 uids, 38 children, 192460 candidates
  Fold 4: 4850 uids, 40 children, 209609 candidates
  Feature groups: text=off, ipa=off, ctc_stats=off, no_ctc_score_feats=off, audio=ON, consensus=off, mbr=off, group_ext=off, align=off, dual=off, tdt_light=ON, tdt_primary_score=off, tdt_nbest_score=off, tdt_exact=off, tdt_group=ON, tdtctc_compare=off, tdt_score_compare=ON, wavlm_group=ON, nemo_group=ON, logprob_proxy=off, word=off, aux_meta=off, no_lm_feats=True, lm_per_word_only=False
  Final features: 212
Relevance levels: 6 (0=binary, N=graded)
Relevance strategy: gap
  Unique relevance values: [0, 1, 2, 3, 4, 5]

  Parallel tree CV enabled: jobs=5, total_cores=128, per_job_tree_threads=25

  Fold 0: train=790786, valid=277796, pid=1500675
[05/21/26 13:35:14] tree.py:610 in create_model()
                    tree_gpu_train: False
[05/21/26 13:35:15] tree.py:765 in create_model()
                    model.get_params(): {'allow_writing_files': False,
                                         'bagging_temperature': 0.8,
                                         'depth': 4,
                                         'grow_policy': 'Lossguide',
                                         'iterations': 500,
                                         'l2_leaf_reg': 5.0,
                                         'learning_rate': 0.05,
                                         'loss_function': 'YetiRank',
                                         'metric_period': 20,
                                         'num_leaves': 31,
                                         'random_seed': 42,
                                         'rsm': 0.6,
                                         'task_type': 'CPU',
                                         'thread_count': 25,
                                         'train_dir': '/home/gezi/pikachu/projects/drivendata/pasketti-phonetic-solution/working/reranker/catboost_info',
                                         'use_best_model': True}
[05/21/26 13:35:15] tree.py:862 in setup()
                    self.model_type: 'cb'
                    self.gpu_infer: False
                    self.save_txt: False
[05/21/26 13:35:15] tree.py:853 in __init__()
                    self.model: <catboost.core.CatBoostRanker object at 0x72cdf1579360>
                    FLAGS.tree_fit: False
                    FLAGS.tree_convert: True
[05/21/26 13:35:15] tree.py:1499 in fit()- callbacks: []
[05/21/26 13:35:15] tree.py:1503 in fit()- eval_metric: None

  Fold 1: train=891752, valid=176830, pid=1500715
Pairwise losses don't support object weights.
Warning: Overfitting detector is active, thus evaluation metric is calculated on every iteration. 'metric_period' is ignored for evaluation metric.
0:	test: 0.8639410	best: 0.8639410 (0)	total: 423ms	remaining: 3m 30s
[05/21/26 13:35:16] tree.py:610 in create_model()
                    tree_gpu_train: False
[05/21/26 13:35:16] tree.py:765 in create_model()
                    model.get_params(): {'allow_writing_files': False,
                                         'bagging_temperature': 0.8,
                                         'depth': 4,
                                         'grow_policy': 'Lossguide',
                                         'iterations': 500,
                                         'l2_leaf_reg': 5.0,
                                         'learning_rate': 0.05,
                                         'loss_function': 'YetiRank',
                                         'metric_period': 20,
                                         'num_leaves': 31,
                                         'random_seed': 42,
                                         'rsm': 0.6,
                                         'task_type': 'CPU',
                                         'thread_count': 25,
                                         'train_dir': '/home/gezi/pikachu/projects/drivendata/pasketti-phonetic-solution/working/reranker/catboost_info',
                                         'use_best_model': True}
[05/21/26 13:35:16] tree.py:862 in setup()
                    self.model_type: 'cb'
                    self.gpu_infer: False
                    self.save_txt: False
[05/21/26 13:35:16] tree.py:853 in __init__()
                    self.model: <catboost.core.CatBoostRanker object at 0x72cdfeafbd00>
                    FLAGS.tree_fit: False
                    FLAGS.tree_convert: True
[05/21/26 13:35:16] tree.py:1499 in fit()- callbacks: []
[05/21/26 13:35:16] tree.py:1503 in fit()- eval_metric: None

  Fold 2: train=856695, valid=211887, pid=1500759
Pairwise losses don't support object weights.
Warning: Overfitting detector is active, thus evaluation metric is calculated on every iteration. 'metric_period' is ignored for evaluation metric.
0:	test: 0.8622060	best: 0.8622060 (0)	total: 465ms	remaining: 3m 52s
[05/21/26 13:35:17] tree.py:610 in create_model()
                    tree_gpu_train: False
[05/21/26 13:35:17] tree.py:765 in create_model()
                    model.get_params(): {'allow_writing_files': False,
                                         'bagging_temperature': 0.8,
                                         'depth': 4,
                                         'grow_policy': 'Lossguide',
                                         'iterations': 500,
                                         'l2_leaf_reg': 5.0,
                                         'learning_rate': 0.05,
                                         'loss_function': 'YetiRank',
                                         'metric_period': 20,
                                         'num_leaves': 31,
                                         'random_seed': 42,
                                         'rsm': 0.6,
                                         'task_type': 'CPU',
                                         'thread_count': 25,
                                         'train_dir': '/home/gezi/pikachu/projects/drivendata/pasketti-phonetic-solution/working/reranker/catboost_info',
                                         'use_best_model': True}
[05/21/26 13:35:17] tree.py:862 in setup()
                    self.model_type: 'cb'
                    self.gpu_infer: False
                    self.save_txt: False
[05/21/26 13:35:18] tree.py:853 in __init__()
                    self.model: <catboost.core.CatBoostRanker object at 0x72cdfeafbdc0>
                    FLAGS.tree_fit: False
                    FLAGS.tree_convert: True
[05/21/26 13:35:18] tree.py:1499 in fit()- callbacks: []
[05/21/26 13:35:18] tree.py:1503 in fit()- eval_metric: None

  Fold 3: train=876122, valid=192460, pid=1500808
Pairwise losses don't support object weights.
Warning: Overfitting detector is active, thus evaluation metric is calculated on every iteration. 'metric_period' is ignored for evaluation metric.
[05/21/26 13:35:18] tree.py:610 in create_model()
                    tree_gpu_train: False
[05/21/26 13:35:19] tree.py:765 in create_model()
                    model.get_params(): {'allow_writing_files': False,
                                         'bagging_temperature': 0.8,
                                         'depth': 4,
                                         'grow_policy': 'Lossguide',
                                         'iterations': 500,
                                         'l2_leaf_reg': 5.0,
                                         'learning_rate': 0.05,
                                         'loss_function': 'YetiRank',
                                         'metric_period': 20,
                                         'num_leaves': 31,
                                         'random_seed': 42,
                                         'rsm': 0.6,
                                         'task_type': 'CPU',
                                         'thread_count': 25,
                                         'train_dir': '/home/gezi/pikachu/projects/drivendata/pasketti-phonetic-solution/working/reranker/catboost_info',
                                         'use_best_model': True}
[05/21/26 13:35:19] tree.py:862 in setup()
                    self.model_type: 'cb'
                    self.gpu_infer: False
                    self.save_txt: False
[05/21/26 13:35:19] tree.py:853 in __init__()
                    self.model: <catboost.core.CatBoostRanker object at 0x72cdfeafbeb0>
                    FLAGS.tree_fit: False
                    FLAGS.tree_convert: True
[05/21/26 13:35:19] tree.py:1499 in fit()- callbacks: []
[05/21/26 13:35:19] tree.py:1503 in fit()- eval_metric: None
0:	test: 0.8853602	best: 0.8853602 (0)	total: 512ms	remaining: 4m 15s

  Fold 4: train=858973, valid=209609, pid=1500860
Pairwise losses don't support object weights.
Warning: Overfitting detector is active, thus evaluation metric is calculated on every iteration. 'metric_period' is ignored for evaluation metric.
[05/21/26 13:35:20] tree.py:610 in create_model()
                    tree_gpu_train: False
[05/21/26 13:35:20] tree.py:765 in create_model()
                    model.get_params(): {'allow_writing_files': False,
                                         'bagging_temperature': 0.8,
                                         'depth': 4,
                                         'grow_policy': 'Lossguide',
                                         'iterations': 500,
                                         'l2_leaf_reg': 5.0,
                                         'learning_rate': 0.05,
                                         'loss_function': 'YetiRank',
                                         'metric_period': 20,
                                         'num_leaves': 31,
                                         'random_seed': 42,
                                         'rsm': 0.6,
                                         'task_type': 'CPU',
                                         'thread_count': 25,
                                         'train_dir': '/home/gezi/pikachu/projects/drivendata/pasketti-phonetic-solution/working/reranker/catboost_info',
                                         'use_best_model': True}
0:	test: 0.8758679	best: 0.8758679 (0)	total: 492ms	remaining: 4m 5s
[05/21/26 13:35:20] tree.py:862 in setup()
                    self.model_type: 'cb'
                    self.gpu_infer: False
                    self.save_txt: False
[05/21/26 13:35:20] tree.py:853 in __init__()
                    self.model: <catboost.core.CatBoostRanker object at 0x72cdfeafbfa0>
                    FLAGS.tree_fit: False
                    FLAGS.tree_convert: True
[05/21/26 13:35:20] tree.py:1499 in fit()- callbacks: []
[05/21/26 13:35:20] tree.py:1503 in fit()- eval_metric: None
Pairwise losses don't support object weights.
Warning: Overfitting detector is active, thus evaluation metric is calculated on every iteration. 'metric_period' is ignored for evaluation metric.
0:	test: 0.8728361	best: 0.8728361 (0)	total: 520ms	remaining: 4m 19s
20:	test: 0.9040444	best: 0.9040444 (20)	total: 8.78s	remaining: 3m 20s
20:	test: 0.9113567	best: 0.9113567 (20)	total: 9.85s	remaining: 3m 44s
20:	test: 0.9136984	best: 0.9136984 (20)	total: 9.03s	remaining: 3m 26s
20:	test: 0.9181356	best: 0.9181356 (20)	total: 10s	remaining: 3m 48s
20:	test: 0.9111181	best: 0.9111181 (20)	total: 9.75s	remaining: 3m 42s
40:	test: 0.9048025	best: 0.9048025 (40)	total: 17.8s	remaining: 3m 18s
40:	test: 0.9145483	best: 0.9145483 (40)	total: 18s	remaining: 3m 21s
40:	test: 0.9123792	best: 0.9123792 (40)	total: 19.7s	remaining: 3m 40s
40:	test: 0.9190173	best: 0.9190173 (40)	total: 19.4s	remaining: 3m 37s
40:	test: 0.9119970	best: 0.9120012 (39)	total: 18.4s	remaining: 3m 26s
60:	test: 0.9050506	best: 0.9050506 (60)	total: 26.8s	remaining: 3m 12s
60:	test: 0.9146176	best: 0.9146473 (57)	total: 26.9s	remaining: 3m 13s
60:	test: 0.9126741	best: 0.9126741 (60)	total: 29.3s	remaining: 3m 31s
60:	test: 0.9126012	best: 0.9126012 (60)	total: 27.1s	remaining: 3m 14s
60:	test: 0.9193504	best: 0.9193504 (60)	total: 28.8s	remaining: 3m 27s
80:	test: 0.9052006	best: 0.9052194 (78)	total: 35.9s	remaining: 3m 5s
80:	test: 0.9149662	best: 0.9149662 (80)	total: 35.8s	remaining: 3m 4s
80:	test: 0.9127223	best: 0.9127955 (76)	total: 39.2s	remaining: 3m 22s
80:	test: 0.9128897	best: 0.9129144 (78)	total: 35.9s	remaining: 3m 5s
80:	test: 0.9197054	best: 0.9197054 (80)	total: 37.9s	remaining: 3m 15s
100:	test: 0.9054614	best: 0.9054811 (99)	total: 44.9s	remaining: 2m 57s
100:	test: 0.9151502	best: 0.9151555 (99)	total: 44.5s	remaining: 2m 55s
100:	test: 0.9129497	best: 0.9129636 (98)	total: 48.8s	remaining: 3m 12s
100:	test: 0.9131188	best: 0.9131492 (95)	total: 44.3s	remaining: 2m 55s
100:	test: 0.9199398	best: 0.9199398 (100)	total: 47.4s	remaining: 3m 7s
120:	test: 0.9056116	best: 0.9056116 (120)	total: 53.8s	remaining: 2m 48s
120:	test: 0.9151424	best: 0.9152215 (115)	total: 53.3s	remaining: 2m 47s
120:	test: 0.9134799	best: 0.9134799 (120)	total: 53s	remaining: 2m 45s
120:	test: 0.9131011	best: 0.9131011 (120)	total: 58.3s	remaining: 3m 2s
120:	test: 0.9199928	best: 0.9200270 (115)	total: 56.7s	remaining: 2m 57s
140:	test: 0.9057693	best: 0.9057693 (140)	total: 1m 2s	remaining: 2m 40s
140:	test: 0.9151171	best: 0.9152243 (136)	total: 1m 1s	remaining: 2m 37s
140:	test: 0.9137161	best: 0.9137183 (135)	total: 1m 1s	remaining: 2m 36s
140:	test: 0.9132013	best: 0.9132141 (139)	total: 1m 8s	remaining: 2m 54s
140:	test: 0.9201440	best: 0.9201440 (140)	total: 1m 6s	remaining: 2m 48s
160:	test: 0.9058937	best: 0.9059494 (154)	total: 1m 11s	remaining: 2m 30s
160:	test: 0.9152482	best: 0.9152482 (160)	total: 1m 10s	remaining: 2m 27s
160:	test: 0.9138625	best: 0.9138697 (157)	total: 1m 10s	remaining: 2m 27s
160:	test: 0.9202336	best: 0.9202336 (160)	total: 1m 15s	remaining: 2m 39s
160:	test: 0.9132878	best: 0.9133304 (152)	total: 1m 18s	remaining: 2m 45s
180:	test: 0.9060035	best: 0.9060035 (180)	total: 1m 20s	remaining: 2m 21s
180:	test: 0.9152885	best: 0.9153024 (167)	total: 1m 18s	remaining: 2m 18s
180:	test: 0.9139610	best: 0.9139960 (178)	total: 1m 18s	remaining: 2m 18s
180:	test: 0.9203063	best: 0.9203251 (179)	total: 1m 24s	remaining: 2m 29s
200:	test: 0.9060616	best: 0.9060616 (200)	total: 1m 29s	remaining: 2m 13s
180:	test: 0.9133785	best: 0.9133785 (180)	total: 1m 28s	remaining: 2m 35s
200:	test: 0.9154036	best: 0.9154044 (198)	total: 1m 27s	remaining: 2m 10s
200:	test: 0.9140654	best: 0.9140956 (195)	total: 1m 27s	remaining: 2m 10s
220:	test: 0.9061045	best: 0.9061045 (220)	total: 1m 38s	remaining: 2m 4s
200:	test: 0.9204389	best: 0.9204389 (200)	total: 1m 34s	remaining: 2m 20s
220:	test: 0.9153621	best: 0.9154146 (202)	total: 1m 36s	remaining: 2m 1s
200:	test: 0.9133964	best: 0.9134010 (194)	total: 1m 37s	remaining: 2m 25s
220:	test: 0.9141506	best: 0.9141506 (220)	total: 1m 35s	remaining: 2m 1s
240:	test: 0.9061364	best: 0.9061430 (237)	total: 1m 47s	remaining: 1m 55s
240:	test: 0.9153788	best: 0.9154146 (202)	total: 1m 44s	remaining: 1m 52s
220:	test: 0.9205571	best: 0.9205571 (220)	total: 1m 43s	remaining: 2m 10s
220:	test: 0.9133887	best: 0.9134778 (203)	total: 1m 47s	remaining: 2m 15s
240:	test: 0.9142566	best: 0.9142566 (240)	total: 1m 44s	remaining: 1m 52s
260:	test: 0.9061589	best: 0.9061869 (258)	total: 1m 56s	remaining: 1m 46s
260:	test: 0.9154843	best: 0.9154843 (260)	total: 1m 53s	remaining: 1m 43s
240:	test: 0.9206043	best: 0.9206119 (239)	total: 1m 53s	remaining: 2m 1s
240:	test: 0.9135532	best: 0.9135532 (240)	total: 1m 57s	remaining: 2m 5s
260:	test: 0.9143778	best: 0.9143778 (260)	total: 1m 52s	remaining: 1m 43s
280:	test: 0.9154806	best: 0.9154843 (260)	total: 2m 1s	remaining: 1m 35s
280:	test: 0.9061997	best: 0.9062243 (275)	total: 2m 5s	remaining: 1m 37s
260:	test: 0.9206589	best: 0.9206719 (257)	total: 2m 2s	remaining: 1m 52s
280:	test: 0.9144895	best: 0.9144895 (280)	total: 2m 1s	remaining: 1m 34s
260:	test: 0.9135752	best: 0.9135778 (255)	total: 2m 6s	remaining: 1m 55s
300:	test: 0.9154926	best: 0.9155195 (284)	total: 2m 10s	remaining: 1m 26s
300:	test: 0.9062049	best: 0.9062265 (293)	total: 2m 14s	remaining: 1m 28s
300:	test: 0.9145214	best: 0.9145526 (288)	total: 2m 9s	remaining: 1m 25s
280:	test: 0.9207097	best: 0.9207097 (280)	total: 2m 11s	remaining: 1m 42s
280:	test: 0.9135563	best: 0.9135942 (261)	total: 2m 15s	remaining: 1m 45s
320:	test: 0.9154925	best: 0.9155219 (304)	total: 2m 19s	remaining: 1m 17s
320:	test: 0.9062180	best: 0.9062398 (309)	total: 2m 23s	remaining: 1m 20s
320:	test: 0.9145757	best: 0.9145757 (320)	total: 2m 18s	remaining: 1m 16s
300:	test: 0.9207625	best: 0.9207710 (292)	total: 2m 20s	remaining: 1m 33s
300:	test: 0.9135831	best: 0.9135960 (287)	total: 2m 24s	remaining: 1m 35s
340:	test: 0.9154063	best: 0.9155219 (304)	total: 2m 27s	remaining: 1m 8s
340:	test: 0.9145672	best: 0.9145796 (327)	total: 2m 26s	remaining: 1m 8s
340:	test: 0.9061756	best: 0.9062398 (309)	total: 2m 32s	remaining: 1m 11s
320:	test: 0.9208020	best: 0.9208192 (313)	total: 2m 30s	remaining: 1m 23s
320:	test: 0.9136733	best: 0.9136965 (315)	total: 2m 34s	remaining: 1m 25s
Stopped by overfitting detector  (50 iterations wait)

bestTest = 0.9155218672
bestIteration = 304

Shrink model to first 305 iterations.
[05/21/26 13:37:53] tree.py:1011 in predict()- df.shape: (211887, 212)
Stopped by overfitting detector  (50 iterations wait)

bestTest = 0.9062398469
bestIteration = 309

Shrink model to first 310 iterations.
360:	test: 0.9146082	best: 0.9146082 (360)	total: 2m 34s	remaining: 59.6s
[05/21/26 13:37:56] tree.py:1011 in predict()- df.shape: (277796, 212)
340:	test: 0.9207797	best: 0.9208192 (313)	total: 2m 38s	remaining: 1m 14s
340:	test: 0.9136436	best: 0.9137018 (324)	total: 2m 43s	remaining: 1m 16s
380:	test: 0.9147232	best: 0.9147232 (380)	total: 2m 41s	remaining: 50.5s
360:	test: 0.9207765	best: 0.9208192 (313)	total: 2m 47s	remaining: 1m 4s
360:	test: 0.9137108	best: 0.9137108 (360)	total: 2m 51s	remaining: 1m 6s
Stopped by overfitting detector  (50 iterations wait)

bestTest = 0.9208192179
bestIteration = 313

Shrink model to first 314 iterations.
[05/21/26 13:38:09] tree.py:1011 in predict()- df.shape: (192460, 212)
400:	test: 0.9147252	best: 0.9147641 (384)	total: 2m 49s	remaining: 41.8s
380:	test: 0.9136741	best: 0.9137172 (363)	total: 3m	remaining: 56.3s
420:	test: 0.9147427	best: 0.9147641 (384)	total: 2m 56s	remaining: 33.2s
400:	test: 0.9136663	best: 0.9137172 (363)	total: 3m 8s	remaining: 46.5s
440:	test: 0.9147537	best: 0.9147732 (433)	total: 3m 4s	remaining: 24.7s
420:	test: 0.9137384	best: 0.9137384 (420)	total: 3m 16s	remaining: 36.8s
460:	test: 0.9147607	best: 0.9147844 (444)	total: 3m 12s	remaining: 16.2s
480:	test: 0.9147608	best: 0.9147844 (444)	total: 3m 19s	remaining: 7.87s
440:	test: 0.9137533	best: 0.9137677 (438)	total: 3m 24s	remaining: 27.3s
Stopped by overfitting detector  (50 iterations wait)

bestTest = 0.9147843909
bestIteration = 444

Shrink model to first 445 iterations.
[05/21/26 13:38:46] tree.py:1011 in predict()- df.shape: (209609, 212)
460:	test: 0.9137533	best: 0.9137698 (454)	total: 3m 32s	remaining: 18s
480:	test: 0.9137079	best: 0.9137698 (454)	total: 3m 40s	remaining: 8.72s
499:	test: 0.9137058	best: 0.9137698 (454)	total: 3m 48s	remaining: 0us

bestTest = 0.9137697638
bestIteration = 454

Shrink model to first 455 iterations.
[05/21/26 13:39:06] tree.py:1011 in predict()- df.shape: (176830, 212)

--- Feature Importance (cb) ---
ctc_score_mean_zscore	mean=30.301128	std=1.425640	nonzero_folds=5
mean_edit_dist_to_best	mean=9.939319	std=0.826653	nonzero_folds=5
ctc_score_diff_from_best	mean=8.264646	std=1.037304	nonzero_folds=5
wavlm_ctc_score_mean	mean=3.605914	std=0.763039	nonzero_folds=5
ctc_score_mean	mean=3.474330	std=0.672779	nonzero_folds=5
min_edit_dist_to_best	mean=3.310944	std=0.140186	nonzero_folds=5
n_candidates	mean=2.863006	std=0.346017	nonzero_folds=5
nonnemo_ctc_score_mean	mean=2.599261	std=0.540164	nonzero_folds=5
nonnemo_ctc_score_mean_zscore	mean=2.116836	std=0.315850	nonzero_folds=5
wavlm_ctc_score_mean_zscore	mean=1.903933	std=0.643912	nonzero_folds=5
edit_dist_to_best_v17.backbone-wavlm-large.ep3.5.leval	mean=1.750007	std=0.641300	nonzero_folds=5
text_len_diff_from_median	mean=1.748822	std=0.248916	nonzero_folds=5
ctc_score_mean_rank	mean=1.538491	std=0.479001	nonzero_folds=5
edit_dist_to_best_v16.dual_bpe.mix2.mix_csss.tdt_only.eval	mean=1.449464	std=0.532552	nonzero_folds=5
edit_dist_to_best_v16.dual_bpe.mix_csss.tdt_only.eval	mean=1.429734	std=0.248002	nonzero_folds=5
tdt_len_diff_min	mean=1.415359	std=0.274304	nonzero_folds=5
ctc_score_v17.backbone-wavlm-large.ep3.5.leval	mean=1.199378	std=0.178638	nonzero_folds=5
edit_dist_to_best_v16.backbone-wavlm-large.dual_bpe.mix4.mix_csss.ep4.5.eval	mean=1.145504	std=0.508607	nonzero_folds=5
wavlm_ctc_score_mean_diff_from_best	mean=1.080913	std=0.234825	nonzero_folds=5
nonnemo_ctc_score_mean_diff_from_best	mean=0.869551	std=0.231592	nonzero_folds=5
edit_dist_to_best_v16.backbone-wavlm-large.dual_bpe.eval	mean=0.866946	std=0.312013	nonzero_folds=5
ctc_score_v16.backbone-wavlm-large.dual_bpe.mix4.mix_csss.ep4.5.eval	mean=0.795430	std=0.394494	nonzero_folds=5
tdt_primary_unique_count	mean=0.783959	std=0.276555	nonzero_folds=5
edit_dist_to_best_v16.dual_bpe.tdt_only.eval	mean=0.772018	std=0.173569	nonzero_folds=5
edit_dist_to_best_v16.backbone-wavlm-large.dual_bpe.mix4.eval	mean=0.699701	std=0.238626	nonzero_folds=5

--- Tree Reranker (cb, 5-fold) Results ---
Overall CER: 0.26307  (n=30645, raw=0.22683)
  score/dd: 0.30666  (n=2417)
  score/ext: 0.21948  (n=28228)

--- Avg CTC Score (baseline) Results ---
Overall CER: 0.26637  (n=30645, raw=0.22922)
  score/dd: 0.31107  (n=2417)
  score/ext: 0.22168  (n=28228)

--- Tree Reranker FullAvg (cb, 5-fold models) Results ---
Overall CER: 0.26086  (n=30645, raw=0.22487)
  score/dd: 0.30415  (n=2417)
  score/ext: 0.21757  (n=28228)

--- Tree Reranker Vote (cb, 5-fold models) Results ---
Overall CER: 0.42612  (n=30645, raw=0.38757)
  score/dd: 0.47248  (n=2417)
  score/ext: 0.37975  (n=28228)

--- Tree Reranker Borda (cb, 5-fold models) Results ---
Overall CER: 0.42612  (n=30645, raw=0.38757)
  score/dd: 0.47248  (n=2417)
  score/ext: 0.37975  (n=28228)

--- Oracle (best candidate) Results ---
Overall CER: 0.15971  (n=30645, raw=0.13413)
  score/dd: 0.19048  (n=2417)
  score/ext: 0.12894  (n=28228)
  Saved cb fold 0: ['model.pkl']
  Saved eval.csv: ../../pasketti-phonetic/working/offline/9/ensemble.feat_nemo_group.feat_tdt_group.feat_wavlm_group.0407/0/eval.csv (30645 rows)
  Saved eval_fullavg.csv: ../../pasketti-phonetic/working/offline/9/ensemble.feat_nemo_group.feat_tdt_group.feat_wavlm_group.0407/0/eval_fullavg.csv (30645 rows)
  Saved eval_vote.csv: ../../pasketti-phonetic/working/offline/9/ensemble.feat_nemo_group.feat_tdt_group.feat_wavlm_group.0407/0/eval_vote.csv (30645 rows)
  Saved eval_borda.csv: ../../pasketti-phonetic/working/offline/9/ensemble.feat_nemo_group.feat_tdt_group.feat_wavlm_group.0407/0/eval_borda.csv (30645 rows)
  Saved strategy_case_analysis.csv: ../../pasketti-phonetic/working/offline/9/ensemble.feat_nemo_group.feat_tdt_group.feat_wavlm_group.0407/0/strategy_case_analysis.csv (30645 rows)
  Saved metrics.csv: ../../pasketti-phonetic/working/offline/9/ensemble.feat_nemo_group.feat_tdt_group.feat_wavlm_group.0407/0/metrics.csv
  Saved reranker_features.txt: ../../pasketti-phonetic/working/offline/9/ensemble.feat_nemo_group.feat_tdt_group.feat_wavlm_group.0407/0/reranker_features.txt
  Saved reranker_feature_importance.csv: ../../pasketti-phonetic/working/offline/9/ensemble.feat_nemo_group.feat_tdt_group.feat_wavlm_group.0407/0/reranker_feature_importance.csv
  Saved reranker_meta.json: ../../pasketti-phonetic/working/offline/9/ensemble.feat_nemo_group.feat_tdt_group.feat_wavlm_group.0407/0/reranker_meta.json
  Saved reranker_experiment.json: ../../pasketti-phonetic/working/offline/9/ensemble.feat_nemo_group.feat_tdt_group.feat_wavlm_group.0407/0/reranker_experiment.json
  Updated experiment_log.csv: ../../pasketti-phonetic/working/offline/9/ensemble-experiments/experiment_log.csv
Copied tree reranker artifacts to tree_reranker
```
