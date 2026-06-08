#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2025-02-16
#   \Description   Shared config for Pasketti ASR (both Phonetic & Word tracks).
#                  Track-specific config.py imports everything from here via:
#                    from src.config_base import *
#                  then overrides MODEL_NAME, train_file, label_column, etc.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
import torch

RUN_VERSION = '1'

# ---------- project paths ----------
flags.DEFINE_string('track', 'phonetic', 'Track: phonetic or word')
flags.DEFINE_string('SUFFIX', '', '')
flags.DEFINE_string('root', None, '')

# ---------- backbone / model ----------
flags.DEFINE_string('backbone', 'openai/whisper-large-v3', 'pretrained ASR model name or path')
flags.DEFINE_alias('bb', 'backbone')
flags.DEFINE_string('model', '', 'model name under src/models/ (auto-inferred from backbone if empty)')
flags.DEFINE_bool('nemo_native_preprocess', False,
                  'When True, skip Wav2Vec2FeatureExtractor normalization for NeMo backbones. '
                  'NeMo models expect raw (un-normalized) waveforms; the default '
                  'do_normalize=True changes the mel spectrum and degrades WER by ~3%%. '
                  'Enable this to match NeMo model.transcribe() preprocessing exactly.')
flags.DEFINE_alias('nnp', 'nemo_native_preprocess')

# ---------- audio ----------
flags.DEFINE_integer('sample_rate', 16000, 'audio sample rate')
flags.DEFINE_float('max_audio_sec', None, 'max audio duration in seconds; 0 means each track sets its own default')
flags.DEFINE_bool('random_crop', False,
                  'When audio > max_audio_sec, take a random crop instead of '
                  'truncating from the start (train only). Acts as positional aug.')
flags.DEFINE_float('crop_prob', 1.0,
                   'Probability of random crop vs head-truncate when random_crop '
                   'is enabled. 1.0 = always random crop; 0.8 = 80% crop / 20% head.')
flags.DEFINE_bool('crop_label', False,
                  'When long audio is truncated/cropped to max_audio_sec during training, '
                  'also truncate labels proportionally. Supports both head-truncate and '
                  'random_crop. Default False = keep full label.')
flags.DEFINE_bool('filter_long_audio', False,
                  'Exclude (not truncate) training samples with audio > max_audio_sec. '
                  'Official baseline filters >25s clips. Default False = truncate/crop.')
flags.DEFINE_bool('eval_truncate_audio', False,
                  'Also truncate eval/test audio to max_audio_sec (head-truncate, no random crop). '
                  'Default False = eval uses full-length audio. '
                  'Enable when using slow autoregressive decoders (TDT/RNNT) to speed up eval.')
flags.DEFINE_alias('evt', 'eval_truncate_audio')
flags.DEFINE_bool('use_soundfile', True, 'use soundfile (faster) instead of librosa for audio loading')

# ---------- generation ----------
flags.DEFINE_integer('num_beams', 5, 'beam search width')
flags.DEFINE_float('length_penalty', 1.0, '')
flags.DEFINE_integer('max_new_tokens', 128, 'max new tokens for generation')
flags.DEFINE_integer('no_repeat_ngram_size', 3, 'prevent repeated n-grams in generation (0=disabled)')
flags.DEFINE_string('language', 'en', 'language for Whisper generation')
flags.DEFINE_string('task', 'transcribe', 'task for Whisper')

# ---------- duration-based output limiting ----------
flags.DEFINE_float('max_tokens_per_sec', 0,
                   'Max label tokens per second of audio duration during training '
                   '(0=disabled). Prevents seq2seq from learning to hallucinate on short audio.')
flags.DEFINE_float('max_words_per_sec', 5,
                   'Max output words per second of audio duration for post-processing '
                   'truncation at submit time (0=disabled). Words split by whitespace.')

# ---------- IPA conversion (phonetic track) ----------
flags.DEFINE_string('ipa_method', 'eng_to_ipa', 'IPA conversion method: eng_to_ipa, g2p, direct')
flags.DEFINE_bool('use_cmudict_fallback', True, 'fallback to CMUdict for unknown words')

# ---------- training strategy ----------
flags.DEFINE_bool('freeze_encoder', False, 'freeze Whisper encoder, only train decoder (saves ~50% memory)')
flags.DEFINE_float('unfreeze_epoch', None,
                   'Two-stage training: freeze encoder for the first N epochs, then '
                   'automatically unfreeze. Implies --freeze_encoder. '
                   'EMA is reset at unfreeze to avoid contamination from frozen stage. '
                   'None = disabled (single-stage training).')

# NeMo adapter training — matches official benchmark adapter approach.
# Inserts lightweight LinearAdapter modules into the frozen model;
# only adapter weights (~0.26 % params) are trainable.
flags.DEFINE_bool('nemo_adapter', False,
                  'Enable NeMo adapter training. Freezes the entire base model '
                  'and inserts small LinearAdapter modules into every encoder layer. '
                  'Overrides freeze_encoder (whole model is frozen, then adapters unfrozen).')
flags.DEFINE_integer('adapter_dim', 32,
                     'Hidden dimension of the LinearAdapter bottleneck. '
                     'Official benchmark uses 32. Smaller = fewer params, larger = more capacity.')
flags.DEFINE_string('adapter_name', 'asr_children',
                    'Globally unique name for the adapter. '
                    'Official benchmark uses "asr_children_orthographic".')
flags.DEFINE_string('adapter_module_name', 'encoder',
                    'Which module(s) to insert adapters into. '
                    '"encoder" (default), "decoder", "joint", or combine with "+".')

# Whisper LoRA (PEFT) — parameter-efficient fine-tuning for HuggingFace Whisper.
# Inserts low-rank adapters into attention layers; only LoRA weights are trainable.
# NOTE: lora_r, lora_alpha are already defined in melt/apps/config.py.
flags.DEFINE_bool('whisper_lora', False,
                  'Enable LoRA (PEFT) for Whisper models. Freezes the entire '
                  'base model and inserts LoRA adapters into attention layers. '
                  'Requires: pip install peft')
flags.DEFINE_float('lora_dropout', 0.05,
                   'Dropout applied to LoRA layers during training.')
flags.DEFINE_string('lora_target_modules', 'q_proj,v_proj',
                    'Comma-separated list of attention modules to apply LoRA. '
                    'Default targets query and value projections. '
                    'Options: q_proj,k_proj,v_proj,out_proj,fc1,fc2')

# NeMo LoRA (PEFT) — parameter-efficient fine-tuning for NeMo Conformer/Parakeet.
# Unlike NeMo adapter (frozen encoder + tiny residual), LoRA inserts low-rank
# matrices into encoder attention/FF layers. LoRA weights CAN be merged back
# into the base model, producing a full-rank updated encoder for downstream use.
flags.DEFINE_bool('nemo_lora', False,
                  'Enable LoRA (PEFT) for NeMo encoder. Freezes the entire '
                  'base model and inserts LoRA adapters into encoder Linear layers. '
                  'Key advantage over nemo_adapter: weights can be merged back, '
                  'producing a full-rank encoder for downstream tasks. '
                  'Requires: pip install peft')
flags.DEFINE_string('nemo_lora_target_modules', 'linear_q,linear_k,linear_v,linear_out',
                    'Comma-separated list of NeMo encoder module names to apply LoRA. '
                    'Parakeet Conformer options: linear_q, linear_k, linear_v, '
                    'linear_out, linear_pos (attention); linear1, linear2 (FFN). '
                    'Default targets attention Q/K/V/Out (~16% encoder params).')
flags.DEFINE_bool('nemo_lora_merge_on_save', True,
                  'Merge LoRA weights back into base model when saving. '
                  'Produces a standard NeMo checkpoint usable as --backbone for '
                  'downstream tasks (e.g. word pretrain -> phonetic fine-tune). '
                  'When False, saves with LoRA adapters (smaller but requires PEFT to load).')

flags.DEFINE_bool('llrd', False,
                  'Enable Layer-wise Learning Rate Decay (LLRD). '
                  'Lower encoder layers get smaller LR, upper layers get larger LR. '
                  'Widely used for fine-tuning pretrained Transformers.')
flags.DEFINE_float('llrd_decay', 0.9,
                   'LLRD decay factor per layer (from top to bottom). '
                   'E.g. 0.9 means each lower layer gets 0.9x the LR of the layer above. '
                   'Typical range: 0.8~0.95. Whisper-large: try 0.85~0.9.')

# ---------- CTC / hybrid (ctc_weight: 0=seq2seq, 0~1=hybrid, 1=ctc_only) ----------
flags.DEFINE_float('ctc_weight', 0.0, 'CTC loss weight: 0=pure seq2seq, 0~1=hybrid, 1=pure CTC')
flags.DEFINE_float('ctc_dropout', 0.1, 'dropout before CTC projection')
flags.DEFINE_bool('nemo_native_ctc', False,
                  'Use NeMo native CTC decoder for training & inference (方案B). '
                  'Reuses NeMo\'s pretrained SentencePiece CTC head when the loaded backbone '
                  'exposes one (pure CTC models via `decoder`, hybrid TDT+CTC models via '
                  '`ctc_decoder`) instead of building a random-initialized project CTC head. '
                  'Labels are re-tokenized with NeMo SentencePiece at training time. '
                  'If the backbone is TDT/RNNT-only or the task uses a custom IPA vocab, '
                  'the code safely falls back to the project CTC head. '
                  'Requires --ctc_weight>0 (implied: auto-set to 1.0 if 0).')

# -- InterCTC: intermediate-layer CTC regularization (ESPnet-style) --
flags.DEFINE_bool('inter_ctc', False,
                  'Enable InterCTC: apply CTC loss at intermediate encoder layers. '
                  'Acts as strong regularization, typically +5~10% relative improvement.')
flags.DEFINE_list('inter_ctc_layers', [],
                  'Encoder layer indices (0-based) to apply InterCTC. '
                  'E.g. "8,16" for Whisper-large (32 layers). '
                  'If empty and inter_ctc=True, auto-selects layer at 1/2 depth.')
flags.DEFINE_float('inter_ctc_weight', 0.3,
                   'Weight for each InterCTC loss term (summed and added to main CTC loss). '
                   'Total InterCTC contribution = inter_ctc_weight * mean(inter_ctc_losses).')

# -- Focal CTC: focus on hard samples --
flags.DEFINE_bool('focal_ctc', False,
                  'Apply focal loss weighting to CTC loss. '
                  'Down-weights easy samples, focuses on hard ones.')
flags.DEFINE_float('focal_ctc_gamma', 2.0,
                   'Focal CTC gamma: higher = more focus on hard samples. '
                   'gamma=0 is equivalent to standard CTC.')

# -- CTC entropy regularization: prevent over-confident CTC predictions --
flags.DEFINE_float('ctc_entropy_reg', 0.0,
                   'CTC output entropy regularization weight. '
                   'Adds -w * H(p) to the loss, encouraging smoother CTC output '
                   'distributions and reducing over-confidence. '
                   'Acts as a CTC-compatible alternative to label smoothing. '
                   'Typical values: 0.01~0.1. 0 = disabled.')
flags.DEFINE_float('phonetic_label_smoothing', 0.0,
                   'Phonetic label smoothing weight for CTC loss (IPA char-level only). '
                   'Blurs CTC output probabilities using phonetic similarity: '
                   'similar phones (e.g. /p/ ↔ /b/) share probability mass. '
                   'Applied as: smooth_prob = (1-α)*prob + α*(prob @ S) where S is '
                   'a phonetic similarity matrix. Typical values: 0.05~0.2. 0 = disabled.')
flags.DEFINE_alias('pls', 'phonetic_label_smoothing')

# -- CTC layer fusion: feed CTC head with weighted sum of multiple encoder layers --
flags.DEFINE_list('ctc_layer_fusion', [],
                  'Encoder layer indices (0-based) for learnable weighted fusion into CTC head. '
                  'E.g. "20,22,23" for last 4 layers of Whisper-large (24 layers). '
                  'Empty = use last layer only (default). '
                  'Learns scalar weights (softmax-normalized) per layer, like ELMo/wav2vec2. '
                  'Middle layers carry richer acoustic/phonetic info vs. final layer.')
flags.DEFINE_integer('ctc_fusion_last_n', None,
                     'Shortcut: fuse the last N encoder layers. '
                     'E.g. 4 = last 4 layers (-4,-3,-2,-1). '
                     'Overrides ctc_layer_fusion if set.')

# ---------- MCER: Minimum Character Error Rate training ----------
flags.DEFINE_float('mcer_weight', 0.0,
                   'MCER (Minimum CER) loss weight. '
                   'When >0, adds REINFORCE-style MCER loss to directly optimize CER. '
                   'Total loss = ctc_loss + mcer_weight * mcer_loss. '
                   'Model should be pre-trained with CTC first. '
                   'Typical values: 0.1~0.3. 0 = disabled.')
flags.DEFINE_integer('mcer_nbest', 8,
                     'Number of N-best hypotheses for MCER training. '
                     'More = lower variance but slower. Typical: 4~16.')
flags.DEFINE_integer('mcer_beam_size', 16,
                     'Beam size for MCER N-best generation (should be >= mcer_nbest). '
                     'Wider beam = more diverse hypotheses.')
flags.DEFINE_integer('mcer_start_epoch', 0,
                     'Epoch to start MCER training. '
                     'Set >0 to warm-start with pure CTC first.')

# ---------- MWER: Minimum Word Error Rate training ----------
flags.DEFINE_float('mwer_weight', 0.0,
                   'MWER (Minimum WER) loss weight. '
                   'When >0, adds REINFORCE-style MWER loss to directly optimize WER '
                   'for non-IPA CTC models. Kept 0 by default so existing training '
                   'flows remain unchanged.')
flags.DEFINE_integer('mwer_nbest', 8,
                     'Number of N-best hypotheses for MWER training. '
                     'More = lower variance but slower. Typical: 4~16.')
flags.DEFINE_integer('mwer_beam_size', 16,
                     'Beam size for MWER N-best generation (should be >= mwer_nbest). '
                     'Wider beam = more diverse hypotheses.')
flags.DEFINE_integer('mwer_start_epoch', 0,
                     'Epoch to start MWER training. '
                     'Set >0 to warm-start with pure CTC first.')

# ---------- CTC decoding ----------
flags.DEFINE_integer('ctc_beam_width', 1, 'CTC beam width: 1=greedy, >1=prefix beam search')
flags.DEFINE_float('ctc_lm_weight', 0.0, 'language model weight for CTC beam search (0=no LM)')
flags.DEFINE_string('ctc_lm_path', '', 'path to character n-gram LM file (.json) for CTC decoding')
flags.DEFINE_bool('ctc_decode_fp32', False,
                  'Use float32 for log_softmax in CTC decoding. '
                  'Improves numerical stability for beam search + LM scoring. '
                  'Minimal impact for greedy decode (argmax is order-preserving).')

# ---------- IPA constrained decoding (phonetic track) ----------
flags.DEFINE_bool('constrain_ipa', False,
                  'constrain seq2seq decoder vocabulary to IPA tokens only')

# ---------- SqueezeFormer (--model=squeezeformer) ----------
flags.DEFINE_integer('sf_dim', 256, 'SqueezeFormer hidden dimension')
flags.DEFINE_integer('sf_depth', 12, 'SqueezeFormer number of layers')
flags.DEFINE_integer('sf_heads', 4, 'SqueezeFormer attention heads')
flags.DEFINE_integer('sf_ff_mult', 4, 'SqueezeFormer feed-forward multiplier')
flags.DEFINE_integer('sf_conv_kernel', 31, 'SqueezeFormer convolution kernel size')
flags.DEFINE_float('sf_attn_dropout', 0.1, 'SqueezeFormer attention dropout')
flags.DEFINE_float('sf_ff_dropout', 0.1, 'SqueezeFormer feed-forward dropout')
flags.DEFINE_float('sf_conv_dropout', 0.1, 'SqueezeFormer convolution dropout')

# ---------- Wav2Vec2 / HuBERT (--model=wav2vec2) ----------
flags.DEFINE_string('tokenizer_backbone', 'openai/whisper-large-v3',
                    'backbone for text tokenizer (used when encoder backbone has no tokenizer). '
                    'Common choices and vocab sizes: '
                    'whisper-large-v3/openai/whisper-large-v3=50257, '
                    'whisper-large-v3-turbo/openai/whisper-large-v3-turbo=50257, '
                    'hubert-large/facebook/hubert-large-ls960-ft=32, '
                    'wav2vec2-large-960h/facebook/wav2vec2-large-960h=32, '
                    'parakeet-ctc-1.1b/nvidia/parakeet-ctc-1.1b=1024. '
                    'wavlm/data2vec encoder checkpoints do not expose a native text tokenizer, '
                    'so they are usually not useful as tokenizer_backbone choices.')
flags.DEFINE_bool('freeze_feature_extractor', True,
                  'freeze wav2vec2/hubert CNN feature extractor (recommended)')
flags.DEFINE_bool('init_ctc_from_pretrained', False,
                  'Initialize CTC head weights from pretrained ForCTC model via vocab mapping. '
                  'Only works with espeak-IPA backbones (wav2vec2-espeak, xlsr-espeak). '
                  'Maps 51/52 IPA chars from 392-class espeak vocab to our 53-class IPA vocab.')
flags.DEFINE_bool('raw_ctc_eval', False,
                  'Use a HuggingFace native ForCTC checkpoint directly for eval-time CTC '
                  'logits/decoding instead of the project CTC head. Intended for word-track '
                  'raw pretrained evaluation such as facebook/wav2vec2-large-960h or '
                  'facebook/hubert-large-ls960-ft. Default False keeps all existing behaviour unchanged.')
flags.DEFINE_bool('native_ctc', False,
                  'Load HuggingFace ForCTC checkpoint (encoder + lm_head) and fine-tune the '
                  'complete native CTC model. Unlike raw_ctc_eval (eval-only), this supports '
                  'training with feature-extractor freezing, gradient checkpointing, etc. '
                  'Use with --backbone=hubert-large --ctc_weight=1 --decode_method=ctc.')

# ---------- data augmentation (train only) ----------
flags.DEFINE_bool('aug', False, 'enable audio data augmentation (master switch)')
flags.DEFINE_bool('aug_show_once', True,
                  'When True, print one concise debug record the first time each '
                  'augmentation actually fires. Useful to verify SpecAugment / '
                  'mix / crop behavior without flooding logs.')
# -- waveform-level augmentation (before mel extraction) --
flags.DEFINE_bool('aug_speed', False, 'speed perturbation (0.9x~1.1x)')
flags.DEFINE_float('aug_speed_min', 0.9, 'min speed factor')
flags.DEFINE_float('aug_speed_max', 1.1, 'max speed factor')
flags.DEFINE_float('aug_speed_prob', 0.5, 'probability of applying speed perturbation')
flags.DEFINE_bool('aug_speed_short_only', False,
                  'only apply speed perturbation to short audio (<= aug_speed_short_dur seconds)')
flags.DEFINE_float('aug_speed_short_dur', 2.0,
                   'duration threshold (seconds) for aug_speed_short_only')
flags.DEFINE_bool('aug_noise', False, 'additive Gaussian noise')
flags.DEFINE_float('aug_noise_snr_min', 10.0, 'min SNR in dB for noise')
flags.DEFINE_float('aug_noise_snr_max', 40.0, 'max SNR in dB for noise')
flags.DEFINE_float('aug_noise_prob', 0.3, 'probability of adding noise')
flags.DEFINE_bool('aug_volume', False, 'random volume/gain perturbation')
flags.DEFINE_float('aug_volume_min', 0.8, 'min gain factor')
flags.DEFINE_float('aug_volume_max', 1.2, 'max gain factor')
flags.DEFINE_float('aug_volume_prob', 0.5, 'probability of volume perturbation')
flags.DEFINE_bool('aug_pitch', False, 'pitch shift (slower, disabled by default)')
flags.DEFINE_integer('aug_pitch_range', 2, 'max pitch shift in semitones (±)')
flags.DEFINE_float('aug_pitch_prob', 0.3, 'probability of pitch shift')
flags.DEFINE_bool('aug_resample', False,
                  'Resample augmentation: downsample to low sr then upsample back. '
                  'Simulates low-quality recording equipment by losing high-frequency info. '
                  'Inspired by Bengali.AI 1st place (16k→8k→16k).')
flags.DEFINE_integer('aug_resample_sr', 8000,
                     'Target sample rate for downsample step (default 8000). '
                     'Lower = more aggressive quality degradation.')
flags.DEFINE_float('aug_resample_prob', 0.5, 'probability of applying resample augmentation')
# -- classroom noise augmentation (RealClass noise data) --
flags.DEFINE_bool('aug_classroom_noise', False,
                  'Add real classroom background noise from RealClass noise dataset. '
                  'Much more realistic than Gaussian noise for noisy classroom WER.')
flags.DEFINE_string('noise_dir', '../input/childrens-classnoise-asr/audio',
                    'Directory containing noise .flac files (from RealClass dataset). '
                    'Required when aug_classroom_noise is enabled.')
flags.DEFINE_float('aug_classroom_noise_prob', 0.5,
                   'probability of adding classroom noise per sample')
flags.DEFINE_float('aug_classroom_snr_min', 10.0,
                   'min SNR in dB for classroom noise mixing (lower = noisier)')
flags.DEFINE_float('aug_classroom_snr_max', 20.0,
                   'max SNR in dB for classroom noise mixing')
flags.DEFINE_bool('aug_classroom_noise_only', False,
                  'When True, ONLY use classroom noise (disable Gaussian noise). '
                  'When False, both can be applied independently.')
# -- spectrogram-level augmentation (after mel extraction, Whisper only) --
flags.DEFINE_bool('aug_spec', False, 'SpecAugment: frequency + time masking on mel')
flags.DEFINE_bool('aug_nemo_spec', False,
                  'Apply SpecAugment on NeMo mel features after NeMo preprocessor '
                  'and before encoder. Separate from aug_spec so existing experiments '
                  'are unaffected.')
flags.DEFINE_float('aug_nemo_spec_time_ratio', 0.0,
                   'Optional NeMo SpecAugment time-mask ratio in [0, 1]. '
                   'When > 0, time mask width is set as a fraction of each sample\'s '
                   'post-preprocessor frame length, instead of the fixed aug_time_mask. '
                   'Default 0 keeps existing fixed-width behavior unchanged.')
flags.DEFINE_integer('aug_freq_mask', 15, 'SpecAugment: max frequency mask width')
flags.DEFINE_alias('freq_masks', 'aug_freq_mask')
flags.DEFINE_integer('aug_freq_num', 2, 'SpecAugment: number of frequency masks')
flags.DEFINE_integer('aug_time_mask', 20, 'SpecAugment: max time mask width')
flags.DEFINE_alias('time_masks', 'aug_time_mask')
flags.DEFINE_integer('aug_time_num', 2, 'SpecAugment: number of time masks')
# -- random proportion masking (inspired by ASLFR) --
flags.DEFINE_bool('aug_temporal_mask', False, 'random temporal (time-step) masking by proportion')
flags.DEFINE_float('aug_temporal_mask_prob', 0.15, 'fraction of time steps to mask (or fixed prob)')
flags.DEFINE_list('aug_temporal_mask_range', [], 'if set, sample mask prob uniformly from [lo, hi]')
flags.DEFINE_bool('aug_spatio_mask', False, 'random spatial (frequency-channel) masking by proportion')
flags.DEFINE_float('aug_spatio_mask_prob', 0.15, 'fraction of frequency channels to mask')
flags.DEFINE_bool('aug_st_mask', False, 'random spatio-temporal 2D masking (each mel cell independently)')
flags.DEFINE_float('aug_st_mask_prob', 0.15, 'fraction of mel cells to mask')
flags.DEFINE_list('aug_st_mask_range', [], 'if set, sample mask prob uniformly from [lo, hi]')
flags.DEFINE_bool('aug_cutmix', False,
                  'CutMix-style mel augmentation: replace a random mel patch with another patch from the same sample')
flags.DEFINE_float('aug_cutmix_prob', 0.3, 'probability of applying aug_cutmix')
flags.DEFINE_integer('aug_cutmix_num', 1, 'number of cutmix patch operations per sample')
flags.DEFINE_list('aug_cutmix_time_ratio', ['0.05', '0.2'],
                  'time patch ratio range [lo, hi] for aug_cutmix')
flags.DEFINE_list('aug_cutmix_freq_ratio', ['0.05', '0.2'],
                  'frequency patch ratio range [lo, hi] for aug_cutmix')
# -- Cross-sample CutMix (Priority 1): splice two samples along time axis --
flags.DEFINE_bool('aug_xcutmix', False,
                  'Cross-sample CutMix: splice mel spectrograms from two different samples '
                  'along time axis with corresponding label concatenation. '
                  'Works with CTC (monotonic alignment).')
flags.DEFINE_float('aug_xcutmix_prob', 0.5, 'probability of applying cross-sample CutMix per sample')
flags.DEFINE_list('aug_xcutmix_ratio', ['0.3', '0.7'],
                  'cut ratio range [lo, hi] for the first sample in cross-sample CutMix')
flags.DEFINE_bool('aug_xcutmix_same_child', False,
                  'When True, cross-sample CutMix only mixes samples from the same child_id / speaker. '
                  'When False (default), randomly sample from the entire training set.')
# -- Alignment-aware SpliceMix (Priority 2): splice at phoneme boundaries --
flags.DEFINE_bool('aug_splicemix', False,
                  'SpliceMix: alignment-aware cross-sample CutMix that splices at phoneme boundaries '
                  'using pre-computed CTC forced alignment. Requires --alignment_file.')
flags.DEFINE_float('aug_splicemix_prob', 0.5, 'probability of applying SpliceMix per sample')
flags.DEFINE_string('alignment_file', '',
                    'Path to pre-computed alignment file (pickle). '
                    'Generated by gen_alignment.py. Required for --aug_splicemix.')
flags.DEFINE_bool('aug_splicemix_same_child', False,
                  'When True, SpliceMix only mixes samples from the same child_id / speaker.')
flags.DEFINE_bool('aug_splicemix_crossfade', True,
                  'Apply short crossfade at splice boundary to smooth transition.')
flags.DEFINE_integer('aug_splicemix_crossfade_frames', 3,
                     'Number of frames for crossfade at splice boundary.')
# -- VTLN (Vocal Tract Length Normalization): frequency-axis warping --
flags.DEFINE_bool('aug_vtln', False,
                  'VTLN augmentation: warp frequency axis by random alpha to simulate '
                  'different vocal tract lengths. Particularly effective for children speech '
                  '(5-10% relative improvement). Works on both waveform (NeMo/wav2vec2) '
                  'and mel (Whisper) pipelines.')
flags.DEFINE_float('aug_vtln_prob', 0.5, 'probability of applying VTLN per sample')
flags.DEFINE_float('aug_vtln_alpha_min', 0.85,
                   'min warping factor (< 1.0 = longer vocal tract / lower formants)')
flags.DEFINE_float('aug_vtln_alpha_max', 1.15,
                   'max warping factor (> 1.0 = shorter vocal tract / higher formants)')
# -- Concat augmentation (MixAug): concatenate random samples at waveform level --
flags.DEFINE_bool('aug_mix', False,
                  'Concat augmentation: randomly concatenate another sample\'s audio '
                  'to the current sample (waveform-level). Labels are joined with a space. '
                  'Helps the model learn word boundaries when training data has many single-word samples.')
flags.DEFINE_float('aug_mix_prob', 0.5, 'Probability of applying concat augmentation per sample.')
flags.DEFINE_integer('aug_mix_num', 1,
                     'Number of extra samples to concatenate (1 = pair, 2 = triplet, etc.).')
flags.DEFINE_bool('aug_mix_cross_source', False,
                  'When True, aug_mix forces cross-source pairing: DD samples mix with EXT '
                  'and EXT samples mix with DD. When False (default), partner is random.')
flags.DEFINE_bool('aug_mix_same_source', False,
                  'When True, aug_mix only mixes within the same source: DD+DD, EXT+EXT. '
                  'Especially useful for ext+ext to create multi-word pseudo-sentences.')
flags.DEFINE_bool('aug_mix_same_session', False,
                  'When True, aug_mix only mixes samples from the same session_id. '
                  'Same session = same child + same mic + same room + same noise floor. '
                  'Creates the most natural-sounding concatenations. Priority: same_session > same_child > same_source > cross_source.')
flags.DEFINE_bool('aug_mix_same_child', False,
                  'When True, aug_mix only mixes samples from the same child_id / speaker.')
flags.DEFINE_bool('aug_mix_random_num', False,
                  'When True, the number of extra samples to concatenate is sampled '
                  'uniformly from [1, aug_mix_num] instead of fixed aug_mix_num. '
                  'Creates more diversity: sometimes pair, sometimes triplet, etc.')
flags.DEFINE_bool('aug_mix_shuffle', False,
                  'When True, shuffle the order of all segments (original + partners) '
                  'before concatenation. E.g. A+B may become B+A, A+B+C may become C+A+B. '
                  'Acts as positional augmentation for CTC training.')
flags.DEFINE_string('aug_mix_strategy', '',
                    'Unified mixing strategy (overrides aug_mix_cross_source/same_child/... bool flags). '
                    'Single: "cross_source". '
                    'Multi equal-prob: "cross_source,same_child" (50/50). '
                    'Multi weighted: "cross_source:0.6,same_child:0.4" (60/40). '
                    'Valid strategies: random, cross_source, same_source, same_child, same_session. '
                    'Empty = use legacy bool flags.')
flags.DEFINE_bool('aug_mix_dd_only', False,
                  'When True, aug_mix only applies to DD samples. '
                  'EXT samples skip concat augmentation entirely.')
flags.DEFINE_bool('aug_mix_ext_only', False,
                  'When True, aug_mix only applies to EXT samples. '
                  'DD samples skip concat augmentation entirely.')
flags.DEFINE_string('aug_mix_dd_strategy', '',
                    'Override mixing strategy for DD samples. '
                    'Same format as aug_mix_strategy: "cross_source", "cross_source,same_child", '
                    '"cross_source:0.6,same_child:0.4". '
                    'Empty = use aug_mix_strategy or global bool flags.')
flags.DEFINE_string('aug_mix_ext_strategy', '',
                    'Override mixing strategy for EXT samples. '
                    'Same format as aug_mix_strategy: "cross_source", "cross_source,same_child", '
                    '"cross_source:0.6,same_child:0.4". '
                    'Empty = use aug_mix_strategy or global bool flags.')
flags.DEFINE_float('aug_mix_max_dur', 0,
                  'Only apply aug_mix when source audio duration <= this value (seconds). '
                  '0 = disabled (all samples eligible). '
                  'Bengali 1st-place style: only merge short audios, leave long ones alone.')
flags.DEFINE_bool('aug_mix_limit_len', False,
                  'Stop adding segments in aug_mix once total audio would exceed max_audio_sec (break). '
                  'Prevents label-audio length mismatch caused by audio truncation with full label kept.')
flags.DEFINE_bool('aug_mix_fit_len', False,
                  'Skip (not break) partners that would push total audio beyond max_audio_sec, '
                  'continue trying other shorter candidates. '
                  'Prevents same label-audio mismatch as aug_mix_limit_len but tries harder to fill.')
flags.DEFINE_bool('aug_mix_fit_label', False,
                  'Skip (not break) partners that would push mixed primary-label units beyond '
                  'aug_mix_max_label_units, then continue trying other candidates. '
                  'Default False keeps existing aug_mix behaviour unchanged.')
flags.DEFINE_integer('aug_mix_max_label_units', 0,
                     'Maximum primary-label units allowed after aug_mix concatenation. '
                     '0 = disabled. Word track uses whitespace word count; phonetic track '
                     'uses non-space character count as a lightweight proxy for TDT/RNNT label length.')
flags.DEFINE_bool('aug_mix_fit_cost', False,
                  'Skip (not break) partners that would push mixed aug_mix cost beyond '
                  'aug_mix_max_cost, then continue trying other candidates. '
                  'Cost proxy = mixed_audio_sec * mixed_label_units. Default False.')
flags.DEFINE_float('aug_mix_max_cost', 0,
                   'Maximum mixed aug_mix cost allowed when aug_mix_fit_cost is enabled. '
                   '0 = disabled. Cost proxy = mixed_audio_sec * mixed_label_units, '
                   'useful for avoiding rare TDT/RNNT OOM tails without changing batch size.')
# Duration-aware concat: target-duration-based partner selection
flags.DEFINE_bool('aug_mix_dur_aware', False,
                  'Duration-aware concat augmentation: sample a target total duration '
                  'from the training data distribution, then greedily pick partners '
                  'whose durations fill the remaining time. Creates more natural '
                  'duration distribution for concatenated utterances. '
                  'Overrides fixed partner count; aug_mix_num serves as max partners cap.')
flags.DEFINE_float('aug_mix_target_dur_pmin', 0.1,
                   'Percentile (0-1) to estimate typical short/1-word sample duration. '
                   'This sets the minimum addition: target_min = cur_dur + percentile(durs, pmin). '
                   'Default 0.1 = p10 of training durations (~typical 1-word length). '
                   'Guarantees at least one partner of this duration gets added.')
flags.DEFINE_float('aug_mix_target_dur_pmax', 0.95,
                   'Percentile (0-1) for upper bound of target total duration. '
                   'target_max = percentile(durs, pmax). Default 0.95 = p95. '
                   'Target is sampled from [cur_dur + dur_short, dur_pmax], '
                   'clamped to max_audio_sec. When cur_dur already near pmax, '
                   'falls back to adding one random short partner.')
flags.DEFINE_bool('aug_mix_dur_sample', False,
                  'Sample target duration from the actual training data distribution '
                  'instead of uniform(target_min, target_max). '
                  'Randomly picks a real sample duration as target (clamped to >= cur_dur + dur_short). '
                  'Produces more natural concat lengths (many short, few long). '
                  'Requires aug_mix_dur_aware=True.')
flags.DEFINE_bool('aug_mix_debug', False,
                  'Log aug_mix details for first N samples: original duration, '
                  'number of partners, each partner duration, final total duration, strategy.')
flags.DEFINE_integer('aug_mix_debug_count', 50,
                     'Number of samples to log when aug_mix_debug is enabled.')

# ---------- data ----------
flags.DEFINE_string('train_file', '', 'training data file (set by track config)')
flags.DEFINE_string('label_column', '', 'label column name (set by track config)')
flags.DEFINE_string('label_column_fallback', '', 'fallback label column')
flags.DEFINE_string('score_metric', '', 'scoring metric: ipa_cer or wer (set by track config)')

flags.DEFINE_integer('samples', 0, 'number of samples to use (0 = all)')
flags.DEFINE_integer('eval_samples', 0, 'number of samples to use (0 = all)')


flags.DEFINE_bool('sort_by_duration', True, 'sort by audio duration for batching')
flags.DEFINE_bool('bucket_batch', False,
                  'Use BucketBatchSampler for training: group similar-length audio '
                  'into the same batch (reduces padding waste, stabilises GPU memory). '
                  'Buckets are shuffled across batches each epoch.')
flags.DEFINE_string('bucket_batch_key', 'audio',
                    'Length proxy used by bucket_batch. '
                    '"audio" = audio_duration_sec only (legacy behaviour). '
                    '"rnnt_cost" = audio_duration_sec * target_units, useful for '
                    'RNNT/TDT where memory scales with both audio length and label length.')
flags.DEFINE_integer('bucket_batch_debug_batches', 3,
                     'How many bucketed training batches to preview in logs at startup. '
                     '0 disables preview. Useful to verify bucket sampler behaviour.')
flags.DEFINE_bool('stress_test_memory', False,
                  'Memory stress test: sort training batches LONGEST-FIRST so the '
                  'first few steps hit worst-case memory. If no OOM in the first '
                  'few batches, the rest of training is safe. Overrides bucket_batch '
                  'and shuffle. Intended for quick BS tuning — run 1-2 minutes then Ctrl-C.')
flags.DEFINE_string('fold_group_key', '', 'group key for GroupKFold (e.g. child_id)')
flags.DEFINE_string('fold_stratify_key', '', 'stratify key for StratifiedGroupKFold (e.g. age_bucket)')
flags.DEFINE_string('sgkf_compat', '1.6.1',
                    'StratifiedGroupKFold compatibility mode. '
                    '"1.6.1" = sklearn <=1.6.1 buggy shuffle (default, backward compat). '
                    '"1.8.0" = sklearn >=1.8.0 fixed shuffle (correct even folds). '
                    'Empty string = use installed sklearn version.')
flags.DEFINE_string('fold_align_file', '',
                    'JSON file mapping group_key -> fold (e.g. child_id -> fold). '
                    'Used for cross-track pretraining: force overlapping children to '
                    'keep the same fold assignment as the other track, preventing '
                    'data leakage. Generate with tools/gen_fold_align.py.')
flags.DEFINE_string('cross_track_child2fold_file', '',
                    'JSON file mapping child_id -> fold from another track '
                    '(for example phonetic fold mapping). Unlike fold_align_file, '
                    'this is used only to filter unsafe eval children when the '
                    'current track uses a different number of folds.')
flags.DEFINE_integer('pretrain_fold', -1,
                     'Fold id of the source-track pretrain model. Used with '
                     'cross_track_child2fold_file to keep overlap children out of '
                     'the current eval fold unless they belonged to the matching '
                     'source-track eval fold. -1 disables the guard.')
flags.DEFINE_bool('cross_track_eval_safe_only', False,
                  'When True, apply a current-run eval guard using '
                  'cross_track_child2fold_file: overlap children whose source-track '
                  'fold != pretrain_fold are excluded from the current eval fold and '
                  'become train-only for this run. This allows word-track 10/20-fold '
                  'CV while still avoiding leakage from 5-fold phonetic pretraining.')

# ---------- extended (TalkBank) data ----------
flags.DEFINE_bool('use_ext', False, 'Include TalkBank extended data for training')
flags.DEFINE_bool('ext_only', False, 'Only use extended data for training')
flags.DEFINE_bool('train_ext_only', False,
                  'Train on EXT data only (filter out DD from training). '
                  'Eval remains on DD (unchanged). Requires --use_ext.')
flags.DEFINE_bool('eval_ext', False,
                  'Include ext data in CV evaluation. '
                  'When False (default): CV only on DrivenData, ext data always in train. '
                  'When True: merge DD+ext then do normal fold split on combined data.')
flags.DEFINE_bool('eval_add_ext', False,
                  'Add sampled EXT data to eval set (no CV on ext). '
                  'Eval: DD eval fold + N=len(DD eval) ext sampled with seed=42. '
                  'Train: DD non-eval folds + ALL remaining ext (not sampled for eval). '
                  'online mode: train = full DD + full ext, eval unchanged.')
flags.DEFINE_bool('ext_eval_group', False,
                  'Group ext eval by child_id to prevent train/eval child leakage. '
                  'When True: ext is fold-split by GroupKFold(child_id), '
                  'eval ext = fold==FLAGS.fold (sampled to n_dd unless eval_ext_full), '
                  'train ext = fold!=FLAGS.fold. '
                  'Requires eval_add_ext=True. Default False for backward compat.')
flags.DEFINE_bool('eval_ext_full', False,
                  'Use ALL ext samples in eval fold instead of downsampling to n_dd. '
                  'Requires ext_eval_group=True. Useful for tree reranker training '
                  'with more eval samples. Use eval_ext_weight to control score balance.')
flags.DEFINE_float('eval_ext_weight', 1.0,
                   'Weight for EXT score in the combined eval metric. '
                   'score = (score_dd + eval_ext_weight * score_ext) / (1 + eval_ext_weight). '
                   'Default 1.0 = equal weight (macro-average DD and EXT). '
                   'Set 0.0 to use DD-only score. Set 0.5 to down-weight EXT.'
                   'Useful when eval_add_ext=True to balance the influence of EXT samples on model selection.'
                   'for online probe its 0.764 vs 0.236 so set 3.2373 or 3 may be fine'
                   )
flags.DEFINE_bool('official_data', False,
                  'Use official baseline data split: ALL data (DD+EXT) minus eval IDs for training, '
                  'no fold-based split. Matches standalone official-baseline.py exactly. '
                  'Eval set = official 1840 DD utterances. Default False = use fold-based CV.')
flags.DEFINE_string('ext_root', '../input/childrens-ext-asr', 'Root dir for extended (TalkBank) data (contains audio/ and train JSONL)')
flags.DEFINE_string('eval_ext_root', '../input/childrens-ext-asr',
                   'Root dir for EXT data used in eval (eval_add_ext). '
                   'When ext_root is overridden (e.g. pseudo-labels), eval still uses real labels from here.')
flags.DEFINE_float('ext_weight', 1.0,
                   'Loss weight for EXT data samples (DD samples always weighted 1.0). '
                   'Set <1 (e.g. 0.5, 0.8) to down-weight EXT contribution to loss.')
flags.DEFINE_float('weak_align_weight', 0.0,
                   'Down-weight for weak-alignment samples (short label + long audio). '
                   'When >0, samples with chars_per_sec < weak_align_cps_threshold '
                   'get weight *= weak_align_weight. 0 = disabled (no down-weighting).')
flags.DEFINE_float('weak_align_cps_threshold', 3.0,
                   'chars_per_sec threshold to identify weak-alignment samples. '
                   'Samples below this threshold are down-weighted by weak_align_weight.')
flags.DEFINE_float('ext_sample_rate', 1.0,
                   'Fraction of EXT training data to use per epoch (0~1). '
                   'Uses cycling index-based selection for uniform coverage across epochs.')
flags.DEFINE_integer('ext_sample_seed', 42,
                     'Base seed for ext_sample_rate cycling permutation.')
flags.DEFINE_bool('temperature_sampler', False,
                  'Enable train-only temperature sampling over data groups. '
                  'Default False keeps existing train/eval behaviour unchanged. '
                  'Eval/valid/test are never affected.')
flags.DEFINE_float('temperature_sampler_alpha', 0.5,
                   'Temperature sampling exponent alpha in p_i ∝ n_i^alpha. '
                   '1.0 = original-size sampling, 0.0 = uniform over groups, '
                   '0.5 = sqrt sampling (recommended for mixed-quality corpora).')
flags.DEFINE_string('temperature_sampler_group', 'source',
                    'How to group training rows for temperature sampling. '
                    'Built-ins: source, label_type, source_label. '
                    'Or provide a dataframe column name.')
flags.DEFINE_integer('temperature_sampler_seed', 42,
                     'Base seed for temperature sampler epoch-wise deterministic draws.')
flags.DEFINE_integer('temperature_sampler_epoch_size', 0,
                     'Number of training samples to draw per epoch when temperature '
                     'sampler is enabled. 0 = keep len(train_ds), preserving current '
                     'epoch/batch budgeting as much as possible.')

# ---------- multi-task learning (cross-track) ----------
flags.DEFINE_float('ipa_weight', 1.0,
                   'Loss weight for IPA (phonetic) decoder task. '
                   'Set to 0 to disable IPA loss (e.g. pure word training).')
flags.DEFINE_float('word_weight', 0.0,
                   'Loss weight for word (orthographic) decoder task. '
                   'Set >0 to enable multi-task learning with word auxiliary loss.')
flags.DEFINE_string('word_tokenizer', None,
                    'Tokenizer for primary CTC head (word track). '
                    'None = try backbone native tokenizer, fallback to --default_word_tokenizer. '
                    '"hubert" = HuBERT CTC char tokenizer (32 vocab). '
                    '"nemo"/"parakeet" = NeMo SentencePiece (1024 vocab). '
                    'Any BACKBONES key or HuggingFace model ID also accepted.')
flags.DEFINE_string('default_word_tokenizer', 'whisper',
                    'Fallback tokenizer when --word_tokenizer=None and backbone has no '
                    'native tokenizer. Default "whisper" = Whisper BPE (50257 vocab).')
flags.DEFINE_bool('word_ctc', False,
                  'Use CTC for word auxiliary task instead of S2S (RNNT). '
                  'Default: char-level (29 vocab). Use --word_ctc_bpe for BPE vocab. '
                  'Requires --use_cross_labels --word_weight>0.')
flags.DEFINE_bool('word_ctc_bpe', False,
                  'Use backbone BPE tokenizer for word CTC instead of char-level. '
                  'NeMo: 1024 SentencePiece vocab. Whisper: uses HF tokenizer vocab. '
                  'Requires --word_ctc.')
flags.DEFINE_bool('word_ctc_bpe_add_blank', None,
                  'Whether auxiliary word_ctc_bpe reserves an extra output slot for '
                  'CTC blank (vocab_size + 1). None = auto: word track always uses '
                  'vocab+1; phonetic track uses backend-family compat defaults '
                  '(NeMo=True, wav2vec2-family=False, others fall back to no extra '
                  'blank). Both modes are supported: '
                  'add_blank=True uses an extra output slot and blank=0 with target '
                  'ids shifted by +1; add_blank=False reuses tokenizer id 0 as blank '
                  'and keeps target ids unshifted. Legacy add_blank=False therefore '
                  'requires real text piece ids to avoid 0.')
flags.DEFINE_bool('pseudo_ipa_ctc', False,
                  'Use pseudo-IPA (eng_to_ipa) as auxiliary CTC target instead of word text. '
                  'Auxiliary head uses same 53-class IPA vocab as primary CTC. '
                  'Enables head ensemble at inference (avg logits from both heads). '
                  'Requires --use_cross_labels --word_weight>0 --word_ctc.')
flags.DEFINE_string('pseudo_ipa_file', '',
                    'Path to pre-generated pseudo-IPA JSONL (from gen_pseudo_ipa.py --include_overlap). '
                    'If empty, converts word text to pseudo-IPA on-the-fly via eng_to_ipa. '
                    'File format: one JSON per line with utterance_id + phonetic_text.')
flags.DEFINE_bool('pseudo_ipa_ensemble', False,
                  'At inference, average logits from primary + pseudo-IPA CTC heads. '
                  'Only works when pseudo_ipa_ctc=True (both heads have same 53-class IPA vocab).')
flags.DEFINE_float('pseudo_ipa_ensemble_weight', 0.5,
                   'Weight for pseudo-IPA head in ensemble: logits = (1-w)*primary + w*pseudo_ipa. '
                   'Default 0.5 = equal weighting. Lower values trust primary head more.')
flags.DEFINE_bool('word_tdt_pseudo_ipa', False,
                  'Use pseudo-IPA text from word_label as the auxiliary target for TDT. '
                  'Requires --s2s_decoder=tdt_reuse. Reuses the 53-class IPA vocabulary '
                  'instead of the backbone word BPE vocab.')
flags.DEFINE_bool('word_aux_ipa', False,
                  'Route cross labels on the word track into the existing word auxiliary '
                  'branch, so the auxiliary head predicts 53-class IPA targets instead of '
                  'orthographic word targets. Used by dual_word_ipa / dual_word_tdt_ipa.')
flags.DEFINE_bool('word_tdt_mixed', False,
                  'Use a mixed auxiliary TDT branch for word targets. '
                  'Main IPA TDT decoder stays unchanged; the word branch shares '
                  'pred_rnn/enc_proj/pred_proj with the main TDT decoder, but uses '
                  'its own BPE prediction embedding and joint output layer. '
                  'Requires --s2s_decoder=tdt_reuse.')
flags.DEFINE_bool('word_tdt_share_decoder', None,
                  'Whether word_tdt_pseudo_ipa shares the main tdt_decoder. '
                  'None = auto. dual_tdt_ipa convenience flag sets this to False unless '
                  'explicitly overridden. True = share main decoder. False = create a '
                  'separate word_tdt_decoder initialized from the main TDT decoder.')
flags.DEFINE_alias('wtdt_sd', 'word_tdt_share_decoder')
flags.DEFINE_bool('word_tdt_half_share_decoder', False,
                  'Whether word_tdt_pseudo_ipa uses half-shared decoder weights. '
                  'Shares pred_rnn/enc_proj/pred_proj with the main tdt_decoder, '
                  'but keeps separate pred_embedding and joint_out for the word branch. '
                  'Requires --word_tdt_pseudo_ipa and is mutually exclusive with '
                  '--word_tdt_share_decoder.')
flags.DEFINE_alias('wtdt_hsd', 'word_tdt_half_share_decoder')
flags.DEFINE_bool('word_tdt_pseudo_ipa_only_nonipa', False,
                  'When True, word_tdt_pseudo_ipa only applies to samples without IPA labels '
                  '(mask overlap samples). Independent from --word_only_loss, which remains '
                  'a broader switch for all word auxiliary losses.')
flags.DEFINE_bool('use_cross_labels', False,
                  'Load cross-track labels for overlapping utterances. '
                  'When True, phonetic track also loads orthographic_text and vice versa.')
flags.DEFINE_string('cross_label_file', '',
                    'Path to the cross-track label JSONL file. '
                    'Auto-detected if empty: for phonetic track -> train_word_transcripts.jsonl, '
                    'for word track -> train_phon_transcripts.jsonl.')
flags.DEFINE_bool('use_word_only_dd', False,
                  'Include DD samples that only have word labels (no IPA). '
                  'Useful for multi-task: IPA loss masked, word loss active, encoder still learns.')
flags.DEFINE_float('word_only_dd_sample_rate', 1.0,
                   'Fraction of word-only DD samples to use per epoch (0~1). '
                   'Uses cycling index-based selection for uniform coverage across epochs.')
flags.DEFINE_alias('wod_sr', 'word_only_dd_sample_rate')
flags.DEFINE_integer('word_only_dd_sample_seed', 42,
                     'Base seed for word_only_dd_sample_rate cycling permutation.')
flags.DEFINE_float('wod_scale', None, '')
flags.DEFINE_bool('use_word_only_ext', False,
                  'Include EXT samples that only have word labels (no IPA). '
                  '114K extra samples, but only word decoder benefits.')
flags.DEFINE_float('word_only_ext_sample_rate', 1.0,
                   'Fraction of word-only EXT samples to use per epoch (0~1). '
                   'Uses cycling index-based selection for uniform coverage across epochs.')
flags.DEFINE_alias('woe_sr', 'word_only_ext_sample_rate')
flags.DEFINE_integer('word_only_ext_sample_seed', 42,
                     'Base seed for word_only_ext_sample_rate cycling permutation.')
flags.DEFINE_float('woe_scale', None, '')
flags.DEFINE_float('wo_scale', None, '')
flags.DEFINE_bool('use_ipa_only_dd', False,
                  'Include DD samples on the word track that only have IPA labels '
                  '(no orthographic label_text). They are added to training only and '
                  'masked out of eval automatically because label_text is empty.')
flags.DEFINE_bool('use_ipa_only_ext', False,
                  'Include EXT samples on the word track that only have IPA labels '
                  '(no orthographic label_text). Disabled by default; kept for symmetry '
                  'with use_ipa_only_dd.')
flags.DEFINE_bool('include_overlap', False,
                  'Include overlap samples (that have both word and IPA labels) in '
                  'auto-generated pseudo-IPA data. When False (default), only word-only '
                  'samples are converted. When True, overlap samples also get pseudo-IPA '
                  'labels from eng_to_ipa (eval fold exclusion handled by dataset.py fold split).')
flags.DEFINE_bool('word_only_loss', False,
                  'Only compute word auxiliary loss for samples that have NO IPA label. '
                  'When True, word_mask is zeroed for samples where ipa_mask=1. '
                  'Motivation: eng_to_ipa conversion is noisy, so word loss on samples '
                  'with correct IPA labels introduces conflicting gradients.')
flags.DEFINE_bool('word_detach_encoder', False,
                  'Detach encoder output before word S2S forward pass. '
                  'Prevents word RNNT gradients from conflicting with CTC gradients on the encoder. '
                  'Word loss only trains the RNNT decoder/joint, not the shared encoder.')
flags.DEFINE_bool('word_loss_normalize', True,
                  'Normalize per-sample RNNT/TDT word loss by target length. '
                  'Makes RNNT loss magnitude comparable to CTC per-frame loss.')
flags.DEFINE_bool('legacy_loss', False, 'Use legacy loss: forward() computes combined scalar loss inline, bypassing get_loss_fn(). For A/B testing.')

# ---------- Metadata auxiliary losses (age/domain classification) ----------
flags.DEFINE_float('aux_age_weight', 0.0,
                   'Weight for age-group auxiliary loss. 0 = disabled (default). '
                   'Coarse 2-class: "3-4"(0) vs "5+"(1). '
                   'Encoder learns age-aware representations; predictions useful for tree reranker.')
flags.DEFINE_string('aux_age_mode', None,
                    'Age auxiliary loss mode: '
                    'classify (2-class softmax CE), '
                    'ordinal (scalar sigmoid, threshold=0.5 for binary, BCE loss), '
                    'regress (scalar, target=3.5/6.0 for 3-4/5+, MSE loss).')
flags.DEFINE_float('aux_domain_weight', 0.0,
                   'Weight for domain (DD vs EXT) auxiliary loss. '
                   '0 = disabled (default). Binary sigmoid: DD=1, EXT=0. '
                   'Helps encoder adapt to domain shift; predictions useful for tree reranker.')
flags.DEFINE_string('aux_pool', 'mean',
                    'Pooling method for auxiliary heads: '
                    'mean (simple mean pooling), '
                    'linear_att (linear attention pooling), '
                    'nonlinear_att (FFN attention pooling).')

# ---------- Length prediction auxiliary losses (nchars / nspaces) ----------
flags.DEFINE_float('aux_nchars_weight', 0.0,
                   'Weight for auxiliary log(1+n_ipa_chars) regression loss (MSE). '
                   '0 = disabled (default). Predicts log(1 + number_of_IPA_characters) '
                   'from encoder pooled representation. Typical range ~0-4.')
flags.DEFINE_float('aux_nspaces_weight', 0.0,
                   'Weight for auxiliary log(1+n_spaces) regression loss (MSE). '
                   '0 = disabled (default). Predicts log(1 + number_of_spaces) '
                   'i.e. log(word_count). Typical range ~0-2.5.')

# ---------- S2S decoder override (for non-native decoder experiments) ----------
flags.DEFINE_string('s2s_decoder', 'native',
                    'S2S decoder type: native (default, subclass decoder), '
                    'aed (custom Transformer AED decoder), '
                    'rnnt_reuse (reuse backbone RNNT decoder with vocab remapping, standard RNNT loss), '
                    'tdt_reuse (like rnnt_reuse but uses TDT loss with duration prediction — '
                    'exact replica of NeMo TDT architecture with IPA vocab), '
                    'tdt_scratch (scratch TDT decoder; uses tokenizer vocab when available, '
                    'otherwise falls back to IPA char vocab), '
                    'rnnt_custom (custom lightweight RNNT decoder from scratch).')
# -- AED decoder params (--s2s_decoder=aed) --
flags.DEFINE_integer('aed_layers', 2, 'AED: number of Transformer decoder layers')
flags.DEFINE_integer('aed_dim', 256, 'AED: decoder hidden dimension')
flags.DEFINE_integer('aed_heads', 4, 'AED: number of attention heads')
flags.DEFINE_float('aed_dropout', 0.1, 'AED: dropout rate')
flags.DEFINE_integer('aed_vocab_size', 0,
                     'AED: output vocab size. '
                     '0 = auto (IPA_CTC_VOCAB_SIZE if constrain_ipa else tokenizer.vocab_size)')
# -- General S2S training/decoding enhancements --
# (Applied to all applicable S2S decoders: native HF, AED, custom RNNT)
# Beam search: native HF models use --num_beams/--length_penalty (above);
#              AED and custom RNNT also respect --num_beams/--length_penalty.
# Label smoothing: uses the existing --label_smoothing flag (defined in melt).
#              Applied to: Whisper, AED, SqueezeFormer. Not applicable to RNNT loss.
flags.DEFINE_float('scheduled_sampling', 0.0,
                   'Scheduled sampling probability (0 = pure teacher forcing, '
                   '0.2~0.4 recommended). Replaces teacher tokens with model predictions '
                   'during training. Currently supported by: AED decoder.')
# -- Custom RNNT decoder params (--s2s_decoder=rnnt_custom) --
flags.DEFINE_integer('rnnt_pred_dim', 256, 'Custom RNNT: prediction network hidden dim')
flags.DEFINE_integer('rnnt_pred_layers', 1, 'Custom RNNT: prediction LSTM layers')
flags.DEFINE_integer('rnnt_joint_dim', 256, 'Custom RNNT: joint network hidden dim')
flags.DEFINE_integer('rnnt_vocab_size', 0,
                     'Custom RNNT: output vocab size. '
                     '0 = auto (IPA_CTC_VOCAB_SIZE if constrain_ipa else tokenizer.vocab_size)')

# -- TDT-specific params (--s2s_decoder=tdt_reuse) --
flags.DEFINE_list('tdt_durations', ['0', '1', '2', '3', '4'],
                  'TDT duration values for skip prediction. '
                  'Default [0,1,2,3,4] matches NeMo parakeet-tdt-0.6b-v2. '
                  'Only used when --s2s_decoder=tdt_reuse.')
flags.DEFINE_float('tdt_sigma', 0.02,
                   'TDT sigma param: Gaussian smoothing for duration targets. '
                   'Matches NeMo default. Only used when --s2s_decoder=tdt_reuse.')
flags.DEFINE_float('tdt_omega', 0.1,
                   'TDT omega param: weight balancing token vs duration loss. '
                   'Matches NeMo default. Only used when --s2s_decoder=tdt_reuse.')
flags.DEFINE_string('tdt_score_method', 'numba',
                    'TDT scoring method for score_targets(): '
                    '"numba" = NeMo TDTLossNumba via Numba CUDA JIT (fastest, default), '
                    '"exact" = pure PyTorch forward algorithm (no Numba, slower), '
                    '"forced_align" = forced greedy alignment (Viterbi approx, fast but approximate). '
                    'Use "exact" or "forced_align" when Numba is unavailable.')
flags.DEFINE_float('tdt_lr', None,
                   'Dedicated learning rate for CustomTDTDecoder. '
                   'When set, TDT decoder params get this LR instead of head_lr. '
                   'Randomly initialized parts (embedding, joint_out) need high LR '
                   'to catch up with copied parts (LSTM, proj). '
                   'Typical: 5e-4 ~ 1e-3 (5~10x head_lr). None = use head_lr.')
flags.DEFINE_string('nemo_trim_vocab_file', '',
                    'Optional JSON file describing a trimmed NeMo tokenizer vocab '
                    'experiment. Accepts either a plain JSON list of kept original '
                    'token ids, or an analyze_nemo_vocab_usage.py report containing '
                    'ranked_token_ids. Empty uses builtin ranked ids when the backbone '
                    'has an embedded trim profile.')
flags.DEFINE_integer('nemo_trim_vocab_topk', 0,
                     'Keep the first top-k original tokenizer ids from either '
                     'nemo_trim_vocab_file or a builtin ranked-id profile. '
                     '0 = use track/backbone auto default if defined, otherwise disabled. '
                     '-1 = explicitly disable auto default. '
                     'Positive values enable trimmed vocab with that top-k.')

flags.DEFINE_string('decode_method', None,
                    'Decoding method at inference time: '
                    'auto (default: CTC if ctc_only, else S2S), '
                    'ctc (always use CTC greedy/beam decode; S2S is auxiliary training loss only), '
                    's2s (always use S2S autoregressive decode), '
                    'tdt (TDT greedy decode with duration-aware frame skipping; requires '
                    '--s2s_decoder=tdt_reuse or --s2s_decoder=tdt_scratch), '
                    'native (NeMo native TDT/RNNT decode with original BPE vocab; for word track eval), '
                    'joint (CTC prefix score + S2S score beam fusion, a la ESPnet).')
flags.DEFINE_float('joint_ctc_decode_weight', 0.3,
                   'CTC weight in joint CTC+S2S beam decode. '
                   'score = (1-w)*log P_s2s + w*log P_ctc. '
                   'Only used when --decode_method=joint. Typical: 0.2~0.5.')

flags.DEFINE_bool('corpus_level_loss', False,
                  'Corpus-level loss reduction: weight each sample by its token count '
                  'so that longer samples contribute more to the loss, matching '
                  'corpus-level CER/WER evaluation metrics. Default False = per-sample mean.')
flags.DEFINE_bool('mean_volume_loss', False,
                  'Use mean_volume reduction for RNNT/TDT loss: losses.sum() / target_lengths.sum(). '
                  'Matches NeMo native training_step reduction. Each TARGET TOKEN contributes '
                  'equally to the gradient (longer utterances weigh more). '
                  'Unlike corpus_level_loss, this correctly handles RNNT per-sample totals '
                  'without double-counting. Recommended for NeMo adapter training.')

flags.DEFINE_float('loss_truncation_ratio', 0.0,
                   'Loss Truncation (Kang & Hashimoto, ACL 2020): zero out the top-K%% '
                   'per-sample losses in each batch before reduction. '
                   'E.g. 0.1 = drop 10%% highest-loss samples per batch. '
                   'Effective for noisy labels: prevents the model from memorizing '
                   'mislabeled examples. Applied BEFORE corpus_level_loss reduction. '
                   '0 = disabled. Typical: 0.05~0.2.')

flags.DEFINE_bool('save_logprobs', False,
                  'Save CTC log-probabilities per utterance during eval. '
                  'Produces ctc_logprobs.pt in model_dir for offline ensemble.')
flags.DEFINE_bool('save_dual_head_preds', False,
                  'When a model has both character CTC and TDT heads, export both '
                  'decoded texts during eval. Also saves dual_head_preds.pt and '
                  'CTC logprobs needed by downstream reranker fusion.')
flags.DEFINE_bool('save_word_head_preds', False,
                  'Export auxiliary word head predictions (word_head_preds.pt) and '
                  'word CTC logprobs (ctc_logprobs_word.pt) during eval. '
                  'Disabled by default: word logprobs are huge for BPE vocab and '
                  'unused by downstream ensemble. Enable only for pseudo_ipa/char word heads.')
flags.DEFINE_bool('save_pred_score', False,
                  'Export the primary decoder 1-best score during eval. '
                  'Default False preserves the current text-only decode behavior. '
                  'Currently supported by NeMo native TDT/RNNT decode and custom '
                  'tdt_reuse decode.')
flags.DEFINE_integer('save_pred_nbest', 0,
                     'Export top-N primary-decoder hypotheses and scores during eval. '
                     '0 = disabled. Native NeMo decode keeps the existing greedy '
                     'primary prediction and runs an extra beam decode only for export. '
                     'Custom tdt_reuse keeps the existing greedy prediction and uses '
                     'an internal beam search to generate N-best candidates.')
flags.DEFINE_bool('infer_extra_heads', True,
                  'During eval generate / fast_infer, whether to run extra '
                  'non-primary heads used only for export or auxiliary outputs. '
                  'Disable for faster online inference when dual-head exports and '
                  'aux metadata outputs are unused.')

# 为了快速实验 不修改增加flagfile 
flags.DEFINE_bool('cnoise', False, 'Add random Gaussian noise to the audio for data augmentation')
flags.DEFINE_bool('cnoise2', False, '')
flags.DEFINE_bool('dual_char', False, 'Dual character modeling: predict both IPA and orthographic chars with separate heads')
flags.DEFINE_bool('dual_bpe', False, 'Dual BPE modeling: predict both IPA and orthographic subword units with separate heads')
flags.DEFINE_bool('dual_ipa', False, 'Dual IPA modeling: predict both phoneme tokens and pseudo-IPA tokens with separate heads')
flags.DEFINE_alias('dual_ipc', 'dual_ipa')
flags.DEFINE_bool('dual_tdt', False, 'Dual TDT modeling: predict both phoneme tokens and duration tokens with separate heads')
flags.DEFINE_bool('dual_tdt_ipa', False,
                  'Convenience config for dual TDT with pseudo-IPA word targets. '
                  'Enables cross labels + word auxiliary loss, sets word_tdt_pseudo_ipa=True, '
                  'and defaults word_tdt_share_decoder to False unless explicitly set.')
flags.DEFINE_bool('dual_word_ipa', False,
                  'Word-track convenience config: main task predicts orthographic text, '
                  'auxiliary branch predicts 53-class IPA with CTC. Also enables '
                  'use_ipa_only_dd training rows by default.')
flags.DEFINE_bool('dual_word_tdt_ipa', False,
                  'Word-track convenience config: main task predicts orthographic text, '
                  'auxiliary branch predicts 53-class IPA with TDT reuse. Also enables '
                  'use_ipa_only_dd training rows by default.')
flags.DEFINE_bool('dual_tdt_mixed', False,
                  'Convenience config for mixed dual TDT. Main IPA branch keeps the '
                  'existing 53-class TDT decoder; word branch uses a half-shared BPE '
                  'TDT decoder with shared pred_rnn/enc_proj/pred_proj and separate '
                  'BPE embedding/joint output.')
flags.DEFINE_bool('mix2', False, 'aug mix max 2')
flags.DEFINE_bool('mix3', False, 'aug mix max 3')
flags.DEFINE_bool('mix4', False, 'aug mix max 4')
flags.DEFINE_bool('mix8', False, 'aug mix max 8')
flags.DEFINE_bool('mix_cross', False, '')
flags.DEFINE_bool('mix_csss', False, '')
flags.DEFINE_bool('mix_sc', False, '')
flags.DEFINE_bool('mix_ss', False, '')
flags.DEFINE_bool('mix_sa', False, '')
flags.DEFINE_bool('mix_dyn', False, '')
flags.DEFINE_bool('tdt', False, '')
flags.DEFINE_bool('tdt2', False, '')
flags.DEFINE_bool('tdt3', False, '')
flags.DEFINE_bool('tdt4', False, '')
flags.DEFINE_bool('tdt_only', False, '')
flags.DEFINE_bool('nemo_spec', False, '')
flags.DEFINE_bool('aux_loss', False, 'Use auxiliary loss head for multi-task learning (e.g. age/domain classification)')
flags.DEFINE_bool('len_loss', False, 'Use length prediction auxiliary loss (nchars/nspaces)')
flags.DEFINE_bool('s2s_only', False, '')
flags.DEFINE_bool('aug_short_speed', False, '')
flags.DEFINE_bool('aug_all_speed', False, '')

BACKBONES = {
  # ---- Whisper (mel spectrogram, seq2seq / CTC / hybrid) ----
  'whisper-tiny': 'openai/whisper-tiny',                      #  39M |  150 MB | 最小Whisper，快速调试
  'whisper-base': 'openai/whisper-base',                      #  74M |  290 MB | 轻量基线
  'whisper-small': 'openai/whisper-small',                    # 244M |  970 MB | 性价比较高
  'whisper-medium': 'openai/whisper-medium',                  # 769M |  3.0 GB | 中等精度
  'whisper-large-v3': 'openai/whisper-large-v3',              # 1.5B |  6.0 GB | 最高精度
  'whisper-large-v3-turbo': 'openai/whisper-large-v3-turbo',  # 809M |  3.1 GB | large精度+medium速度
  # ---- NeMo Parakeet (英文专用, .nemo格式, 需nemo_toolkit) ----
  'parakeet-ctc-0.6b': 'nvidia/parakeet-ctc-0.6b',            # 600M |  2.3 GB | CTC解码，适合音素赛道
  'parakeet-ctc-1.1b': 'nvidia/parakeet-ctc-1.1b',            # 1.1B |  4.3 GB | CTC大模型
  'parakeet-tdt-0.6b': 'nvidia/parakeet-tdt-0.6b-v2',         # 600M |  2.3 GB | TDT解码(CTC+Transducer)
  'parakeet-tdt-1.1b': 'nvidia/parakeet-tdt-1.1b',            # 1.1B |  4.3 GB | TDT大模型，英文SOTA
  'parakeet-tdt-0.6b-v3': 'nvidia/parakeet-tdt-0.6b-v3',      # 600M |  2.3 GB | TDT v3，25语言多语言
  'parakeet-tdt_ctc-1.1b': 'nvidia/parakeet-tdt_ctc-1.1b',    # 1.1B |  4.3 GB | TDT+CTC双目标，encoder更通用
  'parakeet-tdt_ctc-110m': 'nvidia/parakeet-tdt_ctc-110m',    # 110M |  440 MB | TDT+CTC超轻量，快速调试
  'parakeet-rnnt-0.6b': 'nvidia/parakeet-rnnt-0.6b',          # 600M |  2.3 GB | RNNT解码，encoder互补ensemble
  'parakeet-rnnt-1.1b': 'nvidia/parakeet-rnnt-1.1b',          # 1.1B |  4.3 GB | RNNT大模型，encoder多样性
  # ---- NeMo Conformer (原版Conformer, .nemo格式) ----
  'conformer-ctc-small': 'nvidia/stt_en_conformer_ctc_small',                  #  14M |   55 MB | 最小Conformer，快速调试
  'conformer-ctc-medium': 'nvidia/stt_en_conformer_ctc_medium',                #  30M |  120 MB | ≈whisper-tiny级别
  'conformer-ctc-large': 'nvidia/stt_en_conformer_ctc_large',                  # 120M |  480 MB | ≈whisper-small级别
  'conformer-transducer-small': 'nvidia/stt_en_conformer_transducer_small',    #  14M |   55 MB | RNNT版小模型
  'conformer-transducer-medium': 'nvidia/stt_en_conformer_transducer_medium',  #  30M |  120 MB | RNNT版中模型
  'conformer-transducer-large': 'nvidia/stt_en_conformer_transducer_large',    # 120M |  480 MB | RNNT版大模型
  # ---- NeMo FastConformer (Conformer改进版, .nemo格式) ----
  'fastconformer-ctc-small': 'nvidia/stt_en_fastconformer_ctc_small',          #  14M |   55 MB | 最小FastConformer
  'fastconformer-ctc-medium': 'nvidia/stt_en_fastconformer_ctc_medium',        #  30M |  120 MB | ≈whisper-tiny级别
  'fastconformer-ctc-large': 'nvidia/stt_en_fastconformer_ctc_large',          # 114M |  450 MB | FastConformer CTC
  'fastconformer-ctc-xlarge': 'nvidia/stt_en_fastconformer_ctc_xlarge',        # 600M |  2.3 GB | ≈parakeet-0.6b级别
  'fastconformer-transducer-large': 'nvidia/stt_en_fastconformer_transducer_large',  # 114M | 450 MB | FastConformer RNNT
  'fastconformer-hybrid-large': 'nvidia/stt_en_fastconformer_hybrid_large_streaming_multi',  # 114M | CTC+RNNT hybrid
  # ---- NeMo Canary (多语言, .nemo格式) ----
  'canary-1b': 'nvidia/canary-1b',                            # 1.0B |  3.9 GB | 多语言ASR(en/de/es/fr)，AED
  # ---- Wav2Vec2 (self-supervised, raw waveform, CTC-native) ----
  'wav2vec2-base': 'facebook/wav2vec2-base',                  #  95M |  360 MB | 自监督预训练
  'wav2vec2-base-100h': 'facebook/wav2vec2-base-100h',        #  95M |  360 MB | 100h LibriSpeech微调
  'wav2vec2-large': 'facebook/wav2vec2-large',                # 317M |  1.3 GB | 自监督预训练
  'wav2vec2-large-960h': 'facebook/wav2vec2-large-960h',      # 317M |  1.3 GB | 960h微调，推荐
  'wav2vec2-xls-r-300m': 'facebook/wav2vec2-xls-r-300m',      # 300M |  1.2 GB | 128语言多语言
  'wav2vec2-xls-r-1b': 'facebook/wav2vec2-xls-r-1b',          # 1.0B |  3.8 GB | 128语言多语言大模型
  # ---- Wav2Vec2 phoneme-pretrained (espeak IPA CTC, vocab mapping可用) ----
  'wav2vec2-espeak': 'facebook/wav2vec2-lv-60-espeak-cv-ft',  # 317M |  1.3 GB | espeak IPA 392类，wav2vec2-large基础
  'xlsr-espeak': 'facebook/wav2vec2-xlsr-53-espeak-cv-ft',    # 300M |  1.2 GB | espeak IPA 392类，XLS-R多语言基础
  'xls-r-300m-phoneme': 'vitouphy/wav2vec2-xls-r-300m-phoneme', # 300M | 1.2 GB | ARPABET 40类，XLS-R基础
  # ---- HuBERT / DistilHuBERT (self-supervised, raw waveform, CTC-native) ----
  'distilhubert': 'ntu-spml/distilhubert',                    #  24M |   90 MB | HuBERT蒸馏，最小waveform模型
  'hubert-base': 'facebook/hubert-base-ls960',                #  95M |  360 MB | 自监督预训练
  'hubert': 'facebook/hubert-large-ls960-ft',                 # 316M |  1.3 GB | 960h微调，推荐 (alias for hubert-large)
  'hubert-large': 'facebook/hubert-large-ls960-ft',           # 316M |  1.3 GB | 960h微调，推荐
  'hubert-xlarge': 'facebook/hubert-xlarge-ll60k',            # 964M |  3.7 GB | 60k小时预训练
  # ---- WavLM (self-supervised, raw waveform, CTC-native) ----
  'wavlm-base': 'microsoft/wavlm-base',                      #  95M |  360 MB | 自监督预训练
  'wavlm-base-plus': 'microsoft/wavlm-base-plus',            #  95M |  360 MB | 更多数据预训练，推荐
  'wavlm-large': 'microsoft/wavlm-large',                    # 317M |  1.3 GB | 大模型，语音表征SOTA
}

# backbone短名/全名 → model类型的映射规则
_BACKBONE_MODEL_RULES = [
  # (backbone名称中包含的关键词, 对应的 --model 值)
  ('whisper',        'whisper'),       # openai/whisper-*, distil-whisper/*
  ('moonshine',      'moonshine'),     # usefulsensors/moonshine-*
  ('parakeet',       'nemo'),          # nvidia/parakeet-*
  ('fastconformer',  'nemo'),          # nvidia/stt_en_fastconformer_*
  ('canary',         'nemo'),          # nvidia/canary-*
  ('wav2vec2',       'wav2vec2'),      # facebook/wav2vec2-*
  ('hubert',         'wav2vec2'),      # facebook/hubert-*, ntu-spml/distilhubert
  ('wavlm',         'wav2vec2'),      # microsoft/wavlm-*
  ('data2vec',       'wav2vec2'),      # facebook/data2vec-audio-*
  ('squeezeformer',  'squeezeformer'),
]

def infer_model_from_backbone(backbone: str) -> str:
  """根据 --backbone 自动推断 --model 类型。
  
  先查 BACKBONES 短名映射得到全名，再按关键词匹配。
  """
  # 如果用户传了短名，先解析为全名
  full_name = BACKBONES.get(backbone, backbone).lower()
  # 短名本身也参与匹配（如 'parakeet-tdt-0.6b'）
  key = f'{backbone.lower()} {full_name}'
  for keyword, model_type in _BACKBONE_MODEL_RULES:
    if keyword in key:
      return model_type
  return 'whisper'  # 默认回退


def init(model_name, run_version=None):
  """Shared init logic. Called by track config.init()."""
  run_version = run_version or RUN_VERSION
  
  # 自动根据 backbone 推断 model 类型，始终以 backbone 推断为准
  inferred_model = infer_model_from_backbone(FLAGS.backbone)
  allowed_model_overrides = {
      ('nemo', 'nemo_official'),
  }
  keep_explicit_model = bool(FLAGS.model) and (
      FLAGS.model == inferred_model
      or (inferred_model, FLAGS.model) in allowed_model_overrides
  )
  if FLAGS.model and not keep_explicit_model:
    logger.warning(
      f'--model={FLAGS.model} does not match backbone={FLAGS.backbone} '
      f'(inferred: {inferred_model}). Overriding --model to {inferred_model}.')
  FLAGS.model = FLAGS.model if keep_explicit_model else inferred_model
  
  # 短名解析为完整 HF/NeMo 路径
  if FLAGS.backbone in BACKBONES:
    FLAGS.backbone = BACKBONES[FLAGS.backbone]
  wandb = False
  FLAGS.wandb = wandb if FLAGS.wandb is None else FLAGS.wandb
  wandb_mode = 'online'
  FLAGS.wandb_mode = wandb_mode if FLAGS.wandb_mode is None else FLAGS.wandb_mode
  
  write_summary = False
  FLAGS.write_summary = write_summary if FLAGS.write_summary is None else FLAGS.write_summary
  
  ignores = []
  suffix = ''
  gz.init_modeldir(ignores=ignores, 
                   run_version=FLAGS.run_version or run_version,
                   suffix=suffix)
  gz.init_wandb(model_name)
  
  fp16 = True
  FLAGS.fp16 = fp16 if FLAGS.fp16 is None else FLAGS.fp16

  folds = 5
  FLAGS.folds = FLAGS.folds or folds
  fold_seed = 42
  FLAGS.fold_seed = FLAGS.fold_seed if FLAGS.fold_seed is not None else fold_seed
  
  bs = 4
  FLAGS.batch_size = FLAGS.batch_size or bs
  eval_bs = FLAGS.batch_size * 2
  FLAGS.eval_batch_size = FLAGS.eval_batch_size or eval_bs
  
  if inferred_model in ('wav2vec2'):
    FLAGS.eval_batch_size = min(FLAGS.eval_batch_size, 16)  # espeak模型内存较大，eval时减半batch size
    # SSL models (wav2vec2/wavlm/hubert) have fp16 overflow issues in attention layers,
    # use bf16 (same exponent range as fp32, no overflow) by default
    if FLAGS.fp16 and not FLAGS.bfloat16:
      import torch
      if torch.cuda.is_bf16_supported():
        FLAGS.bfloat16 = True
        logger.info(f'wav2vec2-family model detected: auto-enabling bfloat16 for numerical stability')
  
  lr = 1e-5
  FLAGS.learning_rate = FLAGS.learning_rate or lr
  
  optimizer = 'adamw'
  FLAGS.optimizer = FLAGS.optimizer or optimizer
  
  ep = 5
  FLAGS.ep = FLAGS.ep or ep
  
  vie = 1 if not FLAGS.online else FLAGS.ep
  FLAGS.vie = FLAGS.vie or vie
  
  sie = 1
  FLAGS.sie = FLAGS.sie or sie  
  
  acc_steps = 4
  FLAGS.acc_steps = FLAGS.acc_steps or acc_steps
  
  scheduler = 'cosine'
  FLAGS.scheduler = FLAGS.scheduler or scheduler
  
  FLAGS.torch = True
  FLAGS.torch_only = True
  
  device = 'gpu'
  FLAGS.device = FLAGS.device or device
  
  if FLAGS.ema_train:
    if FLAGS.ema_start_epoch is None:
      if not FLAGS.unfreeze_epoch:
        FLAGS.ema_start_epoch = 1  
      else:
        FLAGS.ema_start_epoch = FLAGS.unfreeze_epoch + 1
    if FLAGS.ema_decay is None:
      FLAGS.ema_decay = 0.999
    FLAGS.scheduler = 'constant'  
    # FLAGS.ep = max(FLAGS.ep, 10)
    FLAGS.vie = min(FLAGS.vie, 1)
    # FLAGS.save_ema_init = True
  
  save_best = True
  if FLAGS.save_best is None:
    FLAGS.save_best = save_best
    
  if FLAGS.online:
    FLAGS.save_best = False
    
  if FLAGS.save_best:
    FLAGS.vie = min(FLAGS.vie, 1)
    FLAGS.sie = min(FLAGS.sie, FLAGS.vie)
  
  # lower is better for both CER and WER
  metric_key = 'score'
  FLAGS.metric_key = FLAGS.metric_key or metric_key
  FLAGS.greater_is_better = False
  FLAGS.save_best_only = False
  
  if FLAGS.smoke:
    FLAGS.ep = 1
    FLAGS.vie = 1
    FLAGS.sie = 1
    FLAGS.samples = 1000
    # FLAGS.eval_samples = 100

  early_stop = False
  if FLAGS.early_stop is None:
    FLAGS.early_stop = early_stop
    
  if FLAGS.early_stop:
    FLAGS.patience = int(FLAGS.patience / FLAGS.vie)
    
  # 全量训练 还是按vie固定1 sie保持不变
  if FLAGS.online:
    FLAGS.vie = 1
    FLAGS.early_stop = False
    # FLAGS.eval_ext_full = False
  elif FLAGS.save_logprobs:
    FLAGS.infer_extra_heads = True
    
  if FLAGS.cnoise:
    FLAGS.aug_classroom_noise = True
    # 高snr 对应更轻的噪音 
    FLAGS.aug_classroom_snr_min = 20
    FLAGS.aug_classroom_snr_max = 40
  
  if FLAGS.cnoise2:
    FLAGS.aug_classroom_noise = True
    # 高snr 对应更轻的噪音 相比cnosie稍强
    FLAGS.aug_classroom_snr_min = 15
    FLAGS.aug_classroom_snr_max = 30
    
  if FLAGS.nemo_spec:
    FLAGS.aug_nemo_spec = True
    FLAGS.aug_nemo_spec_time_ratio = 0.05
    FLAGS.aug_time_num = 2
    FLAGS.aug_freq_mask = 10
    FLAGS.aug_freq_num = 2
    
  if FLAGS.mix2:
    FLAGS.aug_mix = True
    FLAGS.aug_mix_num = 2
    FLAGS.aug_mix_random_num = True
    
  if FLAGS.mix3:
    FLAGS.aug_mix = True
    FLAGS.aug_mix_num = 3
    FLAGS.aug_mix_random_num = True
    
  if FLAGS.mix4:
    FLAGS.aug_mix = True
    FLAGS.aug_mix_num = 4
    FLAGS.aug_mix_random_num = True
    
  if FLAGS.mix8:
    FLAGS.aug_mix = True
    FLAGS.aug_mix_num = 8
    FLAGS.aug_mix_random_num = True
    FLAGS.aug_mix_fit_len = True
    
  if FLAGS.mix_cross:
    FLAGS.aug_mix_strategy = 'cross_source'
    
  if FLAGS.mix_csss:
    FLAGS.aug_mix_strategy = 'cross_source,same_source'
  
  if FLAGS.mix_sc:
    FLAGS.aug_mix_strategy = 'same_child'
    
  if FLAGS.mix_ss:
    FLAGS.aug_mix_strategy = 'same_source'

  if FLAGS.mix_sa:
    FLAGS.aug_mix_strategy = 'same_age'
    
  if FLAGS.mix_dyn:
    FLAGS.aug_mix_dur_aware = True
    FLAGS.aug_mix_num = 5
    # FLAGS.aug_mix_fit_len = True

  dual_word_only_dd_sample_rate = 1.0 if getattr(FLAGS, 'temperature_sampler', False) else 0.2
  dual_word_only_ext_sample_rate = 1.0 if getattr(FLAGS, 'temperature_sampler', False) else 0.1
  
  if FLAGS.aug_short_speed:
    FLAGS.aug_speed = True
    FLAGS.aug_speed_min = 0.85
    FLAGS.aug_speed_max = 1.15
    FLAGS.aug_speed_short_only = True
    FLAGS.aug_speed_short_dur = 2.0 
    
  if FLAGS.aug_all_speed:
    FLAGS.aug_speed = True
    FLAGS.aug_speed_min = 0.85
    FLAGS.aug_speed_max = 1.15
    
  if FLAGS.dual_char:
    FLAGS.use_cross_labels = True
    FLAGS.word_ctc = True
    FLAGS.word_weight = 0.3
    FLAGS.use_word_only_dd = True
    FLAGS.use_word_only_ext = True
    FLAGS.word_only_dd_sample_rate = dual_word_only_dd_sample_rate
    FLAGS.word_only_ext_sample_rate = dual_word_only_ext_sample_rate
    
  if FLAGS.dual_bpe:
    FLAGS.use_cross_labels = True
    FLAGS.word_ctc = True
    FLAGS.word_weight = 0.3
    FLAGS.use_word_only_dd = True
    FLAGS.use_word_only_ext = True
    FLAGS.word_only_dd_sample_rate = dual_word_only_dd_sample_rate
    FLAGS.word_only_ext_sample_rate = dual_word_only_ext_sample_rate
    FLAGS.word_ctc_bpe = True

  if FLAGS.word_ctc_bpe_add_blank is None:
    auto_word_ctc_bpe_add_blank = None
    auto_reason = None
    word_tok = getattr(FLAGS, 'word_tokenizer', None)
    resolved_word_tok = BACKBONES.get(word_tok, word_tok) if word_tok else word_tok
    word_tok_name = str(word_tok).lower() if word_tok else ''
    resolved_word_tok_str = str(resolved_word_tok).lower() if resolved_word_tok else ''
    word_tok_is_nemo = word_tok_name in {'nemo', 'parakeet'} or 'nvidia/' in resolved_word_tok_str
    if getattr(FLAGS, 'track', None) == 'word':
      auto_word_ctc_bpe_add_blank = True
      auto_reason = 'word track standardizes auxiliary BPE CTC to vocab+1 blank protocol'
    elif word_tok_is_nemo:
      auto_word_ctc_bpe_add_blank = True
      auto_reason = 'explicit word_tokenizer=nemo/parakeet uses vocab+1 blank protocol even on waveform encoders'
    elif inferred_model == 'nemo':
      auto_word_ctc_bpe_add_blank = True
      auto_reason = 'phonetic NeMo legacy auxiliary BPE checkpoints used vocab+1 blank protocol'
    elif inferred_model == 'wav2vec2':
      auto_word_ctc_bpe_add_blank = False
      auto_reason = 'phonetic wav2vec2/hubert/wavlm legacy auxiliary BPE checkpoints used no-extra-blank protocol'
    else:
      auto_word_ctc_bpe_add_blank = False
      auto_reason = 'phonetic non-NeMo/non-wav2vec2 fallback defaults to no-extra-blank protocol'
    FLAGS.word_ctc_bpe_add_blank = auto_word_ctc_bpe_add_blank
    logger.info(
      'word_ctc_bpe_add_blank auto-set to %s (%s)',
      FLAGS.word_ctc_bpe_add_blank,
      auto_reason,
    )
    
  if FLAGS.dual_ipc:
    FLAGS.use_cross_labels = True
    FLAGS.word_ctc = True
    FLAGS.word_weight = 0.3
    FLAGS.use_word_only_dd = True
    FLAGS.use_word_only_ext = True
    FLAGS.word_only_dd_sample_rate = dual_word_only_dd_sample_rate
    FLAGS.word_only_ext_sample_rate = dual_word_only_ext_sample_rate
    FLAGS.pseudo_ipa_ctc = True
    
  if FLAGS.dual_tdt:
    FLAGS.use_cross_labels = True
    FLAGS.word_ctc = False
    FLAGS.word_weight = 0.3
    FLAGS.use_word_only_dd = True
    FLAGS.use_word_only_ext = True
    FLAGS.word_only_dd_sample_rate = dual_word_only_dd_sample_rate
    FLAGS.word_only_ext_sample_rate = dual_word_only_ext_sample_rate
    # # 4090 oom 5090 need gradc
    # if torch.cuda.is_available():
    #   total_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
    #   if total_mem < 48:
    #     FLAGS.gradc = True
    # FLAGS.acc_steps *= 2

  if FLAGS.dual_tdt_ipa:
    FLAGS.use_cross_labels = True
    FLAGS.word_ctc = False
    FLAGS.word_weight = 0.3
    FLAGS.word_tdt_pseudo_ipa = True
    FLAGS.use_word_only_dd = True
    FLAGS.use_word_only_ext = True
    FLAGS.word_only_dd_sample_rate = dual_word_only_dd_sample_rate
    FLAGS.word_only_ext_sample_rate = dual_word_only_ext_sample_rate
    if FLAGS.word_tdt_share_decoder is None:
      FLAGS.word_tdt_share_decoder = False

  if FLAGS.dual_word_ipa:
    FLAGS.use_cross_labels = True
    FLAGS.word_aux_ipa = True
    FLAGS.ipa_weight = 0.0
    FLAGS.word_ctc = True
    FLAGS.word_weight = 0.3
    FLAGS.pseudo_ipa_ctc = True
    FLAGS.use_ipa_only_dd = True
    if FLAGS.head_lr is None:
      FLAGS.head_lr = 1e-4

  if FLAGS.dual_word_tdt_ipa:
    FLAGS.use_cross_labels = True
    FLAGS.word_aux_ipa = True
    FLAGS.ipa_weight = 0.0
    FLAGS.word_ctc = False
    FLAGS.word_weight = 0.3
    FLAGS.word_tdt_pseudo_ipa = True
    FLAGS.use_ipa_only_dd = True
    if FLAGS.head_lr is None:
      FLAGS.head_lr = 1e-4
    if FLAGS.word_tdt_share_decoder is None:
      FLAGS.word_tdt_share_decoder = False

  if FLAGS.dual_tdt_mixed:
    FLAGS.use_cross_labels = True
    FLAGS.word_ctc = False
    FLAGS.word_weight = 0.3
    FLAGS.word_tdt_mixed = True
    FLAGS.use_word_only_dd = True
    FLAGS.use_word_only_ext = True
    FLAGS.word_only_dd_sample_rate = dual_word_only_dd_sample_rate
    FLAGS.word_only_ext_sample_rate = dual_word_only_ext_sample_rate
    
  if FLAGS.use_cross_labels:
    if FLAGS.wo_scale is not None:
      FLAGS.word_only_dd_sample_rate *= FLAGS.wo_scale
      FLAGS.word_only_ext_sample_rate *= FLAGS.wo_scale
    else:
      if FLAGS.wod_scale is not None:
        FLAGS.word_only_dd_sample_rate *= FLAGS.wod_scale
      if FLAGS.woe_scale is not None:
        FLAGS.word_only_ext_sample_rate *= FLAGS.woe_scale
    
  if FLAGS.aux_loss:
    # 对，归一化后 aux_age=0.1571 已经很小了，乘以 weight=0.1 只贡献 ~0.016，相对 ctc=2.1 完全可以忽略。
    # weight	贡献	占 CTC 比例
    # 0.1	0.016	0.7% ← 几乎无效
    # 1.0	0.157	7.5% ← 合理
    # 2.0	0.314	15%
    # --aux_age_weight=1 比较合适，辅助 loss 占主 loss 的 5-10% 是典型的设置，0.1 太小了几乎没有效果，2.0 可能有点大了（尤其考虑到 age 分类可能比较简单，损失较小）。所以默认 1.0 是个合理的起点。
    if FLAGS.aux_age_mode is None:
      FLAGS.aux_age_mode = 'classify'  
    FLAGS.aux_age_weight = 0.1 if FLAGS.aux_age_mode != 'regress' else 1.0
    FLAGS.aux_domain_weight = 0.1
    FLAGS.aux_pool = 'linear_att'
  
  if FLAGS.len_loss:
    FLAGS.aux_nchars_weight = 0.1
    FLAGS.aux_nspaces_weight = 0.1
    FLAGS.aux_pool = 'linear_att'
  
  # -- Two-stage training: unfreeze_epoch implies freeze_encoder --
  if FLAGS.unfreeze_epoch is not None:
    assert FLAGS.unfreeze_epoch > 0, f'unfreeze_epoch must be > 0, got {FLAGS.unfreeze_epoch}'
    FLAGS.freeze_encoder = True
    
  if FLAGS.tdt:
    FLAGS.s2s_decoder = 'tdt_reuse'
    # for I see tdt much better then ctc decode
    if FLAGS.decode_method is None:
      FLAGS.decode_method = 'tdt'
    FLAGS.ctc_weight = 0.7
    if FLAGS.tdt_lr is None:
      FLAGS.tdt_lr = 5e-4
      
  if FLAGS.tdt2:
    FLAGS.s2s_decoder = 'tdt_reuse'
    if FLAGS.decode_method is None:
      FLAGS.decode_method = 'tdt'
    FLAGS.ctc_weight = 0.5
    if FLAGS.tdt_lr is None:
      FLAGS.tdt_lr = 5e-4
    # FLAGS.gradc = True

  if FLAGS.tdt3:
    FLAGS.s2s_decoder = 'tdt_reuse'
    if FLAGS.decode_method is None:
      FLAGS.decode_method = 'tdt'
    FLAGS.ctc_weight = 0.3
    if FLAGS.tdt_lr is None:
      FLAGS.tdt_lr = 5e-4
    # FLAGS.gradc = True
    
  if FLAGS.tdt4:
    FLAGS.s2s_decoder = 'tdt_reuse'
    if FLAGS.decode_method is None:
      FLAGS.decode_method = 'tdt'
    FLAGS.ctc_weight = 0.1
    if FLAGS.tdt_lr is None:
      FLAGS.tdt_lr = 5e-4
    # FLAGS.gradc = True
    
  if FLAGS.tdt_only:
    FLAGS.s2s_decoder = 'tdt_reuse'
    FLAGS.decode_method = 'tdt'
    FLAGS.ctc_weight = 0.0
    if FLAGS.tdt_lr is None:
      FLAGS.tdt_lr = 5e-4
    # FLAGS.gradc = True
    
  if FLAGS.tdt_lr is None:
    FLAGS.tdt_lr = FLAGS.head_lr
    
  if FLAGS.s2s_only:
    FLAGS.ctc_weight = 0.0
    FLAGS.decode_method = 's2s'

  if FLAGS.word_tdt_pseudo_ipa:
    assert FLAGS.s2s_decoder == 'tdt_reuse', \
      '--word_tdt_pseudo_ipa requires --s2s_decoder=tdt_reuse (or one of --tdt/--tdt2/--tdt3/--tdt4/--tdt_only)'
    assert not FLAGS.word_ctc, \
      '--word_tdt_pseudo_ipa is incompatible with --word_ctc; choose one auxiliary branch'
    assert not (getattr(FLAGS, 'word_tdt_share_decoder', False) and getattr(FLAGS, 'word_tdt_half_share_decoder', False)), \
      '--word_tdt_share_decoder and --word_tdt_half_share_decoder are mutually exclusive'

  if FLAGS.word_tdt_half_share_decoder:
    assert FLAGS.word_tdt_pseudo_ipa, \
      '--word_tdt_half_share_decoder requires --word_tdt_pseudo_ipa'

  if FLAGS.word_tdt_mixed:
    assert FLAGS.s2s_decoder == 'tdt_reuse', \
      '--word_tdt_mixed requires --s2s_decoder=tdt_reuse (or one of --tdt/--tdt2/--tdt3/--tdt4/--tdt_only)'
    assert not FLAGS.word_ctc, \
      '--word_tdt_mixed is incompatible with --word_ctc; choose one auxiliary branch'
    assert not FLAGS.word_tdt_pseudo_ipa, \
      '--word_tdt_mixed is mutually exclusive with --word_tdt_pseudo_ipa'

  if FLAGS.word_aux_ipa:
    assert getattr(FLAGS, 'track', None) == 'word', \
      '--word_aux_ipa is only valid on the word track'
    assert FLAGS.use_cross_labels, \
      '--word_aux_ipa requires --use_cross_labels'
    assert FLAGS.word_weight > 0, \
      '--word_aux_ipa requires --word_weight>0'
    assert FLAGS.pseudo_ipa_ctc or FLAGS.word_tdt_pseudo_ipa, \
      '--word_aux_ipa requires either --pseudo_ipa_ctc (CTC aux) or --word_tdt_pseudo_ipa (TDT aux)'
    
  if FLAGS.decode_method is None:
    FLAGS.decode_method = 'auto' 
    
  if FLAGS.online:
    FLAGS.ep = 20
    FLAGS.sie = 0.1
    FLAGS.vie = 1
    # if FLAGS.mode in (None, 'train'):
    #   FLAGS.force_save = True

def post_restore():
  """Re-apply derived flags that may have been overwritten by restore_configs().
  
  Called from train.py after mt.init() (which calls restore_configs()).
  Command-line flags like save_logprobs survive restore, but programmatic
  derivatives (save_dual_head_preds, infer_extra_heads) get overwritten.
  """
  if FLAGS.save_logprobs:
    FLAGS.infer_extra_heads = True

def show():
  """Shared show logic. Track config can override to add extra fields."""
  ic(FLAGS.backbone, FLAGS.model, FLAGS.sample_rate, FLAGS.max_audio_sec,
     FLAGS.random_crop, FLAGS.crop_prob, FLAGS.crop_label,
     FLAGS.num_beams, FLAGS.max_new_tokens, FLAGS.language,
     FLAGS.score_metric, FLAGS.train_file, FLAGS.label_column,
     FLAGS.batch_size, FLAGS.eval_batch_size, FLAGS.learning_rate,
     FLAGS.ep, FLAGS.folds, FLAGS.fold, FLAGS.metric_key,FLAGS.gradc,
     FLAGS.early_stop, FLAGS.patience,
     FLAGS.vie, FLAGS.sie,
     FLAGS.freeze_encoder, FLAGS.llrd, FLAGS.llrd_decay,
     FLAGS.use_ext, FLAGS.eval_ext, FLAGS.ext_root,
     FLAGS.ext_weight, FLAGS.ext_sample_rate, FLAGS.eval_ext_weight,
     FLAGS.ipa_weight, FLAGS.word_weight,
     FLAGS.use_cross_labels, FLAGS.use_word_only_dd, FLAGS.word_only_dd_sample_rate,
     FLAGS.use_word_only_ext, FLAGS.word_only_ext_sample_rate,
     FLAGS.word_detach_encoder, FLAGS.word_loss_normalize,
     FLAGS.decode_method, FLAGS.joint_ctc_decode_weight,
     FLAGS.word_tokenizer, FLAGS.default_word_tokenizer,
     FLAGS.native_ctc, FLAGS.raw_ctc_eval)
