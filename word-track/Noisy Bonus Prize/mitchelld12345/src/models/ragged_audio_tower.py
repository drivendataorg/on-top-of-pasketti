"""Batched audio tower for Qwen3-ASR: batched conv + batched per-sample SDPA transformer.

Produces identical attention pattern to the default per-sample audio tower
(global attention within each sample), but processes all samples in parallel.
Use this in both training and inference to eliminate drift.
"""
import types

import torch
import torch.nn.functional as F
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    _get_feat_extract_output_lengths,
)


def ragged_audio_tower_forward(tower, input_features, feat_lens):
    n_window = tower.n_window
    chunk_size = n_window * 2
    conv_chunksize = tower.conv_chunksize
    aftercnn_lens = _get_feat_extract_output_lengths(feat_lens)

    # --- Chunked conv (batched across all samples) ---
    chunk_num = torch.ceil(feat_lens / chunk_size).long()
    chunk_lengths = torch.tensor(
        [chunk_size] * chunk_num.sum().item(),
        dtype=torch.long, device=feat_lens.device,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feat_lens % chunk_size
    chunk_lengths[chunk_lengths == 0] = chunk_size

    mel_cat = torch.cat([input_features[i, :, :feat_lens[i]] for i in range(len(feat_lens))], dim=1)
    chunk_list = mel_cat.split(chunk_lengths.tolist(), dim=1)
    padded_feature = torch.nn.utils.rnn.pad_sequence(
        [c.T for c in chunk_list], batch_first=True,
    ).transpose(1, 2)

    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    padded_mask_after_cnn = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(l, dtype=torch.bool, device=feat_lens.device) for l in feature_lens_after_cnn],
        batch_first=True,
    )

    padded_feature = padded_feature.unsqueeze(1)
    padded_embeds = []
    for chunk in padded_feature.split(conv_chunksize, dim=0):
        embed = F.gelu(tower.conv2d1(chunk))
        embed = F.gelu(tower.conv2d2(embed))
        embed = F.gelu(tower.conv2d3(embed))
        padded_embeds.append(embed)
    padded_embed = torch.cat(padded_embeds, dim=0)

    b, c, f, t = padded_embed.size()
    padded_embed = tower.conv_out(
        padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
    )

    pos_emb = tower.positional_embedding(padded_embed.shape[1]).unsqueeze(0).to(padded_embed.dtype)
    padded_embed = padded_embed + pos_emb

    # --- Gather valid tokens per sample ---
    chunk_offset = 0
    sample_tokens = []
    for i in range(len(feat_lens)):
        n_chunks = chunk_num[i].item()
        sample_mask = padded_mask_after_cnn[chunk_offset:chunk_offset + n_chunks]
        sample_embed = padded_embed[chunk_offset:chunk_offset + n_chunks]
        chunk_offset += n_chunks
        sample_tokens.append(sample_embed[sample_mask])

    # --- Batched per-sample global attention (matches default SDPA behavior) ---
    max_len = aftercnn_lens.max().item()
    B = len(feat_lens)
    D = sample_tokens[0].shape[-1]

    batched = torch.zeros(B, max_len, D, dtype=sample_tokens[0].dtype, device=sample_tokens[0].device)
    for i, st in enumerate(sample_tokens):
        batched[i, :st.shape[0]] = st

    padding_mask = None
    if (aftercnn_lens != max_len).any():
        padding_mask = torch.zeros(B, 1, 1, max_len, dtype=batched.dtype, device=batched.device)
        for i, cnn_len in enumerate(aftercnn_lens):
            if cnn_len < max_len:
                padding_mask[i, :, :, cnn_len:] = torch.finfo(batched.dtype).min

    hs = batched
    for encoder_layer in tower.layers:
        hs = _encoder_layer_forward(encoder_layer, hs, padding_mask)

    hs = tower.ln_post(hs)
    hs = tower.proj1(hs)
    hs = tower.act(hs)
    hs = tower.proj2(hs)

    parts = [hs[i, :aftercnn_lens[i]] for i in range(B)]
    return torch.cat(parts, dim=0)


def _encoder_layer_forward(layer, hidden_states, attention_mask):
    attn = layer.self_attn

    residual = hidden_states
    hidden_states = layer.self_attn_layer_norm(hidden_states)

    B, T, D = hidden_states.shape
    q = attn.q_proj(hidden_states).reshape(B, T, attn.num_heads, -1).transpose(1, 2)
    k = attn.k_proj(hidden_states).reshape(B, T, attn.num_heads, -1).transpose(1, 2)
    v = attn.v_proj(hidden_states).reshape(B, T, attn.num_heads, -1).transpose(1, 2)

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attention_mask, scale=attn.scaling, is_causal=False,
    )
    hidden_states = out.transpose(1, 2).reshape(B, T, D)
    hidden_states = attn.out_proj(hidden_states)

    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.final_layer_norm(hidden_states)
    hidden_states = layer.fc1(hidden_states)
    hidden_states = layer.activation_fn(hidden_states)
    hidden_states = layer.fc2(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


def _ragged_get_audio_features(thinker, input_features, feature_attention_mask=None,
                               audio_feature_lengths=None):
    if feature_attention_mask is not None:
        feat_lens = feature_attention_mask.sum(-1)
    else:
        feat_lens = audio_feature_lengths
    return ragged_audio_tower_forward(thinker.audio_tower, input_features, feat_lens)


def patch_audio_tower(thinker):
    thinker.get_audio_features = types.MethodType(_ragged_get_audio_features, thinker)
