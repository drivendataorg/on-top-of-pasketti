"""Inference entrypoint for Qwen3-ASR word track competition submission."""
import json
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn.functional as F

from silero_vad import load_silero_vad, get_speech_timestamps, collect_chunks

DATA_DIR = Path("/code_execution/data")
AUDIO_DIR = DATA_DIR / "audio"
SUBMISSION_FORMAT = DATA_DIR / "submission_format.jsonl"
METADATA = DATA_DIR / "utterance_metadata.jsonl"
OUTPUT_DIR = Path("/code_execution/submission")
OUTPUT_FILE = OUTPUT_DIR / "submission.jsonl"

MODEL_DIR = SRC_DIR / "qwen_model"
CONFIG_FILE = SRC_DIR / "config.yaml"


def load_config():
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


SHORT_PROMPT_CONTEXT = (
    "A child's speech assessment. The child is saying one or two words. "
    "Common words: chair, beard, nurse, ladder, fork, hammer, scarf, sword, dark, weird, clear, "
    "fish, duck, tiger, rabbit, frog, monkey, elephant, butterfly, giraffe, "
    "red, blue, green, yellow, cup, ring, star, door, flower, slide, truck, guitar, drum, "
    "pajamas, umbrella, vegetable, telephone, helicopter, abracadabra, "
    "finger, ear, teeth, five, seven, ten, three, one, two, four."
)
LONG_PROMPT_CONTEXT = "A child speaking in English."
SHORT_DURATION_THRESHOLD = 2.0


def make_text_prompt(context=""):
    if context:
        return (
            f"<|im_start|>system\n{context}<|im_end|>\n"
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\nlanguage English<asr_text>"
        )
    return (
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\nlanguage English<asr_text>"
    )


TEXT_PROMPT = make_text_prompt()
TEXT_PROMPT_SHORT = make_text_prompt(SHORT_PROMPT_CONTEXT)
TEXT_PROMPT_LONG = make_text_prompt(LONG_PROMPT_CONTEXT)


def _ragged_audio_tower_forward(tower, input_features, feat_lens):
    """Batched audio tower: batched conv + batched per-sample SDPA transformer."""
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        _get_feat_extract_output_lengths,
    )

    n_window = tower.n_window
    chunk_size = n_window * 2
    conv_chunksize = tower.conv_chunksize
    aftercnn_lens = _get_feat_extract_output_lengths(feat_lens)

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

    chunk_offset = 0
    sample_tokens = []
    for i in range(len(feat_lens)):
        n_chunks = chunk_num[i].item()
        sample_mask = padded_mask_after_cnn[chunk_offset:chunk_offset + n_chunks]
        sample_embed = padded_embed[chunk_offset:chunk_offset + n_chunks]
        chunk_offset += n_chunks
        sample_tokens.append(sample_embed[sample_mask])

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
        attn = encoder_layer.self_attn
        residual = hs
        hs = encoder_layer.self_attn_layer_norm(hs)
        Bt, T, Dt = hs.shape
        q = attn.q_proj(hs).reshape(Bt, T, attn.num_heads, -1).transpose(1, 2)
        k = attn.k_proj(hs).reshape(Bt, T, attn.num_heads, -1).transpose(1, 2)
        v = attn.v_proj(hs).reshape(Bt, T, attn.num_heads, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=padding_mask, scale=attn.scaling, is_causal=False)
        hs = attn.out_proj(out.transpose(1, 2).reshape(Bt, T, Dt))
        hs = residual + hs
        residual = hs
        hs = encoder_layer.final_layer_norm(hs)
        hs = encoder_layer.fc1(hs)
        hs = encoder_layer.activation_fn(hs)
        hs = encoder_layer.fc2(hs)
        hs = residual + hs

    hs = tower.ln_post(hs)
    hs = tower.proj1(hs)
    hs = tower.act(hs)
    hs = tower.proj2(hs)

    parts = [hs[i, :aftercnn_lens[i]] for i in range(B)]
    return torch.cat(parts, dim=0)


def _ragged_get_audio_features(thinker, input_features, feature_attention_mask=None,
                               audio_feature_lengths=None):
    if feature_attention_mask is not None:
        feat_lens = feature_attention_mask.sum(-1)
    else:
        feat_lens = audio_feature_lengths
    return _ragged_audio_tower_forward(thinker.audio_tower, input_features, feat_lens)


def load_model():
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRThinkerForConditionalGeneration
    from transformers import AutoProcessor

    attn_impl = "eager"
    print(f"Using attention: {attn_impl}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    thinker = Qwen3ASRThinkerForConditionalGeneration.from_pretrained(
        str(MODEL_DIR), device_map=device, dtype=dtype,
        attn_implementation=attn_impl,
    )
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    config = load_config()
    if config.get("batch_audio_tower", True):
        thinker.get_audio_features = types.MethodType(_ragged_get_audio_features, thinker)
        print("Using ragged batched audio tower")

    return thinker, processor


MAX_CHUNK_SEC = 15


def load_audio(audio_path):
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000, dtype="float32", mono=True)
    return audio


def _vad_timestamps(audio, vad_model, threshold=0.5, min_silence_ms=100):
    wav = torch.from_numpy(audio)
    return get_speech_timestamps(
        wav, vad_model,
        threshold=threshold,
        speech_pad_ms=200,
        min_speech_duration_ms=250,
        min_silence_duration_ms=min_silence_ms,
    )


def _chunks_from_timestamps(audio, ts):
    if not ts:
        return [audio]
    chunks = []
    current_start = ts[0]["start"]
    current_end = ts[0]["end"]
    for seg in ts[1:]:
        if (seg["end"] - current_start) / 16000 > MAX_CHUNK_SEC:
            chunks.append(audio[current_start:current_end])
            current_start = seg["start"]
        current_end = seg["end"]
    chunks.append(audio[current_start:current_end])
    return chunks


def _all_chunks_ok(chunks):
    return all(len(c) / 16000 <= MAX_CHUNK_SEC for c in chunks)


def split_audio_vad(audio, vad_model):
    ts = _vad_timestamps(audio, vad_model, threshold=0.5, min_silence_ms=100)
    chunks = _chunks_from_timestamps(audio, ts)
    if _all_chunks_ok(chunks):
        return chunks

    for threshold in (0.3, 0.1):
        ts = _vad_timestamps(audio, vad_model, threshold=threshold, min_silence_ms=50)
        chunks = _chunks_from_timestamps(audio, ts)
        if _all_chunks_ok(chunks):
            return chunks

    max_samples = int(MAX_CHUNK_SEC * 16000)
    return [audio[i:i + max_samples] for i in range(0, len(audio), max_samples)]
    return chunks


def build_batches(entries, max_batch_duration=300, metadata=None):
    if metadata:
        dur_map = {m["utterance_id"]: m["audio_duration_sec"] for m in metadata}
        indexed = [(i, e, dur_map.get(e["utterance_id"], 5.0)) for i, e in enumerate(entries)]
        indexed.sort(key=lambda x: x[2])
    else:
        indexed = [(i, e, 5.0) for i, e in enumerate(entries)]

    batches = []
    current_batch = []
    current_dur = 0
    for orig_idx, entry, dur in indexed:
        if current_batch and current_dur + dur > max_batch_duration:
            batches.append(current_batch)
            current_batch = []
            current_dur = 0
        current_batch.append((orig_idx, entry))
        current_dur += dur
    if current_batch:
        batches.append(current_batch)
    return batches


def _max_new_tokens_for_duration(max_dur_sec):
    """Dynamic token ceiling based on audio duration. ~2 words/sec * 1.5 tokens/word, with headroom."""
    # p98 word counts from smoketest, with 2x safety margin for tokenization + headroom
    if max_dur_sec < 1.5:
        return 16
    elif max_dur_sec < 3:
        return 24
    elif max_dur_sec < 5:
        return 32
    elif max_dur_sec < 10:
        return 48
    elif max_dur_sec < 15:
        return 72
    elif max_dur_sec < 30:
        return 128
    else:
        return 192


def transcribe_audios(thinker, processor, audios, text_prompt=None):
    from qwen_asr.inference.utils import detect_and_fix_repetitions

    if text_prompt is None:
        text_prompt = TEXT_PROMPT

    max_dur = max(len(a) / 16000 for a in audios)
    max_tokens = _max_new_tokens_for_duration(max_dur)

    prompt_inputs = processor(
        text=[text_prompt] * len(audios),
        audio=audios,
        return_tensors="pt",
        padding=True,
    )
    device = next(thinker.parameters()).device
    dtype = next(thinker.parameters()).dtype
    prompt_inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point()
        else v.to(device=device)
        for k, v in prompt_inputs.items()
    }
    prompt_len = prompt_inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated = thinker.generate(**prompt_inputs, max_new_tokens=max_tokens)

    preds = processor.batch_decode(
        generated[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [detect_and_fix_repetitions(p) for p in preds]


def prepare_batch_audio(entries, vad_model, dur_map=None, use_vad=True):
    short_audios, short_map = [], []
    long_audios, long_map = [], []

    for i, entry in enumerate(entries):
        audio_path = AUDIO_DIR / f"{entry['utterance_id']}.flac"
        audio = load_audio(audio_path)
        dur = dur_map.get(entry["utterance_id"], len(audio) / 16000) if dur_map else len(audio) / 16000
        is_short = dur < SHORT_DURATION_THRESHOLD

        audio_dur = len(audio) / 16000
        if audio_dur > 30 or (use_vad and audio_dur > MAX_CHUNK_SEC):
            chunks = split_audio_vad(audio, vad_model)
            target = short_audios if is_short else long_audios
            target_map = short_map if is_short else long_map
            start = len(target)
            target.extend(chunks)
            target_map.append((i, start, len(target)))
        else:
            target = short_audios if is_short else long_audios
            target_map = short_map if is_short else long_map
            start = len(target)
            target.append(audio)
            target_map.append((i, start, start + 1))

    return short_audios, short_map, long_audios, long_map


def reassemble_texts(all_texts, chunk_map):
    results = []
    for entry_idx, start, end in chunk_map:
        text = " ".join(all_texts[start:end])
        results.append((entry_idx, text))
    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    version_file = SRC_DIR / "VERSION"
    if version_file.exists():
        print(f"Version: {version_file.read_text().strip()}")
    t0 = time.time()
    config = load_config()

    print("Loading model...")
    thinker, processor = load_model()

    vad_model = load_silero_vad()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Models loaded on {device} in {time.time() - t0:.0f}s")

    with open(SUBMISSION_FORMAT) as f:
        submission = [json.loads(line) for line in f]

    dur_map = {}
    if METADATA.exists():
        with open(METADATA) as f:
            metadata = [json.loads(line) for line in f]
        dur_map = {m["utterance_id"]: m["audio_duration_sec"] for m in metadata}
        print(f"Loaded metadata for {len(metadata)} utterances")

    use_prompts = config.get("use_prompts", False)
    if use_prompts:
        print(f"Using duration-based prompts (threshold={SHORT_DURATION_THRESHOLD}s)")

    use_vad = config.get("vad", {}).get("enabled", False)
    print(f"VAD chunking: {'enabled' if use_vad else 'disabled'}")

    max_batch_duration = config.get("max_batch_duration", 300)
    batches = build_batches(submission, max_batch_duration, metadata if dur_map else None)
    print(f"Predicting {len(submission)} utterances in {len(batches)} batches")

    predictions = {}
    executor = ThreadPoolExecutor(max_workers=2)

    batch_entries_list = [[entry for _, entry in batch] for batch in batches]
    next_future = executor.submit(prepare_batch_audio, batch_entries_list[0], vad_model, dur_map, use_vad)

    for i, batch_entries in enumerate(batch_entries_list):
        short_audios, short_map, long_audios, long_map = next_future.result()

        if i + 1 < len(batch_entries_list):
            next_future = executor.submit(prepare_batch_audio, batch_entries_list[i + 1], vad_model, dur_map, use_vad)

        try:
            results = []
            if short_audios:
                prompt = TEXT_PROMPT_SHORT if use_prompts else TEXT_PROMPT
                short_texts = transcribe_audios(thinker, processor, short_audios, prompt)
                results.extend(reassemble_texts(short_texts, short_map))
            if long_audios:
                prompt = TEXT_PROMPT_LONG if use_prompts else TEXT_PROMPT
                long_texts = transcribe_audios(thinker, processor, long_audios, prompt)
                results.extend(reassemble_texts(long_texts, long_map))
            results.sort(key=lambda x: x[0])
            texts = [text for _, text in results]
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"OOM on batch {i} (size={len(batch_entries)}), splitting...")
                texts = []
                for entry in batch_entries:
                    sa, sm, la, lm = prepare_batch_audio([entry], vad_model, dur_map, use_vad)
                    if sa:
                        p = TEXT_PROMPT_SHORT if use_prompts else TEXT_PROMPT
                        t = transcribe_audios(thinker, processor, sa, p)
                        texts.extend([x[1] for x in reassemble_texts(t, sm)])
                    if la:
                        p = TEXT_PROMPT_LONG if use_prompts else TEXT_PROMPT
                        t = transcribe_audios(thinker, processor, la, p)
                        texts.extend([x[1] for x in reassemble_texts(t, lm)])
            else:
                raise
        for entry, text in zip(batch_entries, texts):
            predictions[entry["utterance_id"]] = text
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(batches)} batches, {len(predictions)}/{len(submission)} done, {elapsed:.0f}s")

    executor.shutdown()

    with open(OUTPUT_FILE, "w") as f:
        for entry in submission:
            uid = entry["utterance_id"]
            out = {"utterance_id": uid, "orthographic_text": predictions.get(uid, "")}
            f.write(json.dumps(out) + "\n")

    elapsed = time.time() - t0
    print(f"Saved {len(submission)} predictions to {OUTPUT_FILE} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
