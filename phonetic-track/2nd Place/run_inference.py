from itertools import islice
import json
import os
from pathlib import Path
import sys
from typing import Any
import numpy as np
import torch.nn.functional as F

from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import subprocess
import tempfile
import tarfile

from omegaconf import DictConfig, OmegaConf

def install_offline_packages():
    """Extracts bundled wheels from a tar.gz and installs them using uv."""
    
    # Path to the offline_wheels directory
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    wheel_dir = base_dir / "offline_wheels"
    
    # Look for our new compressed archive
    archive_path = wheel_dir / "dependencies.tar.gz"
    
    if not archive_path.exists():
        print(f"Archive {archive_path} not found. Skipping offline install.")
        return

    print(f"Found {archive_path.name}. Extracting and installing...")

    # Create a temporary directory that disappears after installation
    with tempfile.TemporaryDirectory() as tmp_extract_dir:
        try:
            # 1. Extract the tarball
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=tmp_extract_dir)
            
            # 2. Identify extracted wheels
            extracted_wheels = [
                os.path.join(tmp_extract_dir, f) 
                for f in os.listdir(tmp_extract_dir) 
                if f.endswith(".whl") or f.endswith(".tar.gz")
            ]

            if not extracted_wheels:
                print("No wheels found inside the archive.")
                return

            # 3. Use 'uv pip' to install from the temp folder
            # --no-index ensures we don't try to hit the internet
            cmd = [
                "uv", "pip", "install",
                "--no-index",
                "--find-links", tmp_extract_dir
            ]
            
            # Run the command. capture_output=True keeps the 500-line log clean
            subprocess.run(cmd + extracted_wheels, check=True, capture_output=True)
            print(f"Successfully installed {len(extracted_wheels)} offline packages!")

        except tarfile.TarError as e:
            print(f"[!] Failed to extract archive: {e}")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"\n[!] uv install failed. Error:\n{e.stderr.decode()}\n")
            sys.exit(1)

if __name__ == "__main__":
    install_offline_packages()

PROJECT_ROOT = Path(__file__).resolve().parent
# print(f"Project root: {PROJECT_ROOT}")
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataset import prepare_dl_dataset

PROGRESS_STEP_DENOM = 10  # Update progress bar every 1 // PROGRESS_STEP_DENOM
ENABLE_RUNTIME_DEMUCS = False
# RUNTIME_DEMUCS_MODEL = "mdx_extra"
DEMUCS_CHILD_ORIGINAL_MIX = 0.25
# RUNTIME_DEMUCS_REPO_DIR = PROJECT_ROOT / "offline_wheels" / "demucs_repo"

# # ENSEMBLE CONFIG
# ENSEMBLE = False
# FUSION_RULE = "arithmetic"
# WEIGHT_A = 0.20
# TEMP_A = 0.70
# TEMP_B = 0.90
# BLANK_SCALE_A = 1.00
# BLANK_SCALE_B = 0.60

TTA = False
TTA_SPEEDS = [0.96, 1.0, 1.04]

output_paths = [
    # wavLM-large (6) — includes 2 trained on all data
    PROJECT_ROOT / "outputs/submit-day/18-00-56_annoying-Liam-252",
    PROJECT_ROOT / "outputs/submit-day/01-04-13_misogynistic-nuke-21",
    PROJECT_ROOT / "outputs/submit-day/23-41-19_best-Geert-292",                # trained on all data (replaces mini_fridge-371)
    PROJECT_ROOT / "outputs/submit-day/23-50-07_skilled-cloverfitting-294",    # trained on all data (replaces robust-cloverfitting-10)
    PROJECT_ROOT / "outputs/2026-04-04/07-29-58_tyfus-brain-367",
    PROJECT_ROOT / "outputs/submit-day/15-48-50_tyfus-marvin_wants_sweaters-7",
    # Whisper-large (3)
    PROJECT_ROOT / "outputs/submit-day/06-10-00_cute-utrecht-423",
    PROJECT_ROOT / "outputs/submit-day/01-13-22_lovely-computer_science-267",
    PROJECT_ROOT / "outputs/submit-day/14-14-57_vile-Rein-275",
    # Whisper-medium (2)
    PROJECT_ROOT / "outputs/2026-04-05/23-59-25_dumb-Sietse-377",
    PROJECT_ROOT / "outputs/2026-04-06/07-26-07_skilled-guys_no_shakeup_max_0.02-384",
    # HuBERT-large (2)
    PROJECT_ROOT / "outputs/submit-day/11-03-38_lying-utrecht-293",
    PROJECT_ROOT / "outputs/submit-day/20-44-36_lying-just_one_more_run_bro-301",
]

# OUTPUT_PATH_A = PROJECT_ROOT / "outputs" / "2026-03-23" / "18-40-44_japanese-where_merch-259"
# OUTPUT_PATH_B = PROJECT_ROOT / "outputs" / "submit-day" / "20-44-36_lying-just_one_more_run_bro-301"

# # PATH IF NO ENSEMBLE
# # OUTPUT_PATH = PROJECT_ROOT / "outputs" / "2026-03-15" / "14-15-00_robust-cv_over_lb-202"
# OUTPUT_PATH = PROJECT_ROOT / "outputs/2026-03-30/18-00-56_annoying-Liam-252"

import difflib
from collections import Counter

def speed_perturb_waveforms(
        input_features: torch.Tensor,
        input_lengths: torch.Tensor,
        speed: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if abs(speed - 1.0) < 1e-8:
            lengths = input_lengths.clone()
            max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
            features = input_features[:, :max_len].contiguous()
            mask = torch.zeros_like(features, dtype=torch.long)
            for idx, length in enumerate(lengths.tolist()):
                mask[idx, : int(length)] = 1
            return features, lengths, mask

        per_sample: list[torch.Tensor] = []
        new_lengths: list[int] = []
        for idx, src_len in enumerate(input_lengths.tolist()):
            valid = input_features[idx, : int(src_len)]
            target_len = max(1, int(round(float(src_len) / speed)))
            warped = F.interpolate(
                valid.view(1, 1, -1),
                size=target_len,
                mode="linear",
                align_corners=False,
            ).view(-1)
            per_sample.append(warped)
            new_lengths.append(target_len)

        max_len = max(new_lengths)
        batch_size = input_features.shape[0]
        warped_batch = input_features.new_zeros((batch_size, max_len))
        warped_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=input_features.device)
        for idx, warped in enumerate(per_sample):
            valid_len = warped.shape[0]
            warped_batch[idx, :valid_len] = warped
            warped_mask[idx, :valid_len] = 1

        warped_lengths = torch.tensor(new_lengths, device=input_lengths.device, dtype=input_lengths.dtype)
        return warped_batch, warped_lengths, warped_mask

def _edit_distance(a, b) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _align_to_backbone(backbone, hyp):
    _DEL = None
    m, n = len(backbone), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if backbone[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    raw = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and backbone[i - 1] == hyp[j - 1]:
            raw.append((backbone[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            raw.append((backbone[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            raw.append((backbone[i - 1], _DEL)); i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            raw.append((_DEL, hyp[j - 1])); j -= 1
        else:
            break
    raw.reverse()
    votes = {}
    insertions = {}
    bb_pos = -1
    for a_ch, b_ch in raw:
        if a_ch is not _DEL:
            bb_pos += 1
            votes[bb_pos] = b_ch
        else:
            ins_after = max(bb_pos, -1)
            insertions.setdefault(ins_after, []).append(b_ch)
    return votes, insertions


def char_rover(hypotheses, weights=None):
    """Character-level ROVER: align to medoid backbone, weighted majority vote."""
    _DEL = None
    if not hypotheses:
        return ""
    if all(h == hypotheses[0] for h in hypotheses):
        return hypotheses[0]
    n = len(hypotheses)
    if weights is None:
        weights = [1.0] * n

    # Pick medoid backbone
    best_idx, best_dist = 0, float("inf")
    for i in range(n):
        total = sum(_edit_distance(hypotheses[i], hypotheses[j]) for j in range(n) if j != i)
        if total < best_dist:
            best_dist = total
            best_idx = i
    backbone = hypotheses[best_idx]
    if not backbone:
        return Counter(hypotheses).most_common(1)[0][0]

    # Align each hypothesis to backbone independently
    all_votes = []
    all_insertions = []
    for idx, hyp in enumerate(hypotheses):
        if idx == best_idx:
            all_votes.append({p: backbone[p] for p in range(len(backbone))})
            all_insertions.append({})
        else:
            v, ins = _align_to_backbone(backbone, hyp)
            all_votes.append(v)
            all_insertions.append(ins)

    total_weight = sum(weights)
    result = []

    ins_counter = Counter()
    for idx, ins in enumerate(all_insertions):
        if -1 in ins:
            ins_counter[tuple(ins[-1])] += weights[idx]
    for ins_key, w in ins_counter.most_common():
        if w > total_weight / 2:
            result.extend(ins_key)
            break

    for p in range(len(backbone)):
        col = Counter()
        voted_weight = 0.0
        for idx, votes in enumerate(all_votes):
            v = votes.get(p, _DEL)
            col[v] += weights[idx]
            voted_weight += weights[idx]
        col[_DEL] = col.get(_DEL, 0) + (total_weight - voted_weight)
        del_w = col.pop(_DEL, 0)
        if col:
            best_char, best_w = col.most_common(1)[0]
            if best_w >= del_w:
                result.append(best_char)

        ins_counter = Counter()
        for idx, ins in enumerate(all_insertions):
            if p in ins:
                ins_counter[tuple(ins[p])] += weights[idx]
        for ins_key, w in ins_counter.most_common():
            if w > total_weight / 2:
                result.extend(ins_key)
                break

    return "".join(result)

def trim_logits_batch(logits: torch.Tensor, output_lengths: torch.Tensor) -> list[torch.Tensor]:
    logits_cpu = logits.detach().cpu()
    lengths_cpu = output_lengths.detach().cpu().tolist()
    return [seq_logits[: int(seq_len)].contiguous() for seq_logits, seq_len in zip(logits_cpu, lengths_cpu)]


def _trim_log_probs_batch(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
) -> list[torch.Tensor]:
    lengths = output_lengths.detach().cpu().tolist()
    return [
        seq_log_probs[: int(seq_len)].detach().cpu().contiguous()
        for seq_log_probs, seq_len in zip(log_probs, lengths)
    ]


def _decode_with_beam_width(
    model: torch.nn.Module,
    input_features: torch.Tensor,
    input_lengths: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> list[str]:
    
    decoder = model.decoder
    output = model(input_features, attention_mask=attention_mask)
    log_probs = output[0]
    output_lengths = model.get_output_lengths(input_lengths)
    trimmed = _trim_log_probs_batch(log_probs, output_lengths)
    return decoder(trimmed)


def _prediction_token_count(prediction: str, tokenizer: Any) -> int:
    token_ids = tokenizer(prediction)
    blank_id = getattr(tokenizer, "blank_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    return sum(1 for token_id in token_ids if token_id != blank_id and token_id != pad_id)


def _load_runtime_demucs_model(
    model_name: str,
    device: torch.device,
    repo_dir: Path | None = None,
) -> Any:

    from demucs.pretrained import get_model

    model_repo = repo_dir if repo_dir is not None and repo_dir.exists() else None

    def _load_with_demucs() -> Any:
        if model_repo is not None:
            return get_model(model_name, repo=model_repo)
        return get_model(model_name)

    try:
        demucs_model = _load_with_demucs()
    except Exception as exc:
        message = str(exc)
        if "DiffQ" in message or "diffq" in message:
            raise RuntimeError(
                f"Demucs model '{model_name}' requires `diffq`. "
                "Either install it (`uv pip install diffq`) or use a non-quantized model, "
                "for example `mdx_extra`."
            ) from exc

        # PyTorch>=2.6 changed torch.load default to weights_only=True. Demucs
        # checkpoints include module objects and need weights_only=False.
        if "Weights only load failed" in message or "weights_only" in message:
            try:
                import demucs.states as demucs_states

                original_torch_load = demucs_states.torch.load

                def _torch_load_compat(*args, **kwargs):
                    kwargs.setdefault("weights_only", False)
                    return original_torch_load(*args, **kwargs)

                demucs_states.torch.load = _torch_load_compat
                demucs_model = _load_with_demucs()
            except Exception as retry_exc:
                if model_repo is not None:
                    raise RuntimeError(
                        f"Failed to load Demucs model '{model_name}' from local repo '{model_repo}'. "
                        "Model files exist, but checkpoint deserialization still failed under current "
                        "torch/demucs versions."
                    ) from retry_exc
                raise
            finally:
                if "demucs_states" in locals() and "original_torch_load" in locals():
                    demucs_states.torch.load = original_torch_load
        else:
            if model_repo is not None:
                raise RuntimeError(
                    f"Failed to load Demucs model '{model_name}' from local repo '{model_repo}'. "
                    "Ensure the repo contains the matching .yaml bag file and all required .th "
                    "checkpoint files."
                ) from exc
            raise

    demucs_model.to(device)
    demucs_model.eval()
    return demucs_model


def _build_runtime_demucs_batch(
    original_audio: torch.Tensor,
    input_lengths: torch.Tensor,
    demucs_model: Any,
    child_original_mix: float = DEMUCS_CHILD_ORIGINAL_MIX,
) -> torch.Tensor:
    logger.info("[Demucs] Building runtime Demucs features for batch of {} utterances...", original_audio.shape[0])
    from demucs.apply import apply_model


    mix_ratio = float(max(0.0, min(1.0, child_original_mix)))
    max_valid_len = int(input_lengths.max().item())
    if max_valid_len <= 0:
        return original_audio

    mono = original_audio[:, :max_valid_len].float()
    stereo = torch.stack([mono, mono], dim=1)
    print(original_audio.device, stereo.device)

    with torch.autocast(device_type=original_audio.device.type, enabled=False):
        sources = apply_model(
            demucs_model,
            stereo,
            device=original_audio.device,
            split=True,
            overlap=0.25,
            progress=False,
        )
    logger.info("[Demucs] Applied Demucs model to batch.")

    source_names = list(getattr(demucs_model, "sources", []))
    vocals_idx = source_names.index("vocals") if "vocals" in source_names else 0
    demucs_mono = sources[:, vocals_idx].mean(dim=1)

    blended = (1.0 - mix_ratio) * demucs_mono + mix_ratio * mono
    peak = blended.abs().amax(dim=1, keepdim=True)
    scale = torch.where(peak > 1.0, peak, torch.ones_like(peak))
    blended = blended / scale

    logger.info("[Demucs] Blended Demucs vocals with original audio using mix ratio {}.", mix_ratio)

    out = torch.zeros_like(original_audio)
    out[:, :max_valid_len] = blended.to(dtype=original_audio.dtype)
    time_axis = torch.arange(original_audio.shape[1], device=original_audio.device).unsqueeze(0)
    valid_mask = time_axis < input_lengths.unsqueeze(1)
    out = out.masked_fill(~valid_mask, 0.0)
    return out


def choose_demucs_or_original_prediction(
    model: torch.nn.Module,
    tokenizer: Any,
    original_audio: torch.Tensor,
    demucs_audio: torch.Tensor,
    original_input_lengths: torch.Tensor,
    demucs_input_lengths: torch.Tensor | None = None,
    original_attention_mask: torch.Tensor | None = None,
    demucs_attention_mask: torch.Tensor | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    if demucs_input_lengths is None:
        demucs_input_lengths = original_input_lengths

    original_predictions = _decode_with_beam_width(
        model=model,
        input_features=original_audio,
        input_lengths=original_input_lengths,
        attention_mask=original_attention_mask,
    )
    demucs_predictions = _decode_with_beam_width(
        model=model,
        input_features=demucs_audio,
        input_lengths=demucs_input_lengths,
        attention_mask=demucs_attention_mask,
    )

    final_predictions: list[str] = []
    selected_source: list[str] = []
    for pred_orig, pred_demucs in zip(original_predictions, demucs_predictions):
        orig_tokens = _prediction_token_count(pred_orig, tokenizer)
        demucs_tokens = _prediction_token_count(pred_demucs, tokenizer)

        if demucs_tokens < orig_tokens:
            final_predictions.append(pred_orig)
            selected_source.append("original")
        else:
            final_predictions.append(pred_demucs)
            selected_source.append("demucs")

    return final_predictions, selected_source, original_predictions, demucs_predictions


def infer_original_log_probs(
    model: torch.nn.Module,
    input_features: torch.Tensor,
    input_lengths: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    output = model(input_features, attention_mask=attention_mask)
    log_probs = output[0]
    output_lengths = model.get_output_lengths(input_lengths)
    return _trim_log_probs_batch(log_probs, output_lengths)


def decode_log_probs_in_batches(
    decoder: Any,
    utterance_ids: list[str],
    log_probs_by_utt: dict[str, torch.Tensor],
    batch_size: int = 256,
    desc: str = "Decoding",
) -> dict[str, str]:
    decoded: dict[str, str] = {}
    for start_idx in range(0, len(utterance_ids), batch_size):
        end_idx = min(len(utterance_ids), start_idx + batch_size)
        batch_ids = utterance_ids[start_idx:end_idx]
        batch_sequences = [log_probs_by_utt[utt_id] for utt_id in batch_ids]
        batch_predictions = decoder(batch_sequences)
        for utt_id, pred in zip(batch_ids, batch_predictions):
            decoded[utt_id] = pred
    return decoded


def predict(
    model: torch.nn.Module,
    input_features: torch.Tensor,
    input_lengths: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> list[str]:
    output = model(input_features, attention_mask=attention_mask)
    logits = output[1]
    output_lengths = model.get_output_lengths(input_lengths)

    logits = logits.to(torch.float32)
    trimmed_logits = trim_logits_batch(logits, output_lengths)
    return trimmed_logits

def extract_logits(output_path, device, batch_size=64):
    config = output_path / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(config)
    cfg.preprocessing.max_duration_sec = 40 #longest audio is 36 seconds
    cfg.preprocessing.min_duration_sec = 0 # during inference, don't filter out the too short ones
    cfg.training.dataloader.batch_size = 64  # A100 80GB 
    cfg.training.dataloader.num_workers = 16  # 24 vCPUs available in runtime
    cfg.model.decoder._target_ = "src.utils.decoder.MBRDecoder"
    cfg.model.decoder.beam_width = 50
    cfg.model.decoder.mbr_n_best = 50
    # cfg.model.decoder.temperature = 1.0
    # cfg.model.decoder.blank_penalty = 0.25
    # cfg.model.decoder.alpha = 0.102
    # cfg.model.decoder.beta = -0.081
    # cfg.model.decoder.repeat_penalty = -1.736
    model_path = output_path / "fold_1" / "best_model.pth"
    if not model_path.exists():
        model_path = output_path / "fold_0" / "best_model.pth"

    data_loader, _, dataset_info = prepare_dl_dataset(cfg, fold=0, inference=True)
    
    decoder = instantiate(
        cfg.model.decoder,
        tokenizer=dataset_info.get("tokenizer"),
    )

    model = instantiate(cfg.model, vocab_size=dataset_info["vocab_size"], vocab=dataset_info["tokenizer"], inference=True, decoder=decoder)
    state_dict = torch.load(model_path, map_location="cpu")
    if "ema_state_dict" in state_dict and state_dict["ema_state_dict"] is not None:
        logger.info(f"Loading EMA weights for {output_path.name}...")
        state_dict = state_dict['ema_state_dict']
        state_dict = {k[len("ema_model."):]: v for k, v in state_dict.items() if k.startswith("ema_model.")}
        model.load_state_dict(state_dict)
    else:        
        logger.info(f"Loading regular weights for {output_path.name}...")
        model.load_state_dict(state_dict['model_state_dict'], strict=False)
        
    model = model.to(device)
    model.eval()

    logger.info(f"Processing {dataset_info['train_size']} utterances from {output_path.name}")
    step = max(1, dataset_info['train_size'] // PROGRESS_STEP_DENOM)

    all_logits = {}
    next_log = step
    processed = 0

    with torch.no_grad():
        for batch in data_loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                input_features = batch["input_features"].to(device)
                input_lengths = batch["input_lengths"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                if TTA:
                    for speed in TTA_SPEEDS:
                        tta_features, tta_lengths, tta_attention_mask = speed_perturb_waveforms(
                            input_features=input_features,
                            input_lengths=input_lengths,
                            speed=speed,
                        )
                        trimmed_logits = predict(
                            model=model,
                            input_features=tta_features,
                            input_lengths=tta_lengths,
                            attention_mask=tta_attention_mask
                        )
                        for utt_id, logit_seq in zip(batch["utterance_ids"], trimmed_logits):
                            if utt_id not in all_logits:
                                all_logits[utt_id] = []
                            all_logits[utt_id].append(logit_seq.numpy())

                else:
                    trimmed_logits = predict(
                        model=model,
                        input_features=input_features,
                        input_lengths=input_lengths,
                        attention_mask=attention_mask
                    )
                    for utt_id, logit_seq in zip(batch["utterance_ids"], trimmed_logits):
                        if utt_id not in all_logits:
                            all_logits[utt_id] = []
                        all_logits[utt_id].append(logit_seq.numpy())

            
            

            processed += len(batch["utterance_ids"])
            if processed >= next_log:
                logger.info(f"Processed {processed}/{dataset_info['train_size']} utterances...")
                next_log += step

    return all_logits, decoder, dataset_info["tokenizer"].blank_token_id

def main():
    # Diagnostics
    logger.info("Torch version: {}", torch.__version__)
    logger.info("CUDA available: {}", torch.cuda.is_available())
    logger.info("CUDA device count: {}", torch.cuda.device_count())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    predictions = {}

    for output_path in output_paths:
        logger.info(f"Extracting logits from {output_path.name}...")
        logits_by_utt, decoder, blank_id = extract_logits(output_path, device=device)
        logger.info(f"Extracted logits for {len(logits_by_utt)} utterances from {output_path.name}.")
        
        logger.info(f"Decoding predictions for {output_path.name}...")
        utterance_ids_ordered = sorted(logits_by_utt.keys())
        
        if TTA:
            logger.info(f"TTA enabled with {len(TTA_SPEEDS)} speeds: {TTA_SPEEDS}")
            predictions_per_speed = {}
            
            for speed_idx, speed in enumerate(TTA_SPEEDS):
                logger.info(f"  Decoding TTA speed {speed_idx + 1}/{len(TTA_SPEEDS)}: {speed:.2f}x")
                speed_logits = [logits_by_utt[utt_id][speed_idx] for utt_id in utterance_ids_ordered]
                speed_predictions = decoder(speed_logits)
                predictions_per_speed[speed_idx] = speed_predictions
            
            logger.info("Ensembling TTA predictions via alignment voting...")
            decoded_predictions = {}
            for utt_idx, utt_id in enumerate(utterance_ids_ordered):
                hypotheses = [predictions_per_speed[speed_idx][utt_idx] for speed_idx in range(len(TTA_SPEEDS))]
                decoded_predictions[utt_id] = char_rover(hypotheses)
        else:
            speed_logits = [logits_by_utt[utt_id][0] for utt_id in utterance_ids_ordered]
            decoded_predictions_list = decoder(speed_logits)
            decoded_predictions = dict(zip(utterance_ids_ordered, decoded_predictions_list))
        
        for utt_id, pred in decoded_predictions.items():
            if utt_id not in predictions:
                predictions[utt_id] = []
            predictions[utt_id].append(pred)
        logger.info(f"Decoded predictions for {len(decoded_predictions)} utterances from {output_path.name}.")

        # Free memory before loading next model
        del logits_by_utt, decoder, decoded_predictions
        import gc; gc.collect()
        torch.cuda.empty_cache()


    # print(predictions)

    # Char-level ROVER ensemble with whisper 1.5x boost on short sequences
    # Whisper models: indices 6-10 (3 whisper-large + 2 whisper-medium)
    WHISPER_IDX = {6, 7, 8, 9, 10}
    n_models = len(output_paths)
    logger.info(f"Running char-level ROVER on {len(predictions)} utterances across {n_models} models...")
    logger.info(f"Whisper 1.5x boost for median pred length <= 10")
    final_predictions = {}
    for utt_id, hyps in predictions.items():
        med_len = sorted(len(h) for h in hyps)[len(hyps) // 2]
        weights = [1.5 if i in WHISPER_IDX and med_len <= 10 else 1.0 for i in range(n_models)]
        final_predictions[utt_id] = char_rover(hyps, weights)

    logger.success("Transcription complete.")

    # # Write submission file
    # # In the container data/ sits one level above src/ (PROJECT_ROOT), so fall back to parent
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        data_dir = PROJECT_ROOT.parent / "data"
    submission_format_path = data_dir / "submission_format.jsonl"
    submission_path = Path("submission") / "submission.jsonl"
    submission_path.parent.mkdir(parents=True, exist_ok=True)  # create submission/ dir if needed

    logger.info(f"Writing submission file to {submission_path}")
    with submission_format_path.open("r") as fr:
        lines = [json.loads(line) for line in fr]

    with submission_path.open("w") as fw:
        for item in lines:
            item["phonetic_text"] = final_predictions[item["utterance_id"]]
            fw.write(json.dumps(item) + "\n")

    logger.success("Done.")


if __name__ == "__main__":
    main()
