import os
from itertools import islice
import json
import os
from pathlib import Path
import numpy as np
import math
from loguru import logger
import torch
from tqdm import tqdm
from jiwer import cer

from lib.mth import MultiTaskHybridInferenceModel, LayerDropController


BATCH_SIZE = 32
PROGRESS_STEP_DENOM = 100  # Update progress bar every 1 // PROGRESS_STEP_DENOM


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch



logger.info("Torch version: {}", torch.__version__)
logger.info("CUDA available: {}", torch.cuda.is_available())
logger.info("CUDA device count: {}", torch.cuda.device_count())


# Load manifest and process data
data_dir = Path("../data")
manifest_path = data_dir / "nolabel_phon.json"
output_dir = Path("outputs")

with manifest_path.open("r") as fr:
    items = [json.loads(line) for line in fr]#[:100]

for i, item in enumerate(items):
    if "utterance_id" not in item:
        item["utterance_id"] = i

save_path = output_dir / "pseudo_phon.json"
with save_path.open("w") as fw:
    for item in items:
        fw.write(json.dumps(item) + "\n")

# Sort by audio duration for better batching
items.sort(key=lambda x: x["duration"], reverse=True)

logger.info(f"Processing {len(items)} utterances from {manifest_path}")


model_paths = [
    "./models/MTH_both_5ep_0.7alpha_4w_mt03_mf00_snr315",
    "./models/MTH_both_5ep_0.8alpha_4w_mt03_mf005_snr210",
    "./models/MTH_both_5ep_0.75alpha_4w_mt015_mf005_snr210",
    "./models/MTH_both_5ep_0.75alpha_4w_mt03_mf005_snr210"
]


ensemble_preds = []
ensemble_scores = []

for model_path in model_paths:
    step = max(1, len(items) // PROGRESS_STEP_DENOM)
    next_log = step
    processed = 0
    logger.info("Starting transcription...")
    logger.info(f"Loading model from: {model_path}")

    controller = LayerDropController(layerdrop_prob=0)
    w2v_local = Path("./w2v-bert-2.0_mt/")
    wavlm_local = Path("./wavlm-large/")
    model = MultiTaskHybridInferenceModel.load(model_path, w2v_local, wavlm_local, controller=controller)


    all_preds = []
    # all_scores = []
    with open(os.devnull, "w") as devnull:
        with tqdm(total=len(items), file=devnull) as pbar:
            for batch in batched(items, BATCH_SIZE):
                results = model.predict_batch(
                    [data_dir / item["audio_filepath"] for item in batch],
                    [item["text"] for item in batch], ### add when ensemble
                    batch_size=len(batch),
                )
                all_preds.extend(results)
                # all_scores.extend(scores)

                this_batch_size = len(batch)
                pbar.update(this_batch_size)
                processed += this_batch_size
                while processed >= next_log:
                    logger.info(str(pbar))
                    next_log += step
    ensemble_preds.append(all_preds)
    # ensemble_scores.append(all_scores)
    logger.success("Transcription complete.")


# 2. Perform Ensemble (Highest Score Selection)
logger.info("Ensembling results and calculating confidence scores...")
final_items = []
num_items = len(items)

for i in range(num_items):
    # 1. Collect Top-1 from each model
    candidates = [model_res[i]['hypotheses'][0] for model_res in ensemble_preds]
    
    # 2. Pick the candidate with the highest log-likelihood score
    best_cand = max(candidates, key=lambda x: x['score'])
    
    score = math.exp(best_cand['score'])
    pseudo_phoneme = best_cand.get('text', "").strip()
    
    # 4. Construct output entry
    output_item = items[i].copy()
    output_item["text"] = pseudo_phoneme  # Predicted phonemes
    output_item["pseudo_score"] = round(float(score), 4) # Save score for analysis
    
    final_items.append(output_item)

# 3. Save to JSONL
final_save_path = output_dir / "pseudolabel_phon.json"
with final_save_path.open("w", encoding="utf-8") as fw:
    for item in final_items:
        fw.write(json.dumps(item, ensure_ascii=False) + "\n")

logger.success(f"Saved {len(final_items)} entries to {final_save_path}")