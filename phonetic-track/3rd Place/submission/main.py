from itertools import islice
import json
import os
from pathlib import Path
import numpy as np
import gc

from loguru import logger
import torch
from tqdm import tqdm

from lib.mth import MultiTaskHybridInferenceModel, LayerDropController


BATCH_SIZE = 8
PROGRESS_STEP_DENOM = 100  # Update progress bar every 1 // PROGRESS_STEP_DENOM


def batched(iterable, n, *, strict=False):
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def main():
    logger.info("Torch version: {}", torch.__version__)
    logger.info("CUDA available: {}", torch.cuda.is_available())

    # Load manifest and process data
    src_root = Path(__file__).parent.resolve()
    data_dir = Path("data")
    manifest_path = data_dir / "utterance_metadata.jsonl"

    with manifest_path.open("r") as fr:
        items = [json.loads(line) for line in fr]

    durations = np.array([item["audio_duration_sec"] for item in items if "audio_duration_sec" in item])

    print(f"Total utterances: {len(items)}")
    print(f"Min duration: {durations.min():.2f}s")
    print(f"Max duration: {durations.max():.2f}s")
    print(f"Mean duration: {durations.mean():.2f}s")
    print(f"Median duration: {np.median(durations):.2f}s")

    # Sort by audio duration for better batching
    items.sort(key=lambda x: x["audio_duration_sec"], reverse=True)

    model_paths = [
        src_root / "models" / "MTH_all_5ep_0.75alpha_4w_mt015_mf005_snr210",
        src_root / "models" / "MTH_all07_5ep_0.8alpha_4w_mt03_mf005_snr210",
        src_root / "models" / "MTH_all07_5ep_0.75alpha_4w_mt03_mf005_snr310",
        src_root / "models" / "MTH_all07_5ep_0.75alpha_4w_mt015_mf005_snr210",
    ]
    
    ensemble_preds = []
    ensemble_scores = []

    for model_path in model_paths:
        step = max(1, len(items) // PROGRESS_STEP_DENOM)
        next_log = step
        processed = 0
        logger.info(f"Loading model: {model_path.name}")
        
        controller = LayerDropController(layerdrop_prob=0)
        model = MultiTaskHybridInferenceModel.load(model_path, src_root/"models"/"w2v-bert-2.0_mt", src_root/"models"/"wavlm-large", controller=controller)

        all_preds = []
        all_scores = []
        with open(os.devnull, "w") as devnull:
            with tqdm(total=len(items), file=devnull) as pbar:
                for batch in batched(items, BATCH_SIZE):
                    preds, scores = model.predict_batch(
                        [data_dir / item["audio_path"] for item in batch],
                        batch_size=len(batch),
                    )
                    all_preds.extend(preds)
                    all_scores.extend(scores)

                    pbar.update(len(batch))
                    processed += len(batch)
                    while processed >= next_log:
                        logger.info(str(pbar))
                        next_log += step
        
        ensemble_preds.append(all_preds)
        ensemble_scores.append(all_scores)
        
        # Cleanup VRAM for the next model
        del model
        torch.cuda.empty_cache()
        gc.collect()
        logger.success(f"Finished {model_path.name}")

    ### Apply Voting Strategy
    predictions = {}
    num_samples = len(items)
    
    # Transpose lists to iterate per-utterance
    transposed_preds = list(zip(*ensemble_preds))
    transposed_scores = list(zip(*ensemble_scores))

    logger.info("Calculating final ensemble predictions (Majority Vote + Max Score)...")
    for i in range(num_samples):
        sample_preds = transposed_preds[i]
        sample_scores = transposed_scores[i]

        candidate_map = {}
        for pred, score in zip(sample_preds, sample_scores):
            if pred not in candidate_map:
                candidate_map[pred] = []
            candidate_map[pred].append(score)

        # Pick key with: 1. Max frequency, 2. Max individual score as tie-breaker
        best_pred = max(
            candidate_map.keys(),
            key=lambda k: (len(candidate_map[k]), max(candidate_map[k]))
        )

        predictions[items[i]["utterance_id"]] = best_pred
  
    # Write submission file
    submission_format_path = data_dir / "submission_format.jsonl"
    submission_path = Path("submission") / "submission.jsonl"
    submission_path.parent.mkdir(exist_ok=True)
    
    logger.info(f"Writing submission file to {submission_path}")
    with submission_format_path.open("r") as fr, submission_path.open("w") as fw:
        for line in fr:
            item = json.loads(line)
            item["phonetic_text"] = predictions[item["utterance_id"]]
            fw.write(json.dumps(item) + "\n")

    logger.success("Done.")


if __name__ == "__main__":
    main()