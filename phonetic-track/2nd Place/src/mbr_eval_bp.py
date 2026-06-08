"""
MBR-50 decoding with blank penalty sweep on all 9 ensemble models.
Saves parquets for each (model, blank_penalty) combination.

Usage:
    uv run python src/mbr_eval_bp.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

torch.set_num_threads(1)

from src.preprocessing.phoneme_tokenizer import PhonemeTokenizer
from src.utils.decoder import MBRDecoder
from src.utils.score import score_ipa_cer

BEAM_WIDTH = 50
MBR_N_BEST = 50
BLANK_PENALTIES = [0.10, 0.15, 0.20]

RUN_DIRS = [
    # wavLM
    PROJECT_ROOT / "outputs/submit-day/18-00-56_annoying-Liam-252",
    PROJECT_ROOT / "outputs/submit-day/01-04-13_misogynistic-nuke-21",
    PROJECT_ROOT / "outputs/2026-04-05/01-43-42_lovely-mini_fridge-371",
    PROJECT_ROOT / "outputs/submit-day/01-25-33_robust-cloverfitting-10",
    # Whisper
    PROJECT_ROOT / "outputs/submit-day/06-10-00_cute-utrecht-423",
    PROJECT_ROOT / "outputs/submit-day/01-13-22_lovely-computer_science-267",
    PROJECT_ROOT / "outputs/submit-day/14-14-57_vile-Rein-275",
    # HuBERT-large
    PROJECT_ROOT / "outputs/submit-day/11-03-38_lying-utrecht-293",
    PROJECT_ROOT / "outputs/submit-day/20-44-36_lying-just_one_more_run_bro-301",
]

LABEL_FILES = [
    "data/train_phon_transcripts.jsonl",
    "data/train_phon_transcripts_talkbank.jsonl",
]


def load_all_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    for rel in LABEL_FILES:
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        df = pd.read_json(path, lines=True)
        for _, row in df.iterrows():
            labels[row["utterance_id"]] = row["phonetic_text"]
    return labels


def main():
    print("Loading ground truth labels...")
    labels = load_all_labels()
    print(f"Loaded {len(labels)} labels\n")

    tok = PhonemeTokenizer()

    for run_dir in RUN_DIRS:
        npz_path = run_dir / "fold_1/val_logits_best.npz"
        oof_parquet = run_dir / "oof_predictions_best.parquet"
        run_name = run_dir.name

        if not npz_path.exists():
            print(f"[SKIP] No logits: {run_name}")
            continue
        if not oof_parquet.exists():
            print(f"[SKIP] No parquet: {run_name}")
            continue

        print(f"\n--- {run_name} ---")

        # Load logits once per model
        data = np.load(npz_path, allow_pickle=True)
        logits_flat = data["logits"]
        offsets = data["offsets"]
        lengths = data["lengths"]
        utterance_ids = data["utterance_ids"]

        oof_df = pd.read_parquet(oof_parquet)
        oof_uids = set(oof_df["utterance_id"])

        logits_list = []
        refs = []
        valid_uids = []
        for i in range(len(utterance_ids)):
            uid = str(utterance_ids[i])
            if uid not in labels or uid not in oof_uids:
                continue
            start = int(offsets[i])
            length = int(lengths[i])
            sample = logits_flat[start:start + length].astype(np.float32)
            logits_list.append(torch.from_numpy(sample))
            refs.append(labels[uid])
            valid_uids.append(uid)

        for bp in BLANK_PENALTIES:
            bp_str = f"{bp:.2f}".replace(".", "")
            print(f"  MBR-50 blank_penalty={bp}...")

            mbr = MBRDecoder(
                tokenizer=tok,
                beam_width=BEAM_WIDTH,
                mbr_n_best=MBR_N_BEST,
                blank_penalty=bp,
            )

            preds = []
            for i in tqdm(range(0, len(logits_list), 64),
                          desc=f"  bp={bp}", leave=False):
                preds.extend(mbr(logits_list[i:i + 64]))
            mbr.close()

            per = score_ipa_cer(refs, preds)
            print(f"    PER={per:.4f}")

            # Save parquet
            uid_to_pred = dict(zip(valid_uids, preds))
            mbr_df = oof_df.copy()
            mbr_df["prediction"] = mbr_df["utterance_id"].map(
                lambda uid: uid_to_pred.get(uid, "")
            )
            mbr_df["output_length"] = mbr_df["prediction"].str.len()

            save_path = run_dir / f"oof_predictions_mbr50_bp{bp_str}.parquet"
            mbr_df.to_parquet(save_path, index=False)
            print(f"    Saved → {save_path.name}")

        # Free memory
        del logits_flat, data
        import gc; gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
