"""
Evaluate Greedy, BeamSearch, and MBR decoding across multiple runs.
Saves MBR predictions as parquets for downstream ROVER ensembling.

Loads pre-saved val_logits_best.npz from each run directory and reports
PER (IPA-CER via src/utils/score.py) for each decoder.

Usage:
    uv run python src/mbr_eval.py                  # all runs
    uv run python src/mbr_eval.py 367 368           # specific runs (suffix match)
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
from src.utils.decoder import BeamSearchDecoder, GreedyDecoder, MBRDecoder
from src.utils.score import score_ipa_cer

BEAM_WIDTH = 50
MBR_N_BEST = 50

# All ensemble runs (SSL + Whisper + HuBERT-large)
RUN_DIRS = [
    # SSL
    PROJECT_ROOT / "outputs/submit-day/18-00-56_annoying-Liam-252",
    PROJECT_ROOT / "outputs/2026-04-03/15-48-50_annoying-guy-364",
    PROJECT_ROOT / "outputs/2026-04-03/22-31-48_sexist-addiction-365",
    PROJECT_ROOT / "outputs/2026-04-04/00-55-20_vile-turfje-366",
    PROJECT_ROOT / "outputs/2026-04-04/07-29-58_tyfus-brain-367",
    PROJECT_ROOT / "outputs/2026-04-04/09-58-24_smelly-addiction-368",
    PROJECT_ROOT / "outputs/submit-day/09-47-06_lying-where_merch-160",  # SSL (no logits)
    # Whisper
    PROJECT_ROOT / "outputs/2026-04-01/00-36-12_cute-Rein-331",
    PROJECT_ROOT / "outputs/2026-04-02/17-12-22_raging-kaggle-358",
    PROJECT_ROOT / "outputs/submit-day/14-14-57_vile-Rein-275",
    PROJECT_ROOT / "outputs/submit-day/01-13-22_lovely-computer_science-267",
    PROJECT_ROOT / "outputs/submit-day/06-10-00_cute-utrecht-423",
    # HuBERT-large
    PROJECT_ROOT / "outputs/submit-day/20-44-36_lying-just_one_more_run_bro-301",
    PROJECT_ROOT / "outputs/submit-day/11-03-38_lying-utrecht-293",  # (no logits yet)
    # Additional runs
    PROJECT_ROOT / "outputs/submit-day/01-25-33_robust-cloverfitting-10",
    PROJECT_ROOT / "outputs/submit-day/01-04-13_misogynistic-nuke-21",
    PROJECT_ROOT / "outputs/2026-04-05/01-43-42_lovely-mini_fridge-371",
    # TTA runs (Liam-252 speed perturbation)
    PROJECT_ROOT / "outputs/submit-day/tta_096_annoying-Liam-252",
    PROJECT_ROOT / "outputs/submit-day/tta_104_annoying-Liam-252",
    # New models for 13-model ensemble
    PROJECT_ROOT / "outputs/submit-day/05-36-27_best-kaggle_bronze-189",       # wavLM base-plus
    PROJECT_ROOT / "outputs/submit-day/04-27-04_dumb-computer_science-107",    # wavLM base-plus
    PROJECT_ROOT / "outputs/2026-04-05/23-59-25_dumb-Sietse-377",             # whisper-medium
    PROJECT_ROOT / "outputs/2026-04-06/07-26-07_skilled-guys_no_shakeup_max_0.02-384",  # whisper-medium
    PROJECT_ROOT / "outputs/submit-day/23-51-47_japanese-scam-28",             # wavLM (replaces nuke-21)
    PROJECT_ROOT / "outputs/submit-day/23-50-51_homeless-jimmy-27",            # wavLM (replaces Liam-252)
    PROJECT_ROOT / "outputs/submit-day/15-48-50_tyfus-marvin_wants_sweaters-7",  # wavLM-large
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


def load_logits_and_meta(
    npz_path: Path, oof_parquet: Path, labels: dict[str, str]
) -> tuple[list[torch.Tensor], list[str], pd.DataFrame]:
    """Load logits and OOF parquet metadata. Returns (logits_list, refs, oof_df)."""
    data = np.load(npz_path, allow_pickle=True)
    logits_flat = data["logits"]
    offsets = data["offsets"]
    lengths = data["lengths"]
    utterance_ids = data["utterance_ids"]

    # Load OOF parquet for metadata (child_id, fold, etc.)
    oof_df = pd.read_parquet(oof_parquet)
    oof_uids = set(oof_df["utterance_id"])

    all_logits: list[torch.Tensor] = []
    all_refs: list[str] = []
    valid_uids: list[str] = []
    for i in range(len(utterance_ids)):
        uid = str(utterance_ids[i])
        if uid not in labels or uid not in oof_uids:
            continue
        start = int(offsets[i])
        length = int(lengths[i])
        sample = logits_flat[start : start + length].astype(np.float32)
        all_logits.append(torch.from_numpy(sample))
        all_refs.append(labels[uid])
        valid_uids.append(uid)

    return all_logits, all_refs, oof_df, valid_uids


def decode_batched(
    decoder, logits_list: list[torch.Tensor], desc: str, batch_size: int = 64
) -> list[str]:
    preds: list[str] = []
    for i in tqdm(range(0, len(logits_list), batch_size), desc=desc, leave=False):
        preds.extend(decoder(logits_list[i : i + batch_size]))
    return preds


def evaluate_run(run_dir: Path, labels: dict[str, str]) -> dict:
    npz_path = run_dir / "fold_1/val_logits_best.npz"
    oof_parquet = run_dir / "oof_predictions_best.parquet"
    run_name = run_dir.name

    logits_list, refs, oof_df, valid_uids = load_logits_and_meta(npz_path, oof_parquet, labels)
    tok = PhonemeTokenizer()

    # Greedy
    greedy = GreedyDecoder(tok)
    greedy_preds = decode_batched(greedy, logits_list, f"{run_name} | Greedy")
    greedy_per = score_ipa_cer(refs, greedy_preds)

    # Beam
    beam = BeamSearchDecoder(tokenizer=tok, beam_width=BEAM_WIDTH)
    beam_preds = decode_batched(beam, logits_list, f"{run_name} | Beam")
    beam_per = score_ipa_cer(refs, beam_preds)
    beam.close()

    # MBR
    mbr = MBRDecoder(tokenizer=tok, beam_width=BEAM_WIDTH, mbr_n_best=MBR_N_BEST)
    mbr_preds = decode_batched(mbr, logits_list, f"{run_name} | MBR")
    mbr_per = score_ipa_cer(refs, mbr_preds)
    mbr.close()

    # Save MBR predictions as parquet (same format as oof_predictions_best)
    uid_to_mbr = dict(zip(valid_uids, mbr_preds))
    mbr_df = oof_df.copy()
    mbr_df["prediction"] = mbr_df["utterance_id"].map(
        lambda uid: uid_to_mbr.get(uid, "")
    )
    # Update output_length to reflect MBR prediction lengths
    mbr_df["output_length"] = mbr_df["prediction"].str.len()

    save_path = run_dir / "oof_predictions_mbr50.parquet"
    mbr_df.to_parquet(save_path, index=False)
    print(f"  Saved MBR predictions → {save_path}")

    return {
        "run": run_name,
        "n_utts": len(refs),
        "greedy": greedy_per,
        "beam": beam_per,
        "mbr": mbr_per,
        "mbr_vs_greedy": mbr_per - greedy_per,
        "mbr_vs_beam": mbr_per - beam_per,
        "mbr_best": mbr_per <= beam_per and mbr_per <= greedy_per,
    }


def main() -> None:
    # Optional: filter runs by suffix match from CLI args
    filter_suffixes = sys.argv[1:] if len(sys.argv) > 1 else None

    print("Loading ground truth labels...")
    labels = load_all_labels()
    print(f"Loaded {len(labels)} labels\n")

    rows = []
    for run_dir in RUN_DIRS:
        # Filter by suffix if specified
        if filter_suffixes:
            if not any(s in run_dir.name for s in filter_suffixes):
                continue

        npz = run_dir / "fold_1/val_logits_best.npz"
        oof = run_dir / "oof_predictions_best.parquet"
        if not npz.exists():
            print(f"[SKIP] No NPZ: {run_dir.name}")
            continue
        if not oof.exists():
            print(f"[SKIP] No OOF parquet: {run_dir.name}")
            continue

        print(f"--- {run_dir.name} ---")
        row = evaluate_run(run_dir, labels)
        rows.append(row)
        print(
            f"  Greedy: {row['greedy']:.4f}  "
            f"Beam: {row['beam']:.4f}  "
            f"MBR: {row['mbr']:.4f}  "
            f"Δ(MBR-Beam): {row['mbr_vs_beam']:+.4f}\n"
        )

    if not rows:
        print("No runs evaluated.")
        return

    print("\n" + "=" * 80)
    print(f"{'Run':<45} {'Greedy':>7} {'Beam':>7} {'MBR':>7} {'Δ(MBR-Greedy)':>14} {'Δ(MBR-Beam)':>12} {'MBR best?':>10}")
    print("-" * 80)
    for r in rows:
        print(
            f"{r['run']:<45} "
            f"{r['greedy']:>7.4f} "
            f"{r['beam']:>7.4f} "
            f"{r['mbr']:>7.4f} "
            f"{r['mbr_vs_greedy']:>+14.4f} "
            f"{r['mbr_vs_beam']:>+12.4f} "
            f"{'YES' if r['mbr_best'] else 'NO':>10}"
        )
    print("-" * 80)

    mbr_wins = sum(r["mbr_best"] for r in rows)
    avg_delta_beam = sum(r["mbr_vs_beam"] for r in rows) / len(rows)
    avg_delta_greedy = sum(r["mbr_vs_greedy"] for r in rows) / len(rows)
    print(
        f"\nMBR is best in {mbr_wins}/{len(rows)} runs  |  "
        f"avg Δ(MBR-Greedy): {avg_delta_greedy:+.4f}  |  "
        f"avg Δ(MBR-Beam): {avg_delta_beam:+.4f}"
    )
    print(f"\nMBR parquets saved as oof_predictions_mbr50.parquet in each run directory.")


if __name__ == "__main__":
    main()
