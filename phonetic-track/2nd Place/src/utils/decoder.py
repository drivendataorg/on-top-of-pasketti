
from src.preprocessing.phoneme_tokenizer import PhonemeTokenizer
from typing import List
import logging
import torch
import numpy as np
import re
import multiprocessing as mp

import torch
import re
from typing import List
import os, sys
from pathlib import Path
class BeamSearchDecoder:
    def __init__(
        self,
        tokenizer: "PhonemeTokenizer",
        beam_width: int = 50,
        temperature: float = 1.0,
        blank_penalty: float = 0.0,
        alpha: float = 0.0,
        beta: float = 0.0,
        repeat_penalty: float = 0.0,
        decode_workers: int | None = None,
        use_bigram_score: bool = True,
        phoneme_corpus_path: str | None = None,
    ):
        """
        Fast C++ CTC Beam Search using pyctcdecode.

        Supports ultra_optimized_search-style decode tuning params:
        temperature, blank_penalty, alpha, beta, repeat_penalty.
        """
        from pyctcdecode import build_ctcdecoder
        
        self.beam_width = beam_width
        self.tokenizer = tokenizer
        self.temperature = float(temperature)
        self.blank_penalty = float(blank_penalty)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.repeat_penalty = float(repeat_penalty)
        self.decode_workers = (
            max(1, int(decode_workers))
            if decode_workers is not None
            else max(1, mp.cpu_count() - 2)
        )
        self.blank_token_id = tokenizer.blank_token_id
        self._repeat_pattern = re.compile(r"(.)\1\1")
        self._pool = None
        self._pool_size = 0

        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        
        # pyctcdecode strictly expects the CTC blank token to be an empty string "".
        # All other tokens MUST be unique strings.
        labels = []
        for i in range(tokenizer.vocab_size):
            if i == tokenizer.blank_token_id:
                labels.append("")
            else:
                # Decode the single token ID
                char = tokenizer.decode([i])
                
                # If the tokenizer returns empty string (e.g., for padding) 
                # or if the character is somehow already in our list, make it strictly unique.
                if not char or char in labels:
                    char = f"<special_{i}>"
                    
                labels.append(char)
                
        # Build the C++ decoder
        self.decoder = build_ctcdecoder(labels=labels)

        self._log_matrix = None
        if use_bigram_score:
            try:
                counts = self._build_phoneme_bigram_counts(
                    corpus_path=Path(phoneme_corpus_path) if phoneme_corpus_path else None
                )
                self._log_matrix = self._get_bigram_log_matrix(counts=counts)
            except FileNotFoundError as exc:
                logging.warning(
                    "BeamSearchDecoder bigram scoring disabled: %s",
                    exc,
                )

    def _get_pool(self, workers: int):
        workers = max(1, int(workers))
        if self._pool is not None and self._pool_size != workers:
            self._pool.close()
            self._pool.join()
            self._pool = None
            self._pool_size = 0
        if self._pool is None:
            self._pool = mp.Pool(processes=workers)
            self._pool_size = workers
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
            self._pool_size = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _build_phoneme_bigram_counts(
        self,
        corpus_path: Path | None,
    ) -> dict[tuple[int, int], int]:
        if corpus_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            corpus_path = project_root / "phoneme_corpus.txt"
        if not corpus_path.exists():
            raise FileNotFoundError(f"phoneme corpus not found at {corpus_path}")

        counts: dict[tuple[int, int], int] = {}
        with open(corpus_path, "r", encoding="utf-8") as handle:
            for line in handle:
                ids = self.tokenizer(line.strip())
                prev = -1
                for curr in ids:
                    key = (prev, curr)
                    counts[key] = counts.get(key, 0) + 1
                    prev = curr
                final_key = (prev, -2)
                counts[final_key] = counts.get(final_key, 0) + 1
        return counts

    def _bigram_idx(self, token_id: int) -> int:
        vocab_size = self.tokenizer.vocab_size
        if token_id == -1:
            return vocab_size
        if token_id == -2:
            return vocab_size + 1
        return token_id

    def _get_bigram_log_matrix(self, counts: dict[tuple[int, int], int]) -> np.ndarray:
        vocab_size = self.tokenizer.vocab_size
        matrix = np.full((vocab_size + 2, vocab_size + 2), 1e-10, dtype=np.float64)

        for (prev, curr), count in counts.items():
            matrix[self._bigram_idx(prev), self._bigram_idx(curr)] = count

        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        return np.log(matrix)

    def _score_ids_with_bigram(self, ids: list[int]) -> float:
        if self._log_matrix is None or not ids:
            return 0.0

        score = 0.0
        prev = -1
        for curr in ids:
            score += float(self._log_matrix[self._bigram_idx(prev), self._bigram_idx(curr)])
            prev = curr
        score += float(self._log_matrix[self._bigram_idx(prev), self._bigram_idx(-2)])
        return score

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"<special_\d+>", "", text).strip()
        return re.sub(r"\s+", " ", text)

    def _repeat_flag(self, text: str) -> int:
        return int(bool(self._repeat_pattern.search(text.replace(" ", ""))))

    def _prepare_log_probs(self, logits_frame: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(logits_frame, np.ndarray):
            logits_frame = torch.from_numpy(logits_frame)
        logits_frame = logits_frame.detach().cpu().float()

        if self.temperature != 1.0:
            logits_frame = logits_frame / self.temperature
        if self.blank_penalty != 0.0:
            logits_frame[:, self.blank_token_id] -= self.blank_penalty

        return torch.nn.functional.log_softmax(logits_frame, dim=-1).numpy()

    def _best_text_from_beams(self, beams) -> str:
        best_text = ""
        best_score = -float("inf")

        for beam in beams:
            if not isinstance(beam, (tuple, list)):
                continue

            # decode_beams_batch returns: (text, timed_tokens, logit_score, ...)
            # decode_beams returns: (text, lm_state, timed_tokens, logit_score, ...)
            text = beam[0] if len(beam) >= 1 and isinstance(beam[0], str) else ""
            timed_tokens = []
            logit_score = -float("inf")

            if len(beam) >= 3 and isinstance(beam[1], list):
                timed_tokens = beam[1]
                logit_score = float(beam[2])
            elif len(beam) >= 4 and isinstance(beam[2], list):
                timed_tokens = beam[2]
                logit_score = float(beam[3])
            else:
                continue

            ids: list[int] = []
            for token_text, _ in timed_tokens:
                ids.extend(self.tokenizer(token_text))

            score = (
                logit_score
                + (self.alpha * self._score_ids_with_bigram(ids))
                + (self.beta * len(ids))
                + (self.repeat_penalty * self._repeat_flag(text))
            )
            if score > best_score:
                best_score = score
                best_text = text

        return best_text

    def _iter_log_probs(self, logits):
        if isinstance(logits, (list, tuple)):
            for item in logits:
                yield self._prepare_log_probs(item)
        else:
            log_probs = self._prepare_log_probs(logits)
            for b in range(log_probs.shape[0]):
                yield log_probs[b]

    def __call__(self, logits) -> List[str]:
        """
        Args:
            logits: [B, T, vocab_size] tensor or list of [T, vocab_size] tensors.
        """
        log_probs_batch = list(self._iter_log_probs(logits))
        if not log_probs_batch:
            return []

        if len(log_probs_batch) == 1:
            beams_list = [
                self.decoder.decode_beams(
                    log_probs_batch[0],
                    beam_width=self.beam_width,
                )
            ]
        else:
            workers = min(self.decode_workers, len(log_probs_batch))
            pool = self._get_pool(workers)
            beams_list = self.decoder.decode_beams_batch(
                pool,
                log_probs_batch,
                beam_width=self.beam_width,
            )

        decoded = []
        for beams in beams_list:
            text = self._best_text_from_beams(beams) if beams else ""
            decoded.append(self._clean_text(text))
        return decoded
    
class PyCTCDecoder:
    def __init__(
        self,
        tokenizer: "PhonemeTokenizer",
        beam_width: int = 15,
        kenlm_model_path: str = None,
        alpha: float = 0.3,
        beta: float = 1.0,
    ):
        """
        Fast C++ CTC Beam Search using torchaudio (Flashlight).
        
        Natively supports lexicon-free decoding, meaning it scores individual 
        phonemes directly against the KenLM without forcing them into "words".
        """
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        
        try:
            from torchaudio.models.decoder import ctc_decoder
        except ImportError:
            raise ImportError("Please install torchaudio and flashlight-text: uv pip install torchaudio flashlight-text")

        # 1. Prepare tokens list (indices MUST exactly match the model's logits)
        tokens = []
        for i in range(tokenizer.vocab_size):
            if i == tokenizer.blank_token_id:
                # Torchaudio just needs a unique string for the blank token
                tokens.append("<blank>")
            else:
                char = tokenizer.decode([i])
                # Ensure no empty strings sneak in (torchaudio doesn't like them)
                if not char or char == " ": 
                    char = f"<special_{i}>"
                tokens.append(char)

        # 2. Build the C++ Flashlight decoder
        if kenlm_model_path:
            # Note: Torchaudio heavily prefers .binary KenLM models over .arpa
            kenlm_model_path = str(Path(kenlm_model_path).resolve())
            print(f"[PyCTCDecoder] Loading Lexicon-Free KenLM from {kenlm_model_path} (alpha={alpha}, beta={beta})")
            
            self.decoder = ctc_decoder(
                lexicon=None,  # Crucial: Tells the decoder to score individual tokens (phonemes) via LM
                tokens=tokens,
                lm=kenlm_model_path,
                blank_token="<blank>",
                sil_token="<blank>",  # Fixes the ValueError for the missing '|' token
                beam_size=beam_width,
                lm_weight=alpha,
                word_score=beta, # In lexicon-free mode, beta acts as a token insertion bonus/penalty
            )
        else:
            print(f"[PyCTCDecoder] Building pure Beam Search (no LM)")
            self.decoder = ctc_decoder(
                lexicon=None,
                tokens=tokens,
                blank_token="<blank>",
                sil_token="<blank>",
                beam_size=beam_width,
            )

    def _iter_log_probs(self, logits):
        """Yields single-sequence log_softmax tensors to avoid padding hallucinations."""
        if isinstance(logits, (list, tuple)):
            for item in logits:
                if isinstance(item, np.ndarray):
                    item = torch.from_numpy(item)
                # torchaudio C++ backend strictly requires float32 on CPU
                yield torch.nn.functional.log_softmax(item.detach().cpu().float(), dim=-1)
        else:
            if isinstance(logits, np.ndarray):
                logits = torch.from_numpy(logits)
            log_probs = torch.nn.functional.log_softmax(logits.detach().cpu().float(), dim=-1)
            for b in range(log_probs.shape[0]):
                yield log_probs[b]

    def _squash_repetitions(self, text: str, min_repeat: int = 4) -> str:
        """Post-processing to squash EXTREME repeating patterns only."""
        squashed = re.sub(r'(.+?)\1{' + str(min_repeat - 1) + r',}', r'\1', text)
        return squashed

    def __call__(self, logits) -> List[str]:
        """
        Args:
            logits: [B, T, vocab_size] tensor or list of [T, vocab_size] tensors.
        """
        decoded = []
        
        # We iterate over single sequences so Flashlight doesn't try to decode zero-padding
        for emissions in self._iter_log_probs(logits):
            # Flashlight expects shape [B, T, C], so unsqueeze to batch size 1
            emissions_batched = emissions.unsqueeze(0)
            
            # Decode the single sequence
            results = self.decoder(emissions_batched)
            
            # Grab the best hypothesis (beam 0) for this sequence
            best_hyp = results[0][0]
            
            # Flashlight neatly provides the collapsed token IDs (blanks and repeats removed)
            token_ids = best_hyp.tokens.tolist()
            
            # Decode using your exact tokenizer logic so spacing and unknown tokens are perfect
            text = self.tokenizer.decode(token_ids)
            
            # Clean up and squash extreme repetitions
            text = re.sub(r'\s+', ' ', text).strip()
            text = self._squash_repetitions(text)
            
            decoded.append(text)

        return decoded

class MBRDecoder:
    """
    Minimum Bayes Risk (MBR) decoding over CTC beam hypotheses.

    Generates beam_width hypotheses via pyctcdecode CTC beam search, then
    selects the top mbr_n_best by CTC logit score and returns the hypothesis
    that minimises the expected character-level edit distance under the
    approximate posterior:

        P(h | x) ∝ softmax(logit_scores / prob_temperature)

    The edit distance is computed on normalize_ipa-normalised strings so the
    loss matches the IPA-CER evaluation metric.
    """

    def __init__(
        self,
        tokenizer: "PhonemeTokenizer",
        beam_width: int = 50,
        temperature: float = 1.0,
        blank_penalty: float = 0.0,
        mbr_n_best: int = 50,
        prob_temperature: float = 1.0,
        decode_workers: int | None = None,
    ):
        from pyctcdecode import build_ctcdecoder

        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.beam_width = beam_width
        self.tokenizer = tokenizer
        self.temperature = float(temperature)
        self.blank_penalty = float(blank_penalty)
        self.mbr_n_best = mbr_n_best
        self.prob_temperature = float(prob_temperature)
        self.blank_token_id = tokenizer.blank_token_id
        self.decode_workers = (
            max(1, int(decode_workers))
            if decode_workers is not None
            else max(1, mp.cpu_count() - 2)
        )
        self._pool = None
        self._pool_size = 0

        labels = []
        for i in range(tokenizer.vocab_size):
            if i == tokenizer.blank_token_id:
                labels.append("")
            else:
                char = tokenizer.decode([i])
                if not char or char in labels:
                    char = f"<special_{i}>"
                labels.append(char)

        self.decoder = build_ctcdecoder(labels=labels)

    def _get_pool(self, workers: int):
        workers = max(1, int(workers))
        if self._pool is not None and self._pool_size != workers:
            self._pool.close()
            self._pool.join()
            self._pool = None
            self._pool_size = 0
        if self._pool is None:
            self._pool = mp.Pool(processes=workers)
            self._pool_size = workers
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
            self._pool_size = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _prepare_log_probs(self, logits_frame: "torch.Tensor | np.ndarray") -> np.ndarray:
        if isinstance(logits_frame, np.ndarray):
            logits_frame = torch.from_numpy(logits_frame)
        logits_frame = logits_frame.detach().cpu().float()
        if self.temperature != 1.0:
            logits_frame = logits_frame / self.temperature
        if self.blank_penalty != 0.0:
            logits_frame[:, self.blank_token_id] -= self.blank_penalty
        return torch.nn.functional.log_softmax(logits_frame, dim=-1).numpy()

    def _iter_log_probs(self, logits):
        if isinstance(logits, (list, tuple)):
            for item in logits:
                yield self._prepare_log_probs(item)
        else:
            log_probs = self._prepare_log_probs(logits)
            for b in range(log_probs.shape[0]):
                yield log_probs[b]

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"<special_\d+>", "", text).strip()
        return re.sub(r"\s+", " ", text)

    def _extract_hyps(self, beams) -> list[tuple[str, float]]:
        """Return (cleaned_text, logit_score) pairs from beam output."""
        hyps = []
        for beam in beams:
            if not isinstance(beam, (tuple, list)):
                continue
            text = beam[0] if isinstance(beam[0], str) else ""
            logit_score = -float("inf")
            # decode_beams_batch: (text, timed_tokens, logit_score, ...)
            # decode_beams:       (text, lm_state, timed_tokens, logit_score, ...)
            if len(beam) >= 3 and isinstance(beam[1], list):
                logit_score = float(beam[2])
            elif len(beam) >= 4 and isinstance(beam[2], list):
                logit_score = float(beam[3])
            else:
                continue
            hyps.append((self._clean_text(text), logit_score))
        return hyps

    def _mbr_select(self, hyps: list[tuple[str, float]]) -> str:
        """Select the hypothesis with minimum expected edit distance."""
        import editdistance
        from src.utils.score import normalize_ipa

        if not hyps:
            return ""
        if len(hyps) == 1:
            return hyps[0][0]

        texts_raw = [h[0] for h in hyps]
        texts_norm = [normalize_ipa(t) for t in texts_raw]
        scores = np.array([h[1] for h in hyps], dtype=np.float64)

        # Approximate posterior via softmax with prob_temperature
        scores = scores / self.prob_temperature
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()

        # Precompute all pairwise edit distances (symmetric)
        N = len(texts_norm)
        dist_matrix = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                d = float(editdistance.eval(texts_norm[i], texts_norm[j]))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Expected edit distance for each hypothesis
        risks = dist_matrix @ probs

        return texts_raw[int(np.argmin(risks))]

    def __call__(self, logits) -> List[str]:
        """
        Args:
            logits: [B, T, vocab_size] tensor or list of [T, vocab_size] tensors.
        """
        log_probs_batch = list(self._iter_log_probs(logits))
        if not log_probs_batch:
            return []

        if len(log_probs_batch) == 1:
            beams_list = [
                self.decoder.decode_beams(log_probs_batch[0], beam_width=self.beam_width)
            ]
        else:
            workers = min(self.decode_workers, len(log_probs_batch))
            pool = self._get_pool(workers)
            beams_list = self.decoder.decode_beams_batch(
                pool, log_probs_batch, beam_width=self.beam_width
            )

        decoded = []
        for beams in beams_list:
            hyps = self._extract_hyps(beams)
            if self.mbr_n_best < len(hyps):
                hyps = hyps[: self.mbr_n_best]
            decoded.append(self._mbr_select(hyps))
        return decoded


class GreedyDecoder:
    def __init__(self, tokenizer: "PhonemeTokenizer"):
        self.tokenizer = tokenizer
        self.blank_token_id = tokenizer.blank_token_id

    def __call__(self, logits) -> List[str]:
        """
        Greedy CTC decoding: argmax → collapse repeats → remove blank.

        Since PAD and UNK are no longer in the output vocab, the only
        special token to filter is BLANK.

        Args:
            logits: [B, T, vocab_size] tensor, or list of [T, vocab_size] tensors.

        Returns:
            List of decoded phonetic strings, one per batch element.
        """
        blank_id = self.blank_token_id

        if isinstance(logits, list):
            sequences = [t.argmax(dim=-1).detach().cpu() for t in logits]  # list of [T]
        else:
            sequences = list(logits.argmax(dim=-1).detach().cpu())  # list of [T]

        decoded = []
        for seq in sequences:
            #remove blanks:
            # 1. Remove blanks
            not_blank = seq != blank_id
            tokens = seq[not_blank]
            # 2. Remove duplicates
            diff = torch.ones(len(tokens), dtype=torch.bool)
            diff[1:] = tokens[1:] != tokens[:-1]
            output = tokens[diff].tolist()
            #recode to strings
            decoded.append(self.tokenizer.decode(output))

        return decoded


if __name__ == "__main__":
    

    
    import os, sys, torch
    from pathlib import Path
    from tqdm import tqdm
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    import polars as pl

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)

    from src.preprocessing.dataset import prepare_dl_dataset
    from src.utils.score import score_ipa_cer
    



    RUN_DIR = PROJECT_ROOT / "outputs/2026-03-16/17-21-53_screaming-conquest-127"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    base_cfg = OmegaConf.load(PROJECT_ROOT / "configs/default.yaml")
    run_cfg = OmegaConf.load(RUN_DIR / ".hydra/config.yaml")
    cfg = OmegaConf.merge(base_cfg, run_cfg)
    # --- QUICK HARDCODE FIX ---
    if "tokenizer" not in cfg:
        print("[Warning] 'tokenizer' missing from config! Hardcoding fallback...")
        from omegaconf import OmegaConf
        cfg.tokenizer = OmegaConf.create({
            "_target_": "src.preprocessing.phoneme_tokenizer.PhonemeTokenizer",
            # Add any required arguments your PhonemeTokenizer needs here!
            # For example: "vocab_path": str(PROJECT_ROOT / "vocab.json")
        })
    _, val_loader, info = prepare_dl_dataset(cfg, fold=0)
    tok, vs = info["tokenizer"], info["vocab_size"]

    LM_ARPA_PATH = PROJECT_ROOT / "phoneme_lm.arpa"
    LM_BIN_PATH = PROJECT_ROOT / "phoneme_lm.bin"
    LM_PATH = str(LM_BIN_PATH if LM_BIN_PATH.exists() else LM_ARPA_PATH)
    BEAM_WIDTH = 15

    # KenLM sweep settings (tweak as needed)
    RUN_KENLM_GRID = True
    KENLM_BEAMS = [5, 10, 15]
    KENLM_ALPHAS = [0.1, 0.3, 0.5]
    KENLM_BETAS = [0.0, 0.5, 1.0]
    SAVE_BEST_KENLM_PREDICTIONS = True
    GRID_RESULTS_PATH = RUN_DIR / "kenlm_grid_search.parquet"
    # Should squash to "mama"
    # Load model once with a dummy decoder
    # Training-only keys can leak into the model config and break instantiate.
    if "lora_lr" in cfg.model:
        print("[Decoder] Dropping unsupported model arg: lora_lr")
        cfg.model.pop("lora_lr")
    model = instantiate(cfg.model, vocab_size=vs, vocab=tok, decoder=PyCTCDecoder(tokenizer=tok, beam_width=1))
    ckpt = torch.load(RUN_DIR / "fold_1/best_model.pth", map_location=device)
    load_result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if load_result.missing_keys:
        print(f"[Checkpoint Load] Missing keys: {len(load_result.missing_keys)}")
    if load_result.unexpected_keys:
        print(f"[Checkpoint Load] Unexpected keys: {len(load_result.unexpected_keys)}")
    model.to(device).eval()

    def _collect_predictions(decoder_obj, desc: str):
        model.decoder = decoder_obj
        preds, refs = [], []
        utterance_ids, child_ids, age_buckets = [], [], []

        with torch.no_grad():
            for b in tqdm(val_loader, desc=desc):
                mask = b.get("attention_mask")
                output = model(b["input_features"].to(device), attention_mask=mask.to(device) if mask is not None else None)
                if isinstance(output, (list, tuple)):
                    candidate = output[1] if len(output) > 1 else output[0]
                else:
                    candidate = output
                # If we got raw logits, decode them explicitly
                if isinstance(candidate, torch.Tensor):
                    decoded = decoder_obj(candidate)
                else:
                    decoded = candidate
                preds.extend(decoded)
                refs.extend([tok.decode(lbl[lbl != tok.pad_token_id].tolist()) for lbl in b["labels"]])
                utterance_ids.extend(b.get("utterance_ids", []))
                child_ids.extend(b.get("child_ids", []))
                batch_age_buckets = b.get("age_buckets")
                if batch_age_buckets is not None:
                    age_buckets.extend(batch_age_buckets)

        return {
            "preds": preds,
            "refs": refs,
            "utterance_ids": utterance_ids,
            "child_ids": child_ids,
            "age_buckets": age_buckets,
        }

    def _save_predictions(path: Path, results: dict, decoder_name: str | None = None):
        preds = results["preds"]
        refs = results["refs"]
        utterance_ids = results["utterance_ids"]
        child_ids = results["child_ids"]
        age_buckets = results["age_buckets"]

        if len(preds) != len(refs):
            raise ValueError(f"Pred/Ref length mismatch: {len(preds)} != {len(refs)}")
        if utterance_ids and len(utterance_ids) != len(preds):
            raise ValueError(f"Utterance ID length mismatch: {len(utterance_ids)} != {len(preds)}")
        if child_ids and len(child_ids) != len(preds):
            raise ValueError(f"Child ID length mismatch: {len(child_ids)} != {len(preds)}")
        if age_buckets and len(age_buckets) != len(preds):
            raise ValueError(f"Age bucket length mismatch: {len(age_buckets)} != {len(preds)}")

        data = {
            "utterance_id": utterance_ids if utterance_ids else [None] * len(preds),
            "child_id": child_ids if child_ids else [None] * len(preds),
            "ground_truth": refs,
            "prediction": preds,
        }
        if age_buckets:
            data["age_bucket"] = age_buckets
        if decoder_name is not None:
            data["decoder"] = [decoder_name] * len(preds)

        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(data).write_parquet(path)
        print(f"[Decoder] Saved predictions to: {path}")

    def evaluate_decoder(decoder_obj, desc: str, collect: bool = False):
        results = _collect_predictions(decoder_obj, desc)
        cer = score_ipa_cer(results["refs"], results["preds"])
        if collect:
            return cer, results
        return cer

    greedy_cer = evaluate_decoder(GreedyDecoder(tok), "Greedy")
    beam_decoder = BeamSearchDecoder(tokenizer=tok, beam_width=BEAM_WIDTH)
    beam_cer, beam_results = evaluate_decoder(
        beam_decoder,
        f"BeamSearch BW={BEAM_WIDTH}",
        collect=True,
    )
    kenlm_cer = evaluate_decoder(
        PyCTCDecoder(tokenizer=tok, beam_width=BEAM_WIDTH, kenlm_model_path=LM_PATH, alpha=0.3, beta=1.0),
        f"KenLM BW={BEAM_WIDTH} a=0.3 b=1.0",
    )

    # Save beam-search predictions for offline analysis (e.g., oof_predictions_eda.ipynb)
    SAVE_BEAM_PREDICTIONS = True
    if SAVE_BEAM_PREDICTIONS:
        beam_pred_path = RUN_DIR / f"oof_predictions_beam_bw{BEAM_WIDTH}.parquet"
        _save_predictions(
            beam_pred_path,
            beam_results,
            decoder_name=f"beam_bw{BEAM_WIDTH}",
        )

    # Optional KenLM grid search to find a better beam configuration
    best_kenlm = None
    if RUN_KENLM_GRID:
        grid_rows = []
        for bw in KENLM_BEAMS:
            for alpha in KENLM_ALPHAS:
                for beta in KENLM_BETAS:
                    desc = f"KenLM BW={bw} a={alpha} b={beta}"
                    decoder = PyCTCDecoder(
                        tokenizer=tok,
                        beam_width=bw,
                        kenlm_model_path=LM_PATH,
                        alpha=alpha,
                        beta=beta,
                    )
                    cer = evaluate_decoder(decoder, desc)
                    row = {"beam_width": bw, "alpha": alpha, "beta": beta, "cer": cer}
                    grid_rows.append(row)
                    if best_kenlm is None or cer < best_kenlm["cer"]:
                        best_kenlm = row

        pl.DataFrame(grid_rows).write_parquet(GRID_RESULTS_PATH)
        print(f"[KenLM Grid] Saved results to: {GRID_RESULTS_PATH}")

        if best_kenlm is not None:
            print(
                "[KenLM Grid] Best config: "
                f"BW={best_kenlm['beam_width']} a={best_kenlm['alpha']} "
                f"b={best_kenlm['beta']} | CER={best_kenlm['cer']:.4f}"
            )

        if SAVE_BEST_KENLM_PREDICTIONS and best_kenlm is not None:
            best_decoder = PyCTCDecoder(
                tokenizer=tok,
                beam_width=best_kenlm["beam_width"],
                kenlm_model_path=LM_PATH,
                alpha=best_kenlm["alpha"],
                beta=best_kenlm["beta"],
            )
            best_cer, best_results = evaluate_decoder(
                best_decoder,
                f"KenLM Best BW={best_kenlm['beam_width']} a={best_kenlm['alpha']} b={best_kenlm['beta']}",
                collect=True,
            )
            a_tag = str(best_kenlm["alpha"]).replace(".", "p")
            b_tag = str(best_kenlm["beta"]).replace(".", "p")
            best_pred_path = RUN_DIR / f"oof_predictions_kenlm_bw{best_kenlm['beam_width']}_a{a_tag}_b{b_tag}.parquet"
            _save_predictions(
                best_pred_path,
                best_results,
                decoder_name=f"kenlm_bw{best_kenlm['beam_width']}_a{best_kenlm['alpha']}_b{best_kenlm['beta']}",
            )

    print(
        f"Greedy CER: {greedy_cer:.4f} | "
        f"BeamSearch CER: {beam_cer:.4f} | "
        f"KenLM CER: {kenlm_cer:.4f}"
    )
    print(
        f"Δ(Beam-Greedy): {beam_cer - greedy_cer:+.4f} | "
        f"Δ(KenLM-Greedy): {kenlm_cer - greedy_cer:+.4f} | "
        f"Δ(KenLM-Beam): {kenlm_cer - beam_cer:+.4f}"
    )
