from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.score import normalize_ipa, score_ipa_cer

_OUTPUT_PREFIX_RE = re.compile(
    r"^(?:answer|corrected|corrected text|final|final sequence|fused phoneme sequence|output)\s*:\s*",
    flags=re.IGNORECASE,
)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_OPTION_CHOICE_RE = re.compile(r"^(?:option\s*)?([1-9][0-9]*)$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class PredictionBundle:
    label: str
    run_dir: Path
    source: str
    df: pl.DataFrame


@dataclass(frozen=True)
class AlignmentStats:
    shared_utterance_ids: list[str]
    bundle_sizes: dict[str, int]
    largest_bundle_label: str
    largest_bundle_size: int
    shared_size: int

    @property
    def coverage_vs_largest(self) -> float:
        if self.largest_bundle_size == 0:
            return 0.0
        return self.shared_size / self.largest_bundle_size

    @property
    def mismatch_detected(self) -> bool:
        return any(size != self.shared_size for size in self.bundle_sizes.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse phoneme hypotheses from multiple saved ASR runs with an instruction-tuned LLM "
            "served through vLLM, then score the fused predictions on the validation fold."
        )
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Two or more run output directories (or fold_* dirs) to ensemble.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional short labels for the runs. Must match the number of run dirs.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="1-based fold number to evaluate. Defaults to 1.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device used if predictions must be regenerated through src/eval.py.",
    )
    parser.add_argument(
        "--reference-run-index",
        type=int,
        default=1,
        help=(
            "1-based run index whose validation split should be used as the shared reference "
            "when predictions must be regenerated through src/eval.py. Defaults to 1."
        ),
    )
    parser.add_argument(
        "--prediction-source",
        choices=("auto", "oof", "eval"),
        default="auto",
        help=(
            "Where to load base hypotheses from. "
            "'oof' uses saved oof_predictions_best.parquet, "
            "'eval' reruns src/eval.py's inference path, "
            "'auto' prefers OOF and falls back to eval."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="vLLM model name or local path.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor_parallel_size. Defaults to 1.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="vLLM dtype, e.g. auto, float16, or bfloat16.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization fraction. Defaults to 0.9.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to vLLM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature. Defaults to 0.0 for deterministic fusion.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="LLM top-p. Defaults to 1.0.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="LLM repetition penalty. Defaults to 1.0.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum LLM completion length. Defaults to 128.",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=512,
        help="How many prompts to send to vLLM per generate() call.",
    )
    parser.add_argument(
        "--disable-consensus-shortcut",
        action="store_true",
        help="Always call the LLM, even when all non-empty hypotheses already agree.",
    )
    parser.add_argument(
        "--llm-min-unique-hypotheses",
        type=int,
        default=2,
        help=(
            "Only call the LLM when at least this many unique normalized hypotheses remain. "
            "Defaults to 2, meaning any disagreement triggers the LLM. Set to 3 to keep "
            "majority vote for 2-vs-1 rows and reserve the LLM for full 3-way disagreements."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick debugging runs on a subset of the validation fold.",
    )
    parser.add_argument(
        "--output-parquet",
        default=None,
        help="Optional parquet path for utterance-level fused predictions and metadata.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    slug = _SLUG_RE.sub("_", text.lower()).strip("_")
    return slug or "run"


def resolve_existing_path(path_str: str) -> Path:
    raw_path = Path(path_str).expanduser()
    for candidate in (raw_path, PROJECT_ROOT / raw_path):
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / raw_path).resolve()


def resolve_output_path(path_str: str | Path) -> Path:
    raw_path = Path(path_str).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (PROJECT_ROOT / raw_path).resolve()


def resolve_run_and_fold_dirs(output_path: Path, fold_number: int) -> tuple[Path, Path]:
    hydra_config = output_path / ".hydra" / "config.yaml"
    if hydra_config.exists():
        return output_path, output_path / f"fold_{fold_number}"

    if output_path.name.startswith("fold_") and (output_path / "best_model.pth").exists():
        run_dir = output_path.parent
        if (run_dir / ".hydra" / "config.yaml").exists():
            return run_dir, output_path

    raise FileNotFoundError(
        "Could not find a valid run directory. Expected either "
        "`<run_dir>/.hydra/config.yaml` or a `fold_*` directory with `best_model.pth`."
    )


def import_eval_helpers() -> tuple[Any, Any]:
    from src.eval import evaluate_run, prepare_eval_context

    return evaluate_run, prepare_eval_context


def build_default_labels(num_runs: int) -> list[str]:
    return [f"model_{idx}" for idx in range(1, num_runs + 1)]


def resolve_labels(args: argparse.Namespace) -> list[str]:
    labels = args.labels or build_default_labels(len(args.run_dirs))
    if len(labels) != len(args.run_dirs):
        raise ValueError("--labels must have the same length as the run_dirs list.")
    if len(set(labels)) != len(labels):
        raise ValueError("--labels must be unique.")
    return labels


def resolve_default_output_path(labels: list[str], fold: int) -> Path:
    label_slug = "_".join(slugify(label) for label in labels[:4])
    if len(labels) > 4:
        label_slug = f"{label_slug}_{len(labels)}runs"
    return PROJECT_ROOT / "outputs" / "llm_fusion" / f"llm_fusion_fold_{fold}_{label_slug}.parquet"


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def validate_prediction_table(df: pl.DataFrame, label: str, source: str) -> pl.DataFrame:
    required_columns = {"utterance_id", "prediction"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"{label} ({source}) is missing required columns: {sorted(missing_columns)}"
        )
    if df.height == 0:
        raise ValueError(f"{label} ({source}) produced an empty prediction table.")

    utterance_ids = df["utterance_id"]
    if utterance_ids.n_unique() != df.height:
        raise ValueError(f"{label} ({source}) contains duplicate utterance_id rows.")
    return df.sort("utterance_id")


def load_oof_predictions(run_dir: Path, fold: int) -> pl.DataFrame:
    oof_path = run_dir / "oof_predictions_best.parquet"
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing OOF predictions: {oof_path}")

    df = pl.read_parquet(oof_path)
    if "fold" in df.columns:
        df = df.filter(pl.col("fold") == (fold - 1))
    if df.height == 0:
        raise ValueError(f"No OOF rows found in {oof_path} for fold {fold}.")

    keep_columns = ["utterance_id", "prediction"]
    if "child_id" in df.columns:
        keep_columns.append("child_id")
    if "ground_truth" in df.columns:
        keep_columns.append("ground_truth")
    return df.select(keep_columns)


def load_oof_bundle(
    run_path_str: str,
    label: str,
    fold: int,
) -> PredictionBundle:
    requested_path = resolve_existing_path(run_path_str)
    run_dir, _ = resolve_run_and_fold_dirs(requested_path, fold)
    df = validate_prediction_table(load_oof_predictions(run_dir, fold), label=label, source="oof")
    return PredictionBundle(label=label, run_dir=run_dir, source="oof", df=df)


def load_shared_eval_bundles(
    run_paths: list[str],
    labels: list[str],
    fold: int,
    device: str | None,
    reference_run_index: int,
) -> list[PredictionBundle]:
    evaluate_run, prepare_eval_context = import_eval_helpers()
    if not (1 <= reference_run_index <= len(run_paths)):
        raise ValueError(
            f"--reference-run-index must be between 1 and {len(run_paths)}."
        )

    reference_context = prepare_eval_context(
        output_dir=run_paths[reference_run_index - 1],
        fold=fold,
        build_dataset=False,
    )
    shared_val_data = reference_context.val_data

    bundles: list[PredictionBundle] = []
    for run_path_str, label in zip(run_paths, labels):
        requested_path = resolve_existing_path(run_path_str)
        run_dir, _ = resolve_run_and_fold_dirs(requested_path, fold)
        eval_result = evaluate_run(
            output_dir=run_dir,
            fold=fold,
            device=device,
            progress_desc=f"Evaluating {label}",
            val_data_override=shared_val_data,
        )
        df = validate_prediction_table(
            eval_result.to_prediction_frame().select(["utterance_id", "child_id", "ground_truth", "prediction"]),
            label=label,
            source="eval",
        )
        bundles.append(PredictionBundle(label=label, run_dir=run_dir, source="eval", df=df))
    return bundles


def utterance_ids_for_bundle(bundle: PredictionBundle) -> set[str]:
    return set(bundle.df["utterance_id"].to_list())


def compute_alignment_stats(bundles: list[PredictionBundle]) -> AlignmentStats:
    if len(bundles) < 2:
        raise ValueError("LLM fusion requires at least two runs.")

    bundle_sizes = {bundle.label: bundle.df.height for bundle in bundles}
    largest_bundle_label, largest_bundle_size = max(bundle_sizes.items(), key=lambda item: item[1])

    shared_utterance_id_set = set.intersection(*(utterance_ids_for_bundle(bundle) for bundle in bundles))
    shared_utterance_ids = sorted(shared_utterance_id_set)
    if not shared_utterance_ids:
        raise ValueError("No shared utterance_id values were found across the selected prediction tables.")

    return AlignmentStats(
        shared_utterance_ids=shared_utterance_ids,
        bundle_sizes=bundle_sizes,
        largest_bundle_label=largest_bundle_label,
        largest_bundle_size=largest_bundle_size,
        shared_size=len(shared_utterance_ids),
    )


def merge_text_column(df: pl.DataFrame, base_col: str, incoming_col: str) -> pl.DataFrame:
    if incoming_col not in df.columns:
        return df
    if base_col not in df.columns:
        return df.rename({incoming_col: base_col})

    base_has_values = bool(df.select(pl.col(base_col).is_not_null().any()).item())
    if not base_has_values:
        return df.with_columns(pl.col(incoming_col).alias(base_col)).drop(incoming_col)

    mismatch_rows = df.filter(
        pl.col(base_col).is_not_null()
        & pl.col(incoming_col).is_not_null()
        & (pl.col(base_col) != pl.col(incoming_col))
    )
    if mismatch_rows.height:
        raise ValueError(f"Mismatched {base_col} values encountered while aligning prediction tables.")

    return df.with_columns(pl.coalesce([pl.col(base_col), pl.col(incoming_col)]).alias(base_col)).drop(incoming_col)


def join_prediction_bundles(
    bundles: list[PredictionBundle],
    alignment_stats: AlignmentStats,
) -> pl.DataFrame:
    first_bundle = bundles[0]
    shared_ids = alignment_stats.shared_utterance_ids

    base = (
        first_bundle.df
        .filter(pl.col("utterance_id").is_in(shared_ids))
        .rename({"prediction": f"{first_bundle.label}_prediction"})
    )
    if "child_id" not in base.columns:
        base = base.with_columns(pl.lit(None, dtype=pl.String).alias("child_id"))
    if "ground_truth" not in base.columns:
        base = base.with_columns(pl.lit(None, dtype=pl.String).alias("ground_truth"))
    base = base.select(["utterance_id", "child_id", "ground_truth", f"{first_bundle.label}_prediction"])

    for bundle in bundles[1:]:
        current = (
            bundle.df
            .filter(pl.col("utterance_id").is_in(shared_ids))
            .rename({"prediction": f"{bundle.label}_prediction"})
        )
        current = current.select([col for col in current.columns if col in {"utterance_id", "child_id", "ground_truth", f"{bundle.label}_prediction"}])
        joined = base.join(current, on="utterance_id", how="inner", suffix=f"_{bundle.label}")

        joined = merge_text_column(joined, "ground_truth", f"ground_truth_{bundle.label}")
        joined = merge_text_column(joined, "child_id", f"child_id_{bundle.label}")
        base = joined

    if base.height != alignment_stats.shared_size:
        raise ValueError(
            f"Expected {alignment_stats.shared_size} shared utterances after alignment, got {base.height}."
        )
    return base.sort("utterance_id")


def count_unique_hypotheses(predictions: list[str]) -> int:
    normalized = {normalize_ipa(text.strip()) for text in predictions if text and text.strip()}
    return max(1, len(normalized))


def choose_majority_prediction(predictions: list[str]) -> str:
    normalized_counts: dict[str, int] = {}
    normalized_first_text: dict[str, str] = {}
    normalized_first_idx: dict[str, int] = {}

    for idx, text in enumerate(predictions):
        cleaned = text.strip()
        key = normalize_ipa(cleaned)
        if not key:
            continue
        normalized_counts[key] = normalized_counts.get(key, 0) + 1
        if key not in normalized_first_text:
            normalized_first_text[key] = cleaned
            normalized_first_idx[key] = idx

    if not normalized_counts:
        return predictions[0].strip()

    best_key = max(
        normalized_counts,
        key=lambda key: (normalized_counts[key], -normalized_first_idx[key]),
    )
    return normalized_first_text[best_key]


def choose_consensus_prediction(predictions: list[str]) -> str | None:
    normalized_to_text: dict[str, str] = {}
    for text in predictions:
        cleaned = text.strip()
        key = normalize_ipa(cleaned)
        if not key:
            continue
        normalized_to_text.setdefault(key, cleaned)

    if not normalized_to_text:
        return predictions[0].strip()
    if len(normalized_to_text) == 1:
        return next(iter(normalized_to_text.values()))
    return None


def create_prompt(labels: list[str], predictions: list[str]) -> str:
    candidate_lines = []
    for idx, (label, prediction) in enumerate(zip(labels, predictions), start=1):
        candidate_lines.append(f"Option {idx} ({label}): {prediction.strip() or '<empty>'}")

    candidates = "\n".join(candidate_lines)
    return (
        "You are an expert phonetician working on verbatim transcription of young children's speech.\n"
        "Below are multiple ASR phoneme hypotheses for the same child utterance.\n"
        "Choose the single best candidate.\n"
        "Rules:\n"
        "- Preserve the child's actual pronunciation, including developmental or phonological errors.\n"
        "- Do not normalize toward standard or adult English.\n"
        "- You must select exactly one listed candidate.\n"
        "- Do not merge, rewrite, clean up, or invent a new phoneme sequence.\n"
        "- Return only the option number, for example `2`.\n"
        "- Do not explain your reasoning.\n"
        "- If one system is clearly noisy, ignore it.\n\n"
        f"{candidates}\n"
        "Selected option:"
    )


def sanitize_llm_output(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = cleaned.strip("`").strip()
    cleaned = cleaned.splitlines()[0].strip()
    cleaned = _OUTPUT_PREFIX_RE.sub("", cleaned).strip()

    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("'") and cleaned.endswith("'") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def select_llm_prediction(raw_text: str, predictions: list[str]) -> str | None:
    cleaned = sanitize_llm_output(raw_text)
    if not cleaned:
        return None

    stripped_predictions = [prediction.strip() for prediction in predictions]
    match = _OPTION_CHOICE_RE.match(cleaned)
    if match:
        option_idx = int(match.group(1)) - 1
        if 0 <= option_idx < len(stripped_predictions):
            return stripped_predictions[option_idx]

    for prediction in stripped_predictions:
        if cleaned == prediction:
            return prediction

    normalized_to_prediction: dict[str, str] = {}
    for prediction in stripped_predictions:
        normalized = normalize_ipa(prediction)
        if normalized and normalized not in normalized_to_prediction:
            normalized_to_prediction[normalized] = prediction

    return normalized_to_prediction.get(normalize_ipa(cleaned))


def prepare_vllm_import_environment() -> None:
    # In this environment, torchvision is installed but missing the `nms`
    # operator. Recent transformers builds may still try to import torchvision
    # through shared processing/image utilities when vLLM imports transformers.
    # Mark torchvision unavailable before importing vLLM so text-only LLM usage
    # does not fail on an unrelated vision dependency.
    from transformers.utils import import_utils as transformers_import_utils

    transformers_import_utils._torchvision_available = False


def build_vllm(args: argparse.Namespace) -> tuple[Any, Any]:
    prepare_vllm_import_environment()
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        torch_version = "unknown"
        vllm_version = "unknown"
        torch_cuda_version = "unknown"
        try:
            torch_version = importlib_metadata.version("torch")
        except importlib_metadata.PackageNotFoundError:
            pass
        try:
            vllm_version = importlib_metadata.version("vllm")
        except importlib_metadata.PackageNotFoundError:
            pass
        try:
            import torch

            torch_cuda_version = str(torch.version.cuda)
        except Exception:
            pass

        detail = str(exc)
        if "libcudart.so.12" in detail:
            raise ImportError(
                "vLLM is looking for the CUDA 12 runtime (`libcudart.so.12`), but the current "
                f"environment appears to be using torch={torch_version} with CUDA {torch_cuda_version}. "
                "Recreate the vLLM environment with a CUDA 12 torch backend such as `cu128` or "
                "`cu129` instead of `cu130`."
            ) from exc
        if "undefined symbol" in detail or "vllm/_C" in detail:
            raise ImportError(
                "vLLM failed to load its native extension, which usually means the installed "
                f"`vllm` wheel is ABI-incompatible with the current `torch` build "
                f"(torch={torch_version}, cuda={torch_cuda_version}, vllm={vllm_version}). "
                "Use a vLLM build that matches this PyTorch/CUDA stack, build vLLM from source "
                "against the current torch, or run `llm-fusion.py --prediction-source oof` from "
                "a separate clean vLLM environment."
            ) from exc
        raise ImportError(
            "Could not import vLLM after disabling torchvision-dependent transformers paths. "
            f"Installed versions: torch={torch_version}, cuda={torch_cuda_version}, vllm={vllm_version}. "
            "Check the local vLLM/transformers installation in this environment."
        ) from exc

    llm_kwargs: dict[str, Any] = {
        "model": args.llm_model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.dtype:
        llm_kwargs["dtype"] = args.dtype
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        stop=["\n"],
    )
    return llm, sampling_params


def run_llm_fusion(
    metadata_df: pl.DataFrame,
    labels: list[str],
    args: argparse.Namespace,
) -> tuple[list[str], list[str], list[str], list[int], int, int, int]:
    hypothesis_columns = [f"{label}_prediction" for label in labels]
    num_rows = metadata_df.height
    majority_predictions = [""] * num_rows
    fused_predictions = [""] * num_rows
    fusion_sources = [""] * num_rows
    unique_counts = [0] * num_rows
    llm_prompt_count = 0
    llm_empty_fallbacks = 0
    llm_invalid_choice_fallbacks = 0

    llm = None
    sampling_params = None
    pending_prompts: list[str] = []
    pending_indices: list[int] = []
    pending_hypotheses: list[list[str]] = []

    progress = tqdm(total=num_rows, desc="Fusing hypotheses")

    def flush_pending() -> None:
        nonlocal llm, sampling_params, llm_prompt_count, llm_empty_fallbacks
        nonlocal llm_invalid_choice_fallbacks
        nonlocal pending_prompts, pending_indices, pending_hypotheses

        if not pending_prompts:
            return

        if llm is None or sampling_params is None:
            llm, sampling_params = build_vllm(args)

        outputs = llm.generate(pending_prompts, sampling_params)
        llm_prompt_count += len(pending_prompts)

        for row_idx, hypotheses, output in zip(pending_indices, pending_hypotheses, outputs):
            raw_text = output.outputs[0].text if output.outputs else ""
            selected_prediction = select_llm_prediction(raw_text, hypotheses)
            if selected_prediction:
                fused_predictions[row_idx] = selected_prediction
                fusion_sources[row_idx] = "llm"
            else:
                fused_predictions[row_idx] = majority_predictions[row_idx]
                if sanitize_llm_output(raw_text):
                    fusion_sources[row_idx] = "llm_invalid_choice_fallback"
                    llm_invalid_choice_fallbacks += 1
                else:
                    fusion_sources[row_idx] = "llm_empty_fallback"
                    llm_empty_fallbacks += 1

        progress.update(len(pending_prompts))
        pending_prompts = []
        pending_indices = []
        pending_hypotheses = []

    for row_idx, row in enumerate(metadata_df.iter_rows(named=True)):
        hypotheses = [str(row[column] or "") for column in hypothesis_columns]
        majority_predictions[row_idx] = choose_majority_prediction(hypotheses)
        unique_counts[row_idx] = count_unique_hypotheses(hypotheses)

        consensus_prediction = None
        if not args.disable_consensus_shortcut:
            consensus_prediction = choose_consensus_prediction(hypotheses)

        if consensus_prediction is not None:
            fused_predictions[row_idx] = consensus_prediction
            fusion_sources[row_idx] = "consensus"
            progress.update(1)
            continue

        if unique_counts[row_idx] < args.llm_min_unique_hypotheses:
            fused_predictions[row_idx] = majority_predictions[row_idx]
            fusion_sources[row_idx] = "majority"
            progress.update(1)
            continue

        pending_prompts.append(create_prompt(labels=labels, predictions=hypotheses))
        pending_indices.append(row_idx)
        pending_hypotheses.append(hypotheses)
        if len(pending_prompts) >= args.llm_batch_size:
            flush_pending()

    flush_pending()
    progress.close()

    return (
        majority_predictions,
        fused_predictions,
        fusion_sources,
        unique_counts,
        llm_prompt_count,
        llm_empty_fallbacks,
        llm_invalid_choice_fallbacks,
    )


def main() -> None:
    args = parse_args()
    labels = resolve_labels(args)

    if len(args.run_dirs) < 2:
        raise ValueError("Pass at least two run directories for fusion.")
    if args.fold < 1:
        raise ValueError("--fold must be >= 1.")
    if not (1 <= args.reference_run_index <= len(args.run_dirs)):
        raise ValueError(f"--reference-run-index must be between 1 and {len(args.run_dirs)}.")
    if args.tensor_parallel_size < 1:
        raise ValueError("--tensor-parallel-size must be >= 1.")
    if args.llm_batch_size < 1:
        raise ValueError("--llm-batch-size must be >= 1.")
    if args.llm_min_unique_hypotheses < 2:
        raise ValueError("--llm-min-unique-hypotheses must be >= 2.")
    if args.llm_min_unique_hypotheses > len(args.run_dirs):
        raise ValueError(
            f"--llm-min-unique-hypotheses must be <= the number of runs ({len(args.run_dirs)})."
        )
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be >= 1.")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be >= 1 when provided.")
    if args.gpu_memory_utilization is not None and not (0.0 < args.gpu_memory_utilization <= 1.0):
        raise ValueError("--gpu-memory-utilization must be in (0, 1].")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0, 1].")

    if args.prediction_source == "eval":
        bundles = load_shared_eval_bundles(
            run_paths=args.run_dirs,
            labels=labels,
            fold=args.fold,
            device=args.device,
            reference_run_index=args.reference_run_index,
        )
    elif args.prediction_source == "oof":
        bundles = [
            load_oof_bundle(
                run_path_str=run_path,
                label=label,
                fold=args.fold,
            )
            for run_path, label in zip(args.run_dirs, labels)
        ]
    else:
        try:
            bundles = [
                load_oof_bundle(
                    run_path_str=run_path,
                    label=label,
                    fold=args.fold,
                )
                for run_path, label in zip(args.run_dirs, labels)
            ]
        except Exception as exc:
            print(
                f"Could not use saved OOF predictions ({exc}). "
                f"Falling back to shared evaluation on the validation split from run "
                f"{args.reference_run_index} ({labels[args.reference_run_index - 1]})."
            )
            bundles = load_shared_eval_bundles(
                run_paths=args.run_dirs,
                labels=labels,
                fold=args.fold,
                device=args.device,
                reference_run_index=args.reference_run_index,
            )

    alignment_stats = compute_alignment_stats(bundles)
    if alignment_stats.mismatch_detected:
        print_section("Alignment")
        print("Prediction tables do not cover the same utterance_id set.")
        print(
            f"Using only the shared intersection: {alignment_stats.shared_size} / "
            f"{alignment_stats.largest_bundle_size} utterances "
            f"({alignment_stats.coverage_vs_largest:.2%}) relative to the largest run "
            f"({alignment_stats.largest_bundle_label})."
        )
        for label, size in alignment_stats.bundle_sizes.items():
            print(f"{label}: kept {alignment_stats.shared_size} of {size} utterances")

    merged_df = join_prediction_bundles(bundles, alignment_stats=alignment_stats)
    if args.max_samples is not None:
        merged_df = merged_df.head(args.max_samples)

    (
        majority_predictions,
        fused_predictions,
        fusion_sources,
        unique_counts,
        llm_prompt_count,
        llm_empty_fallbacks,
        llm_invalid_choice_fallbacks,
    ) = run_llm_fusion(metadata_df=merged_df, labels=labels, args=args)

    result_df = merged_df.with_columns(
        pl.Series("majority_vote_prediction", majority_predictions),
        pl.Series("llm_fusion_prediction", fused_predictions),
        pl.Series("fusion_source", fusion_sources),
        pl.Series("num_unique_hypotheses", unique_counts),
        pl.lit(args.fold).alias("fold_number"),
        pl.lit(args.llm_model).alias("llm_model"),
        pl.lit(args.prediction_source).alias("prediction_source_mode"),
    )

    output_path = resolve_output_path(args.output_parquet or resolve_default_output_path(labels, args.fold))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(output_path)

    consensus_count = sum(source == "consensus" for source in fusion_sources)
    majority_count = sum(source == "majority" for source in fusion_sources)
    llm_count = sum(
        source in {"llm", "llm_empty_fallback", "llm_invalid_choice_fallback"}
        for source in fusion_sources
    )

    print_section("Runs")
    for bundle in bundles:
        print(f"{bundle.label} [{bundle.source}]: {bundle.run_dir}")
    print(f"fold: {args.fold}")
    print(f"utterances: {result_df.height}")
    print(
        f"shared_utterances: {alignment_stats.shared_size} / "
        f"{alignment_stats.largest_bundle_size} "
        f"({alignment_stats.coverage_vs_largest:.2%} of largest run: "
        f"{alignment_stats.largest_bundle_label})"
    )
    if args.max_samples is not None:
        print(f"sample_limit: {args.max_samples}")

    print_section("Fusion")
    print(f"llm_model: {args.llm_model}")
    print(f"temperature: {args.temperature}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"llm_batch_size: {args.llm_batch_size}")
    print(f"consensus_shortcut: {not args.disable_consensus_shortcut}")
    print(f"llm_min_unique_hypotheses: {args.llm_min_unique_hypotheses}")
    print(f"llm_prompts: {llm_prompt_count}")
    print(f"consensus_rows: {consensus_count}")
    print(f"majority_rows: {majority_count}")
    print(f"llm_rows: {llm_count}")
    print(f"llm_empty_fallbacks: {llm_empty_fallbacks}")
    print(f"llm_invalid_choice_fallbacks: {llm_invalid_choice_fallbacks}")

    has_references = "ground_truth" in result_df.columns and bool(
        result_df.select(pl.col("ground_truth").is_not_null().all()).item()
    )
    if has_references:
        references = result_df["ground_truth"].to_list()
        print_section("PER")
        for label in labels:
            model_per = score_ipa_cer(references, result_df[f"{label}_prediction"].to_list())
            print(f"{label}: {model_per:.5f}")
        majority_per = score_ipa_cer(references, majority_predictions)
        fusion_per = score_ipa_cer(references, fused_predictions)
        print(f"majority_vote: {majority_per:.5f}")
        print(f"llm_fusion: {fusion_per:.5f}")
    else:
        print_section("PER")
        print("ground_truth was not available, so no PER could be computed.")

    print_section("Saved Predictions")
    print(output_path)


if __name__ == "__main__":
    main()
