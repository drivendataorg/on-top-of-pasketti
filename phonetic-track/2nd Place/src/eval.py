from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataset import prepare_dl_dataset
from src.preprocessing.get_json import get_data_from_json
from src.utils.score import score_ipa_cer


@dataclass(frozen=True)
class EvalContext:
    requested_path: Path
    run_dir: Path
    fold_dir: Path
    config_path: Path
    checkpoint_path: Path
    cfg: DictConfig
    fold: int
    fold_index: int
    val_data: list[dict[str, Any]]
    sample_loader: Any | None
    dataset_info: dict[str, Any] | None


@dataclass(frozen=True)
class EvalResult:
    requested_path: Path
    run_dir: Path
    fold_dir: Path
    checkpoint_path: Path
    fold: int
    fold_index: int
    val_data: list[dict[str, Any]]
    predictions_by_id: dict[str, str]
    references: list[str]
    predictions: list[str]
    per: float
    load_info: dict[str, Any]

    def to_prediction_frame(self) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for item in self.val_data:
            utterance_id = item["utterance_id"]
            rows.append(
                {
                    "utterance_id": utterance_id,
                    "child_id": item.get("child_id"),
                    "ground_truth": item["phonetic_text"],
                    "prediction": self.predictions_by_id[utterance_id],
                    "fold": self.fold_index,
                }
            )
        return pl.DataFrame(rows).sort("utterance_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a saved run, rebuild the original validation split, and "
            "compute PER on that full validation set with the saved checkpoint."
        )
    )
    parser.add_argument(
        "output_dir",
        help="Run output directory, e.g. outputs/2026-03-23/02-14-17_skilled-where_merch-247",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="1-based fold number whose checkpoint should be loaded. Defaults to 1.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, e.g. cuda:0 or cpu. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--predictions-output",
        default=None,
        help=(
            "Optional parquet path to save utterance-level validation predictions. "
            "Useful for downstream ensembling/fusion."
        ),
    )
    parser.add_argument(
        "--save-logits",
        action="store_true",
        help="Save val_logits_best.npz to the fold directory (for MBR decoding).",
    )
    return parser.parse_args()


def _trim_logits_batch(logits: torch.Tensor, output_lengths: torch.Tensor) -> list[torch.Tensor]:
    logits_cpu = logits.detach().cpu()
    lengths_cpu = output_lengths.detach().cpu().tolist()
    return [seq_logits[: int(seq_len)].contiguous() for seq_logits, seq_len in zip(logits_cpu, lengths_cpu)]


def resolve_existing_path(path_str: str) -> Path:
    raw_path = Path(path_str).expanduser()
    candidates = [raw_path, PROJECT_ROOT / raw_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


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


def load_model_weights(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ema_state = checkpoint.get("ema_state_dict")

    if ema_state:
        state_dict = {
            key[len("ema_model.") :]: value
            for key, value in ema_state.items()
            if key.startswith("ema_model.")
        }
        weight_source = "ema"
    else:
        state_dict = checkpoint["model_state_dict"]
        weight_source = "model"

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "checkpoint": checkpoint,
        "weight_source": weight_source,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def build_eval_cfg(cfg: DictConfig) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.preprocessing.min_duration_sec = 0
    cfg.training.dataloader.num_workers = 0
    cfg.training.dataloader.pin_memory = False

    # For duration sampler:
    cfg.sampler.val.max_batch_seconds = 120.0
    # For bucket sampler:
    # cfg.sampler.val.batch_size = 16

    return cfg


def prepare_eval_context(
    output_dir: str | Path,
    fold: int,
    val_data_override: list[dict[str, Any]] | None = None,
    build_dataset: bool = True,
) -> EvalContext:
    if fold < 1:
        raise ValueError("--fold must be >= 1")

    requested_path = resolve_existing_path(str(output_dir))
    run_dir, fold_dir = resolve_run_and_fold_dirs(requested_path, fold)
    config_path = run_dir / ".hydra" / "config.yaml"
    checkpoint_path = fold_dir / "best_model.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")

    cfg = build_eval_cfg(OmegaConf.load(config_path))
    fold_index = fold - 1

    if val_data_override is None:
        all_data = get_data_from_json(cfg, inference=False)
        _, val_data = instantiate(cfg.cv.splitter)(all_data=all_data, fold=fold_index)
    else:
        val_data = list(val_data_override)
    if not val_data:
        raise RuntimeError(f"No validation samples found for fold {fold}")

    sample_loader = None
    dataset_info = None
    if build_dataset:
        sample_loader, _, dataset_info = prepare_dl_dataset(
            cfg,
            fold=fold_index,
            inference=True,
            data_override=val_data,
        )

    return EvalContext(
        requested_path=requested_path,
        run_dir=run_dir,
        fold_dir=fold_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        fold=fold,
        fold_index=fold_index,
        val_data=val_data,
        sample_loader=sample_loader,
        dataset_info=dataset_info,
    )


def build_model_for_eval(
    cfg: DictConfig,
    dataset_info: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    decoder = instantiate(cfg.model.decoder, tokenizer=dataset_info["tokenizer"])
    model = instantiate(
        cfg.model,
        vocab_size=dataset_info["vocab_size"],
        vocab=dataset_info["tokenizer"],
        inference=True,
        decoder=decoder,
    )

    load_info = load_model_weights(model, checkpoint_path)
    model = model.to(device)
    model.eval()
    return model, load_info


def extract_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, (tuple, list)):
        if len(model_output) > 1 and isinstance(model_output[1], torch.Tensor):
            return model_output[1]
        tensor_candidates = [value for value in model_output if isinstance(value, torch.Tensor)]
        if tensor_candidates:
            return tensor_candidates[0]
        raise TypeError("Model output did not contain a tensor of logits.")
    raise TypeError(f"Unsupported model output type: {type(model_output)!r}")


def _build_logits_payload(
    trimmed_logits: list[torch.Tensor],
    utterance_ids: list[str],
) -> dict[str, Any]:
    """Pack per-utterance logits into a flat NPZ-ready dict."""
    import numpy as np
    lengths = np.asarray([seq.shape[0] for seq in trimmed_logits], dtype=np.int32)
    offsets = np.zeros(len(lengths), dtype=np.int64)
    if len(lengths) > 1:
        offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
    if trimmed_logits:
        packed = torch.cat([seq.to(dtype=torch.float16) for seq in trimmed_logits], dim=0).numpy()
    else:
        packed = np.empty((0, 0), dtype=np.float16)
    return {
        "logits": packed,
        "offsets": offsets,
        "lengths": lengths,
        "utterance_ids": np.array(utterance_ids, dtype=object),
    }


def generate_predictions(
    model: torch.nn.Module,
    sample_loader: Any,
    device: torch.device,
    progress_desc: str = "Evaluating",
    collect_logits: bool = False,
) -> tuple[dict[str, str], dict[str, Any] | None]:
    predictions_by_id: dict[str, str] = {}
    all_trimmed_logits: list[torch.Tensor] = []
    all_utterance_ids: list[str] = []
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            for batch in tqdm(sample_loader, desc=progress_desc):
                input_features = batch["input_features"].to(device)
                input_lengths = batch["input_lengths"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                logits = extract_logits(model(input_features, attention_mask=attention_mask))
                output_lengths = model.get_output_lengths(input_lengths)
                trimmed_logits = _trim_logits_batch(logits, output_lengths)
                predictions = model.decoder(trimmed_logits)

                for utterance_id, prediction in zip(batch["utterance_ids"], predictions):
                    predictions_by_id[utterance_id] = prediction

                if collect_logits:
                    all_trimmed_logits.extend(trimmed_logits)
                    all_utterance_ids.extend(batch["utterance_ids"])

    logits_payload = None
    if collect_logits:
        logits_payload = _build_logits_payload(all_trimmed_logits, all_utterance_ids)
    return predictions_by_id, logits_payload


def evaluate_run(
    output_dir: str | Path,
    fold: int = 1,
    device: str | None = None,
    progress_desc: str = "Evaluating",
    val_data_override: list[dict[str, Any]] | None = None,
    save_logits: bool = False,
) -> EvalResult:
    context = prepare_eval_context(
        output_dir=output_dir,
        fold=fold,
        val_data_override=val_data_override,
    )
    if context.sample_loader is None or context.dataset_info is None:
        raise RuntimeError("Evaluation context is missing dataset artifacts.")
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, load_info = build_model_for_eval(
        cfg=context.cfg,
        dataset_info=context.dataset_info,
        checkpoint_path=context.checkpoint_path,
        device=device_obj,
    )
    predictions_by_id, logits_payload = generate_predictions(
        model=model,
        sample_loader=context.sample_loader,
        device=device_obj,
        progress_desc=progress_desc,
        collect_logits=save_logits,
    )

    if save_logits and logits_payload is not None:
        import numpy as np
        logits_path = context.fold_dir / "val_logits_best.npz"
        np.savez_compressed(logits_path, **logits_payload)
        print(f"Logits saved to {logits_path}")

    references = [item["phonetic_text"] for item in context.val_data]
    predictions = [predictions_by_id[item["utterance_id"]] for item in context.val_data]
    per = score_ipa_cer(references, predictions)

    return EvalResult(
        requested_path=context.requested_path,
        run_dir=context.run_dir,
        fold_dir=context.fold_dir,
        checkpoint_path=context.checkpoint_path,
        fold=context.fold,
        fold_index=context.fold_index,
        val_data=context.val_data,
        predictions_by_id=predictions_by_id,
        references=references,
        predictions=predictions,
        per=per,
        load_info=load_info,
    )


def save_predictions_parquet(result: EvalResult, output_path: str | Path) -> Path:
    resolved_path = resolve_output_path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_prediction_frame().write_parquet(resolved_path)
    return resolved_path


def main() -> None:
    args = parse_args()
    result = evaluate_run(
        output_dir=args.output_dir,
        fold=args.fold,
        device=args.device,
        save_logits=args.save_logits,
    )

    print(f"Run directory : {result.run_dir}")
    print(f"Checkpoint    : {result.checkpoint_path}")
    print(f"Fold          : {result.fold}")
    print(f"Val size      : {len(result.val_data)}")
    print(f"Weight source : {result.load_info['weight_source']}")
    if result.load_info["missing_keys"]:
        print(f"Missing keys  : {result.load_info['missing_keys']}")
    if result.load_info["unexpected_keys"]:
        print(f"Unexpected keys: {result.load_info['unexpected_keys']}")
    print(f"PER           : {result.per:.5f}")

    if args.predictions_output:
        saved_path = save_predictions_parquet(result, args.predictions_output)
        print(f"Predictions   : {saved_path}")


if __name__ == "__main__":
    main()
