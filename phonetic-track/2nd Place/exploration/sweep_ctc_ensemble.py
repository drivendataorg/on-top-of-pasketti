from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exploration.ensemble_ctc_posteriors import (
    DEFAULT_RUN_A,
    DEFAULT_RUN_B,
    EnsembleRunConfig,
    run_ensemble,
    resolve_path,
)

DEFAULT_DECODER_CONFIGS = [PROJECT_ROOT / "configs/decoder/beam_search_simple.yaml"]
DEFAULT_WEIGHTS_A = [0.4, 0.5, 0.6]
DEFAULT_TEMPERATURES_A = [1.0]
DEFAULT_TEMPERATURES_B = [1.0]
DEFAULT_BLANK_SCALES_A = [1.0]
DEFAULT_BLANK_SCALES_B = [1.0]
DEFAULT_FUSIONS = ["arithmetic", "log_linear"]
DEFAULT_LEADERBOARD_PATH = PROJECT_ROOT / "exploration/reports/ensemble_sweep_leaderboard.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep posterior-fusion ensemble settings and save a leaderboard ordered by PER."
        )
    )
    parser.add_argument("--run-a", default=str(DEFAULT_RUN_A), help="First run directory.")
    parser.add_argument("--run-b", default=str(DEFAULT_RUN_B), help="Second run directory.")
    parser.add_argument("--label-a", default="whisper", help="Short label for run A.")
    parser.add_argument("--label-b", default="hubert", help="Short label for run B.")
    parser.add_argument("--fold", type=int, default=1, help="1-based fold number. Defaults to 1.")
    parser.add_argument(
        "--decoder-configs",
        nargs="+",
        default=[str(path) for path in DEFAULT_DECODER_CONFIGS],
        help="One or more decoder config paths to evaluate.",
    )
    parser.add_argument(
        "--fusions",
        nargs="+",
        choices=("arithmetic", "log_linear"),
        default=DEFAULT_FUSIONS,
        help="Fusion rules to evaluate.",
    )
    parser.add_argument(
        "--weights-a",
        nargs="+",
        type=float,
        default=DEFAULT_WEIGHTS_A,
        help="Weights for run A. Run B gets 1 - weight_a.",
    )
    parser.add_argument(
        "--temperatures-a",
        nargs="+",
        type=float,
        default=DEFAULT_TEMPERATURES_A,
        help="Temperatures for run A.",
    )
    parser.add_argument(
        "--temperatures-b",
        nargs="+",
        type=float,
        default=DEFAULT_TEMPERATURES_B,
        help="Temperatures for run B.",
    )
    parser.add_argument(
        "--blank-scales-a",
        nargs="+",
        type=float,
        default=DEFAULT_BLANK_SCALES_A,
        help="Blank posterior scales for run A.",
    )
    parser.add_argument(
        "--blank-scales-b",
        nargs="+",
        type=float,
        default=DEFAULT_BLANK_SCALES_B,
        help="Blank posterior scales for run B.",
    )
    parser.add_argument(
        "--decode-batch-size",
        type=int,
        default=256,
        help="How many utterances to decode per batch.",
    )
    parser.add_argument(
        "--leaderboard-path",
        default=str(DEFAULT_LEADERBOARD_PATH),
        help="Where to save the sweep leaderboard parquet.",
    )
    parser.add_argument(
        "--save-predictions-dir",
        default=None,
        help="Optional directory to save one parquet of ensemble predictions per sweep run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of sweep runs to execute.",
    )
    return parser.parse_args()


def config_slug(path: Path) -> str:
    return path.stem


def build_run_name(
    decoder_config_path: Path,
    fusion: str,
    weight_a: float,
    temperature_a: float,
    temperature_b: float,
    blank_scale_a: float,
    blank_scale_b: float,
) -> str:
    return (
        f"{config_slug(decoder_config_path)}"
        f"__{fusion}"
        f"__wa{weight_a:.2f}"
        f"__ta{temperature_a:.2f}"
        f"__tb{temperature_b:.2f}"
        f"__ba{blank_scale_a:.2f}"
        f"__bb{blank_scale_b:.2f}"
    ).replace(".", "p")


def main() -> None:
    args = parse_args()

    run_a = resolve_path(args.run_a)
    run_b = resolve_path(args.run_b)
    decoder_config_paths = [resolve_path(path) for path in args.decoder_configs]
    leaderboard_path = resolve_path(args.leaderboard_path)
    save_predictions_dir = resolve_path(args.save_predictions_dir) if args.save_predictions_dir else None

    sweep_space = list(
        itertools.product(
            decoder_config_paths,
            args.fusions,
            args.weights_a,
            args.temperatures_a,
            args.temperatures_b,
            args.blank_scales_a,
            args.blank_scales_b,
        )
    )
    if args.limit is not None:
        sweep_space = sweep_space[: args.limit]

    print(f"Planned runs: {len(sweep_space)}")

    leaderboard_rows: list[dict[str, object]] = []
    total_runs = len(sweep_space)

    for run_index, (decoder_config_path, fusion, weight_a, temperature_a, temperature_b, blank_scale_a, blank_scale_b) in enumerate(sweep_space, start=1):
        run_name = build_run_name(
            decoder_config_path=decoder_config_path,
            fusion=fusion,
            weight_a=weight_a,
            temperature_a=temperature_a,
            temperature_b=temperature_b,
            blank_scale_a=blank_scale_a,
            blank_scale_b=blank_scale_b,
        )
        output_path = None
        if save_predictions_dir is not None:
            output_path = save_predictions_dir / f"{run_name}.parquet"

        print()
        print(f"[{run_index}/{total_runs}] {run_name}")

        run_result = run_ensemble(
            EnsembleRunConfig(
                run_a=run_a,
                run_b=run_b,
                label_a=args.label_a,
                label_b=args.label_b,
                fold=args.fold,
                decoder_config_path=decoder_config_path,
                fusion=fusion,
                weight_a=weight_a,
                weight_b=1.0 - weight_a,
                temperature_a=temperature_a,
                temperature_b=temperature_b,
                blank_scale_a=blank_scale_a,
                blank_scale_b=blank_scale_b,
                decode_batch_size=args.decode_batch_size,
                output_path=output_path,
            )
        )

        leaderboard_rows.append(
            {
                "run_name": run_name,
                "decoder_config": str(decoder_config_path),
                "decoder_target": run_result.decoder_target,
                "fusion": fusion,
                f"{args.label_a}_weight": weight_a,
                f"{args.label_b}_weight": 1.0 - weight_a,
                f"{args.label_a}_temperature": temperature_a,
                f"{args.label_b}_temperature": temperature_b,
                f"{args.label_a}_blank_scale": blank_scale_a,
                f"{args.label_b}_blank_scale": blank_scale_b,
                f"{args.label_a}_per": run_result.model_a_per,
                f"{args.label_b}_per": run_result.model_b_per,
                "ensemble_per": run_result.ensemble_per,
                f"delta_vs_{args.label_a}": (
                    None if run_result.ensemble_per is None or run_result.model_a_per is None
                    else run_result.ensemble_per - run_result.model_a_per
                ),
                f"delta_vs_{args.label_b}": (
                    None if run_result.ensemble_per is None or run_result.model_b_per is None
                    else run_result.ensemble_per - run_result.model_b_per
                ),
                f"trimmed_from_{args.label_a}": run_result.trimmed_counter.get(args.label_a, 0),
                f"trimmed_from_{args.label_b}": run_result.trimmed_counter.get(args.label_b, 0),
                "trimmed_from_none": run_result.trimmed_counter.get("none", 0),
                "prediction_path": str(output_path) if output_path is not None else None,
            }
        )

        if run_result.ensemble_per is not None:
            print(
                f"ensemble_per={run_result.ensemble_per:.5f} | "
                f"delta_vs_{args.label_a}={run_result.ensemble_per - run_result.model_a_per:+.5f} | "
                f"delta_vs_{args.label_b}={run_result.ensemble_per - run_result.model_b_per:+.5f}"
            )

    leaderboard_df = pl.DataFrame(leaderboard_rows).sort("ensemble_per")
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_df.write_parquet(leaderboard_path)

    print()
    print("Top results")
    print("-----------")
    print(leaderboard_df.head(min(10, leaderboard_df.height)))
    print()
    print(f"Saved leaderboard: {leaderboard_path}")


if __name__ == "__main__":
    main()
