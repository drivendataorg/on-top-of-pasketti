import json
import shutil
from pathlib import Path
from typing import Annotated

import torch
import typer
from loguru import logger
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

app = typer.Typer()


def _validate_adapter_configs(paths: list[Path]) -> None:
    """全アダプターの LoRA 設定が一致するか検証する。"""
    configs = []
    for p in paths:
        with (p / "adapter_config.json").open() as f:
            configs.append(json.load(f))

    base = configs[0]
    for i, cfg in enumerate(configs[1:], start=2):
        for key in ("r", "lora_alpha", "target_modules", "peft_type"):
            v1, v2 = base.get(key), cfg.get(key)
            if isinstance(v1, list) and isinstance(v2, list):
                v1, v2 = sorted(v1), sorted(v2)
            if v1 != v2:
                msg = (
                    f"adapter_config.json の '{key}' が不一致 (adapter1 vs adapter{i}): "
                    f"{base.get(key)} vs {cfg.get(key)}"
                )
                raise ValueError(msg)


def _average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """N個の state_dict を単純平均する。"""
    base_keys = state_dicts[0].keys()
    for i, sd in enumerate(state_dicts[1:], start=2):
        if sd.keys() != base_keys:
            only_base = base_keys - sd.keys()
            only_other = sd.keys() - base_keys
            msg = (
                f"state_dict のキーが不一致 (adapter1 vs adapter{i})。"
                f"adapter1のみ: {only_base}, adapter{i}のみ: {only_other}"
            )
            raise ValueError(msg)

    averaged: dict[str, torch.Tensor] = {}
    for key in base_keys:
        stacked = torch.stack([sd[key] for sd in state_dicts])
        averaged[key] = stacked.mean(dim=0)
    return averaged


@app.command()
def main(
    adapter_paths: Annotated[list[str], typer.Argument(help="LoRA アダプターのパス（2個以上）")],
    output_path: Annotated[str, typer.Option("--output", "-o", help="平均済みアダプターの出力先")] = "",
) -> None:
    """N個の LoRA アダプターの重みを単純平均して保存する。"""
    if len(adapter_paths) < 2:
        msg = "アダプターは2個以上指定してください"
        raise typer.BadParameter(msg)

    dirs = [Path(p) for p in adapter_paths]
    if not output_path:
        msg = "--output / -o で出力先を指定してください"
        raise typer.BadParameter(msg)
    out = Path(output_path)

    # 設定の検証
    logger.info(f"Validating {len(dirs)} adapter configs")
    _validate_adapter_configs(dirs)

    # 重みのロード
    state_dicts = []
    for i, d in enumerate(dirs, start=1):
        logger.info(f"Loading adapter {i}/{len(dirs)}: {d}")
        sd = safe_load_file(str(d / "adapter_model.safetensors"), device="cpu")
        state_dicts.append(sd)

    # 平均
    logger.info(f"Averaging {len(state_dicts)} adapters")
    averaged = _average_state_dicts(state_dicts)
    del state_dicts

    # 保存
    out.mkdir(parents=True, exist_ok=True)
    safe_save_file(averaged, str(out / "adapter_model.safetensors"))
    shutil.copy2(dirs[0] / "adapter_config.json", out / "adapter_config.json")

    logger.info(f"Saved averaged adapter to: {out}")


if __name__ == "__main__":
    app()
