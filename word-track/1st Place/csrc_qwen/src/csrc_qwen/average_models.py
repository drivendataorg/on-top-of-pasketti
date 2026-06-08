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


def _load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    """モデルディレクトリから全テンソルをロードする（シャード対応）。"""
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"

    if single_path.exists():
        return safe_load_file(str(single_path), device="cpu")

    if index_path.exists():
        with index_path.open() as f:
            index = json.load(f)
        shard_files = set(index["weight_map"].values())
        state_dict: dict[str, torch.Tensor] = {}
        for shard in sorted(shard_files):
            state_dict.update(safe_load_file(str(model_dir / shard), device="cpu"))
        return state_dict

    msg = f"model.safetensors も model.safetensors.index.json も見つかりません: {model_dir}"
    raise FileNotFoundError(msg)


@app.command()
def main(
    model_paths: Annotated[list[str], typer.Argument(help="平均するモデルのパス（2つ以上）")],
    output_path: Annotated[str, typer.Option("--output", "-o", help="平均済みモデルの出力先")],
    weights: Annotated[
        list[float] | None,
        typer.Option("--weight", "-w", help="各モデルの重み（指定しない場合は均等）"),
    ] = None,
) -> None:
    """複数のモデルの重みを平均して保存する。"""
    if len(model_paths) < 2:
        msg = "モデルパスは2つ以上指定してください"
        raise typer.BadParameter(msg)

    dirs = [Path(p) for p in model_paths]
    out = Path(output_path)
    n = len(dirs)

    # 重みの決定
    if weights is None:
        weights = [1.0 / n] * n
    elif len(weights) != n:
        msg = f"重みの数({len(weights)})がモデル数({n})と一致しません"
        raise typer.BadParameter(msg)

    # 重みのロード
    state_dicts: list[dict[str, torch.Tensor]] = []
    for i, d in enumerate(dirs):
        logger.info(f"Loading model {i + 1}: {d}")
        state_dicts.append(_load_state_dict(d))

    # キーの一致を確認
    ref_keys = state_dicts[0].keys()
    for i, sd in enumerate(state_dicts[1:], start=2):
        if sd.keys() != ref_keys:
            only_ref = ref_keys - sd.keys()
            only_cur = sd.keys() - ref_keys
            msg = f"state_dict のキーが不一致。model1のみ: {only_ref}, model{i}のみ: {only_cur}"
            raise ValueError(msg)

    # 平均
    weights_str = ", ".join(f"{w:.4f}" for w in weights)
    logger.info(f"Averaging {n} models with weights: [{weights_str}]")
    averaged: dict[str, torch.Tensor] = {}
    for key in ref_keys:
        averaged[key] = sum(w * sd[key] for w, sd in zip(weights, state_dicts))  # type: ignore[assignment]
    del state_dicts

    # 最初のモデルのディレクトリをコピー（config, tokenizer 等）
    out.mkdir(parents=True, exist_ok=True)
    for f in dirs[0].iterdir():
        if f.suffix != ".safetensors":
            if f.name == "model.safetensors.index.json":
                continue
            dst = out / f.name
            if f.is_dir():
                shutil.copytree(f, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(f, dst)

    # 平均済み重みを単一ファイルで保存
    logger.info(f"Saving averaged model to: {out}")
    safe_save_file(averaged, str(out / "model.safetensors"))

    logger.info("Done")


if __name__ == "__main__":
    app()
