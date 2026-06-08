"""NeMo Forced Aligner (NFA) のラッパーCLIツール.

NeMoリポジトリの tools/nemo_forced_aligner/align.py をsubprocessで呼び出す。
"""

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()

# NeMoリポジトリ内のNFAスクリプトのデフォルトパス
_DEFAULT_NFA_SCRIPT = "tools/nemo_forced_aligner/align.py"


def _find_nfa_script(nemo_dir: Path | None) -> Path:
    """NFAのalign.pyスクリプトを探す."""
    if nemo_dir is not None:
        script = nemo_dir / _DEFAULT_NFA_SCRIPT
        if script.exists():
            return script
        msg = f"NFA script not found at {script}"
        raise typer.BadParameter(msg)

    msg = (
        "--nemo-dir を指定してください。"
        "NeMoリポジトリのルートディレクトリ（tools/nemo_forced_aligner/align.py が存在する場所）を指定します。"
    )
    raise typer.BadParameter(msg)


@app.command()
def align(
    target_dir: Annotated[Path, typer.Argument(help="target_dir")],
    model_path: Annotated[str, typer.Argument(help="CTCモデルのパス（.nemoファイルまたはNGC名）")],
    nemo_dir: Annotated[Path, typer.Argument(help="NeMoリポジトリのルートディレクトリ")],
    batch_size: Annotated[int, typer.Option("--batch-size", help="バッチサイズ")] = 1,
) -> None:
    """NeMo Forced Aligner (NFA) を実行してアライメントを取得する.

    CTCモデルを使用して音声とテキストのforced alignmentを行い、
    token/word/segmentレベルのタイムスタンプをCTMファイルとして出力する。
    """
    nemo_dir = Path(nemo_dir)
    nfa_script = _find_nfa_script(nemo_dir)
    output_dir = target_dir.joinpath("forced_align")
    manifest_filepath = output_dir.joinpath("forced_align_manifest.jsonl")

    # model_pathがローカルファイルかNGC名かで引数を切り替え
    model_arg = f"model_path={model_path}" if Path(model_path).exists() else f"pretrained_name={model_path}"

    cmd = [
        sys.executable,
        str(nfa_script),
        model_arg,
        f"manifest_filepath={manifest_filepath.resolve()}",
        f"output_dir={output_dir.resolve()}",
        f"batch_size={batch_size}",
    ]

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"NFA failed with return code {result.returncode}", file=sys.stderr, flush=True)
        raise typer.Exit(result.returncode)

    print(f"Alignment completed. Output saved to {output_dir}", flush=True)


if __name__ == "__main__":
    app()
