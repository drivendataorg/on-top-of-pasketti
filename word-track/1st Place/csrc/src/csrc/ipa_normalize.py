import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

app = typer.Typer()

IPA_TO_ASCII: dict[str, str] = {
    "ɑ": "a",
    "ɔ": "o",
    "ɜ": "e",
    "ɪ": "i",
    "ʊ": "u",
    "ʌ": "u",
    "ɛ": "e",
}

IPA_CHARS = set(IPA_TO_ASCII.keys())


def normalize_ipa_to_ascii(text: str) -> str:
    """IPA文字をASCIIに置換する。"""
    for ipa, ascii_char in IPA_TO_ASCII.items():
        text = text.replace(ipa, ascii_char)
    return text


def has_ipa(text: str) -> bool:
    """テキストにIPA文字が含まれるかチェックする。"""
    return any(c in IPA_CHARS for c in text)


def build_reverse_dict(csv_paths: list[Path]) -> dict[str, str]:
    """訓練CSVからASCII→IPA単語辞書を構築する。

    IPA文字を含む単語について、正規化後→元の単語のマッピングを作る。
    同じASCII形に複数のIPA単語がマッピングされる場合は最頻出を採用。
    """
    ascii_to_ipa_counts: dict[str, Counter[str]] = {}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for raw_text in df["text"].dropna():
            cleaned = str(raw_text).replace(" | ", " ").strip()
            cleaned = " ".join(cleaned.split())
            for word in cleaned.split():
                if has_ipa(word):
                    ascii_word = normalize_ipa_to_ascii(word)
                    if ascii_word not in ascii_to_ipa_counts:
                        ascii_to_ipa_counts[ascii_word] = Counter()
                    ascii_to_ipa_counts[ascii_word][word] += 1

    reverse_dict: dict[str, str] = {}
    for ascii_word, counter in ascii_to_ipa_counts.items():
        most_common = counter.most_common(1)[0][0]
        reverse_dict[ascii_word] = most_common

    return reverse_dict


def load_reverse_dict(path: Path) -> dict[str, str]:
    """JSONファイルから逆変換辞書を読み込む。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def reverse_ipa(text: str, reverse_dict: dict[str, str], context_threshold: float = 0.5) -> str:
    """コンテキストベースでASCIIテキストをIPA単語に逆変換する。

    1. 文を単語分割
    2. 各単語が辞書にあるかチェック → IPA候補数カウント
    3. 候補割合 >= 閾値 → 辞書にある単語を逆変換
    4. 閾値未満 → 何もしない
    """
    words = text.split()
    if not words:
        return text

    matches = sum(1 for w in words if w in reverse_dict)
    ratio = matches / len(words)

    if ratio < context_threshold:
        return text

    return " ".join(reverse_dict.get(w, w) for w in words)


@app.command()
def build_dict(
    csv_paths: Annotated[list[Path], typer.Argument(help="訓練CSVファイルパス（複数指定可）")],
    output: Annotated[Path, typer.Option("--output", "-o", help="出力JSONパス")] = Path(
        "output/ipa_reverse_dict.json",
    ),
) -> None:
    """訓練CSVからIPA逆変換辞書を構築してJSONに保存する。"""
    for p in csv_paths:
        if not p.exists():
            typer.echo(f"Error: {p} not found", err=True)
            raise typer.Exit(1)

    rev_dict = build_reverse_dict(csv_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(rev_dict, f, ensure_ascii=False, indent=2)

    typer.echo(f"Built reverse dict with {len(rev_dict)} entries -> {output}")


if __name__ == "__main__":
    app()
