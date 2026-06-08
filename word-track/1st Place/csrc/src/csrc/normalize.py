from .metric import (
    EnglishTextNormalizer,
    english_spelling_normalizer,
)

_english_normalizer = EnglishTextNormalizer(english_spelling_normalizer)


def normalize_orthographic(text: str) -> str:
    """WERスコアリングと同じ正規化を適用する（Whisper EnglishTextNormalizer）。"""
    return _english_normalizer(text)
