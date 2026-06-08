from __future__ import annotations

from typing import List


class WordCharTokenizer:
    """Character-level tokenizer for word transcripts used with CTC.

    Supported symbols are lowercase English letters, space, and apostrophe.
    This keeps the output head compact while preserving common contractions.
    """

    def __init__(self) -> None:
        self.blank_token = "<blank>"
        self.pad_token = "<pad>"

        self.vocab = {
            self.blank_token: 0,
            self.pad_token: 1,
            " ": 2,
        }

        for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz", start=3):
            self.vocab[ch] = i

        self.vocab["'"] = 29
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def normalize_text(self, text: str) -> str:
        text = (text or "").lower()
        chars: List[str] = []
        for ch in text:
            if ch in self.vocab:
                chars.append(ch)
                continue
            # Treat all whitespace as a single CTC separator symbol.
            if ch.isspace():
                chars.append(" ")
        return "".join(chars)

    def __call__(self, text: str) -> List[int]:
        normalized = self.normalize_text(text)
        return [self.vocab[ch] for ch in normalized]

    def decode(self, ids: List[int]) -> str:
        return "".join([self.inverse_vocab.get(i, "") for i in ids])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def blank_token_id(self) -> int:
        return self.vocab[self.blank_token]
