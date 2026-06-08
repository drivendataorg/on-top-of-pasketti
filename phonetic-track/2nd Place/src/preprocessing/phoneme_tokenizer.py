import json
from pathlib import Path
from typing import List, Dict
from omegaconf import OmegaConf
import re
from src.preprocessing.data_split import group_child_id_split
from src.preprocessing.get_json import get_data_from_json

class PhonemeTokenizer:
    """A simple character-level tokenizer for IPA phoneme strings."""
    def __init__(self):
        # Reserve 0 for the CTC Blank token, and 1 for Pad token
        self.blank_token = "<blank>" # note this is not a space!
        self.pad_token = "<pad>" # this needed for the CTC loss, pads sequences to be the same length
        self.vocab = {'<blank>': 0, '<pad>': 1, ' ': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8, 'h': 9, 'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16, 'p': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'z': 25, 'æ': 26, 'ç': 27, 'ð': 28, 'ŋ': 29, 'ɐ': 30, 'ɑ': 31, 'ɔ': 32, 'ə': 33, 'ɚ': 34, 'ɛ': 35, 'ɟ': 36, 'ɪ': 37, 'ɫ': 38, 'ɬ': 39, 'ɹ': 40, 'ɾ': 41, 'ʁ': 42, 'ʃ': 43, 'ʊ': 44, 'ʌ': 45, 'ʒ': 46, 'ʔ': 47, 'ʝ': 48, 'ʤ': 49, 'ʧ': 50, 'ː': 51, 'θ': 52, 'χ': 53}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}  
        
    def build_vocab(self, data: List[Dict]):
        print("*"*50)
        print("Building vocabulary...")
        print("*"*50)
        """Builds vocabulary directly from a list of data dictionaries to prevent re-reading files."""
        unique_chars = set()
        for item in data:
            text = item.get("phonetic_text", "")
            unique_chars.update(list(text))
            
        for i, char in enumerate(sorted(unique_chars), start=len(self.vocab)):
            self.vocab[char] = i
            self.inverse_vocab[i] = char


    def __call__(self, text: str) -> List[int]:
        """Encodes a string into a list of integers."""
        # Note: If a character isn't in vocab, we fall back to blank 
        #print(f"Text: {text}")  # Debug print to check input text
        return [self.vocab.get(char, self.vocab[self.blank_token]) for char in text]
        
    def decode(self, ids: List[int]) -> str:
        """Decodes a list of integers back to a string."""
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
    
class PhonemeTokenizer_long_mark:
    """A character-level tokenizer for IPA phoneme strings that natively supports long mark multi-char tokens."""
    def __init__(self):
        self.blank_token = "<blank>" 
        self.pad_token = "<pad>" 
        self.vocab = {
            '<blank>': 0, '<pad>': 1, ' ': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8, 'h': 9, 
            'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16, 'p': 17, 'r': 18, 's': 19, 
            't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'z': 25, 'æ': 26, 'ç': 27, 'ð': 28, 'ŋ': 29, 
            'ɐ': 30, 'ɑ': 31, 'ɔ': 32, 'ə': 33, 'ɚ': 34, 'ɛ': 35, 'ɟ': 36, 'ɪ': 37, 'ɫ': 38, 'ɬ': 39, 
            'ɹ': 40, 'ɾ': 41, 'ʁ': 42, 'ʃ': 43, 'ʊ': 44, 'ʌ': 45, 'ʒ': 46, 'ʔ': 47, 'ʝ': 48, 'ʤ': 49, 
            'ʧ': 50, 'ː': 51, 'θ': 52, 'χ': 53, 
            'iː': 54, 'aː': 55, 'uː': 56   # Added multi-character long mark tokens
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}  
        
    def build_vocab(self, data: List[Dict]):
        print("*"*50)
        print("Building vocabulary...")
        print("*"*50)
        
        unique_tokens = set()
        for item in data:
            text = item.get("phonetic_text", "")
            # Regex: match iː, aː, uː first. If no match, grab the next single character (.)
            tokens = re.findall(r'iː|aː|uː|.', text)
            unique_tokens.update(tokens)
            
        # Filter out tokens already in the vocab so we don't duplicate/override IDs
        new_tokens = [t for t in sorted(unique_tokens) if t not in self.vocab]
            
        for i, token in enumerate(new_tokens, start=len(self.vocab)):
            self.vocab[token] = i
            self.inverse_vocab[i] = token

    def __call__(self, text: str) -> List[int]:
        """Encodes a string into a list of integers."""
        # Regex chunking prior to encoding
        tokens = re.findall(r'iː|aː|uː|.', text)
        return [self.vocab.get(token, self.vocab[self.blank_token]) for token in tokens]
        
    def decode(self, ids: List[int]) -> str:
        """Decodes a list of integers back to a string."""
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
    
if __name__ == '__main__':
    """
    Debugging.
    """
    cfg = OmegaConf.load("configs/default.yaml")
    all_data = get_data_from_json(cfg)

    # Swap to the new tokenizer for testing
    tokenizer = PhonemeTokenizer_long_mark()
    
    tokenizer.build_vocab(data=all_data)
    
    print("Vocab size:", tokenizer.vocab_size)
    print("\n")

    # Testing to ensure our new multi-char marks work
    sample_text = "tiːst uː aː"
    encoded = tokenizer(sample_text)
    decoded = tokenizer.decode(encoded)

    print("Sample text:", sample_text)
    print("Encoded:", encoded)
    print("Decoded:", decoded)
    print("\nVocab:")
    print(tokenizer.vocab)
    print("\nInverse Vocab:")
    print(tokenizer.inverse_vocab)