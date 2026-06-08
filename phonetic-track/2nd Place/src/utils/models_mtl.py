from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.utils.hub import cached_file


class Wav2Vec2MTL(nn.Module):
    """Shared wav2vec2/wavlm backbone with separate phoneme and word CTC heads."""

    def __init__(
        self,
        phoneme_vocab_size: int,
        word_vocab_size: int,
        pretrained_name: str = "microsoft/wavlm-base-plus",
        classifier_dropout: float = 0.1,
        gradient_checkpointing: bool | None = None,
        augmentation=None,
        backbone_lr: float | None = None,
        head_lr: float | None = None,
        decoder=None,
        phon_decoder=None,
        word_decoder=None,
        phon_vocab=None,
        word_vocab=None,
        LLA: bool = False,
    ):
        super().__init__()
        self.LLA = bool(LLA)
        self.phon_decoder = phon_decoder
        self.word_decoder = word_decoder
        self.phon_vocab = phon_vocab
        self.word_vocab = word_vocab

        mask_time_prob = 0.0
        mask_feature_prob = 0.0
        mask_cfg = augmentation.get("masking", None) if augmentation is not None else None
        if mask_cfg is not None and mask_cfg.get("enabled", False):
            mask_time_prob = float(mask_cfg.get("mask_time_prob", 0.05))
            mask_feature_prob = float(mask_cfg.get("mask_feature_prob", 0.0))

        def _load_backbone_config(model_id: str):
            if Path(model_id).exists():
                config_path = Path(model_id) / "config.json"
            else:
                config_path = cached_file(model_id, "config.json", local_files_only=False)
            config = AutoConfig.from_pretrained(str(config_path))
            config.gradient_checkpointing = False
            if hasattr(config, "mask_time_prob"):
                config.mask_time_prob = mask_time_prob
            if hasattr(config, "mask_feature_prob"):
                config.mask_feature_prob = mask_feature_prob
            return config

        config = _load_backbone_config(pretrained_name)
        self.wav2vec2 = AutoModel.from_pretrained(pretrained_name, config=config)

        if gradient_checkpointing is True:
            self.wav2vec2.gradient_checkpointing_enable()
        elif gradient_checkpointing is False:
            self.wav2vec2.gradient_checkpointing_disable()

        self.wav2vec2.enable_input_require_grads = lambda *a, **kw: None

        hidden_size = self.wav2vec2.config.hidden_size
        num_states = self.wav2vec2.config.num_hidden_layers + 1

        mu = 0.70 * (num_states - 1)
        sigma = 0.15 * num_states
        indices = torch.arange(num_states, dtype=torch.float32)
        initial_logits = 3.0 * torch.exp(-0.5 * ((indices - mu) / sigma) ** 2)
        self.layer_weights = nn.Parameter(initial_logits)

        self.phoneme_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size, phoneme_vocab_size),
        )
        self.word_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size, word_vocab_size),
        )

        with torch.no_grad():
            self.phoneme_head[-1].bias[0] = -2.0
            self.word_head[-1].bias[0] = -2.0

        print(
            "\n[Model Architecture: Wav2Vec2MTL] "
            f"backbone={pretrained_name} phon_vocab={phoneme_vocab_size} word_vocab={word_vocab_size}"
        )
        if backbone_lr is not None and head_lr is not None:
            print(f"Configured LRs | backbone={backbone_lr} head={head_lr}")

    def _encode(self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None):
        outputs = self.wav2vec2(
            input_values=input_features,
            attention_mask=attention_mask,
            output_hidden_states=self.LLA,
        )
        if self.LLA:
            hidden_states = torch.stack(outputs.hidden_states, dim=0)
            normalized_weights = F.softmax(self.layer_weights, dim=-1).view(-1, 1, 1, 1)
            encoded = (hidden_states * normalized_weights).sum(dim=0)
            return encoded, normalized_weights
        return outputs.last_hidden_state, None

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None, task: str = "phon"):
        encoded, normalized_weights = self._encode(input_features=input_features, attention_mask=attention_mask)
        if task == "phon":
            logits = self.phoneme_head(encoded)
        elif task == "word":
            logits = self.word_head(encoded)
        else:
            raise ValueError(f"Unknown task '{task}'. Expected 'phon' or 'word'.")
        return F.log_softmax(logits, dim=-1), logits, normalized_weights

    def decode(self, logits: torch.Tensor, task: str) -> list[str]:
        if task == "phon":
            if self.phon_decoder is None:
                raise ValueError("phon_decoder is not set")
            return self.phon_decoder(logits)
        if task == "word":
            if self.word_decoder is None:
                raise ValueError("word_decoder is not set")
            return self.word_decoder(logits)
        raise ValueError(f"Unknown task '{task}'.")

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        def _conv_out(length, kernel_size, stride):
            return (length - kernel_size) // stride + 1

        base_config = self.wav2vec2.config
        for k, s in zip(base_config.conv_kernel, base_config.conv_stride):
            input_lengths = _conv_out(input_lengths, k, s)
        return input_lengths.to(torch.long)
