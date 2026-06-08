import sys
from pathlib import Path

from src.preprocessing.phoneme_tokenizer import PhonemeTokenizer
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# print(f"Project root: {PROJECT_ROOT}")
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.utils import import_utils as transformers_import_utils

# This repo uses text/audio backbones only, but recent transformers imports can
# eagerly touch torchvision through shared image utilities. In this environment,
# torchvision is installed without a working NMS operator, which breaks even
# pure ASR model loads. Mark torchvision unavailable before loading HF models.
transformers_import_utils._torchvision_available = False

from transformers import AutoConfig, AutoModel
from transformers.utils.hub import cached_file
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.utils.decoder import GreedyDecoder
from src.preprocessing.dataset import prepare_dl_dataset
from src.utils.hf_local import resolve_hf_load_path

class WhisperEncoderCTC(nn.Module):
    """
    Model will take log-Mel spectrograms into the Whisper encoder, 
    take the raw acoustic representations (the last_hidden_state),
    and feed them to a head, 
    where the final layer is a Linear Layer that projects to the token phoneme vocabulary.
    """
    def __init__(
        self, 
        vocab_size: int, 
        whisper_model_id: str = "openai/whisper-tiny", 
        freeze_encoder: bool = False,
        classifier_dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        interpolate_positional_embeddings: bool = True,
        decoder = None,
        augmentation: DictConfig = None,
        cfg: DictConfig = None,
        vocab: PhonemeTokenizer = None,
        inference: bool = False,
        backbone_lr: float | None = None,
        head_lr: float | None = None,
        enable_age_head: bool = False,
        age_head_lambda: float = 0.1,
    ):
        super().__init__()
        from transformers import WhisperModel
        
        self.cfg = cfg
        self.vocab = vocab
        self.decoder = decoder
        self.LLA = False
        self.use_lora = False
        self.gradient_checkpointing = gradient_checkpointing
        self.interpolate_positional_embeddings = interpolate_positional_embeddings
        
        self.enable_age_head = enable_age_head
        self.age_head_lambda = age_head_lambda
        
        mask_cfg = augmentation.get("masking", None) if augmentation is not None else None
        self.specaug_time_prob = 0.0
        self.specaug_feature_prob = 0.0
        self.specaug_time_mask_length = int(mask_cfg.get("mask_time_length", 10)) if mask_cfg is not None else 10
        self.specaug_feature_mask_length = int(mask_cfg.get("mask_feature_length", 10)) if mask_cfg is not None else 10
        if mask_cfg is not None and mask_cfg.get("enabled", False):
            self.specaug_time_prob = float(mask_cfg.get("mask_time_prob", 0.0))
            self.specaug_feature_prob = float(mask_cfg.get("mask_feature_prob", 0.0))

        load_path, local_files_only = resolve_hf_load_path(whisper_model_id, inference=inference)
        if inference:
            if load_path != whisper_model_id:
                print(f"Loading Whisper model in INFERENCE mode from local path: {load_path}")
            else:
                print(
                    f"Local Whisper model path not found for '{whisper_model_id}'; "
                    "trying local Hugging Face cache only."
                )

        # Load the base model and extract just the encoder
        base_model = WhisperModel.from_pretrained(load_path, local_files_only=local_files_only)
        self.encoder = base_model.encoder
        
        # Freeze the encoder weights if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            # Whisper keeps positional embeddings frozen by default; explicitly
            # unfreeze them so "full fine-tuning" really means the full encoder.
            self.encoder.embed_positions.weight.requires_grad = True
                
        # Get hidden size from the loaded encoder's config
        hidden_size = self.encoder.config.d_model
        
        # CTC head: a small MLP is a bit more expressive than a single linear layer
        # while still keeping the same decoder / loss interface as Wav2Vec2.
        self.ctc_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size, vocab_size),
        )

        if self.enable_age_head:
            bottleneck_dim = hidden_size // 4
            self.age_head = nn.Sequential(
                nn.Linear(hidden_size, bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(bottleneck_dim, 4)
            )

        backbone_total = sum(p.numel() for p in self.encoder.parameters())
        backbone_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head_total = sum(p.numel() for p in self.ctc_head.parameters())
        head_trainable = sum(p.numel() for p in self.ctc_head.parameters() if p.requires_grad)
        header = f"{'Component':<18} | {'Total Params':>15} | {'Trainable':>12} | {'Base LR':>10}"
        print(f"\n[Model Architecture: WhisperEncoderCTC - Backbone: {whisper_model_id}]")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        print(
            f"{'Backbone':<18} | {backbone_total:>15,} | {backbone_trainable:>12,} | {str(backbone_lr if backbone_lr is not None else 'N/A'):>10}"
        )
        print(
            f"{'CTC Head':<18} | {head_total:>15,} | {head_trainable:>12,} | {str(head_lr if head_lr is not None else 'N/A'):>10}"
        )
        print("=" * len(header) + "\n")

    def _compute_mask_indices(
        self,
        batch_size: int,
        axis_size: int,
        mask_prob: float,
        mask_length: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if mask_prob <= 0.0 or mask_length <= 0 or axis_size <= 0:
            return None

        mask_length = min(mask_length, axis_size)
        expected_num_masks = mask_prob * axis_size / max(mask_length, 1)
        num_masks = int(expected_num_masks + torch.rand((), device=device).item())
        if num_masks <= 0:
            return None

        max_start = axis_size - mask_length + 1
        starts = torch.randint(0, max_start, (batch_size, num_masks), device=device)
        span_offsets = torch.arange(mask_length, device=device)
        spans = starts.unsqueeze(-1) + span_offsets

        mask = torch.zeros(batch_size, axis_size, dtype=torch.bool, device=device)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, num_masks * mask_length)
        mask[batch_indices, spans.reshape(batch_size, -1)] = True
        return mask

    def _apply_specaugment(self, input_features: torch.Tensor) -> torch.Tensor:
        augmented = input_features.clone()
        batch_size, feature_size, time_size = augmented.shape

        time_mask = self._compute_mask_indices(
            batch_size=batch_size,
            axis_size=time_size,
            mask_prob=self.specaug_time_prob,
            mask_length=self.specaug_time_mask_length,
            device=augmented.device,
        )
        if time_mask is not None:
            augmented = augmented.masked_fill(time_mask.unsqueeze(1), 0.0)

        feature_mask = self._compute_mask_indices(
            batch_size=batch_size,
            axis_size=feature_size,
            mask_prob=self.specaug_feature_prob,
            mask_length=self.specaug_feature_mask_length,
            device=augmented.device,
        )
        if feature_mask is not None:
            augmented = augmented.masked_fill(feature_mask.unsqueeze(-1), 0.0)

        return augmented

    def _position_embeddings(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        embed_pos = self.encoder.embed_positions.weight.to(device=device, dtype=dtype)
        if seq_len <= embed_pos.shape[0]:
            return embed_pos[:seq_len]

        if not self.interpolate_positional_embeddings:
            raise ValueError(
                f"Whisper received {seq_len} encoder frames, but the pretrained positional "
                f"embeddings only cover {embed_pos.shape[0]}. Enable interpolation or reduce "
                "preprocessing.max_duration_sec."
            )

        resized = F.interpolate(
            embed_pos.transpose(0, 1).unsqueeze(0),
            size=seq_len,
            mode="linear",
            align_corners=False,
        )
        return resized.squeeze(0).transpose(0, 1)

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Runs Whisper's encoder manually so we can support long clips and keep a
        plain CTC interface for phoneme decoding.

        Args:
            input_features: Log-Mel spectrograms [B, 80, T_frames].
            attention_mask: Ignored; CTC uses explicit sequence lengths instead.
        """
        if self.training and (self.specaug_time_prob > 0.0 or self.specaug_feature_prob > 0.0):
            input_features = self._apply_specaugment(input_features)

        # 1. Manually run the convolutional stem
        inputs_embeds = F.gelu(self.encoder.conv1(input_features))
        inputs_embeds = F.gelu(self.encoder.conv2(inputs_embeds))
        
        # 2. Permute to match transformer expectations: [Batch, Seq_Len, Hidden_Size]
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        
        # 3. Handle long clips by interpolating positional embeddings when needed.
        seq_len = inputs_embeds.shape[1]
        embed_pos = self._position_embeddings(seq_len, inputs_embeds.device, inputs_embeds.dtype)
        
        # 4. Add positional embeddings and dropout
        hidden_states = inputs_embeds + embed_pos.unsqueeze(0)
        hidden_states = F.dropout(hidden_states, p=self.encoder.dropout, training=self.training)
        
        # 5. Manually pass through the transformer layers so the encoder can stay
        # compatible with arbitrary feature lengths.
        for layer in self.encoder.layers:
            if self.gradient_checkpointing and self.training:
                def layer_forward(states: torch.Tensor, layer_module=layer) -> torch.Tensor:
                    return layer_module(
                        states,
                        attention_mask=None,
                        layer_head_mask=None,
                    )[0]

                hidden_states = checkpoint(
                    layer_forward,
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=None,
                    layer_head_mask=None,
                )
                hidden_states = layer_outputs[0]
            
        # 6. Apply the final layer norm from the encoder
        last_hidden_state = self.encoder.layer_norm(hidden_states)
        
        # 7. Project to vocab and return both log-probs and raw logits so the
        # trainer can compute loss and decode exactly like the Wav2Vec2 path.
        logits = self.ctc_head(last_hidden_state)
        log_probs = F.log_softmax(logits, dim=-1)

        age_logits = None
        if getattr(self, "enable_age_head", False):
            pooled_states = last_hidden_state.mean(dim=1)
            age_logits = self.age_head(pooled_states)

        return log_probs, logits, None, age_logits

    def decode(self, logits: torch.Tensor) -> list[str]:
        return self.decoder(logits)

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute encoder lengths after Whisper's convolutional stem."""

        def _conv_out(length: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
            kernel_size = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
            stride = conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride
            padding = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
            dilation = conv.dilation[0] if isinstance(conv.dilation, tuple) else conv.dilation
            return torch.div(
                length + 2 * padding - dilation * (kernel_size - 1) - 1,
                stride,
                rounding_mode="floor",
            ) + 1

        output_lengths = _conv_out(input_lengths, self.encoder.conv1)
        output_lengths = _conv_out(output_lengths, self.encoder.conv2)
        return output_lengths.to(torch.long)


class Wav2Vec2CTC(nn.Module):
    """
        Wav2Vec2 for phonetic transcription using CTC.

    Architecture:
            - Optional LoRA adapters on attention + FFN layers
            - Otherwise full backbone fine-tuning
            - Custom: 2-layer MLP classification head → vocab_size logits
    """

    def __init__(
        self,
        vocab_size: int,
        pretrained_name: str = "facebook/wav2vec2-base",
        classifier_dropout: float = 0.1,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        decoder = None,
        augmentation: DictConfig = None,
        gradient_checkpointing: bool | None = None,
        backbone_lr: float | None = None,
        head_lr: float | None = None,
        vocab: PhonemeTokenizer = None,
        inference: bool = False,
        LLA: bool = False,
        enable_age_head: bool = False,
        age_head_lambda: float = 0.1,
    ):
        super().__init__()
        self.vocab = vocab
        self.decoder = decoder
        self.LLA = LLA
        self.use_lora = use_lora
        self.enable_age_head = enable_age_head
        self.age_head_lambda = age_head_lambda

        # SpecAugment-style masking for Wav2Vec2 (applied during training)

        aug_cfg = augmentation
        mask_cfg = aug_cfg.get("masking", None) if aug_cfg is not None else None
        if mask_cfg is not None and mask_cfg.get("enabled", False):
            mask_time_prob = float(mask_cfg.get("mask_time_prob", 0.05))
            mask_time_length = int(mask_cfg.get("mask_time_length", 10))
            mask_feature_prob = float(mask_cfg.get("mask_feature_prob", 0.0))
            mask_feature_length = int(mask_cfg.get("mask_feature_length", 10))

        def _load_backbone_config(model_id: str, local_only: bool):
            if Path(model_id).exists():
                config_path = Path(model_id) / "config.json"
            else:
                config_path = cached_file(model_id, "config.json", local_files_only=local_only)
            config = AutoConfig.from_pretrained(str(config_path))
            config.gradient_checkpointing = False
            if hasattr(config, "mask_time_prob"):
                config.mask_time_prob = mask_time_prob
            if hasattr(config, "mask_feature_prob"):
                config.mask_feature_prob = mask_feature_prob
            if hasattr(config, "mask_feature_length"):
                config.mask_feature_length = mask_feature_length
            if hasattr(config, "mask_time_length"):
                config.mask_time_length = mask_time_length
            return config

        # Load pretrained backbone
        if inference:
            local_model_path = Path(__file__).parent.parent.parent / "external" / pretrained_name.split("/")[-1]
            print(f"looking for {local_model_path}")
            load_path = str(local_model_path)
            print(f"found {load_path}")
            config = _load_backbone_config(load_path, local_only=True)
            # print(f"Loading Wav2Vec2 model in INFERENCE mode from: {load_path}")
            self.wav2vec2 = AutoModel.from_pretrained(
                load_path,
                local_files_only=True,
                config=config,
            )
        else:
            # print(f"Loading Wav2Vec2 model in TRAINING mode from Hugging Face: {pretrained_name}")
            config = _load_backbone_config(pretrained_name, local_only=False)
            self.wav2vec2 = AutoModel.from_pretrained(
                pretrained_name,
                config=config,
            )
        if gradient_checkpointing is True:
            self.wav2vec2.gradient_checkpointing_enable()
        elif gradient_checkpointing is False:
            self.wav2vec2.gradient_checkpointing_disable()

        hidden_size = self.wav2vec2.config.hidden_size
        num_states = self.wav2vec2.config.num_hidden_layers + 1
        mu = 0.70 * (num_states - 1)
        indices = torch.arange(num_states, dtype=torch.float32)

        sigma = 0.15 * num_states
        
        # Calculate the Gaussian curve: A * exp(-0.5 * ((x - mu) / sigma)^2)
        # We multiply by a scalar (e.g., 3.0) to make the peak pronounced before softmax is applied
        initial_logits = 3.0 * torch.exp(-0.5 * ((indices - mu) / sigma) ** 2)
        
        # Assign this curve to our learnable parameter
        self.layer_weights = nn.Parameter(initial_logits)
        # # Prevent PEFT / gradient-checkpointing issues
        # # (Wav2Vec2Model has no token embeddings)
        # self.wav2vec2.supports_gradient_checkpointing = False
        # self.wav2vec2.config.gradient_checkpointing = False
        self.wav2vec2.enable_input_require_grads = lambda *a, **kw: None

        if self.use_lora:
            from peft import LoraConfig, get_peft_model

            # Inject LoRA adapters — freezes all base params automatically
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=[
                    "q_proj", "v_proj", "k_proj", "out_proj",
                    "intermediate_dense", "output_dense",
                ],
                bias="none",
            )
            self.wav2vec2 = get_peft_model(self.wav2vec2, lora_cfg)
            # self.wav2vec2.print_trainable_parameters()

        # 2-layer MLP classification head
        self.ctc_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size, vocab_size),
        )

        if self.enable_age_head:
            bottleneck_dim = hidden_size // 4
            self.age_head = nn.Sequential(
                nn.Linear(hidden_size, bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(bottleneck_dim, 4)
            )

        # Anti-blank bias: discourage predicting blank early in training
        with torch.no_grad():
            self.ctc_head[-1].bias[0] = -2.0

        # --- Parameter Statistics Calculation ---
        # 1. Identify trainable params in the backbone
        lora_params = sum(p.numel() for n, p in self.wav2vec2.named_parameters() if "lora_" in n)
        backbone_total = sum(p.numel() for p in self.wav2vec2.parameters())
        backbone_trainable = sum(p.numel() for p in self.wav2vec2.parameters() if p.requires_grad)
        backbone_frozen = backbone_total - backbone_trainable

        # 2. Head params (fully trainable)
        head_total = sum(p.numel() for p in self.ctc_head.parameters())
        head_trainable = sum(p.numel() for p in self.ctc_head.parameters() if p.requires_grad)
        head_frozen = head_total - head_trainable

        # 3. Grand Totals
        total_all = backbone_total + head_total
        total_trainable = backbone_trainable + head_trainable
        total_frozen = backbone_frozen + head_frozen

        # 4. Fetch LRs from config (defaulting to "N/A" if not found)
        backbone_lr = backbone_lr if backbone_lr is not None else "N/A"
        head_lr = head_lr if head_lr is not None else "N/A"

        backbone_name = "Backbone (LoRA)" if self.use_lora else "Backbone (Full FT)"
        backbone_lr_display = backbone_lr
        backbone_trainable_display = lora_params if self.use_lora else backbone_trainable

        # --- Table Formatting ---
        table_data = [
            [backbone_name, f"{backbone_total:,}", f"{backbone_trainable_display:,}", f"{backbone_frozen:,}", f"{backbone_lr_display}"],
            ["CTC Head",        f"{head_total:,}",     f"{head_trainable:,}", f"{head_frozen:,}",   f"{head_lr}"],
            ["---", "---", "---", "---", "---"],
            ["TOTAL",           f"{total_all:,}",      f"{total_trainable:,}", f"{total_frozen:,}", ""],
        ]

        header = f"{'Component':<18} | {'Total Params':>15} | {'Trainable':>12} | {'Frozen':>12} | {'Base LR':>10}"
        print(f"\n[Model Architecture: Wav2Vec2CTC - Backbone: {pretrained_name}]")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for row in table_data:
            if row[0] == "---":
                print("-" * len(header))
                continue
            print(f"{row[0]:<18} | {row[1]:>15} | {row[2]:>12} | {row[3]:>12} | {row[4]:>10}")
        print("=" * len(header) + "\n")

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_features: Raw waveforms [B, T_samples].
            attention_mask:  Binary mask   [B, T_samples] (1 = real, 0 = pad).
        Returns:
            log_probs: [B, T_enc, vocab_size]
        """
        outputs = self.wav2vec2(
            input_values=input_features,
            attention_mask=attention_mask,
            output_hidden_states=self.LLA,
        )
        if self.LLA:
            hidden_states = outputs.hidden_states
            stacked_states = torch.stack(hidden_states, dim=0)
            
            normalized_weights = F.softmax(self.layer_weights, dim=-1)
            normalized_weights = normalized_weights.view(-1, 1, 1, 1)
            weighted_sum_state = (stacked_states * normalized_weights).sum(dim=0)
            logits = self.ctc_head(weighted_sum_state)
        else:
            logits = self.ctc_head(outputs.last_hidden_state)
            normalized_weights = None
            last_hidden_state = outputs.last_hidden_state
            
        age_logits = None
        if getattr(self, "enable_age_head", False):
            if self.LLA:
                pooled_states = weighted_sum_state.mean(dim=1)
            else:
                pooled_states = last_hidden_state.mean(dim=1)
            age_logits = self.age_head(pooled_states)

        return F.log_softmax(logits, dim=-1), logits, normalized_weights, age_logits

    def decode(self, logits: torch.Tensor) -> list[str]:
        """Decode raw logits to phonetic strings using the provided decoder."""
        return self.decoder(logits)
    
    
    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute encoder output lengths after the multi-stage CNN feature extractor."""

        def _conv_out(length, kernel_size, stride):
            return (length - kernel_size) // stride + 1

        if hasattr(self.wav2vec2, "base_model") and hasattr(self.wav2vec2.base_model, "model"):
            base_config = self.wav2vec2.base_model.model.config
        else:
            base_config = self.wav2vec2.config
        for k, s in zip(base_config.conv_kernel, base_config.conv_stride):
            input_lengths = _conv_out(input_lengths, k, s)
        return input_lengths.to(torch.long)


# --- Debugging Block ---
if __name__ == "__main__":
    import os
    from tqdm import tqdm
    from omegaconf import OmegaConf
    from src.utils.score import score_ipa_cer

    # --- Configuration ---
    RUN_DIR = PROJECT_ROOT / "outputs/2026-02-27/23-50-46_thieving-mini_fridge-39"
    CHECKPOINT_PATH = RUN_DIR / "fold_1/best_model.pth"
    DEVICE = "cuda:0"

    # --- 1. Load config & prepare data ---
    cfg = OmegaConf.load(str(RUN_DIR / ".hydra" / "config.yaml"))
    print(cfg)
    # Patch old configs that don't have decoder nested inside model
    # os.chdir(PROJECT_ROOT)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, val_loader, dataset_info = prepare_dl_dataset(cfg, fold=0)
    tokenizer = dataset_info["tokenizer"]

    # --- 2. Instantiate model & load checkpoint ---
    model = instantiate(cfg.model, vocab_size=dataset_info["vocab_size"], vocab=tokenizer)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}  "
          f"(epoch {checkpoint.get('epoch', '?')}, "
          f"val_loss {checkpoint.get('val_loss', '?'):.4f})")

    # --- 3. Decode only the samples from the first batch ---
    # decoder = GreedyDecoder(blank_token_id=tokenizer.blank_token_id)
    all_preds_logits, all_preds_decoded, all_refs = [], [], []

    with torch.no_grad():

        first_batch = next(iter(val_loader))
        # for batch in tqdm(val_loader, desc="Decoding"):
        input_features = first_batch["input_features"].to(device)
        attention_mask = first_batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        output = model(input_features, attention_mask=attention_mask)
        _, logits, _ = output
        decoded = model.decode(logits)
        all_preds_logits.extend(logits.cpu())
        all_preds_decoded.extend(decoded)

        # Decode ground-truth labels (ignore pad tokens)
        for lbl in first_batch["labels"]:
            ids = lbl[lbl != tokenizer.pad_token_id].tolist()
            all_refs.append(tokenizer.decode(ids))
    print(f"Logits preds shape: {len(all_preds_logits)}, Decoded preds shape: {len(all_preds_decoded)}, Refs shape: {len(all_refs)}")
    # --- 4. Score with IPA-CER ---
    ipa_cer = score_ipa_cer(actual=all_refs, predicted=all_preds_decoded)
    print(f"\nIPA-CER for the first batch: {ipa_cer:.4f}")

    # Show a few sample predictions
    print("\n--- Sample predictions ---")
    for i in range(min(10, len(all_preds_decoded))):
        print(f"  REF:  {all_refs[i]}")
        print(f"  PRED: {all_preds_decoded[i]}")
        print()
