"""PyTorch Lightning module for fine-tuning Qwen3-ASR."""
from pathlib import Path

import jiwer
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from loguru import logger
from peft import LoraConfig, get_peft_model
from qwen_asr import Qwen3ASRModel
from whisper_normalizer.english import EnglishTextNormalizer
import librosa

from qwen_asr.inference.utils import detect_and_fix_repetitions
from src.data.augment import build_augment
from src.models.ragged_audio_tower import patch_audio_tower
from src.paths import RAW_AUDIO_DIR, TARGET_SR

normalize = EnglishTextNormalizer()


def max_new_tokens_for_duration(max_dur_sec):
    if max_dur_sec < 1.5:
        return 16
    elif max_dur_sec < 3:
        return 24
    elif max_dur_sec < 5:
        return 32
    elif max_dur_sec < 10:
        return 48
    elif max_dur_sec < 15:
        return 72
    elif max_dur_sec < 30:
        return 128
    else:
        return 192

SHORT_PROMPT = (
    "A child's speech assessment. The child is saying one or two words. "
    "Common words: chair, beard, nurse, ladder, fork, hammer, scarf, sword, dark, weird, clear, "
    "fish, duck, tiger, rabbit, frog, monkey, elephant, butterfly, giraffe, "
    "red, blue, green, yellow, cup, ring, star, door, flower, slide, truck, guitar, drum, "
    "pajamas, umbrella, vegetable, telephone, helicopter, abracadabra, "
    "finger, ear, teeth, five, seven, ten, three, one, two, four."
)
LONG_PROMPT = "A child speaking in English."
SHORT_DURATION_THRESHOLD = 2.0


def build_text_prompt(context=""):
    if context:
        return (
            f"<|im_start|>system\n{context}<|im_end|>\n"
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\nlanguage English<asr_text>"
        )
    return (
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\nlanguage English<asr_text>"
    )


class QwenASRDataset(torch.utils.data.Dataset):

    def __init__(self, entries, processor, text_prompt, use_prompts=False, augment_cfg=None):
        self.entries = [e for e in entries if e["audio_duration_sec"] < 30]
        self.processor = processor
        self.text_prompt = text_prompt
        self.use_prompts = use_prompts
        self.augment = build_augment(augment_cfg) if augment_cfg else None
        if use_prompts:
            self.short_prompt = build_text_prompt(SHORT_PROMPT)
            self.long_prompt = build_text_prompt(LONG_PROMPT)
        self.asr_text_id = processor.tokenizer.convert_tokens_to_ids("<asr_text>")
        self.im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        ap = entry["audio_path"]
        audio_path = Path(ap) if Path(ap).is_absolute() else RAW_AUDIO_DIR.parent / ap
        try:
            audio, sr = librosa.load(audio_path, sr=16000, dtype="float32", mono=True)
            if self.augment is not None:
                audio = self.augment(samples=audio, sample_rate=16000)

            if self.use_prompts:
                dur = entry.get("audio_duration_sec", len(audio) / 16000)
                prompt = self.short_prompt if dur < SHORT_DURATION_THRESHOLD else self.long_prompt
            else:
                prompt = self.text_prompt

            text = entry["orthographic_text"]
            full_text = prompt + text + "<|im_end|>"

            inputs = self.processor(text=[full_text], audio=[audio], return_tensors="pt", padding=False)

            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)
            input_features = inputs["input_features"].squeeze(0)
            feature_attention_mask = inputs["feature_attention_mask"].squeeze(0)

            labels = input_ids.clone()
            asr_pos = (input_ids == self.asr_text_id).nonzero(as_tuple=True)[0][-1].item()
            labels[:asr_pos + 1] = -100

            return input_ids, attention_mask, input_features, feature_attention_mask, labels, audio, text
        except Exception as e:
            uid = entry.get("utterance_id", audio_path.stem)
            logger.warning(f"Skipping {uid}: {e}")
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    input_ids, attention_masks, input_features, feature_attention_masks, labels, audio, raw_texts = zip(*batch)
    max_ids_len = max(x.shape[0] for x in input_ids)
    max_feat_len = max(x.shape[1] for x in input_features)

    pad_token_id = 151643  # Qwen pad token

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_input_features = []
    batch_feature_attention_mask = []

    for ids, mask, feats, feat_mask, lab, _, _ in batch:
        ids_pad = max_ids_len - ids.shape[0]
        feat_pad = max_feat_len - feats.shape[1]

        batch_input_ids.append(F.pad(ids, (ids_pad, 0), value=pad_token_id))
        batch_attention_mask.append(F.pad(mask, (ids_pad, 0), value=0))
        batch_labels.append(F.pad(lab, (ids_pad, 0), value=-100))
        batch_input_features.append(F.pad(feats, (0, feat_pad), value=0.0))
        batch_feature_attention_mask.append(F.pad(feat_mask, (0, feat_pad), value=0))

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "input_features": torch.stack(batch_input_features),
        "feature_attention_mask": torch.stack(batch_feature_attention_mask),
        "labels": torch.stack(batch_labels),
        "audio": audio,
        "raw_texts": list(raw_texts),
    }


class QwenASRModule(pl.LightningModule):

    def __init__(self, model_name="Qwen/Qwen3-ASR-1.7B", lr=1e-5,
                 weight_decay=1e-2, warmup=0.1, freeze_audio_encoder=False,
                 freeze_lm_layers=0, batch_audio_tower=True,
                 lora_rank=0, lora_alpha=16, lora_target_modules=None,
                 kl_alpha=0.0, kl_alpha_min=0.0):
        super().__init__()
        self.save_hyperparameters()

        wrapper = Qwen3ASRModel.from_pretrained(
            model_name, device_map="cuda", max_new_tokens=192, dtype=torch.bfloat16,
        )
        self.thinker = wrapper.model.thinker
        self.processor = wrapper.processor
        self.text_prompt = wrapper._build_text_prompt(context="", force_language="English")
        self.asr_text_id = self.processor.tokenizer.convert_tokens_to_ids("<asr_text>")
        self.thinker.generation_config = wrapper.model.generation_config
        del wrapper

        if lora_rank > 0:
            for p in self.thinker.parameters():
                p.requires_grad = False
            target = lora_target_modules or ["q_proj", "v_proj"]
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target,
                lora_dropout=0.05,
                bias="none",
            )
            self.thinker = get_peft_model(self.thinker, lora_config)
            logger.info(f"LoRA rank={lora_rank} alpha={lora_alpha} targets={target}")
            self.thinker.print_trainable_parameters()
        else:
            self.thinker.train()

            if batch_audio_tower:
                patch_audio_tower(self.thinker)
                logger.info("Using ragged batched audio tower")

            if freeze_audio_encoder:
                for p in self.thinker.audio_tower.parameters():
                    p.requires_grad = False
                logger.info("Froze audio encoder")

            if freeze_lm_layers > 0:
                for layer in self.thinker.model.layers[:freeze_lm_layers]:
                    for p in layer.parameters():
                        p.requires_grad = False
                logger.info(f"Froze first {freeze_lm_layers} LM layers")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params")

        if kl_alpha > 0 and lora_rank == 0:
            import copy
            self._base_thinker = copy.deepcopy(self.thinker)
            self._base_thinker.eval()
            for p in self._base_thinker.parameters():
                p.requires_grad = False
            logger.info("Created frozen base model copy for KL distillation")

        self._val_preds = []
        self._val_refs = []

    def apply_lora(self, rank=16, alpha=32, target_modules=None):
        for p in self.thinker.parameters():
            p.requires_grad = False
        target = target_modules or ["q_proj", "v_proj"]
        lora_config = LoraConfig(
            r=rank, lora_alpha=alpha, target_modules=target,
            lora_dropout=0.05, bias="none",
        )
        self.thinker = get_peft_model(self.thinker, lora_config)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"LoRA applied: rank={rank} alpha={alpha} targets={target}")
        logger.info(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params")

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.startswith("_base_thinker.")}

    def forward(self, batch):
        return self.thinker(
            input_ids=batch["input_ids"],
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask"],
            feature_attention_mask=batch["feature_attention_mask"],
            labels=batch["labels"],
        )

    def _get_base_logits(self, batch):
        if hasattr(self.thinker, 'disable_adapter_layers'):
            self.thinker.disable_adapter_layers()
            base_out = self.forward(batch)
            self.thinker.enable_adapter_layers()
            return base_out.logits
        elif hasattr(self, '_base_thinker'):
            base_out = self._base_thinker(
                input_ids=batch["input_ids"],
                input_features=batch["input_features"],
                attention_mask=batch["attention_mask"],
                feature_attention_mask=batch["feature_attention_mask"],
                labels=batch["labels"],
            )
            return base_out.logits
        return None

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        ce_loss = out.loss

        kl_alpha_max = self.hparams.kl_alpha
        kl_alpha_min = self.hparams.kl_alpha_min
        if kl_alpha_max > 0:
            import math
            progress = self.global_step / max(self.trainer.estimated_stepping_batches, 1)
            kl_alpha = kl_alpha_min + (kl_alpha_max - kl_alpha_min) * 0.5 * (1 + math.cos(math.pi * progress))
            with torch.no_grad():
                base_logits = self._get_base_logits(batch)

            if base_logits is not None:
                labels = batch["labels"]
                mask = labels != -100
                ft_log_probs = F.log_softmax(out.logits[mask], dim=-1)
                base_probs = F.softmax(base_logits[mask], dim=-1)
                kl_loss = F.kl_div(ft_log_probs, base_probs, reduction="batchmean")

                loss = ce_loss + kl_alpha * kl_loss
                self.log("train_ce", ce_loss, prog_bar=False)
                self.log("train_kl", kl_loss, prog_bar=True)
                self.log("kl_alpha", kl_alpha, prog_bar=False)
                self.log("train_loss", loss, prog_bar=True)
                return loss

        self.log("train_loss", ce_loss, prog_bar=True)
        return ce_loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        self.log("val_loss", out.loss, prog_bar=True, sync_dist=True)
        prompt_inputs = self.processor(
            text=[self.text_prompt] * len(batch["audio"]),
            audio=list(batch["audio"]),
            return_tensors="pt",
            padding=True,
        )
        prompt_inputs = {
            k: v.to(device=self.device, dtype=self.dtype) if v.is_floating_point()
            else v.to(device=self.device)
            for k, v in prompt_inputs.items()
        }
        prompt_len = prompt_inputs["input_ids"].shape[1]
        max_dur = max(len(a) / 16000 for a in batch["audio"])
        max_tokens = max_new_tokens_for_duration(max_dur)

        with torch.no_grad():
            generated = self.thinker.generate(
                **prompt_inputs,
                max_new_tokens=max_tokens,
            )
        pred_texts = [
            detect_and_fix_repetitions(t) for t in
            self.processor.batch_decode(
                generated[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ]

        self._val_preds.extend(pred_texts)
        self._val_refs.extend(batch["raw_texts"])

    def on_validation_epoch_end(self):
        if not self._val_refs:
            return
        raw_wer = jiwer.wer(self._val_refs, self._val_preds)
        norm_refs = [normalize(r) for r in self._val_refs]
        norm_preds = [normalize(p) for p in self._val_preds]
        norm_wer = jiwer.wer(norm_refs, norm_preds)
        self.log("val_wer", raw_wer, prog_bar=True)
        self.log("val_wer_norm", norm_wer, prog_bar=True)
        logger.info(f"val_wer={raw_wer:.4f}  val_wer_norm={norm_wer:.4f}  ({len(self._val_refs)} samples)")
        self._val_preds.clear()
        self._val_refs.clear()

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.98),
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup = self.hparams.warmup
        pct_start = warmup if warmup < 1 else warmup / total_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.hparams.lr, total_steps=total_steps,
            pct_start=pct_start, anneal_strategy="cos",
            div_factor=10, final_div_factor=10,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
