import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Union, Any
from jiwer import cer, wer
import torch.nn.functional as F

from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2BertProcessor,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    SeamlessM4TFeatureExtractor,
    Trainer,
    TrainingArguments,
    PretrainedConfig,
    PreTrainedModel
)

from audiomentations import Compose, PitchShift, Gain, AddBackgroundNoise, PolarityInversion

from model import CustomWav2Vec2BertModel, CustomWavLMModel, LayerDropController

# --- Configuration ---
SAMPLING_RATE = 16000
DATA_DIR = Path("data/")
NOISE_DIR = DATA_DIR / "noise_16k"
BERT_MODEL_NAME = "facebook/w2v-bert-2.0"
WAVLM_MODEL_NAME = "microsoft/wavlm-large"

# Training Hyperparameters
EPOCHS = 5
LR = 2e-5
BS = 2
ACCU = 4
MAX_DURATION = 30

ALPHA = 0.8  # Weight for Phoneme and Word Consistency Loss
CR_LOSS_SCALE = 0.2 # Weight for CTC and 4-way Consistency Loss
BB_DROP = 0.25 # Backbone Drop probability

MASK_TIME_PROB = 0.3
MASK_FREQ_PROB = 0.05
MIN_SNR_DB=2
MAX_SNR_DB=10

LOGGING_STEPS = 200

# MODE = "all" # All labelled data
MODE = "all07" # All labelled data + Pseudo IPA label with CER <= 0.7

if MODE == "all07":
    TRAIN_DATA = DATA_DIR / "phon_train_all_filtered_pseudo07.json" 
    EVAL_DATA = DATA_DIR / "val_f0_phon_filtered.json"
else:
    TRAIN_DATA = DATA_DIR / "phon_train_all_filtered.json"
    EVAL_DATA = DATA_DIR / "val_f0_phon_filtered.json"

OUTPUT_DIR = f"./MTH_{MODE}_{EPOCHS}ep_{ALPHA}alpha_4w_mt{MASK_TIME_PROB}_mf{MASK_FREQ_PROB}_snr{MIN_SNR_DB}{MAX_SNR_DB}"
VOCAB_PATH = "ipa_vocab.json"
WORD_VOCAB_PATH = "word_vocab.json"


class MultiTaskHybridConfig(PretrainedConfig):
    model_type = "multi_task_hybrid"
    def __init__(self, phoneme_vocab_size=100, word_vocab_size=100, pad_token_id=0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.phoneme_vocab_size = phoneme_vocab_size
        self.word_vocab_size = word_vocab_size
        self.pad_token_id = pad_token_id
        self.alpha = alpha

class MultiTaskHybridWavBertModel(PreTrainedModel):
    config_class = MultiTaskHybridConfig

    def __init__(self, config, bert_name, wavlm_name, controller):
        super().__init__(config)
        self.controller = controller
        
        # Encoders
        self.bert = CustomWav2Vec2BertModel.from_pretrained(
            bert_name, add_adapter=True, num_adapter_layers=1, layerdrop=0.0,
            mask_time_prob=MASK_TIME_PROB,
            mask_feature_prob=MASK_FREQ_PROB,
        )
        self.wavlm = CustomWavLMModel.from_pretrained(
            wavlm_name, add_adapter=True, num_adapter_layers=1, layerdrop=0.0,
            mask_time_prob=MASK_TIME_PROB,
            mask_feature_prob=MASK_FREQ_PROB,
        )
        self.bert.set_controller(controller)
        self.wavlm.set_controller(controller)
        self.wavlm.freeze_feature_encoder()

        self.bridge_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Multi-Task Heads
        self.phoneme_head = nn.Linear(1024, config.phoneme_vocab_size)
        self.word_head = nn.Linear(1024, config.word_vocab_size)
        
        self.loss_fct = nn.CTCLoss(blank=config.pad_token_id, zero_infinity=True, reduction="sum")

    def forward(self, input_features, input_values, phoneme_labels=None, word_labels=None, **kwargs):
        self.controller.update()
        
        bert_out = self.bert(input_features).last_hidden_state
        wavlm_out = self.wavlm(input_values).last_hidden_state

        # Stochastic Backbone Dropping
        if self.training:
            prob = torch.rand(1, device=bert_out.device)
            if prob <= BB_DROP:
                bert_out = torch.zeros_like(bert_out)
            elif prob >= (1 - BB_DROP):
                wavlm_out = torch.zeros_like(wavlm_out)

        combined = torch.cat([bert_out, wavlm_out], dim=-1)
        features = self.bridge_layer(combined)

        phoneme_logits = self.phoneme_head(features)
        word_logits = self.word_head(features)

        loss = None
        if phoneme_labels is not None and word_labels is not None:
            input_lengths = torch.full((phoneme_logits.size(0),), phoneme_logits.size(1), dtype=torch.long)
            
            # Phoneme Loss
            log_probs_p = F.log_softmax(phoneme_logits, dim=-1).transpose(0, 1)
            p_lengths = (phoneme_labels != -100).sum(dim=-1)
            loss_p = self.loss_fct(log_probs_p, phoneme_labels.masked_fill(phoneme_labels == -100, 0), input_lengths, p_lengths)

            # Word Loss
            log_probs_w = F.log_softmax(word_logits, dim=-1).transpose(0, 1)
            w_lengths = (word_labels != -100).sum(dim=-1)
            loss_w = self.loss_fct(log_probs_w, word_labels.masked_fill(word_labels == -100, 0), input_lengths, w_lengths)

            loss = self.config.alpha * loss_p + (1 - self.config.alpha) * loss_w

        return {
            "loss": loss, 
            "phoneme_logits": phoneme_logits, 
            "word_logits": word_logits
        }


@dataclass
class HybridMultiTaskCollator:
    model: MultiTaskHybridWavBertModel
    bert_processor: Wav2Vec2BertProcessor
    wavlm_processor: Wav2Vec2Processor
    phoneme_tokenizer: Wav2Vec2CTCTokenizer
    word_tokenizer: Wav2Vec2CTCTokenizer

    augmenter = Compose([
        PitchShift(min_semitones=-5.0, max_semitones=5.0, p=1),
        Gain(min_gain_db=-6, max_gain_db=6, p=1),
        AddBackgroundNoise(sounds_path=[str(NOISE_DIR)], min_snr_db=MIN_SNR_DB, max_snr_db=MAX_SNR_DB, p=1)
    ])

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raw_audios = [f["input_features"] for f in features]
        p_labels = [{"input_ids": f["phoneme_labels"]} for f in features]
        w_labels = [{"input_ids": f["word_labels"]} for f in features]

        if self.model.training:
            augmented = [self.augmenter(samples=np.array(a), sample_rate=16000) for a in raw_audios]
            raw_audios = raw_audios + augmented
            p_labels = p_labels + p_labels
            w_labels = w_labels + w_labels

        batch_bert = self.bert_processor(raw_audios, sampling_rate=16000, return_tensors="pt", padding=True)
        batch_wavlm = self.wavlm_processor(raw_audios, sampling_rate=16000, return_tensors="pt", padding=True)
        p_batch = self.phoneme_tokenizer.pad(p_labels, return_tensors="pt")
        w_batch = self.word_tokenizer.pad(w_labels, return_tensors="pt")

        return {
            "input_features": batch_bert["input_features"],
            "input_values": batch_wavlm["input_values"],
            "attention_mask": batch_bert["attention_mask"],
            "phoneme_labels": p_batch["input_ids"].masked_fill(p_batch.attention_mask.ne(1), -100),
            "word_labels": w_batch["input_ids"].masked_fill(w_batch.attention_mask.ne(1), -100),
        }

def compute_metrics(pred, phoneme_tokenizer, word_tokenizer):
    p_logits, w_logits = pred.predictions
    p_labels, w_labels = pred.label_ids

    # Decode Phonemes
    p_ids = np.argmax(p_logits, axis=-1)
    p_labels[p_labels == -100] = phoneme_tokenizer.pad_token_id
    p_cer = cer(phoneme_tokenizer.batch_decode(p_labels, group_tokens=False), phoneme_tokenizer.batch_decode(p_ids))
    
    # Decode Words
    w_ids = np.argmax(w_logits, axis=-1)
    w_labels[w_labels == -100] = word_tokenizer.pad_token_id
    gold_w = word_tokenizer.batch_decode(w_labels, group_tokens=False)
    pred_w = [s if s.strip() != "" else "[EMPTY]" for s in word_tokenizer.batch_decode(w_ids)]
    
    return {"p_cer": p_cer, "w_cer": cer(gold_w, pred_w), "w_wer": wer(gold_w, pred_w)}
    

def make_pad_mask(lengths, max_len):
    batch_size = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    return seq_range.unsqueeze(0).expand(batch_size, max_len) >= lengths.unsqueeze(-1)

def compute_kl_loss(p, q, mask=None):
    p_log = F.log_softmax(p, dim=-1)
    q_prob = F.softmax(q, dim=-1)
    q_log = F.log_softmax(q, dim=-1)
    p_prob = F.softmax(p, dim=-1)
    
    p_loss = F.kl_div(p_log, q_prob, reduction="none")
    q_loss = F.kl_div(q_log, p_prob, reduction="none")
    
    if mask is not None:
        p_loss.masked_fill_(mask, 0.)
        q_loss.masked_fill_(mask, 0.)
    return (p_loss.sum() + q_loss.sum()) / 2

class MultiTaskHybridTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not model.training:
            outputs = model(**inputs)
            return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

        inputs_4n = {k: torch.cat([v, v], dim=0) for k, v in inputs.items()}
        outputs = model(**inputs_4n)
        
        p_logits, w_logits = outputs["phoneme_logits"], outputs["word_logits"]
        n = p_logits.shape[0] // 4
        seq_len = p_logits.shape[1]

        att_mask = inputs["attention_mask"][:n]
        new_lengths = (att_mask.sum(dim=-1).float() / att_mask.shape[1] * seq_len).round().long()
        pad_mask = make_pad_mask(new_lengths, seq_len).unsqueeze(-1)


        def get_consistency(logits):
            a1, a2 = logits[0:n], logits[n:2*n]
            b1, b2 = logits[2*n:3*n], logits[3*n:4*n]
            l_cr = (compute_kl_loss(a1, b1, pad_mask) + compute_kl_loss(a2, b2, pad_mask)) / 2
            l_aug = (compute_kl_loss(a1, a2, pad_mask) + compute_kl_loss(b1, b2, pad_mask)) / 2
            return (l_cr + l_aug) / 2

        p_cons = get_consistency(p_logits)
        w_cons = get_consistency(w_logits)
        
        total_cons = (self.model.config.alpha * p_cons) + ((1 - self.model.config.alpha) * w_cons)
        loss = outputs["loss"] + CR_LOSS_SCALE * total_cons
        
        return (loss, outputs) if return_outputs else loss

# --- 4. Main Script ---

def build_tokenizers():
    # Phoneme Vocab
    ipa = ['[PAD]', '[UNK]', '|','b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z', 'æ', 'ç', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɚ', 'ɛ', 'ɟ', 'ɪ', 'ɫ', 'ɬ', 'ɹ', 'ɾ', 'ʁ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʔ', 'ʝ', 'ʤ', 'ʧ', 'ː', 'θ', 'χ']
    with open(VOCAB_PATH, "w") as f: json.dump({s: i for i, s in enumerate(ipa)}, f)
    
    # Word Vocab
    words = ['[PAD]', '[UNK]', '|', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    with open(WORD_VOCAB_PATH, "w") as f: json.dump({s: i for i, s in enumerate(words)}, f)
    
    return Wav2Vec2CTCTokenizer(VOCAB_PATH, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"), \
           Wav2Vec2CTCTokenizer(WORD_VOCAB_PATH, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

def main():
    p_tok, w_tok = build_tokenizers()
        
    bert_fe = SeamlessM4TFeatureExtractor.from_pretrained(BERT_MODEL_NAME)
    wavlm_fe = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL_NAME)
    
    bert_proc = Wav2Vec2BertProcessor(feature_extractor=bert_fe, tokenizer=p_tok)
    wavlm_proc = Wav2Vec2Processor(feature_extractor=wavlm_fe, tokenizer=p_tok)


    dataset = load_dataset("json", data_files={"train": str(DATA_DIR / TRAIN_DATA), "eval": str(DATA_DIR / EVAL_DATA)})
    dataset = dataset.filter(lambda x: x["duration"] <= MAX_DURATION)
    dataset = dataset.map(lambda x: {"audio_filepath": str(DATA_DIR / x["audio_filepath"])})
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=SAMPLING_RATE))

    def transform(batch):
        batch["input_features"] = batch["audio_filepath"]["array"]
        batch["phoneme_labels"] = p_tok(batch["text"]).input_ids
        batch["word_labels"] = w_tok(batch["word"]).input_ids
        return batch

    dataset = dataset.map(transform, remove_columns=dataset["train"].column_names)

    controller = LayerDropController(layerdrop_prob=0.1)
    config = MultiTaskHybridConfig(phoneme_vocab_size=len(p_tok), word_vocab_size=len(w_tok), pad_token_id=p_tok.pad_token_id, alpha=ALPHA)
    model = MultiTaskHybridWavBertModel(config, BERT_MODEL_NAME, WAVLM_MODEL_NAME, controller)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BS,
        gradient_accumulation_steps=ACCU,
        learning_rate=LR,
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        metric_for_best_model="eval_p_cer",
        greater_is_better=False,
        save_total_limit=2,     
        gradient_checkpointing=False,
        report_to="none",
        load_best_model_at_end=True,
    )

    trainer = MultiTaskHybridTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=HybridMultiTaskCollator(model, bert_proc, wavlm_proc, p_tok, w_tok),
        compute_metrics=lambda p: compute_metrics(p, p_tok, w_tok),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()
