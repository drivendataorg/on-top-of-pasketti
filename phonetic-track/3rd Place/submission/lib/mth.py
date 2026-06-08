import numpy as np
import torch
import librosa
import torch.nn as nn
from pathlib import Path
from loguru import logger
from transformers import (
    Wav2Vec2BertProcessor, 
    Wav2Vec2FeatureExtractor, 
    AutoConfig,
    AutoModel,
    WavLMModel,
    Wav2Vec2BertModel,
    PretrainedConfig,
    PreTrainedModel
)

from safetensors.torch import load_file
from transformers.models.wavlm.modeling_wavlm import WavLMBaseModelOutput
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertBaseModelOutput, Wav2Vec2BertAdapter

class LayerDropController:
    def __init__(self, layerdrop_prob=0.0):
        self.layerdrop_prob = layerdrop_prob
        self.keep_adapter = True

    def update(self):
        self.keep_adapter = torch.rand(1).item() >= self.layerdrop_prob

class CustomWavLMModel(WavLMModel):
    def __init__(self, config):
        super().__init__(config)
        self.controller = None
        config.adapter_act = getattr(config, "adapter_act", "relu")
        config.adapter_layers = getattr(config, "adapter_layers", 1)
        config.adapter_stride = getattr(config, "adapter_stride", 2)
        config.conformer_conv_dropout = getattr(config, "conformer_conv_dropout", 0.1)
        config.adapter_kernel_size = getattr(config, "adapter_kernel_size", 3)
        self.adapter = Wav2Vec2BertAdapter(config)
    
    def set_controller(self, controller):
        self.controller = controller

    def forward(
            self,
            input_values: torch.Tensor | None,
            attention_mask: torch.Tensor | None = None,
            mask_time_indices: torch.FloatTensor | None = None,
            output_attentions: bool | None = None,
            output_hidden_states: bool | None = None,
            return_dict: bool | None = None,
            **kwargs,
        ) -> tuple | WavLMBaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None and (not self.training or self.controller.keep_adapter):
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return WavLMBaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CustomWav2Vec2BertModel(Wav2Vec2BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.controller = None

    def set_controller(self, controller):
        self.controller = controller

    def forward(
            self,
            input_features: torch.Tensor | None,
            attention_mask: torch.Tensor | None = None,
            mask_time_indices: torch.FloatTensor | None = None,
            output_attentions: bool | None = None,
            output_hidden_states: bool | None = None,
            return_dict: bool | None = None,
            **kwargs,
        ) -> tuple | Wav2Vec2BertBaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states, extract_features = self.feature_projection(input_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.intermediate_ffn:
            expanded_hidden_states = self.intermediate_ffn(hidden_states)
            hidden_states = hidden_states + 0.5 * expanded_hidden_states

        if self.adapter is not None and (not self.training or self.controller.keep_adapter):
            hidden_states = self.adapter(hidden_states, attention_mask=attention_mask)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BertBaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    
class MultiTaskHybridConfig(PretrainedConfig):
    model_type = "multi_task_hybrid"
    def __init__(self, phoneme_vocab_size=100, word_vocab_size=100, pad_token_id=0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.phoneme_vocab_size = phoneme_vocab_size
        self.word_vocab_size = word_vocab_size
        self.pad_token_id = pad_token_id
        self.alpha = alpha


class MultiTaskHybridModel(PreTrainedModel):
    config_class = MultiTaskHybridConfig

    def __init__(self, config, bert_name, wavlm_name, controller):
        super().__init__(config)
        self.controller = controller
        
        # Encoders
        bert_config = AutoConfig.from_pretrained(bert_name / "config.json", add_adapter=True, num_adapter_layers=1)
        wavlm_config = AutoConfig.from_pretrained(wavlm_name / "config.json", add_adapter=True, num_adapter_layers=1)
        self.bert = CustomWav2Vec2BertModel(bert_config)
        self.wavlm = CustomWavLMModel(wavlm_config)
        self.bert.set_controller(controller)
        self.wavlm.set_controller(controller)
        self.wavlm.freeze_feature_encoder()

        # Fusion Bridge
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
    

class MultiTaskHybridInferenceModel:
    def __init__(self, model, bert_processor, wavlm_processor):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.bert_processor = bert_processor
        self.wavlm_processor = wavlm_processor

    @classmethod
    def load(cls, 
             checkpoint_path: str, 
             bert_local_path: str, 
             wavlm_local_path: str, 
             controller):
        
        logger.info(f"Loading Hybrid Model from: {checkpoint_path}")
        
        config = MultiTaskHybridConfig.from_pretrained(checkpoint_path)
        
        model = MultiTaskHybridModel(
            config=config,
            bert_name=bert_local_path,
            wavlm_name=wavlm_local_path,
            controller=controller
        )
        
        safetensors_path = Path(checkpoint_path) / "model.safetensors"
        state_dict = load_file(str(safetensors_path), device="cpu")
        
        model.load_state_dict(state_dict, strict=True)
        
        bert_processor = Wav2Vec2BertProcessor.from_pretrained(bert_local_path)
        bert_processor.tokenizer.pad_token_id = 0
        bert_processor.tokenizer.pad_token = "[PAD]"
        wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(wavlm_local_path)
        
        return cls(model, bert_processor, wavlm_processor)

    def predict(self, audio_path: Path):
        transcriptions, scores = self.predict_batch([audio_path], batch_size=1)
        return transcriptions[0], scores[0]

    def predict_batch(self, audio_paths: list[Path], batch_size: int = 4):        
        audio_inputs = []
        for p in audio_paths:
            speech, _ = librosa.load(str(p), sr=16000)
            audio_inputs.append(speech)

        bert_inputs = self.bert_processor(
            audio_inputs, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        wavlm_inputs = self.wavlm_processor(
            audio_inputs,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_features=bert_inputs.input_features, 
                input_values=wavlm_inputs.input_values
            )
            logits = outputs["phoneme_logits"]


        predicted_ids = torch.argmax(logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        selected_log_probs = torch.gather(log_probs, dim=-1, index=predicted_ids.unsqueeze(-1)).squeeze(-1)

        mask = (predicted_ids != self.bert_processor.tokenizer.pad_token_id).float()

        scores = []
        for i in range(selected_log_probs.shape[0]):
            total_log_prob = (selected_log_probs[i] * mask[i]).sum().item()            
            token_count = mask[i].sum().item()
            
            if token_count > 0:
                mean_log_prob = total_log_prob / token_count
                score = np.exp(mean_log_prob) 
            else:
                score = 0.0
                
            scores.append(score)

        transcriptions = self.bert_processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcriptions, scores
    


