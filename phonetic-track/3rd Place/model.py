import torch
import torch.nn as nn
from transformers import WavLMModel, Wav2Vec2BertModel
from transformers.models.wavlm.modeling_wavlm import WavLMBaseModelOutput
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertBaseModelOutput, Wav2Vec2BertAdapter


class LayerDropController:
    def __init__(self, layerdrop_prob=0.1):
        self.layerdrop_prob = layerdrop_prob
        self.keep_adapter = True

    def update(self):
        self.keep_adapter = torch.rand(1).item() > self.layerdrop_prob


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