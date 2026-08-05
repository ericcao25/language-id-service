"""
Model architecture, copied from train.py.

This must match the training script exactly — the checkpoints are raw
state_dicts, not HuggingFace save_pretrained() output, so the service has to
reconstruct this exact class before calling load_state_dict().
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model


class Wav2Vec2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = getattr(config, "pooling_mode", "mean")
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.init_weights()

    @property
    def all_tied_weights_keys(self):
        # Compatibility shim for some transformers versions — without this,
        # newer transformers' init_weights()/tie_weights() call chain raises
        # AttributeError on this custom model class. Present in the original
        # train.py; dropped by mistake in an earlier version of this file.
        keys = getattr(self, "_tied_weights_keys", None)
        if keys is None:
            return {}
        if isinstance(keys, dict):
            return keys
        return {k: True for k in keys}

    def merged_strategy(self, hidden_states, attention_mask=None, mode="mean"):
        if mode == "mean":
            if attention_mask is None:
                return hidden_states.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            return summed / denom
        elif mode == "sum":
            if attention_mask is None:
                return hidden_states.sum(dim=1)
            mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
            return (hidden_states * mask).sum(dim=1)
        elif mode == "max":
            if attention_mask is None:
                return hidden_states.max(dim=1)[0]
            mask = attention_mask.unsqueeze(-1).bool()
            neg_inf = torch.finfo(hidden_states.dtype).min
            masked = hidden_states.masked_fill(~mask, neg_inf)
            return masked.max(dim=1)[0]
        raise ValueError("pooling_mode must be one of ['mean','sum','max']")

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        pooled = self.merged_strategy(hidden_states, attention_mask, self.pooling_mode)
        logits = self.classifier(pooled)
        return logits
