#! /usr/bin/env python

import torch
from torch.nn import Identity
import torchvision.models as models
from transformers import ElectraModel, ElectraConfig
from typing import Optional

class SequenceSummary(torch.nn.Module):
    r"""
    Compute a single vector summary of a sequence hidden states.
    Args:
        config ([`PretrainedConfig`]):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):
            - **summary_type** (`str`) -- The method to use to make this summary. Accepted values are:
                - `"last"` -- Take the last token hidden state (like XLNet)
                - `"first"` -- Take the first token hidden state (like Bert)
                - `"mean"` -- Take the mean of all tokens hidden states
                - `"cls_index"` -- Supply a Tensor of classification token position (GPT/GPT-2)
                - `"attn"` -- Not implemented now, use multi-head attention
            - **summary_use_proj** (`bool`) -- Add a projection after the vector extraction.
            - **summary_proj_to_labels** (`bool`) -- If `True`, the projection outputs to `config.num_labels` classes
              (otherwise to `config.hidden_size`).
            - **summary_activation** (`Optional[str]`) -- Set to `"tanh"` to add a tanh activation to the output,
              another string or `None` will add no activation.
            - **summary_first_dropout** (`float`) -- Optional dropout probability before the projection and activation.
            - **summary_last_dropout** (`float`)-- Optional dropout probability after the projection and activation.
    """

    def __init__(self, config):
        super().__init__()

        self.summary_type = getattr(config, "summary_type", "last")

        self.summary = Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = torch.nn.Linear(config.hidden_size, num_classes)

        self.activation = torch.nn.GELU()

        self.first_dropout = Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = torch.nn.Dropout(config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = torch.nn.Dropout(config.summary_last_dropout)

    def forward(
        self, hidden_states: torch.FloatTensor, cls_index: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        """
        Compute a single vector summary of a sequence hidden states.
        Args:
            hidden_states (`torch.FloatTensor` of shape `[batch_size, seq_len, hidden_size]`):
                The hidden states of the last layer.
            cls_index (`torch.LongTensor` of shape `[batch_size]` or `[batch_size, ...]` where ... are optional leading dimensions of `hidden_states`, *optional*):
                Used if `summary_type == "cls_index"` and takes the last token of the sequence as classification token.
        Returns:
            `torch.FloatTensor`: The summary of the sequence hidden states.
        """
        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = hidden_states.mean(dim=1)
        elif self.summary_type == "cls_index":
            if cls_index is None:
                cls_index = torch.full_like(
                    hidden_states[..., :1, :],
                    hidden_states.shape[-2] - 1,
                    dtype=torch.long,
                )
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


class ElectraMulti(torch.nn.Module):
    def __init__(self):

        super(ElectraMulti, self).__init__()

        electra_base = "google/electra-base-discriminator"
        electra_large = "google/electra-large-discriminator"
        self.electra = ElectraModel.from_pretrained(electra_large)
        self.sequence_summary = SequenceSummary(self.electra.config)
        self.classifier_qa = torch.nn.Linear(self.electra.config.hidden_size, 1)
        self.classifier_complexity = torch.nn.Linear(self.electra.config.hidden_size, 3)


    def forward(self, input_ids, attention_mask, token_type_ids):

        num_choices = input_ids.shape[1] 

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) 
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        discriminator_hidden_states = self.electra(input_ids, attention_mask, token_type_ids)

        sequence_output = discriminator_hidden_states[0]

        pooled_output = self.sequence_summary(sequence_output)

        logits_qa = self.classifier_qa(pooled_output)
        logits_qa = logits_qa.view(-1, num_choices)

        logits_complexity = self.classifier_complexity(pooled_output)
        logits_complexity = logits_complexity.view(-1, num_choices, 3)
        logits_complexity = torch.mean(logits_complexity, -2)

        return logits_qa, logits_complexity