import torch
import torch.nn as nn
from transformers import (
    BertPreTrainedModel,
    BertModel
)
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict
from transformers.modeling_outputs import ModelOutput


@dataclass
class MultiHeadTokenClassifierOutput(ModelOutput):
  
    loss: Optional[torch.FloatTensor] = None
    
    loss_edits: Optional[torch.FloatTensor] = None
    loss_areta13: Optional[torch.FloatTensor] = None
    loss_areta43: Optional[torch.FloatTensor] = None

    logits_edits: torch.FloatTensor = None
    logits_areta13: torch.FloatTensor = None
    logits_areta43: torch.FloatTensor = None

    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class BertForTokenMultiLabelClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [ r'pooler' ]

    def __init__(self, config, edits_class_weights=None, areta13_class_weights=None, areta43_class_weights=None):
        super().__init__(config)
        
        self.edits_num_labels = config.edits_num_labels
        
        self.areta13_num_labels = config.areta13_num_labels
        
        self.areta43_num_labels = config.areta43_num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        
        self.dropout = nn.Dropout(classifier_dropout)
        
        self.edits_classifier = nn.Linear(config.hidden_size, config.edits_num_labels)
        
        self.areta13_classifier = nn.Linear(config.hidden_size, config.areta13_num_labels)
        
        self.areta43_classifier = nn.Linear(config.hidden_size, config.areta43_num_labels)
        
        
        
        self.loss_weight_edits = float(getattr(config, "loss_weight_edits", 1.0))
        
        self.loss_weight_areta13 = float(getattr(config, "loss_weight_areta13", 1.0))
        
        self.loss_weight_areta43 = float(getattr(config, "loss_weight_areta43", 1.0))
        
        self.label_smoothing = float(getattr(config, "label_smoothing", 0.0))
        
        self.ignore_index = getattr(config, "ignore_index", -100)

        
        # TODO: FIND A BETTER WAY
        
        self.edits_class_weights = edits_class_weights if edits_class_weights is not None else None
        
        self.areta13_class_weights = areta13_class_weights if areta13_class_weights is not None else None
        
        self.areta43_class_weights = areta43_class_weights if areta43_class_weights is not None else None
        
        # Initialize weights and apply final processing
        self.post_init()

    def _head_forward(self, sequence_output, classifier):
        x = self.dropout(sequence_output)
        return classifier(x)

    def _loss(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        class_weights: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if labels is None:
            return None
        # CE with ignore_index ensures padding/special tokens don't contribute.
        loss_fct = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=self.ignore_index, label_smoothing=self.label_smoothing
        )
        # logits: (B, L, C) -> (B*L, C); labels: (B, L) -> (B*L)
        return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        edits_labels: Optional[torch.Tensor] = None,
        areta13_labels: Optional[torch.Tensor] = None,
        areta43_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, MultiHeadTokenClassifierOutput]:

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state  # (B, L, H)

        # Per-head logits
        logits_edits = self._head_forward(sequence_output, self.edits_classifier)
        logits_areta13 = self._head_forward(sequence_output, self.areta13_classifier)
        logits_areta43 = self._head_forward(sequence_output, self.areta43_classifier)

        # Per-head losses (computed only if labels are provided)
        loss_edits = self._loss(logits_edits, edits_labels, self.edits_class_weights)
        loss_areta13 = self._loss(logits_areta13, areta13_labels, self.areta13_class_weights)
        loss_areta43 = self._loss(logits_areta43, areta43_labels, self.areta43_class_weights)

        # Weighted sum (only for heads that had labels)
        losses = []
        if loss_edits is not None:
            losses.append(self.loss_weight_edits * loss_edits)
        if loss_areta13 is not None:
            losses.append(self.loss_weight_areta13 * loss_areta13)
        if loss_areta43 is not None:
            losses.append(self.loss_weight_areta43 * loss_areta43)

        total_loss = sum(losses) if losses else None

        if not return_dict:
            output = (
                logits_edits,
                logits_areta13,
                logits_areta43,
                outputs.hidden_states,
                outputs.attentions,
            )
            return ((total_loss,) + output) if total_loss is not None else output

        return MultiHeadTokenClassifierOutput(
            loss=total_loss,
            loss_edits=loss_edits,
            loss_areta13=loss_areta13,
            loss_areta43=loss_areta43,
            logits_edits=logits_edits,
            logits_areta13=logits_areta13,
            logits_areta43=logits_areta43,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

