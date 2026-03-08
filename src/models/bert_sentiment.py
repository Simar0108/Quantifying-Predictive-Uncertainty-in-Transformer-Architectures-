"""
BERT-based binary sentiment classifier for SST-2.
Uses HuggingFace BertForSequenceClassification; same weights used for baseline and MC Dropout.
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, BertForSequenceClassification


def get_bert_sentiment(
    model_name: str = "bert-base-uncased",
    num_labels: int = 2,
    dropout_prob: float = 0.1,
) -> BertForSequenceClassification:
    """Load BERT with sequence classification head (for SST-2)."""
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, hidden_dropout_prob=dropout_prob)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    return model


class BERTSentimentClassifier(nn.Module):
    """
    Thin wrapper around HuggingFace BertForSequenceClassification so we have
    a consistent interface (forward returns logits; we can swap in MC wrapper).
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.backbone = get_bert_sentiment(model_name=model_name, num_labels=num_labels, dropout_prob=dropout_prob)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out  # loss, logits, ...

    @property
    def config(self):
        return self.backbone.config
