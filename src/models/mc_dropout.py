"""
Monte Carlo Dropout wrapper: run T forward passes with dropout enabled and compute
predictive mean μ and variance σ². Use these as uncertainty signals for OOD and calibration.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MCOutput:
    logits: torch.Tensor
    mean_probs: torch.Tensor
    var_probs: torch.Tensor
    mean_positive: torch.Tensor
    var_positive: torch.Tensor
    var_logit_positive: torch.Tensor  # variance of positive-class logit across MC samples (often more discriminative for OOD)


class MCDropoutWrapper(nn.Module):
    """
    Stochastic inference wrapper around a classifier that outputs logits.
    Keeps dropout active for T forward passes and returns:
      - mean prediction (μ)
      - predictive variance (σ²) per class / for positive class
    """

    def __init__(self, model: nn.Module, num_samples: int = 30):
        super().__init__()
        self.model = model
        self.num_samples = num_samples

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """
        If labels provided, run standard forward (one pass, for training).
        If no labels, run MC Dropout inference: T passes, return mean logits and variance.
        """
        if labels is not None:
            return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # MC Dropout inference: keep dropout active
        was_training = self.model.training
        self.model.train()
        logits_list = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits if hasattr(out, "logits") else out[0]
                logits_list.append(logits)
        if not was_training:
            self.model.eval()

        # Stack: (T, B, C)
        logits_stack = torch.stack(logits_list, dim=0)
        probs_stack = F.softmax(logits_stack, dim=-1)

        # Predictive mean and variance (per class, then we often care about positive class)
        mean_probs = probs_stack.mean(dim=0)  # (B, C)
        var_probs = probs_stack.var(dim=0)   # (B, C)

        # For binary: variance of P(positive) and of positive-class logit
        if mean_probs.size(-1) == 2:
            mean_pos = mean_probs[:, 1]
            var_pos = var_probs[:, 1]
            var_logit_pos = logits_stack[:, :, 1].var(dim=0)  # (B,)
        else:
            mean_pos = mean_probs
            var_pos = var_probs
            var_logit_pos = logits_stack.var(dim=0).mean(dim=-1)

        return MCOutput(
            logits=logits_stack.mean(dim=0),
            mean_probs=mean_probs,
            var_probs=var_probs,
            mean_positive=mean_pos,
            var_positive=var_pos,
            var_logit_positive=var_logit_pos,
        )

    def predict_with_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience: return (mean_probs, mean_positive, var_positive) for evaluation.
        """
        out = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return out.mean_probs, out.mean_positive, out.var_positive
