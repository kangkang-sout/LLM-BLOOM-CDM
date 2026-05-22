from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SCDM(nn.Module):
    """Simple Cognitive Diagnosis Model.

    This is a lightweight baseline that keeps the same input contract as
    NCDM, making it convenient for quick comparisons.
    """

    def __init__(
        self,
        num_students: int,
        num_items: int,
        num_knowledge: int,
    ) -> None:
        super().__init__()
        self.student_mastery = nn.Embedding(num_students, num_knowledge)
        self.item_difficulty = nn.Embedding(num_items, num_knowledge)
        self.item_discrimination = nn.Embedding(num_items, 1)
        self.output_bias = nn.Embedding(num_items, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.student_mastery.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_difficulty.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_discrimination.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.output_bias.weight)

    def predict_logits(
        self,
        student_ids: torch.Tensor,
        item_ids: torch.Tensor,
        knowledge_mask: torch.Tensor,
    ) -> torch.Tensor:
        knowledge_mask = knowledge_mask.float()

        mastery = torch.sigmoid(self.student_mastery(student_ids))
        difficulty = torch.sigmoid(self.item_difficulty(item_ids))
        discrimination = F.softplus(self.item_discrimination(item_ids)).squeeze(-1) + 1e-6
        bias = self.output_bias(item_ids).squeeze(-1)

        signal = ((mastery - difficulty) * knowledge_mask).sum(dim=-1)
        logits = discrimination * signal + bias
        return logits

    def forward(
        self,
        student_ids: torch.Tensor,
        item_ids: torch.Tensor,
        knowledge_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_logits(student_ids, item_ids, knowledge_mask)

    def predict_proba(
        self,
        student_ids: torch.Tensor,
        item_ids: torch.Tensor,
        knowledge_mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.predict_logits(student_ids, item_ids, knowledge_mask))

    def loss(
        self,
        student_ids: torch.Tensor,
        item_ids: torch.Tensor,
        knowledge_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.predict_logits(student_ids, item_ids, knowledge_mask)
        return F.binary_cross_entropy_with_logits(logits, labels.float())
