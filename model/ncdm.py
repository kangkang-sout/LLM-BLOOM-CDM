from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NCDM(nn.Module):
    """Neural Cognitive Diagnosis Model.

    The model expects `student_ids` and `item_ids` to be remapped to
    contiguous 0-based indices. `knowledge_mask` should be a multi-hot tensor
    with shape `[batch_size, num_knowledge]`.
    """

    def __init__(
        self,
        num_students: int,
        num_items: int,
        num_knowledge: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_knowledge = num_knowledge

        self.student_mastery = nn.Embedding(num_students, num_knowledge)
        self.item_difficulty = nn.Embedding(num_items, num_knowledge)
        self.item_discrimination = nn.Embedding(num_items, 1)

        self.prediction = nn.Sequential(
            nn.Linear(num_knowledge, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.student_mastery.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_difficulty.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_discrimination.weight, mean=0.0, std=0.1)

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

        interaction = (mastery - difficulty) * knowledge_mask
        logits = self.prediction(discrimination.unsqueeze(-1) * interaction).squeeze(-1)
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
