from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class GCDM(nn.Module):
    """Graph-based Cognitive Diagnosis Model."""

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
        self.concept_embedding = nn.Embedding(num_knowledge, hidden_dim)

        self.gcn_1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn_2 = nn.Linear(hidden_dim, hidden_dim)
        self.graph_gate = nn.Linear(hidden_dim, num_knowledge)

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
        nn.init.normal_(self.concept_embedding.weight, mean=0.0, std=0.1)

    def _normalize_graph(
        self,
        knowledge_graph: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if knowledge_graph is None:
            graph = torch.eye(self.num_knowledge, device=device)
        else:
            graph = knowledge_graph.float().to(device)
            graph = graph + torch.eye(self.num_knowledge, device=device)

        degree = graph.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return graph / degree

    def _encode_concepts(self, knowledge_graph: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
        adjacency = self._normalize_graph(knowledge_graph, device)
        concept_states = self.concept_embedding.weight

        hidden = torch.relu(self.gcn_1(adjacency @ concept_states))
        hidden = self.gcn_2(adjacency @ hidden)
        return hidden

    def predict_logits(
        self,
        student_ids: torch.Tensor,
        item_ids: torch.Tensor,
        knowledge_mask: torch.Tensor,
        knowledge_graph: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        knowledge_mask = knowledge_mask.float()
        device = knowledge_mask.device

        mastery = torch.sigmoid(self.student_mastery(student_ids))
        difficulty = torch.sigmoid(self.item_difficulty(item_ids))
        discrimination = F.softplus(self.item_discrimination(item_ids)).squeeze(-1) + 1e-6

        concept_states = self._encode_concepts(knowledge_graph, device)
        normalized_mask = knowledge_mask / knowledge_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        question_context = normalized_mask @ concept_states
        graph_gate = torch.sigmoid(self.graph_gate(question_context))

        interaction = (mastery - difficulty) * knowledge_mask * graph_gate
        logits = self.prediction(discrimination.unsqueeze(-1) * interaction).squeeze(-1)
        return logits

    def forward(
        self,
        student_ids: torch.Tensor,
        item_ids: torch.Tensor,
        knowledge_mask: torch.Tensor,
        knowledge_graph: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.predict_logits(student_ids, item_ids, knowledge_mask, knowledge_graph)

    def predict_proba(
        self,
        student_ids: torch.Tensor,
        item_ids: torch.Tensor,
        knowledge_mask: torch.Tensor,
        knowledge_graph: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.predict_logits(student_ids, item_ids, knowledge_mask, knowledge_graph)
        )

    def loss(
        self,
        student_ids: torch.Tensor,
        item_ids: torch.Tensor,
        knowledge_mask: torch.Tensor,
        labels: torch.Tensor,
        knowledge_graph: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.predict_logits(student_ids, item_ids, knowledge_mask, knowledge_graph)
        return F.binary_cross_entropy_with_logits(logits, labels.float())
