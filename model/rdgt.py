from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class RDGT(nn.Module):
    """Response-Driven Graph Tracing model for cognitive diagnosis."""

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

        self.update_scale = nn.Parameter(torch.ones(num_knowledge))
        self.trace_mixer = nn.Sequential(
            nn.Linear(num_knowledge, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_knowledge),
        )
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
        nn.init.ones_(self.update_scale)

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

    def _trace_state(
        self,
        history_correctness: Optional[torch.Tensor],
        history_knowledge_mask: Optional[torch.Tensor],
        knowledge_graph: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if history_correctness is None or history_knowledge_mask is None:
            return torch.zeros(batch_size, self.num_knowledge, device=device)

        if history_knowledge_mask.size(1) == 0:
            return torch.zeros(batch_size, self.num_knowledge, device=device)

        adjacency = self._normalize_graph(knowledge_graph, device)
        state = torch.zeros(batch_size, self.num_knowledge, device=device)

        for step in range(history_knowledge_mask.size(1)):
            q_mask = history_knowledge_mask[:, step, :].float()
            outcome = history_correctness[:, step].float().unsqueeze(-1)
            signed_update = q_mask * (outcome * 2.0 - 1.0) * self.update_scale.unsqueeze(0)

            propagated_state = torch.matmul(state, adjacency)
            state = torch.tanh(self.trace_mixer(propagated_state) + signed_update)

        return state

    def predict_logits(
        self,
        student_ids: torch.Tensor,
        item_ids: torch.Tensor,
        knowledge_mask: torch.Tensor,
        history_correctness: Optional[torch.Tensor] = None,
        history_knowledge_mask: Optional[torch.Tensor] = None,
        knowledge_graph: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        knowledge_mask = knowledge_mask.float()
        device = knowledge_mask.device
        batch_size = student_ids.size(0)

        traced_state = self._trace_state(
            history_correctness,
            history_knowledge_mask,
            knowledge_graph,
            batch_size,
            device,
        )
        adjacency = self._normalize_graph(knowledge_graph, device)
        propagated_question = torch.matmul(knowledge_mask, adjacency)

        mastery = torch.sigmoid(self.student_mastery(student_ids) + traced_state)
        difficulty = torch.sigmoid(self.item_difficulty(item_ids))
        discrimination = F.softplus(self.item_discrimination(item_ids)).squeeze(-1) + 1e-6

        interaction = (mastery - difficulty) * propagated_question
        logits = self.prediction(discrimination.unsqueeze(-1) * interaction).squeeze(-1)
        return logits

    def forward(
        self,
        student_ids: torch.Tensor,
        item_ids: torch.Tensor,
        knowledge_mask: torch.Tensor,
        history_correctness: Optional[torch.Tensor] = None,
        history_knowledge_mask: Optional[torch.Tensor] = None,
        knowledge_graph: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.predict_logits(
            student_ids,
            item_ids,
            knowledge_mask,
            history_correctness,
            history_knowledge_mask,
            knowledge_graph,
        )

    def predict_proba(
        self,
        student_ids: torch.Tensor,
        item_ids: torch.Tensor,
        knowledge_mask: torch.Tensor,
        history_correctness: Optional[torch.Tensor] = None,
        history_knowledge_mask: Optional[torch.Tensor] = None,
        knowledge_graph: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.predict_logits(
                student_ids,
                item_ids,
                knowledge_mask,
                history_correctness,
                history_knowledge_mask,
                knowledge_graph,
            )
        )

    def loss(
        self,
        student_ids: torch.Tensor,
        item_ids: torch.Tensor,
        knowledge_mask: torch.Tensor,
        labels: torch.Tensor,
        history_correctness: Optional[torch.Tensor] = None,
        history_knowledge_mask: Optional[torch.Tensor] = None,
        knowledge_graph: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.predict_logits(
            student_ids,
            item_ids,
            knowledge_mask,
            history_correctness,
            history_knowledge_mask,
            knowledge_graph,
        )
        return F.binary_cross_entropy_with_logits(logits, labels.float())
