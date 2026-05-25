"""Learned fiber lift."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from hgraphml.graph import EdgeFiber
from hgraphml.lifts.uniform import _validate_lift_inputs


class LearnedFiberLift(nn.Module):
    """Learn fine messages from coarse messages and fine-edge context."""

    def __init__(
        self,
        *,
        message_dim: int,
        context_dim: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if message_dim <= 0:
            raise ValueError("message_dim must be positive.")
        if context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.message_dim = message_dim
        self.context_dim = context_dim
        score_input_dim = message_dim + context_dim
        decoder_input_dim = message_dim + context_dim + 1
        self.score_mlp = nn.Sequential(
            nn.Linear(score_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim),
        )
        self.last_weights_by_fiber: dict[int, torch.Tensor] = {}

    def lift(
        self,
        *,
        coarse_messages: torch.Tensor,
        edge_fibers: Mapping[int, EdgeFiber],
        fine_edge_count: int,
        fine_edge_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if fine_edge_context is None:
            raise ValueError("LearnedFiberLift requires fine_edge_context.")
        if coarse_messages.shape[1] != self.message_dim:
            raise ValueError("coarse_messages message dimension does not match lift.")
        if fine_edge_context.ndim != 2 or fine_edge_context.shape[1] != self.context_dim:
            raise ValueError("fine_edge_context shape does not match lift context_dim.")
        if fine_edge_context.shape[0] != fine_edge_count:
            raise ValueError("fine_edge_context first dimension must match fine_edge_count.")
        _validate_lift_inputs(coarse_messages, edge_fibers, fine_edge_count)

        fine_messages = coarse_messages.new_zeros((fine_edge_count, self.message_dim))
        self.last_weights_by_fiber = {}
        for coarse_edge, fiber in edge_fibers.items():
            fine_edge_indices = torch.tensor(
                fiber.fine_edges,
                dtype=torch.long,
                device=coarse_messages.device,
            )
            context = fine_edge_context.index_select(0, fine_edge_indices)
            repeated_coarse = coarse_messages[coarse_edge].expand(context.shape[0], -1)
            score_input = torch.cat((repeated_coarse, context), dim=1)
            scores = self.score_mlp(score_input).squeeze(1)
            weights = torch.softmax(scores, dim=0).unsqueeze(1)
            decoder_input = torch.cat((repeated_coarse, context, weights), dim=1)
            decoded = self.decoder(decoder_input)
            fine_messages[fine_edge_indices] = decoded
            self.last_weights_by_fiber[coarse_edge] = weights.squeeze(1)
        return fine_messages
