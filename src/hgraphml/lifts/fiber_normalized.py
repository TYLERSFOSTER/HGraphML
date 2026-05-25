"""Fiber-normalized aggregate lift."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from hgraphml.graph import EdgeFiber
from hgraphml.lifts.uniform import _validate_lift_inputs


class FiberNormalizedLift:
    """Distribute each coarse aggregate message uniformly over its fine fiber."""

    def lift(
        self,
        *,
        coarse_messages: torch.Tensor,
        edge_fibers: Mapping[int, EdgeFiber],
        fine_edge_count: int,
        fine_edge_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del fine_edge_context
        _validate_lift_inputs(coarse_messages, edge_fibers, fine_edge_count)
        fine_messages = coarse_messages.new_zeros((fine_edge_count, coarse_messages.shape[1]))
        for coarse_edge, fiber in edge_fibers.items():
            fine_messages[list(fiber.fine_edges)] = coarse_messages[coarse_edge] / len(
                fiber.fine_edges
            )
        return fine_messages
