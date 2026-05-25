"""Uniform pullback lift."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from hgraphml.graph import EdgeFiber


class UniformPullbackLift:
    """Copy each coarse message to every fine edge in its fiber."""

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
            fine_messages[list(fiber.fine_edges)] = coarse_messages[coarse_edge]
        return fine_messages


def _validate_lift_inputs(
    coarse_messages: torch.Tensor,
    edge_fibers: Mapping[int, EdgeFiber],
    fine_edge_count: int,
) -> None:
    if coarse_messages.ndim != 2:
        raise ValueError("coarse_messages must be rank-2.")
    if fine_edge_count < 0:
        raise ValueError("fine_edge_count must be nonnegative.")
    for coarse_edge, fiber in edge_fibers.items():
        if coarse_edge < 0 or coarse_edge >= coarse_messages.shape[0]:
            raise ValueError("edge_fibers contain a coarse edge without a message row.")
        if any(fine_edge >= fine_edge_count for fine_edge in fiber.fine_edges):
            raise ValueError("edge_fibers contain a fine edge outside fine_edge_count.")
