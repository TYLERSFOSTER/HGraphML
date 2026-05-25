"""Shared lift protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import torch

from hgraphml.graph import EdgeFiber


class MessageLift(Protocol):
    """Protocol for lifting coarse edge messages to fine edge messages."""

    def lift(
        self,
        *,
        coarse_messages: torch.Tensor,
        edge_fibers: Mapping[int, EdgeFiber],
        fine_edge_count: int,
        fine_edge_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return fine edge messages aligned to the original graph."""
