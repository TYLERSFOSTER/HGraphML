"""Message tensor containers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hgraphml.graph import TensorGraph


@dataclass(frozen=True, slots=True)
class MessageBatch:
    """Edge-aligned messages for one tensor graph."""

    graph: TensorGraph
    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("MessageBatch.values must be a rank-2 tensor.")
        if self.values.shape[0] != self.graph.edge_count:
            raise ValueError("MessageBatch.values first dimension must match edge_count.")
