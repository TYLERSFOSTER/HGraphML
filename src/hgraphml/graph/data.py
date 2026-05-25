"""Tensor graph container used by the first HGraphML surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclass(frozen=True, slots=True)
class TensorGraph:
    """Small directed graph represented by edge-index tensors."""

    node_count: int
    edge_index: torch.Tensor
    edge_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.node_count < 0:
            raise ValueError("TensorGraph.node_count must be nonnegative.")
        if self.edge_index.ndim != 2:
            raise ValueError("TensorGraph.edge_index must be a rank-2 tensor.")
        if self.edge_index.shape[0] != 2:
            raise ValueError("TensorGraph.edge_index must have shape (2, edge_count).")
        if self.edge_index.dtype not in _INTEGER_DTYPES:
            raise TypeError("TensorGraph.edge_index must use an integer dtype.")
        if self.edge_labels is not None and len(self.edge_labels) != self.edge_count:
            raise ValueError("TensorGraph.edge_labels length must match edge_count.")
        if self.edge_count == 0:
            return
        min_index = int(self.edge_index.min().item())
        max_index = int(self.edge_index.max().item())
        if min_index < 0:
            raise ValueError("TensorGraph.edge_index cannot contain negative indices.")
        if max_index >= self.node_count:
            raise ValueError("TensorGraph.edge_index contains node indices out of range.")

    @property
    def edge_count(self) -> int:
        """Return the number of directed edges."""

        return int(self.edge_index.shape[1])

    @property
    def sources(self) -> torch.Tensor:
        """Return source-node indices for all edges."""

        return self.edge_index[0]

    @property
    def targets(self) -> torch.Tensor:
        """Return target-node indices for all edges."""

        return self.edge_index[1]
