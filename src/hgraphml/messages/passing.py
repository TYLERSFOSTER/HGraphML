"""PyTorch message-passing helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from torch import nn

from hgraphml.graph import NodeFiber, TensorGraph


class EdgeMessageMLP(nn.Module):
    """Tiny source/target edge-message MLP for the first HGraphML proof."""

    def __init__(
        self,
        *,
        feature_dim: int,
        message_dim: int,
        hidden_dim: int = 32,
        edge_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if message_dim <= 0:
            raise ValueError("message_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if edge_feature_dim < 0:
            raise ValueError("edge_feature_dim must be nonnegative.")
        self.feature_dim = feature_dim
        self.message_dim = message_dim
        self.edge_feature_dim = edge_feature_dim
        input_dim = 2 * feature_dim + edge_feature_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim),
        )

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return edge-aligned messages."""

        if source_features.shape != target_features.shape:
            raise ValueError("source_features and target_features must have the same shape.")
        if source_features.ndim != 2 or source_features.shape[1] != self.feature_dim:
            raise ValueError("source_features shape does not match feature_dim.")
        pieces = [source_features, target_features]
        if self.edge_feature_dim > 0:
            if edge_features is None:
                raise ValueError("edge_features are required by this EdgeMessageMLP.")
            if edge_features.ndim != 2 or edge_features.shape[1] != self.edge_feature_dim:
                raise ValueError("edge_features shape does not match edge_feature_dim.")
            if edge_features.shape[0] != source_features.shape[0]:
                raise ValueError("edge_features first dimension must match edge count.")
            pieces.append(edge_features)
        elif edge_features is not None:
            raise ValueError("edge_features were provided but edge_feature_dim is zero.")
        return cast(torch.Tensor, self.network(torch.cat(pieces, dim=1)))


def run_edge_message_model(
    graph: TensorGraph,
    node_features: torch.Tensor,
    message_model: nn.Module,
    edge_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather source/target features and run an edge-message module."""

    _validate_node_features(graph, node_features)
    if edge_features is not None and edge_features.shape[0] != graph.edge_count:
        raise ValueError("edge_features first dimension must match edge_count.")
    source_features = node_features.index_select(0, graph.sources.to(node_features.device))
    target_features = node_features.index_select(0, graph.targets.to(node_features.device))
    return cast(torch.Tensor, message_model(source_features, target_features, edge_features))


def mean_pool_node_features(
    node_features: torch.Tensor,
    node_fibers: Mapping[int, NodeFiber],
) -> torch.Tensor:
    """Mean-pool fine node features into contiguous coarse-node features."""

    if node_features.ndim != 2:
        raise ValueError("node_features must be rank-2.")
    pooled: list[torch.Tensor] = []
    for expected_coarse_node, coarse_node in enumerate(sorted(node_fibers)):
        if coarse_node != expected_coarse_node:
            raise ValueError("node_fibers must use contiguous coarse-node ids.")
        fine_nodes = torch.tensor(
            node_fibers[coarse_node].fine_nodes,
            dtype=torch.long,
            device=node_features.device,
        )
        pooled.append(node_features.index_select(0, fine_nodes).mean(dim=0))
    if not pooled:
        return node_features.new_zeros((0, node_features.shape[1]))
    return torch.stack(pooled, dim=0)


def _validate_node_features(graph: TensorGraph, node_features: torch.Tensor) -> None:
    if node_features.ndim != 2:
        raise ValueError("node_features must be rank-2.")
    if node_features.shape[0] != graph.node_count:
        raise ValueError("node_features first dimension must match node_count.")
