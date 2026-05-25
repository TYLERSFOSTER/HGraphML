"""Tests for message passing helpers."""

from __future__ import annotations

import pytest
import torch

from hgraphml.graph import NodeFiber, TensorGraph
from hgraphml.messages import EdgeMessageMLP, mean_pool_node_features, run_edge_message_model


def test_run_edge_message_model_returns_edge_aligned_messages() -> None:
    torch.manual_seed(0)
    graph = TensorGraph(node_count=3, edge_index=torch.tensor([[0, 1], [1, 2]]))
    features = torch.randn((3, 4))
    model = EdgeMessageMLP(feature_dim=4, message_dim=5, hidden_dim=8)

    messages = run_edge_message_model(graph, features, model)

    assert messages.shape == (2, 5)


def test_mean_pool_node_features_uses_fibers() -> None:
    features = torch.tensor([[1.0, 3.0], [3.0, 5.0], [10.0, 20.0]])
    fibers = {
        0: NodeFiber(tier=1, coarse_node=0, fine_nodes=(0, 1)),
        1: NodeFiber(tier=1, coarse_node=1, fine_nodes=(2,)),
    }

    pooled = mean_pool_node_features(features, fibers)

    assert torch.equal(pooled, torch.tensor([[2.0, 4.0], [10.0, 20.0]]))


def test_mean_pool_node_features_rejects_noncontiguous_ids() -> None:
    features = torch.ones((2, 2))
    fibers = {1: NodeFiber(tier=1, coarse_node=1, fine_nodes=(0,))}

    with pytest.raises(ValueError, match="contiguous"):
        mean_pool_node_features(features, fibers)
