"""Tests for tensor graph data."""

from __future__ import annotations

import pytest
import torch

from hgraphml.graph import TensorGraph


def test_tensor_graph_accepts_valid_edge_index() -> None:
    graph = TensorGraph(
        node_count=3,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_labels=("a", "b"),
    )

    assert graph.edge_count == 2
    assert graph.sources.tolist() == [0, 1]
    assert graph.targets.tolist() == [1, 2]


def test_tensor_graph_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        TensorGraph(node_count=3, edge_index=torch.tensor([[0, 1]], dtype=torch.long))


def test_tensor_graph_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="out of range"):
        TensorGraph(node_count=2, edge_index=torch.tensor([[0], [2]], dtype=torch.long))


def test_tensor_graph_rejects_edge_label_mismatch() -> None:
    with pytest.raises(ValueError, match="edge_labels"):
        TensorGraph(
            node_count=2,
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_labels=("only", "too-many"),
        )
