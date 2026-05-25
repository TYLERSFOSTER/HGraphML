"""Tests for message containers."""

from __future__ import annotations

import pytest
import torch

from hgraphml.graph import TensorGraph
from hgraphml.messages import MessageBatch


def test_message_batch_accepts_edge_aligned_values() -> None:
    graph = TensorGraph(node_count=2, edge_index=torch.tensor([[0], [1]], dtype=torch.long))
    messages = MessageBatch(graph=graph, values=torch.ones((1, 4)))

    assert messages.values.shape == (1, 4)


def test_message_batch_rejects_wrong_rank() -> None:
    graph = TensorGraph(node_count=2, edge_index=torch.tensor([[0], [1]], dtype=torch.long))
    with pytest.raises(ValueError, match="rank-2"):
        MessageBatch(graph=graph, values=torch.ones((1, 4, 1)))


def test_message_batch_rejects_wrong_edge_count() -> None:
    graph = TensorGraph(node_count=2, edge_index=torch.tensor([[0], [1]], dtype=torch.long))
    with pytest.raises(ValueError, match="edge_count"):
        MessageBatch(graph=graph, values=torch.ones((2, 4)))
