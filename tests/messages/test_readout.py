"""Tests for readout helpers."""

from __future__ import annotations

import torch

from hgraphml.graph import TensorGraph
from hgraphml.messages import incoming_sum_readout


def test_incoming_sum_readout_aggregates_by_target() -> None:
    graph = TensorGraph(node_count=3, edge_index=torch.tensor([[0, 1], [2, 2]]))
    messages = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    readout = incoming_sum_readout(graph, messages)

    assert torch.equal(readout[0], torch.zeros(2))
    assert torch.equal(readout[2], torch.tensor([4.0, 6.0]))
