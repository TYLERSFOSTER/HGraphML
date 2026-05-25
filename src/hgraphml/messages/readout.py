"""Readout helpers from fine messages back to nodes."""

from __future__ import annotations

import torch

from hgraphml.graph import TensorGraph


def incoming_sum_readout(graph: TensorGraph, fine_messages: torch.Tensor) -> torch.Tensor:
    """Aggregate incoming fine-edge messages at target nodes."""

    if fine_messages.ndim != 2:
        raise ValueError("fine_messages must be rank-2.")
    if fine_messages.shape[0] != graph.edge_count:
        raise ValueError("fine_messages first dimension must match edge_count.")
    readout = fine_messages.new_zeros((graph.node_count, fine_messages.shape[1]))
    if graph.edge_count == 0:
        return readout
    targets = graph.targets.to(fine_messages.device)
    readout.index_add_(0, targets, fine_messages)
    return readout
