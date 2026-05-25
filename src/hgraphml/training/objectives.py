"""Toy teacher objectives."""

from __future__ import annotations

import torch

from hgraphml.graph import TensorGraph
from hgraphml.messages import EdgeMessageMLP, incoming_sum_readout, run_edge_message_model


def make_teacher_node_targets(
    graph: TensorGraph,
    node_features: torch.Tensor,
    *,
    output_dim: int,
    seed: int = 0,
) -> torch.Tensor:
    """Generate graph-structured node targets from a fixed teacher model."""

    if node_features.ndim != 2:
        raise ValueError("node_features must be rank-2.")
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    teacher = EdgeMessageMLP(
        feature_dim=int(node_features.shape[1]),
        message_dim=output_dim,
        hidden_dim=32,
    )
    torch.random.set_rng_state(generator_state)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        fine_messages = run_edge_message_model(graph, node_features, teacher)
        return incoming_sum_readout(graph, fine_messages)
