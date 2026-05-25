"""Tests for collapse_messages."""

from __future__ import annotations

import torch
from state_collapser.tower.partition import LabelBlockSchema

from hgraphml import collapse_messages
from hgraphml.adapters import build_tower_bundle
from hgraphml.examples import make_repeated_motif_graph, make_toy_node_features
from hgraphml.lifts import LearnedFiberLift, UniformPullbackLift
from hgraphml.messages import EdgeMessageMLP


def test_collapse_messages_forward_pass() -> None:
    torch.manual_seed(0)
    graph = make_repeated_motif_graph()
    features = make_toy_node_features(seed=0)
    model = EdgeMessageMLP(feature_dim=8, message_dim=8, hidden_dim=16)
    lift = UniformPullbackLift()

    result = collapse_messages(graph=graph, node_features=features, message_model=model, lift=lift)

    assert result.fine_messages.shape == (graph.edge_count, 8)
    assert result.node_readout.shape == (graph.node_count, 8)


def test_collapse_messages_backpropagates_through_learned_lift() -> None:
    torch.manual_seed(0)
    graph = make_repeated_motif_graph()
    features = make_toy_node_features(seed=0)
    bundle = build_tower_bundle(graph, contraction_schema=LabelBlockSchema.from_labels(("motif",)))
    model = EdgeMessageMLP(feature_dim=8, message_dim=8, hidden_dim=16)
    lift = LearnedFiberLift(message_dim=8, context_dim=16, hidden_dim=16)

    result = collapse_messages(
        graph=graph,
        node_features=features,
        message_model=model,
        lift=lift,
        tower_bundle=bundle,
    )
    result.node_readout.sum().backward()

    assert any(parameter.grad is not None for parameter in lift.parameters())
    assert result.coarse_tier == len(bundle.node_fibers_by_tier) - 1
