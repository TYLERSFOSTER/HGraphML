"""Tests for toy teacher objectives."""

from __future__ import annotations

from hgraphml.examples import make_repeated_motif_graph, make_toy_node_features
from hgraphml.training import make_teacher_node_targets


def test_teacher_targets_are_deterministic_and_node_aligned() -> None:
    graph = make_repeated_motif_graph()
    features = make_toy_node_features(seed=0)

    first = make_teacher_node_targets(graph, features, output_dim=8, seed=10)
    second = make_teacher_node_targets(graph, features, output_dim=8, seed=10)

    assert first.shape == (graph.node_count, 8)
    assert first.equal(second)
