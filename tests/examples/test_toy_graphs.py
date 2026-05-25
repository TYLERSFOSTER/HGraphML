"""Tests for toy graph helpers."""

from __future__ import annotations

from hgraphml.examples import make_repeated_motif_graph, make_toy_node_features


def test_repeated_motif_graph_shape() -> None:
    graph = make_repeated_motif_graph()

    assert graph.node_count == 6
    assert graph.edge_count == 7


def test_toy_node_features_are_deterministic() -> None:
    first = make_toy_node_features(seed=2)
    second = make_toy_node_features(seed=2)

    assert first.shape == (6, 8)
    assert first.equal(second)
