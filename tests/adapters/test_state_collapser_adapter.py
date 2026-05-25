"""Tests for state_collapser tower adapter."""

from __future__ import annotations

from state_collapser.tower.partition import LabelBlockSchema

from hgraphml.adapters import build_tower_bundle
from hgraphml.examples import make_repeated_motif_graph


def test_build_tower_bundle_registers_full_graph() -> None:
    graph = make_repeated_motif_graph()
    bundle = build_tower_bundle(graph)

    assert len(bundle.partition_tower.registry.state_ids) == graph.node_count
    assert len(bundle.partition_tower.registry.edge_ids) == graph.edge_count
    assert bundle.coarse_graphs_by_tier[0].node_count == graph.node_count
    assert bundle.coarse_graphs_by_tier[0].edge_count == graph.edge_count


def test_build_tower_bundle_with_schema_creates_higher_tier() -> None:
    graph = make_repeated_motif_graph()
    bundle = build_tower_bundle(graph, contraction_schema=LabelBlockSchema.from_labels(("motif",)))

    assert len(bundle.node_fibers_by_tier) >= 2
    assert len(bundle.edge_fibers_by_tier) >= 2


def test_fiber_maps_cover_all_fine_indices() -> None:
    graph = make_repeated_motif_graph()
    bundle = build_tower_bundle(graph, contraction_schema=LabelBlockSchema.from_labels(("motif",)))

    for node_fibers in bundle.node_fibers_by_tier:
        covered_nodes = sorted(
            fine_node for fiber in node_fibers.values() for fine_node in fiber.fine_nodes
        )
        assert covered_nodes == list(range(graph.node_count))
    for edge_fibers in bundle.edge_fibers_by_tier:
        covered_edges = sorted(
            fine_edge for fiber in edge_fibers.values() for fine_edge in fiber.fine_edges
        )
        assert covered_edges == list(range(graph.edge_count))
