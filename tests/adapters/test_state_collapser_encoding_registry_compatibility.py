"""Compatibility tests for state_collapser's shared encoding registry."""

from __future__ import annotations

import json

from state_collapser.tower.partition import LabelBlockSchema
from state_collapser.training import EncodingRegistry

from hgraphml.adapters import build_encoding_registry, build_tower_bundle
from hgraphml.examples import make_repeated_motif_graph


def test_build_encoding_registry_from_tower_bundle_is_json_safe() -> None:
    graph = make_repeated_motif_graph()
    schema = LabelBlockSchema.from_labels(("motif",))
    bundle = build_tower_bundle(graph, contraction_schema=schema)

    registry = build_encoding_registry(bundle)

    assert registry.registry_id
    json.dumps(registry.to_dict())


def test_build_encoding_registry_matches_direct_state_collapser_construction() -> None:
    graph = make_repeated_motif_graph()
    schema = LabelBlockSchema.from_labels(("motif",))
    bundle = build_tower_bundle(graph, contraction_schema=schema)

    direct = EncodingRegistry.from_tower(bundle.partition_tower)
    via_helper = build_encoding_registry(bundle)

    assert via_helper.registry_id == direct.registry_id
    assert via_helper.to_dict() == direct.to_dict()


def test_encoding_registry_encodes_base_states_and_edges() -> None:
    graph = make_repeated_motif_graph()
    schema = LabelBlockSchema.from_labels(("motif",))
    bundle = build_tower_bundle(graph, contraction_schema=schema)
    registry = build_encoding_registry(bundle)
    tower = bundle.partition_tower

    for state_id in tower.registry.state_ids:
        state = tower.registry.state_for_id(state_id)
        assert registry.encode_state(state) is not None

    for edge_id in tower.registry.edge_ids:
        edge = tower.registry.edge_for_id(edge_id)
        assert registry.encode_edge(edge) is not None


def test_encoding_registry_encodes_state_and_action_cells() -> None:
    graph = make_repeated_motif_graph()
    schema = LabelBlockSchema.from_labels(("motif",))
    bundle = build_tower_bundle(graph, contraction_schema=schema)
    registry = build_encoding_registry(bundle)
    tower = bundle.partition_tower

    for state_layer in tower.state_layers:
        for state_cell_id in state_layer.all_cell_ids():
            assert registry.encode_state_cell(state_cell_id) is not None

    action_cell_ids = [
        action_cell_id
        for action_layer in tower.action_layers
        for action_cell_id in action_layer.edge_ids_by_action_cell
    ]
    assert action_cell_ids
    for action_cell_id in action_cell_ids:
        assert registry.encode_action_cell(action_cell_id) is not None


def test_encoding_registry_is_compatible_with_hgraphml_fibers() -> None:
    graph = make_repeated_motif_graph()
    schema = LabelBlockSchema.from_labels(("motif",))
    bundle = build_tower_bundle(graph, contraction_schema=schema)
    registry = build_encoding_registry(bundle)
    tower = bundle.partition_tower

    for node_fibers in bundle.node_fibers_by_tier:
        covered_nodes = sorted(
            fine_node for fiber in node_fibers.values() for fine_node in fiber.fine_nodes
        )
        assert covered_nodes == list(range(graph.node_count))
        for fiber in node_fibers.values():
            assert all(0 <= fine_node < graph.node_count for fine_node in fiber.fine_nodes)

    for edge_fibers in bundle.edge_fibers_by_tier:
        covered_edges = sorted(
            fine_edge for fiber in edge_fibers.values() for fine_edge in fiber.fine_edges
        )
        assert covered_edges == list(range(graph.edge_count))
        for fiber in edge_fibers.values():
            assert all(0 <= fine_edge < graph.edge_count for fine_edge in fiber.fine_edges)

    for edge_id in tower.registry.edge_ids:
        edge = tower.registry.edge_for_id(edge_id)
        assert registry.encode_edge(edge) is not None
