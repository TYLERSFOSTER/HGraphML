"""Adapter from complete HGraphML tensor graphs to state_collapser partition towers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from state_collapser.core.action import PrimitiveAction
from state_collapser.core.edges import BaseEdge
from state_collapser.core.state import State
from state_collapser.tower.partition import ContractionSchema, PartitionTower
from state_collapser.tower.partition.tower import build_partition_tower_full
from state_collapser.training import EncodingRegistry

from hgraphml.graph import EdgeFiber, NodeFiber, TensorGraph


@dataclass(frozen=True, slots=True)
class TowerBundle:
    """State-collapser-backed tower plus HGraphML fiber/readout maps."""

    graph: TensorGraph
    partition_tower: PartitionTower
    node_fibers_by_tier: tuple[Mapping[int, NodeFiber], ...]
    edge_fibers_by_tier: tuple[Mapping[int, EdgeFiber], ...]
    coarse_graphs_by_tier: tuple[TensorGraph, ...]


def build_tower_bundle(
    graph: TensorGraph,
    *,
    contraction_schema: ContractionSchema | None = None,
) -> TowerBundle:
    """Build a partition tower by treating the full graph as already explored."""

    states = tuple(
        State(payload=("node", node_index), identity=("node", node_index))
        for node_index in range(graph.node_count)
    )
    edges = tuple(
        _edge_for_index(graph, states, edge_index) for edge_index in range(graph.edge_count)
    )
    tower = build_partition_tower_full(
        states=states,
        edges=edges,
        current_state=None,
        schema=contraction_schema,
    )
    node_index_by_state = {state: index for index, state in enumerate(states)}
    edge_index_by_edge = {edge: index for index, edge in enumerate(edges)}

    node_fibers_by_tier = []
    edge_fibers_by_tier = []
    coarse_graphs_by_tier = []
    for tier, state_layer in enumerate(tower.state_layers):
        node_fibers, cell_to_coarse = _node_fibers_for_tier(
            tower,
            tier,
            node_index_by_state,
        )
        edge_fibers, coarse_edge_sources, coarse_edge_targets = _edge_fibers_for_tier(
            tower,
            tier,
            edge_index_by_edge,
            cell_to_coarse,
        )
        node_fibers_by_tier.append(node_fibers)
        edge_fibers_by_tier.append(edge_fibers)
        coarse_graphs_by_tier.append(
            _coarse_graph(
                node_count=len(state_layer.all_cell_ids()),
                sources=coarse_edge_sources,
                targets=coarse_edge_targets,
            )
        )
    return TowerBundle(
        graph=graph,
        partition_tower=tower,
        node_fibers_by_tier=tuple(node_fibers_by_tier),
        edge_fibers_by_tier=tuple(edge_fibers_by_tier),
        coarse_graphs_by_tier=tuple(coarse_graphs_by_tier),
    )


def build_encoding_registry(bundle: TowerBundle) -> EncodingRegistry:
    """Build the shared state_collapser encoding registry for a tower bundle."""

    return EncodingRegistry.from_tower(bundle.partition_tower)


def _edge_for_index(graph: TensorGraph, states: tuple[State, ...], edge_index: int) -> BaseEdge:
    source_index = int(graph.sources[edge_index].item())
    target_index = int(graph.targets[edge_index].item())
    label = graph.edge_labels[edge_index] if graph.edge_labels is not None else None
    labels = () if label is None else (label,)
    action = PrimitiveAction(
        payload=("edge", edge_index),
        identity=("edge", edge_index),
        labels=labels,
    )
    return BaseEdge(
        source=states[source_index],
        action=action,
        target=states[target_index],
        labels=labels,
    )


def _node_fibers_for_tier(
    tower: PartitionTower,
    tier: int,
    node_index_by_state: Mapping[State, int],
) -> tuple[dict[int, NodeFiber], dict[object, int]]:
    state_layer = tower.state_layers[tier]
    node_fibers: dict[int, NodeFiber] = {}
    cell_to_coarse: dict[object, int] = {}
    for coarse_node, state_cell_id in enumerate(state_layer.all_cell_ids()):
        cell_to_coarse[state_cell_id] = coarse_node
        fine_nodes = tuple(
            sorted(
                node_index_by_state[tower.registry.state_for_id(state_id)]
                for state_id in state_layer.members(state_cell_id)
            )
        )
        node_fibers[coarse_node] = NodeFiber(
            tier=tier,
            coarse_node=coarse_node,
            fine_nodes=fine_nodes,
        )
    return node_fibers, cell_to_coarse


def _edge_fibers_for_tier(
    tower: PartitionTower,
    tier: int,
    edge_index_by_edge: Mapping[BaseEdge, int],
    cell_to_coarse: Mapping[object, int],
) -> tuple[dict[int, EdgeFiber], list[int], list[int]]:
    edge_fibers: dict[int, EdgeFiber] = {}
    coarse_edge_sources: list[int] = []
    coarse_edge_targets: list[int] = []
    state_layer = tower.state_layers[tier]
    edges_by_coarse_endpoints: dict[tuple[int, int], list[int]] = {}

    for edge_id in tower.registry.edge_ids:
        source_cell = state_layer.cell_of_state_id[tower.registry.source_state_id(edge_id)]
        target_cell = state_layer.cell_of_state_id[tower.registry.target_state_id(edge_id)]
        source_coarse = cell_to_coarse[source_cell]
        target_coarse = cell_to_coarse[target_cell]
        fine_edge = edge_index_by_edge[tower.registry.edge_for_id(edge_id)]
        edges_by_coarse_endpoints.setdefault((source_coarse, target_coarse), []).append(fine_edge)

    for coarse_edge, ((source_coarse, target_coarse), fine_edges) in enumerate(
        sorted(edges_by_coarse_endpoints.items())
    ):
        edge_fibers[coarse_edge] = EdgeFiber(
            tier=tier,
            coarse_edge=coarse_edge,
            fine_edges=tuple(sorted(fine_edges)),
        )
        coarse_edge_sources.append(source_coarse)
        coarse_edge_targets.append(target_coarse)
    return edge_fibers, coarse_edge_sources, coarse_edge_targets


def _coarse_graph(*, node_count: int, sources: list[int], targets: list[int]) -> TensorGraph:
    if sources:
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return TensorGraph(node_count=node_count, edge_index=edge_index)
