"""Main hierarchical message-passing orchestration call."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hgraphml.adapters import TowerBundle, build_tower_bundle
from hgraphml.graph import TensorGraph
from hgraphml.lifts import MessageLift
from hgraphml.messages import incoming_sum_readout, mean_pool_node_features, run_edge_message_model


@dataclass(frozen=True, slots=True)
class HGraphMLResult:
    """Structured result from one hierarchical message pass."""

    graph: TensorGraph
    tower_bundle: TowerBundle
    coarse_tier: int
    coarse_messages: torch.Tensor
    fine_messages: torch.Tensor
    node_readout: torch.Tensor


def collapse_messages(
    *,
    graph: TensorGraph,
    node_features: torch.Tensor,
    message_model: nn.Module,
    lift: MessageLift,
    edge_features: torch.Tensor | None = None,
    contraction_schema: object | None = None,
    tower_bundle: TowerBundle | None = None,
    coarse_tier: int | None = None,
    refinement_steps: int = 0,
) -> HGraphMLResult:
    """Run one quotient-tower-backed message-passing pass."""

    if refinement_steps != 0:
        raise NotImplementedError("refinement_steps are not implemented in milestone one.")
    _validate_features(graph, node_features, edge_features)
    bundle = tower_bundle or build_tower_bundle(
        graph,
        contraction_schema=contraction_schema,  # type: ignore[arg-type]
    )
    if bundle.graph.node_count != graph.node_count or bundle.graph.edge_count != graph.edge_count:
        raise ValueError("tower_bundle graph shape does not match graph.")
    tier = _select_tier(coarse_tier, len(bundle.node_fibers_by_tier))
    node_fibers = bundle.node_fibers_by_tier[tier]
    edge_fibers = bundle.edge_fibers_by_tier[tier]
    coarse_graph = bundle.coarse_graphs_by_tier[tier]
    coarse_node_features = mean_pool_node_features(node_features, node_fibers)
    coarse_messages = run_edge_message_model(coarse_graph, coarse_node_features, message_model)
    fine_context = _fine_edge_context(graph, node_features, edge_features)
    fine_messages = lift.lift(
        coarse_messages=coarse_messages,
        edge_fibers=edge_fibers,
        fine_edge_count=graph.edge_count,
        fine_edge_context=fine_context,
    )
    node_readout = incoming_sum_readout(graph, fine_messages)
    return HGraphMLResult(
        graph=graph,
        tower_bundle=bundle,
        coarse_tier=tier,
        coarse_messages=coarse_messages,
        fine_messages=fine_messages,
        node_readout=node_readout,
    )


def _select_tier(coarse_tier: int | None, tier_count: int) -> int:
    if tier_count <= 0:
        raise ValueError("tower bundle must contain at least one tier.")
    if coarse_tier is None:
        return tier_count - 1
    tier = coarse_tier if coarse_tier >= 0 else tier_count + coarse_tier
    if tier < 0 or tier >= tier_count:
        raise ValueError("coarse_tier is out of range.")
    return tier


def _fine_edge_context(
    graph: TensorGraph,
    node_features: torch.Tensor,
    edge_features: torch.Tensor | None,
) -> torch.Tensor:
    source_features = node_features.index_select(0, graph.sources.to(node_features.device))
    target_features = node_features.index_select(0, graph.targets.to(node_features.device))
    pieces = [source_features, target_features]
    if edge_features is not None:
        pieces.append(edge_features)
    return torch.cat(pieces, dim=1)


def _validate_features(
    graph: TensorGraph,
    node_features: torch.Tensor,
    edge_features: torch.Tensor | None,
) -> None:
    if node_features.ndim != 2:
        raise ValueError("node_features must be rank-2.")
    if node_features.shape[0] != graph.node_count:
        raise ValueError("node_features first dimension must match node_count.")
    if edge_features is not None:
        if edge_features.ndim != 2:
            raise ValueError("edge_features must be rank-2.")
        if edge_features.shape[0] != graph.edge_count:
            raise ValueError("edge_features first dimension must match edge_count.")
