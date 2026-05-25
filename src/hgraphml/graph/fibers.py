"""Fiber data structures connecting coarse graph cells to fine graph indices."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NodeFiber:
    """Fine node indices represented by one coarse node at one tier."""

    tier: int
    coarse_node: int
    fine_nodes: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_index_tuple(
            name="NodeFiber.fine_nodes",
            tier=self.tier,
            coarse_index=self.coarse_node,
            fine_indices=self.fine_nodes,
        )


@dataclass(frozen=True, slots=True)
class EdgeFiber:
    """Fine edge indices represented by one coarse edge at one tier."""

    tier: int
    coarse_edge: int
    fine_edges: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_index_tuple(
            name="EdgeFiber.fine_edges",
            tier=self.tier,
            coarse_index=self.coarse_edge,
            fine_indices=self.fine_edges,
        )


NodeFiberMap = Mapping[int, NodeFiber]
EdgeFiberMap = Mapping[int, EdgeFiber]


def _validate_index_tuple(
    *,
    name: str,
    tier: int,
    coarse_index: int,
    fine_indices: tuple[int, ...],
) -> None:
    if tier < 0:
        raise ValueError("fiber tier must be nonnegative.")
    if coarse_index < 0:
        raise ValueError("fiber coarse index must be nonnegative.")
    if not fine_indices:
        raise ValueError(f"{name} cannot be empty.")
    if any(index < 0 for index in fine_indices):
        raise ValueError(f"{name} cannot contain negative indices.")
    if len(set(fine_indices)) != len(fine_indices):
        raise ValueError(f"{name} cannot contain duplicate indices.")
