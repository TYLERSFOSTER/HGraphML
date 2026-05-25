"""Tests for fiber maps."""

from __future__ import annotations

import pytest

from hgraphml.graph import EdgeFiber, NodeFiber


def test_node_fiber_stores_fine_nodes() -> None:
    fiber = NodeFiber(tier=1, coarse_node=0, fine_nodes=(0, 1))

    assert fiber.fine_nodes == (0, 1)


def test_edge_fiber_stores_fine_edges() -> None:
    fiber = EdgeFiber(tier=1, coarse_edge=2, fine_edges=(3, 4))

    assert fiber.fine_edges == (3, 4)


def test_fiber_rejects_empty_preimage() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        EdgeFiber(tier=1, coarse_edge=0, fine_edges=())


def test_fiber_rejects_duplicate_indices() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        NodeFiber(tier=1, coarse_node=0, fine_nodes=(1, 1))
