"""Tests for fiber-normalized lift."""

from __future__ import annotations

import torch

from hgraphml.graph import EdgeFiber
from hgraphml.lifts import FiberNormalizedLift, UniformPullbackLift


def test_fiber_normalized_lift_divides_message_mass() -> None:
    lift = FiberNormalizedLift()
    coarse = torch.tensor([[2.0, 4.0]])
    fibers = {0: EdgeFiber(tier=1, coarse_edge=0, fine_edges=(0, 1))}

    fine = lift.lift(coarse_messages=coarse, edge_fibers=fibers, fine_edge_count=2)

    assert torch.equal(fine[0], torch.tensor([1.0, 2.0]))
    assert torch.equal(fine.sum(dim=0), coarse[0])


def test_fiber_normalized_differs_from_uniform_when_fiber_has_many_edges() -> None:
    coarse = torch.tensor([[2.0, 4.0]])
    fibers = {0: EdgeFiber(tier=1, coarse_edge=0, fine_edges=(0, 1))}

    normalized = FiberNormalizedLift().lift(
        coarse_messages=coarse,
        edge_fibers=fibers,
        fine_edge_count=2,
    )
    uniform = UniformPullbackLift().lift(
        coarse_messages=coarse,
        edge_fibers=fibers,
        fine_edge_count=2,
    )

    assert not torch.equal(normalized, uniform)


def test_fiber_normalized_lift_preserves_gradient() -> None:
    coarse = torch.ones((1, 2), requires_grad=True)
    fibers = {0: EdgeFiber(tier=1, coarse_edge=0, fine_edges=(0, 1))}

    fine = FiberNormalizedLift().lift(
        coarse_messages=coarse,
        edge_fibers=fibers,
        fine_edge_count=2,
    )
    fine.sum().backward()

    assert coarse.grad is not None
    assert torch.equal(coarse.grad, torch.tensor([[1.0, 1.0]]))
