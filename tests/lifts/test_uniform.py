"""Tests for uniform pullback lift."""

from __future__ import annotations

import pytest
import torch

from hgraphml.graph import EdgeFiber
from hgraphml.lifts import UniformPullbackLift


def test_uniform_lift_copies_messages_to_fiber() -> None:
    lift = UniformPullbackLift()
    coarse = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    fibers = {
        0: EdgeFiber(tier=1, coarse_edge=0, fine_edges=(0, 2)),
        1: EdgeFiber(tier=1, coarse_edge=1, fine_edges=(1,)),
    }

    fine = lift.lift(coarse_messages=coarse, edge_fibers=fibers, fine_edge_count=3)

    assert torch.equal(fine[0], coarse[0])
    assert torch.equal(fine[2], coarse[0])
    assert torch.equal(fine[1], coarse[1])


def test_uniform_lift_preserves_gradient() -> None:
    lift = UniformPullbackLift()
    coarse = torch.ones((1, 2), requires_grad=True)
    fibers = {0: EdgeFiber(tier=1, coarse_edge=0, fine_edges=(0, 1))}

    fine = lift.lift(coarse_messages=coarse, edge_fibers=fibers, fine_edge_count=2)
    fine.sum().backward()

    assert coarse.grad is not None
    assert torch.equal(coarse.grad, torch.tensor([[2.0, 2.0]]))


def test_uniform_lift_rejects_missing_coarse_message() -> None:
    lift = UniformPullbackLift()
    fibers = {1: EdgeFiber(tier=1, coarse_edge=1, fine_edges=(0,))}

    with pytest.raises(ValueError, match="without a message"):
        lift.lift(coarse_messages=torch.ones((1, 2)), edge_fibers=fibers, fine_edge_count=1)
