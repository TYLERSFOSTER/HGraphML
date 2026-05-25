"""Tests for learned fiber lift."""

from __future__ import annotations

import torch

from hgraphml.graph import EdgeFiber
from hgraphml.lifts import LearnedFiberLift


def test_learned_lift_outputs_fine_edge_messages_and_normalized_weights() -> None:
    torch.manual_seed(0)
    lift = LearnedFiberLift(message_dim=3, context_dim=4, hidden_dim=8)
    coarse = torch.randn((2, 3))
    context = torch.randn((3, 4))
    fibers = {
        0: EdgeFiber(tier=1, coarse_edge=0, fine_edges=(0, 2)),
        1: EdgeFiber(tier=1, coarse_edge=1, fine_edges=(1,)),
    }

    fine = lift.lift(
        coarse_messages=coarse,
        edge_fibers=fibers,
        fine_edge_count=3,
        fine_edge_context=context,
    )

    assert fine.shape == (3, 3)
    assert torch.isclose(lift.last_weights_by_fiber[0].sum(), torch.tensor(1.0))
    assert torch.isclose(lift.last_weights_by_fiber[1].sum(), torch.tensor(1.0))


def test_learned_lift_parameters_receive_gradients_and_update() -> None:
    torch.manual_seed(0)
    lift = LearnedFiberLift(message_dim=2, context_dim=4, hidden_dim=8)
    optimizer = torch.optim.SGD(lift.parameters(), lr=0.1)
    coarse = torch.randn((1, 2))
    context = torch.randn((2, 4))
    fibers = {0: EdgeFiber(tier=1, coarse_edge=0, fine_edges=(0, 1))}
    before = [parameter.detach().clone() for parameter in lift.parameters()]

    fine = lift.lift(
        coarse_messages=coarse,
        edge_fibers=fibers,
        fine_edge_count=2,
        fine_edge_context=context,
    )
    loss = fine.pow(2).sum()
    loss.backward()
    optimizer.step()

    assert any(parameter.grad is not None for parameter in lift.parameters())
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, lift.parameters(), strict=True)
    )
