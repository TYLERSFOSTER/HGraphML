"""Tiny train-step helper for the first HGraphML proof."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain

import torch
from torch import nn

from hgraphml.adapters import TowerBundle
from hgraphml.collapse import HGraphMLResult, collapse_messages
from hgraphml.diagnostics import (
    GradientDiagnostics,
    ViabilityDiagnostics,
    collect_gradient_diagnostics,
    parameters_changed,
    snapshot_parameters,
)
from hgraphml.graph import TensorGraph
from hgraphml.lifts import MessageLift


@dataclass(frozen=True, slots=True)
class TrainStepResult:
    """Result of one tiny HGraphML training step."""

    loss: float
    hgraphml_result: HGraphMLResult
    gradient_diagnostics: GradientDiagnostics
    viability_diagnostics: ViabilityDiagnostics


def train_step(
    *,
    graph: TensorGraph,
    node_features: torch.Tensor,
    target: torch.Tensor,
    message_model: nn.Module,
    lift: MessageLift,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    tower_bundle: TowerBundle | None = None,
) -> TrainStepResult:
    """Run one train step through collapse_messages."""

    trainable_parameters = tuple(_trainable_parameters(message_model, lift))
    before = snapshot_parameters(trainable_parameters)
    optimizer.zero_grad()
    result = collapse_messages(
        graph=graph,
        node_features=node_features,
        message_model=message_model,
        lift=lift,
        tower_bundle=tower_bundle,
    )
    loss = loss_fn(result.node_readout, target)
    loss.backward()  # type: ignore[no-untyped-call]
    gradient_diagnostics = collect_gradient_diagnostics(trainable_parameters)
    optimizer.step()
    viability_diagnostics = ViabilityDiagnostics(
        loss_is_finite=bool(torch.isfinite(loss).item()),
        parameters_changed=parameters_changed(before, trainable_parameters),
        fine_message_shape=tuple(result.fine_messages.shape),
        coarse_message_shape=tuple(result.coarse_messages.shape),
    )
    return TrainStepResult(
        loss=float(loss.detach().item()),
        hgraphml_result=result,
        gradient_diagnostics=gradient_diagnostics,
        viability_diagnostics=viability_diagnostics,
    )


def _trainable_parameters(
    message_model: nn.Module,
    lift: MessageLift,
) -> tuple[nn.Parameter, ...]:
    lift_parameters = lift.parameters() if isinstance(lift, nn.Module) else ()
    return tuple(
        parameter
        for parameter in chain(message_model.parameters(), lift_parameters)
        if parameter.requires_grad
    )
