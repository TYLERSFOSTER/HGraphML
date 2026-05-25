"""Viability diagnostics for the first training proof."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

ParameterSnapshot = tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class ViabilityDiagnostics:
    """Basic finite-loss and parameter-change diagnostics."""

    loss_is_finite: bool
    parameters_changed: bool
    fine_message_shape: tuple[int, ...]
    coarse_message_shape: tuple[int, ...]


def snapshot_parameters(parameters: Iterable[nn.Parameter]) -> ParameterSnapshot:
    """Return detached copies of trainable parameters."""

    return tuple(
        parameter.detach().clone()
        for parameter in parameters
        if parameter.requires_grad
    )


def parameters_changed(
    before: ParameterSnapshot,
    parameters: Iterable[nn.Parameter],
) -> bool:
    """Return whether any trainable parameter differs from its snapshot."""

    after = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if len(before) != len(after):
        return True
    return any(not torch.equal(old, new.detach()) for old, new in zip(before, after, strict=True))
