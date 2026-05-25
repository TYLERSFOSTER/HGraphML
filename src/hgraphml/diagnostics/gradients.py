"""Gradient diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class GradientDiagnostics:
    """Summary of gradient presence and magnitude."""

    parameter_count: int
    parameters_with_grad: int
    nonzero_gradient_count: int
    total_gradient_norm: float


def collect_gradient_diagnostics(
    parameters: Iterable[nn.Parameter],
) -> GradientDiagnostics:
    """Collect simple gradient diagnostics for trainable parameters."""

    parameter_count = 0
    parameters_with_grad = 0
    nonzero_gradient_count = 0
    total_gradient_norm = 0.0
    for parameter in parameters:
        if not parameter.requires_grad:
            continue
        parameter_count += 1
        if parameter.grad is None:
            continue
        parameters_with_grad += 1
        grad_norm = float(torch.linalg.vector_norm(parameter.grad).item())
        total_gradient_norm += grad_norm
        if grad_norm > 0.0:
            nonzero_gradient_count += 1
    return GradientDiagnostics(
        parameter_count=parameter_count,
        parameters_with_grad=parameters_with_grad,
        nonzero_gradient_count=nonzero_gradient_count,
        total_gradient_norm=total_gradient_norm,
    )
