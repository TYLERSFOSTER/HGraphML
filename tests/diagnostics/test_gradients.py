"""Tests for gradient diagnostics."""

from __future__ import annotations

import torch

from hgraphml.diagnostics import collect_gradient_diagnostics


def test_collect_gradient_diagnostics_counts_nonzero_gradients() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    loss = parameter.sum()
    loss.backward()

    diagnostics = collect_gradient_diagnostics((parameter,))

    assert diagnostics.parameter_count == 1
    assert diagnostics.parameters_with_grad == 1
    assert diagnostics.nonzero_gradient_count == 1
    assert diagnostics.total_gradient_norm > 0.0
