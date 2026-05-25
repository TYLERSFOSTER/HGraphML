"""Tests for viability diagnostics."""

from __future__ import annotations

import torch

from hgraphml.diagnostics import parameters_changed, snapshot_parameters


def test_parameters_changed_detects_update() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    before = snapshot_parameters((parameter,))

    with torch.no_grad():
        parameter.add_(1.0)

    assert parameters_changed(before, (parameter,))
