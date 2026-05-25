"""Tests for learned lift demo."""

from __future__ import annotations

import math

from hgraphml.examples import run_learned_lift_demo


def test_learned_lift_demo_runs_and_trains() -> None:
    result = run_learned_lift_demo(steps=3, seed=0)

    assert math.isfinite(result.initial_loss)
    assert math.isfinite(result.final_loss)
    assert result.steps[-1].gradient_diagnostics.nonzero_gradient_count > 0
    assert result.steps[-1].viability_diagnostics.parameters_changed
