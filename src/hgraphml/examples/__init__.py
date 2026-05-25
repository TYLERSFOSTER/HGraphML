"""Examples for HGraphML.

Runnable example modules are imported lazily so ``python -m`` can execute them
without a prior package-level import registering the module in ``sys.modules``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hgraphml.examples.toy_graphs import make_repeated_motif_graph, make_toy_node_features

if TYPE_CHECKING:
    from hgraphml.examples.learned_lift_demo import (
        LearnedLiftDemoResult,
        run_learned_lift_demo,
    )

__all__ = [
    "LearnedLiftDemoResult",
    "make_repeated_motif_graph",
    "make_toy_node_features",
    "run_learned_lift_demo",
]


def __getattr__(name: str) -> object:
    """Lazily expose runnable demo helpers."""

    if name in {"LearnedLiftDemoResult", "run_learned_lift_demo"}:
        from hgraphml.examples import learned_lift_demo

        return getattr(learned_lift_demo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
