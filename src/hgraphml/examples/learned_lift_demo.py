"""Runnable learned-lift viability demo."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from state_collapser.tower.partition import LabelBlockSchema

from hgraphml.adapters import build_tower_bundle
from hgraphml.examples.toy_graphs import make_repeated_motif_graph, make_toy_node_features
from hgraphml.lifts import LearnedFiberLift
from hgraphml.messages import EdgeMessageMLP
from hgraphml.training import TrainStepResult, make_teacher_node_targets, train_step


@dataclass(frozen=True, slots=True)
class LearnedLiftDemoResult:
    """Summary of a tiny learned-lift run."""

    initial_loss: float
    final_loss: float
    steps: tuple[TrainStepResult, ...]


def run_learned_lift_demo(*, steps: int = 10, seed: int = 0) -> LearnedLiftDemoResult:
    """Run a tiny deterministic trainability proof."""

    if steps <= 0:
        raise ValueError("steps must be positive.")
    torch.manual_seed(seed)
    graph = make_repeated_motif_graph()
    node_features = make_toy_node_features(seed=seed, feature_dim=8)
    target = make_teacher_node_targets(graph, node_features, output_dim=8, seed=seed + 100)
    schema = LabelBlockSchema.from_labels(("motif",))
    tower_bundle = build_tower_bundle(graph, contraction_schema=schema)
    message_model = EdgeMessageMLP(feature_dim=8, message_dim=8, hidden_dim=32)
    lift = LearnedFiberLift(message_dim=8, context_dim=16, hidden_dim=32)
    optimizer = torch.optim.Adam(
        list(message_model.parameters()) + list(lift.parameters()),
        lr=0.01,
    )
    loss_fn = torch.nn.MSELoss()
    results = []
    for _ in range(steps):
        results.append(
            train_step(
                graph=graph,
                node_features=node_features,
                target=target,
                message_model=message_model,
                lift=lift,
                optimizer=optimizer,
                loss_fn=loss_fn,
                tower_bundle=tower_bundle,
            )
        )
    return LearnedLiftDemoResult(
        initial_loss=results[0].loss,
        final_loss=results[-1].loss,
        steps=tuple(results),
    )


def main() -> None:
    """CLI entry point for the demo."""

    result = run_learned_lift_demo()
    print(f"initial_loss={result.initial_loss:.6f}")
    print(f"final_loss={result.final_loss:.6f}")


if __name__ == "__main__":  # pragma: no cover
    main()
