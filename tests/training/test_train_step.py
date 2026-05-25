"""Tests for train_step."""

from __future__ import annotations

import torch
from state_collapser.tower.partition import LabelBlockSchema

from hgraphml.adapters import build_tower_bundle
from hgraphml.examples import make_repeated_motif_graph, make_toy_node_features
from hgraphml.lifts import LearnedFiberLift
from hgraphml.messages import EdgeMessageMLP
from hgraphml.training import make_teacher_node_targets, train_step


def test_train_step_updates_parameters_and_reports_diagnostics() -> None:
    torch.manual_seed(0)
    graph = make_repeated_motif_graph()
    features = make_toy_node_features(seed=0)
    target = make_teacher_node_targets(graph, features, output_dim=8, seed=1)
    bundle = build_tower_bundle(graph, contraction_schema=LabelBlockSchema.from_labels(("motif",)))
    model = EdgeMessageMLP(feature_dim=8, message_dim=8, hidden_dim=16)
    lift = LearnedFiberLift(message_dim=8, context_dim=16, hidden_dim=16)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(lift.parameters()), lr=0.01)

    result = train_step(
        graph=graph,
        node_features=features,
        target=target,
        message_model=model,
        lift=lift,
        optimizer=optimizer,
        loss_fn=torch.nn.MSELoss(),
        tower_bundle=bundle,
    )

    assert result.viability_diagnostics.loss_is_finite
    assert result.viability_diagnostics.parameters_changed
    assert result.gradient_diagnostics.nonzero_gradient_count > 0
