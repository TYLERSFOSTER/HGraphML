"""Tiny graph fixtures for the first HGraphML proof."""

from __future__ import annotations

import torch

from hgraphml.graph import TensorGraph


def make_repeated_motif_graph() -> TensorGraph:
    """Return two repeated motifs connected by one bridge edge."""

    edge_index = torch.tensor(
        [
            [0, 1, 0, 3, 4, 3, 2],
            [1, 2, 2, 4, 5, 5, 3],
        ],
        dtype=torch.long,
    )
    return TensorGraph(
        node_count=6,
        edge_index=edge_index,
        edge_labels=("motif", "motif", "motif", "motif", "motif", "motif", "bridge"),
    )


def make_toy_node_features(*, seed: int = 0, feature_dim: int = 8) -> torch.Tensor:
    """Return deterministic random node features for the toy graph."""

    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive.")
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((6, feature_dim), generator=generator)
