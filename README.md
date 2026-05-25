<h1 align="center"><strong>H</strong><em>ierarchical</em> • <strong>Graph</strong> • <strong>M</strong><em>achine</em> • <strong>L</strong><em>earning</em></br>(HGraphML)</h1>


<p align="center">
  <a href="https://github.com/TYLERSFOSTER/HGraphML/actions/workflows/ci.yml">
    <img src="https://github.com/TYLERSFOSTER/HGraphML/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <!-- <a href="https://pypi.org/project/hgraphml/">
    <img src="https://img.shields.io/pypi/v/hgraphml" alt="PyPI">
  </a> -->
  <a href="https://pypi.org/project/hgraphml/">
    <img src="https://img.shields.io/pypi/pyversions/hgraphml" alt="Python">
  </a>
  <a href="https://github.com/TYLERSFOSTER/HGraphML/releases">
    <img src="https://img.shields.io/github/v/release/TYLERSFOSTER/HGraphML?label=release" alt="Release">
  </a>
  <!-- <a href="https://github.com/TYLERSFOSTER/HGraphML/issues">
    <img src="https://img.shields.io/github/issues/TYLERSFOSTER/HGraphML" alt="GitHub issues">
  </a> -->
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/badge/lint-ruff-46aef7" alt="Linted with Ruff">
  </a>
</p>

`HGraphML` is a lightweight Python package that realizes an insight that
[Abdullah N. Malik](https://abdullahnaeemmalik.github.io/) made while thinking
about the relationship between graph ML and the quotient-tower machinery behind
[`state_collapser`](https://github.com/TYLERSFOSTER/state_collapser). The basic
idea is that the same hierarchical graph-collapse tools used by
`state_collapser` for reinforcement-learning state/action graphs should also be
useful for graph ML systems where data flows across a known graph.

For `state_collapser`'s original RL application, the graph that data flows
across is the state/action graph of the underlying problem. The moving data can
be understood as the agent itself, as a Dirac delta of a single run, or as the
probabilistic flow induced by a policy over the state/action graph.

For `HGraphML`, the first target is graph message passing. A known graph is
handed to `state_collapser` as an already-discovered graph; `state_collapser`
builds a quotient tower; `HGraphML` runs message passing on a coarse tier; and a
lift maps coarse messages back to the fine graph. This is intended as the first
executable bridge toward hierarchical versions of graph ML methods such as
[belief propagation](https://en.wikipedia.org/wiki/Belief_propagation).

This repository does not yet claim speed-up. The current milestone is stricter
and smaller: prove that quotient-tower-backed graph message passing can sit
inside a trainable PyTorch computation without breaking autograd.

## Current Milestone

The package currently provides:

- a minimal `TensorGraph` surface for known directed graphs,
- a direct `state_collapser` adapter that builds quotient towers from known
  graphs,
- node-fiber and edge-fiber readouts from tower tiers,
- deterministic uniform and fiber-normalized lifts,
- a learned PyTorch lift,
- a small `collapse_messages(...)` orchestration call,
- a tiny train-step helper proving gradients and optimizer updates work.

The central call is:

```python
from hgraphml import collapse_messages

result = collapse_messages(
    graph=graph,
    node_features=node_features,
    message_model=message_model,
    lift=lift,
    contraction_schema=schema,
)
```

## Tiny Demo

```python
from hgraphml.examples.learned_lift_demo import run_learned_lift_demo

result = run_learned_lift_demo(steps=10, seed=0)
print(result.initial_loss, result.final_loss)
```

From a local checkout:

```bash
uv run --extra dev python -m hgraphml.examples.learned_lift_demo
```

## Installation

Python `3.11` or `3.12` is required.

For local development from this repository:

```bash
uv sync --extra dev --group dev
```

Then run the local checks:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run --extra dev mypy
```

The current development setup expects a sibling checkout of `state_collapser`:

```text
../state_collapser
```

That local editable source is declared in `pyproject.toml` for `uv` workflows.
Once `state-collapser` and `hgraphml` are both published in the required form,
this can become a normal package installation path.

## Quick Start

Run the built-in trainability demo:

```bash
uv run --extra dev python -m hgraphml.examples.learned_lift_demo
```

Expected output shape:

```text
initial_loss=...
final_loss=...
```

The exact values may change as the toy objective evolves. The important first
checks are that the run completes, the loss is finite, gradients reach trainable
parameters, and the optimizer changes those parameters.

## Real Minimal Use

The shortest useful example is:

```python
import torch
from state_collapser.tower.partition import LabelBlockSchema

from hgraphml import collapse_messages
from hgraphml.adapters import build_tower_bundle
from hgraphml.examples.toy_graphs import make_repeated_motif_graph, make_toy_node_features
from hgraphml.lifts import LearnedFiberLift
from hgraphml.messages import EdgeMessageMLP

graph = make_repeated_motif_graph()
node_features = make_toy_node_features(seed=0, feature_dim=8)

schema = LabelBlockSchema.from_labels(("motif",))
tower_bundle = build_tower_bundle(graph, contraction_schema=schema)

message_model = EdgeMessageMLP(feature_dim=8, message_dim=8, hidden_dim=32)
lift = LearnedFiberLift(message_dim=8, context_dim=16, hidden_dim=32)

result = collapse_messages(
    graph=graph,
    node_features=node_features,
    message_model=message_model,
    lift=lift,
    tower_bundle=tower_bundle,
)

loss = result.node_readout.pow(2).mean()
loss.backward()

print(result.coarse_messages.shape)
print(result.fine_messages.shape)
print(bool(torch.isfinite(loss)))
```

This example does the real package move:

```text
fine graph -> quotient tower -> coarse messages -> learned lift -> fine readout
```

## Train-Step Example

For a tiny supervised viability loop:

```python
import torch
from state_collapser.tower.partition import LabelBlockSchema

from hgraphml.adapters import build_tower_bundle
from hgraphml.examples.toy_graphs import make_repeated_motif_graph, make_toy_node_features
from hgraphml.lifts import LearnedFiberLift
from hgraphml.messages import EdgeMessageMLP
from hgraphml.training import make_teacher_node_targets, train_step

graph = make_repeated_motif_graph()
node_features = make_toy_node_features(seed=0, feature_dim=8)
target = make_teacher_node_targets(graph, node_features, output_dim=8, seed=100)

schema = LabelBlockSchema.from_labels(("motif",))
tower_bundle = build_tower_bundle(graph, contraction_schema=schema)

message_model = EdgeMessageMLP(feature_dim=8, message_dim=8, hidden_dim=32)
lift = LearnedFiberLift(message_dim=8, context_dim=16, hidden_dim=32)
optimizer = torch.optim.Adam(
    list(message_model.parameters()) + list(lift.parameters()),
    lr=0.01,
)

step = train_step(
    graph=graph,
    node_features=node_features,
    target=target,
    message_model=message_model,
    lift=lift,
    optimizer=optimizer,
    loss_fn=torch.nn.MSELoss(),
    tower_bundle=tower_bundle,
)

print(step.loss)
print(step.gradient_diagnostics.nonzero_gradient_count)
print(step.viability_diagnostics.parameters_changed)
```

## Current Implemented Surface

The package currently contains real code for:

- `hgraphml.graph`
  - `TensorGraph`, `NodeFiber`, `EdgeFiber`, and fiber-map aliases
- `hgraphml.adapters`
  - direct `state_collapser` tower construction and tensor-friendly fiber readouts
- `hgraphml.messages`
  - message containers, edge MLP message passing, node pooling, and readout
- `hgraphml.lifts`
  - uniform pullback, fiber-normalized, and learned fiber lifts
- `hgraphml.training`
  - teacher-target generation and a tiny train-step helper
- `hgraphml.diagnostics`
  - gradient and viability diagnostics
- `hgraphml.examples`
  - a repeated-motif graph and learned-lift trainability demo

The top-level public-feeling surface is intentionally small:

```python
from hgraphml import HGraphMLResult, __version__, collapse_messages
```

Most current entry points live in explicit subpackages.

## Repository Status

This project is `pre-alpha`.

What is solid enough to rely on:

- package layout and typed source structure
- local test/lint/type workflow
- the first direct `state_collapser` adapter
- deterministic lift baselines
- learned lift trainability
- end-to-end `collapse_messages(...)`
- toy training diagnostics

What should still be treated as unstable:

- broad public API shape
- lift semantics beyond the first implemented baselines
- benchmark and speed-up claims
- graph ML framework adapters
- long-term naming of some surfaces
- packaging/release workflow until the first public release is cut

## Project Structure

Current top-level source layout:

```text
src/hgraphml/
  adapters/
  diagnostics/
  examples/
  graph/
  lifts/
  messages/
  training/
```

Test layout mirrors the package areas:

```text
tests/
  adapters/
  diagnostics/
  examples/
  graph/
  lifts/
  messages/
  training/
```

## Documentation

Where to go next:

- New to the package: start with [`docs/usage/01_001_first_hack.md`](./docs/usage/01_001_first_hack.md).
- Looking for exact implemented surfaces: read [`docs/api_notes/01_001_first_surfaces.md`](./docs/api_notes/01_001_first_surfaces.md).
- Looking for the implementation record: read [`docs/design/01_004_hgraphml_first_import_implementation_log.md`](./docs/design/01_004_hgraphml_first_import_implementation_log.md).
- Planning to contribute: read [`CONTRIBUTING.md`](./CONTRIBUTING.md).

Design and implementation docs:

- [`docs/design/01_001_lifting_strategies_for_hierarchical_graph_message_passing.md`](./docs/design/01_001_lifting_strategies_for_hierarchical_graph_message_passing.md)
- [`docs/design/01_002_hgraphml_first_import_blueprint.md`](./docs/design/01_002_hgraphml_first_import_blueprint.md)
- [`docs/design/01_003_hgraphml_first_import_implementation_gameplan.md`](./docs/design/01_003_hgraphml_first_import_implementation_gameplan.md)
- [`docs/design/01_004_hgraphml_first_import_implementation_log.md`](./docs/design/01_004_hgraphml_first_import_implementation_log.md)

Related upstream package:

- [`state_collapser`](https://github.com/TYLERSFOSTER/state_collapser)

## Development

Common local checks:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run --extra dev mypy
```

The project expects:

- typed Python,
- tests for new runtime behavior,
- explicit handling of `state_collapser` compatibility,
- honest distinction between viability demos and benchmark claims,
- care with quotient, fiber, lift, and message-passing vocabulary.

## Current Non-Goals

`HGraphML` does not yet provide:

- speed-up benchmarks,
- exact belief propagation,
- PyTorch Geometric or DGL adapters,
- dynamic graph updates,
- batching/vectorization conventions,
- checkpointing,
- mature experiment tracking,
- GPU/device hardening beyond ordinary PyTorch tensor behavior.

Those are next-stage engineering targets, not current claims.

## License

This project is released under the [MIT License](./LICENSE).
