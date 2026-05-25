# HGraphML First Import Implementation Log

## Status

Implementation log for:

```text
docs/design/01_003_hgraphml_first_import_implementation_gameplan.md
```

Branch:

```text
codex/hgraphml-first-import
```

Implementation date:

```text
2026-05-25
```

## Executive Summary

The first `HGraphML` package milestone has been implemented.

The repository now contains a real Python package under `src/hgraphml` that
connects:

```text
known graph data
    -> direct state_collapser quotient tower construction
    -> coarse graph message passing
    -> deterministic or learned lift back to fine edges
    -> fine graph readout
    -> PyTorch loss/backprop/optimizer step
```

The milestone is viability-first. It does not claim speed-up. It proves that the
core architectural bridge is executable and differentiable.

## Package Skeleton Decisions

Created:

```text
pyproject.toml
src/hgraphml/
tests/
```

Package metadata:

- package name: `hgraphml`,
- version: `0.1.0`,
- Python: `>=3.11,<3.13`,
- build backend: `hatchling`,
- runtime dependencies: `torch`, `state-collapser`,
- local development source: an unpublished `state_collapser` checkout was used
  during initial implementation; committed public-release metadata should not
  assume any machine-specific sibling path.

Development tooling:

- `pytest`,
- `ruff`,
- `mypy`,
- typed package marker `py.typed`.

## Direct `state_collapser` Adapter Decision

The gameplan contained a non-negotiable stop condition: do not silently
reimplement `state_collapser` inside `HGraphML`.

Direct import worked, so implementation proceeded without fallback.

Used `state_collapser` surfaces:

```python
from state_collapser.core.action import PrimitiveAction
from state_collapser.core.edges import BaseEdge
from state_collapser.core.state import State
from state_collapser.tower.partition import ContractionSchema, PartitionTower
from state_collapser.tower.partition.tower import build_partition_tower_full
```

The adapter lives at:

```text
src/hgraphml/adapters/state_collapser.py
```

The adapter treats the HGraphML graph as fully explored:

- every HGraphML node is registered as a `state_collapser.core.state.State`,
- every HGraphML directed edge is registered as a `BaseEdge`,
- edge labels are forwarded into `PrimitiveAction` and `BaseEdge` labels,
- the supplied contraction schema is passed to `build_partition_tower_full`,
- no environment loop or incremental exploration surface is used.

## ID Mapping

HGraphML node IDs are integer positions in `TensorGraph`:

```text
0, 1, ..., node_count - 1
```

They are mapped to `State` objects as:

```python
State(payload=("node", node_index), identity=("node", node_index))
```

HGraphML edge IDs are integer positions in `edge_index`:

```text
0, 1, ..., edge_count - 1
```

They are mapped to actions and base edges as:

```python
PrimitiveAction(
    payload=("edge", edge_index),
    identity=("edge", edge_index),
    labels=labels,
)

BaseEdge(
    source=states[source_index],
    action=action,
    target=states[target_index],
    labels=labels,
)
```

Reverse maps are then built from registered `State`/`BaseEdge` objects back to
HGraphML integer node and edge indices.

## Node Fiber Recovery

For each tower tier, HGraphML reads the active `state_layer` from the
`PartitionTower`.

Each active state cell becomes one compact HGraphML coarse node. Its fine-node
fiber is recovered by reading the members of the state cell and mapping those
state IDs back to original HGraphML node indices.

The resulting surface is:

```python
dict[int, NodeFiber]
```

where the keys are compact coarse node IDs.

## Edge Fiber Recovery

The first implementation intentionally does not expose internal transient
action-cell IDs as stable HGraphML coarse-edge IDs.

During validation, the initial action-cell-based readout failed because
`state_collapser` action layers can retain transitional/internal action-cell
labels that are meaningful inside the tower but not appropriate as HGraphML
coarse graph endpoints.

The corrected HGraphML rule is:

```text
At tier i, group original fine edges by the active tier-i state cells of their
source and target endpoints.
```

So for each original edge:

```text
u -> v
```

HGraphML finds:

```text
cell_i(u), cell_i(v)
```

then groups all fine edges with the same coarse endpoint pair. Each endpoint
pair becomes a compact HGraphML coarse edge with an `EdgeFiber` listing all fine
edges represented by that coarse edge.

This is the correct graph-message-passing readout because it aligns coarse
message edges with the active quotient graph induced by the tier's state
partition.

## Implemented Source Surfaces

Graph surfaces:

- `TensorGraph`,
- `NodeFiber`,
- `EdgeFiber`,
- `NodeFiberMap`,
- `EdgeFiberMap`,
- `TowerBundle`,
- `build_tower_bundle(...)`.

Message surfaces:

- `MessageBatch`,
- `EdgeMessageMLP`,
- `run_edge_message_model(...)`,
- `mean_pool_node_features(...)`,
- `incoming_sum_readout(...)`.

Lift surfaces:

- `MessageLift`,
- `UniformPullbackLift`,
- `FiberNormalizedLift`,
- `LearnedFiberLift`.

Orchestration:

- `collapse_messages(...)`,
- `HGraphMLResult`.

Training and diagnostics:

- `GradientDiagnostics`,
- `ViabilityDiagnostics`,
- `collect_gradient_diagnostics(...)`,
- `snapshot_parameters(...)`,
- `parameters_changed(...)`,
- `make_teacher_node_targets(...)`,
- `train_step(...)`,
- `TrainStepResult`.

Example surfaces:

- `make_repeated_motif_graph(...)`,
- `make_toy_node_features(...)`,
- `run_learned_lift_demo(...)`,
- `LearnedLiftDemoResult`.

## Documentation Added

README now explains:

- how `HGraphML` relates to `state_collapser`,
- why the first target is graph message passing,
- what the current milestone proves,
- that no speed-up claim is made yet,
- how to run the tiny demo,
- where to find deeper docs.

Usage doc:

```text
docs/usage/01_001_first_hack.md
```

API notes:

```text
docs/api_notes/01_001_first_surfaces.md
```

## Tests Added

The test suite covers:

- package import/version,
- `TensorGraph` validation,
- node and edge fiber validation,
- message container validation,
- direct `state_collapser` adapter behavior,
- deterministic lifts,
- learned lift shape/weights/gradients,
- message model and pooling,
- readout,
- `collapse_messages(...)`,
- gradient diagnostics,
- viability diagnostics,
- teacher target determinism,
- train-step viability,
- toy graph construction,
- learned-lift demo viability.

## Validation Results

Full test suite:

```text
uv run --extra dev pytest
36 passed
```

Lint:

```text
uv run --extra dev ruff check .
All checks passed
```

Typing:

```text
uv run --extra dev mypy
Success: no issues found in 27 source files
```

Demo:

```text
uv run --extra dev python -m hgraphml.examples.learned_lift_demo
initial_loss=0.216840
final_loss=0.010117
```

The demo also emitted a PyTorch warning about NumPy not being installed in the
local virtual environment. This warning did not affect the milestone validation.

## Gameplan Completion Audit

Phase 0, branch and baseline reality:

```text
completed
```

The work was performed on `codex/hgraphml-first-import`. Existing README and
docs changes were treated as Project Owner context rather than reset.

Phase 1, package skeleton and tooling:

```text
completed
```

Phase 2, tensor graph and fiber data:

```text
completed
```

Phase 3, early `state_collapser` adapter gate:

```text
completed
```

Direct import worked. No fallback tower implementation was introduced.

Phase 4, lift operators:

```text
completed
```

Phase 5, message passing and readout:

```text
completed
```

Phase 6, `collapse_messages(...)`:

```text
completed
```

`refinement_steps` is present as a reserved argument but raises
`NotImplementedError` unless set to `0`.

Phase 7, training helpers and diagnostics:

```text
completed
```

Phase 8, toy example and viability benchmark:

```text
completed
```

The benchmark is viability-only, not speed-oriented.

Phase 9, documentation:

```text
completed
```

README, usage doc, API notes, and this implementation log were added or
updated.

Phase 10, validation and implementation log:

```text
completed
```

Phase 11, final review:

```text
completed
```

## Known Limitations

This milestone does not include:

- speed-up benchmarks,
- exact belief propagation,
- PyG/DGL adapters,
- dynamic graph updates,
- checkpointing,
- batching/vectorization conventions,
- mature experiment manifests,
- GPU/device hardening beyond ordinary PyTorch tensor behavior.

The learned lift is intentionally tiny. It proves differentiability and training
viability, not modeling quality.

## Next Recommended Work

Recommended next steps:

1. Add a first real graph ML example that is not the toy repeated-motif graph.
2. Design exact-vs-approximate lift semantics more explicitly.
3. Add a benchmark harness that compares flat message passing with quotient
   message passing under controlled graph families.
4. Decide whether the next adapter should target PyTorch Geometric, belief
   propagation directly, or a hand-rolled small factor-graph surface.
5. Add artifact output for benchmark runs before making any public speed-up
   claim.

## Final Implementation Note

The central package claim is now executable:

```text
HGraphML can use state_collapser quotient towers as scaffolding for trainable
hierarchical graph message passing.
```

The implementation is intentionally modest, but the bridge is real.
