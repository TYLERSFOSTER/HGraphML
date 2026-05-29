# HGraphML Follow-Up Instructions For `state_collapser` Tensorization

Date: 2026-05-29

Status: downstream implementation instructions, not yet executed

## Purpose

This document captures the HGraphML-side follow-up implied by the
`state_collapser` tensorization discussion and implementation.

The resolved decision was:

```text
HGraphML gets first-pass compatibility coverage,
not first-pass full tensorization.
```

That means the immediate HGraphML work is not to turn graph message passing into
an RL-style `ActionSelectionInput -> model -> ActionDecision` pipeline.

The immediate HGraphML work is to protect and prepare the shared tower-encoding
surface so that HGraphML can later reuse the same stable numeric vocabulary for
graph message passing.

## Upstream Context

The relevant upstream `state_collapser` work is on the branch:

```text
codex/tensorization-boundary
```

The planned upstream surface is:

```python
from state_collapser.training import EncodingRegistry
from state_collapser.training import LinearizationConfig
from state_collapser.training import LinearizationState
from state_collapser.training import NumericBackend
from state_collapser.training import TensorDeviceKind
```

The HGraphML-relevant object is:

```python
EncodingRegistry
```

The HGraphML-non-relevant first-scope objects are:

```python
ActionSelectionInput
TrainingTransition
TorchDecisionBatch
TorchTransitionBatch
ActionDecision
```

Those objects are RL/training-boundary surfaces. HGraphML should not be forced
through them.

## Core Instruction

HGraphML should treat `state_collapser.training.EncodingRegistry` as a shared
tower/cell/edge vocabulary, not as an RL tensorization API.

The desired downstream relationship is:

```text
TensorGraph
    -> HGraphML state_collapser adapter
        -> PartitionTower
            -> EncodingRegistry.from_tower(...)
                -> graph-message-passing-compatible numeric ids
```

The forbidden downstream relationship is:

```text
TensorGraph
    -> fake RL ActionSelectionInput
        -> linearize_action_selection_input(...)
            -> TorchDecisionBatch
                -> graph message passing
```

That would be a category mistake. HGraphML is not trying to pretend graph
message passing is an RL action-decision surface.

## Why This Matters

The `state_collapser` tensorization work introduced a general numeric boundary.
Part of the point of the design was that the shared encoding vocabulary should
not be only an RL observation/action encoder.

HGraphML is the reason that matters.

For graph ML, the relevant stable ids are:

- graph node ids
- graph edge ids
- state-cell ids
- action-cell ids
- action-collection ids
- tower-tier ids
- node-fiber ids or coarse-node ids
- edge-fiber ids or coarse-edge ids

Those are exactly the objects HGraphML already manipulates in
`TowerBundle`, `NodeFiber`, `EdgeFiber`, and `TensorGraph`.

## Current HGraphML Reality

The current package already has:

- `src/hgraphml/adapters/state_collapser.py`
- `TowerBundle`
- `build_tower_bundle(...)`
- `TensorGraph`
- `NodeFiber`
- `EdgeFiber`
- `collapse_messages(...)`
- deterministic and learned lifts
- train-step tests
- state-collapser-backed quotient-tower construction

The current adapter builds a full known graph as an already-discovered
`state_collapser` graph:

```text
TensorGraph
    -> State / BaseEdge wrappers
        -> build_partition_tower_full(...)
            -> node_fibers_by_tier
            -> edge_fibers_by_tier
            -> coarse_graphs_by_tier
```

That is still the correct architecture.

## What To Do In HGraphML

## Phase 0: Dependency And Branch Discipline

### Stage 0.1: Wait For Upstream Availability

Do not implement this until `state_collapser` exposes `EncodingRegistry` on a
commit or tag that HGraphML can install.

Acceptable upstream sources:

- local editable checkout during development
- a branch reference while actively testing
- a public tag after `state_collapser` release

Do not make HGraphML depend permanently on a short-lived Codex branch.

### Stage 0.2: Create A Dedicated Branch

Use a branch such as:

```text
codex/state-collapser-tensorization-compat
```

Record:

```bash
git status --short
git log --oneline --decorate --max-count=5
```

before edits.

### Stage 0.3: Confirm Dependency Version

Inspect:

```text
pyproject.toml
uv.lock
```

Confirm whether the pinned `state-collapser` dependency contains:

```python
state_collapser.training.EncodingRegistry
```

If not, either:

- update the dependency pin to the correct released tag, or
- use a local editable dependency only for development.

Do not claim public compatibility until the public dependency pin contains the
surface being tested.

## Phase 1: Add A Narrow Compatibility Test

### Stage 1.1: Test Import Shape

Add a test file:

```text
tests/adapters/test_state_collapser_encoding_registry_compatibility.py
```

The first test should verify:

```python
from state_collapser.training import EncodingRegistry
```

This test must not import:

```python
ActionSelectionInput
TrainingTransition
TorchDecisionBatch
TorchTransitionBatch
```

Completion criterion:

```text
HGraphML can import the shared encoding surface without touching RL-specific
training records.
```

### Stage 1.2: Test Registry Construction From Existing TowerBundle

Use the existing HGraphML adapter path:

```python
from state_collapser.training import EncodingRegistry
from state_collapser.tower.partition import LabelBlockSchema

from hgraphml.adapters import build_tower_bundle
from hgraphml.examples.toy_graphs import make_repeated_motif_graph
```

Test shape:

```python
graph = make_repeated_motif_graph()
schema = LabelBlockSchema.from_labels(("motif",))
bundle = build_tower_bundle(graph, contraction_schema=schema)
registry = EncodingRegistry.from_tower(bundle.partition_tower)
```

Assertions:

- `registry.registry_id` is a nonempty string.
- `registry.to_dict()` is JSON-safe.
- every original graph node has a corresponding encoded state id.
- every original graph edge has a corresponding encoded edge id.
- at least one state-cell id is encodable at each tower tier.
- at least one action-cell id is encodable if the tower has action cells.

Important:

The test should not build fake RL action-selection objects.

### Stage 1.3: Test HGraphML Fiber Compatibility

Extend the same test file to verify that HGraphML's own fiber maps remain
compatible with the registry.

For every tier in `bundle.node_fibers_by_tier`:

- each coarse node still has a `NodeFiber`
- each `NodeFiber.fine_nodes` entry maps back to a graph node
- the corresponding state cell can be encoded through the upstream registry

For every tier in `bundle.edge_fibers_by_tier`:

- each coarse edge still has an `EdgeFiber`
- each `EdgeFiber.fine_edges` entry maps back to a graph edge
- the corresponding base edge can be encoded through the upstream registry

Completion criterion:

```text
The new upstream registry does not disturb HGraphML's node-fiber and edge-fiber
readout assumptions.
```

## Phase 2: Optionally Attach Encoding Metadata To TowerBundle

This phase is optional for the first compatibility pass.

Only do it if it makes HGraphML easier to use without creating tight coupling.

### Stage 2.1: Add Optional Field

Consider extending `TowerBundle`:

```python
encoding_registry: EncodingRegistry | None = None
```

Then update `build_tower_bundle(...)`:

```python
encoding_registry = EncodingRegistry.from_tower(tower)
```

and include it in the returned bundle.

Pros:

- downstream code can access the stable numeric vocabulary immediately
- avoids repeated registry construction
- makes compatibility visible in the adapter object

Cons:

- imports `state_collapser.training.EncodingRegistry` directly in the adapter
- makes `TowerBundle` depend on a newer state-collapser surface
- may force HGraphML to raise its minimum state-collapser pin immediately

Recommendation:

Do not add this field in the first follow-up unless the test-only compatibility
pass feels too weak.

### Stage 2.2: Safer Alternative

Prefer a helper:

```python
def build_encoding_registry(bundle: TowerBundle) -> EncodingRegistry:
    return EncodingRegistry.from_tower(bundle.partition_tower)
```

Potential location:

```text
src/hgraphml/adapters/state_collapser.py
```

Potential export:

```text
src/hgraphml/adapters/__init__.py
```

This keeps `TowerBundle` stable while making the new bridge explicit.

Recommendation:

If HGraphML code needs a runtime surface, implement the helper rather than
changing `TowerBundle` fields.

## Phase 3: Add Documentation

### Stage 3.1: Update API Notes

Update:

```text
docs/api_notes/01_001_first_surfaces.md
```

Add a short section:

```text
State-collapser Encoding Compatibility
```

It should say:

- HGraphML uses `state_collapser` for quotient-tower construction.
- HGraphML may use `EncodingRegistry.from_tower(...)` for stable numeric ids.
- This is not RL tensorization.
- HGraphML does not require `ActionSelectionInput` or `TrainingTransition`.
- HGraphML does not route graph message passing through `TorchDecisionBatch`.

### Stage 3.2: Update README If Public Dependency Supports It

Only update README after HGraphML's dependency pin points to a
`state_collapser` commit/tag containing `EncodingRegistry`.

Add a small note near the current `state_collapser` dependency discussion:

```text
HGraphML also checks compatibility with state_collapser's shared
EncodingRegistry, which provides stable numeric ids for tower states, edges,
cells, and tiers. HGraphML uses this as graph-message-passing infrastructure,
not as an RL ActionSelectionInput tensorization path.
```

Do not make this sound like a speed-up claim.

Do not make this sound like full tensorized graph message passing is complete.

## Phase 4: Do Not Implement Full HGraphML Tensorization Yet

The following are explicitly deferred:

- tensorized graph-message-passing kernels based on `EncodingRegistry`
- replacing `TensorGraph` ids with state-collapser ids throughout the package
- using `LinearizedActionSelectionInput` in HGraphML
- using `LinearizedTrainingTransition` in HGraphML
- using `TorchDecisionBatch` in HGraphML
- adding an RL learner path
- benchmarking claimed speed-up from tensorized registry use
- changing lift semantics
- changing `collapse_messages(...)`

The first compatibility pass is about preserving the interface and preparing
the future path.

## Phase 5: Validation

Run:

```bash
uv run --extra dev pytest tests/adapters/test_state_collapser_adapter.py
uv run --extra dev pytest tests/adapters/test_state_collapser_encoding_registry_compatibility.py
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run --extra dev mypy
```

Expected result:

```text
all tests pass
ruff passes
mypy passes
```

If the new compatibility test fails because HGraphML's state-collapser
dependency pin does not expose `EncodingRegistry`, update the dependency pin or
defer the test until the correct upstream release exists.

Do not work around that by reimplementing `EncodingRegistry` in HGraphML.

## Final Acceptance Criteria

This HGraphML follow-up is complete when:

- HGraphML has a test importing `state_collapser.training.EncodingRegistry`.
- HGraphML can build an encoding registry from a real `TowerBundle`.
- HGraphML can verify graph nodes and graph edges remain encodable.
- HGraphML can verify node fibers and edge fibers remain compatible.
- No HGraphML code constructs fake RL `ActionSelectionInput` objects.
- No HGraphML code depends on `TrainingTransition`.
- No HGraphML code routes graph message passing through `TorchDecisionBatch`.
- Documentation explains that this is shared tower encoding, not RL
  tensorization.
- Full HGraphML tests, Ruff, and mypy pass.

## Conceptual Summary

The state-collapser tensorization work creates two different downstream
possibilities:

```text
RL/counterpoint:
    ActionSelectionInput
        -> LinearizedActionSelectionInput
            -> TorchDecisionBatch
```

and:

```text
HGraphML:
    TensorGraph
        -> PartitionTower
            -> EncodingRegistry
                -> graph-message-passing-compatible stable ids
```

HGraphML belongs to the second path.

The whole point of the HGraphML follow-up is to protect that distinction before
the next round of tensor work begins.
