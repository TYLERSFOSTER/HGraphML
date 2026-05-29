# State-Collapser Encoding Compatibility Blueprint

Date: 2026-05-29

Status: blueprint, not implementation gameplan

Source documents:

- `docs/design/tensorization/01_005_hgraphml_state_collapser_tensorization_followup_instructions.md`
- `state_collapser/docs/design/tensorization/01_005_hgraphml_tensorization_followup_bridge.md`
- `docs/api_notes/01_001_first_surfaces.md`
- `docs/usage/01_001_first_hack.md`
- `docs/design/01_004_hgraphml_first_import_implementation_log.md`

## Purpose

This blueprint defines the HGraphML-side follow-up for the
`state_collapser` tensorization boundary.

The resolved scope is narrow:

```text
HGraphML gets first-pass compatibility coverage,
not first-pass full tensorization.
```

The goal is to prove that HGraphML can consume the shared upstream
`EncodingRegistry` vocabulary from a real HGraphML-built `TowerBundle` without
turning graph message passing into an RL action-selection pipeline.

## Current Reality

HGraphML currently has a working first graph-message-passing vertical slice:

```text
TensorGraph
    -> HGraphML state_collapser adapter
        -> PartitionTower
            -> coarse TensorGraph readouts
                -> message model
                    -> lift
                        -> fine readout / train step
```

The adapter lives at:

```text
src/hgraphml/adapters/state_collapser.py
```

It already builds:

- `TowerBundle`
- `node_fibers_by_tier`
- `edge_fibers_by_tier`
- `coarse_graphs_by_tier`

That architecture remains correct.

Upstream `state_collapser` has now released the tensorization boundary as:

```text
state_collapser v0.7.0
```

That release exports:

```python
from state_collapser.training import EncodingRegistry
```

and provides:

```python
EncodingRegistry.from_tower(...)
```

HGraphML still currently pins:

```text
state-collapser @ git+https://github.com/TYLERSFOSTER/state_collapser.git@v0.6.0
```

in:

```text
pyproject.toml
uv.lock
```

Therefore, the implementation must begin by updating HGraphML's dependency to
an installable upstream target that exposes `EncodingRegistry`.

## Design Target

The desired relationship is:

```text
TensorGraph
    -> build_tower_bundle(...)
        -> TowerBundle.partition_tower
            -> EncodingRegistry.from_tower(...)
                -> stable numeric ids for graph-message-passing infrastructure
```

The first compatibility pass should prove that HGraphML can reuse stable ids
for:

- original graph nodes through upstream `State` encodings;
- original graph edges through upstream `BaseEdge` encodings;
- tower tiers;
- state cells;
- action cells and action collections where present;
- HGraphML node-fiber and edge-fiber assumptions.

This is preparation for later tensor-aware graph message passing. It is not
that later graph message-passing implementation.

## Explicit Non-Target

The forbidden relationship is:

```text
TensorGraph
    -> fake RL ActionSelectionInput
        -> linearize_action_selection_input(...)
            -> TorchDecisionBatch
                -> graph message passing
```

HGraphML must not construct fake RL records to use the encoding registry.

The following upstream symbols are out of scope for the first HGraphML
compatibility pass:

- `ActionSelectionInput`
- `TrainingTransition`
- `LinearizedActionSelectionInput`
- `LinearizedTrainingTransition`
- `TorchDecisionBatch`
- `TorchTransitionBatch`
- `ActionDecision`

These are RL and learner-boundary surfaces. HGraphML's first pass only needs
the shared tower encoding vocabulary.

## Proposed Package Surface

The minimum first pass can be test-only plus documentation.

If a runtime helper is useful, add:

```python
def build_encoding_registry(bundle: TowerBundle) -> EncodingRegistry:
    return EncodingRegistry.from_tower(bundle.partition_tower)
```

Preferred location:

```text
src/hgraphml/adapters/state_collapser.py
```

Preferred export:

```text
src/hgraphml/adapters/__init__.py
```

This helper is preferable to adding an `encoding_registry` field to
`TowerBundle` because it:

- keeps `TowerBundle`'s object shape stable;
- avoids making registry construction mandatory;
- makes the bridge explicit at the adapter boundary;
- lets tests and downstream code opt into the new vocabulary.

Do not add an `encoding_registry` field to `TowerBundle` in the first pass
unless the implementation gameplan explicitly revisits that decision.

## Phase 0: Dependency Gate

### Goal

Make HGraphML install a `state_collapser` dependency that exposes
`EncodingRegistry`.

### Required Checks

Before implementation, record:

```bash
git status --short
git log --oneline --decorate --max-count=5
```

Then update the dependency target from:

```text
v0.6.0
```

to:

```text
v0.7.0
```

in:

```text
pyproject.toml
uv.lock
```

After syncing, this command must pass:

```bash
uv run --extra dev python -c "from state_collapser.training import EncodingRegistry; print(EncodingRegistry.__name__)"
```

Expected output:

```text
EncodingRegistry
```

### Stop Condition

If HGraphML cannot import `EncodingRegistry` from the selected dependency, stop.

Do not work around the failure by reimplementing `EncodingRegistry` locally.

## Phase 1: Compatibility Tests

### Test File

Add:

```text
tests/adapters/test_state_collapser_encoding_registry_compatibility.py
```

### Test 1: Import Shape

The first test should verify:

```python
from state_collapser.training import EncodingRegistry
```

It should not import RL-specific training or Torch batch surfaces.

Completion criterion:

```text
HGraphML can import the shared encoding surface without touching RL-specific
training records.
```

### Test 2: Registry From Existing TowerBundle

Use HGraphML's existing adapter path:

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
- every original graph node has a corresponding encoded upstream state id.
- every original graph edge has a corresponding encoded upstream edge id.
- at least one state-cell id is encodable at each tower tier.
- at least one action-cell id is encodable when the tower has action cells.

### Test 3: HGraphML Fiber Compatibility

Extend the same file to verify HGraphML's existing fiber maps remain compatible
with the registry.

For every tier in `bundle.node_fibers_by_tier`:

- each coarse node has a `NodeFiber`;
- each `NodeFiber.fine_nodes` entry maps back to a graph node;
- the corresponding tier state cell can be encoded through the registry.

For every tier in `bundle.edge_fibers_by_tier`:

- each coarse edge has an `EdgeFiber`;
- each `EdgeFiber.fine_edges` entry maps back to a graph edge;
- the corresponding upstream base edge can be encoded through the registry.

Completion criterion:

```text
EncodingRegistry.from_tower(...) does not disturb HGraphML's node-fiber and
edge-fiber readout assumptions.
```

## Phase 2: Optional Runtime Helper

If the compatibility tests need a named HGraphML surface, add:

```python
def build_encoding_registry(bundle: TowerBundle) -> EncodingRegistry:
    return EncodingRegistry.from_tower(bundle.partition_tower)
```

This helper should be covered by the compatibility test file.

The helper should not change:

- `TowerBundle` fields;
- `build_tower_bundle(...)` return shape;
- `collapse_messages(...)`;
- lift behavior;
- message-passing behavior.

If implementation reality makes the helper awkward, skip it. A test-only
compatibility pass is acceptable.

## Phase 3: Documentation

### API Notes

Update:

```text
docs/api_notes/01_001_first_surfaces.md
```

Add a short section:

```text
State-Collapser Encoding Compatibility
```

It should say:

- HGraphML uses `state_collapser` for quotient-tower construction.
- HGraphML can build `EncodingRegistry.from_tower(...)` from a `TowerBundle`.
- This is shared tower encoding, not RL tensorization.
- HGraphML does not require `ActionSelectionInput` or `TrainingTransition`.
- HGraphML does not route graph message passing through `TorchDecisionBatch`.

### README

After HGraphML pins a dependency that exposes `EncodingRegistry`, update the
README near the existing `state_collapser` dependency discussion.

Suggested wording:

```text
HGraphML also checks compatibility with state_collapser's shared
EncodingRegistry, which provides stable numeric ids for tower states, edges,
cells, and tiers. HGraphML uses this as graph-message-passing infrastructure,
not as an RL ActionSelectionInput tensorization path.
```

Do not make this sound like:

- full tensorized graph message passing is complete;
- HGraphML has benchmark-proven acceleration;
- HGraphML has become an RL learner package.

## Validation

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

If dependency sync requires network access, use the ordinary project dependency
workflow and record the exact command and outcome in the implementation log.

## Acceptance Criteria

This follow-up is complete when:

- HGraphML pins or otherwise uses a `state_collapser` dependency exposing
  `EncodingRegistry`;
- HGraphML has a test importing `state_collapser.training.EncodingRegistry`;
- HGraphML can build an encoding registry from a real `TowerBundle`;
- HGraphML verifies graph nodes and graph edges remain encodable;
- HGraphML verifies node fibers and edge fibers remain compatible;
- no HGraphML code constructs fake RL `ActionSelectionInput` objects;
- no HGraphML code depends on `TrainingTransition`;
- no HGraphML code routes graph message passing through `TorchDecisionBatch`;
- documentation explains that this is shared tower encoding, not RL
  tensorization;
- full tests, Ruff, and mypy pass.

## Deferred Work

Explicitly defer:

- tensorized graph-message-passing kernels based on `EncodingRegistry`;
- replacing `TensorGraph` ids with `state_collapser` ids throughout the package;
- using `LinearizedActionSelectionInput` in HGraphML;
- using `LinearizedTrainingTransition` in HGraphML;
- using `TorchDecisionBatch` in HGraphML;
- adding an RL learner path;
- benchmarking speed-up from registry compatibility;
- changing lift semantics;
- changing `collapse_messages(...)`;
- changing the `TowerBundle` dataclass shape.

## Blueprint Conclusion

The upstream dependency blocker has been resolved by `state_collapser v0.7.0`.

The HGraphML work is now ready for an implementation gameplan.

The implementation should stay small: update the dependency, prove registry
compatibility from the existing tower bundle, optionally expose a helper, and
document the boundary.

