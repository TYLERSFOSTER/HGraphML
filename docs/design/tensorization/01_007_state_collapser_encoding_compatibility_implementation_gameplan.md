# State-Collapser Encoding Compatibility Implementation Gameplan

Date: 2026-05-29

Status: implementation gameplan, not executed

Blueprint:

- `docs/design/tensorization/01_006_state_collapser_encoding_compatibility_blueprint.md`

Source documents:

- `docs/design/tensorization/01_005_hgraphml_state_collapser_tensorization_followup_instructions.md`
- `state_collapser/docs/design/tensorization/01_005_hgraphml_tensorization_followup_bridge.md`
- `docs/api_notes/01_001_first_surfaces.md`
- `docs/usage/01_001_first_hack.md`
- `docs/design/01_004_hgraphml_first_import_implementation_log.md`
- `docs/prime_directive/prime_directive.md`
- `docs/prime_directive/git_practices.md`
- `docs/prime_directive/common_failure_mode_002_implementation_without_owner_approval.md`
- `docs/prime_directive/common_failure_mode_003_gameplan_rewrite_during_implementation.md`

## Execution Authority

The Project Owner approved converting the blueprint into a gameplan.

This document is not itself approval to implement source changes. Before
implementation begins, the Project Owner must explicitly request execution of
this gameplan.

Once execution is requested, this gameplan is the implementation authority unless
the Project Owner explicitly changes it.

## Goal

Implement the first HGraphML compatibility pass for the `state_collapser`
tensorization boundary.

The implementation must prove:

```text
TensorGraph
    -> build_tower_bundle(...)
        -> TowerBundle.partition_tower
            -> EncodingRegistry.from_tower(...)
                -> graph-message-passing-compatible stable ids
```

without routing HGraphML through RL-specific training or Torch tensor surfaces.

## Fixed Decisions

These decisions are fixed for this implementation unless the Project Owner
changes them before execution.

1. Use a dedicated implementation branch:

   ```text
   codex/state-collapser-encoding-compat
   ```

2. Update the `state-collapser` dependency from:

   ```text
   v0.6.0
   ```

   to:

   ```text
   v0.7.0
   ```

3. Add the explicit HGraphML helper:

   ```python
   def build_encoding_registry(bundle: TowerBundle) -> EncodingRegistry:
       return EncodingRegistry.from_tower(bundle.partition_tower)
   ```

4. Export that helper from:

   ```text
   src/hgraphml/adapters/__init__.py
   ```

5. Do not add an `encoding_registry` field to `TowerBundle`.

6. Do not change `collapse_messages(...)`.

7. Do not change lift semantics.

8. Do not use:

   ```python
   ActionSelectionInput
   TrainingTransition
   LinearizedActionSelectionInput
   LinearizedTrainingTransition
   TorchDecisionBatch
   TorchTransitionBatch
   ActionDecision
   ```

   in HGraphML.

## Explicit Non-Goals

Do not implement:

- tensorized graph-message-passing kernels;
- replacement of `TensorGraph` ids with `state_collapser` ids throughout the
  package;
- an RL learner path;
- Torch batch usage;
- speed-up benchmarks;
- changes to `TowerBundle` fields;
- changes to `collapse_messages(...)`;
- changes to deterministic or learned lift behavior;
- local reimplementation of `EncodingRegistry`.

## Global Stop Conditions

Implementation must stop and report if:

- HGraphML cannot install or sync against `state_collapser v0.7.0`;
- `from state_collapser.training import EncodingRegistry` fails after sync;
- adding the helper requires changing `TowerBundle` shape;
- compatibility tests require constructing fake RL records;
- any test failure suggests the upstream registry is not compatible with the
  existing HGraphML adapter assumptions;
- dependency sync modifies unrelated package metadata in an unexpected way;
- network or sandbox restrictions prevent dependency update.

If a dependency command fails because of network or sandbox restrictions,
rerun the same command with escalation according to the active tool policy and
record the outcome.

## Phase 0: Branch And Baseline Reality

### Stage 0.1: Create Implementation Branch

#### Action 0.1.1

Create or switch to:

```bash
git checkout -b codex/state-collapser-encoding-compat
```

If the branch already exists, switch to it without overwriting work.

### Stage 0.2: Record Starting State

#### Action 0.2.1

Record:

```bash
git status --short
git log --oneline --decorate --max-count=5
```

Expected current reality before implementation:

```text
pyproject.toml pins state-collapser v0.6.0
uv.lock pins state-collapser v0.6.0
docs/design/tensorization/01_006_state_collapser_encoding_compatibility_blueprint.md exists
```

#### Action 0.2.2

Confirm the new upstream tag exists conceptually in the design record:

```text
state_collapser v0.7.0 exposes state_collapser.training.EncodingRegistry
```

No source edit may proceed if the implementer has reason to believe the public
tag no longer exposes that surface.

## Phase 1: Dependency Update Gate

### Stage 1.1: Update Dependency Pin

#### Action 1.1.1

Update `pyproject.toml` so the direct Git dependency points to:

```text
state_collapser.git@v0.7.0
```

Target file:

```text
pyproject.toml
```

### Stage 1.2: Regenerate Lock Metadata

#### Action 1.2.1

Regenerate `uv.lock` using the repository's normal `uv` workflow.

Expected target file:

```text
uv.lock
```

The resulting lock metadata should point at:

```text
v0.7.0
```

not:

```text
v0.6.0
```

### Stage 1.3: Verify Import

#### Action 1.3.1

Run:

```bash
uv run --extra dev python -c "from state_collapser.training import EncodingRegistry; print(EncodingRegistry.__name__)"
```

Expected output:

```text
EncodingRegistry
```

#### Action 1.3.2

If the import fails, stop. Do not proceed to code changes.

## Phase 2: Adapter Helper

### Stage 2.1: Add Helper

#### Action 2.1.1

Edit:

```text
src/hgraphml/adapters/state_collapser.py
```

Add:

```python
def build_encoding_registry(bundle: TowerBundle) -> EncodingRegistry:
    """Build the shared state_collapser encoding registry for a tower bundle."""

    return EncodingRegistry.from_tower(bundle.partition_tower)
```

Import:

```python
from state_collapser.training import EncodingRegistry
```

Do not modify `TowerBundle` fields.

Do not modify `build_tower_bundle(...)` behavior.

### Stage 2.2: Export Helper

#### Action 2.2.1

Edit:

```text
src/hgraphml/adapters/__init__.py
```

Export:

```python
build_encoding_registry
```

Update `__all__` accordingly.

## Phase 3: Compatibility Tests

### Stage 3.1: Add Test File

#### Action 3.1.1

Create:

```text
tests/adapters/test_state_collapser_encoding_registry_compatibility.py
```

### Stage 3.2: Test Import Shape

#### Action 3.2.1

Add a test that imports only the shared encoding surface:

```python
from state_collapser.training import EncodingRegistry
```

The test must not import:

```python
ActionSelectionInput
TrainingTransition
LinearizedActionSelectionInput
LinearizedTrainingTransition
TorchDecisionBatch
TorchTransitionBatch
ActionDecision
```

### Stage 3.3: Test Registry Construction From TowerBundle

#### Action 3.3.1

Use:

```python
from state_collapser.tower.partition import LabelBlockSchema

from hgraphml.adapters import build_encoding_registry, build_tower_bundle
from hgraphml.examples.toy_graphs import make_repeated_motif_graph
```

Construct:

```python
graph = make_repeated_motif_graph()
schema = LabelBlockSchema.from_labels(("motif",))
bundle = build_tower_bundle(graph, contraction_schema=schema)
registry = build_encoding_registry(bundle)
```

Assert:

- `registry.registry_id` is a nonempty string.
- `registry.to_dict()` is JSON-safe.
- every original graph node maps to an encoded upstream `State`.
- every original graph edge maps to an encoded upstream `BaseEdge`.
- every tier has at least one encodable state cell.
- at least one action cell is encodable when action cells exist.

Use the existing `bundle.partition_tower.registry` to recover upstream
`State` and `BaseEdge` objects. Do not add new adapter state just to make the
test easier.

### Stage 3.4: Test Fiber Compatibility

#### Action 3.4.1

In the same test file, verify:

- every `NodeFiber.fine_nodes` value is within `range(graph.node_count)`;
- every `EdgeFiber.fine_edges` value is within `range(graph.edge_count)`;
- each state cell represented by a tier can be encoded;
- each base edge represented by a fine edge can be encoded;
- all fine nodes remain covered at every tier;
- all fine edges remain covered at every tier.

### Stage 3.5: Test Helper Identity

#### Action 3.5.1

Add a focused helper test:

```python
assert build_encoding_registry(bundle).registry_id == EncodingRegistry.from_tower(
    bundle.partition_tower
).registry_id
```

This proves the HGraphML helper is a transparent bridge, not a new registry
implementation.

## Phase 4: Documentation

### Stage 4.1: API Notes

#### Action 4.1.1

Edit:

```text
docs/api_notes/01_001_first_surfaces.md
```

Add a section:

```text
State-Collapser Encoding Compatibility
```

It must state:

- HGraphML uses `state_collapser` for quotient-tower construction.
- HGraphML exposes `build_encoding_registry(...)` as a helper around
  `EncodingRegistry.from_tower(...)`.
- This is shared tower encoding, not RL tensorization.
- HGraphML does not require `ActionSelectionInput` or `TrainingTransition`.
- HGraphML does not route graph message passing through `TorchDecisionBatch`.

### Stage 4.2: README

#### Action 4.2.1

Edit:

```text
README.md
```

Update the dependency discussion from `v0.6.0` to `v0.7.0`.

Add a short note near the `state_collapser` dependency discussion:

```text
HGraphML also checks compatibility with state_collapser's shared
EncodingRegistry, which provides stable numeric ids for tower states, edges,
cells, and tiers. HGraphML uses this as graph-message-passing infrastructure,
not as an RL ActionSelectionInput tensorization path.
```

Do not add speed-up language.

Do not claim full HGraphML tensorization.

## Phase 5: Validation

### Stage 5.1: Focused Adapter Tests

#### Action 5.1.1

Run:

```bash
uv run --extra dev pytest tests/adapters/test_state_collapser_adapter.py
```

Expected result:

```text
passes
```

#### Action 5.1.2

Run:

```bash
uv run --extra dev pytest tests/adapters/test_state_collapser_encoding_registry_compatibility.py
```

Expected result:

```text
passes
```

### Stage 5.2: Full Validation

#### Action 5.2.1

Run:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run --extra dev mypy
```

Expected result:

```text
all pass
```

### Stage 5.3: Dependency Verification

#### Action 5.3.1

Verify no committed dependency metadata still references:

```text
v0.6.0
```

for `state-collapser`.

Representative check:

```bash
rg -n "state_collapser.git@v0.6.0|rev=v0.6.0|state-collapser.*v0.6.0" pyproject.toml uv.lock README.md
```

Expected result:

```text
no matches
```

## Phase 6: Implementation Log

### Stage 6.1: Add Log

#### Action 6.1.1

Create:

```text
docs/design/tensorization/01_008_state_collapser_encoding_compatibility_implementation_log.md
```

Record:

- branch name;
- starting git status and commit;
- dependency update details;
- whether `EncodingRegistry` import passed;
- helper surface added;
- compatibility test summary;
- documentation updates;
- validation command outcomes;
- any skipped or failed command;
- final git status.

## Phase 7: Final Review

### Stage 7.1: Scope Audit

#### Action 7.1.1

Confirm no HGraphML code imports or uses:

```text
ActionSelectionInput
TrainingTransition
LinearizedActionSelectionInput
LinearizedTrainingTransition
TorchDecisionBatch
TorchTransitionBatch
ActionDecision
```

### Stage 7.2: Public Claim Audit

#### Action 7.2.1

Confirm README and docs do not claim:

- graph-ML speed-up;
- full graph-message tensorization;
- RL learner support in HGraphML.

### Stage 7.3: Completion Summary

#### Action 7.3.1

Prepare a concise summary for the Project Owner:

- dependency now targets `state_collapser v0.7.0`;
- helper added;
- compatibility tests added;
- docs updated;
- validation results;
- remaining deferred work.

## Completion Criteria

The implementation is complete when:

- HGraphML installs a `state_collapser` dependency exposing
  `EncodingRegistry`;
- `build_encoding_registry(...)` exists and is exported;
- compatibility tests pass;
- existing adapter tests pass;
- full test suite passes;
- Ruff passes;
- mypy passes;
- docs explain the encoding boundary accurately;
- no RL tensorization surfaces are used in HGraphML;
- implementation log is written.

