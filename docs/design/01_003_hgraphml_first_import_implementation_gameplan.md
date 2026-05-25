# HGraphML First Import Implementation Gameplan

## Status

Implementation gameplan for the first real `HGraphML` package milestone.

This document follows:

- [01_001_lifting_strategies_for_hierarchical_graph_message_passing.md](./01_001_lifting_strategies_for_hierarchical_graph_message_passing.md)
- [01_002_hgraphml_first_import_blueprint.md](./01_002_hgraphml_first_import_blueprint.md)

This is a Phase.Stage.Action gameplan. During execution, the gameplan should be
treated as the implementation authority unless the Project Owner explicitly
authorizes a change.

## Prime Directive Alignment

Per repository git practice:

```text
approved blueprint + approved gameplan + execution request
    -> create/switch to dedicated implementation branch before source edits
```

Implementation should not begin on `main`.

Expected branch form:

```text
codex/hgraphml-first-import
```

## Goal

Implement the first `HGraphML` package milestone:

```text
state_collapser-style quotient/tower scaffolding
    + PyTorch graph message passing
    + deterministic lifts
    + learned lift
    + trainability proof
```

The core proof is:

```text
collapse_messages(...) builds or receives a tower, runs coarse message passing,
lifts messages back to the fine graph, computes a loss, backpropagates through
the learned lift/message path, and updates parameters.
```

No speed-up claim is required in this milestone.

## Non-Negotiable Stop Condition

The `state_collapser` adapter investigation is an early blocking phase.

If direct use of `state_collapser` partition/tower surfaces works cleanly, the
implementation proceeds with that adapter.

If direct use does not work cleanly, implementation must stop before creating a
local fallback tower/fiber adapter. The correct action is:

```text
write a concrete mismatch note,
identify the smallest missing state_collapser surface,
and request Project Owner approval before implementing any fallback.
```

The implementation must not silently reimplement `state_collapser` under an
`HGraphML` name.

## Phase 0: Branch And Baseline Reality

### Stage 0.1: Branch Discipline

#### Action 0.1.1

Create a dedicated implementation branch:

```bash
git checkout -b codex/hgraphml-first-import
```

#### Action 0.1.2

Record the starting git state:

```bash
git status --short
git log --oneline --decorate --max-count=5
```

#### Action 0.1.3

Confirm that existing uncommitted Project Owner changes are not overwritten.

Expected current state before implementation:

```text
README.md may already be modified.
docs/design/ may contain the design and gameplan documents.
docs/engineer_continuity/ may already exist locally.
```

### Stage 0.2: Repository Shape Check

#### Action 0.2.1

Inspect the current repo:

```bash
find . -maxdepth 3 -type f | sort
```

#### Action 0.2.2

Confirm whether `pyproject.toml`, `src/`, and `tests/` exist.

Expected result:

```text
They likely do not exist before this implementation.
```

#### Action 0.2.3

Read repo-local prime directive documents before source edits:

```text
docs/prime_directive/prime_directive.md
docs/prime_directive/git_practices.md
docs/prime_directive/common_failure_mode_002_implementation_without_owner_approval.md
docs/prime_directive/common_failure_mode_003_gameplan_rewrite_during_implementation.md
```

## Phase 1: Package Skeleton And Tooling

### Stage 1.1: Project Metadata

#### Action 1.1.1

Create `pyproject.toml` using a `src/` package layout.

Required metadata:

- package name: `hgraphml`
- Python requirement: `>=3.11`
- license: MIT
- typed package marker support
- build backend: hatchling or equivalent simple backend

#### Action 1.1.2

Add dependencies:

- `torch`
- `state-collapser` if available as installable dependency

If `state-collapser` cannot be expressed as a normal dependency because the
local package is not published in the needed way, document the local editable
install expectation in README/developer docs instead of inventing a misleading
dependency.

#### Action 1.1.3

Add development dependencies:

- `pytest`
- `ruff`
- `mypy`

#### Action 1.1.4

Configure `pytest`, `ruff`, and `mypy` in `pyproject.toml`.

Expected policy:

- tests live under `tests`
- package source lives under `src`
- mypy checks `hgraphml`
- ruff targets Python 3.11

### Stage 1.2: Initial Package Files

#### Action 1.2.1

Create:

```text
src/hgraphml/__init__.py
src/hgraphml/_version.py
src/hgraphml/py.typed
```

#### Action 1.2.2

Export only:

```python
__version__
collapse_messages
```

from the top-level package.

If `collapse_messages` is not implemented until a later phase, export it only
after implementation or provide a temporary import-safe placeholder that raises
`NotImplementedError` and is removed before milestone completion.

Preferred path:

```text
Do not export placeholder APIs unless tests require early import shape.
```

### Stage 1.3: Test Skeleton

#### Action 1.3.1

Create initial test package structure:

```text
tests/
  test_package.py
```

#### Action 1.3.2

Add package smoke tests:

- `import hgraphml`
- `hgraphml.__version__` exists

#### Action 1.3.3

Run:

```bash
uv run pytest tests/test_package.py
uv run ruff check .
uv run mypy src
```

If `uv` is not configured for this repo yet, use the repo's available Python
environment and document the chosen command in the implementation log.

## Phase 2: Tensor Graph And Fiber Data

### Stage 2.1: TensorGraph

#### Action 2.1.1

Create:

```text
src/hgraphml/graph/__init__.py
src/hgraphml/graph/data.py
tests/graph/test_data.py
```

#### Action 2.1.2

Implement `TensorGraph`:

```python
@dataclass(frozen=True, slots=True)
class TensorGraph:
    node_count: int
    edge_index: torch.Tensor
    edge_labels: tuple[str, ...] | None = None
```

#### Action 2.1.3

Validate:

- `node_count >= 0`
- `edge_index.ndim == 2`
- `edge_index.shape[0] == 2`
- edge indices are integer-like tensors
- edge indices are in range when edge count is nonzero
- edge labels length matches edge count when provided

#### Action 2.1.4

Add convenience properties:

```python
edge_count
sources
targets
```

#### Action 2.1.5

Test valid and invalid graph construction.

### Stage 2.2: MessageBatch

#### Action 2.2.1

Create:

```text
src/hgraphml/messages/__init__.py
src/hgraphml/messages/containers.py
tests/messages/test_containers.py
```

#### Action 2.2.2

Implement `MessageBatch`:

```python
@dataclass(frozen=True, slots=True)
class MessageBatch:
    graph: TensorGraph
    values: torch.Tensor
```

#### Action 2.2.3

Validate:

- message values rank is `2`
- first dimension equals `graph.edge_count`

#### Action 2.2.4

Test message shape validation.

### Stage 2.3: Fiber Maps

#### Action 2.3.1

Create:

```text
src/hgraphml/graph/fibers.py
tests/graph/test_fibers.py
```

#### Action 2.3.2

Implement:

```python
@dataclass(frozen=True, slots=True)
class NodeFiber:
    tier: int
    coarse_node: int
    fine_nodes: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class EdgeFiber:
    tier: int
    coarse_edge: int
    fine_edges: tuple[int, ...]
```

#### Action 2.3.3

Add type aliases:

```python
NodeFiberMap = Mapping[int, NodeFiber]
EdgeFiberMap = Mapping[int, EdgeFiber]
```

#### Action 2.3.4

Validate:

- `tier >= 0`
- coarse index is nonnegative
- fine index tuples are nonempty
- fine index tuples contain no duplicates

#### Action 2.3.5

Test validation and basic storage behavior.

## Phase 3: Early `state_collapser` Adapter Gate

### Stage 3.1: Inspect `state_collapser` Surfaces

#### Action 3.1.1

Inspect installed/importable `state_collapser` package.

Commands may include:

```bash
uv run python - <<'PY'
import state_collapser
print(state_collapser.__version__)
PY
```

#### Action 3.1.2

Inspect relevant partition/tower modules.

Target modules:

```text
state_collapser.tower.partition
state_collapser.tower.partition.tower
state_collapser.tower.partition.schema
state_collapser.tower.partition.base_registry
state_collapser.tower.partition.ids
state_collapser.tower.partition.readout
```

#### Action 3.1.3

Determine whether current `state_collapser` exposes a clean way to:

- register all nodes/states up front
- register all directed edges/actions up front
- apply a schema over the full graph
- retrieve node/state fibers by tier
- retrieve edge/action fibers by tier

### Stage 3.2: Adapter Decision Gate

#### Action 3.2.1

Write a short adapter feasibility note in the implementation log.

The note must answer:

- Which `state_collapser` classes/functions will be used?
- How are HGraphML node IDs mapped to `state_collapser` state IDs?
- How are HGraphML edge IDs mapped to `state_collapser` edge/action IDs?
- How are node fibers recovered?
- How are edge fibers recovered?
- Is any fallback needed?

#### Action 3.2.2

If direct import works, proceed to Stage 3.3.

#### Action 3.2.3

If direct import does not work cleanly, stop implementation and ask PO for
approval before creating any local fallback tower/fiber adapter.

Stop report must include:

- concrete missing API or mismatch
- smallest fallback needed
- whether the fallback would be temporary
- what later `state_collapser` change would remove the fallback

### Stage 3.3: Implement Direct Adapter

Only execute this stage if Stage 3.2 approves direct import.

#### Action 3.3.1

Create:

```text
src/hgraphml/adapters/__init__.py
src/hgraphml/adapters/state_collapser.py
src/hgraphml/graph/tower_adapter.py
tests/adapters/test_state_collapser_adapter.py
```

#### Action 3.3.2

Implement `TowerBundle`:

```python
@dataclass(frozen=True, slots=True)
class TowerBundle:
    graph: TensorGraph
    partition_tower: object
    node_fibers_by_tier: tuple[Mapping[int, NodeFiber], ...]
    edge_fibers_by_tier: tuple[Mapping[int, EdgeFiber], ...]
```

Use a more specific `state_collapser` type for `partition_tower` if available.

#### Action 3.3.3

Implement:

```python
def build_tower_bundle(
    graph: TensorGraph,
    *,
    contraction_schema: object | None = None,
) -> TowerBundle:
    ...
```

#### Action 3.3.4

Ensure the adapter treats the graph as fully explored:

- every node registered
- every edge registered
- no environment reset
- no environment step
- no incremental discovery loop

#### Action 3.3.5

Test:

- all nodes appear in tier-0 fibers
- all edges appear in tier-0 fibers
- every fine node is covered by every tier's node fiber maps
- every fine edge is covered by every tier's edge fiber maps
- no exploration loop is required

## Phase 4: Lift Operators

### Stage 4.1: Lift Interface

#### Action 4.1.1

Create:

```text
src/hgraphml/lifts/__init__.py
src/hgraphml/lifts/base.py
tests/lifts/test_base.py
```

#### Action 4.1.2

Define a protocol or abstract base for message lifts:

```python
class MessageLift(Protocol):
    def lift(
        self,
        *,
        coarse_messages: torch.Tensor,
        edge_fibers: Mapping[int, EdgeFiber],
        fine_edge_count: int,
        fine_edge_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
```

#### Action 4.1.3

Ensure interface is compatible with both stateless objects and `torch.nn.Module`
learned lifts.

### Stage 4.2: Uniform Pullback Lift

#### Action 4.2.1

Create:

```text
src/hgraphml/lifts/uniform.py
tests/lifts/test_uniform.py
```

#### Action 4.2.2

Implement `UniformPullbackLift`.

#### Action 4.2.3

Test:

- copies coarse message to every fine edge in fiber
- handles singleton fibers
- rejects missing coarse-message row for fiber key
- output shape is `(fine_edge_count, message_dim)`
- gradient from summed fine output reaches coarse messages

### Stage 4.3: Fiber-Normalized Lift

#### Action 4.3.1

Create:

```text
src/hgraphml/lifts/fiber_normalized.py
tests/lifts/test_fiber_normalized.py
```

#### Action 4.3.2

Implement `FiberNormalizedLift`.

First behavior:

```text
fine_message[e] = coarse_message[c] / len(fiber(c))
```

#### Action 4.3.3

Test:

- divides message across fiber
- differs from uniform pullback for fiber size greater than one
- preserves total aggregate mass under sum over fiber
- preserves gradient
- handles singleton fibers

### Stage 4.4: Learned Fiber Lift

#### Action 4.4.1

Create:

```text
src/hgraphml/lifts/learned.py
tests/lifts/test_learned.py
```

#### Action 4.4.2

Implement `LearnedFiberLift` as `torch.nn.Module`.

Suggested constructor:

```python
class LearnedFiberLift(nn.Module):
    def __init__(
        self,
        *,
        message_dim: int,
        context_dim: int,
        hidden_dim: int = 32,
    ) -> None:
        ...
```

#### Action 4.4.3

Implement score network:

```text
score[e] = score_mlp([coarse_message[c], fine_edge_context[e]])
```

#### Action 4.4.4

Implement within-fiber softmax normalization.

#### Action 4.4.5

Implement decoder:

```text
fine_message[e] = decoder([coarse_message[c], fine_edge_context[e], weight[e]])
```

#### Action 4.4.6

Expose optional diagnostics for learned weights if useful:

```python
last_weights_by_fiber
```

This must not detach tensors needed for gradients.

#### Action 4.4.7

Test:

- output shape
- weights normalize within each fiber
- parameters receive gradients
- optimizer step changes parameters
- no accidental detach

## Phase 5: Message Passing And Readout

### Stage 5.1: Message Model

#### Action 5.1.1

Create:

```text
src/hgraphml/messages/passing.py
tests/messages/test_passing.py
```

#### Action 5.1.2

Implement `EdgeMessageMLP`.

Inputs:

- source node features
- target node features
- optional edge features

Output:

- edge-aligned messages

#### Action 5.1.3

Implement helper:

```python
def run_edge_message_model(
    graph: TensorGraph,
    node_features: torch.Tensor,
    message_model: nn.Module,
    edge_features: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
```

#### Action 5.1.4

Test source/target gather and output shape.

### Stage 5.2: Coarse Feature Pooling

#### Action 5.2.1

Implement:

```python
def mean_pool_node_features(
    node_features: torch.Tensor,
    node_fibers: Mapping[int, NodeFiber],
) -> torch.Tensor:
    ...
```

#### Action 5.2.2

Test mean pooling over node fibers.

#### Action 5.2.3

Ensure output coarse node order is stable and aligned to coarse node IDs.

If coarse node IDs are non-contiguous, either:

- compact them explicitly and record mapping
- or require adapter to emit contiguous coarse node IDs

Prefer contiguous IDs for milestone one.

### Stage 5.3: Readout

#### Action 5.3.1

Create:

```text
src/hgraphml/messages/readout.py
tests/messages/test_readout.py
```

#### Action 5.3.2

Implement:

```python
def incoming_sum_readout(
    graph: TensorGraph,
    fine_messages: torch.Tensor,
) -> torch.Tensor:
    ...
```

#### Action 5.3.3

Test:

- aggregates messages by target node
- returns shape `(node_count, message_dim)`
- handles nodes with no incoming messages by returning zeros

## Phase 6: `collapse_messages(...)`

### Stage 6.1: Result And Diagnostics Types

#### Action 6.1.1

Create:

```text
src/hgraphml/collapse.py
tests/test_collapse.py
```

#### Action 6.1.2

Implement `HGraphMLResult`.

Fields:

- `graph`
- `tower_bundle`
- `coarse_tier`
- `coarse_messages`
- `fine_messages`
- `node_readout`
- optional diagnostics

#### Action 6.1.3

Implement minimal diagnostics type if needed.

### Stage 6.2: Coarse Tier Selection

#### Action 6.2.1

Implement coarse-tier selection:

- if `coarse_tier is None`, choose largest available tier
- if `coarse_tier < 0`, index from end
- validate tier bounds

#### Action 6.2.2

Test tier selection behavior.

### Stage 6.3: Coarse Graph Construction

#### Action 6.3.1

Implement construction of a coarse `TensorGraph` from the selected tier's node
and edge fiber maps.

#### Action 6.3.2

Ensure coarse graph edge count equals number of coarse edge fibers.

#### Action 6.3.3

Ensure coarse node count matches contiguous coarse node IDs.

#### Action 6.3.4

Test coarse graph construction on simple hand-written fibers.

### Stage 6.4: Fine Edge Context

#### Action 6.4.1

Implement fine edge context construction:

```text
context[e] = concat(source_node_features[e], target_node_features[e], optional edge_features[e])
```

#### Action 6.4.2

Test shape with and without edge features.

### Stage 6.5: Orchestration

#### Action 6.5.1

Implement `collapse_messages(...)`.

Flow:

1. validate graph and features
2. build `tower_bundle` if not supplied
3. select tier
4. build coarse graph
5. pool fine node features to coarse nodes
6. run message model on coarse graph
7. build fine edge context
8. lift coarse messages to fine edge messages
9. compute incoming-sum node readout
10. return `HGraphMLResult`

#### Action 6.5.2

Test end-to-end forward pass.

#### Action 6.5.3

Test that gradients flow from `result.node_readout.sum()` to message model and
learned lift parameters.

#### Action 6.5.4

Test that a prebuilt `TowerBundle` can be reused.

## Phase 7: Training Helpers And Diagnostics

### Stage 7.1: Gradient Diagnostics

#### Action 7.1.1

Create:

```text
src/hgraphml/diagnostics/__init__.py
src/hgraphml/diagnostics/gradients.py
tests/diagnostics/test_gradients.py
```

#### Action 7.1.2

Implement `GradientDiagnostics`.

Fields:

- `parameter_count`
- `parameters_with_grad`
- `nonzero_gradient_count`
- `total_gradient_norm`

#### Action 7.1.3

Implement:

```python
def collect_gradient_diagnostics(parameters: Iterable[nn.Parameter]) -> GradientDiagnostics:
    ...
```

#### Action 7.1.4

Test zero-grad and nonzero-grad cases.

### Stage 7.2: Viability Diagnostics

#### Action 7.2.1

Create:

```text
src/hgraphml/diagnostics/viability.py
tests/diagnostics/test_viability.py
```

#### Action 7.2.2

Implement `ViabilityDiagnostics`.

Fields:

- `loss_is_finite`
- `parameters_changed`
- `fine_message_shape`
- `coarse_message_shape`

#### Action 7.2.3

Implement parameter snapshot/changed helpers.

Must avoid interfering with gradients.

### Stage 7.3: Train Step

#### Action 7.3.1

Create:

```text
src/hgraphml/training/__init__.py
src/hgraphml/training/train_step.py
tests/training/test_train_step.py
```

#### Action 7.3.2

Implement `TrainStepResult`.

Fields:

- `loss`
- `hgraphml_result`
- `gradient_diagnostics`
- `viability_diagnostics`

#### Action 7.3.3

Implement `train_step(...)`.

Flow:

1. snapshot trainable parameters
2. `optimizer.zero_grad()`
3. call `collapse_messages(...)`
4. compute loss
5. call `loss.backward()`
6. collect gradient diagnostics
7. call `optimizer.step()`
8. collect viability diagnostics
9. return result

#### Action 7.3.4

Test:

- finite loss
- nonzero gradients
- parameters changed
- output diagnostics populated

## Phase 8: Toy Example And Viability Benchmark

### Stage 8.1: Toy Graphs

#### Action 8.1.1

Create:

```text
src/hgraphml/examples/__init__.py
src/hgraphml/examples/toy_graphs.py
tests/examples/test_toy_graphs.py
```

#### Action 8.1.2

Implement:

```python
def make_repeated_motif_graph() -> TensorGraph:
    ...
```

Use a small graph such as:

```text
motif 0: 0 -> 1 -> 2, 0 -> 2
motif 1: 3 -> 4 -> 5, 3 -> 5
bridge: 2 -> 3
```

#### Action 8.1.3

Provide deterministic node feature generation:

```python
def make_toy_node_features(seed: int = 0, feature_dim: int = 8) -> torch.Tensor:
    ...
```

#### Action 8.1.4

Test graph and feature shapes.

### Stage 8.2: Teacher Objective

#### Action 8.2.1

Create:

```text
src/hgraphml/training/objectives.py
tests/training/test_objectives.py
```

#### Action 8.2.2

Implement a fixed teacher message/readout target:

```python
def make_teacher_node_targets(
    graph: TensorGraph,
    node_features: torch.Tensor,
    *,
    output_dim: int,
    seed: int = 0,
) -> torch.Tensor:
    ...
```

#### Action 8.2.3

Ensure teacher parameters are fixed and not part of the trained model.

#### Action 8.2.4

Test target shape and deterministic behavior.

### Stage 8.3: Learned Lift Demo

#### Action 8.3.1

Create:

```text
src/hgraphml/examples/learned_lift_demo.py
tests/examples/test_learned_lift_demo.py
```

#### Action 8.3.2

Implement:

```python
def run_learned_lift_demo(
    *,
    steps: int = 10,
    seed: int = 0,
) -> LearnedLiftDemoResult:
    ...
```

#### Action 8.3.3

The demo should:

1. build toy graph
2. build or reuse tower bundle
3. generate node features
4. generate teacher target
5. initialize message model and learned lift
6. run several `train_step(...)` calls
7. return initial loss, final loss, and diagnostics

#### Action 8.3.4

Test:

- demo runs
- loss values are finite
- learned lift gets gradients
- parameters change
- final loss is not `NaN`

Do not require monotonic loss decrease on every run unless it is stable. Prefer
requiring objective movement or final loss below a loose threshold discovered
from deterministic seeds.

## Phase 9: Documentation

### Stage 9.1: README Update

#### Action 9.1.1

Update `README.md` without overwriting existing Project Owner edits.

README should explain:

- HGraphML applies `state_collapser` quotient towers to graph message passing
- first milestone is trainable hierarchical message passing
- first proof is autograd viability
- no speed-up claim yet

#### Action 9.1.2

Add a minimal usage snippet:

```python
from hgraphml.examples.learned_lift_demo import run_learned_lift_demo

result = run_learned_lift_demo(steps=10, seed=0)
print(result.initial_loss, result.final_loss)
```

### Stage 9.2: Usage Doc

#### Action 9.2.1

Create:

```text
docs/usage/01_001_first_hack.md
```

#### Action 9.2.2

Document:

- build toy graph
- make node features
- initialize message model
- initialize learned lift
- call `collapse_messages`
- call `train_step`
- interpret diagnostics

### Stage 9.3: API Notes

#### Action 9.3.1

Create:

```text
docs/api_notes/01_001_first_surfaces.md
```

#### Action 9.3.2

Document:

- `TensorGraph`
- `MessageBatch`
- `NodeFiber`
- `EdgeFiber`
- `TowerBundle`
- `UniformPullbackLift`
- `FiberNormalizedLift`
- `LearnedFiberLift`
- `collapse_messages`
- `train_step`

## Phase 10: Validation And Implementation Log

### Stage 10.1: Focused Tests

#### Action 10.1.1

Run focused tests by package area:

```bash
uv run pytest tests/graph
uv run pytest tests/messages
uv run pytest tests/lifts
uv run pytest tests/diagnostics
uv run pytest tests/training
uv run pytest tests/examples
```

If a directory does not exist yet at the time of the first run, run the
available focused tests and update the implementation log.

### Stage 10.2: Full Local Validation

#### Action 10.2.1

Run:

```bash
uv run pytest tests
uv run ruff check .
uv run mypy src
```

#### Action 10.2.2

Run the learned-lift demo manually:

```bash
uv run python -m hgraphml.examples.learned_lift_demo
```

If the demo is not implemented as a module CLI, run the equivalent Python import
snippet and record it.

### Stage 10.3: Implementation Log

#### Action 10.3.1

Create:

```text
docs/design/01_004_hgraphml_first_import_implementation_log.md
```

#### Action 10.3.2

Record:

- branch name
- package skeleton decisions
- `state_collapser` adapter decision and exact surfaces used
- any PO-approved deviations
- test results
- known limitations
- next recommended work

## Phase 11: Final Review

### Stage 11.1: Gameplan Completion Audit

#### Action 11.1.1

Review every Phase.Stage.Action item and classify it as:

- completed
- explicitly deferred with reason
- blocked with reason
- changed by PO authorization

#### Action 11.1.2

Ensure no unapproved fallback tower implementation was introduced.

#### Action 11.1.3

Ensure no speed-up claim was added to README or docs.

### Stage 11.2: Git Summary

#### Action 11.2.1

Run:

```bash
git status --short
git diff --stat
```

#### Action 11.2.2

Prepare a concise final implementation summary for the Project Owner.

The summary must include:

- what was implemented
- what tests passed
- what was not run
- whether direct `state_collapser` import was used
- any remaining risks

## Completion Criteria

The implementation is complete only when:

- package imports
- `collapse_messages` runs
- deterministic lifts work and are tested
- learned lift works and is tested
- train step produces finite loss
- gradients reach learned lift parameters
- optimizer changes parameters
- toy demo runs
- direct `state_collapser` adapter is implemented or PO-approved fallback is
  explicitly documented
- README and usage/API docs are updated
- full validation has been run or any inability to run it is clearly reported

## Explicit Non-Goals For This Gameplan

Do not implement:

- speed benchmark claims
- PyG adapter as a hard dependency
- pgmpy adapter
- DGL adapter
- mature experiment tracking
- checkpointing
- GPU/device abstraction beyond ordinary PyTorch tensors
- exact BP theorem/proof layer
- dynamic graph updates

## Final Note

This gameplan is deliberately viability-first.

The first honest win is:

```text
HGraphML makes the state_collapser quotient-tower idea executable inside a
trainable graph-message-passing loop.
```

Once that is real, speed-up experiments and richer graph ML adapters become
ordinary next-stage engineering.
