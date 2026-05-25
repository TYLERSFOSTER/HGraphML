# HGraphML First Import Blueprint

## Status

Blueprint for the first real implementation of `HGraphML`.

This document follows:

- [01_001_lifting_strategies_for_hierarchical_graph_message_passing.md](./01_001_lifting_strategies_for_hierarchical_graph_message_passing.md)

It converts the lifting-strategy discussion into an implementation architecture
for the first package milestone.

## Executive Summary

`HGraphML` should begin as a very small Python package that proves one central
claim:

```text
A state_collapser-style quotient tower can sit inside a trainable graph
message-passing computation without breaking PyTorch autograd.
```

The first implementation is not a speed benchmark. It is not a full graph ML
framework. It is not a complete belief-propagation library.

The first implementation is a proof that the package-defining call is real:

```python
from hgraphml import collapse_messages

result = collapse_messages(
    graph=graph,
    node_features=x,
    message_model=message_model,
    lift=learned_lift,
    contraction_schema=schema,
)
```

The first success condition is:

```text
The call builds a quotient tower, runs coarse message passing, lifts coarse
messages back to the fine graph, computes a loss, backpropagates through the
learned lift/message path, and updates parameters.
```

## PO Decisions Captured

The following Project Owner decisions control this blueprint.

### Approximate First

The first implementation should target approximate hierarchical message passing.
Exact quotient message passing is valuable, but it is a special sanity case,
not the first milestone.

The implementation should still include exact/symmetric tests where practical,
because those are useful for detecting broken lift semantics.

### Use Serious Existing Tools

The package should use industrial-grade existing tools wherever that helps.

For the first milestone, this means:

- use `torch` for tensors, modules, gradients, and training
- keep graph storage minimal and explicit
- avoid inventing a full graph ML framework
- avoid taking on `torch_geometric` until the first small PyTorch-native proof
  works

`torch_geometric` is a likely next adapter. It should not be required for the
first proof unless implementation reality makes it simpler than the internal
minimal graph surface.

### Import `state_collapser` Directly If Possible

The first implementation should attempt to use `state_collapser` directly rather
than conceptually reimplementing it.

The graph ML adaptation is:

```text
state_collapser RL case:
    graph is discovered incrementally through environment interaction

HGraphML first case:
    graph is fully known up front and treated as already discovered
```

So `HGraphML` should build a complete base registry from the known graph and
then ask `state_collapser` partition machinery to build a quotient tower from
that fully registered graph.

### First Benchmark Is Viability

The first benchmark is not speed-up.

It should answer:

```text
Does this train at all?
```

The benchmark can use tiny 8-dimensional feature vectors. It should show:

- forward pass succeeds
- loss is finite
- gradients reach learned lift parameters
- optimizer step changes parameters
- repeated steps move the toy objective

### Learned Lift Is Milestone One

Learned lift weights are not optional later work.

The first implementation should include a small differentiable learned lift. The
deterministic lifts should exist as baselines and sanity checks, but the honest
first proof is trainable hierarchical message passing.

## Scope

### In Scope

- Python package skeleton under `src/hgraphml`
- `pyproject.toml` with `torch` dependency
- direct dependency or editable-development expectation for `state_collapser`
- minimal tensor graph data object
- minimal edge-fiber representation
- adapter that registers a fully known graph into a `state_collapser`-style
  tower
- deterministic uniform lift
- deterministic fiber-normalized lift
- learned PyTorch lift
- simple PyTorch message model
- `collapse_messages(...)` orchestration call
- one trainable toy example
- tests proving gradient viability
- docs showing the first hack

### Out Of Scope

- speed-up claims
- large benchmarks
- mature graph ML framework design
- full belief propagation implementation
- exact BP theorem layer
- GPU performance hardening
- `torch_geometric` as a hard dependency
- `pgmpy` integration
- distributed graph computation
- dynamic graph updates
- public API stability beyond the tiny first package surface

## Proposed Source Layout

```text
src/
└── hgraphml/
    ├── __init__.py
    ├── collapse.py
    ├── py.typed
    │
    ├── graph/
    │   ├── __init__.py
    │   ├── data.py
    │   ├── fibers.py
    │   └── tower_adapter.py
    │
    ├── messages/
    │   ├── __init__.py
    │   ├── containers.py
    │   ├── passing.py
    │   └── readout.py
    │
    ├── lifts/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── uniform.py
    │   ├── fiber_normalized.py
    │   └── learned.py
    │
    ├── adapters/
    │   ├── __init__.py
    │   ├── state_collapser.py
    │   └── torch_geometric.py
    │
    ├── training/
    │   ├── __init__.py
    │   ├── losses.py
    │   ├── objectives.py
    │   └── train_step.py
    │
    ├── examples/
    │   ├── __init__.py
    │   ├── toy_graphs.py
    │   └── learned_lift_demo.py
    │
    └── diagnostics/
        ├── __init__.py
        ├── gradients.py
        └── viability.py
```

This layout is intentionally a little larger than the first code may require.
The point is to keep conceptual surfaces separated so the first package does not
immediately become a tangled demo script.

## Package Surface

The top-level package should export only a tiny number of names:

```python
from hgraphml import collapse_messages
from hgraphml import __version__
```

Optionally, it may also export:

```python
from hgraphml import HGraphMLResult
```

All other surfaces can live under explicit subpackages.

The main public-feeling call is:

```python
result = collapse_messages(
    graph=graph,
    node_features=node_features,
    edge_features=edge_features,
    message_model=message_model,
    lift=lift,
    contraction_schema=contraction_schema,
    coarse_tier=-1,
    refinement_steps=1,
)
```

The first version may support fewer arguments, but it should be designed toward
this shape.

## Data Model

### `TensorGraph`

The first internal graph object should be small and tensor-friendly.

Proposed fields:

```python
@dataclass(frozen=True, slots=True)
class TensorGraph:
    node_count: int
    edge_index: torch.Tensor
    edge_labels: tuple[str, ...] | None = None
```

Expected shape:

```text
edge_index.shape == (2, edge_count)
```

where:

```text
edge_index[0, e] = source node index
edge_index[1, e] = target node index
```

This mirrors common PyTorch Geometric convention without requiring PyG in the
first implementation.

Validation should ensure:

- `node_count >= 0`
- `edge_index` is rank `2`
- first dimension is `2`
- indices are in range
- edge labels, if present, match edge count

### `MessageBatch`

Messages should be represented as tensors aligned to graph edges.

Proposed fields:

```python
@dataclass(frozen=True, slots=True)
class MessageBatch:
    graph: TensorGraph
    values: torch.Tensor
```

Expected shape:

```text
values.shape == (edge_count, message_dim)
```

The graph stores combinatorics. The message batch stores differentiable data.

### `NodeFeatureBatch`

The first implementation can pass node features as bare tensors:

```text
node_features.shape == (node_count, feature_dim)
```

If repeated plumbing becomes awkward, introduce:

```python
@dataclass(frozen=True, slots=True)
class NodeFeatureBatch:
    graph: TensorGraph
    values: torch.Tensor
```

But avoid adding this wrapper unless it keeps the first implementation cleaner.

## Tower Adapter

### Execution Gate

The `state_collapser` adapter investigation must happen early in
implementation, before the package builds too much scaffolding around an
assumed tower interface.

This is a hard execution gate:

```text
If direct state_collapser import works cleanly:
    implement the adapter against the real state_collapser partition surfaces.

If direct state_collapser import does not work cleanly:
    stop implementation,
    write a concrete mismatch note,
    and ask the Project Owner whether to authorize a local fallback tower/fiber
    adapter for milestone one.
```

The implementation must not silently reimplement `state_collapser` tower
semantics under an HGraphML name.

### Goal

`HGraphML` should reuse `state_collapser` partition/tower machinery if possible.
The adapter's job is to translate a fully known tensor graph into the graph and
edge identity surfaces expected by `state_collapser`.

### Full-Graph Registration

The adapter should treat the input graph as already discovered:

```text
for every node in TensorGraph:
    register state/node identity

for every edge in TensorGraph:
    register directed edge/action identity
```

There should be no environment interaction loop and no incremental discovery in
milestone one.

### Identity Mapping

The adapter must preserve stable maps:

```text
HGraphML node index -> state_collapser state id
HGraphML edge index -> state_collapser edge id
state_collapser state cell -> HGraphML node fiber
state_collapser edge/action cell -> HGraphML edge fiber
```

This mapping is the core bridge.

### Output

The adapter should produce an object like:

```python
@dataclass(frozen=True, slots=True)
class TowerBundle:
    graph: TensorGraph
    partition_tower: object
    node_fibers_by_tier: tuple[NodeFiberMap, ...]
    edge_fibers_by_tier: tuple[EdgeFiberMap, ...]
```

The actual `partition_tower` type should be as specific as possible if
`state_collapser` exports a suitable name. If it does not, the adapter can keep
it internal and expose only the fiber maps.

### Fallback If Direct Import Is Awkward

If the current `state_collapser` tower API resists direct use because it is too
RL-shaped, the implementation should stop and document the mismatch.

The fallback should not silently reimplement the whole tower. The authorized
fallback for milestone one should be only a minimal local partition/fiber adapter
that preserves the conceptual contract while clearly marking that direct import
needs a later compatibility pass.

This is a likely implementation risk and should be surfaced early in the
gameplan.

## Fiber Maps

The lift operators need fast access to fine preimages of coarse nodes and edges.

### `NodeFiber`

```python
@dataclass(frozen=True, slots=True)
class NodeFiber:
    tier: int
    coarse_node: int
    fine_nodes: tuple[int, ...]
```

### `EdgeFiber`

```python
@dataclass(frozen=True, slots=True)
class EdgeFiber:
    tier: int
    coarse_edge: int
    fine_edges: tuple[int, ...]
```

### Fiber Collection

The first implementation can use simple tuple/dict maps:

```python
NodeFiberMap = Mapping[int, NodeFiber]
EdgeFiberMap = Mapping[int, EdgeFiber]
```

where the key is coarse node/edge index.

This does not have to be maximally optimized. It has to be correct, stable, and
easy to inspect.

## Message Passing

### First Message Model

The first message model should be a tiny PyTorch module:

```python
class EdgeMessageMLP(nn.Module):
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
```

It should output edge-aligned messages:

```text
messages.shape == (edge_count, message_dim)
```

### Coarse Message Pass

The first coarse message pass can be simple:

```text
coarse node features
    -> source/target gather on coarse edges
        -> message_model(...)
            -> coarse edge messages
```

The blueprint does not require a full GNN layer. The important thing is that
coarse messages are differentiable tensors that can be lifted back to fine edge
messages.

### Coarse Node Features

The first implementation needs a way to produce coarse node features from fine
node features.

Default:

```text
coarse_node_feature = mean(fine_node_features in node fiber)
```

This is not the final theory of graph quotient features. It is a minimal,
differentiable, understandable first aggregator.

Future options:

- sum
- max
- learned pooling
- attention pooling
- feature-type-specific aggregation

## Lift Operators

### Shared Interface

The lift interface should be PyTorch-friendly:

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

Expected output:

```text
fine_messages.shape == (fine_edge_count, message_dim)
```

This interface focuses on edge messages because edge-aligned data is the most
direct import of message passing. Node readouts can be built on top.

### `UniformPullbackLift`

Behavior:

```text
for each coarse edge c:
    for each fine edge e in fiber(c):
        fine_message[e] = coarse_message[c]
```

This lift should:

- be deterministic
- have no trainable parameters
- preserve gradient flow from fine loss back to coarse messages
- be tested as the simplest baseline

### `FiberNormalizedLift`

Behavior:

```text
for each coarse edge c:
    weights = normalize(weight_rule(fine edges in fiber(c)))
    for each fine edge e in fiber(c):
        fine_message[e] = weights[e] * coarse_message[c]
```

The first deterministic weight rule can be uniform:

```text
weights[e] = 1 / len(fiber(c))
```

This may look similar to uniform pullback, but it has different semantics:

- uniform pullback copies the whole message to every fine edge
- fiber-normalized lift divides aggregate message mass over the fiber

The test suite should explicitly distinguish those two behaviors.

### `LearnedFiberLift`

The learned lift is the first milestone's important lift.

Suggested behavior:

```text
for each coarse edge c:
    for each fine edge e in fiber(c):
        score[e] = MLP([coarse_message[c], fine_edge_context[e]])
    weights = softmax(score over fiber(c))
    fine_message[e] = Decoder(coarse_message[c], fine_edge_context[e], weight[e])
```

The simplest version can do:

```text
fine_message[e] = weight[e] * coarse_message[c]
```

but it may be better to include a small decoder:

```text
fine_message[e] = decoder([coarse_message[c], fine_edge_context[e], weight[e]])
```

The decoder form is more expressive and better proves that the lift can learn.

### Learned Lift Inputs

Milestone-one learned lift can use:

- coarse message
- source fine node feature
- target fine node feature
- optional edge feature

This is enough to make the lift trainable without requiring a full graph ML
backend.

### Learned Lift Diagnostics

Tests must prove:

- learned lift parameters receive gradients
- gradient norms are nonzero in ordinary runs
- one optimizer step changes parameters
- output shape is correct
- no tensors are accidentally detached

## `collapse_messages(...)`

### Responsibility

`collapse_messages(...)` orchestrates one hierarchical message-passing pass.

It should not own a long training loop.

It should:

1. validate inputs
2. build or receive a tower bundle
3. choose a coarse tier
4. pool fine node features to coarse node features
5. run the message model on coarse edges
6. lift coarse messages to fine edges
7. optionally run fine refinement
8. return structured outputs

### Proposed Signature

```python
def collapse_messages(
    *,
    graph: TensorGraph,
    node_features: torch.Tensor,
    message_model: nn.Module,
    lift: MessageLift,
    edge_features: torch.Tensor | None = None,
    contraction_schema: object | None = None,
    tower_bundle: TowerBundle | None = None,
    coarse_tier: int | None = None,
    refinement_steps: int = 0,
) -> HGraphMLResult:
    ...
```

### `tower_bundle`

Allowing a prebuilt `tower_bundle` is important because tower construction should
not necessarily run on every training step. The first demo can build internally,
but the package should support:

```python
tower_bundle = build_tower_bundle(graph, contraction_schema=schema)

for batch in batches:
    result = collapse_messages(
        graph=graph,
        node_features=batch.x,
        message_model=model,
        lift=lift,
        tower_bundle=tower_bundle,
    )
```

### Result Object

```python
@dataclass(frozen=True, slots=True)
class HGraphMLResult:
    graph: TensorGraph
    tower_bundle: TowerBundle
    coarse_tier: int
    coarse_messages: torch.Tensor
    fine_messages: torch.Tensor
    node_readout: torch.Tensor | None = None
    diagnostics: HGraphMLDiagnostics | None = None
```

The first implementation can omit `node_readout` if it complicates the pass.

## Readout

The first readout should be simple and edge-message based.

Options:

- aggregate incoming fine messages at each target node
- aggregate outgoing fine messages at each source node
- concatenate incoming/outgoing aggregates

Proposed first readout:

```text
node_readout[v] = sum_{u -> v} fine_message[u -> v]
```

This gives a node-aligned tensor suitable for a simple supervised toy loss.

Future readouts can add:

- mean aggregation
- max aggregation
- attention aggregation
- BP-style marginal extraction
- user-supplied readout functions

## Toy Example

### Goal

The first example should prove trainability, not speed.

### Graph

Use a small repeated directed graph, for example:

```text
two or four repeated motifs
each motif has the same local edge pattern
motifs are connected by a few bridge edges
```

This graph should be small enough to inspect by hand.

Example:

```text
motif 0: 0 -> 1 -> 2, 0 -> 2
motif 1: 3 -> 4 -> 5, 3 -> 5
bridge: 2 -> 3
```

### Features

Use 8-dimensional node features.

### Target

The target should be simple but nontrivial.

Good first target:

```text
predict a node-level target generated by a fixed teacher message model
```

This avoids needing a real graph ML dataset.

Procedure:

1. Generate random node features.
2. Run a fixed teacher message/readout model on the fine graph.
3. Train the hierarchical model to match the teacher's node readout.

This tests whether the hierarchical lifted path can learn to approximate a
fine-graph message computation.

### Why Teacher Target Is Better Than Arbitrary Labels

An arbitrary random label target may not depend on graph structure. A teacher
message model guarantees that the target is graph-message-passing-shaped.

This makes the viability test meaningful.

## Training Step

The package should provide a tiny helper:

```python
def train_step(
    *,
    graph: TensorGraph,
    node_features: torch.Tensor,
    target: torch.Tensor,
    message_model: nn.Module,
    lift: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    tower_bundle: TowerBundle | None = None,
) -> TrainStepResult:
    ...
```

This should:

1. call `optimizer.zero_grad()`
2. call `collapse_messages(...)`
3. compute loss
4. call `loss.backward()`
5. collect gradient diagnostics
6. call `optimizer.step()`
7. return loss and diagnostics

This helper should remain tiny. It exists to prove the first loop, not to become
a training framework.

## Diagnostics

### `GradientDiagnostics`

Fields:

```python
@dataclass(frozen=True, slots=True)
class GradientDiagnostics:
    parameter_count: int
    parameters_with_grad: int
    nonzero_gradient_count: int
    total_gradient_norm: float
```

### `ViabilityDiagnostics`

Fields:

```python
@dataclass(frozen=True, slots=True)
class ViabilityDiagnostics:
    loss_is_finite: bool
    parameters_changed: bool
    fine_message_shape: tuple[int, ...]
    coarse_message_shape: tuple[int, ...]
```

### First Viability Assertion

The first demo should assert:

```python
assert diagnostics.loss_is_finite
assert diagnostics.parameters_changed
assert diagnostics.nonzero_gradient_count > 0
```

## Testing Blueprint

### Package Smoke

Tests:

- package imports
- `hgraphml.__version__` exists
- `collapse_messages` imports from top level

### Tensor Graph

Tests:

- valid graph construction
- invalid edge shape rejected
- out-of-range edge index rejected
- edge label count mismatch rejected

### Fiber Maps

Tests:

- node fiber stores fine node tuple
- edge fiber stores fine edge tuple
- empty fiber rejected unless explicitly allowed

### Uniform Lift

Tests:

- copies each coarse message to every fine edge in its fiber
- preserves gradient from fine loss to coarse messages
- rejects missing fiber for coarse message
- returns expected shape

### Fiber-Normalized Lift

Tests:

- divides aggregate message over uniform weights
- distinguishes behavior from uniform pullback when fiber size exceeds one
- preserves gradient
- handles singleton fibers

### Learned Lift

Tests:

- output shape matches `(fine_edge_count, message_dim)`
- parameters receive gradients
- softmax weights normalize within each fiber
- one optimizer step changes parameters
- repeated calls remain deterministic under fixed seed if initialized the same

### Message Passing

Tests:

- gathers source and target features correctly
- produces edge-aligned messages
- node readout aggregates incoming messages correctly

### Tower Adapter

Tests:

- full graph registration includes all nodes
- full graph registration includes all edges
- fiber maps cover all fine nodes and edges
- no exploration loop is required

If direct `state_collapser` import is awkward, tests should capture the exact
fallback behavior and document what remains to be integrated.

### `collapse_messages`

Tests:

- end-to-end forward pass succeeds
- result contains coarse messages and fine messages
- result fine-message count equals graph edge count
- prebuilt tower bundle can be reused
- loss can backpropagate through result fine messages

### Trainability

Tests:

- one train step produces finite loss
- learned lift gradients are nonzero
- optimizer changes learned lift parameters
- toy teacher objective moves over several steps

## Documentation Blueprint

### README

The README should explain:

- HGraphML is a lightweight package applying `state_collapser` quotient towers
  to graph message passing
- the first target is trainable hierarchical message passing
- the package is not yet making speed-up claims
- the first proof is autograd viability

### Usage Doc

Add:

```text
docs/usage/01_001_first_hack.md
```

This should show:

- create toy graph
- create node features
- create message model
- create learned lift
- call `collapse_messages`
- train for a few steps

### API Notes

Add:

```text
docs/api_notes/01_001_first_surfaces.md
```

This should document:

- `TensorGraph`
- `collapse_messages`
- `UniformPullbackLift`
- `FiberNormalizedLift`
- `LearnedFiberLift`
- `train_step`

## Dependency Blueprint

### Required

- Python `3.11+`
- `torch`
- `state_collapser`

### Development

- `pytest`
- `ruff`
- `mypy`

### Deferred

- `torch_geometric`
- `networkx`
- `pgmpy`
- `dgl`

The first `pyproject.toml` should avoid heavyweight graph dependencies until
the package proves its core loop.

## Relationship To `state_collapser`

`HGraphML` should treat `state_collapser` as the quotient-tower provider.

The conceptual mapping is:

```text
state_collapser state
    -> HGraphML node

state_collapser action / edge
    -> HGraphML directed computational edge

state_collapser partition tower
    -> HGraphML message-passing hierarchy

state_collapser lift-to-finer-tier idea
    -> HGraphML message lift
```

The largest practical difference is:

```text
state_collapser discovers graphs through interaction.
HGraphML starts with the graph already known.
```

So the adapter must bypass exploration and construct a full base graph registry
at initialization.

## Implementation Risks

### Risk 1: `state_collapser` Tower API Is Too RL-Shaped

The current `state_collapser` partition tower may not expose exactly the static
full-graph construction API that `HGraphML` wants.

Mitigation:

- attempt direct import first
- identify the smallest required adapter surface
- avoid silently reimplementing `state_collapser`
- document any incompatibility precisely

### Risk 2: Lift Semantics Become Ambiguous

Uniform pullback and fiber-normalized lift have different semantics. Copying a
message and distributing aggregate mass are not the same operation.

Mitigation:

- test both separately
- name them clearly
- make learned lift explicit about whether it emits weights, decoded messages,
  or both

### Risk 3: Autograd Breaks Through Indexing Or Adapter Boundaries

Tower construction is non-differentiable combinatorics. That is fine. Message
values and lift outputs must remain differentiable.

Mitigation:

- keep tower construction outside gradient expectations
- use PyTorch tensor operations for message/lift data
- write explicit gradient diagnostics
- avoid `.detach()`, `.item()`, or Python scalar conversion in the differentiable
  path

### Risk 4: First Demo Accidentally Proves Nothing

A toy target that ignores graph structure would not prove the architecture.

Mitigation:

- use a teacher message model to generate graph-structured targets
- train hierarchical path to match teacher readout
- inspect gradient and parameter-change diagnostics

### Risk 5: Dependency Drag

Pulling in PyG, pgmpy, DGL, or networkx too early may bury the core proof under
installation and adapter complexity.

Mitigation:

- start with PyTorch-only graph tensors
- add external adapters after core viability
- keep adapter modules optional

## First Milestone Definition

Milestone one is complete when the repo has:

- installable package skeleton
- `TensorGraph`
- `UniformPullbackLift`
- `FiberNormalizedLift`
- `LearnedFiberLift`
- minimal message model
- state-collapser-backed or explicitly documented fallback tower adapter
- `collapse_messages(...)`
- node readout
- `train_step(...)`
- toy teacher-message example
- tests proving forward pass, lift behavior, gradients, optimizer updates, and
  toy objective movement
- README showing the first hack

Milestone one is not required to show speed-up.

## Proposed Implementation Sequence

This is not yet a Phase.Stage.Action gameplan, but the blueprint implies this
rough order:

1. Package skeleton and dependency setup
2. Tensor graph data model
3. Fiber maps
4. Deterministic lifts
5. Learned lift
6. Simple message model and readout
7. `state_collapser` tower adapter investigation and implementation
8. `collapse_messages(...)`
9. Train step and diagnostics
10. Toy teacher example
11. Tests
12. README/usage/API docs

## Blueprint Conclusion

The design is ready for a detailed Phase.Stage.Action implementation gameplan.

The only major implementation uncertainty is the exact shape of the direct
`state_collapser` tower import. That uncertainty should not block the gameplan;
it should become an early explicit phase with a stop condition:

```text
If direct import is compatible, proceed with it.
If direct import is not compatible, document the mismatch and request PO approval
before using a local fallback tower/fiber adapter.
```

Everything else is straightforward package engineering around the core learned
lift proof.
