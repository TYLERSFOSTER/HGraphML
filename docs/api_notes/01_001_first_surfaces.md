# HGraphML First API Surfaces

## Status

These are the first package surfaces for `HGraphML`.

They are intentionally small. The goal is not to define a mature graph ML
framework yet. The goal is to make the first `state_collapser`-backed graph
message-passing experiment possible, inspectable, and trainable.

The surfaces below should be read as milestone-one APIs. They are usable, typed,
and tested, but not yet promised as long-term stable public interfaces.

## Top-Level Package

```python
from hgraphml import HGraphMLResult, __version__, collapse_messages
```

The top-level package exports only the orchestration call, its result type, and
the version string.

Everything else lives under explicit subpackages so the package does not pretend
to be more mature than it is.

## `TensorGraph`

Defined in:

```text
src/hgraphml/graph/data.py
```

Import:

```python
from hgraphml.graph import TensorGraph
```

Shape:

```python
TensorGraph(
    node_count: int,
    edge_index: torch.Tensor,
    edge_labels: tuple[str, ...] | None = None,
)
```

`edge_index` is a rank-2 integer tensor of shape `(2, edge_count)`. Row `0`
contains source node indices; row `1` contains target node indices.

Validation rules:

- `node_count` must be nonnegative,
- `edge_index` must have shape `(2, E)`,
- edge indices must be integer tensors,
- nonempty edge indices must lie in `[0, node_count)`,
- `edge_labels`, when present, must have length `edge_count`.

Convenience properties:

- `edge_count`,
- `sources`,
- `targets`.

## `MessageBatch`

Defined in:

```text
src/hgraphml/messages/containers.py
```

Import:

```python
from hgraphml.messages import MessageBatch
```

Shape:

```python
MessageBatch(
    graph: TensorGraph,
    values: torch.Tensor,
)
```

`values` must be a rank-2 tensor whose first dimension equals
`graph.edge_count`.

This container is deliberately minimal. It exists so later APIs can distinguish
edge-aligned messages from arbitrary tensors.

## `NodeFiber` And `EdgeFiber`

Defined in:

```text
src/hgraphml/graph/fibers.py
```

Import:

```python
from hgraphml.graph import EdgeFiber, NodeFiber
```

Shapes:

```python
NodeFiber(
    tier: int,
    coarse_node: int,
    fine_nodes: tuple[int, ...],
)

EdgeFiber(
    tier: int,
    coarse_edge: int,
    fine_edges: tuple[int, ...],
)
```

Fiber maps tell `HGraphML` how a coarse tier sits over the original fine graph.
A node fiber maps one coarse node to the fine nodes it represents. An edge fiber
maps one coarse edge to the fine edges it represents.

Validation rules:

- tier indices must be nonnegative,
- coarse indices must be nonnegative,
- fine-index tuples must be nonempty,
- fine-index tuples must not contain duplicates.

Type aliases:

```python
NodeFiberMap = Mapping[int, NodeFiber]
EdgeFiberMap = Mapping[int, EdgeFiber]
```

Milestone-one adapter policy: coarse node IDs and coarse edge IDs are compacted
to contiguous integer ranges so tensors can be built without a second indexing
layer.

## `TowerBundle`

Defined in:

```text
src/hgraphml/adapters/state_collapser.py
```

Import:

```python
from hgraphml.adapters import TowerBundle, build_tower_bundle
```

Shape:

```python
TowerBundle(
    graph: TensorGraph,
    partition_tower: PartitionTower,
    node_fibers_by_tier: tuple[Mapping[int, NodeFiber], ...],
    edge_fibers_by_tier: tuple[Mapping[int, EdgeFiber], ...],
    coarse_graphs_by_tier: tuple[TensorGraph, ...],
)
```

`TowerBundle` is the bridge object between `state_collapser` and HGraphML.

It contains both the underlying `state_collapser` `PartitionTower` and the
tensor-friendly readouts that message passing needs.

## `build_tower_bundle(...)`

Defined in:

```text
src/hgraphml/adapters/state_collapser.py
```

Import:

```python
from hgraphml.adapters import build_tower_bundle
```

Signature:

```python
def build_tower_bundle(
    graph: TensorGraph,
    *,
    contraction_schema: ContractionSchema | None = None,
) -> TowerBundle:
    ...
```

This function treats the HGraphML graph as fully known up front. It registers
all nodes and directed edges into `state_collapser`, builds a partition tower,
then recovers node and edge fibers for every tier.

Important implementation detail: HGraphML edge fibers are grouped by active
tier state-cell endpoints. This avoids exposing transient action-cell labels
from the internal `state_collapser` action layer as stable HGraphML coarse-edge
IDs.

## Lift Interface

Defined in:

```text
src/hgraphml/lifts/base.py
```

Import:

```python
from hgraphml.lifts import MessageLift
```

Protocol shape:

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

This protocol is intentionally compatible with both stateless lift objects and
`torch.nn.Module` learned lifts.

## `UniformPullbackLift`

Defined in:

```text
src/hgraphml/lifts/uniform.py
```

Import:

```python
from hgraphml.lifts import UniformPullbackLift
```

Behavior:

```text
fine_message[e] = coarse_message[c]
```

for every fine edge `e` in the fiber of coarse edge `c`.

This is the simplest sanity-check lift. It copies a coarse message to all fine
edges represented by that coarse edge.

## `FiberNormalizedLift`

Defined in:

```text
src/hgraphml/lifts/fiber_normalized.py
```

Import:

```python
from hgraphml.lifts import FiberNormalizedLift
```

Behavior:

```text
fine_message[e] = coarse_message[c] / |fiber(c)|
```

This lift preserves total message mass under summation over each edge fiber.

## `LearnedFiberLift`

Defined in:

```text
src/hgraphml/lifts/learned.py
```

Import:

```python
from hgraphml.lifts import LearnedFiberLift
```

Constructor:

```python
LearnedFiberLift(
    message_dim: int,
    context_dim: int,
    hidden_dim: int = 32,
)
```

The learned lift uses a small score network and decoder. For each coarse edge
fiber, it computes scores from:

```text
[coarse_message, fine_edge_context]
```

then normalizes scores with a within-fiber softmax and decodes fine-edge
messages from:

```text
[coarse_message, fine_edge_context, within_fiber_weight]
```

Diagnostics:

```python
lift.last_weights_by_fiber
```

stores the most recent within-fiber weights. These tensors are not detached by
the lift.

## `EdgeMessageMLP`

Defined in:

```text
src/hgraphml/messages/passing.py
```

Import:

```python
from hgraphml.messages import EdgeMessageMLP
```

Constructor:

```python
EdgeMessageMLP(
    feature_dim: int,
    message_dim: int,
    hidden_dim: int = 32,
    edge_feature_dim: int = 0,
)
```

This is a tiny first message model. It consumes source-node features,
target-node features, and optional edge features, then emits edge-aligned
messages.

It is not intended as a complete graph neural network layer.

## `collapse_messages(...)`

Defined in:

```text
src/hgraphml/collapse.py
```

Import:

```python
from hgraphml import collapse_messages
```

Signature:

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

Milestone-one behavior:

1. Validate graph and feature tensors.
2. Build a `TowerBundle` unless one is supplied.
3. Select a coarse tier; default is the largest available tier.
4. Mean-pool fine node features into coarse node features.
5. Run the message model on the selected coarse graph.
6. Build fine-edge contexts from fine source/target features and optional edge
   features.
7. Lift coarse messages back to fine edges.
8. Compute incoming-sum readout on the fine graph.
9. Return `HGraphMLResult`.

`refinement_steps` is reserved for later work and must currently be `0`.

## `HGraphMLResult`

Defined in:

```text
src/hgraphml/collapse.py
```

Fields:

```python
graph: TensorGraph
tower_bundle: TowerBundle
coarse_tier: int
coarse_messages: torch.Tensor
fine_messages: torch.Tensor
node_readout: torch.Tensor
```

This object is deliberately plain. It keeps enough information to inspect what
happened without introducing an experiment framework.

## `train_step(...)`

Defined in:

```text
src/hgraphml/training/train_step.py
```

Import:

```python
from hgraphml.training import train_step
```

Signature:

```python
def train_step(
    *,
    graph: TensorGraph,
    node_features: torch.Tensor,
    target: torch.Tensor,
    message_model: nn.Module,
    lift: MessageLift,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    tower_bundle: TowerBundle | None = None,
) -> TrainStepResult:
    ...
```

Flow:

1. Snapshot trainable parameters.
2. Zero gradients.
3. Call `collapse_messages(...)`.
4. Compute loss.
5. Backpropagate.
6. Collect gradient diagnostics.
7. Step optimizer.
8. Collect viability diagnostics.

## Diagnostics

Defined in:

```text
src/hgraphml/diagnostics/
```

Current diagnostics are deliberately viability-oriented:

- `GradientDiagnostics` reports parameter count, gradient coverage, nonzero
  gradient count, and total gradient norm.
- `ViabilityDiagnostics` reports finite loss, whether parameters changed, and
  coarse/fine message shapes.

These are not experiment-tracking artifacts. They are first-pass proof tools.

## Toy Example Surfaces

Defined in:

```text
src/hgraphml/examples/
```

Useful imports:

```python
from hgraphml.examples.toy_graphs import make_repeated_motif_graph
from hgraphml.examples.toy_graphs import make_toy_node_features
from hgraphml.examples.learned_lift_demo import run_learned_lift_demo
```

The toy example exists to prove trainability, not speed.

## Stability Warning

The current surfaces are suitable for first experiments and follow-up design.
They should not yet be treated as a stable external graph ML framework API.

The likely next surface work is:

- richer graph adapters,
- clearer exact-vs-approximate lift semantics,
- benchmark harnesses,
- stronger device and batching conventions,
- real graph ML examples beyond the toy motif graph.
